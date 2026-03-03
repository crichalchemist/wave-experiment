"""
FastAPI application with three analysis endpoints.

Data minimization is a first-class constraint: every response model
exposes only the fields the client strictly needs, no more.

The module degrades gracefully when fastapi/pydantic are not installed —
FastAPI and BaseModel are set to None, app is None, and callers that
import only create_app() will receive None, leaving the rest of the
package unaffected.

Note: when fastapi IS installed, the module-level `app = create_app()`
at the bottom of this file executes on every import. That call invokes
`graph_store_from_env()` which reads environment variables. This is
intentional — it enables `uvicorn src.api.routes:app` ASGI entry-point
convention. In test code, always use `create_app()` via a fresh fixture
to avoid shared state between test runs.
"""
from __future__ import annotations

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:
    FastAPI = None  # type: ignore[assignment,misc]
    BaseModel = None  # type: ignore[assignment]

from src.core.providers import MockProvider, ModelProvider, provider_from_env
from src.data.graph_store import GraphStore, graph_store_from_env
from src.detective.experience import EMPTY_LIBRARY
from src.detective.evolution import evolve_hypothesis
from src.detective.hypothesis import Hypothesis
from src.inference.pipeline import analyze

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------

API_TITLE: str = "Detective LLM API"
API_VERSION: str = "0.1.0"

# Default confidence for a freshly created hypothesis at the /evolve boundary.
# 0.5 represents maximum uncertainty — the prior before any evidence update.
_INITIAL_HYPOTHESIS_CONFIDENCE: float = 0.5

# MockProvider response used at the API layer while no real provider is wired.
_API_MOCK_RESPONSE: str = "Analysis: gap detected between 2013-2017."


# ---------------------------------------------------------------------------
# Request / response models (defined only when pydantic is available)
# ---------------------------------------------------------------------------

if BaseModel is not None:

    class AnalyzeRequest(BaseModel):  # type: ignore[valid-type]
        claim: str
        phi_metrics: dict[str, float] | None = None  # optional Φ construct levels

    class AnalyzeResponse(BaseModel):  # type: ignore[valid-type]
        verdict: str
        confidence: float
        evidence_count: int  # number of evidence nodes retrieved — not gap count

    class NetworkResponse(BaseModel):  # type: ignore[valid-type]
        entity: str
        relationships: list[str]  # immediate successor node IDs (1-hop)

    class EvolveRequest(BaseModel):  # type: ignore[valid-type]
        evidence_path: str  # text of new evidence; used as hypothesis statement
        phi_metrics: dict[str, float] | None = None  # optional Φ construct levels

    class EvolveResponse(BaseModel):  # type: ignore[valid-type]
        hypothesis_id: str
        statement: str
        confidence: float

else:
    # Satisfy module-level names so imports never raise NameError
    AnalyzeRequest = None  # type: ignore[assignment,misc]
    AnalyzeResponse = None  # type: ignore[assignment,misc]
    NetworkResponse = None  # type: ignore[assignment,misc]
    EvolveRequest = None  # type: ignore[assignment,misc]
    EvolveResponse = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# App factory — wraps route registration so the app can be re-created in tests
# ---------------------------------------------------------------------------


def create_app(
    graph: "GraphStore | None" = None,
    provider: "ModelProvider | None" = None,
) -> "FastAPI":  # type: ignore[name-defined]
    """
    Build and return the FastAPI application instance.

    Registering routes inside a factory rather than at module scope allows
    tests to create isolated instances and avoids shared global state between
    test runs.

    graph:    optional pre-built GraphStore.  When None, graph_store_from_env()
              is called once at app-creation time.
    provider: optional ModelProvider.  When None, provider_from_env() is tried
              first; falls back to MockProvider when env vars are not configured
              (safe for tests and local dev without a running Ollama instance).
    All route handlers share the same provider instance via closure.
    """
    if FastAPI is None:
        raise ImportError("fastapi is required to create the Detective LLM API app.")

    _graph: GraphStore = graph if graph is not None else graph_store_from_env()

    if provider is not None:
        _provider: ModelProvider = provider
    else:
        try:
            _provider = provider_from_env()
        except (ValueError, KeyError, ImportError):
            _provider = MockProvider(response=_API_MOCK_RESPONSE)

    app = FastAPI(title=API_TITLE, version=API_VERSION)

    # ------------------------------------------------------------------
    # POST /analyze
    # ------------------------------------------------------------------

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze_endpoint(request: AnalyzeRequest) -> AnalyzeResponse:  # type: ignore[name-defined]
        """
        Run the 4-layer analytical pipeline on the supplied claim.

        Data minimized to: verdict, confidence, gaps_found.
        The full AnalysisResult (including reasoning chain, intent, raw evidence)
        is deliberately not surfaced — consumers receive only what they act on.
        """
        # No constitution wired at API layer; verify_inline falls back to
        # the default principle when constitution lacks a .critique() method.
        constitution = object()

        result = analyze(
            claim=request.claim,
            provider=_provider,
            graph=_graph,
            library=EMPTY_LIBRARY,
            constitution=constitution,
            phi_metrics=request.phi_metrics,
        )

        return AnalyzeResponse(
            verdict=result.verdict,
            confidence=result.confidence,
            evidence_count=len(result.evidence),
        )

    # ------------------------------------------------------------------
    # GET /network/{entity}
    # ------------------------------------------------------------------

    @app.get("/network/{entity}", response_model=NetworkResponse)
    def network_endpoint(entity: str) -> NetworkResponse:  # type: ignore[name-defined]
        """
        Return 1-hop successors of the named entity from the knowledge graph.

        Data minimized to: entity, relationships (list of neighbour IDs).
        If the entity is not in the graph, relationships is an empty list —
        absence is investigatively significant and should not raise an error.
        """
        neighbours = _graph.successors(entity)
        return NetworkResponse(entity=entity, relationships=neighbours)

    # ------------------------------------------------------------------
    # POST /evolve
    # ------------------------------------------------------------------

    @app.post("/evolve", response_model=EvolveResponse)
    async def evolve_endpoint(request: EvolveRequest) -> EvolveResponse:  # type: ignore[name-defined]
        """
        Evolve a fresh hypothesis against the supplied evidence text.

        Optionally accepts phi_metrics for welfare-aware hypothesis evolution.

        A base hypothesis is created at maximum-uncertainty confidence (0.5)
        and then updated by evolve_parallel() with k=1 (single branch).
        The experience record from that call is discarded at this layer —
        the API surface is stateless; callers that need lineage tracking
        should use the Python API directly.

        Data minimized to: hypothesis_id, statement, confidence.
        """
        base = Hypothesis.create(
            text=request.evidence_path,
            confidence=_INITIAL_HYPOTHESIS_CONFIDENCE,
        )

        from src.detective.parallel_evolution import evolve_parallel

        results = await evolve_parallel(
            hypothesis=base,
            evidence_list=[request.evidence_path],
            provider=_provider,
            k=1,
            library=EMPTY_LIBRARY,
            phi_metrics=request.phi_metrics,
        )

        if results:
            evolved = results[0].hypothesis
        else:
            evolved = base

        return EvolveResponse(
            hypothesis_id=evolved.id,
            statement=evolved.text,
            confidence=evolved.confidence,
        )

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (None when fastapi unavailable)
# ---------------------------------------------------------------------------

app = create_app() if FastAPI is not None else None
