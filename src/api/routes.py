"""
FastAPI application with analysis + trace streaming endpoints.

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

import asyncio
import json
import os
from dataclasses import asdict

try:
    from fastapi import FastAPI, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    FastAPI = None  # type: ignore[assignment,misc]
    BaseModel = None  # type: ignore[assignment]

from src.core.providers import MockProvider, ModelProvider, provider_from_env
from src.core.trace_store import TraceStore
from src.data.graph_store import GraphStore, graph_store_from_env
from src.detective.experience import EMPTY_LIBRARY
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

    class InvestigateRequest(BaseModel):  # type: ignore[valid-type]
        trigger_mode: str  # "hypothesis", "topic", "reactive"
        seed: str
        max_steps: int = 50
        max_pages: int = 200
        max_llm_calls: int = 300
        max_time: int = 3600
        source_ids: list[str] | None = None
        constitution_path: str | None = None
        phi_metrics: dict[str, float] | None = None

    class InvestigateStartResponse(BaseModel):  # type: ignore[valid-type]
        investigation_id: str
        status: str

    class InvestigationStatusResponse(BaseModel):  # type: ignore[valid-type]
        id: str
        steps: int
        findings: int
        hypotheses: int
        pages: int
        llm_calls: int
        elapsed_seconds: float
        running: bool
        assumptions_detected: int = 0

else:
    # Satisfy module-level names so imports never raise NameError
    AnalyzeRequest = None  # type: ignore[assignment,misc]
    AnalyzeResponse = None  # type: ignore[assignment,misc]
    NetworkResponse = None  # type: ignore[assignment,misc]
    EvolveRequest = None  # type: ignore[assignment,misc]
    EvolveResponse = None  # type: ignore[assignment,misc]
    InvestigateRequest = None  # type: ignore[assignment,misc]
    InvestigateStartResponse = None  # type: ignore[assignment,misc]
    InvestigationStatusResponse = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# App factory — wraps route registration so the app can be re-created in tests
# ---------------------------------------------------------------------------


def create_app(
    graph: "GraphStore | None" = None,
    provider: "ModelProvider | None" = None,
    trace_store: "TraceStore | None" = None,
) -> "FastAPI":  # type: ignore[name-defined]
    """
    Build and return the FastAPI application instance.

    Registering routes inside a factory rather than at module scope allows
    tests to create isolated instances and avoids shared global state between
    test runs.

    graph:      optional pre-built GraphStore.  When None, graph_store_from_env()
                is called once at app-creation time.
    provider:   optional ModelProvider.  When None, provider_from_env() is tried
                first; falls back to MockProvider when env vars are not configured
                (safe for tests and local dev without a running Ollama instance).
    trace_store: optional TraceStore.  When provided, enables /traces/* endpoints
                for SSE streaming, recent queries, and JSONL history.
    All route handlers share the same provider/store instances via closure.
    """
    # Load .env.local so uvicorn-launched apps get the same config as CLI
    try:
        from dotenv import load_dotenv
        load_dotenv(".env.local", override=False)
    except ImportError:
        pass

    from src.core.log import configure_logging
    configure_logging()

    if FastAPI is None:
        raise ImportError("fastapi is required to create the Detective LLM API app.")

    _graph: GraphStore = graph if graph is not None else graph_store_from_env()

    if provider is not None:
        _provider: ModelProvider = provider
    else:
        try:
            _provider = provider_from_env()
        except (ValueError, KeyError, ImportError) as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Provider configuration failed (%s), using MockProvider. "
                "API will return synthetic responses.",
                exc,
            )
            _provider = MockProvider(response=_API_MOCK_RESPONSE)

    # Resolve trace store: explicit arg > provider attribute > DETECTIVE_TRACE_PATH env
    if trace_store is not None:
        _trace_store = trace_store
    else:
        _trace_store = getattr(_provider, "_trace_store", None)
        if _trace_store is None:
            _trace_path = os.environ.get("DETECTIVE_TRACE_PATH")
            if _trace_path:
                from pathlib import Path
                _trace_store = TraceStore(path=Path(_trace_path))

    app = FastAPI(title=API_TITLE, version=API_VERSION)

    _DEFAULT_CORS_ORIGINS = [
        "https://www.crichalchemist.com",
        "https://crichalchemist.com",
        "https://crichalchemist-maninagarden.hf.space",
        "http://localhost:7860",
    ]

    cors_origins_raw = os.environ.get("CORS_ORIGINS")
    cors_origins = (
        [o.strip() for o in cors_origins_raw.split(",") if o.strip()]
        if cors_origins_raw
        else _DEFAULT_CORS_ORIGINS
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

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

    # ------------------------------------------------------------------
    # GET /traces/recent — last N traces from in-memory deque
    # ------------------------------------------------------------------

    @app.get("/traces/recent")
    def traces_recent(
        n: int = Query(default=50, ge=1, le=500),
    ) -> list[dict]:
        if _trace_store is None:
            return []
        return [asdict(t) for t in _trace_store.recent(n=n)]

    # ------------------------------------------------------------------
    # GET /traces/history — paginated JSONL read (newest first)
    # ------------------------------------------------------------------

    @app.get("/traces/history")
    def traces_history(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> list[dict]:
        if _trace_store is None:
            return []
        return [asdict(t) for t in _trace_store.historical(offset=offset, limit=limit)]

    # ------------------------------------------------------------------
    # GET /traces/stream — SSE live stream with 30s keepalive
    # ------------------------------------------------------------------

    @app.get("/traces/stream")
    async def traces_stream() -> StreamingResponse:
        if _trace_store is None:
            return StreamingResponse(
                iter([": no trace store configured\n\n"]),
                media_type="text/event-stream",
            )

        async def _event_generator():
            q = _trace_store.subscribe()
            try:
                while True:
                    try:
                        trace = await asyncio.wait_for(q.get(), timeout=30)
                        yield f"data: {json.dumps(asdict(trace))}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                _trace_store.unsubscribe(q)

        return StreamingResponse(
            _event_generator(),
            media_type="text/event-stream",
        )

    # ------------------------------------------------------------------
    # GET /graph/export — full graph for space visualization
    # ------------------------------------------------------------------

    @app.get("/graph/export")
    def graph_export() -> dict:
        """
        Export the full knowledge graph as nodes + edges for visualization.

        Designed for the HF Space Entity Network Explorer tab. Returns
        all nodes, all edges with relation type and confidence, and
        summary statistics.
        """
        nodes = _graph.nodes()
        edges = []
        for node in nodes:
            for target in _graph.successors(node):
                edge = _graph.get_edge(node, target)
                if edge:
                    edges.append({
                        "source": edge.source,
                        "target": edge.target,
                        "relation": edge.relation.value,
                        "confidence": edge.confidence,
                    })
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        }

    # ------------------------------------------------------------------
    # Investigation endpoints
    # ------------------------------------------------------------------

    # In-memory registry of running/completed investigations
    _investigations: dict = {}
    _investigation_reports: dict[str, dict] = {}
    _investigation_tasks: dict[str, asyncio.Task] = {}

    @app.post("/investigate", response_model=InvestigateStartResponse)
    async def investigate_start(request: InvestigateRequest) -> InvestigateStartResponse:  # type: ignore[name-defined]
        """Start an autonomous investigation asynchronously."""
        from src.detective.investigation.agent import InvestigationAgent
        from src.detective.investigation.types import InvestigationBudget, InvestigationConfig

        budget = InvestigationBudget(
            max_steps=request.max_steps,
            max_pages=request.max_pages,
            max_llm_calls=request.max_llm_calls,
            max_time_seconds=request.max_time,
        )
        source_ids = tuple(request.source_ids) if request.source_ids else ("foia_fbi_vault", "graph_neighbourhood")

        config = InvestigationConfig(
            trigger_mode=request.trigger_mode,  # type: ignore[arg-type]
            seed=request.seed,
            budget=budget,
            source_ids=source_ids,
            constitution_path=request.constitution_path,
            phi_metrics=request.phi_metrics,
        )

        from src.detective.investigation.source_protocol import build_sources
        sources = build_sources(config.source_ids, _graph)
        try:
            from src.detective.constitution import load_constitution
            constitution = load_constitution()
        except FileNotFoundError:
            constitution = "Epistemic honesty above analytical comfort."

        agent = InvestigationAgent(
            config=config,
            provider=_provider,
            graph=_graph,
            sources=sources,
            constitution=constitution,
            trace_store=_trace_store,
        )
        _investigations[config.id] = agent

        async def _run_and_store(inv_id: str, ag: object) -> None:
            report = await ag.run()
            _investigation_reports[inv_id] = asdict(report)

        task = asyncio.create_task(_run_and_store(config.id, agent))
        _investigation_tasks[config.id] = task

        return InvestigateStartResponse(
            investigation_id=config.id,
            status="started",
        )

    @app.get("/investigation/{investigation_id}/status", response_model=InvestigationStatusResponse)
    def investigation_status(investigation_id: str) -> InvestigationStatusResponse:  # type: ignore[name-defined]
        """Poll investigation progress."""
        from fastapi import HTTPException

        agent = _investigations.get(investigation_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Investigation not found")
        return InvestigationStatusResponse(**agent.status)

    @app.get("/investigation/{investigation_id}/report")
    def investigation_report(investigation_id: str) -> dict:
        """Get the final investigation report."""
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse

        report = _investigation_reports.get(investigation_id)
        if report is not None:
            return report

        if investigation_id in _investigations:
            return JSONResponse(status_code=202, content={"status": "running"})

        raise HTTPException(status_code=404, detail="Investigation not found")

    @app.get("/investigation/{investigation_id}/stream")
    async def investigation_stream(investigation_id: str) -> StreamingResponse:
        """SSE stream of investigation steps."""
        agent = _investigations.get(investigation_id)
        if agent is None:
            return StreamingResponse(
                iter([f": investigation {investigation_id} not found\n\n"]),
                media_type="text/event-stream",
            )

        async def _step_generator():
            seen = 0
            while True:
                steps = agent._steps  # noqa: SLF001 — direct access for streaming
                if len(steps) > seen:
                    for step in steps[seen:]:
                        yield f"data: {json.dumps(asdict(step), default=str)}\n\n"
                    seen = len(steps)

                # Check if investigation is complete
                task = _investigation_tasks.get(investigation_id)
                if task and task.done():
                    yield f"data: {json.dumps({'event': 'complete'})}\n\n"
                    break

                try:
                    await asyncio.sleep(1)
                except asyncio.CancelledError:
                    break

        return StreamingResponse(
            _step_generator(),
            media_type="text/event-stream",
        )

    return app


# ---------------------------------------------------------------------------
# Module-level app instance (None when fastapi unavailable)
# ---------------------------------------------------------------------------

app = create_app() if FastAPI is not None else None
