"""Autonomous investigation agent — plan → gather → analyze → evolve → enrich loop."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from src.core.providers import ModelProvider, provider_from_env
from src.core.trace_store import TraceStore
from src.data.entity_filter import filter_entities
from src.data.graph_store import GraphStore, graph_store_from_env
from src.detective.constitution import load_constitution
from src.detective.evolution import branching_rule
from src.detective.experience import (
    EMPTY_LIBRARY,
    ExperienceLibrary,
    Experience,
    add_experience,
)
from src.detective.hypothesis import Hypothesis, WEIGHTS_BRIDGE, WEIGHTS_DEFAULT
from src.detective.investigation.planner import (
    generate_leads,
    generate_seed_hypotheses,
    hypotheses_from_graph_event,
    spawn_alternatives,
)
from src.detective.investigation.source_protocol import (
    InvestigationSource,
    build_sources,
)
from src.detective.investigation.types import (
    AssumptionDetection,
    AssumptionScanResult,
    DocumentEvidence,
    Finding,
    HypothesisSnapshot,
    InvestigationBudget,
    InvestigationConfig,
    InvestigationReport,
    InvestigationStep,
    Lead,
    SourceResult,
    TerminationReason,
)
from src.detective.module_a import detect_cognitive_biases
from src.detective.module_b import detect_historical_determinism
from src.detective.module_c import detect_geopolitical_presumptions
from src.detective.parallel_evolution import evolve_parallel
from src.inference.pipeline import AnalysisResult, analyze
from src.inference.welfare_scoring import (
    infer_threatened_constructs,
    score_hypothesis_welfare,
)

_logger = logging.getLogger(__name__)

# Max leads to execute per gather phase
_GATHER_BATCH_SIZE = 5

# Confidence thresholds
_FINDING_CONFIDENCE_THRESHOLD = 0.7
_PRUNE_CONFIDENCE_THRESHOLD = 0.05

# Parallel evolution branch count
_EVOLUTION_K = 3

# Assumption scan limits
_ASSUMPTION_SCAN_MAX_DOCS = 5
_ASSUMPTION_MAX_FINDINGS_PER_SCAN = 10
_MAX_COUNTER_LEADS = 3
_ASSUMPTION_TEXT_LIMIT = 2000

_COUNTER_LEAD_PROMPT = (
    "An assumption was detected in investigative evidence:\n"
    "[{module}] {assumption_type}: {detail}\n"
    "Source: {source_text}\n\n"
    "Generate a focused search query to investigate whether this assumption "
    "is valid or masks a gap in the evidence. Reply with ONLY: query: <text>"
)


# ---------------------------------------------------------------------------
# BudgetTracker — the only mutable state in the agent
# ---------------------------------------------------------------------------

@dataclass
class BudgetTracker:
    """Mutable budget counter. Not exposed outside the agent."""

    budget: InvestigationBudget
    steps: int = 0
    pages: int = 0
    llm_calls: int = 0
    start_time: float = 0.0

    def __post_init__(self) -> None:
        if self.start_time == 0.0:
            self.start_time = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.start_time

    def check(self) -> TerminationReason | None:
        """Return a termination reason if any budget is exceeded, else None."""
        if self.steps >= self.budget.max_steps:
            return "budget_max_steps"
        if self.pages >= self.budget.max_pages:
            return "budget_max_pages"
        if self.llm_calls >= self.budget.max_llm_calls:
            return "budget_max_llm_calls"
        if self.elapsed >= self.budget.max_time_seconds:
            return "budget_max_time"
        return None

    def record_gather(self, pages: int) -> None:
        self.pages += pages

    def record_llm_call(self, count: int = 1) -> None:
        self.llm_calls += count

    def record_step(self) -> None:
        self.steps += 1


# ---------------------------------------------------------------------------
# InvestigationAgent
# ---------------------------------------------------------------------------

class InvestigationAgent:
    """Autonomous investigation loop connecting all detective-llm subsystems."""

    def __init__(
        self,
        config: InvestigationConfig,
        provider: ModelProvider,
        graph: GraphStore,
        sources: dict[str, InvestigationSource],
        constitution: str,
        trace_store: TraceStore | None = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._graph = graph
        self._sources = sources
        self._constitution = constitution
        self._trace_store = trace_store
        self._budget = BudgetTracker(budget=config.budget)

        # Accumulated immutable state
        self._hypotheses: list[Hypothesis] = []
        self._findings: list[Finding] = []
        self._steps: list[InvestigationStep] = []
        self._library: ExperienceLibrary = EMPTY_LIBRARY
        self._lead_queue: list[Lead] = []
        self._total_documents: int = 0
        self._graph_edges_added: int = 0
        self._assumption_results: list[AssumptionScanResult] = []
        self._total_assumptions: int = 0

    @classmethod
    def from_env(cls, config: InvestigationConfig) -> InvestigationAgent:
        """Factory: construct agent from environment variables."""
        provider = provider_from_env()
        graph = graph_store_from_env()
        sources = build_sources(config.source_ids, graph)

        constitution_path = config.constitution_path or "docs/constitution.md"
        try:
            constitution = load_constitution(Path(constitution_path))
        except FileNotFoundError:
            _logger.warning("Constitution not found at %s; using default", constitution_path)
            constitution = "Epistemic honesty above analytical comfort."

        trace_store = None
        import os
        trace_path = os.environ.get("DETECTIVE_TRACE_PATH")
        if trace_path:
            trace_store = TraceStore(path=Path(trace_path))

        return cls(
            config=config,
            provider=provider,
            graph=graph,
            sources=sources,
            constitution=constitution,
            trace_store=trace_store,
        )

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    async def run(self) -> InvestigationReport:
        """Execute the autonomous investigation loop until termination."""
        self._seed_hypotheses()

        while True:
            # 1. Check termination
            reason = self._budget.check()
            if reason:
                self._record_step("budget_halt")
                return self._build_report(reason)

            if not self._hypotheses and not self._lead_queue:
                self._record_step("budget_halt")
                return self._build_report("leads_exhausted")

            # 2. PLAN — generate leads
            plan_leads, plan_llm = self._plan_phase()
            self._budget.record_llm_call(plan_llm)
            self._lead_queue.extend(plan_leads)
            self._record_step("plan", leads_generated=len(plan_leads), llm_calls=plan_llm)

            reason = self._budget.check()
            if reason:
                return self._build_report(reason)

            # 3. GATHER — execute top leads
            docs, gather_pages, gather_findings, gather_llm = self._gather_phase()
            self._budget.record_gather(gather_pages)
            self._budget.record_llm_call(gather_llm)
            self._total_documents += len(docs)
            self._findings.extend(gather_findings)
            self._record_step(
                "gather",
                documents_gathered=len(docs),
                pages_consumed=gather_pages,
            )

            if not docs:
                # No documents gathered — check if we have more leads
                if not self._lead_queue:
                    return self._build_report("leads_exhausted")
                self._budget.record_step()
                continue

            reason = self._budget.check()
            if reason:
                return self._build_report(reason)

            # 4. ANALYZE — run documents through pipeline + assumption scan
            analysis_results, analyze_llm, halt, assumption_results = self._analyze_phase(docs)
            self._budget.record_llm_call(analyze_llm)
            assumptions_detected = sum(len(r.detections) for r in assumption_results)
            self._record_step(
                "analyze",
                llm_calls=analyze_llm,
                assumptions_detected=assumptions_detected,
            )

            if halt:
                self._record_step("constitutional_halt")
                return self._build_report("constitutional_halt")

            reason = self._budget.check()
            if reason:
                return self._build_report(reason)

            # 5. REFLECT — constitutional critique of analysis results
            reflect_llm, reflect_halt = self._reflect_phase(analysis_results)
            self._budget.record_llm_call(reflect_llm)
            self._record_step("reflect", llm_calls=reflect_llm)

            if reflect_halt:
                self._record_step("constitutional_halt")
                return self._build_report("constitutional_halt")

            reason = self._budget.check()
            if reason:
                return self._build_report(reason)

            # 6. EVOLVE — evolve hypotheses with evidence
            evolve_llm, evolve_findings, evolve_hyps = await self._evolve_phase(
                docs, analysis_results
            )
            self._budget.record_llm_call(evolve_llm)
            self._findings.extend(evolve_findings)
            self._record_step(
                "evolve",
                hypotheses_evolved=evolve_hyps,
                findings_produced=len(evolve_findings),
                llm_calls=evolve_llm,
            )

            reason = self._budget.check()
            if reason:
                return self._build_report(reason)

            # 7. ENRICH — extract entities and add to graph
            edges_added = self._enrich_phase(docs)
            self._graph_edges_added += edges_added

            self._budget.record_step()

    # -------------------------------------------------------------------
    # Phase implementations
    # -------------------------------------------------------------------

    def _seed_hypotheses(self) -> None:
        """Create initial hypotheses based on trigger mode."""
        mode = self._config.trigger_mode

        if mode == "hypothesis":
            h = Hypothesis.create(
                text=self._config.seed,
                confidence=0.5,
            )
            self._hypotheses = [h]

        elif mode == "topic":
            self._hypotheses = generate_seed_hypotheses(
                self._config.seed, self._provider
            )
            self._budget.record_llm_call(1)

        elif mode == "reactive":
            self._hypotheses = hypotheses_from_graph_event(
                self._config.seed, self._provider
            )
            self._budget.record_llm_call(1)

    def _plan_phase(self) -> tuple[list[Lead], int]:
        """Generate new leads from current hypotheses. Returns (leads, llm_calls)."""
        if not self._hypotheses:
            return [], 0

        source_ids = tuple(self._sources.keys())
        leads = generate_leads(
            hypotheses=self._hypotheses,
            graph=self._graph,
            available_sources=source_ids,
            provider=self._provider,
            step=self._budget.steps,
        )
        return leads, 1

    def _gather_phase(
        self,
    ) -> tuple[list[DocumentEvidence], int, list[Finding], int]:
        """Execute top leads from queue. Returns (docs, pages, findings, llm_calls)."""
        batch = sorted(
            self._lead_queue[:_GATHER_BATCH_SIZE],
            key=lambda lead: lead.priority,
            reverse=True,
        )
        self._lead_queue = self._lead_queue[_GATHER_BATCH_SIZE:]

        all_docs: list[DocumentEvidence] = []
        total_pages = 0
        injection_findings: list[Finding] = []

        remaining_pages = self._config.budget.max_pages - self._budget.pages

        for lead in batch:
            source = self._sources.get(lead.source_id)
            if not source:
                _logger.warning("Source %r not available — skipping lead", lead.source_id)
                continue

            max_pages = min(10, remaining_pages)
            if max_pages <= 0:
                break

            result: SourceResult = source.search(query=lead.query, max_pages=max_pages)
            result = replace(result, lead_id=lead.id)

            all_docs.extend(result.documents)
            total_pages += result.pages_consumed
            remaining_pages -= result.pages_consumed

            # Record injection attempts as findings
            for finding_text in result.injection_findings:
                injection_findings.append(
                    Finding.create(
                        description=f"Injection attempt detected: {finding_text}",
                        confidence=1.0,
                        supporting_document_urls=tuple(
                            d.source_url for d in result.documents
                        ),
                        is_injection_finding=True,
                    )
                )

        # Deduplicate gathered documents
        if len(all_docs) > 1:
            from src.data.dedup import DedupIndex

            dedup_idx = DedupIndex(threshold=0.5)
            unique_docs: list[DocumentEvidence] = []
            for doc in all_docs:
                if not dedup_idx.is_duplicate(doc.text):
                    unique_docs.append(doc)
                    dedup_idx.add(doc.source_url or doc.title, doc.text)
            if len(unique_docs) < len(all_docs):
                _logger.info(
                    "Dedup: %d → %d documents", len(all_docs), len(unique_docs)
                )
            all_docs = unique_docs

        return all_docs, total_pages, injection_findings, 0

    def _analyze_phase(
        self,
        docs: list[DocumentEvidence],
    ) -> tuple[list[AnalysisResult], int, bool, list[AssumptionScanResult]]:
        """Analyze documents via 4-layer pipeline + assumption scan.

        Returns (results, llm_calls, halt, assumption_results).
        """
        results: list[AnalysisResult] = []
        llm_calls = 0
        halt = False

        # Build a duck-typed constitution object for verify_inline
        constitution_obj = _ConstitutionWrapper(self._constitution)

        for doc in docs:
            for hyp in self._hypotheses[:3]:  # Limit cross-product
                claim = f"{hyp.text} — Evidence: {doc.text[:500]}"
                try:
                    result = analyze(
                        claim=claim,
                        provider=self._provider,
                        graph=self._graph,
                        library=self._library,
                        constitution=constitution_obj,
                        phi_metrics=self._config.phi_metrics,
                    )
                    results.append(result)
                    llm_calls += 2  # fuse_reasoning + verify_inline

                    # Check for constitutional halt signal
                    if "HALT" in result.verdict.upper():
                        halt = True
                        break
                except Exception:
                    _logger.exception("Analysis failed for claim: %s", claim[:100])
                    llm_calls += 1

            if halt:
                break

        # Assumption scan (after pipeline, once per unique doc)
        assumption_results: list[AssumptionScanResult] = []
        if not halt:
            scan_results, findings, counter_leads, scan_llm = self._assumption_scan(docs)
            assumption_results = scan_results
            llm_calls += scan_llm
            self._findings.extend(findings)
            self._lead_queue.extend(counter_leads)
            self._assumption_results.extend(scan_results)
            detected = sum(len(r.detections) for r in scan_results)
            self._total_assumptions += detected

        return results, llm_calls, halt, assumption_results

    def _reflect_phase(
        self,
        analysis_results: list[AnalysisResult],
    ) -> tuple[int, bool]:
        """Constitutional critique of analysis results. Returns (llm_calls, halt)."""
        from src.detective.constitution import critique_against_constitution

        if not analysis_results:
            return 0, False

        llm_calls = 0
        halt = False

        # Critique a sample of analysis results (not all — budget efficiency)
        sample = analysis_results[:3]
        for result in sample:
            analysis_text = (
                f"Claim: {result.claim}\n"
                f"Verdict: {result.verdict}\n"
                f"Confidence: {result.confidence:.2f}\n"
                f"Reasoning: {' → '.join(result.reasoning.steps)}"
            )
            try:
                critique = critique_against_constitution(
                    analysis=analysis_text,
                    constitution=self._constitution,
                    critic_provider=self._provider,
                )
                llm_calls += 1

                # Check for halt signals in critique
                if any(
                    signal in critique.upper()
                    for signal in ("HALT", "STOP INVESTIGATION", "CONSTITUTIONAL VIOLATION")
                ):
                    halt = True
                    _logger.warning("Constitutional halt during reflect phase")
                    break

            except Exception:
                _logger.exception("Reflect phase critique failed")
                llm_calls += 1

        return llm_calls, halt

    async def _evolve_phase(
        self,
        docs: list[DocumentEvidence],
        analysis_results: list[AnalysisResult],
    ) -> tuple[int, list[Finding], int]:
        """Evolve hypotheses with evidence. Returns (llm_calls, findings, hypotheses_evolved)."""
        evidence_texts = [d.text[:1000] for d in docs if d.text.strip()]
        if not evidence_texts or not self._hypotheses:
            return 0, [], 0

        llm_calls = 0
        new_findings: list[Finding] = []
        evolved_hypotheses: list[Hypothesis] = []

        for hyp in self._hypotheses:
            try:
                results = await evolve_parallel(
                    hypothesis=hyp,
                    evidence_list=evidence_texts[:_EVOLUTION_K],
                    provider=self._provider,
                    k=_EVOLUTION_K,
                    library=self._library,
                    phi_metrics=self._config.phi_metrics,
                )
                llm_calls += len(results)

                for result in results:
                    evolved = result.hypothesis

                    # Welfare scoring
                    if self._config.phi_metrics:
                        constructs = infer_threatened_constructs(evolved.text)
                        welfare = score_hypothesis_welfare(evolved, self._config.phi_metrics)
                        evolved = replace(
                            evolved,
                            threatened_constructs=constructs,
                            welfare_relevance=welfare,
                        )

                    # Record experience
                    action = "confirmed" if evolved.confidence > hyp.confidence else "refuted"
                    exp = Experience(
                        hypothesis_id=hyp.id,
                        hypothesis_text=hyp.text,
                        evidence=result.evidence_used[:500],
                        action=action,
                        confidence_delta=evolved.confidence - hyp.confidence,
                        outcome_quality=evolved.confidence,
                    )
                    self._library = add_experience(self._library, exp)

                    evolved_hypotheses.append(evolved)

                    # Extract findings from high-confidence hypotheses
                    if evolved.confidence >= _FINDING_CONFIDENCE_THRESHOLD:
                        new_findings.append(
                            Finding.create(
                                description=evolved.text,
                                confidence=evolved.confidence,
                                supporting_hypothesis_ids=(evolved.id,),
                                supporting_document_urls=tuple(
                                    d.source_url for d in docs[:5]
                                ),
                                threatened_constructs=evolved.threatened_constructs,
                                welfare_relevance=evolved.welfare_relevance,
                            )
                        )

            except Exception:
                _logger.exception("Evolution failed for hypothesis %s", hyp.id)
                evolved_hypotheses.append(hyp)
                llm_calls += 1

        # Spawn alternatives for low-confidence hypotheses (breadth strategy)
        for hyp in evolved_hypotheses:
            if branching_rule(hyp.confidence) == "breadth":
                summary = "; ".join(r.verdict for r in analysis_results[:3])
                alternatives = spawn_alternatives(hyp, summary, self._provider)
                llm_calls += 1
                evolved_hypotheses.extend(alternatives)

        # Prune dead hypotheses
        self._hypotheses = [
            h for h in evolved_hypotheses
            if h.confidence >= _PRUNE_CONFIDENCE_THRESHOLD
        ]

        return llm_calls, new_findings, len(evolved_hypotheses)

    def _enrich_phase(self, docs: list[DocumentEvidence]) -> int:
        """Extract entities from documents and add edges to graph."""
        from src.core.types import RelationType
        from src.data.ner import extract_entities

        edges_added = 0
        all_entities: list[str] = []

        for doc in docs:
            ner_result = extract_entities(doc.text)
            for ent in ner_result.entities:
                if ent.label in ("PERSON", "ORG"):
                    all_entities.append(ent.text)

        # Filter through entity filter
        clean_entities = filter_entities(all_entities)

        # Add CO_MENTIONED edges between entities appearing in same batch
        seen: list[str] = []
        for entity in clean_entities[:50]:  # Limit to prevent graph explosion
            for other in seen:
                if entity != other:
                    self._graph.add_edge(
                        source=entity,
                        target=other,
                        relation=RelationType.CO_MENTIONED,
                        confidence=0.5,
                    )
                    edges_added += 1
            seen.append(entity)

        return edges_added

    # -------------------------------------------------------------------
    # Assumption scanning
    # -------------------------------------------------------------------

    def _assumption_scan(
        self,
        docs: list[DocumentEvidence],
    ) -> tuple[list[AssumptionScanResult], list[Finding], list[Lead], int]:
        """Scan documents for assumptions via modules A/B/C.

        Returns (scan_results, findings, counter_leads, llm_calls).
        """
        if not self._config.enable_assumption_scan:
            return [], [], [], 0

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_docs: list[DocumentEvidence] = []
        for doc in docs:
            if doc.source_url not in seen_urls:
                seen_urls.add(doc.source_url)
                unique_docs.append(doc)

        threshold = self._config.assumption_threshold
        scan_results: list[AssumptionScanResult] = []
        all_findings: list[Finding] = []
        all_detections: list[AssumptionDetection] = []
        total_llm_calls = 0
        findings_this_scan = 0

        for doc in unique_docs[:_ASSUMPTION_SCAN_MAX_DOCS]:
            if self._budget.check() is not None:
                break

            text = doc.text[:_ASSUMPTION_TEXT_LIMIT]
            doc_detections: list[AssumptionDetection] = []
            doc_llm_calls = 0
            remaining = self._budget.budget.max_llm_calls - self._budget.llm_calls - total_llm_calls

            # Module A: cognitive biases (keyword-only when budget < 10)
            try:
                use_provider = self._provider if remaining >= 10 else None
                bias_results = detect_cognitive_biases(
                    text, provider=use_provider, threshold=threshold,
                )
                if use_provider is not None:
                    doc_llm_calls += len(bias_results)
                for b in bias_results:
                    doc_detections.append(AssumptionDetection(
                        module="A",
                        assumption_type=b.assumption_type,
                        score=b.score,
                        source_text=b.source_text[:300],
                        detail=b.bias_type,
                    ))
            except Exception:
                _logger.exception("Module A scan failed for %s", doc.source_url)

            remaining = self._budget.budget.max_llm_calls - self._budget.llm_calls - total_llm_calls - doc_llm_calls

            # Module B: historical determinism (requires provider)
            if remaining >= 5:
                try:
                    det_results = detect_historical_determinism(
                        text, provider=self._provider, threshold=threshold,
                    )
                    doc_llm_calls += len(det_results)
                    for d in det_results:
                        doc_detections.append(AssumptionDetection(
                            module="B",
                            assumption_type=d.assumption_type,
                            score=d.score,
                            source_text=d.source_text[:300],
                            detail=d.trigger_phrase,
                        ))
                except Exception:
                    _logger.exception("Module B scan failed for %s", doc.source_url)

            remaining = self._budget.budget.max_llm_calls - self._budget.llm_calls - total_llm_calls - doc_llm_calls

            # Module C: geopolitical presumptions (requires provider)
            if remaining >= 5:
                try:
                    geo_results = detect_geopolitical_presumptions(
                        text, provider=self._provider, threshold=threshold,
                    )
                    doc_llm_calls += len(geo_results)
                    for g in geo_results:
                        doc_detections.append(AssumptionDetection(
                            module="C",
                            assumption_type=g.assumption_type,
                            score=g.score,
                            source_text=g.source_text[:300],
                            detail=g.presumed_actor,
                        ))
                except Exception:
                    _logger.exception("Module C scan failed for %s", doc.source_url)

            scan_results.append(AssumptionScanResult(
                document_url=doc.source_url,
                detections=tuple(doc_detections),
                llm_calls=doc_llm_calls,
            ))
            total_llm_calls += doc_llm_calls
            all_detections.extend(doc_detections)

            # Create findings (capped per scan)
            for det in doc_detections:
                if findings_this_scan >= _ASSUMPTION_MAX_FINDINGS_PER_SCAN:
                    break
                all_findings.append(Finding.create(
                    description=(
                        f"[{det.module}] {det.assumption_type.value}: {det.detail} "
                        f"(score {det.score:.2f})"
                    ),
                    confidence=det.score,
                    supporting_document_urls=(doc.source_url,),
                    is_assumption_finding=True,
                    assumption_module=det.module,
                ))
                findings_this_scan += 1

        # Generate counter-leads
        counter_leads, lead_llm = self._generate_counter_leads(
            all_detections, step=self._budget.steps,
        )
        total_llm_calls += lead_llm

        return scan_results, all_findings, counter_leads, total_llm_calls

    def _generate_counter_leads(
        self,
        detections: list[AssumptionDetection],
        step: int,
    ) -> tuple[list[Lead], int]:
        """Generate counter-leads to probe detected assumptions.

        Returns (leads, llm_calls).
        """
        if not detections:
            return [], 0

        leads: list[Lead] = []
        llm_calls = 0

        # Sort by score descending, take top N
        sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)

        for det in sorted_dets[:_MAX_COUNTER_LEADS]:
            if self._budget.check() is not None:
                break

            prompt = _COUNTER_LEAD_PROMPT.format(
                module=det.module,
                assumption_type=det.assumption_type.value,
                detail=det.detail,
                source_text=det.source_text[:300],
            )

            try:
                response = self._provider.complete(prompt)
                llm_calls += 1
            except Exception:
                _logger.exception("Counter-lead generation failed")
                llm_calls += 1
                continue

            # Parse query from response
            query = response.strip()
            if "query:" in query.lower():
                query = query.split(":", 1)[1].strip()
            if not query:
                continue

            # Route to best source
            source_id = self._route_counter_lead(det.module)

            leads.append(Lead.create(
                query=query,
                source_id=source_id,
                priority=det.score * 0.8,
                generation_step=step,
            ))

        return leads, llm_calls

    def _route_counter_lead(self, module: str) -> str:
        """Pick the best source for a counter-lead based on module type."""
        available = set(self._sources.keys())

        if module == "C":
            # Geopolitical → prefer institutional sources
            for source_id in ("court_listener", "sec_edgar", "web_search"):
                if source_id in available:
                    return source_id
        else:
            # Modules A/B → prefer web search
            for source_id in ("web_search", "news_search"):
                if source_id in available:
                    return source_id

        # Fallback: first available source
        if available:
            return next(iter(available))
        return "web_search"

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _record_step(
        self,
        action: str,
        leads_generated: int = 0,
        documents_gathered: int = 0,
        hypotheses_evolved: int = 0,
        findings_produced: int = 0,
        pages_consumed: int = 0,
        llm_calls: int = 0,
        assumptions_detected: int = 0,
    ) -> None:
        step = InvestigationStep(
            step_number=len(self._steps),
            action=action,  # type: ignore[arg-type]
            timestamp=datetime.now(timezone.utc),
            leads_generated=leads_generated,
            documents_gathered=documents_gathered,
            hypotheses_evolved=hypotheses_evolved,
            findings_produced=findings_produced,
            pages_consumed=pages_consumed,
            llm_calls=llm_calls,
            assumptions_detected=assumptions_detected,
        )
        self._steps.append(step)

    def _build_report(self, reason: TerminationReason) -> InvestigationReport:
        """Assemble the final investigation report."""
        weights = WEIGHTS_BRIDGE if self._config.phi_metrics else WEIGHTS_DEFAULT

        snapshots = tuple(
            HypothesisSnapshot(
                id=h.id,
                text=h.text,
                confidence=h.confidence,
                parent_id=h.parent_id,
                welfare_relevance=h.welfare_relevance,
                threatened_constructs=h.threatened_constructs,
                combined_score=h.combined_score(**weights),
            )
            for h in self._hypotheses
        )

        return InvestigationReport(
            config=self._config,
            findings=tuple(self._findings),
            hypothesis_tree=snapshots,
            steps=tuple(self._steps),
            total_pages=self._budget.pages,
            total_llm_calls=self._budget.llm_calls,
            total_documents=self._total_documents,
            elapsed_seconds=self._budget.elapsed,
            termination_reason=reason,
            graph_edges_added=self._graph_edges_added,
            total_assumptions_detected=self._total_assumptions,
        )

    @property
    def status(self) -> dict:
        """Return current investigation status for polling."""
        return {
            "id": self._config.id,
            "steps": self._budget.steps,
            "findings": len(self._findings),
            "hypotheses": len(self._hypotheses),
            "pages": self._budget.pages,
            "llm_calls": self._budget.llm_calls,
            "elapsed_seconds": round(self._budget.elapsed, 1),
            "running": self._budget.check() is None,
            "assumptions_detected": self._total_assumptions,
        }


class _ConstitutionWrapper:
    """Duck-typed wrapper so verify_inline() can call .critique()."""

    def __init__(self, text: str) -> None:
        self._text = text

    def critique(self, analysis: str) -> str:
        return self._text
