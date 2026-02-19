from __future__ import annotations
import sys
from pathlib import Path

import click

from src.core.providers import provider_from_env
from src.detective.constitution import load_constitution
from src.security.sanitizer import sanitize_document

_RISK_LEVELS_WARN: frozenset[str] = frozenset({"high", "critical"})


@click.group()
def cli() -> None:
    """Detective LLM — information gap analysis for investigative datasets."""


@cli.command()
@click.argument("doc_path", type=click.Path(exists=True, readable=True, path_type=Path))
@click.option("--query", "-q", default="What information is absent from this document?",
              help="The analytical question to investigate.")
@click.option("--constitution", "-c", default="docs/constitution.md",
              help="Path to the moral compass constitution.")
def analyze(doc_path: Path, query: str, constitution: str) -> None:
    """Analyze a document for information gaps."""
    text = doc_path.read_text(encoding="utf-8")
    result = sanitize_document(text)

    if result.injection_detected and result.risk_level in _RISK_LEVELS_WARN:
        click.echo(
            f"WARNING: Injection detected in document ({result.risk_level} risk). "
            f"Findings: {', '.join(result.findings)}",
            err=True,
        )

    constitution_text = load_constitution(Path(constitution))
    provider = provider_from_env()
    from src.security.prompt_guard import build_analysis_prompt
    prompt = build_analysis_prompt(
        document_text=result.safe_text,
        constitution=constitution_text,
        query=query,
    )
    response = provider.complete(prompt)
    click.echo(response)


@cli.command()
@click.argument("doc_path", type=click.Path(exists=True, readable=True, path_type=Path))
def network(doc_path: Path) -> None:
    """Build and display the knowledge graph entities from a document."""
    text = doc_path.read_text(encoding="utf-8")
    result = sanitize_document(text)

    if result.injection_detected:
        click.echo(
            f"NOTE: Injection patterns detected ({result.risk_level}). "
            "Document content is sandboxed.",
            err=True,
        )

    # Entity extraction placeholder — wired to NER in a later phase
    click.echo(f"Document: {doc_path.name}")
    click.echo(f"Characters: {len(result.safe_text)}")
    click.echo(f"Injection detected: {result.injection_detected}")
    if result.findings:
        click.echo(f"Security findings: {', '.join(result.findings)}")


@cli.command()
@click.argument("analysis_text")
@click.option("--constitution", "-c", default="docs/constitution.md",
              help="Path to the moral compass constitution.")
def critique(analysis_text: str, constitution: str) -> None:
    """Run constitutional self-critique on an analysis."""
    # The analysis_text itself is treated as potential injection surface
    result = sanitize_document(analysis_text)
    constitution_text = load_constitution(Path(constitution))
    critic = provider_from_env()
    from src.detective.constitution import critique_against_constitution
    response = critique_against_constitution(
        analysis=result.safe_text,
        constitution=constitution_text,
        critic_provider=critic,
    )
    click.echo(response)


@cli.command()
@click.option("--output", default="data/training/constitutional_pairs.jsonl", help="Output JSONL path")
@click.option("--max-examples", default=200, help="Maximum preference pairs to generate")
@click.option("--constitution", default="docs/constitution.md", help="Constitution path")
def warmup(output: str, max_examples: int, constitution: str) -> None:
    """Generate constitutional preference pairs (run before SFT)."""
    from src.training.constitutional_warmup import run_constitutional_warmup, ConstitutionalWarmupConfig
    from src.core.providers import AzureFoundryProvider
    import os

    local = provider_from_env()
    critic_endpoint = os.environ.get("AZURE_ENDPOINT")
    critic_key = os.environ.get("AZURE_API_KEY")
    critic_model = os.environ.get("AZURE_MODEL", "claude-3-5-sonnet")

    if not critic_endpoint:
        raise click.ClickException("Set AZURE_ENDPOINT for the constitutional critic (Claude).")

    critic = AzureFoundryProvider(endpoint=critic_endpoint, credential=critic_key, model=critic_model)
    cfg = ConstitutionalWarmupConfig(
        output_path=output,
        max_examples=max_examples,
        constitution_path=constitution,
    )
    count = run_constitutional_warmup(cfg, local_provider=local, critic_provider=critic)
    click.echo(f"Generated {count} constitutional preference pairs → {output}")
