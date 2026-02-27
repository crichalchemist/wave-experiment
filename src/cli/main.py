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
@click.option("--document-file", default=None, help="Text file with pre-extracted documents (one per line/chunk)")
@click.option("--local-only", is_flag=True, help="Skip remote sources (HF, DOJ, international)")
def warmup(output: str, max_examples: int, constitution: str, document_file: str | None, local_only: bool) -> None:
    """Generate constitutional preference pairs (run before SFT)."""
    from src.training.constitutional_warmup import run_constitutional_warmup, ConstitutionalWarmupConfig
    from src.core.providers import critic_provider_from_env

    local = provider_from_env()
    critic = critic_provider_from_env()
    cfg = ConstitutionalWarmupConfig(
        output_path=output,
        max_examples=max_examples,
        constitution_path=constitution,
        document_file=document_file,
        use_huggingface=not local_only,
        use_doj=not local_only,
        use_international=not local_only,
    )
    count = run_constitutional_warmup(cfg, local_provider=local, critic_provider=critic)
    click.echo(f"Generated {count} constitutional preference pairs → {output}")


@cli.command()
@click.option("--data", default="data/training/constitutional_pairs.jsonl",
              help="JSONL file with preference pairs")
@click.option("--model-id", default=None,
              help="HuggingFace model ID (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)")
@click.option("--output-dir", default="checkpoints/dpo", help="Directory for checkpoints")
@click.option("--eval-split", default=0.1, type=float, help="Fraction of data for evaluation")
def dpo(data: str, model_id: str | None, output_dir: str, eval_split: float) -> None:
    """Run DPO training on constitutional preference pairs."""
    from src.training.train_dpo import (
        load_preference_pairs,
        preference_pairs_to_dataset,
        build_dpo_trainer,
        DEFAULT_MODEL_ID,
        DPO_OUTPUT_DIR,
    )

    model_id = model_id or DEFAULT_MODEL_ID

    click.echo(f"Loading preference pairs from {data}")
    samples = load_preference_pairs(data)
    click.echo(f"Loaded {len(samples)} preference pairs")

    dataset = preference_pairs_to_dataset(samples)

    # Train/eval split
    if eval_split > 0 and len(samples) > 10:
        split = dataset.train_test_split(test_size=eval_split, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
        click.echo(f"Split: {len(train_ds)} train, {len(eval_ds)} eval")
    else:
        train_ds, eval_ds = dataset, None

    click.echo(f"Loading model: {model_id}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    click.echo("Building DPO trainer with LoRA")
    trainer = build_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    click.echo("Starting DPO training...")
    trainer.train()

    save_path = output_dir or DPO_OUTPUT_DIR
    trainer.save_model(save_path)
    click.echo(f"Training complete. Model saved to {save_path}")


@cli.command("legal-warmup")
@click.option("--output", default="data/training/legal_pairs.jsonl", help="Output JSONL path")
@click.option("--examples-per-domain", default=200, type=int, help="Pairs per legal domain")
@click.option("--domains", default="criminal_justice,territorial_rights,foia_transparency",
              help="Comma-separated legal domains")
@click.option("--constitution", default="docs/constitution.md", help="Constitution path")
def legal_warmup(output: str, examples_per_domain: int, domains: str, constitution: str) -> None:
    """Generate legal domain DPO preference pairs (written vs. applied)."""
    from src.training.legal_warmup import run_legal_warmup, LegalWarmupConfig
    from src.core.providers import critic_provider_from_env

    local = provider_from_env()
    critic = critic_provider_from_env()

    domain_tuple = tuple(d.strip() for d in domains.split(","))

    cfg = LegalWarmupConfig(
        output_path=output,
        examples_per_domain=examples_per_domain,
        domains=domain_tuple,
        constitution_path=constitution,
    )
    count = run_legal_warmup(cfg, local_provider=local, critic_provider=critic)
    click.echo(f"Generated {count} legal preference pairs → {output}")


@cli.command("scrape-foia")
@click.option("--portal", type=click.Choice(["fbi_vault", "nara", "state_dept"]),
              required=True, help="FOIA portal to scrape")
@click.option("--collection", type=str, default=None,
              help="Specific collection to crawl (e.g., 'jeffrey-epstein')")
@click.option("--query", type=str, default=None,
              help="Search query for the portal")
@click.option("--max-pages", type=int, default=100, help="Maximum pages to crawl")
@click.option("--output", "-o", type=click.Path(), default="data/foia",
              help="Output directory for scraped documents")
@click.option("--ingest/--no-ingest", default=True,
              help="Run document ingestion (OCR + text extraction) after scraping")
def scrape_foia(portal, collection, query, max_pages, output, ingest):
    """Scrape FOIA portal for investigation evidence and training data."""
    import json
    from pathlib import Path
    from src.data.sourcing.foia_scraper import FOIAScraper

    scraper = FOIAScraper(portal=portal, output_dir=output)

    click.echo(f"Crawling {portal}...")
    documents = scraper.crawl(collection=collection, query=query, max_pages=max_pages)
    click.echo(f"Found {len(documents)} documents")

    if ingest and documents:
        click.echo("Downloading and ingesting documents...")
        documents = scraper.download_and_ingest(documents)
        with_text = [d for d in documents if d.text]
        click.echo(f"Ingested {len(with_text)} documents with extracted text")

    # Save document index
    index_path = Path(output) / portal / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump([
            {"title": d.title, "url": d.url, "collection": d.collection,
             "has_text": bool(d.text), "date": d.date}
            for d in documents
        ], f, indent=2)
    click.echo(f"Saved document index to {index_path}")


@cli.command("ingest")
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--evidence/--no-evidence", default=True, help="Run evidence pipeline")
@click.option("--training/--no-training", default=True, help="Run training pipeline")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output JSONL path for training data")
def ingest(input_dir, evidence, training, output):
    """Process scraped FOIA documents through evidence + training pipelines.

    Reads extracted text from INPUT_DIR, runs welfare scoring on each document,
    and outputs evidence results and/or training data JSONL.
    """
    import json
    from pathlib import Path
    from src.data.sourcing.foia_scraper import FOIADocument
    from src.data.sourcing.dual_pipeline import run_dual_pipeline

    input_path = Path(input_dir)

    # Find text files to process
    text_files = sorted(input_path.glob("**/*.txt"))
    if not text_files:
        click.echo(f"No .txt files found in {input_dir}")
        return

    # Build FOIADocument list from text files
    docs = []
    for text_file in text_files:
        text = text_file.read_text(encoding="utf-8", errors="replace")
        docs.append(FOIADocument(
            source_portal="local",
            title=text_file.stem,
            url=str(text_file),
            date=None,
            collection=None,
            text=text,
            pdf_path=None,
        ))

    click.echo(f"Processing {len(docs)} documents from {input_dir}...")
    result = run_dual_pipeline(docs)

    if evidence:
        click.echo(f"  Evidence results: {len(result['evidence'])} documents scored")

    if training:
        output_path = output or "data/training/foia_welfare_scored.jsonl"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for item in result["training"]:
                f.write(json.dumps(item) + "\n")
        click.echo(f"  Training data: {len(result['training'])} examples -> {output_path}")


@cli.command("extract-scenarios")
@click.argument("corpus_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default="spaces/maninagarden/extracted_scenarios.json",
              show_default=True, help="Output JSON path for extracted scenario templates")
@click.option("--length", type=int, default=200, help="Scenario trajectory length")
def extract_scenarios(corpus_path: str, output: str, length: int) -> None:
    """Extract welfare trajectory patterns from a text corpus.

    Runs the extraction pipeline on CORPUS_PATH, identifies trajectory
    patterns in construct scores, and saves scenario templates for the forecaster.
    """
    import json
    from src.inference.scenario_extraction import run_extraction_pipeline

    click.echo(f"Extracting scenarios from {corpus_path}...")
    result = run_extraction_pipeline(corpus_path, scenario_length=length)

    click.echo(f"  Profiles: {len(result['profiles'])}")
    click.echo(f"  Patterns: {len(result['patterns'])}")
    click.echo(f"  Scenarios: {len(result['scenarios'])}")

    templates = result["patterns"]
    with open(output, "w") as f:
        json.dump(templates, f, indent=2)

    click.echo(f"Saved {len(templates)} scenario templates to {output}")
