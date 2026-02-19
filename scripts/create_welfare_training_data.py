#!/usr/bin/env python3
"""
Generate training data for welfare classifier using Claude API.

Usage:
    python scripts/create_welfare_training_data.py \
        --corpus data/training/smiles_and_cries_extracted.txt \
        --output data/training/welfare_training_data.jsonl \
        --n-samples 1000

Cost: ~$10, Time: ~20 minutes
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.providers import critic_provider_from_env

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_critic_provider_env():
    """
    Map ANTHROPIC_FOUNDRY_* to AZURE_CRITIC_* for critic_provider_from_env().

    This allows the script to work with the standard Foundry variables
    without requiring separate AZURE_CRITIC_* configuration.
    """
    import os

    # Map ANTHROPIC_FOUNDRY_* → AZURE_CRITIC_* if AZURE_CRITIC_* not already set
    if not os.getenv('AZURE_CRITIC_ENDPOINT'):
        foundry_resource = os.getenv('ANTHROPIC_FOUNDRY_RESOURCE')
        if foundry_resource:
            os.environ['AZURE_CRITIC_ENDPOINT'] = f"https://{foundry_resource}.services.ai.azure.com/"

    if not os.getenv('AZURE_CRITIC_KEY'):
        foundry_key = os.getenv('ANTHROPIC_FOUNDRY_API_KEY')
        if foundry_key:
            os.environ['AZURE_CRITIC_KEY'] = foundry_key

    if not os.getenv('AZURE_CRITIC_MODEL'):
        foundry_model = os.getenv('ANTHROPIC_DEFAULT_SONNET_MODEL', 'claude-sonnet-4-5-2')
        os.environ['AZURE_CRITIC_MODEL'] = foundry_model


LABELING_PROMPT = """You are analyzing text for welfare relevance across 8 constructs from the Φ(humanity) framework.

For EACH construct below, rate 0.0-1.0 how strongly this text addresses threats to that construct:
- 0.0 = Not relevant at all
- 0.3 = Mentioned tangentially
- 0.5 = Moderately relevant
- 0.7 = Substantially relevant
- 1.0 = Core focus

CONSTRUCTS (from docs/humanity-phi-formalized.md):

1. **Care (c)**: Resource allocation meeting basic needs
   Examples: poverty, deprivation, healthcare access, housing, education

2. **Compassion (kappa)**: Responsive support to acute distress
   Examples: crisis response, emergency aid, disaster relief

3. **Joy (j)**: Positive affect above subsistence
   Examples: wellbeing, life satisfaction, happiness

4. **Purpose (p)**: Alignment of actions with chosen goals
   Examples: autonomy, agency, self-determination, meaning

5. **Empathy (eps)**: Perspective-taking across groups
   Examples: intergroup understanding, discrimination, bias, othering

6. **Love (lam_L)**: Active extension for growth (bell hooks)
   Examples: developmental support, mutual aid, capacity building, nurturing

7. **Protection (lam_P)**: Safeguarding from harm
   Examples: violence prevention, safety, security, rights protection

8. **Truth (xi)**: Epistemic integrity
   Examples: suppression, concealment, falsification, contradictions

TEXT TO ANALYZE:
{text}

Return ONLY a JSON object with scores:
{{"c": 0.0, "kappa": 0.0, "j": 0.0, "p": 0.0, "eps": 0.0, "lam_L": 0.0, "lam_P": 0.0, "xi": 0.0}}"""


def load_corpus(corpus_path: Path) -> List[Dict[str, Any]]:
    """
    Load corpus from extracted text file.

    Format: Each chunk separated by "="*70
    Metadata in comments: # Source: filename | Chunk: N/M
    """
    with open(corpus_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by separator
    chunks = content.split("=" * 70)

    parsed_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Extract metadata from comment lines
        lines = chunk.split('\n')
        metadata = {}
        text_lines = []

        for line in lines:
            if line.startswith('# Source:'):
                # Parse: # Source: filename | Chunk: N/M
                parts = line[len('# Source:'):].split('|')
                metadata['source'] = parts[0].strip()
                if len(parts) > 1 and 'Chunk:' in parts[1]:
                    chunk_info = parts[1].split(':')[1].strip()
                    metadata['chunk_info'] = chunk_info
            elif not line.startswith('#'):
                text_lines.append(line)

        text = '\n'.join(text_lines).strip()
        # Only add chunks that have both metadata and text
        if text and metadata:
            parsed_chunks.append({
                'text': text,
                'metadata': metadata
            })

    logger.info(f"Loaded {len(parsed_chunks)} chunks from {corpus_path}")
    return parsed_chunks


def stratified_sample(
    chunks: List[Dict[str, Any]],
    n_samples: int = 1000
) -> List[Dict[str, Any]]:
    """
    Stratified sampling for diversity:
    - Balanced across source documents
    - Mixed lengths (short/medium/long)
    """
    if not chunks:
        logger.error("No chunks to sample from!")
        return []

    # Group by source
    by_source = defaultdict(list)
    for chunk in chunks:
        source = chunk['metadata'].get('source', 'unknown')
        by_source[source].append(chunk)

    logger.info(f"Found {len(by_source)} unique sources")

    if len(by_source) == 0:
        logger.error("No sources found!")
        return []

    # Calculate samples per source
    samples_per_source = n_samples // len(by_source)
    samples = []

    for source, source_chunks in by_source.items():
        # Stratify by length
        short = [c for c in source_chunks if len(c['text']) < 1000]
        medium = [c for c in source_chunks if 1000 <= len(c['text']) < 1500]
        long_chunks = [c for c in source_chunks if len(c['text']) >= 1500]

        # Sample proportionally
        n_short = min(len(short), samples_per_source // 3)
        n_medium = min(len(medium), samples_per_source // 3)
        n_long = min(len(long_chunks), samples_per_source // 3)

        if short:
            samples.extend(random.sample(short, n_short))
        if medium:
            samples.extend(random.sample(medium, n_medium))
        if long_chunks:
            samples.extend(random.sample(long_chunks, n_long))

    # If we didn't get enough, sample more randomly
    if len(samples) < n_samples:
        remaining = n_samples - len(samples)
        available = [c for c in chunks if c not in samples]
        samples.extend(random.sample(available, min(remaining, len(available))))

    # Shuffle to avoid source clustering
    random.shuffle(samples)

    logger.info(f"Sampled {len(samples)} diverse examples")
    return samples[:n_samples]


def label_with_claude(
    text: str,
    provider,
    max_retries: int = 3
) -> Dict[str, float] | None:
    """Label single example with Claude API."""
    prompt = LABELING_PROMPT.format(text=text[:2000])

    for attempt in range(max_retries):
        try:
            response = provider.complete(
                prompt,
                max_tokens=150,
                temperature=0.0
            )

            # Parse JSON - handle markdown code blocks
            response_clean = response.strip()
            if response_clean.startswith('```'):
                # Extract JSON from markdown code block
                lines = response_clean.split('\n')
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                response_clean = '\n'.join(json_lines)

            scores = json.loads(response_clean)

            # Validate structure
            required = {"c", "kappa", "j", "p", "eps", "lam_L", "lam_P", "xi"}
            if not required.issubset(scores.keys()):
                missing = required - scores.keys()
                logger.warning(f"Missing constructs: {missing}, retrying...")
                continue

            # Validate ranges
            all_valid = True
            for construct, score in scores.items():
                if not 0.0 <= score <= 1.0:
                    logger.warning(f"{construct} out of range: {score}, retrying...")
                    all_valid = False
                    break

            if not all_valid:
                continue

            return scores

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt+1}: JSON parse failed: {e}")
            logger.warning(f"Response was: {response[:200]}")
            if attempt == max_retries - 1:
                logger.error(f"Failed after {max_retries} attempts")
                return None
            time.sleep(2 ** attempt)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)

    return None


def label_all_samples(
    samples: List[Dict[str, Any]],
    provider,
    checkpoint_path: Path | None = None
) -> List[Dict[str, Any]]:
    """
    Label all samples with rate limiting and checkpointing.

    Rate limit: 50 requests/minute = 1.2s spacing
    """
    # Load checkpoint if exists
    labeled = []
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            labeled = [json.loads(line) for line in f]
        logger.info(f"Resumed from checkpoint: {len(labeled)} already labeled")

    start_idx = len(labeled)
    failed_indices = []

    # Process samples
    for i, sample in enumerate(samples):
        if i < start_idx:
            continue

        logger.info(f"Labeling {i+1}/{len(samples)}: {sample['metadata'].get('source', 'unknown')}")

        scores = label_with_claude(sample['text'], provider)

        if scores:
            result = {
                'text': sample['text'],
                'scores': scores,
                'metadata': sample['metadata'],
                'idx': i
            }
            labeled.append(result)

            # Checkpoint every 100
            if len(labeled) % 100 == 0 and checkpoint_path:
                save_checkpoint(labeled, checkpoint_path)
        else:
            failed_indices.append(i)

        # Rate limiting
        time.sleep(1.2)

    logger.info(f"Labeled: {len(labeled)}, Failed: {len(failed_indices)}")

    # Retry failures
    if failed_indices:
        logger.info(f"Retrying {len(failed_indices)} failed samples...")
        for idx in failed_indices:
            sample = samples[idx]
            logger.info(f"Retry {idx+1}: {sample['metadata'].get('source', 'unknown')}")
            scores = label_with_claude(sample['text'], provider)
            if scores:
                result = {
                    'text': sample['text'],
                    'scores': scores,
                    'metadata': sample['metadata'],
                    'idx': idx
                }
                labeled.append(result)
            time.sleep(1.2)

    return labeled


def save_checkpoint(labeled: List[Dict], checkpoint_path: Path):
    """Save checkpoint."""
    with open(checkpoint_path, 'w') as f:
        for item in labeled:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Checkpoint saved: {len(labeled)} examples")


def main():
    parser = argparse.ArgumentParser(description='Generate welfare training data')
    parser.add_argument('--corpus', type=Path, required=True,
                       help='Path to extracted corpus')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output JSONL file')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples to label')
    parser.add_argument('--checkpoint', type=Path, default=None,
                       help='Checkpoint file for resume capability')

    args = parser.parse_args()

    # Setup environment
    setup_critic_provider_env()

    # Load corpus
    chunks = load_corpus(args.corpus)

    # Sample
    samples = stratified_sample(chunks, args.n_samples)

    logger.info(f"Will label {len(samples)} examples")
    logger.info(f"Estimated cost: $8-10")
    logger.info(f"Estimated time: 20 minutes")

    # Load Claude provider
    logger.info("Loading Claude provider from environment...")
    provider = critic_provider_from_env()

    # Checkpoint path
    checkpoint_path = args.checkpoint or args.output.with_suffix('.checkpoint.jsonl')

    # Label all samples
    labeled = label_all_samples(samples, provider, checkpoint_path)

    # Save final output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for item in labeled:
            f.write(json.dumps(item) + '\n')

    logger.info(f"✓ Saved {len(labeled)} labeled examples to {args.output}")
    logger.info(f"✓ Training data generation complete!")


if __name__ == '__main__':
    main()
