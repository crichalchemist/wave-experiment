1#!/usr/bin/env python3
"""Run constitutional warmup with welfare filtering."""

import os
import sys
from pathlib import Path

# Set up environment
os.environ.setdefault("AZURE_MODEL", "claude-sonnet-4-5-2")

# Import after environment is set
from src.training.constitutional_warmup import (
    run_constitutional_warmup,
    ConstitutionalWarmupConfig,
)
from src.core.providers import AzureFoundryProvider, MockProvider

def main():
    print("\n" + "="*70)
    print("CONSTITUTIONAL WARMUP WITH WELFARE FILTERING")
    print("="*70)

    # Check environment variables
    azure_endpoint = os.environ.get("AZURE_ENDPOINT") or os.environ.get("AZURE_CRITIC_ENDPOINT")
    azure_key = os.environ.get("AZURE_API_KEY") or os.environ.get("AZURE_CRITIC_KEY")
    azure_model = os.environ.get("AZURE_MODEL", "claude-sonnet-4-5-2")

    if not azure_endpoint or not azure_key:
        print("\n⚠️  Missing Azure credentials. Using MockProvider for testing.")
        print("   Set AZURE_ENDPOINT and AZURE_API_KEY for live critique.\n")
        local_provider = MockProvider(response="Analysis: gap detected")
        critic_provider = MockProvider(response="Critique: meets standards")
    else:
        print(f"\n✓ Azure Endpoint: {azure_endpoint[:50]}...")
        print(f"✓ Azure Model: {azure_model}\n")

        local_provider = MockProvider(response="Analysis: gap detected")
        critic_provider = AzureFoundryProvider(
            endpoint=azure_endpoint,
            api_key=azure_key,
            model=azure_model,
        )

    # Configure warmup
    output_path = "data/training/constitutional_pairs_test.jsonl"
    document_file = "data/training/warmup_test_data.txt"

    config = ConstitutionalWarmupConfig(
        output_path=output_path,
        max_examples=5,  # Small test run
        constitution_path="docs/constitution.md",
        document_file=document_file,  # NEW: Load from test data file
        use_huggingface=False,  # Disable external sources for testing
        use_doj=False,
        use_international=False,
    )

    print("Configuration:")
    print(f"  Output: {output_path}")
    print(f"  Document file: {document_file}")
    print(f"  Max examples: {config.max_examples}")
    print(f"  Constitution: {config.constitution_path}")
    print(f"\nWelfare filtering enabled:")
    print(f"  Threshold: 0.3 (default)")
    print(f"  Phi metrics: all at 0.5 (conservative defaults)")
    print("\n" + "="*70)
    print("Starting warmup...\n")

    try:
        count = run_constitutional_warmup(
            cfg=config,
            local_provider=local_provider,
            critic_provider=critic_provider,
        )

        print("\n" + "="*70)
        print("WARMUP COMPLETE")
        print("="*70)
        print(f"\nGenerated {count} constitutional preference pairs")
        print(f"Output: {output_path}")

        # Check output file
        if Path(output_path).exists():
            size = Path(output_path).stat().st_size
            print(f"File size: {size} bytes")

            with open(output_path) as f:
                lines = f.readlines()
                print(f"Lines written: {len(lines)}")

        print("\n✅ Constitutional warmup with welfare filtering successful!\n")

    except Exception as e:
        print(f"\n❌ Error during warmup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
