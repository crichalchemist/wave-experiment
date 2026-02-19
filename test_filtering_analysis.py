#!/usr/bin/env python3
"""Analyze which documents were filtered by welfare scoring."""

from src.training.constitutional_warmup import should_include_example
from src.inference.welfare_scoring import infer_threatened_constructs, score_hypothesis_welfare
from src.detective.hypothesis import Hypothesis

# Test data from warmup_test_data.txt
test_docs = [
    "Evidence of resource deprivation affecting vulnerable communities 2013-2017.",
    "Meeting scheduled for Tuesday at 3pm in conference room B.",
    "Testimony about ongoing violence against marginalized populations.",
    "Administrative note about parking permits for staff.",
    "Suppressed testimony about exploitation and abuse patterns.",
    "Quarterly financial report showing routine expenditures.",
    "Evidence of concealed records regarding protective services.",
]

# Phi metrics used in warmup (all at 0.5)
phi_metrics = {
    "c": 0.5,
    "kappa": 0.5,
    "j": 0.5,
    "p": 0.5,
    "eps": 0.5,
    "lam": 0.5,
    "xi": 0.5,
}

threshold = 0.3

print("\n" + "="*70)
print("WELFARE FILTERING ANALYSIS")
print("="*70)
print(f"\nPhi metrics: all at 0.5 (baseline)")
print(f"Welfare threshold: {threshold}")
print(f"Max examples: 5 (processed first 5 documents)")
print("\n" + "="*70)

for i, text in enumerate(test_docs[:5], 1):  # Only first 5 processed
    constructs = infer_threatened_constructs(text)
    should_include = should_include_example(text, phi_metrics, threshold)

    # Get detailed scoring
    h = Hypothesis.create(text[:100], 0.5)
    welfare_score = score_hypothesis_welfare(h, phi_metrics) if constructs else 0.0

    status = "✓ INCLUDED" if should_include else "✗ FILTERED"

    print(f"\nLine {i}: {status}")
    print(f"  Text: {text[:60]}...")
    print(f"  Constructs: {constructs if constructs else 'none'}")
    print(f"  Welfare score: {welfare_score:.3f}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Decision: {'PASS' if welfare_score >= threshold else 'FAIL'}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nProcessed first 5 documents (max_examples=5):")
included = sum(1 for text in test_docs[:5] if should_include_example(text, phi_metrics, threshold))
filtered = 5 - included
print(f"  ✓ Included: {included}")
print(f"  ✗ Filtered: {filtered}")
print(f"\nRemaining 2 documents not processed (beyond max_examples limit)")
