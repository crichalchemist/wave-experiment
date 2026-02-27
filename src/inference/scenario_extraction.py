"""
Detective-generated scenario extraction.

Extracts welfare construct patterns from real text via the classifier,
identifies trajectory patterns (declining, rising, diverging constructs),
and generates synthetic training scenarios from those patterns.

This is Layer 3 of the Detective-Forecaster bridge: the detective's
findings enrich the forecaster's training data.
"""
from typing import Dict, List, Optional
import logging

import numpy as np
import pandas as pd

from src.inference.welfare_scoring import (
    ALL_CONSTRUCTS, compute_phi, get_construct_scores,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE_WORDS = 500


def extract_construct_profiles(
    corpus_text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
) -> List[Dict]:
    """
    Split corpus into chunks and score each for welfare constructs.

    Args:
        corpus_text: Full text corpus.
        chunk_size: Target words per chunk.

    Returns:
        List of dicts with keys: chunk_index, scores, text_preview.
    """
    words = corpus_text.split()
    profiles = []

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < chunk_size // 4:
            break  # skip tiny trailing chunks

        chunk_text = " ".join(chunk_words)
        scores = get_construct_scores(chunk_text)

        profiles.append({
            "chunk_index": len(profiles),
            "scores": scores,
            "text_preview": chunk_text[:100],
        })

    return profiles


def identify_trajectory_patterns(
    profiles: List[Dict],
    min_run_length: int = 3,
    change_threshold: float = 0.15,
) -> List[Dict]:
    """
    Find trajectory patterns in consecutive construct profiles.

    Looks for runs of min_run_length chunks where a construct changes
    by at least change_threshold total. Each pattern becomes a scenario template.

    Args:
        profiles: Output from extract_construct_profiles.
        min_run_length: Minimum consecutive chunks to form a pattern.
        change_threshold: Minimum total change to qualify.

    Returns:
        List of scenario templates with label, start_levels, end_levels,
        dominant_construct, direction.
    """
    if len(profiles) < min_run_length:
        return []

    patterns = []

    for construct in ALL_CONSTRUCTS:
        values = [p["scores"][construct] for p in profiles]

        # Sliding window: find runs where construct changes significantly
        for start in range(len(values) - min_run_length + 1):
            end = start + min_run_length
            # Extend run while trend continues
            while end < len(values):
                delta = values[end] - values[start]
                prev_delta = values[end - 1] - values[start]
                if abs(delta) < abs(prev_delta):
                    break
                end += 1

            total_change = values[min(end, len(values)) - 1] - values[start]

            if abs(total_change) >= change_threshold:
                direction = "declining" if total_change < 0 else "rising"
                start_levels = profiles[start]["scores"].copy()
                end_levels = profiles[min(end, len(profiles)) - 1]["scores"].copy()

                label = f"{direction}_{construct}"
                if not any(p["label"] == label for p in patterns):
                    patterns.append({
                        "label": label,
                        "start_levels": start_levels,
                        "end_levels": end_levels,
                        "dominant_construct": construct,
                        "direction": direction,
                        "run_length": end - start,
                    })

    return patterns


def generate_from_template(
    template: Dict,
    length: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic trajectory from a scenario template.

    Interpolates between start and end levels, adds noise calibrated
    to the observed change magnitude, and computes Phi.

    Args:
        template: Dict with start_levels, end_levels keys.
        length: Number of time steps.
        rng: Random generator for noise.

    Returns:
        DataFrame with 8 construct columns + phi column.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    data = {}
    for c in ALL_CONSTRUCTS:
        start = template["start_levels"].get(c, 0.5)
        end = template["end_levels"].get(c, 0.5)
        noise_scale = max(0.005, abs(end - start) * 0.05)
        data[c] = np.linspace(start, end, length) + rng.normal(0, noise_scale, length)
        data[c] = np.clip(data[c], 0.0, 1.0)

    df = pd.DataFrame(data)

    # Compute Phi (no derivatives — welfare_scoring.compute_phi doesn't accept them)
    phi_vals = np.array([
        compute_phi({c: df.at[i, c] for c in ALL_CONSTRUCTS})
        for i in range(len(df))
    ])
    df["phi"] = phi_vals
    return df
