#!/usr/bin/env python3
"""
Flatten sequence data to final-week selected features.

Input:
  - sequences.npz with X shaped (N, T, F)

Output:
  - flattened.npz with X shaped (N, K), where K is the number of selected features
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


SELECTED_FEATURES = ["FilledSST", "TSA", "TSA_DHW", "TSA_Frequency"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten sequences.npz to final-week selected features."
    )
    parser.add_argument(
        "--input",
        default="datasets/sequences.npz",
        help="Path to input .npz file (default: datasets/sequences.npz).",
    )
    parser.add_argument(
        "--output",
        default="flattened.npz",
        help="Path to output .npz file (default: flattened.npz).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = np.load(input_path, allow_pickle=True)
    if "X" not in data.files:
        raise KeyError("Input .npz is missing required array: 'X'")

    X = data["X"]
    if X.ndim != 3:
        raise ValueError(f"Expected X with 3 dims (N, T, F), got shape {X.shape}")

    if "feature_names" not in data.files:
        raise KeyError("Input .npz is missing required array: 'feature_names'")

    feature_names = [str(name) for name in data["feature_names"]]
    feature_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    missing = [name for name in SELECTED_FEATURES if name not in feature_to_idx]
    if missing:
        raise KeyError(f"Missing selected features in input: {missing}")

    selected_idx = [feature_to_idx[name] for name in SELECTED_FEATURES]

    # Keep only the final week (t = -1), then keep selected feature columns.
    X_flat = X[:, -1, :][:, selected_idx].astype(np.float32, copy=False)

    save_payload = {
        "X": X_flat,
        "feature_names": np.array(SELECTED_FEATURES, dtype=object),
    }
    if "y" in data.files:
        save_payload["y"] = data["y"]
    if "meta" in data.files:
        save_payload["meta"] = data["meta"]
    if "bleach_bins" in data.files:
        save_payload["bleach_bins"] = data["bleach_bins"]

    np.savez_compressed(output_path, **save_payload)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Original X shape: {X.shape}")
    print(f"Flattened X shape: {X_flat.shape}")
    print(f"Selected features: {SELECTED_FEATURES}")


if __name__ == "__main__":
    main()
