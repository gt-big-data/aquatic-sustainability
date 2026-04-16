#!/usr/bin/env python3
"""
Rebuild reduced sequences with 52-week windows and selected features.

By default:
  - Reads 52-week source data from datasets/sequences.npz
  - Reads reduced feature spec + labels from datasets/sequences_reduced_16.npz
  - Overwrites datasets/sequences_reduced_16.npz (creates .bak backup first)

Also merges bleaching labels by combining low+moderate:
  - 4-class labels {0,1,2,3} -> 3-class labels {0,1,2}
    mapping: 0->0, 1->1, 2->1, 3->2
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild sequences_reduced_16.npz to use 52-week windows for reduced features."
    )
    parser.add_argument(
        "--source",
        default="datasets/sequences.npz",
        help="52-week source dataset with full feature set (default: datasets/sequences.npz).",
    )
    parser.add_argument(
        "--reduced",
        default="datasets/sequences_reduced_16.npz",
        help="Reduced dataset providing target feature list and labels (default: datasets/sequences_reduced_16.npz).",
    )
    parser.add_argument(
        "--output",
        default="datasets/sequences_reduced_16.npz",
        help="Output dataset path (default: overwrite reduced file).",
    )
    parser.add_argument(
        "--features",
        default="",
        help=(
            "Comma-separated feature names to keep from source (e.g. "
            "'FilledSST,TSA,TSA_DHW'). If omitted, uses reduced feature_names."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak file when overwriting --reduced in place.",
    )
    return parser.parse_args()


def require_keys(data: np.lib.npyio.NpzFile, name: str, keys: list[str]) -> None:
    missing = [k for k in keys if k not in data.files]
    if missing:
        raise KeyError(f"{name} is missing required keys: {missing}")


def merge_low_moderate(y: np.ndarray) -> np.ndarray:
    """Merge classes 1 and 2 into one class for 3-class setup."""
    y = y.astype(np.int64, copy=True)
    unique = set(np.unique(y).tolist())
    if unique.issubset({0, 1, 2}):
        return y
    if not unique.issubset({0, 1, 2, 3}):
        raise ValueError(f"Unexpected label set for merge: {sorted(unique)}")
    y[y == 2] = 1
    y[y == 3] = 2
    return y


def main() -> None:
    args = parse_args()

    source_path = Path(args.source).resolve()
    reduced_path = Path(args.reduced).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")
    if not reduced_path.exists():
        raise FileNotFoundError(f"Reduced dataset not found: {reduced_path}")

    source = np.load(source_path, allow_pickle=True)
    reduced = np.load(reduced_path, allow_pickle=True)

    require_keys(source, "source", ["X", "feature_names", "meta", "y"])
    require_keys(reduced, "reduced", ["X", "feature_names", "y", "meta"])

    X_source = source["X"]
    meta_source = source["meta"]
    source_features = [str(x) for x in source["feature_names"]]

    X_reduced = reduced["X"]
    y_reduced = reduced["y"]
    y_source = source["y"]
    meta_reduced = reduced["meta"]
    reduced_features = [str(x) for x in reduced["feature_names"]]

    if X_source.ndim != 3:
        raise ValueError(f"source X must be 3D (N, T, F), got {X_source.shape}")
    if X_reduced.ndim != 3:
        raise ValueError(f"reduced X must be 3D (N, T, F), got {X_reduced.shape}")
    if X_source.shape[0] != X_reduced.shape[0]:
        raise ValueError(
            f"Sample count mismatch: source N={X_source.shape[0]} vs reduced N={X_reduced.shape[0]}"
        )
    if not np.array_equal(meta_source, meta_reduced):
        raise ValueError("source and reduced meta arrays do not match; refusing to merge.")

    if args.features.strip():
        output_features = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        output_features = reduced_features

    source_idx = {name: i for i, name in enumerate(source_features)}
    missing_features = [f for f in output_features if f not in source_idx]
    if missing_features:
        raise KeyError(
            f"Reduced feature names not found in source feature_names: {missing_features}"
        )

    keep_idx = [source_idx[f] for f in output_features]
    X_out = X_source[:, :, keep_idx].astype(np.float32, copy=False)

    y_out = merge_low_moderate(y_source)
    if y_out.shape != y_reduced.shape:
        raise ValueError(
            f"Merged source y shape {y_out.shape} != reduced y shape {y_reduced.shape}"
        )

    payload = {
        "X": X_out,
        "y": y_out,
        "meta": meta_reduced,
        "feature_names": np.array(output_features, dtype=object),
    }
    payload["bleach_bins"] = np.array([-1, 1, 50, 100], dtype=np.float32)

    if output_path == reduced_path and not args.no_backup:
        backup_path = reduced_path.with_suffix(reduced_path.suffix + ".bak")
        shutil.copy2(reduced_path, backup_path)
        print(f"Backup created: {backup_path}")

    np.savez_compressed(output_path, **payload)

    print(f"Source: {source_path}")
    print(f"Reduced: {reduced_path}")
    print(f"Output: {output_path}")
    print(f"Source X shape: {X_source.shape}")
    print(f"Original reduced X shape: {X_reduced.shape}")
    print(f"New reduced X shape: {X_out.shape}")
    print(f"Features kept: {output_features}")
    print(f"Original source y classes: {sorted(np.unique(y_source).tolist())}")
    print(f"Output y classes: {sorted(np.unique(y_out).tolist())}")
    print("Done.")


if __name__ == "__main__":
    main()
