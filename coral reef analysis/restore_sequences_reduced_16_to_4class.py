#!/usr/bin/env python3
"""
Restore 4-class bleaching labels in sequences_reduced_16.npz.

This script keeps reduced features/X from the target NPZ, but replaces:
  - y           with labels from source sequences.npz (classes 0,1,2,3)
  - bleach_bins with bins from source sequences.npz ([-1, 1, 10, 50, 100])

Safety checks:
  - same sample count between source and target
  - exact meta match to ensure labels are copied to the correct rows
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore sequences_reduced_16.npz to use the same 4 bleaching levels "
            "as sequences.npz."
        )
    )
    parser.add_argument(
        "--source",
        default="datasets/sequences.npz",
        help="Source NPZ with original 4-class labels (default: datasets/sequences.npz).",
    )
    parser.add_argument(
        "--target",
        default="datasets/sequences_reduced_16.npz",
        help="Target reduced NPZ to update (default: datasets/sequences_reduced_16.npz).",
    )
    parser.add_argument(
        "--output",
        default="datasets/sequences_reduced_16.npz",
        help="Output NPZ path (default: overwrite target).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a .bak file when overwriting target in place.",
    )
    return parser.parse_args()


def require_keys(data: np.lib.npyio.NpzFile, name: str, keys: list[str]) -> None:
    missing = [k for k in keys if k not in data.files]
    if missing:
        raise KeyError(f"{name} is missing required keys: {missing}")


def main() -> None:
    args = parse_args()

    source_path = Path(args.source).resolve()
    target_path = Path(args.target).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target not found: {target_path}")

    with np.load(source_path, allow_pickle=True) as source, np.load(
        target_path, allow_pickle=True
    ) as target:
        require_keys(source, "source", ["y", "meta", "bleach_bins"])
        require_keys(target, "target", ["X", "y", "meta", "feature_names"])

        # Materialize arrays in memory so in-place writes do not read from a file
        # that is being overwritten.
        y_source = np.array(source["y"], copy=True)
        meta_source = np.array(source["meta"], copy=True)
        meta_target = np.array(target["meta"], copy=True)
        bins_source = np.array(source["bleach_bins"], copy=True)
        y_before = np.array(target["y"], copy=True)
        payload = {k: np.array(target[k], copy=True) for k in target.files}

    if meta_source.shape[0] != meta_target.shape[0]:
        raise ValueError(
            f"Row count mismatch: source={meta_source.shape[0]}, target={meta_target.shape[0]}"
        )
    if not np.array_equal(meta_source, meta_target):
        raise ValueError(
            "Meta arrays are not identical; refusing to transfer labels to avoid misalignment."
        )

    src_classes = sorted(np.unique(y_source).tolist())
    if not set(src_classes).issubset({0, 1, 2, 3}):
        raise ValueError(f"Unexpected source classes: {src_classes}")

    payload["y"] = y_source.astype(np.int32, copy=False)
    payload["bleach_bins"] = bins_source

    if output_path == target_path and not args.no_backup:
        backup_path = target_path.with_suffix(target_path.suffix + ".bak")
        shutil.copy2(target_path, backup_path)
        print(f"Backup created: {backup_path}")

    np.savez_compressed(output_path, **payload)

    y_after = payload["y"]
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"Output: {output_path}")
    print(f"Classes before: {sorted(np.unique(y_before).tolist())}")
    print(f"Classes after:  {sorted(np.unique(y_after).tolist())}")
    print(f"Bleach bins after: {payload['bleach_bins'].tolist()}")
    print("Done.")


if __name__ == "__main__":
    main()
