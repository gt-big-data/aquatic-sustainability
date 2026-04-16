#!/usr/bin/env python3
"""
Filter an NPZ sequence dataset to rows ending on the most recent available week.

Expected NPZ structure (same as crw_gbr_sequences_reduced_16.npz):
  - X: (N, lookback, features)
  - y: (N,)
  - meta: (N, 4) -> [lat, lon, year, month]
  - feature_names, bleach_bins

Selection logic:
  1) If an exact end-time key exists (e.g., end_time/timestamps), keep rows
     matching max timestamp.
  2) Otherwise, fall back to meta year/month and keep rows from the latest
     (year, month). This is month-level, not exact week-level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


TIMESTAMP_KEY_CANDIDATES = ["end_time", "end_week", "timestamps", "time"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter NPZ sequences to the most recent end-week rows."
    )
    parser.add_argument(
        "--input",
        default="datasets/crw_gbr_sequences_reduced_16.npz",
        help="Input NPZ path.",
    )
    parser.add_argument(
        "--output",
        default="datasets/crw_gbr_sequences_reduced_16_latest_week.npz",
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report counts; do not write output file.",
    )
    return parser.parse_args()


def parse_timestamp_array(arr: np.ndarray) -> np.ndarray | None:
    """Return datetime64[ns] array if parseable; otherwise None."""
    if arr.ndim != 1:
        return None

    # Already datetime-like
    if np.issubdtype(arr.dtype, np.datetime64) or arr.dtype.kind in ("M", "m"):
        return arr.astype("datetime64[ns]")

    # Try parsing string/object arrays to datetime
    if arr.dtype.kind in ("U", "S", "O"):
        try:
            parsed = np.asarray(arr).astype("datetime64[ns]")
        except (TypeError, ValueError):
            return None
        if parsed.shape != arr.shape:
            return None
        if np.all(np.isnat(parsed)):
            return None
        return parsed

    return None


def find_timestamp_key(
    data: np.lib.npyio.NpzFile, n_rows: int
) -> tuple[str | None, np.ndarray | None]:
    for key in TIMESTAMP_KEY_CANDIDATES:
        if key not in data.files:
            continue
        arr = data[key]
        if arr.ndim != 1 or len(arr) != n_rows:
            continue
        parsed = parse_timestamp_array(arr)
        if parsed is not None:
            return key, parsed
    return None, None


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {input_path}")

    data = np.load(input_path, allow_pickle=True)
    required = {"X", "y", "meta"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(f"Input NPZ is missing required keys: {sorted(missing)}")

    X = data["X"]
    y = data["y"]
    meta = data["meta"]
    n_rows = X.shape[0]
    if meta.ndim != 2 or meta.shape[0] != n_rows:
        raise ValueError(f"Invalid meta shape {meta.shape} for X rows {n_rows}")

    ts_key, ts = find_timestamp_key(data, n_rows)
    if ts_key is not None and ts is not None:
        valid = ~np.isnat(ts)
        if not np.any(valid):
            raise ValueError(
                f"Timestamp key '{ts_key}' found but all values are NaT; cannot select latest week."
            )
        latest = ts[valid].max()
        keep = ts == latest
        mode = f"exact timestamp key '{ts_key}'"
        latest_desc = str(latest)
    else:
        # Fallback: use year/month from meta.
        if meta.shape[1] < 4:
            raise ValueError(
                "No timestamp key found and meta has fewer than 4 cols; "
                "cannot infer end period."
            )
        years = meta[:, 2].astype(np.int32)
        months = meta[:, 3].astype(np.int32)
        ym = years * 100 + months
        latest_ym = int(ym.max())
        keep = ym == latest_ym
        mode = "meta year/month fallback"
        latest_desc = f"{latest_ym // 100:04d}-{latest_ym % 100:02d}"

    keep_count = int(np.sum(keep))
    print("=" * 70)
    print("Latest-end filtering")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Rows: {n_rows:,}")
    print(f"Selection mode: {mode}")
    print(f"Latest period: {latest_desc}")
    print(f"Kept rows: {keep_count:,} ({keep_count / n_rows:.2%})")

    if args.dry_run:
        print("Dry-run enabled; no file written.")
        return

    payload = {}
    for key in data.files:
        arr = data[key]
        if getattr(arr, "ndim", 0) >= 1 and len(arr) == n_rows:
            payload[key] = arr[keep]
        else:
            payload[key] = arr

    np.savez_compressed(output_path, **payload)
    print(f"Output: {output_path}")
    print(f"Output X shape: {payload['X'].shape}")


if __name__ == "__main__":
    main()
