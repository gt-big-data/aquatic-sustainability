#!/usr/bin/env python3
"""
Extract one sequence per GBR cluster centroid for every end_time.

Input:
  - Sequence NPZ containing X, meta, and end_time
  - Cluster summary CSV with centroid coordinates

Output:
  - NPZ where first dimension equals (num_clusters * num_end_times)
    with one nearest sequence per centroid per end_time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match cluster centroids to nearest sequence rows per end_time."
    )
    parser.add_argument(
        "--input-npz",
        default="datasets/crw_gbr_sequences_reduced_16_latest_week.npz",
        help="Sequence NPZ containing X/meta/end_time.",
    )
    parser.add_argument(
        "--clusters-csv",
        default="outputs/gbr_reef_eda/gbr_cluster_summary.csv",
        help="Cluster summary CSV containing centroid coordinates.",
    )
    parser.add_argument(
        "--output-npz",
        default="datasets/crw_gbr_sequences_reduced_16_latest_week_centroids.npz",
        help="Output NPZ with one sequence per cluster centroid.",
    )
    parser.add_argument(
        "--lat-col",
        default="lat_centroid",
        help="Centroid latitude column in clusters CSV.",
    )
    parser.add_argument(
        "--lon-col",
        default="lon_centroid",
        help="Centroid longitude column in clusters CSV.",
    )
    parser.add_argument(
        "--cluster-id-col",
        default="cluster_id",
        help="Cluster ID column in clusters CSV.",
    )
    parser.add_argument(
        "--cluster-size-col",
        default="n_records",
        help="Cluster size column in clusters CSV (optional, saved if present).",
    )
    parser.add_argument(
        "--allow-zero-match",
        action="store_true",
        help=(
            "Allow nearest match from all rows, including all-zero sequences. "
            "Default behavior filters candidate rows to non-zero sequence signal."
        ),
    )
    return parser.parse_args()


def haversine_km_single_to_many(lat: float, lon: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Compute haversine distance from one (lat, lon) to many points (lats, lons) in km.
    """
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(lats)
    lon2 = np.radians(lons)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return EARTH_RADIUS_KM * c


def haversine_km_many_to_many(
    src_lats: np.ndarray, src_lons: np.ndarray, dst_lats: np.ndarray, dst_lons: np.ndarray
) -> np.ndarray:
    """
    Pairwise haversine distance matrix from source points to destination points.
    Returns shape (len(src_lats), len(dst_lats)).
    """
    lat1 = np.radians(src_lats)[:, None]
    lon1 = np.radians(src_lons)[:, None]
    lat2 = np.radians(dst_lats)[None, :]
    lon2 = np.radians(dst_lons)[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return EARTH_RADIUS_KM * c


def nonzero_sequence_mask(X: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """
    Return boolean mask (N,) where each row has any non-zero signal
    across all timesteps/features.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N, T, F); got {X.shape}")
    return np.any(np.abs(np.nan_to_num(X, nan=0.0)) > eps, axis=(1, 2))


def main() -> None:
    args = parse_args()
    input_npz = Path(args.input_npz).resolve()
    clusters_csv = Path(args.clusters_csv).resolve()
    output_npz = Path(args.output_npz).resolve()
    output_npz.parent.mkdir(parents=True, exist_ok=True)

    if not input_npz.exists():
        raise FileNotFoundError(f"Input NPZ not found: {input_npz}")
    if not clusters_csv.exists():
        raise FileNotFoundError(f"Clusters CSV not found: {clusters_csv}")

    data = np.load(input_npz, allow_pickle=True)
    required_npz_keys = {"X", "meta", "end_time"}
    missing_npz = sorted(required_npz_keys.difference(data.files))
    if missing_npz:
        raise KeyError(f"Input NPZ missing required keys: {missing_npz}")

    X = data["X"]
    meta = data["meta"]
    end_time = data["end_time"].astype("datetime64[ns]")
    n_rows = X.shape[0]
    if meta.ndim != 2 or meta.shape[0] != n_rows or meta.shape[1] < 2:
        raise ValueError(f"Expected meta shape (N, >=2), got {meta.shape}")
    if end_time.ndim != 1 or end_time.shape[0] != n_rows:
        raise ValueError(f"Expected end_time shape (N,), got {end_time.shape}")
    row_has_signal = nonzero_sequence_mask(X)

    unique_end_times = np.sort(np.unique(end_time))
    T = len(unique_end_times)
    if T == 0:
        raise ValueError("No end_time values found in input NPZ.")

    clusters = pd.read_csv(clusters_csv)
    for col in [args.lat_col, args.lon_col, args.cluster_id_col]:
        if col not in clusters.columns:
            raise KeyError(f"Column '{col}' not found in {clusters_csv}")

    clusters = clusters.dropna(subset=[args.lat_col, args.lon_col]).copy()
    clusters = clusters.sort_values(args.cluster_id_col).reset_index(drop=True)

    centroid_lats = clusters[args.lat_col].to_numpy(dtype=np.float64)
    centroid_lons = clusters[args.lon_col].to_numpy(dtype=np.float64)
    cluster_ids = clusters[args.cluster_id_col].to_numpy()
    K = len(clusters)

    if K == 0:
        raise ValueError("No valid centroid rows found in clusters CSV.")

    # One output row per (end_time, cluster centroid).
    chosen_idx = np.empty((T, K), dtype=np.int64)
    chosen_dist_km = np.empty((T, K), dtype=np.float64)
    week_used_nonzero = np.zeros((T,), dtype=bool)
    week_valid_counts = np.zeros((T,), dtype=np.int64)
    fallback_weeks = 0
    for t_i, ts in enumerate(unique_end_times):
        mask = end_time == ts
        row_idx = np.flatnonzero(mask)
        if row_idx.size == 0:
            raise RuntimeError(f"No rows found for end_time={ts}")

        if args.allow_zero_match:
            candidate_idx = row_idx
            week_used_nonzero[t_i] = False
        else:
            candidate_mask = row_has_signal[row_idx]
            candidate_idx = row_idx[candidate_mask]
            if candidate_idx.size == 0:
                # Safety fallback: if a week has no non-zero candidates, keep behavior robust.
                candidate_idx = row_idx
                week_used_nonzero[t_i] = False
                fallback_weeks += 1
            else:
                week_used_nonzero[t_i] = True
        week_valid_counts[t_i] = candidate_idx.size

        week_lats = meta[candidate_idx, 0].astype(np.float64)
        week_lons = meta[candidate_idx, 1].astype(np.float64)
        dist = haversine_km_many_to_many(centroid_lats, centroid_lons, week_lats, week_lons)
        nearest_local = np.argmin(dist, axis=1)

        chosen_idx[t_i, :] = candidate_idx[nearest_local]
        chosen_dist_km[t_i, :] = dist[np.arange(K), nearest_local]

        if (t_i + 1) % 10 == 0 or (t_i + 1) == T:
            print(f"  Matched end_times: {t_i + 1}/{T}")

    chosen_idx_flat = chosen_idx.reshape(-1)
    chosen_dist_flat = chosen_dist_km.reshape(-1)
    repeated_cluster_ids = np.tile(cluster_ids, T)
    repeated_centroid_lats = np.tile(centroid_lats.astype(np.float32), T)
    repeated_centroid_lons = np.tile(centroid_lons.astype(np.float32), T)

    target_end_times = np.repeat(unique_end_times, K)
    selected_end_times = end_time[chosen_idx_flat]
    if not np.array_equal(selected_end_times, target_end_times):
        raise RuntimeError("Selected rows are not aligned to requested end_time groups.")

    payload: dict[str, np.ndarray] = {}
    for key in data.files:
        arr = data[key]
        if getattr(arr, "ndim", 0) >= 1 and len(arr) == n_rows:
            payload[key] = arr[chosen_idx_flat]
        else:
            payload[key] = arr

    # Add centroid matching metadata
    payload["cluster_id"] = np.asarray(repeated_cluster_ids)
    payload["centroid_lat"] = repeated_centroid_lats
    payload["centroid_lon"] = repeated_centroid_lons
    payload["matched_lat"] = payload["meta"][:, 0].astype(np.float32)
    payload["matched_lon"] = payload["meta"][:, 1].astype(np.float32)
    payload["centroid_match_distance_km"] = chosen_dist_flat.astype(np.float32)
    payload["target_end_time"] = target_end_times
    payload["match_used_nonzero_filter"] = np.repeat(week_used_nonzero, K)
    if args.cluster_size_col in clusters.columns:
        payload["cluster_n_records"] = np.tile(clusters[args.cluster_size_col].to_numpy(), T)

    np.savez_compressed(output_npz, **payload)

    unique_matches = len(np.unique(chosen_idx_flat))
    print("=" * 70)
    print("Centroid sequence extraction complete")
    print("=" * 70)
    print(f"Input NPZ rows: {n_rows:,}")
    print(f"Unique end_time values: {T:,}")
    print(f"Clusters: {K:,}")
    print(f"Output X shape: {payload['X'].shape}")
    print(f"Output rows (end_time x clusters): {T:,} x {K:,} = {T * K:,}")
    print(f"Unique matched sequence rows: {unique_matches:,} / {T * K:,}")
    if args.allow_zero_match:
        print("Candidate filter: disabled (--allow-zero-match)")
    else:
        print(
            "Candidate filter: non-zero rows only "
            f"(weekly candidate count min/median/max="
            f"{int(week_valid_counts.min())}/"
            f"{int(np.median(week_valid_counts))}/"
            f"{int(week_valid_counts.max())})"
        )
        if fallback_weeks > 0:
            print(f"Fallback weeks (no non-zero candidates): {fallback_weeks}")
    print(
        "Match distance (km): "
        f"min={chosen_dist_flat.min():.4f}, "
        f"median={np.median(chosen_dist_flat):.4f}, "
        f"max={chosen_dist_flat.max():.4f}"
    )
    print(f"Saved: {output_npz}")


if __name__ == "__main__":
    main()
