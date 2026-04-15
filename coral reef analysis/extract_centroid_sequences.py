#!/usr/bin/env python3
"""
Extract one latest-week sequence per GBR cluster centroid.

Input:
  - Latest-week sequence NPZ (e.g., crw_gbr_sequences_reduced_16_latest_week.npz)
  - Cluster summary CSV with centroid coordinates

Output:
  - NPZ where first dimension equals number of clusters (one row per centroid)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match cluster centroids to nearest latest-week sequences."
    )
    parser.add_argument(
        "--input-npz",
        default="datasets/crw_gbr_sequences_reduced_16_latest_week.npz",
        help="Latest-week sequence NPZ.",
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
    required_npz_keys = {"X", "meta"}
    missing_npz = sorted(required_npz_keys.difference(data.files))
    if missing_npz:
        raise KeyError(f"Input NPZ missing required keys: {missing_npz}")

    X = data["X"]
    meta = data["meta"]
    n_rows = X.shape[0]
    if meta.ndim != 2 or meta.shape[0] != n_rows or meta.shape[1] < 2:
        raise ValueError(f"Expected meta shape (N, >=2), got {meta.shape}")

    site_lats = meta[:, 0].astype(np.float64)
    site_lons = meta[:, 1].astype(np.float64)

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

    chosen_idx = np.empty(K, dtype=np.int64)
    chosen_dist_km = np.empty(K, dtype=np.float64)
    for i in range(K):
        d = haversine_km_single_to_many(
            centroid_lats[i], centroid_lons[i], site_lats, site_lons
        )
        j = int(np.argmin(d))
        chosen_idx[i] = j
        chosen_dist_km[i] = float(d[j])

    # One output row per cluster centroid (duplicates allowed if same nearest site).
    payload: dict[str, np.ndarray] = {}
    for key in data.files:
        arr = data[key]
        if getattr(arr, "ndim", 0) >= 1 and len(arr) == n_rows:
            payload[key] = arr[chosen_idx]
        else:
            payload[key] = arr

    # Add centroid matching metadata
    payload["cluster_id"] = np.asarray(cluster_ids)
    payload["centroid_lat"] = centroid_lats.astype(np.float32)
    payload["centroid_lon"] = centroid_lons.astype(np.float32)
    payload["matched_lat"] = payload["meta"][:, 0].astype(np.float32)
    payload["matched_lon"] = payload["meta"][:, 1].astype(np.float32)
    payload["centroid_match_distance_km"] = chosen_dist_km.astype(np.float32)
    if args.cluster_size_col in clusters.columns:
        payload["cluster_n_records"] = clusters[args.cluster_size_col].to_numpy()

    np.savez_compressed(output_npz, **payload)

    unique_matches = len(np.unique(chosen_idx))
    print("=" * 70)
    print("Centroid sequence extraction complete")
    print("=" * 70)
    print(f"Input NPZ rows: {n_rows:,}")
    print(f"Clusters: {K:,}")
    print(f"Output X shape: {payload['X'].shape}")
    print(f"Unique matched sequence rows: {unique_matches:,} / {K:,}")
    print(
        "Match distance (km): "
        f"min={chosen_dist_km.min():.4f}, "
        f"median={np.median(chosen_dist_km):.4f}, "
        f"max={chosen_dist_km.max():.4f}"
    )
    print(f"Saved: {output_npz}")


if __name__ == "__main__":
    main()
