#!/usr/bin/env python3
"""
Exploratory data analysis + spatial clustering for GBR reef positions.

Purpose:
  - Filter global_bleaching_environmental.csv to the Great Barrier Reef bbox
  - Quantify coordinate density / nearest-neighbor spacing
  - Cluster nearby lat/lon points likely representing the same reef

Clustering method:
  - DBSCAN with haversine distance
  - eps_km controls max within-cluster neighbor distance
  - min_samples controls density requirement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


EARTH_RADIUS_KM = 6371.0088


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA + clustering of likely same-reef GBR coordinates."
    )
    parser.add_argument(
        "--input-csv",
        default="datasets/global_bleaching_environmental.csv",
        help="Path to global_bleaching_environmental.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gbr_reef_eda",
        help="Directory to write EDA outputs",
    )
    parser.add_argument("--lat-min", type=float, default=-24.5)
    parser.add_argument("--lat-max", type=float, default=-10.0)
    parser.add_argument("--lon-min", type=float, default=142.0)
    parser.add_argument("--lon-max", type=float, default=154.0)
    parser.add_argument(
        "--eps-km",
        type=float,
        default=2.0,
        help="DBSCAN neighborhood radius in kilometers (default: 2.0)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples (default: 2)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="How many largest clusters to print/save in quick preview outputs",
    )
    return parser.parse_args()


def nearest_neighbor_quantiles(coords_rad: np.ndarray) -> dict[str, float]:
    if len(coords_rad) < 2:
        return {k: float("nan") for k in ["q50", "q75", "q90", "q95", "q99"]}
    nn = NearestNeighbors(n_neighbors=2, metric="haversine")
    nn.fit(coords_rad)
    dists, _ = nn.kneighbors(coords_rad)
    # dists[:, 0] is self-distance (0), dists[:, 1] is nearest non-self
    nn_km = dists[:, 1] * EARTH_RADIUS_KM
    qs = np.quantile(nn_km, [0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "q50": float(qs[0]),
        "q75": float(qs[1]),
        "q90": float(qs[2]),
        "q95": float(qs[3]),
        "q99": float(qs[4]),
    }


def add_cluster_sizes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cluster_counts = out["cluster_id"].value_counts(dropna=False).to_dict()
    out["cluster_size"] = out["cluster_id"].map(cluster_counts).astype(int)
    out["likely_same_reef"] = (out["cluster_id"] >= 0) & (out["cluster_size"] >= 2)
    return out


def build_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    clustered = df[df["cluster_id"] >= 0].copy()
    if clustered.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "n_records",
                "n_unique_coords",
                "lat_centroid",
                "lon_centroid",
                "lat_min",
                "lat_max",
                "lon_min",
                "lon_max",
                "year_min",
                "year_max",
            ]
        )

    grouped = clustered.groupby("cluster_id", as_index=False)
    summary = grouped.agg(
        n_records=("cluster_id", "size"),
        n_unique_coords=("coord_key", "nunique"),
        lat_centroid=("Latitude_Degrees", "mean"),
        lon_centroid=("Longitude_Degrees", "mean"),
        lat_min=("Latitude_Degrees", "min"),
        lat_max=("Latitude_Degrees", "max"),
        lon_min=("Longitude_Degrees", "min"),
        lon_max=("Longitude_Degrees", "max"),
    )

    if "Date_Year" in clustered.columns:
        yr = grouped["Date_Year"].agg(year_min="min", year_max="max")
        summary = summary.merge(yr, on="cluster_id", how="left")
    else:
        summary["year_min"] = np.nan
        summary["year_max"] = np.nan

    summary = summary.sort_values(["n_records", "n_unique_coords"], ascending=False)
    return summary


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print("=" * 70)
    print("Loading data")
    print("=" * 70)
    print(f"Input: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Rows (global): {len(df):,}")

    required = ["Latitude_Degrees", "Longitude_Degrees"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df["Latitude_Degrees"] = pd.to_numeric(df["Latitude_Degrees"], errors="coerce")
    df["Longitude_Degrees"] = pd.to_numeric(df["Longitude_Degrees"], errors="coerce")
    df = df.dropna(subset=["Latitude_Degrees", "Longitude_Degrees"]).copy()

    gbr = df[
        (df["Latitude_Degrees"] >= args.lat_min)
        & (df["Latitude_Degrees"] <= args.lat_max)
        & (df["Longitude_Degrees"] >= args.lon_min)
        & (df["Longitude_Degrees"] <= args.lon_max)
    ].copy()

    if gbr.empty:
        raise ValueError("No records inside provided GBR bounding box.")

    gbr["coord_key"] = (
        gbr["Latitude_Degrees"].round(5).astype(str)
        + ","
        + gbr["Longitude_Degrees"].round(5).astype(str)
    )

    print("\n" + "=" * 70)
    print("GBR EDA")
    print("=" * 70)
    print(f"Rows (GBR bbox): {len(gbr):,}")
    print(f"Unique exact coords: {gbr[['Latitude_Degrees','Longitude_Degrees']].drop_duplicates().shape[0]:,}")
    if "Site_ID" in gbr.columns:
        print(f"Unique Site_ID: {gbr['Site_ID'].nunique():,}")
    if "Date_Year" in gbr.columns:
        year = pd.to_numeric(gbr["Date_Year"], errors="coerce")
        print(f"Date_Year range: {int(year.min())} - {int(year.max())}")

    coords_deg = gbr[["Latitude_Degrees", "Longitude_Degrees"]].to_numpy(dtype=np.float64)
    coords_rad = np.radians(coords_deg)

    q = nearest_neighbor_quantiles(coords_rad)
    print("\nNearest-neighbor distance quantiles (km):")
    for k, v in q.items():
        print(f"  {k}: {v:.3f}")

    print("\n" + "=" * 70)
    print("DBSCAN clustering")
    print("=" * 70)
    print(f"eps_km={args.eps_km}, min_samples={args.min_samples}")

    model = DBSCAN(
        eps=args.eps_km / EARTH_RADIUS_KM,
        min_samples=args.min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    labels = model.fit_predict(coords_rad)
    gbr["cluster_id"] = labels
    gbr = add_cluster_sizes(gbr)

    n_noise = int((labels == -1).sum())
    n_clustered = int((labels >= 0).sum())
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    print(f"Clusters found: {n_clusters}")
    print(f"Clustered records: {n_clustered:,}")
    print(f"Noise records: {n_noise:,}")

    cluster_summary = build_cluster_summary(gbr)
    preview = cluster_summary.head(args.top_n).copy()
    print(f"\nTop {min(args.top_n, len(cluster_summary))} clusters by size:")
    if not preview.empty:
        for _, r in preview.iterrows():
            print(
                f"  cluster={int(r.cluster_id):4d} "
                f"n={int(r.n_records):5d} "
                f"unique_coords={int(r.n_unique_coords):4d} "
                f"centroid=({r.lat_centroid:.4f}, {r.lon_centroid:.4f})"
            )
    else:
        print("  (No non-noise clusters)")

    # Save outputs
    clustered_csv = output_dir / "gbr_records_with_clusters.csv"
    summary_csv = output_dir / "gbr_cluster_summary.csv"
    preview_csv = output_dir / "gbr_cluster_summary_top.csv"
    nn_json = output_dir / "gbr_nearest_neighbor_quantiles_km.json"
    meta_json = output_dir / "gbr_eda_metadata.json"

    gbr.to_csv(clustered_csv, index=False)
    cluster_summary.to_csv(summary_csv, index=False)
    preview.to_csv(preview_csv, index=False)
    nn_json.write_text(json.dumps(q, indent=2))

    meta = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "bbox": {
            "lat_min": args.lat_min,
            "lat_max": args.lat_max,
            "lon_min": args.lon_min,
            "lon_max": args.lon_max,
        },
        "dbscan": {"eps_km": args.eps_km, "min_samples": args.min_samples},
        "counts": {
            "rows_global": int(len(df)),
            "rows_gbr": int(len(gbr)),
            "clusters": n_clusters,
            "clustered_records": n_clustered,
            "noise_records": n_noise,
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2))

    print("\nSaved:")
    print(f"  {clustered_csv}")
    print(f"  {summary_csv}")
    print(f"  {preview_csv}")
    print(f"  {nn_json}")
    print(f"  {meta_json}")


if __name__ == "__main__":
    main()
