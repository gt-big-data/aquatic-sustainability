#!/usr/bin/env python3
"""
Lightweight visualization for GBR clustering results.

Plots:
  - All GBR record locations in a light color
  - Cluster centroids (size/color by cluster record count)
  - Coastline outline derived from local SST NaN mask contour
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GBR record points and cluster centroids.")
    parser.add_argument(
        "--records-csv",
        default="outputs/gbr_reef_eda/gbr_records_with_clusters.csv",
        help="CSV with record-level lat/lon and cluster labels.",
    )
    parser.add_argument(
        "--summary-csv",
        default="outputs/gbr_reef_eda/gbr_cluster_summary.csv",
        help="CSV with cluster centroids and sizes.",
    )
    parser.add_argument(
        "--sst-file",
        default="",
        help=(
            "Optional SST NetCDF file for coastline extraction. "
            "If omitted, script tries first file in crw_gbr_nc_yearly/sst/*.nc."
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/gbr_reef_eda/gbr_clusters_map.png",
        help="Output figure path.",
    )
    parser.add_argument("--lat-min", type=float, default=-24.5)
    parser.add_argument("--lat-max", type=float, default=-10.0)
    parser.add_argument("--lon-min", type=float, default=142.0)
    parser.add_argument("--lon-max", type=float, default=154.0)
    parser.add_argument(
        "--annotate-top-n",
        type=int,
        default=15,
        help="Annotate top N largest clusters by n_records.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    return parser.parse_args()


def resolve_sst_file(candidate: str) -> Path | None:
    if candidate:
        p = Path(candidate).resolve()
        return p if p.exists() else None
    default_dir = Path("crw_gbr_nc_yearly/sst")
    files = sorted(default_dir.glob("*.nc"))
    return files[0].resolve() if files else None


def draw_coastline_from_sst_mask(ax: plt.Axes, sst_file: Path) -> None:
    try:
        import xarray as xr
    except Exception as exc:  # pragma: no cover
        print(f"[warn] xarray unavailable, skipping coastline: {exc}")
        return

    try:
        ds = xr.open_dataset(sst_file)
        if "analysed_sst" not in ds:
            print(f"[warn] 'analysed_sst' not found in {sst_file}, skipping coastline")
            return
        da = ds["analysed_sst"].isel(time=0)
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        land = np.isnan(da.values).astype(float)
        # Contour at 0.5 approximates land/ocean boundary.
        ax.contour(
            lons,
            lats,
            land,
            levels=[0.5],
            colors="black",
            linewidths=0.8,
            alpha=0.9,
            zorder=2,
        )
        print(f"[info] coastline drawn from {sst_file}")
    except Exception as exc:  # pragma: no cover
        print(f"[warn] failed to draw coastline from {sst_file}: {exc}")


def main() -> None:
    args = parse_args()
    records_csv = Path(args.records_csv).resolve()
    summary_csv = Path(args.summary_csv).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records_csv.exists():
        raise FileNotFoundError(f"Records CSV not found: {records_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    records = pd.read_csv(records_csv)
    summary = pd.read_csv(summary_csv)

    for col in ["Latitude_Degrees", "Longitude_Degrees"]:
        if col not in records.columns:
            raise KeyError(f"Missing required column '{col}' in {records_csv}")
    for col in ["cluster_id", "n_records", "lat_centroid", "lon_centroid"]:
        if col not in summary.columns:
            raise KeyError(f"Missing required column '{col}' in {summary_csv}")

    # Keep map within GBR bbox
    records = records[
        (records["Latitude_Degrees"] >= args.lat_min)
        & (records["Latitude_Degrees"] <= args.lat_max)
        & (records["Longitude_Degrees"] >= args.lon_min)
        & (records["Longitude_Degrees"] <= args.lon_max)
    ].copy()
    summary = summary[
        (summary["lat_centroid"] >= args.lat_min)
        & (summary["lat_centroid"] <= args.lat_max)
        & (summary["lon_centroid"] >= args.lon_min)
        & (summary["lon_centroid"] <= args.lon_max)
    ].copy()

    if records.empty:
        raise ValueError("No records in requested bbox.")

    fig, ax = plt.subplots(figsize=(10, 8))

    # 1) Background records
    ax.scatter(
        records["Longitude_Degrees"].values,
        records["Latitude_Degrees"].values,
        s=8,
        c="#9ecae1",
        alpha=0.35,
        edgecolors="none",
        label="GBR records",
        zorder=1,
    )

    # 2) Coastline outline from local SST mask
    sst_file = resolve_sst_file(args.sst_file)
    if sst_file is not None:
        draw_coastline_from_sst_mask(ax, sst_file)
    else:
        print("[warn] no SST file found for coastline outline")

    # 3) Cluster centroids
    if not summary.empty:
        # Marker size scaling keeps large clusters visible but bounded.
        sizes = 25.0 + 6.0 * np.sqrt(summary["n_records"].values.astype(float))
        sc = ax.scatter(
            summary["lon_centroid"].values,
            summary["lat_centroid"].values,
            s=sizes,
            c=summary["n_records"].values,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.4,
            alpha=0.95,
            label="Cluster centroids",
            zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax, shrink=0.88, pad=0.02)
        cbar.set_label("Cluster size (n_records)")

        if args.annotate_top_n > 0:
            top = summary.sort_values("n_records", ascending=False).head(args.annotate_top_n)
            for _, r in top.iterrows():
                ax.text(
                    float(r["lon_centroid"]) + 0.03,
                    float(r["lat_centroid"]) + 0.03,
                    f"{int(r['cluster_id'])}",
                    fontsize=7,
                    color="black",
                    zorder=4,
                )

    ax.set_xlim(args.lon_min, args.lon_max)
    ax.set_ylim(args.lat_min, args.lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("GBR Records and Likely Same-Reef Cluster Centroids")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    print(f"[info] saved figure: {output_path}")


if __name__ == "__main__":
    main()
