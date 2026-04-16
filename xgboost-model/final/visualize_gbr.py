"""
Create GBR prediction maps:
1. Point-level centroid map
2. Grid-level heatmap (nearest-centroid interpolation)
3. Risk score continuous map
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from xgboost import XGBClassifier

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("cartopy not installed -- maps will be without coastlines")

from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_PATH = ROOT / "coral reef analysis" / "datasets" / "crw_gbr_sequences_reduced_16_latest_week_centroids.npz"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

CLASS_LABELS = ["None (0%)", "Moderate (1-50%)", "Severe (>50%)"]
CLASS_COLORS = {
    "None (0%)": "#4393c3",
    "Moderate (1-50%)": "#f4a582",
    "Severe (>50%)": "#d6604d",
}

# GBR bounding box
EXTENT = [141.5, 154.5, -25.5, -9.5]


def make_ax(fig, extent):
    if HAS_CARTOPY:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="black", linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.gridlines(draw_labels=True, alpha=0.3)
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3)
    return ax


def scatter_kwargs(lon, lat):
    if HAS_CARTOPY:
        return {"transform": ccrs.PlateCarree()}
    return {}


def plot_centroid_map(results, save_path):
    """Map 1: Point-level centroid predictions."""
    fig = plt.figure(figsize=(14, 10))
    ax = make_ax(fig, EXTENT)

    for label in CLASS_LABELS:
        mask = results["predicted_label"] == label
        subset = results[mask]
        ax.scatter(
            subset["centroid_lon"], subset["centroid_lat"],
            c=CLASS_COLORS[label], s=40, alpha=0.8,
            label=f"{label} (n={mask.sum()})",
            edgecolors="black", linewidth=0.3, zorder=5,
            **scatter_kwargs(subset["centroid_lon"], subset["centroid_lat"]),
        )

    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_title("XGBoost Predicted Bleaching Severity -- GBR Reef Clusters (April 2026)",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def build_grid_predictions(model, inf_data, feature_names):
    """Predict on a dense grid using nearest-centroid interpolation."""
    lat_range = np.arange(-25.0, -9.5, 0.05)
    lon_range = np.arange(142.0, 154.0, 0.05)

    centroid_coords = np.column_stack([inf_data["centroid_lat"], inf_data["centroid_lon"]])
    tree = cKDTree(centroid_coords)

    X_seq_centroids = inf_data["X"]
    lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
    grid_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    distances, indices = tree.query(grid_points)
    MAX_DISTANCE_DEG = 1.5
    valid_mask = distances < MAX_DISTANCE_DEG

    print(f"Grid: {len(lat_range)} x {len(lon_range)} = {len(grid_points)} points")
    print(f"Valid (within {MAX_DISTANCE_DEG} deg of centroid): {valid_mask.sum()}")

    valid_indices = indices[valid_mask]
    valid_points = grid_points[valid_mask]

    X_seq_grid = X_seq_centroids[valid_indices]
    meta_grid = np.column_stack([
        valid_points[:, 0], valid_points[:, 1],
        np.full(len(valid_points), 2026.0),
        np.full(len(valid_points), 4.0),
    ])

    flat = {}
    for w in range(16):
        for i, fname in enumerate(feature_names):
            flat[f"{fname}_week{w:02d}"] = X_seq_grid[:, w, i]
    flat["latitude"] = meta_grid[:, 0]
    flat["longitude"] = meta_grid[:, 1]
    flat["year"] = meta_grid[:, 2]
    flat["month"] = meta_grid[:, 3]
    X_grid_flat = pd.DataFrame(flat).fillna(0)

    y_pred = model.predict(X_grid_flat)
    y_proba = model.predict_proba(X_grid_flat)

    print(f"Grid predictions: None={np.sum(y_pred == 0)}, "
          f"Moderate={np.sum(y_pred == 1)}, Severe={np.sum(y_pred == 2)}")

    return lat_grid, lon_grid, grid_points, valid_mask, y_pred, y_proba


def plot_grid_heatmap(lat_grid, lon_grid, grid_points, valid_mask, y_pred, save_path):
    """Map 2: Grid-level discrete heatmap."""
    pred_grid = np.full(len(grid_points), np.nan)
    pred_grid[valid_mask] = y_pred
    pred_grid_2d = pred_grid.reshape(lat_grid.shape)

    cmap = mcolors.ListedColormap(["#4393c3", "#f4a582", "#d6604d"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(14, 10))
    ax = make_ax(fig, EXTENT)

    kwargs = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    ax.pcolormesh(lon_grid, lat_grid, pred_grid_2d,
                  cmap=cmap, norm=norm, alpha=0.7, **kwargs)

    legend_elements = [
        Patch(facecolor="#4393c3", label=f"None (0%) (n={np.sum(y_pred == 0)})"),
        Patch(facecolor="#f4a582", label=f"Moderate (1-50%) (n={np.sum(y_pred == 1)})"),
        Patch(facecolor="#d6604d", label=f"Severe (>50%) (n={np.sum(y_pred == 2)})"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_title("Predicted Bleaching Severity -- GBR Grid (April 2026)",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_risk_score_map(lat_grid, lon_grid, grid_points, valid_mask, y_proba, save_path):
    """Map 3: Continuous risk score heatmap."""
    risk_scores = 1.0 - y_proba[:, 0]

    risk_grid = np.full(len(grid_points), np.nan)
    risk_grid[valid_mask] = risk_scores
    risk_grid_2d = risk_grid.reshape(lat_grid.shape)

    fig = plt.figure(figsize=(14, 10))
    ax = make_ax(fig, EXTENT)

    kwargs = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    im = ax.pcolormesh(lon_grid, lat_grid, risk_grid_2d,
                       cmap="RdYlBu_r", vmin=0, vmax=1, alpha=0.8, **kwargs)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="Bleaching Risk Score")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0% risk", "25%", "50%", "75%", "100% risk"])

    ax.set_title("Bleaching Risk Score -- GBR Grid (April 2026)\n"
                 "Risk = P(Moderate) + P(Severe)",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    results = pd.read_csv(RESULTS_DIR / "gbr_predictions.csv")

    model = XGBClassifier()
    model.load_model(str(RESULTS_DIR / "model_3class.json"))

    with open(RESULTS_DIR / "model_config.json") as f:
        config = json.load(f)

    inf_data = np.load(INFERENCE_PATH, allow_pickle=True)
    feature_names = list(inf_data["feature_names"])

    print("Map 1: Centroid predictions...")
    plot_centroid_map(results, RESULTS_DIR / "gbr_centroid_map.png")

    print("\nMap 2: Grid heatmap...")
    lat_grid, lon_grid, grid_points, valid_mask, y_pred, y_proba = \
        build_grid_predictions(model, inf_data, feature_names)
    plot_grid_heatmap(lat_grid, lon_grid, grid_points, valid_mask, y_pred,
                      RESULTS_DIR / "gbr_grid_heatmap.png")

    print("\nMap 3: Risk score map...")
    plot_risk_score_map(lat_grid, lon_grid, grid_points, valid_mask, y_proba,
                        RESULTS_DIR / "gbr_risk_score_map.png")

    print("\nAll maps saved to results/")


if __name__ == "__main__":
    main()
