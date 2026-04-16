"""
Generate 52-week temporal evolution of bleaching predictions across GBR.

Uses crw_gbr_sequences_reduced_16_centroids_all_weeks.npz which contains
52 weekly snapshots × 133 reef clusters, each with a full 16-week lookback.

Outputs:
  - weekly_evolution_grid.png   — Multi-panel GBR maps (sampled weeks)
  - weekly_stacked_area.png     — Stacked area chart of severity distribution
  - weekly_risk_heatmap.png     — Heatmap: clusters × weeks, colored by risk
  - weekly_evolution.gif        — Animated GIF cycling through all 52 weeks
  - weekly_summary.csv          — Per-week counts, percentages, risk scores
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from xgboost import XGBClassifier

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("cartopy not installed -- maps will be without coastlines")

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("imageio not installed -- skipping GIF")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_WEEKS_PATH = RESULTS_DIR / "crw_gbr_sequences_reduced_16_centroids_all_weeks.npz"

# Fallback: check experiments/results if not in final/results
if not ALL_WEEKS_PATH.exists():
    ALT_PATH = Path(__file__).resolve().parents[1] / "experiments" / "results" / "crw_gbr_sequences_reduced_16_centroids_all_weeks.npz"
    if ALT_PATH.exists():
        ALL_WEEKS_PATH = ALT_PATH

CLASS_LABELS = ["None (0%)", "Moderate (1-50%)", "Severe (>50%)"]
CLASS_COLORS = {0: "#4393c3", 1: "#f4a582", 2: "#d6604d"}
EXTENT = [141.5, 154.5, -25.5, -9.5]


def load_data():
    """Load model, config, and 52-week inference data."""
    model = XGBClassifier()
    model.load_model(str(RESULTS_DIR / "model_3class.json"))

    with open(RESULTS_DIR / "model_config.json") as f:
        config = json.load(f)

    inf_data = np.load(ALL_WEEKS_PATH, allow_pickle=True)
    return model, config, inf_data


def run_52week_inference(model, config, inf_data):
    """Run inference for each of the 52 weekly snapshots."""
    X_all = inf_data["X"]           # (6916, 16, 4)
    meta_all = inf_data["meta"]     # (6916, 4)
    feature_names = list(inf_data["feature_names"])
    FEATURE_COLS = config["feature_columns"]

    end_times = inf_data["end_time"]
    unique_dates = np.unique(end_times)
    n_weeks = len(unique_dates)
    cluster_ids = np.unique(inf_data["cluster_id"])
    n_clusters = len(cluster_ids)

    print(f"Data: {n_weeks} weeks x {n_clusters} clusters = {len(X_all)} rows")
    print(f"Date range: {unique_dates[0]} -> {unique_dates[-1]}")

    # Get centroid coords from the first week's entries
    first_week_mask = end_times == unique_dates[0]
    centroid_lats = inf_data["centroid_lat"][first_week_mask]
    centroid_lons = inf_data["centroid_lon"][first_week_mask]

    week_dates = [pd.Timestamp(d) for d in unique_dates]
    weekly_predictions = []
    weekly_probabilities = []

    for wi, date in enumerate(unique_dates):
        mask = end_times == date
        X_seq = X_all[mask]       # (133, 16, 4)
        meta = meta_all[mask]     # (133, 4)

        # Flatten raw weekly
        flat = {}
        for w in range(16):
            for i, fname in enumerate(feature_names):
                flat[f"{fname}_week{w:02d}"] = X_seq[:, w, i]
        flat["latitude"] = meta[:, 0]
        flat["longitude"] = meta[:, 1]
        flat["year"] = meta[:, 2]
        flat["month"] = meta[:, 3]
        X_flat = pd.DataFrame(flat).fillna(0)

        assert list(X_flat.columns) == FEATURE_COLS, f"Feature mismatch at {date}"

        y_pred = model.predict(X_flat)
        y_proba = model.predict_proba(X_flat)

        weekly_predictions.append(y_pred)
        weekly_probabilities.append(y_proba)

        counts = np.bincount(y_pred.astype(int), minlength=3)
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        if wi % 4 == 0 or wi == n_weeks - 1:
            print(f"  Week {wi+1:2d} ({date_str}): None={counts[0]:3d}, "
                  f"Moderate={counts[1]:3d}, Severe={counts[2]:3d}")

    return weekly_predictions, weekly_probabilities, week_dates, centroid_lats, centroid_lons


def plot_evolution_grid(weekly_predictions, week_dates, centroid_lats, centroid_lons, save_path):
    """Multi-panel GBR maps — sample 16 evenly spaced weeks from the 52."""
    n_weeks = len(week_dates)
    # Pick 16 evenly spaced indices
    sample_indices = np.linspace(0, n_weeks - 1, 16, dtype=int)

    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    axes_flat = axes.flatten()

    for panel_idx, week_idx in enumerate(sample_indices):
        ax = axes_flat[panel_idx]
        y_pred = weekly_predictions[week_idx]
        date_str = week_dates[week_idx].strftime("%b %d '%y")

        for cls in [0, 1, 2]:
            mask = y_pred == cls
            if mask.sum() > 0:
                ax.scatter(
                    centroid_lons[mask], centroid_lats[mask],
                    c=CLASS_COLORS[cls], s=15, alpha=0.8, edgecolors="none",
                )

        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_title(f"{date_str}", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        counts = np.bincount(y_pred.astype(int), minlength=3)
        count_text = f"N:{counts[0]} M:{counts[1]} S:{counts[2]}"
        ax.text(0.02, 0.02, count_text, transform=ax.transAxes, fontsize=7,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    legend_elements = [
        Patch(facecolor="#4393c3", label="None (0%)"),
        Patch(facecolor="#f4a582", label="Moderate (1-50%)"),
        Patch(facecolor="#d6604d", label="Severe (>50%)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Bleaching Prediction Evolution Over 52 Weeks\n"
        f'{week_dates[0].strftime("%b %d, %Y")} \u2192 {week_dates[-1].strftime("%b %d, %Y")}',
        fontsize=16, fontweight="bold", y=1.01,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_stacked_area(weekly_predictions, week_dates, n_clusters, save_path):
    """Stacked area chart of severity distribution over 52 weeks."""
    n_weeks = len(week_dates)
    fig, ax = plt.subplots(figsize=(16, 6))

    none_counts, mod_counts, sev_counts = [], [], []
    for y_pred in weekly_predictions:
        counts = np.bincount(y_pred.astype(int), minlength=3)
        none_counts.append(counts[0])
        mod_counts.append(counts[1])
        sev_counts.append(counts[2])

    weeks = np.arange(n_weeks)
    none_pct = np.array(none_counts) / n_clusters * 100
    mod_pct = np.array(mod_counts) / n_clusters * 100
    sev_pct = np.array(sev_counts) / n_clusters * 100

    ax.stackplot(weeks, none_pct, mod_pct, sev_pct,
                 labels=CLASS_LABELS,
                 colors=["#4393c3", "#f4a582", "#d6604d"], alpha=0.85)

    # Show monthly tick labels
    tick_positions = []
    tick_labels = []
    for i, d in enumerate(week_dates):
        if d.day <= 7 or i == 0 or i == n_weeks - 1:
            tick_positions.append(i)
            tick_labels.append(d.strftime("%b '%y"))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

    ax.set_ylabel(f"Percentage of {n_clusters} Reef Clusters", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("Bleaching Severity Distribution Across GBR (52 Weeks)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, n_weeks - 1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_risk_heatmap(weekly_probabilities, week_dates, centroid_lats, save_path):
    """Heatmap: 133 clusters (y, sorted N->S) x 52 weeks (x), colored by risk score."""
    n_clusters = len(centroid_lats)
    n_weeks = len(week_dates)

    risk_matrix = np.zeros((n_clusters, n_weeks))
    for w in range(n_weeks):
        risk_matrix[:, w] = 1.0 - weekly_probabilities[w][:, 0]

    lat_order = np.argsort(centroid_lats)[::-1]  # north first
    risk_matrix_sorted = risk_matrix[lat_order]
    lats_sorted = centroid_lats[lat_order]

    fig, ax = plt.subplots(figsize=(20, 8))
    im = ax.imshow(risk_matrix_sorted, aspect="auto", cmap="RdYlBu_r",
                   vmin=0, vmax=1, interpolation="nearest")

    # X-axis: monthly labels
    tick_positions = []
    tick_labels = []
    for i, d in enumerate(week_dates):
        if d.day <= 7 or i == 0 or i == n_weeks - 1:
            tick_positions.append(i)
            tick_labels.append(d.strftime("%b '%y"))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

    # Y-axis: latitude labels
    ytick_positions = np.arange(0, n_clusters, 15)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels([f"{lats_sorted[i]:.1f}\u00b0" for i in ytick_positions], fontsize=8)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Reef Cluster (by Latitude, N\u2192S)", fontsize=12)
    ax.set_title("Bleaching Risk Score per Cluster Over 52 Weeks\n"
                 "(Red = high risk, Blue = low risk)",
                 fontsize=13, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Risk Score (P(Moderate) + P(Severe))")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def make_gif(weekly_predictions, week_dates, centroid_lats, centroid_lons, save_path):
    """Animated GIF cycling through all 52 weeks."""
    if not HAS_IMAGEIO:
        print("Skipping GIF (imageio not installed)")
        return

    n_weeks = len(week_dates)
    frames = []

    for week_idx in range(n_weeks):
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pred = weekly_predictions[week_idx]
        date_str = week_dates[week_idx].strftime("%B %d, %Y")

        for cls, label, color in [(0, "None", "#4393c3"), (1, "Moderate", "#f4a582"), (2, "Severe", "#d6604d")]:
            mask = y_pred == cls
            if mask.sum() > 0:
                ax.scatter(centroid_lons[mask], centroid_lats[mask],
                           c=color, s=50, alpha=0.8, label=f"{label} ({mask.sum()})",
                           edgecolors="black", linewidth=0.3)

        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{date_str}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.3)

        frame_path = str(RESULTS_DIR / f"frame_{week_idx:02d}.png")
        fig.tight_layout()
        fig.savefig(frame_path, dpi=100)
        plt.close()
        frames.append(imageio.imread(frame_path))

    # Hold last frame longer
    frames_extended = frames + [frames[-1]] * 5
    imageio.mimsave(str(save_path), frames_extended, duration=0.4)

    for w in range(n_weeks):
        os.remove(str(RESULTS_DIR / f"frame_{w:02d}.png"))

    print(f"Saved: {save_path}")


def save_summary_csv(weekly_predictions, weekly_probabilities, week_dates, n_clusters, save_path):
    """Per-week counts, percentages, mean/max risk scores."""
    rows = []
    for w in range(len(week_dates)):
        counts = np.bincount(weekly_predictions[w].astype(int), minlength=3)
        risk_scores = 1.0 - weekly_probabilities[w][:, 0]
        rows.append({
            "week": w + 1,
            "date": week_dates[w].strftime("%Y-%m-%d"),
            "n_none": int(counts[0]),
            "n_moderate": int(counts[1]),
            "n_severe": int(counts[2]),
            "pct_none": round(counts[0] / n_clusters * 100, 1),
            "pct_moderate": round(counts[1] / n_clusters * 100, 1),
            "pct_severe": round(counts[2] / n_clusters * 100, 1),
            "mean_risk_score": round(float(risk_scores.mean()), 4),
            "max_risk_score": round(float(risk_scores.max()), 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    print("\n" + "=" * 60)
    print("WEEKLY EVOLUTION SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))


def main():
    model, config, inf_data = load_data()

    print(f"Loaded 52-week inference data from {ALL_WEEKS_PATH.name}\n")

    weekly_predictions, weekly_probabilities, week_dates, centroid_lats, centroid_lons = \
        run_52week_inference(model, config, inf_data)

    n_clusters = len(centroid_lats)
    print(f"\nGenerating visualizations for {len(week_dates)} weeks, {n_clusters} clusters...\n")

    plot_evolution_grid(weekly_predictions, week_dates, centroid_lats, centroid_lons,
                        RESULTS_DIR / "weekly_evolution_grid.png")

    plot_stacked_area(weekly_predictions, week_dates, n_clusters,
                      RESULTS_DIR / "weekly_stacked_area.png")

    plot_risk_heatmap(weekly_probabilities, week_dates, centroid_lats,
                      RESULTS_DIR / "weekly_risk_heatmap.png")

    make_gif(weekly_predictions, week_dates, centroid_lats, centroid_lons,
             RESULTS_DIR / "weekly_evolution.gif")

    save_summary_csv(weekly_predictions, weekly_probabilities, week_dates, n_clusters,
                     RESULTS_DIR / "weekly_summary.csv")

    print("\nAll temporal visualizations saved to results/")


if __name__ == "__main__":
    main()
