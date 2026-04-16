#!/usr/bin/env python3
"""
Run LSTM inference on processed NPZ sequence data and save:
  1) location-aware tabular predictions
  2) machine-readable NPZ outputs
  3) severity map visualization (similar style to visualize_gbr_clusters.py)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import imageio.v2 as imageio

    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, lstm_output: torch.Tensor):
        scores = self.attn(lstm_output).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, weights


class BleachingLSTMFlexible(nn.Module):
    """
    Inference-time model definition supporting uni/bidirectional checkpoints.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        n_layers: int,
        n_classes: int,
        bidirectional: bool,
        n_static: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        num_dirs = 2 if bidirectional else 1

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.attention = TemporalAttention(hidden_size * num_dirs)
        head_input = hidden_size * num_dirs + hidden_size * num_dirs + n_static
        self.classifier = nn.Sequential(
            nn.Linear(head_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, n_classes),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor):
        x = self.input_proj(x_seq)
        lstm_out, _ = self.lstm(x)
        attn_context, attn_weights = self.attention(lstm_out)
        last_hidden = lstm_out[:, -1, :]
        combined = torch.cat([attn_context, last_hidden, x_static], dim=1)
        logits = self.classifier(combined)
        return logits, attn_weights


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, static: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.static = torch.from_numpy(static.astype(np.float32, copy=False))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.static[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bleaching LSTM inference + map output.")
    parser.add_argument(
        "--input-npz",
        default="datasets/crw_gbr_sequences_reduced_16_latest_week_centroids.npz",
        help="Input sequence NPZ (must contain X and meta).",
    )
    parser.add_argument(
        "--checkpoint",
        default="best_model.pt",
        help="Path to trained model checkpoint (.pt state_dict).",
    )
    parser.add_argument(
        "--stats-path",
        default="",
        help=(
            "Optional NPZ containing feat_mean/feat_std/static_mean/static_std. "
            "If omitted, auto-detects <checkpoint_dir>/normalization_stats.npz "
            "then <checkpoint_dir>/results.npz."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/inference",
        help="Directory for CSV/NPZ/PNG outputs.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--use-centroid-coords",
        action="store_true",
        help="Use centroid_lat/centroid_lon for plotting if present; otherwise use meta lat/lon.",
    )
    parser.add_argument(
        "--sst-file",
        default="",
        help="Optional SST NetCDF path for coastline extraction. If omitted, auto-detect first file.",
    )
    parser.add_argument("--lat-min", type=float, default=-24.5)
    parser.add_argument("--lat-max", type=float, default=-10.0)
    parser.add_argument("--lon-min", type=float, default=142.0)
    parser.add_argument("--lon-max", type=float, default=154.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--skip-temporal-viz",
        action="store_true",
        help="Skip temporal visualization outputs for multi-week centroid inputs.",
    )
    parser.add_argument(
        "--write-temporal-gif",
        action="store_true",
        help="Also write weekly_evolution.gif for temporal inputs (requires imageio).",
    )
    return parser.parse_args()


def resolve_sst_file(candidate: str) -> Path | None:
    if candidate:
        p = Path(candidate).resolve()
        return p if p.exists() else None
    files = sorted(Path("crw_gbr_nc_yearly/sst").glob("*.nc"))
    return files[0].resolve() if files else None


def draw_coastline_from_sst_mask(ax: plt.Axes, sst_file: Path) -> None:
    try:
        import xarray as xr
    except Exception as exc:
        print(f"[warn] xarray unavailable, skipping coastline: {exc}")
        return
    try:
        ds = xr.open_dataset(sst_file)
        if "analysed_sst" not in ds:
            print(f"[warn] analysed_sst missing in {sst_file}, coastline skipped")
            return
        da = ds["analysed_sst"].isel(time=0)
        land = np.isnan(da.values).astype(float)
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        ax.contour(lons, lats, land, levels=[0.5], colors="black", linewidths=0.8, zorder=2)
    except Exception as exc:
        print(f"[warn] failed coastline draw: {exc}")


def class_names_for(n_classes: int) -> list[str]:
    if n_classes == 4:
        return ["None (0%)", "Low (1-10%)", "Moderate (10-50%)", "Severe (>50%)"]
    if n_classes == 3:
        return ["None (0%)", "Moderate (1-50%)", "Severe (>50%)"]
    return [f"Class {i}" for i in range(n_classes)]


def infer_model_config(state_dict: dict[str, torch.Tensor]) -> dict:
    req = ["input_proj.0.weight", "classifier.6.weight", "lstm.weight_hh_l0"]
    for k in req:
        if k not in state_dict:
            raise KeyError(f"Checkpoint missing required key: {k}")

    n_features = int(state_dict["input_proj.0.weight"].shape[1])
    hidden_size = int(state_dict["lstm.weight_hh_l0"].shape[1])
    n_classes = int(state_dict["classifier.6.weight"].shape[0])

    layer_ids = set()
    pat = re.compile(r"^lstm\.weight_ih_l(\d+)$")
    for key in state_dict:
        m = pat.match(key)
        if m:
            layer_ids.add(int(m.group(1)))
    n_layers = (max(layer_ids) + 1) if layer_ids else 1

    bidirectional = any("_reverse" in k for k in state_dict.keys())

    return {
        "n_features": n_features,
        "hidden_size": hidden_size,
        "n_classes": n_classes,
        "n_layers": n_layers,
        "bidirectional": bidirectional,
    }


def load_state_dict_compat(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def resolve_stats_path(stats_path: str, checkpoint_path: Path) -> Path:
    if stats_path:
        p = Path(stats_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Stats file not found: {p}")
        return p

    candidates = [
        checkpoint_path.parent / "normalization_stats.npz",
        checkpoint_path.parent / "results.npz",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        "Normalization stats not found. Provide --stats-path or place "
        "'normalization_stats.npz' next to the checkpoint."
    )


def load_normalization_stats(
    stats_file: Path, n_features: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    with np.load(stats_file, allow_pickle=True) as d:
        required = ["feat_mean", "feat_std", "static_mean", "static_std"]
        missing = [k for k in required if k not in d.files]
        if missing:
            raise KeyError(f"Stats file missing required keys {missing}: {stats_file}")

        feat_mean = np.asarray(d["feat_mean"], dtype=np.float32).reshape(-1)
        feat_std = np.asarray(d["feat_std"], dtype=np.float32).reshape(-1)
        static_mean = np.asarray(d["static_mean"], dtype=np.float32).reshape(-1)
        static_std = np.asarray(d["static_std"], dtype=np.float32).reshape(-1)

    if feat_mean.shape[0] != n_features or feat_std.shape[0] != n_features:
        raise ValueError(
            f"Feature stats shape mismatch in {stats_file}: "
            f"feat_mean={feat_mean.shape}, feat_std={feat_std.shape}, expected=({n_features},)"
        )
    if static_mean.shape[0] < 2 or static_std.shape[0] < 2:
        raise ValueError(
            f"Static stats shape mismatch in {stats_file}: "
            f"static_mean={static_mean.shape}, static_std={static_std.shape}, expected at least (2,)"
        )

    feat_std = feat_std.copy()
    static_std = static_std.copy()
    feat_std[feat_std == 0] = 1.0
    static_std[static_std == 0] = 1.0
    return feat_mean, feat_std, static_mean[:2], static_std[:2], f"stats file ({stats_file})"


def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_feature_idx(feature_names: list[str], candidates: list[str]) -> int | None:
    norm_names = [normalize_name(n) for n in feature_names]
    for cand in candidates:
        key = normalize_name(cand)
        if key in norm_names:
            return norm_names.index(key)
    return None


def temporal_mode_available(data: np.lib.npyio.NpzFile) -> bool:
    if "end_time" not in data.files:
        return False
    if "cluster_id" not in data.files:
        return False
    unique_end_times = np.unique(data["end_time"].astype("datetime64[ns]"))
    return len(unique_end_times) > 1


def build_temporal_predictions_df(
    data: np.lib.npyio.NpzFile,
    X: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    if probs.shape[1] != 3:
        raise ValueError(
            "Temporal CSV format requires 3-class probabilities to match "
            "gbr_predictions_temporal.csv."
        )

    end_time = data["end_time"].astype("datetime64[ns]")
    unique_dates = np.sort(np.unique(end_time))
    week_lookup = {ts: i + 1 for i, ts in enumerate(unique_dates)}
    week_num = np.array([week_lookup[ts] for ts in end_time], dtype=np.int64)
    date_str = pd.to_datetime(end_time).strftime("%Y-%m-%d")

    if "centroid_lat" in data.files and "centroid_lon" in data.files:
        centroid_lat = data["centroid_lat"].astype(np.float64)
        centroid_lon = data["centroid_lon"].astype(np.float64)
    else:
        centroid_lat = data["meta"][:, 0].astype(np.float64)
        centroid_lon = data["meta"][:, 1].astype(np.float64)

    n_records = (
        data["cluster_n_records"].astype(np.float64)
        if "cluster_n_records" in data.files
        else np.full((len(preds),), np.nan, dtype=np.float64)
    )
    match_distance = (
        data["centroid_match_distance_km"].astype(np.float64)
        if "centroid_match_distance_km" in data.files
        else np.full((len(preds),), np.nan, dtype=np.float64)
    )

    feature_names = [str(x) for x in data["feature_names"].tolist()] if "feature_names" in data.files else []
    dhw_idx = find_feature_idx(feature_names, ["tsa_dhw", "TSA_DHW", "crw_dhw"])
    sst_idx = find_feature_idx(feature_names, ["filled_sst", "FilledSST", "crw_sst", "analysed_sst"])

    if dhw_idx is None:
        tsa_dhw_max = np.full((len(preds),), np.nan, dtype=np.float64)
        tsa_dhw_last = np.full((len(preds),), np.nan, dtype=np.float64)
    else:
        tsa_dhw_max = X[:, :, dhw_idx].max(axis=1).astype(np.float64)
        tsa_dhw_last = X[:, -1, dhw_idx].astype(np.float64)

    if sst_idx is None:
        filled_sst_last = np.full((len(preds),), np.nan, dtype=np.float64)
    else:
        filled_sst_last = X[:, -1, sst_idx].astype(np.float64)

    out = pd.DataFrame(
        {
            "week": week_num,
            "date": date_str,
            "cluster_id": data["cluster_id"].astype(np.int64),
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "n_records": n_records,
            "match_distance_km": np.round(match_distance, 2),
            "predicted_class": preds.astype(np.int64),
            "predicted_label": [class_names[i] for i in preds.tolist()],
            "prob_none": np.round(probs[:, 0], 4),
            "prob_moderate": np.round(probs[:, 1], 4),
            "prob_severe": np.round(probs[:, 2], 4),
            "risk_score": np.round(1.0 - probs[:, 0], 4),
            "TSA_DHW_max": tsa_dhw_max,
            "TSA_DHW_last": tsa_dhw_last,
            "FilledSST_last": filled_sst_last,
        }
    )
    out = out.sort_values(["week", "cluster_id"]).reset_index(drop=True)
    return out


def monthly_tick_positions(dates: list[pd.Timestamp]) -> tuple[list[int], list[str]]:
    ticks: list[int] = []
    labels: list[str] = []
    for i, d in enumerate(dates):
        if d.day <= 7 or i == 0 or i == len(dates) - 1:
            ticks.append(i)
            labels.append(d.strftime("%b '%y"))
    return ticks, labels


def make_temporal_plots(
    temporal_df: pd.DataFrame,
    output_dir: Path,
    class_names: list[str],
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    dpi: int,
    write_gif: bool,
) -> list[Path]:
    saved: list[Path] = []
    class_colors = {0: "#4393c3", 1: "#f4a582", 2: "#d6604d"}
    class_order = [0, 1, 2]
    legend_labels = class_names[:3]

    by_week = {int(w): g.copy() for w, g in temporal_df.groupby("week")}
    weeks = sorted(by_week.keys())
    week_dates = [pd.to_datetime(by_week[w]["date"].iloc[0]) for w in weeks]
    n_weeks = len(weeks)

    # 1) multi-panel weekly evolution grid
    sample_idx = np.linspace(0, n_weeks - 1, min(16, n_weeks), dtype=int)
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    axes_flat = axes.flatten()
    for panel_i, ax in enumerate(axes_flat):
        if panel_i >= len(sample_idx):
            ax.axis("off")
            continue
        week = weeks[int(sample_idx[panel_i])]
        wk = by_week[week]
        for c in class_order:
            m = wk["predicted_class"].to_numpy() == c
            if np.any(m):
                ax.scatter(
                    wk.loc[m, "centroid_lon"],
                    wk.loc[m, "centroid_lat"],
                    c=class_colors[c],
                    s=15,
                    alpha=0.85,
                    edgecolors="none",
                )
        counts = np.bincount(wk["predicted_class"].to_numpy().astype(np.int64), minlength=3)
        ax.text(
            0.02,
            0.02,
            f"N:{counts[0]} M:{counts[1]} S:{counts[2]}",
            transform=ax.transAxes,
            fontsize=7,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(week_dates[int(sample_idx[panel_i])].strftime("%b %d '%y"), fontsize=10, fontweight="bold")
    legend_handles = [
        mpatches.Patch(facecolor=class_colors[c], label=legend_labels[c]) for c in class_order
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        "Bleaching Prediction Evolution Over Weeks\n"
        f"{week_dates[0].strftime('%b %d, %Y')} -> {week_dates[-1].strftime('%b %d, %Y')}",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    p1 = output_dir / "weekly_evolution_grid.png"
    fig.savefig(p1, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # 2) stacked area chart
    counts = []
    for w in weeks:
        arr = by_week[w]["predicted_class"].to_numpy().astype(np.int64)
        counts.append(np.bincount(arr, minlength=3))
    counts_arr = np.asarray(counts)
    denom = counts_arr.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1
    pct = counts_arr / denom * 100.0
    x = np.arange(n_weeks)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.stackplot(
        x,
        pct[:, 0],
        pct[:, 1],
        pct[:, 2],
        labels=legend_labels,
        colors=[class_colors[0], class_colors[1], class_colors[2]],
        alpha=0.85,
    )
    xt, xl = monthly_tick_positions(week_dates)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, n_weeks - 1)
    ax.set_ylabel("Percentage of Reef Clusters")
    ax.set_xlabel("Date")
    ax.set_title("Bleaching Severity Distribution Over Time", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    p2 = output_dir / "weekly_stacked_area.png"
    fig.savefig(p2, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p2)

    # 3) risk heatmap (cluster x week)
    cluster_meta = (
        temporal_df.groupby("cluster_id", as_index=False)["centroid_lat"].first().sort_values("centroid_lat", ascending=False)
    )
    ordered_clusters = cluster_meta["cluster_id"].to_numpy()
    risk_pivot = temporal_df.pivot(index="cluster_id", columns="week", values="risk_score")
    risk_pivot = risk_pivot.reindex(index=ordered_clusters, columns=weeks)
    fig, ax = plt.subplots(figsize=(20, 8))
    im = ax.imshow(risk_pivot.to_numpy(dtype=np.float32), aspect="auto", cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    xt, xl = monthly_tick_positions(week_dates)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=9)
    yt = np.arange(0, len(ordered_clusters), max(1, len(ordered_clusters) // 9))
    ax.set_yticks(yt)
    lat_labels = cluster_meta["centroid_lat"].to_numpy()
    ax.set_yticklabels([f"{lat_labels[i]:.1f}°" for i in yt], fontsize=8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Reef Cluster (North -> South)")
    ax.set_title("Bleaching Risk Score per Cluster Over Time", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Risk Score (P(Moderate)+P(Severe))")
    fig.tight_layout()
    p3 = output_dir / "weekly_risk_heatmap.png"
    fig.savefig(p3, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(p3)

    # 4) optional GIF
    if write_gif:
        if not HAS_IMAGEIO:
            print("[warn] imageio unavailable; skipping weekly_evolution.gif")
        else:
            frames = []
            for i, w in enumerate(weeks):
                wk = by_week[w]
                fig, ax = plt.subplots(figsize=(10, 8))
                for c in class_order:
                    m = wk["predicted_class"].to_numpy() == c
                    if np.any(m):
                        ax.scatter(
                            wk.loc[m, "centroid_lon"],
                            wk.loc[m, "centroid_lat"],
                            c=class_colors[c],
                            s=50,
                            alpha=0.8,
                            edgecolors="black",
                            linewidth=0.3,
                            label=f"{legend_labels[c]} ({int(np.sum(m))})",
                        )
                ax.set_xlim(lon_min, lon_max)
                ax.set_ylim(lat_min, lat_max)
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.set_title(week_dates[i].strftime("%B %d, %Y"), fontsize=14, fontweight="bold")
                ax.legend(loc="lower left", fontsize=9)
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.canvas.draw()
                # Use direct RGBA array view from canvas to avoid backend/DPI-dependent
                # width/height mismatches when manually reshaping raw bytes.
                rgba = np.asarray(fig.canvas.buffer_rgba())
                if rgba.ndim != 3 or rgba.shape[2] != 4:
                    raise RuntimeError(f"Unexpected canvas RGBA shape: {rgba.shape}")
                frames.append(rgba[:, :, :3].copy())
                plt.close(fig)
            frames.extend([frames[-1]] * 5)
            p4 = output_dir / "weekly_evolution.gif"
            imageio.mimsave(p4, frames, duration=0.4)
            saved.append(p4)

    return saved


def make_temporal_summary(temporal_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for week, wk in temporal_df.groupby("week"):
        date = wk["date"].iloc[0]
        arr = wk["predicted_class"].to_numpy().astype(np.int64)
        counts = np.bincount(arr, minlength=3)
        n = len(wk)
        risk = wk["risk_score"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "week": int(week),
                "date": str(date),
                "n_none": int(counts[0]),
                "n_moderate": int(counts[1]),
                "n_severe": int(counts[2]),
                "pct_none": round(float(counts[0] / n * 100.0), 1),
                "pct_moderate": round(float(counts[1] / n * 100.0), 1),
                "pct_severe": round(float(counts[2] / n * 100.0), 1),
                "mean_risk_score": round(float(np.mean(risk)), 4),
                "max_risk_score": round(float(np.max(risk)), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("week").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_path = Path(args.input_npz).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {input_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    data = np.load(input_path, allow_pickle=True)
    if "X" not in data.files or "meta" not in data.files:
        raise KeyError("Input NPZ must contain X and meta arrays.")

    X = data["X"]
    meta = data["meta"]
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (N, T, F); got {X.shape}")
    if meta.ndim != 2 or meta.shape[0] != X.shape[0] or meta.shape[1] < 2:
        raise ValueError(f"Expected meta shape (N, >=2); got {meta.shape}")

    state_dict = load_state_dict_compat(ckpt_path, device)
    cfg = infer_model_config(state_dict)
    if cfg["n_features"] != X.shape[2]:
        raise ValueError(
            f"Feature mismatch: checkpoint expects {cfg['n_features']} features, "
            f"but input X has {X.shape[2]}."
        )

    stats_file = resolve_stats_path(args.stats_path, ckpt_path)
    feat_mean, feat_std, static_mean, static_std, stats_src = load_normalization_stats(
        stats_file, X.shape[-1]
    )

    X_norm = (X - feat_mean) / feat_std
    static = (meta[:, :2] - static_mean) / static_std

    ds = SequenceDataset(X_norm, static)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = BleachingLSTMFlexible(
        n_features=cfg["n_features"],
        hidden_size=cfg["hidden_size"],
        n_layers=cfg["n_layers"],
        n_classes=cfg["n_classes"],
        bidirectional=cfg["bidirectional"],
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print("=" * 70)
    print("Inference")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(
        f"Model config: n_features={cfg['n_features']}, n_classes={cfg['n_classes']}, "
        f"hidden_size={cfg['hidden_size']}, n_layers={cfg['n_layers']}, "
        f"bidirectional={cfg['bidirectional']}"
    )
    print(f"X shape: {X.shape}")
    print(f"Feature normalization: {stats_src}")
    print(f"Static normalization: {stats_src}")
    print(f"Device: {device}")

    logits_all = []
    probs_all = []
    preds_all = []
    attn_all = []
    with torch.no_grad():
        for x_seq, x_static in loader:
            x_seq = x_seq.to(device)
            x_static = x_static.to(device)
            logits, attn = model(x_seq, x_static)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            logits_all.append(logits.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            preds_all.append(preds.cpu().numpy())
            attn_all.append(attn.cpu().numpy())

    logits = np.concatenate(logits_all, axis=0)
    probs = np.concatenate(probs_all, axis=0)
    preds = np.concatenate(preds_all, axis=0)
    attn = np.concatenate(attn_all, axis=0)
    conf = probs.max(axis=1)

    n_classes = probs.shape[1]
    class_names = class_names_for(n_classes)
    temporal_mode = temporal_mode_available(data)
    latest_mask = None
    latest_date = None
    if temporal_mode:
        end_time_all = data["end_time"].astype("datetime64[ns]")
        latest_date = np.max(end_time_all)
        latest_mask = end_time_all == latest_date

    # Use centroid coordinates for map if requested and available.
    if args.use_centroid_coords and "centroid_lat" in data.files and "centroid_lon" in data.files:
        plot_lat = data["centroid_lat"].astype(np.float32)
        plot_lon = data["centroid_lon"].astype(np.float32)
        loc_source = "centroid_lat/lon"
    else:
        plot_lat = meta[:, 0].astype(np.float32)
        plot_lon = meta[:, 1].astype(np.float32)
        loc_source = "meta lat/lon"

    # Save NPZ outputs
    out_npz = output_dir / "inference_outputs.npz"
    payload = {
        "predictions": preds,
        "probabilities": probs,
        "logits": logits,
        "attention_weights": attn,
        "confidence": conf,
        "meta": meta,
        "plot_lat": plot_lat,
        "plot_lon": plot_lon,
        "class_names": np.array(class_names, dtype=object),
        "feature_names": data["feature_names"] if "feature_names" in data.files else np.array([], dtype=object),
    }
    for key in ["end_time", "cluster_id", "centroid_lat", "centroid_lon", "matched_lat", "matched_lon", "cluster_n_records"]:
        if key in data.files:
            payload[key] = data[key]
    np.savez_compressed(out_npz, **payload)

    # Save CSV outputs
    out_csv = output_dir / "inference_predictions.csv"
    out_df = pd.DataFrame(
        {
            "sample_idx": np.arange(len(preds), dtype=np.int64),
            "lat": plot_lat,
            "lon": plot_lon,
            "meta_lat": meta[:, 0],
            "meta_lon": meta[:, 1],
            "meta_year": meta[:, 2] if meta.shape[1] > 2 else np.nan,
            "meta_month": meta[:, 3] if meta.shape[1] > 3 else np.nan,
            "predicted_class": preds,
            "predicted_label": [class_names[i] for i in preds.tolist()],
            "confidence": conf,
        }
    )
    if "end_time" in data.files:
        out_df["end_time"] = data["end_time"].astype("datetime64[ns]").astype(str)
    if "cluster_id" in data.files:
        out_df["cluster_id"] = data["cluster_id"]
    if "centroid_match_distance_km" in data.files:
        out_df["centroid_match_distance_km"] = data["centroid_match_distance_km"]
    for c in range(n_classes):
        out_df[f"prob_class_{c}"] = probs[:, c]
    out_df.to_csv(out_csv, index=False)

    temporal_csv = None
    temporal_summary_csv = None
    temporal_plot_paths: list[Path] = []
    if temporal_mode:
        temporal_df = build_temporal_predictions_df(data, X, preds, probs, class_names)
        temporal_csv = output_dir / "gbr_predictions_temporal.csv"
        temporal_df.to_csv(temporal_csv, index=False)

        temporal_summary = make_temporal_summary(temporal_df)
        temporal_summary_csv = output_dir / "weekly_summary.csv"
        temporal_summary.to_csv(temporal_summary_csv, index=False)

        if args.skip_temporal_viz:
            print("[info] temporal visualizations skipped (--skip-temporal-viz)")
        else:
            temporal_plot_paths = make_temporal_plots(
                temporal_df=temporal_df,
                output_dir=output_dir,
                class_names=class_names,
                lat_min=args.lat_min,
                lat_max=args.lat_max,
                lon_min=args.lon_min,
                lon_max=args.lon_max,
                dpi=args.dpi,
                write_gif=args.write_temporal_gif,
            )

    # Summary JSON
    counts = np.bincount(preds, minlength=n_classes)
    summary = {
        "input_npz": str(input_path),
        "checkpoint": str(ckpt_path),
        "num_samples": int(len(preds)),
        "class_names": class_names,
        "predicted_counts": {class_names[i]: int(counts[i]) for i in range(n_classes)},
        "predicted_percent": {class_names[i]: float(counts[i] / len(preds) * 100.0) for i in range(n_classes)},
        "location_source": loc_source,
        "feature_norm_source": stats_src,
        "static_norm_source": stats_src,
        "temporal_mode": bool(temporal_mode),
    }
    if temporal_mode and latest_date is not None:
        summary["num_weeks"] = int(len(np.unique(data["end_time"].astype("datetime64[ns]"))))
        summary["latest_date"] = str(pd.Timestamp(latest_date).strftime("%Y-%m-%d"))
    out_json = output_dir / "inference_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Visualization
    out_png = output_dir / "inference_severity_map.png"
    fig, ax = plt.subplots(figsize=(10, 8))

    # Colors by severity class index.
    palette_4 = ["#2b8cbe", "#fee08b", "#f46d43", "#d73027"]
    palette_3 = ["#2b8cbe", "#fdae61", "#d73027"]
    palette = palette_3 if n_classes == 3 else palette_4
    if n_classes > len(palette):
        palette = [plt.cm.tab20(i) for i in range(n_classes)]

    # Coastline
    sst_file = resolve_sst_file(args.sst_file)
    if sst_file is not None:
        draw_coastline_from_sst_mask(ax, sst_file)
    else:
        print("[warn] no SST file found for coastline")

    if temporal_mode and latest_mask is not None:
        map_mask = latest_mask
        map_preds = preds[map_mask]
        map_lat = plot_lat[map_mask]
        map_lon = plot_lon[map_mask]
        map_suffix = f" (Latest Week: {pd.Timestamp(latest_date).strftime('%Y-%m-%d')})"
    else:
        map_mask = slice(None)
        map_preds = preds
        map_lat = plot_lat
        map_lon = plot_lon
        map_suffix = ""

    n_points = len(map_preds)
    point_size = 18 if n_points <= 5000 else 6
    alpha = 0.9 if n_points <= 5000 else 0.5
    for c in range(n_classes):
        m = map_preds == c
        if not np.any(m):
            continue
        label = f"{class_names[c]} (n={int(np.sum(m))})"
        ax.scatter(
            map_lon[m],
            map_lat[m],
            s=point_size,
            c=[palette[c]],
            alpha=alpha,
            edgecolors="none",
            label=label,
            zorder=3,
        )

    ax.set_xlim(args.lon_min, args.lon_max)
    ax.set_ylim(args.lat_min, args.lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Predicted Bleaching Severity{map_suffix}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=args.dpi)

    print("\nSaved:")
    print(f"  {out_npz}")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print(f"  {out_png}")
    if temporal_csv is not None:
        print(f"  {temporal_csv}")
    if temporal_summary_csv is not None:
        print(f"  {temporal_summary_csv}")
    for p in temporal_plot_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
