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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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
    }
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

    n_points = len(preds)
    point_size = 18 if n_points <= 5000 else 6
    alpha = 0.9 if n_points <= 5000 else 0.5
    for c in range(n_classes):
        m = preds == c
        if not np.any(m):
            continue
        label = f"{class_names[c]} (n={int(np.sum(m))})"
        ax.scatter(
            plot_lon[m],
            plot_lat[m],
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
    ax.set_title("Predicted Bleaching Severity")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=args.dpi)

    print("\nSaved:")
    print(f"  {out_npz}")
    print(f"  {out_csv}")
    print(f"  {out_json}")
    print(f"  {out_png}")


if __name__ == "__main__":
    main()
