#!/usr/bin/env python3
"""
Create metric charts and summary files from a training/evaluation results.npz.

Expected keys (common):
  - predictions, labels
Optional keys:
  - probabilities, attention_weights
  - history_train_loss, history_val_loss, history_val_acc
  - class_names
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze results.npz and export charts/metrics.")
    parser.add_argument(
        "--results-npz",
        default="outputs/trained_model_2/results.npz",
        help="Path to results.npz file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/trained_model_2/analysis",
        help="Directory to save plots and summary artifacts.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Model Evaluation",
        help="Title prefix for generated figures.",
    )
    return parser.parse_args()


def resolve_class_names(data: np.lib.npyio.NpzFile, n_classes: int) -> list[str]:
    if "class_names" in data.files:
        return [str(x) for x in data["class_names"].tolist()]
    return [f"Class {i}" for i in range(n_classes)]


def plot_confusion(
    cm: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str,
    normalize: bool = False,
) -> None:
    vals = cm.astype(np.float64)
    if normalize:
        row_sums = vals.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        vals = vals / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(vals, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            txt = f"{vals[i, j]:.2f}" if normalize else f"{int(vals[i, j])}"
            color = "white" if vals[i, j] > (vals.max() * 0.6) else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    train_loss: np.ndarray,
    val_loss: np.ndarray,
    val_acc: np.ndarray | None,
    out_path: Path,
    title: str,
) -> None:
    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3, linestyle="--")
    ax1.set_title(title)

    handles = []
    labels = []
    h, l = ax1.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

    if val_acc is not None and len(val_acc) == len(epochs):
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_acc, label="Val Accuracy", color="#2ca02c", linewidth=2, alpha=0.8)
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.0)
        h2, l2 = ax2.get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)

    ax1.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_label_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    n_classes = len(class_names)
    true_counts = np.bincount(y_true.astype(np.int64), minlength=n_classes)
    pred_counts = np.bincount(y_pred.astype(np.int64), minlength=n_classes)
    x = np.arange(n_classes)
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, true_counts, width, label="True", color="#4c72b0")
    ax.bar(x + width / 2, pred_counts, width, label="Pred", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_hist(
    probs: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    conf = probs.max(axis=1)
    correct = y_true == y_pred
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(conf[correct], bins=20, alpha=0.7, label="Correct", color="#2ca02c")
    ax.hist(conf[~correct], bins=20, alpha=0.7, label="Incorrect", color="#d62728")
    ax.set_xlabel("Max Predicted Probability")
    ax.set_ylabel("Samples")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_attention_profile(
    attn: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    seq_len = attn.shape[1]
    x = np.arange(seq_len)
    fig, ax = plt.subplots(figsize=(10, 5))
    overall = attn.mean(axis=0)
    ax.plot(x, overall, linewidth=2.5, color="black", label="Overall")
    for c, name in enumerate(class_names):
        mask = y_true == c
        if np.any(mask):
            ax.plot(x, attn[mask].mean(axis=0), linewidth=1.7, label=name)
    ax.set_xlabel("Timestep Index (0=oldest)")
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_npz).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        raise FileNotFoundError(f"results.npz not found: {results_path}")

    data = np.load(results_path, allow_pickle=True)
    required = {"predictions", "labels"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"Missing required keys in {results_path}: {missing}")

    y_pred = data["predictions"].astype(np.int64)
    y_true = data["labels"].astype(np.int64)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch predictions {y_pred.shape} vs labels {y_true.shape}")

    n_classes = int(max(np.max(y_true), np.max(y_pred))) + 1
    class_names = resolve_class_names(data, n_classes)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Scalar metrics
    metrics = {
        "results_npz": str(results_path),
        "num_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    metrics_path = out_dir / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / "classification_report.csv", index=True)

    # Charts
    plot_confusion(
        cm,
        class_names,
        out_dir / "confusion_matrix_counts.png",
        f"{args.title_prefix} - Confusion Matrix (Counts)",
        normalize=False,
    )
    plot_confusion(
        cm,
        class_names,
        out_dir / "confusion_matrix_normalized.png",
        f"{args.title_prefix} - Confusion Matrix (Row Normalized)",
        normalize=True,
    )
    plot_label_distribution(
        y_true,
        y_pred,
        class_names,
        out_dir / "class_distribution_true_vs_pred.png",
        f"{args.title_prefix} - True vs Predicted Class Distribution",
    )

    if "probabilities" in data.files:
        probs = data["probabilities"].astype(np.float32)
        if probs.ndim == 2 and probs.shape[0] == len(y_true):
            plot_confidence_hist(
                probs,
                y_true,
                y_pred,
                out_dir / "confidence_histogram.png",
                f"{args.title_prefix} - Confidence Histogram",
            )

    train_loss = data["history_train_loss"] if "history_train_loss" in data.files else None
    val_loss = data["history_val_loss"] if "history_val_loss" in data.files else None
    val_acc = data["history_val_acc"] if "history_val_acc" in data.files else None
    if train_loss is not None and val_loss is not None:
        plot_training_curves(
            np.asarray(train_loss),
            np.asarray(val_loss),
            np.asarray(val_acc) if val_acc is not None else None,
            out_dir / "training_curves.png",
            f"{args.title_prefix} - Training History",
        )

    if "attention_weights" in data.files:
        attn = data["attention_weights"].astype(np.float32)
        if attn.ndim == 2 and attn.shape[0] == len(y_true):
            plot_attention_profile(
                attn,
                y_true,
                class_names,
                out_dir / "attention_profile.png",
                f"{args.title_prefix} - Mean Attention by Timestep",
            )

    print("=" * 70)
    print("Analysis complete")
    print("=" * 70)
    print(f"Input: {results_path}")
    print(f"Output dir: {out_dir}")
    print("Saved:")
    for name in [
        "metrics_summary.json",
        "classification_report.csv",
        "confusion_matrix_counts.png",
        "confusion_matrix_normalized.png",
        "class_distribution_true_vs_pred.png",
        "confidence_histogram.png",
        "training_curves.png",
        "attention_profile.png",
    ]:
        p = out_dir / name
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
