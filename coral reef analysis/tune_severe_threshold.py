#!/usr/bin/env python3
"""
Tune a severe-bleaching decision threshold with a recall constraint.

Given model outputs (probabilities or logits) and true labels, sweep thresholds
on the severe class score and pick the threshold that:
  1) satisfies severe recall >= target (if possible), then
  2) maximizes severe precision, then severe F1, then accuracy.

This is useful for recall-first monitoring systems where severe misses are costly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def severe_metrics(y_true: np.ndarray, y_pred: np.ndarray, severe_class: int) -> dict:
    tp = int(np.sum((y_true == severe_class) & (y_pred == severe_class)))
    fp = int(np.sum((y_true != severe_class) & (y_pred == severe_class)))
    fn = int(np.sum((y_true == severe_class) & (y_pred != severe_class)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def apply_severe_threshold(scores: np.ndarray, threshold: float, severe_class: int) -> np.ndarray:
    """
    Threshold rule:
      - If severe_score >= threshold -> predict severe
      - Else -> argmax among non-severe classes
    """
    preds = np.argmax(scores, axis=1)
    severe_score = scores[:, severe_class]
    is_severe = severe_score >= threshold

    if np.any(~is_severe):
        non_severe_scores = scores[~is_severe].copy()
        non_severe_scores[:, severe_class] = -np.inf
        preds[~is_severe] = np.argmax(non_severe_scores, axis=1)
    preds[is_severe] = severe_class
    return preds


def class_names_for(n_classes: int):
    if n_classes == 4:
        return ["None (0%)", "Low (1-10%)", "Moderate (10-50%)", "Severe (>50%)"]
    if n_classes == 3:
        return ["None (0%)", "Moderate (1-50%)", "Severe (>50%)"]
    return [f"Class {i}" for i in range(n_classes)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune severe threshold with recall constraint.")
    parser.add_argument(
        "--results-path",
        default="results.npz",
        help="Path to .npz file containing labels and probabilities/logits.",
    )
    parser.add_argument(
        "--labels-key",
        default="labels",
        help="Array key for true labels (default: labels).",
    )
    parser.add_argument(
        "--scores-key",
        default="",
        help="Optional array key for class scores; if omitted uses probabilities then logits.",
    )
    parser.add_argument(
        "--severe-class",
        type=int,
        default=-1,
        help="Severe class index. Default -1 means last class index.",
    )
    parser.add_argument(
        "--min-severe-recall",
        type=float,
        default=0.85,
        help="Minimum severe recall target (default: 0.85).",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="Optional output .npz to save tuned predictions and chosen threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_path).resolve()
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    data = np.load(results_path, allow_pickle=True)
    if args.labels_key not in data.files:
        raise KeyError(f"Missing labels key '{args.labels_key}' in {results_path}")
    y_true = data[args.labels_key].astype(int)

    if args.scores_key:
        if args.scores_key not in data.files:
            raise KeyError(f"Missing scores key '{args.scores_key}' in {results_path}")
        raw_scores = data[args.scores_key]
        used_key = args.scores_key
    elif "probabilities" in data.files:
        raw_scores = data["probabilities"]
        used_key = "probabilities"
    elif "logits" in data.files:
        raw_scores = data["logits"]
        used_key = "logits"
    else:
        raise KeyError(
            "No scores found. Expected one of: probabilities, logits, or --scores-key."
        )

    if raw_scores.ndim != 2:
        raise ValueError(f"Scores array must be 2D (N, C), got shape {raw_scores.shape}")
    if raw_scores.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Score/label length mismatch: scores N={raw_scores.shape[0]} labels N={y_true.shape[0]}"
        )

    if used_key == "logits":
        scores = softmax_np(raw_scores.astype(np.float64, copy=False))
    else:
        scores = raw_scores.astype(np.float64, copy=False)

    n_classes = scores.shape[1]
    severe_class = n_classes - 1 if args.severe_class < 0 else args.severe_class
    if not (0 <= severe_class < n_classes):
        raise ValueError(f"Invalid severe class index {severe_class} for n_classes={n_classes}")

    class_names = class_names_for(n_classes)
    label_order = list(range(n_classes))

    # Baseline argmax metrics
    baseline_pred = np.argmax(scores, axis=1)
    baseline_sv = severe_metrics(y_true, baseline_pred, severe_class)
    baseline_acc = accuracy_score(y_true, baseline_pred)

    severe_scores = scores[:, severe_class]
    thresholds = np.unique(np.concatenate([severe_scores, np.array([0.0, 1.0])]))

    candidates = []
    for t in thresholds:
        pred_t = apply_severe_threshold(scores, float(t), severe_class)
        sv = severe_metrics(y_true, pred_t, severe_class)
        candidates.append(
            {
                "threshold": float(t),
                "pred": pred_t,
                "severe_precision": sv["precision"],
                "severe_recall": sv["recall"],
                "severe_f1": sv["f1"],
                "accuracy": accuracy_score(y_true, pred_t),
                "macro_f1": f1_score(y_true, pred_t, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_true, pred_t, average="weighted", zero_division=0),
            }
        )

    viable = [c for c in candidates if c["severe_recall"] >= args.min_severe_recall]
    if viable:
        best = sorted(
            viable,
            key=lambda c: (c["severe_precision"], c["severe_f1"], c["accuracy"]),
            reverse=True,
        )[0]
        met_target = True
    else:
        best = sorted(
            candidates,
            key=lambda c: (c["severe_recall"], c["severe_precision"], c["severe_f1"]),
            reverse=True,
        )[0]
        met_target = False

    tuned_pred = best["pred"]

    print("=" * 70)
    print("Severe Threshold Tuning")
    print("=" * 70)
    print(f"Results file: {results_path}")
    print(f"Scores key used: {used_key}")
    print(f"Num samples: {len(y_true)}")
    print(f"Num classes: {n_classes}")
    print(f"Severe class index: {severe_class}")
    print(f"Recall target: {args.min_severe_recall:.3f}")
    print()
    print("Baseline (argmax):")
    print(
        f"  Severe precision={baseline_sv['precision']:.3f} "
        f"recall={baseline_sv['recall']:.3f} "
        f"f1={baseline_sv['f1']:.3f} | accuracy={baseline_acc:.3f}"
    )
    print()
    print("Selected threshold:")
    print(f"  threshold={best['threshold']:.6f}")
    print(f"  meets recall target: {'yes' if met_target else 'no'}")
    print(
        f"  Severe precision={best['severe_precision']:.3f} "
        f"recall={best['severe_recall']:.3f} "
        f"f1={best['severe_f1']:.3f} | accuracy={best['accuracy']:.3f}"
    )
    print(
        f"  macro_f1={best['macro_f1']:.3f} weighted_f1={best['weighted_f1']:.3f}"
    )

    print("\nTuned Classification Report:")
    print(
        classification_report(
            y_true,
            tuned_pred,
            labels=label_order,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Tuned Confusion Matrix:")
    cm = confusion_matrix(y_true, tuned_pred, labels=label_order)
    print(f"{'':>15} | " + " | ".join(f"{n:>16}" for n in class_names))
    print("-" * (18 + 19 * len(class_names)))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>15} | " + " | ".join(f"{v:>16d}" for v in row))

    if args.output_path:
        output_path = Path(args.output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            labels=y_true,
            baseline_predictions=baseline_pred,
            tuned_predictions=tuned_pred,
            severe_scores=severe_scores,
            threshold=best["threshold"],
            severe_class=severe_class,
            min_severe_recall=args.min_severe_recall,
            class_names=np.array(class_names, dtype=object),
        )
        print(f"\nSaved tuned outputs to: {output_path}")


if __name__ == "__main__":
    main()
