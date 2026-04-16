"""
Post-hoc evaluation: confusion matrix plots, feature importance, SHAP analysis.

Usage:
    python src/evaluate.py                          # uses outputs/ defaults
    python src/evaluate.py --model outputs/model.json --data data/sequences_reduced_16.npz
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from flatten import load_and_flatten

SEED = 42
CLASS_NAMES = ["None (0%)", "Moderate (1-50%)", "Severe (>50%)"]
PLOT_LABELS = ["None (0%)", "Moderate\n(1-50%)", "Severe\n(>50%)"]


def get_test_set(data_path: str):
    """Reproduce the same test split used during training."""
    X_flat, y, _ = load_and_flatten(data_path)
    _, X_temp, _, y_temp = train_test_split(
        X_flat, y, test_size=0.30, stratify=y, random_state=SEED
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    return X_test, y_test


def plot_confusion_matrix(y_test, y_pred, output_dir: Path):
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
        xticklabels=PLOT_LABELS, yticklabels=PLOT_LABELS,
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix (Counts)")

    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=axes[1],
        xticklabels=PLOT_LABELS, yticklabels=PLOT_LABELS,
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix (Row-Normalized %)")

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir / 'confusion_matrix.png'}")


def plot_feature_importance(model, output_dir: Path):
    importance = model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp])
    ax.set_xlabel("Gain")
    ax.set_title("Top 20 Feature Importance (Gain)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir / 'feature_importance.png'}")


def plot_shap(model, X_test, output_dir: Path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)  # (N, features, classes)

    # Convert (N, F, C) to list of (N, F) per class for summary_plot
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    else:
        shap_list = shap_values  # already a list

    # Severe bleaching (class 2)
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_list[2], X_test, show=False, max_display=20)
    plt.title("SHAP Values -- Severe Bleaching (Class 2)")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_severe.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / 'shap_severe.png'}")

    # All classes
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_list, X_test, show=False, max_display=15,
        class_names=["None", "Moderate", "Severe"],
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_all_classes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / 'shap_all_classes.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots and SHAP analysis")
    parser.add_argument("--model", default="outputs/model.json", help="Saved XGBoost model path")
    parser.add_argument("--data", default="data/sequences_reduced_16.npz", help="Dataset path")
    parser.add_argument("--output-dir", default="outputs", help="Directory for plots")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP analysis (faster)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = XGBClassifier()
    model.load_model(args.model)
    print(f"Loaded model from {args.model}")

    # Get test set
    X_test, y_test = get_test_set(args.data)
    y_pred = model.predict(X_test)

    # Plots
    plot_confusion_matrix(y_test, y_pred, output_dir)
    plot_feature_importance(model, output_dir)

    if not args.skip_shap:
        plot_shap(model, X_test, output_dir)
    else:
        print("Skipping SHAP analysis (--skip-shap)")

    print("\nDone.")


if __name__ == "__main__":
    main()
