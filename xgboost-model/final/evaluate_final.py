"""
Generate evaluation plots: confusion matrix, feature importance, SHAP.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "xgboost-model" / "data" / "sequences_reduced_16.npz"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 42


def flatten_raw(X_seq, meta, feature_names):
    flat = {}
    for w in range(X_seq.shape[1]):
        for i, fname in enumerate(feature_names):
            flat[f"{fname}_week{w:02d}"] = X_seq[:, w, i]
    flat["latitude"] = meta[:, 0]
    flat["longitude"] = meta[:, 1]
    flat["month"] = meta[:, 3]
    return pd.DataFrame(flat).fillna(0)


def get_test_set():
    """Reproduce the same test split used during training."""
    data = np.load(DATA_PATH, allow_pickle=True)
    X_seq = data["X"]
    y = data["y"]
    meta = data["meta"]
    feature_names = list(data["feature_names"])

    _, X_temp, _, y_temp, _, meta_temp = train_test_split(
        X_seq, y, meta, test_size=0.30, stratify=y, random_state=SEED
    )
    _, X_test_seq, _, y_test, _, meta_test = train_test_split(
        X_temp, y_temp, meta_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    X_test = flatten_raw(X_test_seq, meta_test, feature_names)
    return X_test, y_test


def plot_confusion(ax, data, title, xticklabels, yticklabels, fmt):
    """Plot confusion matrix with seaborn when available, matplotlib otherwise."""
    if HAS_SEABORN:
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            ax=ax,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )
    else:
        im = ax.imshow(data, cmap="Blues")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                text = f"{int(val)}" if fmt == "d" else f"{val:.1f}"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)


def main():
    # Load model and config
    model = XGBClassifier()
    model.load_model(str(RESULTS_DIR / "model_3class.json"))

    with open(RESULTS_DIR / "model_config.json") as f:
        config = json.load(f)

    CLASS_LABELS = config["class_labels"]
    if "year" in config.get("feature_columns", []):
        raise ValueError(
            "Loaded model_config still includes 'year' as a feature. "
            "Retrain with final/train_final.py to produce year-free artifacts."
        )
    PLOT_LABELS = ["None (0%)", "Moderate\n(1-50%)", "Severe\n(>50%)"]
    if not HAS_SEABORN:
        print(
            f"seaborn not installed in current Python ({sys.executable}); "
            "using matplotlib fallback for confusion matrix plots."
        )

    # Load test results
    test_data = np.load(RESULTS_DIR / "test_results.npz", allow_pickle=True)
    y_test = test_data["y_test"]
    y_pred = test_data["y_pred"]

    # Get test features for SHAP
    X_test, _ = get_test_set()

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_confusion(
        axes[0], cm, "Confusion Matrix (Counts)", PLOT_LABELS, PLOT_LABELS, "d"
    )
    plot_confusion(
        axes[1], cm_pct, "Confusion Matrix (Row-Normalized %)", PLOT_LABELS, PLOT_LABELS, ".1f"
    )

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved {RESULTS_DIR / 'confusion_matrix.png'}")

    # === Feature Importance ===
    importance = model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp])
    ax.set_xlabel("Gain")
    ax.set_title("Top 20 Feature Importance -- 3-Class Final Model")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved {RESULTS_DIR / 'feature_importance.png'}")

    # === SHAP ===
    if not HAS_SHAP:
        raise ModuleNotFoundError(
            "shap is not installed in current Python "
            f"({sys.executable}). Install shap in the environment used to run this script."
        )
    print("Computing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Convert (N, F, C) to list of (N, F) per class if needed
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    else:
        shap_list = shap_values

    # All classes summary
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_list, X_test, show=False, max_display=15,
                      class_names=["None", "Moderate", "Severe"])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {RESULTS_DIR / 'shap_summary.png'}")

    # Severe class (class 2)
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_list[2], X_test, show=False, max_display=15)
    plt.title("SHAP Values -- Severe Bleaching (>50%)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_severe.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {RESULTS_DIR / 'shap_severe.png'}")

    print("\nAll evaluation plots saved to results/")


if __name__ == "__main__":
    main()
