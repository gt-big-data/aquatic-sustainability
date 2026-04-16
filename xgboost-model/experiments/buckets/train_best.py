"""
Step 2: Train the final model with the winning bucket scheme.

Reads winner_config.json from analyze_buckets.py, trains on the full
dataset with raw-weekly flattening (68 features), saves model + plots.
"""

import json
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ── paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
TRAIN_NPZ = ROOT / "xgboost-model" / "data" / "sequences_reduced_16.npz"
GCBD_CSV = ROOT / "coral reef analysis" / "datasets" / "global_bleaching_environmental.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def flatten_raw(X_seq, meta, feature_names):
    flat = {}
    for w in range(X_seq.shape[1]):
        for i, fname in enumerate(feature_names):
            flat[f"{fname}_week{w:02d}"] = X_seq[:, w, i]
    flat["latitude"] = meta[:, 0]
    flat["longitude"] = meta[:, 1]
    flat["year"] = meta[:, 2]
    flat["month"] = meta[:, 3]
    return pd.DataFrame(flat).fillna(0)


def recover_continuous_bleaching(npz_path, gcbd_path):
    data = np.load(npz_path, allow_pickle=True)
    X_seq = data["X"]
    meta = data["meta"]
    y_orig = data["y"]
    feature_names = list(data["feature_names"])

    gcbd = pd.read_csv(gcbd_path, low_memory=False)
    gcbd["Percent_Bleaching"] = pd.to_numeric(gcbd["Percent_Bleaching"], errors="coerce")
    gcbd = gcbd.dropna(subset=["Percent_Bleaching", "Date_Year", "Latitude_Degrees", "Longitude_Degrees"])
    gcbd = gcbd[gcbd["Date_Year"] >= 1983]
    gcbd["event_month"] = gcbd["Date_Month"].fillna(6).astype(int).clip(1, 12)
    gcbd["lat_r"] = np.round(gcbd["Latitude_Degrees"].values, 3)
    gcbd["lon_r"] = np.round(gcbd["Longitude_Degrees"].values, 3)

    gcbd_by_key = defaultdict(list)
    for _, row in gcbd.iterrows():
        k = f"{row['lat_r']}_{row['lon_r']}_{int(row['Date_Year'])}_{row['event_month']}"
        gcbd_by_key[k].append(row["Percent_Bleaching"])

    npz_lat = np.round(meta[:, 0].astype(float), 3)
    npz_lon = np.round(meta[:, 1].astype(float), 3)
    npz_year = meta[:, 2].astype(int)
    npz_month = meta[:, 3].astype(int)

    key_counter = defaultdict(int)
    pct_recovered = np.full(len(meta), np.nan)
    for i in range(len(meta)):
        k = f"{npz_lat[i]}_{npz_lon[i]}_{npz_year[i]}_{npz_month[i]}"
        idx = key_counter[k]
        if k in gcbd_by_key and idx < len(gcbd_by_key[k]):
            pct_recovered[i] = gcbd_by_key[k][idx]
        key_counter[k] += 1

    mask = ~np.isnan(pct_recovered)
    print(f"Recovered: {mask.sum()} / {len(meta)} ({mask.sum()/len(meta)*100:.1f}%)")
    return X_seq, meta, feature_names, pct_recovered


def main():
    # Load winner config
    config_path = RESULTS_DIR / "winner_config.json"
    if not config_path.exists():
        print("ERROR: Run analyze_buckets.py first to produce winner_config.json")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    edges = config["edges"]
    class_labels = config["class_labels"]
    n_classes = config["n_classes"]
    print(f"Winner scheme: {config['scheme']} ({n_classes} classes)")
    print(f"Edges: {edges}")
    print(f"Labels: {class_labels}")

    # ── Load & prepare data ───────────────────────────────────
    X_seq, meta, feature_names, pct_recovered = recover_continuous_bleaching(
        TRAIN_NPZ, GCBD_CSV
    )
    mask = ~np.isnan(pct_recovered)
    X_seq_m = X_seq[mask]
    meta_m = meta[mask]
    pct_m = pct_recovered[mask]

    # Bin with winning scheme
    y = pd.cut(
        pd.Series(pct_m).clip(0, 100),
        bins=edges, labels=list(range(n_classes)), include_lowest=True,
    ).astype(int).values

    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, label, count in zip(unique, class_labels, counts):
        print(f"  Class {cls} [{label}]: {count} ({count/len(y)*100:.1f}%)")

    # ── Split 70/15/15 ────────────────────────────────────────
    X_train_seq, X_temp_seq, y_train, y_temp, meta_train, meta_temp = train_test_split(
        X_seq_m, y, meta_m, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val_seq, X_test_seq, y_val, y_test, meta_val, meta_test = train_test_split(
        X_temp_seq, y_temp, meta_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    X_train_flat = flatten_raw(X_train_seq, meta_train, feature_names)
    X_val_flat = flatten_raw(X_val_seq, meta_val, feature_names)
    X_test_flat = flatten_raw(X_test_seq, meta_test, feature_names)

    print(f"\nSplit sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

    # ── Train ─────────────────────────────────────────────────
    weights = compute_sample_weight("balanced", y_train)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        max_depth=6, learning_rate=0.1, n_estimators=500,
        subsample=0.9, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.1, random_state=SEED, early_stopping_rounds=30,
    )
    model.fit(
        X_train_flat, y_train, sample_weight=weights,
        eval_set=[(X_val_flat, y_val)], verbose=False,
    )

    y_pred = model.predict(X_test_flat)
    y_proba = model.predict_proba(X_test_flat)

    # ── Evaluation ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL MODEL — TEST SET")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_labels, digits=3))
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1: {macro_f1:.4f}")

    # ── Confusion matrix plot ─────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Wrap long labels for plotting
    plot_labels = [l.replace(" (", "\n(") for l in class_labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=plot_labels, yticklabels=plot_labels)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix (Counts)")

    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=axes[1],
                xticklabels=plot_labels, yticklabels=plot_labels)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix (Row-Normalized %)")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"\nSaved {RESULTS_DIR / 'confusion_matrix.png'}")

    # ── Feature importance ────────────────────────────────────
    importance = model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp])
    ax.set_xlabel("Gain")
    ax.set_title("Top 20 Feature Importance — Final Model")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"Saved {RESULTS_DIR / 'feature_importance.png'}")

    # ── Save model + config ───────────────────────────────────
    model.save_model(str(RESULTS_DIR / "best_model.json"))
    print(f"Saved {RESULTS_DIR / 'best_model.json'}")

    model_config = {
        "bucket_scheme": edges,
        "class_labels": class_labels,
        "n_classes": n_classes,
        "flattening": "raw_weekly",
        "n_features": X_train_flat.shape[1],
        "feature_columns": list(X_train_flat.columns),
        "macro_f1": round(macro_f1, 4),
        "scheme_name": config["scheme"],
    }
    with open(RESULTS_DIR / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Saved {RESULTS_DIR / 'model_config.json'}")


if __name__ == "__main__":
    main()
