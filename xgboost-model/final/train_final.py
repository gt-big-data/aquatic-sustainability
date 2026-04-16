"""
Train final 3-class XGBoost model with raw weekly flattening (67 features).
Best configuration from ablation experiments, excluding year as a predictor.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "xgboost-model" / "data" / "sequences_reduced_16.npz"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
CLASS_LABELS = ["None (0%)", "Moderate (1-50%)", "Severe (>50%)"]
N_CLASSES = 3


def flatten_raw(X_seq, meta, feature_names):
    """Raw weekly flattening: 16 weeks x 4 features + 3 metadata = 67 features."""
    flat = {}
    for w in range(X_seq.shape[1]):
        for i, fname in enumerate(feature_names):
            flat[f"{fname}_week{w:02d}"] = X_seq[:, w, i]
    flat["latitude"] = meta[:, 0]
    flat["longitude"] = meta[:, 1]
    flat["month"] = meta[:, 3]
    return pd.DataFrame(flat).fillna(0)


def main():
    # Load
    data = np.load(DATA_PATH, allow_pickle=True)
    X_seq = data["X"]
    y = data["y"]
    meta = data["meta"]
    feature_names = list(data["feature_names"])

    print(f"Loaded: {X_seq.shape[0]} samples, {X_seq.shape[1]} weeks, {X_seq.shape[2]} features")
    print(f"Classes: {np.bincount(y.astype(int))} ({CLASS_LABELS})")

    # Split 70/15/15 stratified
    X_train_seq, X_temp_seq, y_train, y_temp, meta_train, meta_temp = train_test_split(
        X_seq, y, meta, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val_seq, X_test_seq, y_val, y_test, meta_val, meta_test = train_test_split(
        X_temp_seq, y_temp, meta_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    # Flatten raw weekly
    X_train = flatten_raw(X_train_seq, meta_train, feature_names)
    X_val = flatten_raw(X_val_seq, meta_val, feature_names)
    X_test = flatten_raw(X_test_seq, meta_test, feature_names)

    FEATURE_COLS = list(X_train.columns)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Features: {len(FEATURE_COLS)}")

    # Class weights
    weights = compute_sample_weight("balanced", y_train)

    # Train with best known hyperparameters (from ablation experiments)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        max_depth=6,
        learning_rate=0.1,
        n_estimators=500,
        subsample=0.9,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        random_state=SEED,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n" + "=" * 60)
    print("FINAL 3-CLASS MODEL -- TEST SET")
    print("=" * 60)
    report = classification_report(y_test, y_pred, target_names=CLASS_LABELS, digits=3)
    print(report)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Pred None  Pred Mod  Pred Sev")
    print(f"  Act None    {cm[0,0]:>8}  {cm[0,1]:>8}  {cm[0,2]:>8}")
    print(f"  Act Mod     {cm[1,0]:>8}  {cm[1,1]:>8}  {cm[1,2]:>8}")
    print(f"  Act Sev     {cm[2,0]:>8}  {cm[2,1]:>8}  {cm[2,2]:>8}")

    # Save model
    model.save_model(str(RESULTS_DIR / "model_3class.json"))

    # Save config
    config = {
        "class_labels": CLASS_LABELS,
        "n_classes": N_CLASSES,
        "bin_edges": [0, 1, 50, 100],
        "flattening": "raw_weekly_no_year",
        "n_features": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "excluded_features": ["year"],
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "hyperparameters": {
            "max_depth": 6, "learning_rate": 0.1, "n_estimators": 500,
            "subsample": 0.9, "colsample_bytree": 0.7, "min_child_weight": 3, "gamma": 0.1,
        },
    }
    with open(RESULTS_DIR / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save test predictions
    np.savez(
        RESULTS_DIR / "test_results.npz",
        y_test=y_test, y_pred=y_pred, y_proba=y_proba,
        meta_test=meta_test, feature_cols=np.array(FEATURE_COLS),
    )

    # Save report text
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write("FINAL 3-CLASS MODEL -- TEST SET\n")
        f.write("=" * 60 + "\n")
        f.write(report)
        f.write(f"\nMacro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n")

    print(f"\nModel saved to {RESULTS_DIR / 'model_3class.json'}")
    print(f"Config saved to {RESULTS_DIR / 'model_config.json'}")


if __name__ == "__main__":
    main()
