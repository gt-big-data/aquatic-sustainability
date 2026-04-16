"""
XGBoost training pipeline for coral bleaching classification.

Usage:
    python src/train.py                    # baseline only
    python src/train.py --tune             # baseline + hyperparameter search
    python src/train.py --tune --n-iter 100  # more search iterations
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from flatten import load_and_flatten

SEED = 42
CLASS_NAMES = ["None (0%)", "Moderate (1-50%)", "Severe (>50%)"]
OUTPUT_DIR = Path("outputs")


def split_data(X, y):
    """70/15/15 stratified split matching William's LSTM."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    print(f"Train: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
    print(f"Val:   {len(X_val)} ({len(X_val)/len(X)*100:.0f}%)")
    print(f"Test:  {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")

    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = np.bincount(labels.astype(int))
        pcts = counts / len(labels) * 100
        print(f"  {name} classes: {counts} ({np.array2string(pcts, precision=1)}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_baseline(X_train, y_train, X_val, y_val, sample_weights):
    """Train baseline XGBoost with sensible defaults."""
    print("\n" + "=" * 60)
    print("BASELINE MODEL")
    print("=" * 60)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        colsample_bytree=0.8,
        subsample=0.8,
        min_child_weight=5,
        random_state=SEED,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_val_pred = model.predict(X_val)

    print("\n--- Validation Set ---")
    print(classification_report(y_val, y_val_pred, target_names=CLASS_NAMES))
    val_f1 = f1_score(y_val, y_val_pred, average="macro")
    print(f"Macro F1: {val_f1:.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_val, y_val_pred)}")

    return model, val_f1


def tune_model(X_train, y_train, sample_weights, n_iter=50):
    """RandomizedSearchCV over hyperparameters, scoring by f1_macro."""
    print("\n" + "=" * 60)
    print(f"HYPERPARAMETER SEARCH ({n_iter} iterations)")
    print("=" * 60)

    param_grid = {
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [300, 500, 800],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "min_child_weight": [3, 5, 10],
        "gamma": [0, 0.1, 0.3],
    }

    search = RandomizedSearchCV(
        XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=SEED,
        ),
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=5,
        scoring="f1_macro",
        random_state=SEED,
        verbose=1,
        n_jobs=-1,
        refit=True,
    )

    t0 = time.time()
    search.fit(X_train, y_train, sample_weight=sample_weights)
    elapsed = time.time() - t0

    print(f"\nSearch completed in {elapsed:.0f}s")
    print(f"Best macro F1 (CV): {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    return search.best_estimator_, search


def evaluate_final(model, X_test, y_test):
    """Final evaluation on held-out test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)

    print(
        classification_report(
            y_test, y_pred, target_names=CLASS_NAMES, digits=3
        )
    )

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Pred None  Pred Mod  Pred Sev")
    print(f"  Act None    {cm[0,0]:>8}  {cm[0,1]:>8}  {cm[0,2]:>8}")
    print(f"  Act Mod     {cm[1,0]:>8}  {cm[1,1]:>8}  {cm[1,2]:>8}")
    print(f"  Act Sev     {cm[2,0]:>8}  {cm[2,1]:>8}  {cm[2,2]:>8}")

    for i, name in enumerate(["None", "Moderate", "Severe"]):
        class_mask = y_test == i
        class_acc = (y_pred[class_mask] == i).mean()
        print(f"  {name} recall: {class_acc:.3f} ({class_mask.sum()} samples)")

    return y_pred, y_proba, macro_f1


def save_artifacts(model, search, y_test, y_pred, y_proba, baseline_val_f1, tuned_val_f1, test_f1, best_params):
    """Save model, results, and CV results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(OUTPUT_DIR / "model.json"))
    print(f"\nModel saved to {OUTPUT_DIR / 'model.json'}")

    np.savez(
        OUTPUT_DIR / "results.npz",
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )
    print(f"Results saved to {OUTPUT_DIR / 'results.npz'}")

    if search is not None:
        pd.DataFrame(search.cv_results_).to_csv(
            OUTPUT_DIR / "cv_results.csv", index=False
        )
        print(f"CV results saved to {OUTPUT_DIR / 'cv_results.csv'}")

    # Feature importance
    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame(
        sorted(importance.items(), key=lambda x: x[1], reverse=True),
        columns=["feature", "gain"],
    )
    imp_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    print(f"Feature importance saved to {OUTPUT_DIR / 'feature_importance.csv'}")

    summary = {
        "baseline_val_macro_f1": baseline_val_f1,
        "tuned_val_macro_f1": tuned_val_f1,
        "test_macro_f1": test_f1,
        "best_params": best_params,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {OUTPUT_DIR / 'summary.json'}")


def print_summary(baseline_val_f1, tuned_val_f1, test_f1, best_params):
    print("\n" + "=" * 60)
    print("XGBoost Coral Bleaching Classification -- Final Results")
    print("=" * 60)
    print("Dataset:    sequences_reduced_16.npz (28,539 samples)")
    print("Features:   36 (4 satellite x 8 stats + 4 metadata)")
    print("Classes:    None (58.2%) / Moderate (35.0%) / Severe (6.8%)")
    print("Split:      70/15/15 stratified (seed=42)")
    print()
    print(f"Baseline Val Macro F1:  {baseline_val_f1:.4f}")
    if tuned_val_f1 is not None:
        print(f"Tuned CV Macro F1:      {tuned_val_f1:.4f}")
    print(f"Test Macro F1:          {test_f1:.4f}")
    print()
    if best_params:
        print("Best Hyperparameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    print()
    print("Comparison with William's models:")
    print("  LSTM:    [fill in from William's results]")
    print("  MLP:     [fill in from William's results]")
    print(f"  XGBoost: {test_f1:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost coral bleaching classifier")
    parser.add_argument("--data-path", default="data/sequences_reduced_16.npz")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search")
    parser.add_argument("--n-iter", type=int, default=50, help="RandomizedSearchCV iterations")
    args = parser.parse_args()

    # Step 1: Load and flatten
    print("=" * 60)
    print("LOADING AND FLATTENING DATA")
    print("=" * 60)
    X_flat, y, meta = load_and_flatten(args.data_path)

    # Step 2: Split
    print("\n" + "=" * 60)
    print("TRAIN/VAL/TEST SPLIT")
    print("=" * 60)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_flat, y)

    # Step 3: Class weights
    sample_weights = compute_sample_weight("balanced", y_train)
    classes, counts = np.unique(y_train, return_counts=True)
    class_weights = {int(c): len(y_train) / (len(classes) * count) for c, count in zip(classes, counts)}
    print(f"\nClass weights: {class_weights}")

    # Step 4: Baseline
    baseline_model, baseline_val_f1 = train_baseline(
        X_train, y_train, X_val, y_val, sample_weights
    )

    # Step 5: Tune (optional)
    search = None
    tuned_val_f1 = None
    best_params = None

    if args.tune:
        tuned_model, search = tune_model(X_train, y_train, sample_weights, n_iter=args.n_iter)

        # Evaluate tuned model on val set
        y_val_pred = tuned_model.predict(X_val)
        tuned_val_f1 = f1_score(y_val, y_val_pred, average="macro")
        print(f"\nTuned model val macro F1: {tuned_val_f1:.4f}")
        best_params = search.best_params_

        # Use tuned model for final eval if it's better
        if tuned_val_f1 > baseline_val_f1:
            print("Tuned model is better — using it for final evaluation.")
            final_model = tuned_model
        else:
            print("Baseline is better — using it for final evaluation.")
            final_model = baseline_model
    else:
        final_model = baseline_model

    # Step 6: Final evaluation on test set
    y_pred, y_proba, test_f1 = evaluate_final(final_model, X_test, y_test)

    # Step 7: Save
    save_artifacts(
        final_model, search, y_test, y_pred, y_proba,
        baseline_val_f1, tuned_val_f1, test_f1, best_params
    )

    # Summary
    print_summary(baseline_val_f1, tuned_val_f1, test_f1, best_params)


if __name__ == "__main__":
    main()
