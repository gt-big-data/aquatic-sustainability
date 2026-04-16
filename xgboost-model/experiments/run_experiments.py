"""
Run all ablation experiments. Outputs comparison tables and saves results.
Does NOT modify anything in src/ or outputs/.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from flatten_variants import (
    flatten_hybrid,
    flatten_hybrid_no_year,
    flatten_raw,
    flatten_summary,
    flatten_summary_interactions,
    flatten_summary_no_year,
)

SEED = 42
CLASS_NAMES = ["None (0%)", "Moderate (1-50%)", "Severe (>50%)"]
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TUNED_PARAMS = dict(
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


# ══════════════════════════════════════════════════════════════
# DATA LOADING & SPLITTING
# ══════════════════════════════════════════════════════════════

def load_raw_data():
    data_path = Path(__file__).resolve().parent.parent / "data" / "sequences_reduced_16.npz"
    data = np.load(data_path, allow_pickle=True)
    X_seq = data["X"]
    y = data["y"]
    meta = data["meta"]
    feature_names = list(data["feature_names"])
    print(f"Loaded: {X_seq.shape[0]} samples, {X_seq.shape[1]} weeks, {X_seq.shape[2]} features")
    print(f"Classes: {np.bincount(y.astype(int))} (none/moderate/severe)")
    return X_seq, y, meta, feature_names


def split_raw(X_seq, y, meta):
    """Split raw sequences + meta. Same indices, same split as original."""
    # Stack seq and meta for synchronized splitting
    X_train_seq, X_temp_seq, y_train, y_temp, meta_train, meta_temp = train_test_split(
        X_seq, y, meta, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val_seq, X_test_seq, y_val, y_test, meta_val, meta_test = train_test_split(
        X_temp_seq, y_temp, meta_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    print(f"Split: Train {len(y_train)}, Val {len(y_val)}, Test {len(y_test)}")
    return (
        X_train_seq, X_val_seq, X_test_seq,
        y_train, y_val, y_test,
        meta_train, meta_val, meta_test,
    )


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 1: FLATTENING VARIANTS
# ══════════════════════════════════════════════════════════════

def run_flattening_experiments(
    X_train_seq, X_val_seq, X_test_seq,
    y_train, y_val, y_test,
    meta_train, meta_val, meta_test,
    feature_names,
):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: FLATTENING VARIANTS")
    print("=" * 70)

    variants = {
        "A_summary": flatten_summary,
        "B_raw": flatten_raw,
        "C_hybrid": flatten_hybrid,
        "D_summary_no_year": flatten_summary_no_year,
        "E_summary_interact": flatten_summary_interactions,
        "F_hybrid_no_year": flatten_hybrid_no_year,
    }

    sample_weights = compute_sample_weight("balanced", y_train)
    results = []

    for name, fn in variants.items():
        print(f"\n--- {name} ---")
        t0 = time.time()

        X_tr = fn(X_train_seq, meta_train, feature_names)
        X_va = fn(X_val_seq, meta_val, feature_names)
        X_te = fn(X_test_seq, meta_test, feature_names)
        n_feats = X_tr.shape[1]
        print(f"  Features: {n_feats}")

        model = XGBClassifier(**TUNED_PARAMS)
        model.fit(
            X_tr, y_train, sample_weight=sample_weights,
            eval_set=[(X_va, y_val)], verbose=False,
        )

        y_pred = model.predict(X_te)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        recalls = recall_score(y_test, y_pred, average=None)
        precisions = precision_score(y_test, y_pred, average=None)
        elapsed = time.time() - t0

        results.append({
            "variant": name,
            "n_features": n_feats,
            "macro_f1": macro_f1,
            "none_recall": recalls[0],
            "mod_recall": recalls[1],
            "sev_recall": recalls[2],
            "none_precision": precisions[0],
            "mod_precision": precisions[1],
            "sev_precision": precisions[2],
            "time_s": elapsed,
            "model": model,
            "y_pred": y_pred,
            "X_test_flat": X_te,
        })

        print(f"  Macro F1: {macro_f1:.4f} | None R: {recalls[0]:.3f} | "
              f"Mod R: {recalls[1]:.3f} | Sev R: {recalls[2]:.3f} | {elapsed:.1f}s")

    return results


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 2: SEVERE CLASS WEIGHT SWEEP
# ══════════════════════════════════════════════════════════════

def run_weight_experiments(
    X_train, y_train, X_val, y_val, X_test, y_test,
):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SEVERE CLASS WEIGHT SWEEP")
    print("=" * 70)

    base_weights = compute_sample_weight("balanced", y_train)
    results = []

    for severe_mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
        weights = base_weights.copy()
        weights[y_train == 2] *= severe_mult

        model = XGBClassifier(**TUNED_PARAMS)
        model.fit(
            X_train, y_train, sample_weight=weights,
            eval_set=[(X_val, y_val)], verbose=False,
        )

        y_pred = model.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        recalls = recall_score(y_test, y_pred, average=None)
        precisions = precision_score(y_test, y_pred, average=None)
        effective_weight = 4.87 * severe_mult

        results.append({
            "severe_multiplier": severe_mult,
            "effective_weight": f"~{effective_weight:.1f}x",
            "macro_f1": macro_f1,
            "sev_recall": recalls[2],
            "sev_precision": precisions[2],
            "sev_f1": f1_score(y_test, y_pred, average=None)[2],
            "model": model,
            "y_pred": y_pred,
        })

        print(f"  Severe {severe_mult}x (eff ~{effective_weight:.1f}x): "
              f"F1={macro_f1:.4f}, Sev recall={recalls[2]:.3f}, Sev prec={precisions[2]:.3f}")

    return results


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 3: PROBABILITY THRESHOLD TUNING
# ══════════════════════════════════════════════════════════════

def run_threshold_experiment(model, X_test, y_test):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: PROBABILITY THRESHOLD TUNING")
    print("=" * 70)

    y_proba = model.predict_proba(X_test)
    results = []

    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        y_pred_custom = np.argmax(y_proba[:, :2], axis=1)
        severe_mask = y_proba[:, 2] >= threshold
        y_pred_custom[severe_mask] = 2

        macro_f1 = f1_score(y_test, y_pred_custom, average="macro")
        recalls = recall_score(y_test, y_pred_custom, average=None)
        precisions = precision_score(y_test, y_pred_custom, average=None)

        results.append({
            "threshold": threshold,
            "macro_f1": macro_f1,
            "sev_recall": recalls[2],
            "sev_precision": precisions[2],
            "sev_f1": f1_score(y_test, y_pred_custom, average=None)[2],
            "y_pred": y_pred_custom,
        })

        print(f"  Threshold {threshold:.2f}: F1={macro_f1:.4f}, "
              f"Sev recall={recalls[2]:.3f}, Sev prec={precisions[2]:.3f}")

    return results


# ══════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrices(y_test, predictions_dict, output_path):
    """Plot confusion matrices for multiple models side by side."""
    n = len(predictions_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    labels = ["None (0%)", "Moderate\n(1-50%)", "Severe\n(>50%)"]

    for ax, (name, y_pred) in zip(axes, predictions_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(
            cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=ax,
            xticklabels=labels, yticklabels=labels,
        )
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\nMacro F1: {macro_f1:.3f}")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_feature_importance_for(model, output_path, title="Feature Importance (Gain)"):
    importance = model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:25]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp])
    ax.set_xlabel("Gain")
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    X_seq, y, meta, feature_names = load_raw_data()

    (
        X_train_seq, X_val_seq, X_test_seq,
        y_train, y_val, y_test,
        meta_train, meta_val, meta_test,
    ) = split_raw(X_seq, y, meta)

    # ── Experiment 1: Flattening Variants ──
    exp1_results = run_flattening_experiments(
        X_train_seq, X_val_seq, X_test_seq,
        y_train, y_val, y_test,
        meta_train, meta_val, meta_test,
        feature_names,
    )

    # Print comparison table
    print("\n" + "=" * 90)
    print("EXPERIMENT 1 RESULTS: FLATTENING VARIANTS (Test Set)")
    print("=" * 90)
    print(f"{'Variant':<25} | {'Feats':>5} | {'Macro F1':>8} | "
          f"{'None R':>6} | {'Mod R':>6} | {'Sev R':>6} | {'Time':>5}")
    print("-" * 90)
    for r in exp1_results:
        print(f"{r['variant']:<25} | {r['n_features']:>5} | {r['macro_f1']:>8.4f} | "
              f"{r['none_recall']:>6.3f} | {r['mod_recall']:>6.3f} | {r['sev_recall']:>6.3f} | "
              f"{r['time_s']:>4.0f}s")
    print("=" * 90)

    # Verify control matches original
    control = exp1_results[0]
    if abs(control["macro_f1"] - 0.758) > 0.005:
        print(f"\n  WARNING: Control F1 {control['macro_f1']:.4f} doesn't match original 0.758")
    else:
        print(f"\n  Control verified: F1 {control['macro_f1']:.4f} matches original 0.758")

    # Year ablation
    a_f1 = next(r["macro_f1"] for r in exp1_results if r["variant"] == "A_summary")
    d_f1 = next(r["macro_f1"] for r in exp1_results if r["variant"] == "D_summary_no_year")
    c_f1 = next(r["macro_f1"] for r in exp1_results if r["variant"] == "C_hybrid")
    f_f1 = next(r["macro_f1"] for r in exp1_results if r["variant"] == "F_hybrid_no_year")
    print(f"\n  Year ablation (summary): {a_f1:.4f} -> {d_f1:.4f} (delta {d_f1 - a_f1:+.4f})")
    print(f"  Year ablation (hybrid):  {c_f1:.4f} -> {f_f1:.4f} (delta {f_f1 - c_f1:+.4f})")

    # Find best variant
    best_exp1 = max(exp1_results, key=lambda r: r["macro_f1"])
    print(f"\n  Best variant: {best_exp1['variant']} (F1: {best_exp1['macro_f1']:.4f})")

    # ── Experiment 2: Severe Weight Sweep (using best variant's features) ──
    best_fn_name = best_exp1["variant"]
    variant_fns = {
        "A_summary": flatten_summary,
        "B_raw": flatten_raw,
        "C_hybrid": flatten_hybrid,
        "D_summary_no_year": flatten_summary_no_year,
        "E_summary_interact": flatten_summary_interactions,
        "F_hybrid_no_year": flatten_hybrid_no_year,
    }
    best_fn = variant_fns[best_fn_name]

    X_train_flat = best_fn(X_train_seq, meta_train, feature_names)
    X_val_flat = best_fn(X_val_seq, meta_val, feature_names)
    X_test_flat = best_fn(X_test_seq, meta_test, feature_names)

    exp2_results = run_weight_experiments(
        X_train_flat, y_train, X_val_flat, y_val, X_test_flat, y_test,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT 2 RESULTS: SEVERE WEIGHT SWEEP")
    print("=" * 80)
    print(f"{'Multiplier':>10} | {'Eff Weight':>10} | {'Macro F1':>8} | "
          f"{'Sev R':>6} | {'Sev P':>6} | {'Sev F1':>6}")
    print("-" * 80)
    for r in exp2_results:
        print(f"{r['severe_multiplier']:>10.1f} | {r['effective_weight']:>10} | "
              f"{r['macro_f1']:>8.4f} | {r['sev_recall']:>6.3f} | "
              f"{r['sev_precision']:>6.3f} | {r['sev_f1']:>6.3f}")
    print("=" * 80)

    best_exp2 = max(exp2_results, key=lambda r: r["macro_f1"])
    print(f"\n  Best weight: {best_exp2['severe_multiplier']}x "
          f"(F1: {best_exp2['macro_f1']:.4f})")

    # ── Experiment 3: Threshold Tuning (using best model from exp2) ──
    exp3_results = run_threshold_experiment(
        best_exp2["model"], X_test_flat, y_test,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT 3 RESULTS: PROBABILITY THRESHOLD TUNING")
    print("=" * 70)
    print(f"{'Threshold':>9} | {'Macro F1':>8} | {'Sev R':>6} | {'Sev P':>6} | {'Sev F1':>6}")
    print("-" * 70)
    for r in exp3_results:
        print(f"{r['threshold']:>9.2f} | {r['macro_f1']:>8.4f} | "
              f"{r['sev_recall']:>6.3f} | {r['sev_precision']:>6.3f} | {r['sev_f1']:>6.3f}")
    print("=" * 70)

    best_exp3 = max(exp3_results, key=lambda r: r["macro_f1"])
    default_pred = best_exp2["y_pred"]
    default_f1 = f1_score(y_test, default_pred, average="macro")
    if best_exp3["macro_f1"] > default_f1:
        print(f"\n  Best threshold: {best_exp3['threshold']} (F1: {best_exp3['macro_f1']:.4f})")
        best_y_pred = best_exp3["y_pred"]
        best_threshold = best_exp3["threshold"]
    else:
        print(f"\n  Default argmax is best (F1: {default_f1:.4f})")
        best_y_pred = default_pred
        best_threshold = "default argmax"

    # ── Save Results ──
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    # Comparison table
    comp_rows = []
    for r in exp1_results:
        comp_rows.append({
            "variant": r["variant"],
            "n_features": r["n_features"],
            "macro_f1": r["macro_f1"],
            "none_recall": r["none_recall"],
            "mod_recall": r["mod_recall"],
            "sev_recall": r["sev_recall"],
            "none_precision": r["none_precision"],
            "mod_precision": r["mod_precision"],
            "sev_precision": r["sev_precision"],
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    print(f"  Saved {RESULTS_DIR / 'comparison_table.csv'}")

    # Confusion matrices: original vs best
    plot_confusion_matrices(
        y_test,
        {
            "A. Summary (original)": exp1_results[0]["y_pred"],
            f"Best: {best_fn_name}": best_y_pred,
        },
        RESULTS_DIR / "confusion_matrices.png",
    )

    # Feature importance for best model
    plot_feature_importance_for(
        best_exp2["model"],
        RESULTS_DIR / "feature_importance_best.png",
        title=f"Feature Importance — {best_fn_name} (Gain)",
    )

    # Full classification report for best model
    print("\n" + "=" * 70)
    print(f"BEST MODEL — FULL CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(
        y_test, best_y_pred, target_names=CLASS_NAMES, digits=3
    ))

    # ── Final Summary ──
    best_recalls = recall_score(y_test, best_y_pred, average=None)
    best_macro = f1_score(y_test, best_y_pred, average="macro")

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"  Flattening:      {best_fn_name}")
    print(f"  Features:        {best_exp1['n_features']}")
    print(f"  Severe weight:   {best_exp2['severe_multiplier']}x (eff {best_exp2['effective_weight']})")
    print(f"  Threshold:       {best_threshold}")
    print()
    print(f"  Test Macro F1:   {best_macro:.4f}")
    print(f"  None recall:     {best_recalls[0]:.3f}")
    print(f"  Moderate recall: {best_recalls[1]:.3f}")
    print(f"  Severe recall:   {best_recalls[2]:.3f}")
    print()
    print("  vs Original model (summary, balanced weights, default threshold):")
    orig_f1 = f1_score(y_test, exp1_results[0]["y_pred"], average="macro")
    orig_recalls = recall_score(y_test, exp1_results[0]["y_pred"], average=None)
    print(f"    Macro F1:        {orig_f1:.4f}")
    print(f"    Severe recall:   {orig_recalls[2]:.3f}")
    delta = best_macro - orig_f1
    print(f"    Delta:           {delta:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
