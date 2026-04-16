"""
Step 1: Analyze different bleaching severity bucket schemes.

Recovers continuous Percent_Bleaching from the GCBD CSV by matching
(lat, lon, year, month) keys to the NPZ meta array, then tests 5
bucket schemes with XGBoost using raw-weekly flattening (68 features).
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier

# ── paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]  # aquatic-sustainability/
TRAIN_NPZ = ROOT / "xgboost-model" / "data" / "sequences_reduced_16.npz"
GCBD_CSV = ROOT / "coral reef analysis" / "datasets" / "global_bleaching_environmental.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ── flatten: raw weekly (68 features, best from ablation) ─────
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


# ── recover continuous bleaching values ───────────────────────
def recover_continuous_bleaching(npz_path, gcbd_path):
    """Match NPZ samples to GCBD CSV to recover Percent_Bleaching."""
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

    # Build ordered lookup from GCBD (preserves row order from extract_sequence.py)
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

    matched = np.sum(~np.isnan(pct_recovered))
    print(f"Recovered continuous values: {matched} / {len(meta)} ({matched/len(meta)*100:.1f}%)")

    # Verify binning consistency with original labels
    mask = ~np.isnan(pct_recovered)
    y_check = pd.cut(
        pd.Series(pct_recovered[mask]).clip(0, 100),
        bins=[-1, 1, 50, 100], labels=[0, 1, 2], include_lowest=True,
    ).astype(int).values
    agreement = (y_check == y_orig[mask]).sum()
    print(f"Binning agreement: {agreement} / {mask.sum()} ({agreement/mask.sum()*100:.1f}%)")

    return X_seq, meta, y_orig, feature_names, pct_recovered


# ── bucket schemes ────────────────────────────────────────────
SCHEMES = {
    "3_original":    {"edges": [-1, 1, 50, 100],
                      "labels": ["None (0%)", "Moderate (1-50%)", "Severe (>50%)"]},
    "4_split_mod":   {"edges": [-1, 1, 25, 50, 100],
                      "labels": ["None (0%)", "Low (1-25%)", "High (26-50%)", "Severe (>50%)"]},
    "4_split_sev":   {"edges": [-1, 1, 50, 75, 100],
                      "labels": ["None (0%)", "Moderate (1-50%)", "High (51-75%)", "Extreme (>75%)"]},
    "5_granular":    {"edges": [-1, 1, 25, 50, 75, 100],
                      "labels": ["None (0%)", "Low (1-25%)", "Mod (26-50%)", "High (51-75%)", "Extreme (>75%)"]},
    "4_ecological":  {"edges": [-1, 5, 30, 100],
                      "labels": ["Healthy (0-5%)", "Stressed (6-30%)", "Severe (>30%)"]},
}


def bin_target(pct_values, edges):
    """Bin continuous Percent_Bleaching using pd.cut (same logic as extract_sequence.py)."""
    n_classes = len(edges) - 1
    labels = list(range(n_classes))
    binned = pd.cut(
        pd.Series(pct_values).clip(0, 100),
        bins=edges, labels=labels, include_lowest=True,
    ).astype(int).values
    return binned


def main():
    print("=" * 70)
    print("STEP 1: Recover continuous bleaching values")
    print("=" * 70)
    X_seq, meta, y_orig, feature_names, pct_recovered = recover_continuous_bleaching(
        TRAIN_NPZ, GCBD_CSV
    )

    # Keep only samples where we recovered the continuous value
    mask = ~np.isnan(pct_recovered)
    X_seq_m = X_seq[mask]
    meta_m = meta[mask]
    pct_m = pct_recovered[mask]
    print(f"\nUsing {mask.sum()} samples with recovered continuous values")

    print(f"\nContinuous bleaching distribution:")
    print(f"  0%:      {(pct_m == 0).sum()}")
    print(f"  0-1%:    {((pct_m > 0) & (pct_m <= 1)).sum()}")
    print(f"  1-5%:    {((pct_m > 1) & (pct_m <= 5)).sum()}")
    print(f"  5-25%:   {((pct_m > 5) & (pct_m <= 25)).sum()}")
    print(f"  25-50%:  {((pct_m > 25) & (pct_m <= 50)).sum()}")
    print(f"  50-75%:  {((pct_m > 50) & (pct_m <= 75)).sum()}")
    print(f"  75-100%: {((pct_m > 75) & (pct_m <= 100)).sum()}")

    print("\n" + "=" * 70)
    print("STEP 2: Test bucket schemes")
    print("=" * 70)

    # Fixed hyperparameters (tuned in ablation experiments)
    BASE_PARAMS = dict(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        max_depth=6, learning_rate=0.1, n_estimators=500,
        subsample=0.9, colsample_bytree=0.7, min_child_weight=3,
        gamma=0.1, random_state=SEED,
    )

    results = []

    for name, scheme in SCHEMES.items():
        edges = scheme["edges"]
        class_labels = scheme["labels"]
        n_classes = len(edges) - 1

        y_binned = bin_target(pct_m, edges)

        unique, counts = np.unique(y_binned, return_counts=True)
        min_pct = (counts / len(y_binned) * 100).min()

        print(f"\n{'─' * 50}")
        print(f"{name} ({n_classes} classes):")
        for cls, label, count in zip(unique, class_labels, counts):
            pct = count / len(y_binned) * 100
            print(f"  Class {cls} [{label}]: {count} ({pct:.1f}%)")

        if min_pct < 2.0:
            print(f"  WARNING: smallest class is {min_pct:.1f}% — may be too small for reliable training")

        # Split 70/15/15 stratified BEFORE flattening
        X_train_seq, X_temp_seq, y_train, y_temp, meta_train, meta_temp = train_test_split(
            X_seq_m, y_binned, meta_m, test_size=0.30, stratify=y_binned, random_state=SEED
        )
        X_val_seq, X_test_seq, y_val, y_test, meta_val, meta_test = train_test_split(
            X_temp_seq, y_temp, meta_temp, test_size=0.50, stratify=y_temp, random_state=SEED
        )

        # Flatten raw weekly
        X_train_flat = flatten_raw(X_train_seq, meta_train, feature_names)
        X_val_flat = flatten_raw(X_val_seq, meta_val, feature_names)
        X_test_flat = flatten_raw(X_test_seq, meta_test, feature_names)

        # Train
        weights = compute_sample_weight("balanced", y_train)

        params = {**BASE_PARAMS, "num_class": n_classes}
        model = XGBClassifier(**params, early_stopping_rounds=30)
        model.fit(
            X_train_flat, y_train, sample_weight=weights,
            eval_set=[(X_val_flat, y_val)], verbose=False,
        )

        y_pred = model.predict(X_test_flat)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        per_class_f1 = f1_score(y_test, y_pred, average=None)

        results.append({
            "scheme": name,
            "n_classes": n_classes,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "min_class_pct": min_pct,
            "per_class_f1": per_class_f1.tolist(),
            "class_labels": class_labels,
            "edges": edges,
        })

        print(f"  Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
        print(f"  Per-class F1: {[f'{f:.3f}' for f in per_class_f1]}")
        print(classification_report(y_test, y_pred, target_names=class_labels, digits=3))

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BUCKET SCHEME COMPARISON")
    print("=" * 70)
    for r in results:
        print(f"  {r['scheme']:20s} | {r['n_classes']} cls | Macro F1: {r['macro_f1']:.4f} | "
              f"Weighted F1: {r['weighted_f1']:.4f} | Min class: {r['min_class_pct']:.1f}%")

    # ── Decision: pick best ───────────────────────────────────
    baseline_f1 = next(r["macro_f1"] for r in results if r["scheme"] == "3_original")
    candidates = [
        r for r in results
        if r["min_class_pct"] >= 3.0 and r["macro_f1"] >= baseline_f1 - 0.02
    ]
    # Among candidates, prefer more classes, then higher macro F1
    candidates.sort(key=lambda r: (r["n_classes"], r["macro_f1"]), reverse=True)

    if candidates:
        best = candidates[0]
    else:
        # Fall back to 3-class baseline
        best = next(r for r in results if r["scheme"] == "3_original")
        print("\n  No scheme met all criteria. Falling back to 3-class baseline.")

    print(f"\n  WINNER: {best['scheme']} ({best['n_classes']} classes, macro F1 = {best['macro_f1']:.4f})")
    print(f"  Edges: {best['edges']}")
    print(f"  Labels: {best['class_labels']}")

    # ── Save ──────────────────────────────────────────────────
    import json

    comparison_df = pd.DataFrame([
        {
            "scheme": r["scheme"],
            "n_classes": r["n_classes"],
            "macro_f1": round(r["macro_f1"], 4),
            "weighted_f1": round(r["weighted_f1"], 4),
            "min_class_pct": round(r["min_class_pct"], 1),
            "per_class_f1": str(r["per_class_f1"]),
        }
        for r in results
    ])
    comparison_df.to_csv(RESULTS_DIR / "bucket_comparison.csv", index=False)
    print(f"\n  Saved {RESULTS_DIR / 'bucket_comparison.csv'}")

    # Save winner config for train_best.py
    winner_config = {
        "scheme": best["scheme"],
        "edges": best["edges"],
        "class_labels": best["class_labels"],
        "n_classes": best["n_classes"],
        "macro_f1": best["macro_f1"],
        "baseline_3class_f1": baseline_f1,
    }
    with open(RESULTS_DIR / "winner_config.json", "w") as f:
        json.dump(winner_config, f, indent=2)
    print(f"  Saved {RESULTS_DIR / 'winner_config.json'}")


if __name__ == "__main__":
    main()
