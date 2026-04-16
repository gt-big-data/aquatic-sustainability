"""
Step 3: Run GBR inference on 133 reef cluster centroids (April 2026).

Loads the trained model from train_best.py and runs predictions on
live CRW satellite data. Produces predictions with both year=2026
(actual) and year=2020 (clamped) to quantify year extrapolation effect.
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

# ── paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
INFERENCE_NPZ = ROOT / "coral reef analysis" / "datasets" / "crw_gbr_sequences_reduced_16_latest_week_centroids.npz"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    # ── Load model & config ───────────────────────────────────
    config_path = RESULTS_DIR / "model_config.json"
    model_path = RESULTS_DIR / "best_model.json"

    if not config_path.exists() or not model_path.exists():
        print("ERROR: Run train_best.py first to produce best_model.json and model_config.json")
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    model = XGBClassifier()
    model.load_model(str(model_path))

    CLASS_LABELS = config["class_labels"]
    FEATURE_COLUMNS = config["feature_columns"]
    N_FEATURES = config["n_features"]

    print(f"Model: {config['scheme_name']} ({config['n_classes']} classes)")
    print(f"Features: {N_FEATURES} (raw weekly)")

    # ── Load inference data ───────────────────────────────────
    if not INFERENCE_NPZ.exists():
        print(f"ERROR: Inference data not found at {INFERENCE_NPZ}")
        sys.exit(1)

    inf_data = np.load(INFERENCE_NPZ, allow_pickle=True)
    X_seq = inf_data["X"]
    meta = inf_data["meta"]
    feature_names = list(inf_data["feature_names"])
    cluster_ids = inf_data["cluster_id"]
    centroid_lats = inf_data["centroid_lat"]
    centroid_lons = inf_data["centroid_lon"]
    match_distances = inf_data["centroid_match_distance_km"]
    cluster_sizes = inf_data["cluster_n_records"]

    print(f"\nInference data: {X_seq.shape[0]} reef clusters")
    print(f"Date: {inf_data['end_time'][0]}")
    print(f"NaN count: {np.isnan(X_seq).sum()}")

    # ── Flatten (must match training exactly) ─────────────────
    flat_features = {}
    for w in range(16):
        for i, fname in enumerate(feature_names):
            flat_features[f"{fname}_week{w:02d}"] = X_seq[:, w, i]

    flat_features["latitude"] = meta[:, 0]
    flat_features["longitude"] = meta[:, 1]
    flat_features["year"] = meta[:, 2]
    flat_features["month"] = meta[:, 3]

    X_inference = pd.DataFrame(flat_features).fillna(0)

    # Verify feature alignment
    if list(X_inference.columns) != FEATURE_COLUMNS:
        print("WARNING: Feature column mismatch!")
        print(f"  Expected: {FEATURE_COLUMNS[:5]}...")
        print(f"  Got:      {list(X_inference.columns)[:5]}...")
        sys.exit(1)

    print(f"Flattened: {X_inference.shape}")

    # ── Predict with actual year (2026) ───────────────────────
    y_pred = model.predict(X_inference)
    y_proba = model.predict_proba(X_inference)

    # ── Predict with clamped year (2020) ──────────────────────
    X_clamped = X_inference.copy()
    X_clamped["year"] = 2020.0
    y_pred_clamped = model.predict(X_clamped)
    y_proba_clamped = model.predict_proba(X_clamped)

    # ── Year extrapolation analysis ───────────────────────────
    year_shift = (y_pred != y_pred_clamped).sum()
    print(f"\nYear extrapolation effect:")
    print(f"  {year_shift} of {len(y_pred)} clusters changed class when year clamped to 2020")

    pred_dist_2026 = pd.Series([CLASS_LABELS[c] for c in y_pred]).value_counts()
    pred_dist_2020 = pd.Series([CLASS_LABELS[c] for c in y_pred_clamped]).value_counts()
    print(f"  Year=2026: {pred_dist_2026.to_dict()}")
    print(f"  Year=2020: {pred_dist_2020.to_dict()}")

    # ── Build results table ───────────────────────────────────
    results = pd.DataFrame({
        "cluster_id": cluster_ids,
        "centroid_lat": centroid_lats,
        "centroid_lon": centroid_lons,
        "n_historical_records": cluster_sizes,
        "match_distance_km": np.round(match_distances, 2),
        "predicted_class": y_pred,
        "predicted_label": [CLASS_LABELS[c] for c in y_pred],
        "predicted_class_clamped": y_pred_clamped,
        "predicted_label_clamped": [CLASS_LABELS[c] for c in y_pred_clamped],
    })

    # Per-class probabilities
    for i, label in enumerate(CLASS_LABELS):
        col_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct").replace(">", "gt").replace("-", "_")
        results[f"prob_{col_name}"] = np.round(y_proba[:, i], 4)

    # Key satellite features
    # Find index of TSA_DHW in feature_names
    dhw_idx = list(feature_names).index("TSA_DHW") if "TSA_DHW" in feature_names else 2
    sst_idx = list(feature_names).index("FilledSST") if "FilledSST" in feature_names else 0

    results["TSA_DHW_week15"] = X_seq[:, 15, dhw_idx]
    results["TSA_DHW_week00"] = X_seq[:, 0, dhw_idx]
    results["TSA_DHW_max"] = X_seq[:, :, dhw_idx].max(axis=1)
    results["FilledSST_last"] = X_seq[:, 15, sst_idx]

    # Sort by predicted severity
    results = results.sort_values("predicted_class", ascending=False)

    # ── Print summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GBR REEF CLUSTER PREDICTIONS — April 2026")
    print("=" * 60)

    print(f"\nPrediction distribution (year=2026):")
    for label in CLASS_LABELS:
        count = (results["predicted_label"] == label).sum()
        pct = count / len(results) * 100
        print(f"  {label}: {count} clusters ({pct:.1f}%)")

    print(f"\nPrediction distribution (year=2020, clamped):")
    for label in CLASS_LABELS:
        count = (results["predicted_label_clamped"] == label).sum()
        pct = count / len(results) * 100
        print(f"  {label}: {count} clusters ({pct:.1f}%)")

    # Find the severe/highest-risk probability column
    prob_cols = [c for c in results.columns if c.startswith("prob_")]
    severe_prob_col = prob_cols[-1]  # last class = most severe

    print(f"\nTop 10 highest risk clusters (by P({CLASS_LABELS[-1]})):")
    top_risk = results.nlargest(10, severe_prob_col)
    for _, row in top_risk.iterrows():
        print(f"  Cluster {int(row['cluster_id']):3d} | "
              f"lat={row['centroid_lat']:.2f}, lon={row['centroid_lon']:.2f} | "
              f"{row['predicted_label']:20s} | "
              f"DHW_max={row['TSA_DHW_max']:.1f} | "
              f"P(severe)={row[severe_prob_col]:.3f}")

    # ── Save ──────────────────────────────────────────────────
    results.to_csv(RESULTS_DIR / "gbr_predictions.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'gbr_predictions.csv'}")

    map_data = results[["cluster_id", "centroid_lat", "centroid_lon",
                         "predicted_label", "predicted_class",
                         "TSA_DHW_max", "n_historical_records"]].copy()
    map_data.to_csv(RESULTS_DIR / "gbr_risk_map_data.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'gbr_risk_map_data.csv'}")


if __name__ == "__main__":
    main()
