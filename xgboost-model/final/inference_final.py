"""
Run inference on 133 GBR reef cluster centroids (April 2026).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_PATH = ROOT / "coral reef analysis" / "datasets" / "crw_gbr_sequences_reduced_16_latest_week_centroids.npz"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    # Load model and config
    model = XGBClassifier()
    model.load_model(str(RESULTS_DIR / "model_3class.json"))

    with open(RESULTS_DIR / "model_config.json") as f:
        config = json.load(f)

    CLASS_LABELS = config["class_labels"]
    FEATURE_COLS = config["feature_columns"]

    # Load inference data
    inf_data = np.load(INFERENCE_PATH, allow_pickle=True)
    X_seq = inf_data["X"]
    meta = inf_data["meta"]
    feature_names = list(inf_data["feature_names"])

    print(f"Inference data: {X_seq.shape[0]} reef clusters")
    print(f"Date: {inf_data['end_time'][0]}")

    # Flatten raw weekly (must match training)
    flat = {}
    for w in range(16):
        for i, fname in enumerate(feature_names):
            flat[f"{fname}_week{w:02d}"] = X_seq[:, w, i]
    flat["latitude"] = meta[:, 0]
    flat["longitude"] = meta[:, 1]
    flat["year"] = meta[:, 2]
    flat["month"] = meta[:, 3]
    X_inference = pd.DataFrame(flat).fillna(0)

    assert list(X_inference.columns) == FEATURE_COLS, "Feature mismatch!"

    # Predict
    y_pred = model.predict(X_inference)
    y_proba = model.predict_proba(X_inference)

    # Build results table
    results = pd.DataFrame({
        "cluster_id": inf_data["cluster_id"],
        "centroid_lat": inf_data["centroid_lat"],
        "centroid_lon": inf_data["centroid_lon"],
        "n_records": inf_data["cluster_n_records"],
        "match_distance_km": np.round(inf_data["centroid_match_distance_km"], 2),
        "predicted_class": y_pred,
        "predicted_label": [CLASS_LABELS[c] for c in y_pred],
        "prob_none": np.round(y_proba[:, 0], 4),
        "prob_moderate": np.round(y_proba[:, 1], 4),
        "prob_severe": np.round(y_proba[:, 2], 4),
        "risk_score": np.round(1.0 - y_proba[:, 0], 4),
        "TSA_DHW_max": X_seq[:, :, 2].max(axis=1),
        "TSA_DHW_last": X_seq[:, 15, 2],
        "FilledSST_last": X_seq[:, 15, 0],
    })

    results = results.sort_values("risk_score", ascending=False)
    results.to_csv(RESULTS_DIR / "gbr_predictions.csv", index=False)

    # Print summary
    print(f"\nPrediction distribution:")
    for label in CLASS_LABELS:
        count = (results["predicted_label"] == label).sum()
        print(f"  {label}: {count} ({count / len(results) * 100:.1f}%)")

    print(f"\nTop 10 highest risk clusters:")
    for _, row in results.head(10).iterrows():
        print(f"  Cluster {int(row['cluster_id']):3d} | "
              f"lat={row['centroid_lat']:.2f}, lon={row['centroid_lon']:.2f} | "
              f"risk={row['risk_score']:.3f} | DHW_max={row['TSA_DHW_max']:.1f} | "
              f"{row['predicted_label']}")

    print(f"\nSaved {RESULTS_DIR / 'gbr_predictions.csv'}")


if __name__ == "__main__":
    main()
