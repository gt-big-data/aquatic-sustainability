"""
Run inference on 133 GBR reef cluster centroids (April 2026).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ALL_WEEKS_NAME = "crw_gbr_sequences_reduced_16_centroids_all_weeks.npz"


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_feature_idx(feature_names, candidates):
    norm_names = [_normalize_name(n) for n in feature_names]
    for cand in candidates:
        cand_norm = _normalize_name(cand)
        if cand_norm in norm_names:
            return norm_names.index(cand_norm)
    raise ValueError(
        f"Could not find any of {candidates} in feature_names={list(feature_names)}"
    )


def _resolve_all_weeks_path() -> Path:
    candidates = [
        RESULTS_DIR / ALL_WEEKS_NAME,
        Path(__file__).resolve().parents[1] / "experiments" / "results" / ALL_WEEKS_NAME,
        ROOT / "coral reef analysis" / "datasets" / ALL_WEEKS_NAME,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {ALL_WEEKS_NAME} in any expected location: "
        + ", ".join(str(p) for p in candidates)
    )


def _latest_week_indices(inf_data):
    end_times = inf_data["end_time"]
    latest_date = np.max(end_times)
    latest_mask = end_times == latest_date
    latest_idx = np.where(latest_mask)[0]
    if len(latest_idx) == 0:
        raise ValueError("No rows found for latest end_time in all-weeks dataset.")
    # Deterministic ordering for downstream CSV output
    cluster_ids = inf_data["cluster_id"][latest_idx]
    latest_idx = latest_idx[np.argsort(cluster_ids)]
    return latest_date, latest_idx


def main():
    # Load model and config
    model = XGBClassifier()
    model.load_model(str(RESULTS_DIR / "model_3class.json"))

    with open(RESULTS_DIR / "model_config.json") as f:
        config = json.load(f)

    CLASS_LABELS = config["class_labels"]
    FEATURE_COLS = config["feature_columns"]
    if "year" in FEATURE_COLS:
        raise ValueError(
            "Loaded model_config still includes 'year' as a feature. "
            "Retrain with final/train_final.py to produce year-free artifacts."
        )

    # Load all-weeks data, then select latest weekly snapshot
    all_weeks_path = _resolve_all_weeks_path()
    inf_data = np.load(all_weeks_path, allow_pickle=True)
    latest_date, latest_idx = _latest_week_indices(inf_data)
    X_seq = inf_data["X"][latest_idx]
    meta = inf_data["meta"][latest_idx]
    feature_names = list(inf_data["feature_names"])
    dhw_idx = _find_feature_idx(feature_names, ["tsa_dhw", "TSA_DHW"])
    sst_idx = _find_feature_idx(feature_names, ["filled_sst", "FilledSST"])

    print(f"Inference data: {X_seq.shape[0]} reef clusters")
    print(f"Date: {pd.Timestamp(latest_date).strftime('%Y-%m-%d')}")
    print(f"Source: {all_weeks_path}")

    # Flatten raw weekly (must match training)
    flat = {}
    for w in range(16):
        for i, fname in enumerate(feature_names):
            flat[f"{fname}_week{w:02d}"] = X_seq[:, w, i]
    flat["latitude"] = meta[:, 0]
    flat["longitude"] = meta[:, 1]
    flat["month"] = meta[:, 3]
    X_inference = pd.DataFrame(flat).fillna(0)

    assert list(X_inference.columns) == FEATURE_COLS, "Feature mismatch!"

    # Predict
    y_pred = model.predict(X_inference)
    y_proba = model.predict_proba(X_inference)

    # Build results table
    results = pd.DataFrame({
        "cluster_id": inf_data["cluster_id"][latest_idx],
        "centroid_lat": inf_data["centroid_lat"][latest_idx],
        "centroid_lon": inf_data["centroid_lon"][latest_idx],
        "n_records": inf_data["cluster_n_records"][latest_idx],
        "match_distance_km": np.round(inf_data["centroid_match_distance_km"][latest_idx], 2),
        "predicted_class": y_pred,
        "predicted_label": [CLASS_LABELS[c] for c in y_pred],
        "prob_none": np.round(y_proba[:, 0], 4),
        "prob_moderate": np.round(y_proba[:, 1], 4),
        "prob_severe": np.round(y_proba[:, 2], 4),
        "risk_score": np.round(1.0 - y_proba[:, 0], 4),
        "TSA_DHW_max": X_seq[:, :, dhw_idx].max(axis=1),
        "TSA_DHW_last": X_seq[:, 15, dhw_idx],
        "FilledSST_last": X_seq[:, 15, sst_idx],
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
