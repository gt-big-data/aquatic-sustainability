# Final 3-Class XGBoost Model + GBR Visualization

## Model

3-class coral bleaching classifier using raw weekly flattening (67 features) with tuned hyperparameters from ablation experiments.

- **Bins:** 0% / 1-50% / >50% (None / Moderate / Severe)
- **Flattening:** Raw weekly (16 weeks x 4 features + 3 metadata = 67, excluding `year`)
- **Split:** 70/15/15 stratified (seed=42, matches William's LSTM)
- **Class weights:** Balanced via `compute_sample_weight`

### Test Set Results

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| None (0%) | 0.907 | 0.900 | 0.903 | 2,491 |
| Moderate (1-50%) | 0.783 | 0.745 | 0.764 | 1,497 |
| Severe (>50%) | 0.551 | 0.720 | 0.624 | 293 |
| **Macro avg** | 0.747 | 0.788 | **0.764** | 4,281 |

### Hyperparameters

max_depth=6, learning_rate=0.1, n_estimators=500, subsample=0.9, colsample_bytree=0.7, min_child_weight=3, gamma=0.1

## GBR Inference (April 2026)

Ran on 133 reef cluster centroids from CRW satellite data (week ending 2026-04-12).

| Prediction | Count | % |
|------------|-------|---|
| None (0%) | 58 | 43.6% |
| Moderate (1-50%) | 75 | 56.4% |
| Severe (>50%) | 0 | 0.0% |

Highest risk cluster: Cluster 2 (lat=-10.22, lon=150.85) with DHW_max=10.5 and risk score 0.834.

## Visualization Maps

Three maps in `results/`:

1. **gbr_centroid_map.png** — Point-level predictions at 133 reef cluster centroids
2. **gbr_grid_heatmap.png** — Dense grid (0.05 deg resolution, 34K valid points) with discrete class predictions via nearest-centroid interpolation
3. **gbr_risk_score_map.png** — Continuous risk score (P(moderate) + P(severe)) showing gradient of bleaching risk

## Files

| File | Description |
|------|-------------|
| `train_final.py` | Train model, save artifacts |
| `evaluate_final.py` | Confusion matrix, feature importance, SHAP plots |
| `inference_final.py` | GBR centroid inference |
| `visualize_gbr.py` | 3 GBR maps |
| `results/model_3class.json` | Trained XGBoost model |
| `results/model_config.json` | Feature columns, class labels, hyperparams |
| `results/test_results.npz` | Test set predictions/probabilities |
| `results/classification_report.txt` | Full classification report |
| `results/confusion_matrix.png` | Counts + row-normalized % |
| `results/feature_importance.png` | Top 20 features by gain |
| `results/shap_summary.png` | SHAP all-class summary |
| `results/shap_severe.png` | SHAP severe class |
| `results/gbr_predictions.csv` | 133 cluster predictions with probabilities |
| `results/gbr_centroid_map.png` | Point-level map |
| `results/gbr_grid_heatmap.png` | Grid heatmap |
| `results/gbr_risk_score_map.png` | Risk score map |
