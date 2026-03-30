# Sprint 2 Audit Results

**Date:** 2026-03-30
**Status:** Complete

## CV Strategy Deviation from PRD

The PRD specified **sector-based GroupKFold (3 folds, Northern/Central/Southern)**. This was designed for GBR-only training where adjacent reefs could leak across folds. With global training (28,552 rows from 89 countries), spatial leakage between GBR sectors is no longer a meaningful risk — GBR is only ~6% of the training data. Additionally, most training rows have `Sector = 'Non-GBR'`, making GroupKFold on Sector useless for the global set.

**Adopted:** Standard 5-fold KFold (`shuffle=True, random_state=42`) on the global training set. Sector-level performance is evaluated on the held-out GBR test set instead.

## Train/Test Split

| Set | Rows | Year Range | Source |
|-----|------|------------|--------|
| Train | 28,552 | 1983–2015 | Global (all rows with `Date_Year < 2016`) |
| Test | 182 | 2016–2017 | GBR only (`is_GBR == True` and `Date_Year >= 2016`) |
| Discarded at test time | 5,621 | 2016–2019 | Non-GBR rows from 2016+ (not part of evaluation) |

No data leakage: zero overlapping indices between train and test.

### Target Distribution Shift (Train vs Test)

| Stat | Train (n=28,552) | Test (n=182) |
|------|-------------------|--------------|
| Mean | 10.30% | 16.05% |
| % zeros | 45.7% | 5.5% |

The test set has **higher mean bleaching** and **far fewer zeros** than training data. This is expected — 2016–2017 were mass bleaching years on the GBR, whereas the global training set is dominated by low/zero bleaching observations. This distribution shift is the core challenge: the model must extrapolate from a predominantly low-bleaching training distribution to predict elevated bleaching during mass events.

## Feature Scaling

RobustScaler fit on training data only, saved to `outputs/scalers/robust_scaler.pkl`.

| Feature | Center (Median) | Scale (IQR) |
|---------|-----------------|-------------|
| ClimSST | 300.62 | 3.53 |
| Temperature_Kelvin | 301.62 | 2.48 |
| Temperature_Mean | 300.64 | 1.59 |
| Temperature_Minimum | 296.67 | 3.37 |
| Temperature_Maximum | 305.07 | 1.37 |

## Baseline Model

Default XGBoost configuration:

```
objective: reg:tweedie
tweedie_variance_power: 1.5
n_estimators: 300
max_depth: 5
learning_rate: 0.05
colsample_bytree: 0.7
tree_method: hist
random_state: 42
```

### Baseline Results

| Metric | Value |
|--------|-------|
| CV MAE (5-fold, global train) | 0.0656 +/- 0.0018 |
| GBR Test MAE | 14.43% |
| GBR Test RMSE | 25.19% |
| GBR Test R² | -0.2436 |
| Naive MAE (predict training mean) | 14.63% |

The baseline barely beats the naive predictor (14.43% vs 14.63% — a 0.20% improvement). The negative R² means the model explains less variance than a horizontal line at the test mean. This is a direct consequence of the train/test distribution shift: the model learned from a zero-inflated global distribution and struggles to predict elevated GBR bleaching.

## Hyperparameter Search

**Method:** RandomizedSearchCV, 50 iterations, 5-fold KFold, `neg_mean_absolute_error` scoring.

**Search space:**

| Parameter | Values |
|-----------|--------|
| tweedie_variance_power | 1.2, 1.3, 1.4, 1.5, 1.6 |
| max_depth | 3, 4, 5, 6, 7 |
| learning_rate | 0.01, 0.03, 0.05, 0.1 |
| subsample | 0.7, 0.8, 0.9 |
| colsample_bytree | 0.5, 0.6, 0.7, 0.8 |
| n_estimators | 300, 500, 800 |
| min_child_weight | 3, 5, 10 |

**Total fits:** 250 (50 combinations x 5 folds)

### Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| tweedie_variance_power | 1.5 |
| subsample | 0.8 |
| n_estimators | 800 |
| min_child_weight | 10 |
| max_depth | 7 |
| learning_rate | 0.1 |
| colsample_bytree | 0.6 |

Notable: the search selected a deeper model (`max_depth=7`) with more trees (`n_estimators=800`) and higher learning rate (`learning_rate=0.1`) than the baseline. The `min_child_weight=10` acts as regularization against the deep trees.

### Tuned Model Results

| Metric | Baseline | Tuned | Delta |
|--------|----------|-------|-------|
| CV MAE (global train) | 0.0656 | 0.0544 | -0.0112 (17% better) |
| GBR Test MAE | 14.43% | 15.52% | +1.09% (worse) |
| GBR Test RMSE | 25.19% | 26.48% | +1.29% (worse) |
| GBR Test R² | -0.2436 | -0.3741 | -0.1305 (worse) |

## CV vs Test Divergence — Analysis

The tuned model substantially improved CV MAE (17% better) but **degraded GBR test performance** by ~1%. This is the most important finding of Sprint 2.

**Root causes:**

1. **Distribution mismatch.** The CV optimizes on global data (48% zeros, mean=10.3%). The GBR test set is a different distribution entirely (5.5% zeros, mean=16.1%). A model that is better at predicting the global average is not necessarily better at predicting GBR mass bleaching events.

2. **Overfitting to global patterns.** The deeper, more complex tuned model (`max_depth=7, n_estimators=800`) fits global patterns more tightly, but those patterns don't transfer to the GBR 2016–2017 test regime. The simpler baseline (`max_depth=5, n_estimators=300`) generalizes slightly better to the out-of-distribution GBR test set.

3. **Small test set.** With only 182 rows, the 1.09% MAE difference is likely within noise. This is not a statistically significant degradation — but the direction is informative.

**Implication for Sprint 3:** The baseline model is the more honest benchmark for GBR test evaluation. The tuned model's CV improvement is real on global data but doesn't translate to the GBR test set. Sprint 3 should evaluate both and consider whether GBR-weighted CV scoring or a smaller search space (constraining depth) would help.

## Per-Sector GBR Test Breakdown

| Sector | n | Mean Actual | MAE | Notes |
|--------|---|-------------|-----|-------|
| Northern | 2 | 1.5% | 1.28% | Too few rows to interpret |
| Central | 96 | 23.2% | 23.66% | Highest bleaching, highest error |
| Southern | 84 | 8.3% | 6.56% | Moderate bleaching, reasonable error |

**Central sector** drives most of the test error. With mean actual bleaching of 23.2%, the model's MAE of 23.66% indicates it is systematically underpredicting — likely predicting values near 0 for many rows that actually experienced significant bleaching. This is consistent with the zero-inflated training distribution biasing predictions downward.

**Northern** has only 2 test rows — no meaningful conclusions.

**Southern** performs best (MAE=6.56% vs actual mean=8.3%), likely because Southern GBR experienced less extreme bleaching in 2016–2017 and falls closer to the training distribution.

## Artifacts Saved

| File | Size | Contents |
|------|------|----------|
| `outputs/models/best_model.json` | 4.9 MB | Tuned XGBoost model (800 trees, depth 7) |
| `outputs/models/baseline_model.json` | 1.1 MB | Baseline XGBoost model (300 trees, depth 5) |
| `outputs/scalers/robust_scaler.pkl` | 2.1 KB | RobustScaler fit on training data |
| `outputs/reports/cv_results.csv` | 50 rows | Full RandomizedSearchCV results |
| `outputs/reports/test_predictions.csv` | 182 rows | Per-row predictions with residuals |

## Code Files

| File | Role |
|------|------|
| `src/train.py` | `load_and_split()`, `scale_features()`, `train_baseline()` |
| `src/tune.py` | `run_hyperparameter_search()` with param grid |
| `run_sprint2.py` | End-to-end pipeline: split → scale → baseline → tune → evaluate → save |

Sprint 1 files (`src/ingest.py`, `src/features.py`, `src/preprocess.py`, `run_sprint1.py`) were not modified.

## Exit Criteria Checklist

- [x] Train/test split verified: 28,552 train (global pre-2016), 182 test (GBR 2016–2017)
- [x] RobustScaler fit on train, saved to `outputs/scalers/`
- [x] Baseline model CV MAE (0.0656) and GBR test MAE (14.43%) computed
- [x] Naive baseline MAE (14.63%) computed
- [x] 50-combo RandomizedSearchCV completed
- [x] Best hyperparameters identified and printed
- [x] Tuned model GBR test MAE (15.52%) computed — compared against baseline and naive
- [x] Per-sector MAE breakdown printed
- [x] `best_model.json` saved
- [x] `cv_results.csv` saved
- [x] `test_predictions.csv` saved with residuals
- [x] Summary report printed with all metrics

## Sprint 3 Prerequisites

1. Load tuned model from `outputs/models/best_model.json` and baseline from `outputs/models/baseline_model.json`
2. Load scaler from `outputs/scalers/robust_scaler.pkl`
3. Load test predictions from `outputs/reports/test_predictions.csv` for residual analysis
4. Feature importance (gain-based and SHAP) on the GBR test set
5. Residual plots: distribution, predicted-vs-actual, residuals-by-sector
6. Consider evaluating **both** baseline and tuned models in Sprint 3, given the CV/test divergence
7. Post-2020 inference using NOAA CRW satellite data at AIMS LTMP reef locations
8. AIMS coral cover cross-validation
