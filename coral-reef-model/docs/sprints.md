# Sprint Implementation Guide: GBR Coral Bleaching Prediction

**Total Sprints:** 4 (Sprint 4 optional)
**Sprint Length:** ~1 week each
**Tooling:** Python, XGBoost, scikit-learn, pandas, matplotlib/seaborn
**Compute:** Local machine
**IDE Agent:** Claude Code

---

## Sprint 0: Repository Setup & Data Audit

**Goal:** Get the repo structure in place, inventory the GCBD data, and identify what features are actually available before writing any model code.

**Duration:** Half a day.

### Tasks

#### 0.1 — Project Structure

```
gbr-bleaching/
├── data/
│   ├── raw/              # Original downloaded files (never modify)
│   │   ├── gcbd/         # BCO-DMO global_bleaching_environmental.csv
│   │   └── aims/         # AIMS LTMP manta-tow-by-reef.csv (validation only)
│   ├── processed/        # Cleaned, filtered outputs
│   └── README.md         # Data dictionary: source, date range, columns
├── src/
│   ├── ingest.py         # GCBD loading and GBR filtering
│   ├── preprocess.py     # NaN handling, pH transform, scaling
│   ├── features.py       # Feature selection and assembly
│   ├── train.py          # XGBoost training with spatial CV
│   ├── tune.py           # Hyperparameter search
│   ├── evaluate.py       # Test set evaluation and reporting
│   ├── inference.py      # Post-2020 NOAA CRW inference + AIMS validation
│   └── utils.py          # Shared helpers (plotting, IO, constants)
├── notebooks/            # EDA and one-off analysis
├── outputs/
│   ├── models/           # Saved model weights (.json)
│   ├── scalers/          # Fitted scaler objects (.pkl)
│   ├── figures/          # Plots and visualizations
│   └── reports/          # Evaluation markdown/CSVs
├── tests/
├── PRD.md
├── SPRINT_GUIDE.md
├── requirements.txt
└── .gitignore
```

#### 0.2 — Data Inventory

**GCBD (primary — training + test):**
Download from https://www.bco-dmo.org/dataset/773466. Place `global_bleaching_environmental.csv` in `data/raw/gcbd/`. Then audit:

- Total row count
- Row count after GBR bounding box filter (Lat ∈ [−25, −10], Lon ∈ [142, 154])
- Row count after dropping NaN `Bleaching_Percentage`
- Year range
- List ALL columns — especially the CoRTAD environmental covariates (SST mean, max, anomaly, TSA, TSA_DHW, etc.)
- NaN rate per column (GBR-filtered subset only)
- Check if a sector/region column already exists or needs to be derived from latitude
- Target distribution: histogram of `Bleaching_Percentage` — confirm zero-inflation

**AIMS LTMP reef summary (validation only):**
Place `manta-tow-by-reef.csv` in `data/raw/aims/`. Quick audit:

- Confirm 2,646 rows, 1993–2023, columns: MEAN_LIVE_CORAL, MEAN_DEAD_CORAL, LATITUDE, LONGITUDE, REPORT_YEAR, SECTOR
- Confirm NO bleaching column exists (this is why it's validation-only)
- Count post-2020 rows (expect ~411 across 2020–2023)

Print clean summary tables for both. Don't eyeball it — compute actual numbers.

#### 0.3 — Dependencies

```
# requirements.txt
pandas>=2.0
numpy>=1.24
xgboost>=2.0
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
shap>=0.43
joblib>=1.3
requests>=2.31
```

#### 0.4 — Write `data/README.md`

Document everything discovered above. Include:

- Source, file path, row counts, year range for each dataset
- Full column list for GCBD with NaN rates
- Feature availability matrix: which CoRTAD features are present, which are missing
- Note that AIMS data has no bleaching column — validation use only

### Exit Criteria

- [ ] Repo structure created
- [ ] GCBD downloaded, GBR-filtered row count confirmed
- [ ] All GCBD columns inventoried with NaN rates
- [ ] CoRTAD feature columns identified (SST, DHW, TSA, etc.)
- [ ] AIMS summary inspected and confirmed as validation-only
- [ ] `data/README.md` written with all findings
- [ ] `requirements.txt` installed and importable

---

## Sprint 1: Data Ingestion, Cleaning, and Feature Assembly

**Goal:** Go from raw GCBD CSV to a single clean `gbr_training_data.csv` with target + features, filtered to GBR, NaN-purged, and split by sector.

### Tasks

#### 1.1 — GBR Geographic Filter (`src/ingest.py`)

```python
GBR_BOUNDS = {
    'lat_min': -25.0,
    'lat_max': -10.0,
    'lon_min': 142.0,
    'lon_max': 154.0
}

def filter_gbr(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows within the GBR bounding box."""
    mask = (
        (df['Latitude'] >= GBR_BOUNDS['lat_min']) &
        (df['Latitude'] <= GBR_BOUNDS['lat_max']) &
        (df['Longitude'] >= GBR_BOUNDS['lon_min']) &
        (df['Longitude'] <= GBR_BOUNDS['lon_max'])
    )
    return df[mask].copy()
```

#### 1.2 — Target Variable Processing

- Load GCBD, apply `filter_gbr()`
- Drop rows where `Bleaching_Percentage` is NaN
- Scale target to [0.0, 1.0] if it's currently on a [0, 100] scale
- Log the before/after row counts

#### 1.3 — Sector Assignment

```python
def assign_sector(lat: float) -> str:
    if lat >= -16.0:
        return 'Northern'
    elif lat >= -20.0:
        return 'Central'
    else:
        return 'Southern'
```

Verify the distribution — if one sector has <50 surveys after filtering, spatial CV won't be meaningful and you'll need to fall back to a 2-fold split.

#### 1.4 — Feature Selection

Based on the Sprint 0 column inventory, select the predictor features. Expected CoRTAD columns in the BCO-DMO CSV include (exact names may vary — verify):

- `SST`, `SST_Max`, `SST_Mean`, `SST_Min`, `SST_Standard_Deviation`
- `TSA` (Thermal Stress Anomaly), `TSA_DHW` (Degree Heating Weeks)
- `TSA_DHWMax`, `TSA_DHWMean`, `TSA_Standard_Deviation`
- `SSTA` (SST Anomaly), `SSTA_Max`, `SSTA_Mean`, `SSTA_Standard_Deviation`

Plus site metadata: `Distance_to_Shore`, `Turbidity`, `Cyclone_Frequency`, `Depth_m`, `Exposure`

Drop any columns that are IDs, dates, or metadata not useful for prediction (`Site_Name`, `Country`, `Source`, etc.).

#### 1.5 — pH Transformation (if applicable)

```python
if 'pH' in df.columns:
    df['H_ion'] = 10 ** (-df['pH'])
    df = df.drop(columns=['pH'])
```

#### 1.6 — NaN Audit on Features

For each predictor column in the GBR-filtered dataset:

- If NaN rate < 5%: drop the affected rows
- If NaN rate 5–30%: flag it, investigate, decide column vs. row drop
- If NaN rate > 30%: drop the column entirely

Document every decision in `data/README.md`.

#### 1.7 — Save Processed Data

```python
df.to_csv('data/processed/gbr_training_data.csv', index=False)

print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
print(f"Target distribution:\n{df['Bleaching_Percentage'].describe()}")
print(f"Sector distribution:\n{df['Sector'].value_counts()}")
print(f"Year range: {df['Year'].min()} – {df['Year'].max()}")
print(f"Feature columns: {[c for c in df.columns if c not in ['Bleaching_Percentage', 'Sector', 'Year', 'Latitude', 'Longitude']]}")
```

### Exit Criteria

- [ ] `gbr_training_data.csv` exists with target + all usable features + sector column
- [ ] Zero NaN values in the final file
- [ ] pH converted to [H⁺] (if applicable)
- [ ] Sector distribution logged and viable for 3-fold spatial CV
- [ ] Feature columns documented in `data/README.md`

---

## Sprint 2: Baseline Model, Spatial CV, and Hyperparameter Tuning

**Goal:** Train a baseline XGBoost model, implement sector-based spatial cross-validation, run a hyperparameter search, and select the best configuration.

### Tasks

#### 2.1 — Temporal Train/Test Split (`src/preprocess.py`)

```python
TEMPORAL_CUTOFF = 2016

df = pd.read_csv('data/processed/gbr_training_data.csv')

train_df = df[df['Year'] < TEMPORAL_CUTOFF].copy()
test_df = df[df['Year'] >= TEMPORAL_CUTOFF].copy()

print(f"Train: {len(train_df)} rows ({train_df['Year'].min()}–{train_df['Year'].max()})")
print(f"Test:  {len(test_df)} rows ({test_df['Year'].min()}–{test_df['Year'].max()})")
```

**Sanity checks:**

- If test set has <30 rows, the cutoff is too aggressive — move it earlier.
- Verify the test set includes 2016 and/or 2017 mass bleaching event data.

#### 2.2 — Feature/Target Separation

```python
TARGET = 'Bleaching_Percentage'
GROUP_COL = 'Sector'
DROP_COLS = [TARGET, GROUP_COL, 'Year', 'Latitude', 'Longitude']
# Add any other ID/metadata columns discovered in Sprint 0

feature_cols = [c for c in train_df.columns if c not in DROP_COLS]
X_train = train_df[feature_cols]
y_train = train_df[TARGET]
groups = train_df[GROUP_COL]
```

#### 2.3 — Feature Scaling

```python
from sklearn.preprocessing import RobustScaler
import joblib

scaler = RobustScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=feature_cols,
    index=X_train.index
)
joblib.dump(scaler, 'outputs/scalers/robust_scaler.pkl')
```

#### 2.4 — Baseline Model

```python
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, cross_val_score

baseline = XGBRegressor(
    objective='reg:tweedie',
    tweedie_variance_power=1.5,
    tree_method='hist',
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

cv = GroupKFold(n_splits=3)
scores = cross_val_score(
    baseline, X_train_scaled, y_train,
    groups=groups, cv=cv,
    scoring='neg_mean_absolute_error'
)
print(f"Baseline CV MAE: {-scores.mean():.4f} ± {scores.std():.4f}")
```

#### 2.5 — Hyperparameter Search (`src/tune.py`)

```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'tweedie_variance_power': [1.2, 1.3, 1.4, 1.5, 1.6],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
    'n_estimators': [300, 500, 800],
    'min_child_weight': [3, 5, 10],
}

search = RandomizedSearchCV(
    XGBRegressor(objective='reg:tweedie', tree_method='hist', random_state=42),
    param_distributions=param_grid,
    n_iter=50,
    cv=GroupKFold(n_splits=3),
    scoring='neg_mean_absolute_error',
    random_state=42,
    verbose=1,
    n_jobs=-1,
    refit=True
)

search.fit(X_train_scaled, y_train, groups=groups)
print(f"Best MAE: {-search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")
```

#### 2.6 — Save CV Results

```python
cv_results = pd.DataFrame(search.cv_results_)
cv_results.to_csv('outputs/reports/cv_results.csv', index=False)

best_model = search.best_estimator_
best_model.save_model('outputs/models/best_model_cv.json')
```

### Exit Criteria

- [ ] Baseline model CV MAE established
- [ ] 50-combo hyperparameter search completed
- [ ] Best params identified and logged
- [ ] `cv_results.csv` saved
- [ ] Best model saved as `.json`
- [ ] Tuned model beats baseline by a meaningful margin

---

## Sprint 3: Final Evaluation, Post-2020 Inference, and Reporting

**Goal:** Evaluate the tuned model on the held-out temporal test set, run inference on post-2020 satellite data, cross-reference against AIMS coral cover, and produce the final evaluation report.

### Tasks

#### 3.1 — Test Set Evaluation (`src/evaluate.py`)

```python
import joblib, numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = XGBRegressor()
model.load_model('outputs/models/best_model_cv.json')
scaler = joblib.load('outputs/scalers/robust_scaler.pkl')

X_test = test_df[feature_cols]
y_test = test_df[TARGET]
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

y_pred = np.clip(model.predict(X_test_scaled), 0.0, 1.0)

y_test_100 = y_test * 100
y_pred_100 = y_pred * 100

mae = mean_absolute_error(y_test_100, y_pred_100)
rmse = np.sqrt(mean_squared_error(y_test_100, y_pred_100))
r2 = r2_score(y_test_100, y_pred_100)

print(f"Test MAE:  {mae:.2f}%")
print(f"Test RMSE: {rmse:.2f}%")
print(f"Test R²:   {r2:.4f}")
```

#### 3.2 — Naive Baseline Comparison

```python
naive_pred = np.full_like(y_test_100, y_train.mean() * 100)
naive_mae = mean_absolute_error(y_test_100, naive_pred)

print(f"Naive baseline MAE: {naive_mae:.2f}%")
print(f"Model improvement:  {naive_mae - mae:.2f}% absolute")
```

#### 3.3 — Feature Importance

```python
import matplotlib.pyplot as plt

importance = model.get_booster().get_score(importance_type='gain')
sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp])
ax.set_xlabel('Gain')
ax.set_title('Feature Importance (Gain)')
ax.invert_yaxis()
fig.tight_layout()
fig.savefig('outputs/figures/feature_importance.png', dpi=150)
```

Optional SHAP:

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, show=False)
plt.savefig('outputs/figures/shap_summary.png', dpi=150, bbox_inches='tight')
```

#### 3.4 — Residual Analysis

```python
residuals = y_pred_100 - y_test_100

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(residuals, bins=30, edgecolor='black')
axes[0].set_xlabel('Residual (%)')
axes[0].set_title('Residual Distribution')
axes[0].axvline(0, color='red', linestyle='--')

axes[1].scatter(y_test_100, y_pred_100, alpha=0.5, s=20)
axes[1].plot([0, 100], [0, 100], 'r--')
axes[1].set_xlabel('Actual Bleaching %')
axes[1].set_ylabel('Predicted Bleaching %')
axes[1].set_title('Predicted vs Actual')

for sector in test_df['Sector'].unique():
    mask = test_df['Sector'] == sector
    axes[2].scatter(y_test_100[mask], residuals[mask], label=sector, alpha=0.5, s=20)
axes[2].axhline(0, color='red', linestyle='--')
axes[2].set_xlabel('Actual Bleaching %')
axes[2].set_ylabel('Residual')
axes[2].set_title('Residuals by Sector')
axes[2].legend()

fig.tight_layout()
fig.savefig('outputs/figures/residual_analysis.png', dpi=150)
```

#### 3.5 — Post-2020 Satellite Inference (`src/inference.py`)

This is the "pseudo-live" test. Use NOAA CRW satellite data as input features for the trained model at GBR reef locations during 2020–2024 peak-stress periods.

```python
# Step 1: Build a roster of GBR reef coordinates
# Use the AIMS LTMP reef locations (from manta-tow-by-reef.csv)
aims = pd.read_csv('data/raw/aims/manta-tow-by-reef.csv')
reef_roster = aims[['REEF_NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()

# Step 2: For each reef, fetch NOAA CRW SST/DHW at peak-stress dates
# Target dates: Feb-Mar 2020, Feb-Mar 2022, Feb-Mar 2024
# Access: https://coralreefwatch.noaa.gov/product/vs/data.php
# Or use ERDDAP: https://coastwatch.pfeg.noaa.gov/erddap/

# Step 3: Construct feature vectors matching GCBD training columns
# Map CRW product columns to CoRTAD column names used in training
# This requires manual alignment — document the mapping

# Step 4: Run inference
# X_inference_scaled = scaler.transform(X_inference)
# predictions = model.predict(X_inference_scaled)
# predictions_pct = np.clip(predictions * 100, 0, 100)

# Step 5: Save predictions
# inference_df = reef_roster.copy()
# inference_df['predicted_bleaching_pct'] = predictions_pct
# inference_df.to_csv('outputs/reports/post2020_inference.csv', index=False)
```

**Key challenge:** The GCBD training features come from CoRTAD v6, while NOAA CRW provides a different (related) product suite. The feature definitions may not perfectly align. Document the mapping and any discrepancies in the evaluation report.

#### 3.6 — AIMS Coral Cover Cross-Validation

After generating post-2020 predictions, cross-reference against AIMS manta tow data:

```python
aims = pd.read_csv('data/raw/aims/manta-tow-by-reef.csv')

# For reefs where model predicted high bleaching (e.g., >30%),
# check: did MEAN_LIVE_CORAL drop in subsequent year?
# For reefs where model predicted low bleaching (<10%),
# check: did MEAN_LIVE_CORAL stay stable?

# This is qualitative — report as a table in evaluation_report.md
# Example: "Of 15 reefs where model predicted >30% bleaching in 2020,
#           12 showed a decline in live coral cover by 2021"
```

#### 3.7 — Evaluation Report (`outputs/reports/evaluation_report.md`)

Generate a markdown report containing:

1. **Model configuration** — best hyperparameters from CV
2. **Test set metrics** — MAE, RMSE, R², Tweedie deviance (on 2016–2020 GCBD data)
3. **Baseline comparison** — naive mean vs. tuned model
4. **Feature importance** — top features by gain, embedded plots
5. **Residual analysis** — bias patterns, sector-level performance
6. **Post-2020 inference results** — predicted bleaching map for 2020/2022/2024 peak-stress periods
7. **AIMS coral cover cross-validation** — consistency between predicted bleaching and observed coral cover changes
8. **Dataset summary** — train/test sizes, year ranges, sector distributions
9. **Limitations** — GCBD temporal coverage gap, out-of-distribution extrapolation risk for extreme DHW values, CoRTAD vs. CRW feature alignment
10. **Next steps** — whether Sprint 4 slope features are warranted, what AIMS full data access would unlock

### Exit Criteria

- [ ] Test set metrics computed and logged
- [ ] Model beats naive baseline
- [ ] Feature importance and SHAP plots generated
- [ ] Residual analysis plot generated
- [ ] Post-2020 inference predictions generated for at least one peak-stress period
- [ ] AIMS coral cover cross-validation completed (qualitative)
- [ ] `evaluation_report.md` written with all results
- [ ] Final model saved as `outputs/models/final_model.json`

---

## Sprint 4 (Optional): Trend Feature Enhancement

**Trigger:** Run this sprint if Sprint 3 evaluation shows the model struggles with trend-driven bleaching events — e.g., it underpredicts severe bleaching at sites where SST was rapidly rising.

**Goal:** Add 90-day linear trend (slope) features for SST and DHW, retrain, and compare.

### Prerequisites

Requires time-series data per survey site (daily or weekly readings for the 90 days preceding each survey). If your dataset only has point-in-time values matched to survey dates, this sprint isn't possible without going back to raw satellite APIs.

### Tasks

#### 4.1 — Slope Feature Computation

```python
from scipy.stats import linregress

def compute_slope(series: pd.Series) -> float:
    x = np.arange(len(series))
    slope, _, _, _, _ = linregress(x, series.values)
    return slope
```

#### 4.2 — Add to Feature Set and Retrain

Add `SST_slope_90d` and `DHW_slope_90d`, re-run the full pipeline (scaling, spatial CV, hyperparameter search), compare test set metrics.

#### 4.3 — Decide

If slope features improve MAE by >0.5% absolute on the test set, keep them. Otherwise the added complexity isn't worth it.

### Exit Criteria

- [ ] Slope features computed (if time-series data available)
- [ ] Model retrained with slope features
- [ ] Side-by-side comparison documented
- [ ] Final decision in evaluation report

---

## Quick Reference: Key Decisions

| Decision             | Choice                                                          | Rationale                                                                                        |
| -------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Training data        | BCO-DMO GCBD (1980–2020) only                                   | Only public dataset with continuous bleaching percentages + pre-matched environmental covariates |
| AIMS LTMP role       | Post-2020 validation only                                       | No bleaching column in public data; coral cover changes used as consistency check                |
| Post-2020 evaluation | NOAA CRW satellite inference + AIMS coral cover cross-reference | No ground-truth bleaching data available after 2020 without AIMS full access                     |
| Feature set          | CoRTAD v6 columns already in GCBD                               | SST/DHW metrics are dominant predictors; Copernicus chemistry is nice-to-have                    |
| Feature engineering  | Raw values only (no rolling windows)                            | Simpler, faster; slope as optional Sprint 4                                                      |
| NaN handling         | Zero-tolerance on target; pragmatic thresholds on features      | Preserves signal integrity                                                                       |
| Train/test split     | Temporal (pre-2016 / 2016–2020)                                 | Tests forward prediction into the accelerated bleaching era                                      |
| CV during training   | Sector-based GroupKFold (3 sectors)                             | Prevents spatial leakage                                                                         |
| Tuning method        | RandomizedSearchCV, 50 combos                                   | Feasible on local machine                                                                        |
| Scaling              | RobustScaler                                                    | Handles extreme outliers                                                                         |
| Objective            | Tweedie                                                         | Handles zero-inflated target distribution                                                        |
| pH representation    | [H⁺] = 10^(−pH) if pH available                                 | Linearizes log scale for tree splits                                                             |
