# PRD: Great Barrier Reef Coral Bleaching Prediction Model

**Author:** Vaibhav
**Last Updated:** March 30, 2026
**Status:** Active Development

---

## 1. Problem Statement

Mass coral bleaching events on the Great Barrier Reef (GBR) are increasing in frequency and severity. Current monitoring relies on field surveys that are expensive, slow, and geographically sparse. This project builds a machine learning model that predicts bleaching severity at GBR reef sites using satellite-derived environmental data, enabling faster risk assessment without waiting for divers to confirm damage.

## 2. Objective

Train an XGBoost regression model to predict `Bleaching_Percentage` (0–100% scale) at GBR reef sites using historical environmental features. The model trains on the BCO-DMO Global Coral Bleaching Database (1980–2020) and is evaluated both on a held-out temporal slice of that database and through satellite-driven inference on the post-2020 period, cross-referenced against AIMS coral cover monitoring data.

**Success criteria:**

- Mean Absolute Error (MAE) ≤ 10% on the held-out temporal test set
- Model generalizes across GBR sectors (Northern, Central, Southern) without memorizing spatial coordinates
- Tweedie deviance on test set improves over a naive baseline (predicting the training mean)
- Post-2020 inference predictions qualitatively align with documented bleaching events (2020, 2022, 2024)

## 3. Scope

### In Scope

- GBR-only data (no global reefs)
- Historical data ingestion, cleaning, and filtering from BCO-DMO GCBD
- XGBoost model training with Tweedie objective
- Hyperparameter tuning via RandomizedSearchCV (local machine)
- Sector-based spatial cross-validation during training
- Temporal train/test split for final evaluation (within GCBD)
- Post-2020 satellite inference using NOAA CRW data, validated against AIMS coral cover trends
- Local Python scripts — no deployment, no dashboard, no API

### Out of Scope

- Live/daily inference pipeline (Phase 5 of original doc)
- Streamlit or web dashboard
- Distributed training (Dask, cluster compute)
- Rolling window feature engineering (using raw feature values only)
- Data imputation of any kind
- AIMS LTMP as a training target source (no bleaching column in public data)

## 4. Data Architecture

### 4.1 Target Variable

| Attribute             | Detail                                                                            |
| --------------------- | --------------------------------------------------------------------------------- |
| **Source**            | BCO-DMO Global Coral Bleaching Database (GCBD)                                    |
| **File**              | `global_bleaching_environmental.csv` (16 MB, CC-BY 4.0)                           |
| **Download**          | https://www.bco-dmo.org/dataset/773466                                            |
| **Column**            | `Bleaching_Percentage` — percentage of coral colonies showing bleaching at a site |
| **Scale**             | 0.0–1.0 (decimal), converted from 0–100 raw percentage                            |
| **Coverage**          | 1980–2020, ~34,846 records globally                                               |
| **Geographic Filter** | GBR bounding box only: Lat ∈ [−25.0°, −10.0°], Lon ∈ [142.0°, 154.0°]             |
| **NaN Policy**        | Drop any row missing `Bleaching_Percentage`. No imputation.                       |

The BCO-DMO version is preferred over the figshare version because it's a direct CSV download with no authentication, has an ERDDAP API endpoint for programmatic queries, and includes pre-matched CoRTAD v6 environmental covariates.

**What the GCBD does NOT cover:** The database ends at 2020. It misses the 2022, 2024, and 2025 GBR mass bleaching events — the most severe on record (2024 saw DHW of 12–15.5°C-weeks in the southern GBR, far exceeding anything in the training distribution). This is an inherent limitation: the model has never seen conditions resembling the current bleaching regime, and XGBoost cannot extrapolate beyond its training target range. We address this through post-2020 satellite inference (Section 4.5), not by supplementing the training target.

### 4.2 Predictor Variables

The model uses raw feature values from the dataset — no rolling window aggregations. Each row in the training data corresponds to a single survey observation at a specific site and date.

**Core environmental features (pre-matched in GCBD via CoRTAD v6):**

The BCO-DMO GCBD comes with a suite of SST-derived metrics already matched to each survey row. Inspect the CSV during Sprint 0 to identify the exact column names, but expect variables including SST mean, SST max, SST anomaly, Thermal Stress Anomaly (TSA), TSA DHW (Degree Heating Weeks), SST standard deviation, and related thermal metrics. These are derived from the Coral Reef Temperature Anomaly Database (CoRTAD) at ~4km resolution.

**Additional pre-matched site metadata in GCBD:**

| Column              | Use                                        |
| ------------------- | ------------------------------------------ |
| `Distance_to_Shore` | Nearshore vs. offshore exposure            |
| `Turbidity`         | Light attenuation / sediment stress        |
| `Cyclone_Frequency` | Historical cyclone exposure                |
| `Depth_m`           | Survey depth — shallow reefs bleach faster |
| `Exposure`          | Site exposure to open ocean                |

**Features NOT in the GCBD that would require separate sourcing:**

| Variable         | Source            | Status                                        |
| ---------------- | ----------------- | --------------------------------------------- |
| Ocean pH         | Copernicus Marine | Not in GCBD; add only if feasible in Sprint 1 |
| Dissolved Oxygen | Copernicus Marine | Not in GCBD; add only if feasible in Sprint 1 |
| Chlorophyll-a    | Copernicus Marine | Not in GCBD; add only if feasible in Sprint 1 |
| Wind Speed       | NOAA / ERA5       | Not in GCBD; add only if feasible in Sprint 1 |

**Decision:** Start with the features already in the GCBD (SST metrics, DHW, site metadata). These are the strongest predictors in the literature. Add Copernicus chemistry features only if Sprint 0 confirms API access is ready and Sprint 1 has time — they're nice-to-have, not essential.

**pH transformation (if pH is added):** Convert to raw hydrogen ion concentration ([H⁺] = 10^(−pH)) before modeling. This linearizes the logarithmic pH scale for tree-based splits.

### 4.3 NaN Policy

**Zero-tolerance on the target:** Any survey row missing `Bleaching_Percentage` is dropped. No exceptions.

**Pragmatic on predictors:** If a feature column has >10% NaN across the GBR-filtered dataset, investigate whether it's systemic. If so, either drop the column entirely or drop the affected rows. Do not interpolate or forward-fill training data.

**Column-level threshold:** If >30% of GBR-filtered rows are missing a metadata column (e.g., `Depth_m`), drop that column entirely rather than imputing.

### 4.4 Train/Test Split

**Strategy: Temporal split within the GCBD.**

- **Training set:** All surveys before 2016
- **Test set:** All surveys from 2016–2020 (captures the 2016, 2017, and 2020 mass bleaching events)

This tests the model's ability to predict into the accelerated bleaching era using only pre-2016 training data. The 2016–2017 back-to-back events were unprecedented at the time; if the model can predict their severity from environmental features, it has learned real thermal stress dynamics.

**Alternative three-way split** (if the dataset is large enough after GBR filtering): train (pre-2014), validation (2014–2016), test (2017–2020).

> **Important:** The temporal split is for final model evaluation. During training, use sector-based spatial cross-validation (see Section 5.2) for hyperparameter tuning on the training set only.

### 4.5 Post-2020 Satellite Inference and Validation

The model cannot be formally evaluated on 2022–2024 bleaching events (no ground-truth bleaching percentages available in public datasets). Instead, we run inference using satellite-derived features and validate qualitatively.

**Inference inputs:** NOAA Coral Reef Watch provides daily 5km-resolution SST and DHW at GBR virtual station points, freely downloadable for 2020–2025. Extract the same feature set used during training (SST metrics, DHW) for a roster of GBR reef coordinates at peak-stress dates (e.g., February–March 2024). Access: https://coralreefwatch.noaa.gov/product/vs/data.php

**Validation source:** AIMS LTMP manta tow reef summary (`manta-tow-by-reef.csv`, 2,646 rows, 1993–2023). This dataset has no bleaching column, but it records `MEAN_LIVE_CORAL` and `MEAN_DEAD_CORAL` as continuous percentages per reef per year. After the model predicts bleaching severity at these reef locations, cross-reference whether reefs flagged as high-bleaching-risk subsequently showed drops in live coral cover or spikes in dead coral. This is a consistency check, not a formal evaluation metric.

**AIMS full data access (pending):** An email has been sent to adc@aims.gov.au requesting the full per-tow manta tow data and fixed site survey data (which records bleaching as a continuous percentage). If granted, this can be incorporated as proper test-set ground truth for 2021+. The pipeline is designed so AIMS data can be slotted in without restructuring.

## 5. Model Architecture

### 5.1 Algorithm: XGBoost with Tweedie Objective

**Why Tweedie:** The target distribution is zero-inflated — many GBR sites record 0% bleaching in any given survey. Standard MSE loss treats the mass of zeros and the spread of positive values with equal weight, which biases the model toward predicting moderate values everywhere. The Tweedie distribution naturally handles this spike-at-zero shape.

**Configuration:**

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    objective='reg:tweedie',
    tweedie_variance_power=1.5,   # tuned via CV; search [1.2, 1.6]
    tree_method='hist',           # histogram-based binning, fast on local machine
    colsample_bytree=0.7,         # prevents over-reliance on DHW
    n_estimators=500,             # with early stopping
    early_stopping_rounds=30,
    eval_metric='tweedie-nll@1.5',
    random_state=42
)
```

### 5.2 Validation: Sector-Based Spatial Cross-Validation

Random K-Fold will leak spatial information — adjacent reefs share nearly identical environmental conditions.

**Approach:** Use `GroupKFold` with GBR sectors as the grouping variable.

| Sector   | Approximate Latitude Range | Characteristics                              |
| -------- | -------------------------- | -------------------------------------------- |
| Northern | −10° to −16°               | Warmest, most exposed to Coral Sea heating   |
| Central  | −16° to −20°               | Moderate, includes Townsville/Cairns systems |
| Southern | −20° to −25°               | Cooler, influenced by southern ocean mixing  |

### 5.3 Feature Scaling

**RobustScaler** for all predictor features. Median/IQR-based scaling is resistant to extreme outliers without compressing the normal distribution. Note: XGBoost is tree-based and technically invariant to monotonic feature scaling. The scaler primarily aids interpretability and downstream model comparison.

### 5.4 Hyperparameter Tuning

**Method:** `sklearn.model_selection.RandomizedSearchCV` with sector-based `GroupKFold`.

| Parameter                | Values                    | Rationale                                     |
| ------------------------ | ------------------------- | --------------------------------------------- |
| `tweedie_variance_power` | [1.2, 1.3, 1.4, 1.5, 1.6] | Controls zero-inflation penalty shape         |
| `max_depth`              | [3, 4, 5, 6, 7]           | Shallow trees prevent spatial memorization    |
| `learning_rate`          | [0.01, 0.03, 0.05, 0.1]   | Slower = smoother convergence on Tweedie loss |
| `subsample`              | [0.7, 0.8, 0.9]           | Row-level stochasticity per boosting round    |
| `colsample_bytree`       | [0.5, 0.6, 0.7, 0.8]      | Forces learning from non-DHW features         |
| `n_estimators`           | [300, 500, 800]           | Paired with early stopping (30 rounds)        |
| `min_child_weight`       | [3, 5, 10]                | Regularization against small leaf nodes       |

**Search budget:** 50 random combinations × 3-fold spatial CV = 150 model fits.

### 5.5 Evaluation Metrics

| Metric                   | Purpose                                                         |
| ------------------------ | --------------------------------------------------------------- |
| **Tweedie Deviance**     | Primary optimization target; handles zero-inflated distribution |
| **MAE (on 0–100 scale)** | Interpretable error — "the model is off by X% on average"       |
| **RMSE**                 | Penalizes large errors; catches catastrophic mispredictions     |
| **R²**                   | Variance explained; context for signal-to-noise ratio           |

## 6. Recommended Enhancement: 90-Day Trend Slope

**Verdict: Yes, include it — but as a separate, clearly scoped step.**

The 90-day linear trend (slope) captures momentum — whether conditions are getting worse or improving leading up to the survey. If your dataset has time-series data per site, compute the slope of a simple linear regression over the 90-day window for SST and DHW.

**Implementation:** Optional Sprint 4 enhancement. If Sprint 3 evaluation shows the model underfitting or missing trend-driven bleaching events, add the slope features and re-run.

## 7. Output Artifacts

| Artifact                 | Format              | Description                                                                  |
| ------------------------ | ------------------- | ---------------------------------------------------------------------------- |
| `model.json`             | XGBoost native JSON | Frozen model weights                                                         |
| `scaler.pkl`             | Pickle              | Fitted RobustScaler                                                          |
| `cv_results.csv`         | CSV                 | Full cross-validation results                                                |
| `evaluation_report.md`   | Markdown            | Test set metrics, feature importance, residual analysis, post-2020 inference |
| `feature_importance.png` | PNG                 | SHAP or native feature importance visualization                              |
| `post2020_inference.csv` | CSV                 | Model predictions at GBR reef sites for 2020–2024 using NOAA CRW features    |

## 8. Assumptions and Risks

| Assumption                                                                    | Risk if Wrong                                                           | Mitigation                                      |
| ----------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------- |
| GCBD has sufficient GBR survey density after NaN filtering                    | Dataset may be too small for robust training                            | Relax NaN policy on low-importance predictors   |
| CoRTAD v6 SST/DHW features in the GCBD are sufficient predictors              | Chemistry features (pH, oxygen) may carry marginal signal               | Add Copernicus features in Sprint 1 if feasible |
| DHW is available as a precomputed column in the BCO-DMO CSV                   | Column names may differ from expectations                               | Audit in Sprint 0                               |
| Temporal split (pre-2016 / 2016–2020) doesn't create fatal distribution shift | 2016–2017 events were unprecedented; model may underpredict at extremes | Document as informative failure mode            |
| NOAA CRW virtual station features are compatible with GCBD training features  | Feature definitions may differ between CoRTAD and CRW products          | Align feature definitions during Sprint 3       |
| Local machine can handle 150 model fits                                       | GBR-only dataset should be small enough                                 | Reduce to 30 combos                             |
| AIMS may grant full data access                                               | May not respond                                                         | Pipeline works without AIMS; it's additive      |

## 9. Non-Goals (Explicit)

- This is not a forecasting system. It predicts bleaching given known environmental conditions, not future conditions.
- This does not replace field surveys. It complements them by identifying where to prioritize monitoring.
- This does not model reef recovery, only bleaching severity.
- No live data pipelines, APIs, or dashboards in this phase.
- The AIMS LTMP reef summary is not used as a training target (no bleaching column). It is used only for post-2020 consistency validation.
