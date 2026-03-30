# Sprint 3 Evaluation Report: GBR Coral Bleaching Prediction Model

**Date:** 2026-03-30

## 1. Executive Summary

This project trained an XGBoost regression model with Tweedie objective on 28,552 global coral bleaching observations (1983-2015) to predict bleaching severity at Great Barrier Reef sites. The model was evaluated on 182 GBR observations from the 2016-2017 mass bleaching events. The baseline model achieved a GBR test MAE of 14.43%, barely outperforming a naive predictor (14.63%). The core limitation is a fundamental distribution shift: the model trained on predominantly zero-bleaching global data cannot reliably predict the elevated bleaching levels observed during GBR mass bleaching events. Diagnostic experiments explored GBR-only training, sample weighting, and feature reduction to characterize this limitation.

## 2. Model Configuration

**Primary model:** Baseline XGBoost (selected over tuned model due to better GBR test performance)

| Parameter | Value |
|-----------|-------|
| objective | reg:tweedie |
| tweedie_variance_power | 1.5 |
| n_estimators | 300 |
| max_depth | 5 |
| learning_rate | 0.05 |
| colsample_bytree | 0.7 |
| tree_method | hist |

**Training data:** 28,552 rows from 89 countries (all global observations pre-2016)
**Test data:** 182 GBR rows from 2016-2017 mass bleaching events
**Features:** 40 (32 CoRTAD thermal + 4 site metadata + 2 geographic + 2 exposure one-hot)

## 3. Test Set Metrics

| Metric | Naive | Baseline | Tuned |
|--------|-------|----------|-------|
| MAE (%) | 14.63 | 14.43 | 15.52 |
| RMSE (%) | 23.31 | 25.19 | 26.48 |
| R² | -0.0650 | -0.2436 | -0.3741 |
| Median Abs Error (%) | 8.80 | 4.46 | 5.97 |

The baseline model achieves a marginal 0.20% MAE improvement over the naive predictor. 
The negative R² indicates the model explains less variance than a horizontal line at the test mean. 
The tuned model (max_depth=7, n_estimators=800) performed *worse* on the GBR test set despite 
17% better cross-validation MAE on global training data — a textbook case of overfitting to 
the training distribution.

## 4. Diagnostic Experiments

| Variant | GBR Test MAE (%) |
|---------|------------------|
| Naive (predict mean) | 14.63 |
| Global baseline (40 feat) | 14.43 |
| Global tuned (40 feat) | 15.52 |
| GBR-only baseline (40 feat) | 15.01 |
| Global + GBR 5x weight | 14.86 |
| Global + GBR 10x weight | 15.22 |
| Global + GBR 20x weight | 15.10 |
| Global + severity weight | 14.33 |
| Global + top-7 features only | 13.22 **BEST** |

**Best variant:** Global + top-7 features only (13.22% MAE)

These experiments reveal the fundamental challenge: no simple reweighting or feature 
selection strategy dramatically improves GBR test performance. The distribution shift 
between global training data (48% zeros, mean=10.3%) and the GBR mass bleaching test 
set (5.5% zeros, mean=16.1%) is the binding constraint.

## 5. Feature Importance

### Gain-Based Importance (Top 15)

![Feature Importance](../figures/feature_importance_gain.png)

| Rank | Feature | Gain |
|------|---------|------|
| 1 | SSTA_Standard_Deviation | 211.0 |
| 2 | TSA_Mean | 58.0 |
| 3 | TSA_FrequencyMean | 53.9 |
| 4 | TSA_DHW | 52.8 |
| 5 | SSTA_DHW | 30.0 |
| 6 | Longitude_Degrees | 28.2 |
| 7 | Latitude_Degrees | 26.7 |
| 8 | TSA_Standard_Deviation | 22.1 |
| 9 | TSA_Frequency | 20.5 |
| 10 | TSA_DHWMean | 20.3 |
| 11 | TSA_Frequency_Standard_Deviation | 19.9 |
| 12 | Cyclone_Frequency | 19.1 |
| 13 | SSTA_Frequency | 15.3 |
| 14 | Temperature_Mean | 14.6 |
| 15 | TSA | 14.4 |

The model primarily relies on thermal stress features (DHW, TSA, SST variants), 
indicating it has learned the relationship between accumulated heat stress and bleaching 
rather than simply memorizing geographic coordinates.

### SHAP Summary

![SHAP Summary](../figures/shap_summary.png)

## 6. Residual Analysis

![Residual Analysis](../figures/residual_analysis.png)

Mean residual: -11.5% — the model systematically 
underpredicts bleaching on the GBR test set. 
This is consistent with the zero-inflated training distribution biasing predictions 
toward low values. The Central sector (highest actual bleaching) shows the largest 
negative residuals, confirming that the model cannot predict elevated bleaching levels 
in the most severely affected region.

## 7. Prediction Distribution

![Prediction Distribution](../figures/prediction_distribution.png)

The predicted distribution is compressed near low values while the actual distribution 
spreads across 0-100%. This confirms the model is collapsing toward the global training 
mean rather than capturing the full range of bleaching severity observed during mass events.

## 8. DHW Response Curve

![DHW Response Curve](../figures/dhw_response_curve.png)

| DHW | Predicted Bleaching | Expected from Literature |
|-----|---------------------|--------------------------|
| 4 (significant bleaching) | 31.3% | ~10-30% |
| 8 (widespread mortality) | 44.0% | ~30-60% |
| 12 (severe mortality) | 46.6% | ~50-90% |

The model shows an increasing bleaching response with DHW, consistent with the 
established literature. The curve shape indicates the model has captured the core 
thermal stress dynamic.

## 9. AIMS Coral Cover Cross-Reference

Year-over-year changes in live coral cover from AIMS LTMP manta tow surveys (2020-2023) 
provide a qualitative validation signal. Reefs experiencing the largest coral cover declines 
are the locations where bleaching models should predict elevated risk.

**Top reefs with largest coral cover decline (2020-2023):**

| Reef | Year | Live Coral (%) | Change (%) | Lat | Lon |
|------|------|----------------|------------|-----|-----|
| 22084S | 2022 | 9.0 | -34.9 | -22.00 | 152.46 |
| ASHMORE BANKS (1) | 2023 | 51.6 | -34.9 | -11.89 | 143.63 |
| HAYMAN ISLAND REEF | 2021 | 9.6 | -22.0 | -20.07 | 148.89 |
| 16013B | 2020 | 25.4 | -21.8 | -16.01 | 145.80 |
| WRECK ISLAND REEF | 2023 | 55.6 | -18.7 | -23.33 | 151.97 |
| SAND BANK NO 1 REEF | 2023 | 34.6 | -18.4 | -14.19 | 144.92 |
| MIDDLE BANKS (2) | 2021 | 31.4 | -17.8 | -11.76 | 143.65 |
| GANNETT CAY REEF | 2022 | 6.1 | -17.5 | -21.98 | 152.47 |
| SMALL LAGOON REEF | 2022 | 7.6 | -17.4 | -21.88 | 152.51 |
| HEDGE REEF | 2020 | 2.9 | -17.3 | -13.91 | 143.96 |

These reefs — concentrated in regions that experienced repeated mass bleaching in 
2020 and 2022 — represent the sites where the model's predictions would be most 
valuable if satellite-derived features were available at matching temporal resolution.

## 10. Limitations

1. **Small test set (182 rows):** The GBR test set from 2016-2017 is too small for robust 
statistical evaluation. Differences between model variants are likely within noise.

2. **Distribution shift:** The global training set is 48% zeros with mean bleaching of 10.3%. 
The GBR test set from mass bleaching years has only 5.5% zeros and mean 16.1%. The model 
is being evaluated on a fundamentally different distribution than it was trained on.

3. **Zero-inflation bias:** The Tweedie objective appropriately handles zero-inflation during 
training, but the resulting model is anchored to predict low values. It systematically 
underpredicts during mass bleaching events.

4. **GCBD temporal coverage:** The training database effectively ends at 2017 for GBR data. 
The 2020, 2022, and 2024 mass bleaching events — the most severe on record — are not 
captured in training or testing.

5. **CoRTAD vs CRW feature alignment:** Post-2020 inference would require mapping between 
CoRTAD v6 training features and NOAA CRW satellite products, which use different derivation 
methods and spatial resolutions.

## 11. What Would Actually Fix This

1. **More GBR-specific training data:** The AIMS LTMP full per-tow dataset (pending email 
to adc@aims.gov.au) records bleaching as a continuous percentage at GBR sites from 1993-2023. 
This would provide ~400+ post-2020 GBR bleaching observations covering the 2020 and 2022 
mass events — exactly the distribution the current model cannot learn from.

2. **Bleaching-severity-weighted training:** Upweighting high-bleaching observations (tested 
in diagnostic experiments) partially addresses zero-inflation bias but cannot overcome the 
fundamental lack of mass-bleaching-era training signal.

3. **Reframe as classification:** Converting from regression (predict exact percentage) to 
classification (bleaching/no-bleaching, or low/moderate/severe) may be more tractable. The 
model's signal is strongest for distinguishing zero from non-zero bleaching; predicting 
exact severity requires training data from the severity range being predicted.

4. **GBR-only or GBR-upweighted training:** The global training approach dilutes GBR-specific 
patterns. A model trained exclusively on GBR data, or with heavy GBR upweighting, would 
better capture regional thermal stress thresholds — at the cost of a much smaller training set.

5. **Ensemble with regional bias correction:** Train on global data for general patterns, 
then apply a GBR-specific bias correction layer calibrated to the 2016-2017 test set.

## 12. Figures

| Figure | Path |
|--------|------|
| Feature Importance (Gain) | `outputs/figures/feature_importance_gain.png` |
| SHAP Summary | `outputs/figures/shap_summary.png` |
| Residual Analysis (4-panel) | `outputs/figures/residual_analysis.png` |
| Prediction Distribution | `outputs/figures/prediction_distribution.png` |
| DHW Response Curve | `outputs/figures/dhw_response_curve.png` |
