# Sprint 1 Audit Results

**Date:** 2026-03-30
**Status:** Complete

## Architecture Change

Training on the **full global GCBD dataset** (not GBR-only). The model learns universal thermal-stress → bleaching relationships from reefs worldwide, with lat/lon as features for regional differentiation. Test evaluation remains GBR-only (2016–2017).

## Pipeline Summary

| Step | Action | Rows |
|------|--------|------|
| Raw load | `global_bleaching_environmental.csv` | 41,361 |
| Coerce + drop invalid target | `pd.to_numeric(Percent_Bleaching, errors='coerce')`, drop NaN | 34,515 |
| Drop high-NaN columns | `Depth_m`, `Percent_Cover` | — |
| Coerce all numeric features | `pd.to_numeric(errors='coerce')` on 38 columns | — |
| One-hot encode Exposure | Sheltered (baseline), Exposed, Sometimes | — |
| Drop rows with NaN features | 160 rows had NaN in CoRTAD/Windspeed/etc. | 34,355 |
| Scale target | `/100.0`, clip to [0, 1] | — |
| Tag regions | `is_GBR`, `Sector` | — |

**Output:** `data/processed/global_training_data.csv` — 34,355 rows, 40 features, zero NaN.

## Processed Dataset Stats

| Attribute | Value |
|-----------|-------|
| Total rows | 34,355 |
| GBR rows | 1,949 |
| Non-GBR rows | 32,406 |
| Feature columns | 40 |
| Metadata columns | 3 (`Date_Year`, `is_GBR`, `Sector`) |
| Year range | 1983–2019 |
| Countries | 89 |
| Ecoregions | 114 |

## Feature Columns (40)

| Group | Columns | Count |
|-------|---------|-------|
| CoRTAD v6 thermal | ClimSST, Temperature_Kelvin/Mean/Min/Max/StdDev, SSTA/StdDev/Mean/Min/Max, SSTA_Frequency/StdDev/Max/Mean, SSTA_DHW/StdDev/Max/Mean, TSA/StdDev/Min/Max/Mean, TSA_Frequency/StdDev/Max/Mean, TSA_DHW/StdDev/Max/Mean | 32 |
| Site metadata | Distance_to_Shore, Turbidity, Cyclone_Frequency, Windspeed | 4 |
| Geographic (features) | Latitude_Degrees, Longitude_Degrees | 2 |
| Exposure (one-hot) | Exposure_Exposed, Exposure_Sometimes | 2 |

**Lat/lon are model features**, not just metadata. `is_GBR` and `Sector` are metadata only (for splitting/eval).

## Column Drop Decisions

| Column | GBR NaN % | Global NaN % | Decision |
|--------|-----------|--------------|----------|
| `Depth_m` | 40.4% | 4.9% | DROP — 40.4% NaN in GBR (test region) makes it unreliable |
| `Percent_Cover` | 44.8% | 34.3% | DROP — exceeds 30% threshold both globally and in GBR |

Note: `Depth_m` is only 4.9% NaN globally but 40.4% in GBR. Since GBR is our test region and the model must generalize there, keeping it would create a feature that's missing for ~40% of GBR test rows.

## NaN Rates Before Row Drop (Global, n=34,515)

All CoRTAD columns: 0.3–0.4% NaN (95–142 rows each, overlapping)
- `Distance_to_Shore`: 2 rows (0.0%)
- `Turbidity`: 6 rows (0.0%)
- `Windspeed`: 111 rows (0.3%)
- `Exposure`: 0 NaN

Total rows dropped for feature NaN: 160 (0.5%) — acceptable.

## Target Distribution

| Stat | Global (n=34,355) | GBR (n=1,949) |
|------|--------------------|----------------|
| Mean | 0.0964 | 0.2004 |
| Median | 0.0025 | 0.0550 |
| Std | 0.2021 | 0.2833 |
| Min | 0.0 | 0.0 |
| Max | 1.0 | 1.0 |
| % zeros | 48.1% | 14.2% |
| % > 0.5 | 6.9% | 19.1% |

Global data is heavily zero-inflated (48.1% zeros) — confirms Tweedie objective is appropriate. GBR has higher mean bleaching than global average.

## GBR Sector Distribution

| Sector | Lat Range | Rows | Mean Bleaching |
|--------|-----------|------|----------------|
| Northern | >= -16.0° | 291 | 0.1644 |
| Central | -16.0° to -20.0° | 1,095 | 0.1633 |
| Southern | < -20.0° | 563 | 0.2911 |

All sectors > 50 rows — viable for GroupKFold. Southern has highest mean bleaching, likely driven by heavy sampling during 1998/2002 mass events.

## Mean Bleaching by Year (GBR)

| Year | Mean | n | | Year | Mean | n |
|------|------|---|-|------|------|---|
| 1992 | 0.055 | 1 | | 2007 | 0.008 | 55 |
| 1996 | 0.518 | 6 | | 2008 | 0.006 | 60 |
| **1998** | **0.428** | **433** | | 2009 | 0.035 | 96 |
| 1999 | 0.118 | 4 | | 2010 | 0.024 | 79 |
| 2000 | 0.305 | 1 | | 2011 | 0.062 | 40 |
| **2002** | **0.366** | **381** | | 2012 | 0.023 | 25 |
| 2003 | 0.008 | 49 | | 2013 | 0.035 | 65 |
| 2004 | 0.027 | 55 | | 2014 | 0.043 | 64 |
| 2005 | 0.010 | 123 | | 2015 | 0.091 | 62 |
| 2006 | 0.058 | 168 | | **2016** | **0.164** | **82** |
| | | | | **2017** | **0.158** | **100** |

1998 and 2002 are the dominant mass bleaching years in the data. 2016–2017 are elevated relative to background but lower than 1998/2002 in this dataset. The temporal test split (2016+) captures 182 GBR rows.

## Top-10 Feature Correlations with Target

### Global (n=34,355)

| Rank | Feature | |Pearson r| |
|------|---------|------------|
| 1 | SSTA_DHW | 0.272 |
| 2 | TSA_DHW | 0.272 |
| 3 | SSTA_Frequency | 0.189 |
| 4 | TSA_Frequency | 0.162 |
| 5 | Longitude_Degrees | 0.145 |
| 6 | TSA | 0.143 |
| 7 | SSTA_Standard_Deviation | 0.137 |
| 8 | SSTA | 0.119 |
| 9 | Exposure_Sometimes | 0.109 |
| 10 | Temperature_Kelvin | 0.108 |

### GBR Only (n=1,949)

| Rank | Feature | |Pearson r| |
|------|---------|------------|
| 1 | SSTA_DHW | 0.332 |
| 2 | TSA_Standard_Deviation | 0.294 |
| 3 | Temperature_Kelvin_Standard_Deviation | 0.294 |
| 4 | TSA_DHW | 0.290 |
| 5 | Temperature_Minimum | 0.249 |
| 6 | TSA_Minimum | 0.246 |
| 7 | TSA | 0.219 |
| 8 | Temperature_Kelvin | 0.218 |
| 9 | TSA_Mean | 0.214 |
| 10 | SSTA | 0.183 |

DHW (Degree Heating Weeks) dominates both globally and for GBR — thermal stress accumulation is the primary bleaching driver, as expected from the literature. GBR shows stronger correlations overall (r=0.33 vs 0.27 for top feature).

## Collinearity Pairs (|r| > 0.95)

| Feature A | Feature B | |r| |
|-----------|-----------|-----|
| Temperature_Kelvin_Standard_Deviation | TSA_Standard_Deviation | 0.999 |
| TSA_FrequencyMean | TSA_DHWMean | 0.975 |
| TSA_DHW_Standard_Deviation | TSA_DHWMean | 0.970 |
| TSA_Standard_Deviation | TSA_Minimum | 0.964 |
| Temperature_Kelvin_Standard_Deviation | TSA_Minimum | 0.963 |
| TSA_FrequencyMean | TSA_DHW_Standard_Deviation | 0.963 |

6 pairs total. Not dropped — XGBoost's `colsample_bytree` handles collinearity by randomly sampling feature subsets per tree.

## Global Geographic Coverage

| Ocean | Rows |
|-------|------|
| Pacific | 17,311 |
| Atlantic | 13,311 |
| Indian | 2,322 |
| Red Sea | 1,042 |
| Arabian Gulf | 369 |

## Files Written

- `src/utils.py` — column name constants, GBR bounds, feature lists, sector logic
- `src/ingest.py` — `load_gcbd()`: reads CSV, coerces target, drops invalid rows
- `src/features.py` — `prepare_features()`: feature selection, coercion, one-hot encoding, NaN drop
- `src/preprocess.py` — `scale_target()`, `tag_regions()`: target scaling, GBR/sector tagging
- `run_sprint1.py` — end-to-end pipeline + EDA checks
- `data/processed/global_training_data.csv` — clean processed dataset
- `outputs/figures/correlation_top10.png` — top-10 feature correlation bar plots
- `outputs/figures/global_coverage.png` — world scatter plot of training data
- `outputs/figures/target_distribution.png` — target histograms (global + GBR)

## Sprint 2 Prerequisites

1. Load `data/processed/global_training_data.csv`
2. Train set: all rows where `Date_Year < 2016`
3. Test set: rows where `is_GBR == True` and `Date_Year >= 2016` (182 rows)
4. Feature columns: everything except `Percent_Bleaching`, `Date_Year`, `is_GBR`, `Sector`
5. Apply RobustScaler fit on train only
6. Use `Sector` column (GBR rows) for GroupKFold spatial CV during tuning
7. Tweedie objective — 48.1% zeros globally confirms this is the right choice
