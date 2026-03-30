# Sprint 0 Audit Results

**Date:** 2026-03-30
**Status:** Complete

## Project Overview

Building an XGBoost regression model (Tweedie objective) to predict coral bleaching severity (Percent_Bleaching, 0–100%) on the Great Barrier Reef using satellite-derived environmental data from the BCO-DMO GCBD.

## GCBD — Primary Dataset

- **File:** `data/raw/gcbd/global_bleaching_environmental.csv` (16.7 MB, 62 columns)
  - NOT `Global_Coral_Bleaching_Database.csv` (21 columns, no covariates — ignore this file)
- **Raw rows:** 41,361 globally
- **GBR-filtered rows (bounding box Lat ∈ [-25, -10], Lon ∈ [142, 154]):** 2,192
- **GBR + valid target:** 1,950 (242 rows had `"nd"` sentinel in Percent_Bleaching)
- **Year range (GBR):** 1992–2017
- **Column names use:** `Latitude_Degrees`, `Longitude_Degrees`, `Percent_Bleaching`, `Date_Year` (not the `Latitude`/`Bleaching_Percentage` names the PRD originally assumed)
- **Sentinel value:** `"nd"` for missing data — must use `pd.to_numeric(col, errors='coerce')` on all feature columns

## Target Distribution (GBR, n=1,950)

- Mean: 20.03%, Median: 5.50%, Max: 100%
- 14.2% of rows have exactly 0% bleaching — Tweedie objective appropriate
- 75th percentile: 30.50% — most observations are low-bleaching

## Sector Distribution (latitude-based)

- Northern (>= -16.0°): 291 rows (14.9%)
- Central (-16.0° to -20.0°): 1,096 rows (56.2%)
- Southern (< -20.0°): 563 rows (28.9%)
- All >50 — viable for 3-fold GroupKFold

## Temporal Split (per PRD: pre-2016 / 2016+)

- Train: 1,768 rows (1992–2015)
- Test: 182 rows (2016–2017)
- Big years: 1998 (433 rows), 2002 (382 rows), 2006 (168 rows)

## Feature Columns — 37 numeric + 1 categorical

**32 CoRTAD v6 thermal columns (all 0.1% NaN = 1 row):**
- ClimSST
- Temperature_Kelvin, Temperature_Mean, Temperature_Minimum, Temperature_Maximum, Temperature_Kelvin_Standard_Deviation
- SSTA, SSTA_Standard_Deviation, SSTA_Mean, SSTA_Minimum, SSTA_Maximum
- SSTA_Frequency, SSTA_Frequency_Standard_Deviation, SSTA_FrequencyMax, SSTA_FrequencyMean
- SSTA_DHW, SSTA_DHW_Standard_Deviation, SSTA_DHWMax, SSTA_DHWMean
- TSA, TSA_Standard_Deviation, TSA_Minimum, TSA_Maximum, TSA_Mean
- TSA_Frequency, TSA_Frequency_Standard_Deviation, TSA_FrequencyMax, TSA_FrequencyMean
- TSA_DHW, TSA_DHW_Standard_Deviation, TSA_DHWMax, TSA_DHWMean

**Site metadata (0% NaN):** Distance_to_Shore, Turbidity, Cyclone_Frequency

**Windspeed:** 0.1% NaN (1 row)

**Exposure:** Categorical — Sheltered (1748), Exposed (360), Sometimes (84) — needs one-hot encoding

**DROP columns (>30% NaN):** Depth_m (40.4%), Percent_Cover (44.8%)

**Not in GCBD (nice-to-have, would need Copernicus Marine):** Ocean pH, Dissolved Oxygen, Chlorophyll-a

## AIMS LTMP — Validation Only

- **File:** `data/raw/aims/manta-tow-by-reef.csv` (2,646 rows, 22 columns)
- **NO bleaching column** — validation-only via coral cover trends
- **Year range:** 1993–2023 (REPORT_YEAR column)
- **Post-2020 rows:** 411 (2020: 86, 2021: 127, 2022: 87, 2023: 111)
- **Key columns:** MEAN_LIVE_CORAL, MEAN_DEAD_CORAL, LATITUDE, LONGITUDE, REPORT_YEAR, SECTOR
- **SECTOR codes:** CA, CB, CG, CL, CU, IN, PC, PO, SW, TO, WH (AIMS internal, not our lat-based scheme)
- **YEAR_CODE** is academic-year format (e.g., 202223) — use REPORT_YEAR instead

## Repo Structure

```
coral-reef-model/
├── data/raw/gcbd/global_bleaching_environmental.csv  ← USE THIS
├── data/raw/gcbd/Global_Coral_Bleaching_Database.csv ← ignore (no covariates)
├── data/raw/aims/manta-tow-by-reef.csv
├── data/processed/
├── data/README.md                ← full audit results
├── src/{ingest,preprocess,features,train,tune,evaluate,inference,utils}.py
├── docs/{PRD.md, sprints.md}
├── outputs/{models,scalers,figures,reports}/
├── notebooks/, tests/
├── requirements.txt              ← pandas,numpy,xgboost,sklearn,matplotlib,seaborn,shap,joblib,requests
└── venv/                         ← Python 3.14, all deps installed and importable
```

## Sprint 1 Prerequisites

1. Use `global_bleaching_environmental.csv` (not the other CSV)
2. Coerce all feature columns with `pd.to_numeric(errors='coerce')` to handle `"nd"` sentinels
3. Drop `Depth_m` and `Percent_Cover` (>30% NaN)
4. Drop the ~1 row with NaN across all CoRTAD + Windspeed columns
5. One-hot encode `Exposure` (Sheltered/Exposed/Sometimes)
6. No external API sourcing needed — 37 features available in the GCBD
