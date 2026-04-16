# Data Dictionary

## Sources

### GCBD — BCO-DMO (Primary: Training + Test)
- **File:** `data/raw/gcbd/global_bleaching_environmental.csv`
- **Source:** BCO-DMO Global Coral Bleaching Database
- **Download:** https://www.bco-dmo.org/dataset/773466
- **Raw rows:** 41,361 globally
- **Columns:** 62
- **GBR-filtered rows:** 2,192
- **GBR + valid target rows:** 1,950 (242 rows had `"nd"` in Percent_Bleaching)
- **Year range (full):** 1980–2020
- **Year range (GBR):** 1992–2017

Note: `data/raw/gcbd/Global_Coral_Bleaching_Database.csv` also exists (33,244 rows, 21 columns) but lacks environmental covariates. Use `global_bleaching_environmental.csv` for all modeling work.

#### Column Inventory (62 columns)

**Target:**
- `Percent_Bleaching` — percentage of coral showing bleaching (0–100)

**Location/Time:**
- `Site_ID`, `Sample_ID` — record identifiers
- `Latitude_Degrees`, `Longitude_Degrees` — survey coordinates
- `Date_Year`, `Date_Month`, `Date_Day`, `Date` — survey date

**Geographic/Taxonomic metadata:**
- `Data_Source`, `Ocean_Name`, `Reef_ID`, `Realm_Name`, `Ecoregion_Name`
- `Country_Name`, `State_Island_Province_Name`, `City_Town_Name`, `Site_Name`
- `Substrate_Name`, `Bleaching_Level`, `Percent_Cover`

**Text fields:**
- `Site_Comments`, `Sample_Comments`, `Bleaching_Comments`

**CoRTAD v6 Thermal Features (32 columns, all 0.1% NaN = 1 row):**

| Group | Columns |
|-------|---------|
| Climatological SST | `ClimSST` |
| Temperature | `Temperature_Kelvin`, `Temperature_Mean`, `Temperature_Minimum`, `Temperature_Maximum`, `Temperature_Kelvin_Standard_Deviation` |
| SSTA (SST Anomaly) | `SSTA`, `SSTA_Standard_Deviation`, `SSTA_Mean`, `SSTA_Minimum`, `SSTA_Maximum` |
| SSTA Frequency | `SSTA_Frequency`, `SSTA_Frequency_Standard_Deviation`, `SSTA_FrequencyMax`, `SSTA_FrequencyMean` |
| SSTA DHW | `SSTA_DHW`, `SSTA_DHW_Standard_Deviation`, `SSTA_DHWMax`, `SSTA_DHWMean` |
| TSA (Thermal Stress Anomaly) | `TSA`, `TSA_Standard_Deviation`, `TSA_Minimum`, `TSA_Maximum`, `TSA_Mean` |
| TSA Frequency | `TSA_Frequency`, `TSA_Frequency_Standard_Deviation`, `TSA_FrequencyMax`, `TSA_FrequencyMean` |
| TSA DHW | `TSA_DHW`, `TSA_DHW_Standard_Deviation`, `TSA_DHWMax`, `TSA_DHWMean` |

**Site Metadata:**

| Column | NaN % | NaN Count | Decision |
|--------|-------|-----------|----------|
| `Distance_to_Shore` | 0.0% | 0 | Keep |
| `Turbidity` | 0.0% | 0 | Keep |
| `Cyclone_Frequency` | 0.0% | 0 | Keep |
| `Windspeed` | 0.1% | 1 | Keep (drop 1 row) |
| `Exposure` | 0.0% | 0 | Categorical: Sheltered (1748), Exposed (360), Sometimes (84) — one-hot encode |
| `Depth_m` | 40.4% | 787 | **DROP** (>30% NaN threshold) |
| `Percent_Cover` | 44.8% | 874 | **DROP** (>30% NaN threshold) |

#### Target Distribution (GBR, n=1,950)

| Statistic | Value |
|-----------|-------|
| Mean | 20.03% |
| Std | 28.33% |
| Min | 0.0% |
| 25th | 0.75% |
| Median | 5.50% |
| 75th | 30.50% |
| Max | 100.0% |
| **% rows = 0** | **14.2%** |

Zero-inflation present (14.2% exact zeros) — Tweedie objective appropriate.

#### Sector Distribution (GBR, n=1,950)

| Sector | Latitude Range | Rows | % |
|--------|---------------|------|-----|
| Northern | >= -16.0 | 291 | 14.9% |
| Central | -16.0 to -20.0 | 1,096 | 56.2% |
| Southern | < -20.0 | 563 | 28.9% |

All sectors >50 rows — viable for 3-fold GroupKFold spatial CV. Northern is smallest but 291 is sufficient.

#### GBR Rows by Year

| Year | Rows | | Year | Rows |
|------|------|-|------|------|
| 1992 | 1 | | 2007 | 55 |
| 1996 | 6 | | 2008 | 60 |
| 1998 | 433 | | 2009 | 96 |
| 1999 | 4 | | 2010 | 79 |
| 2000 | 1 | | 2011 | 40 |
| 2002 | 382 | | 2012 | 25 |
| 2003 | 49 | | 2013 | 65 |
| 2004 | 55 | | 2014 | 64 |
| 2005 | 123 | | 2015 | 62 |
| 2006 | 168 | | 2016 | 82 |
| | | | 2017 | 100 |

**Temporal split (pre-2016 / 2016+):** Train = 1,768 rows (1992–2015), Test = 182 rows (2016–2017).

#### NaN Summary (GBR, n=1,950, `"nd"` coerced to NaN)

- **Depth_m:** 40.4% (787 rows) — DROP column
- **Percent_Cover:** 44.8% (874 rows) — DROP column
- **All 32 CoRTAD thermal columns:** 0.1% (1 row each, same row) — drop that 1 row
- **Windspeed:** 0.1% (1 row) — drop that row
- **Distance_to_Shore, Turbidity, Cyclone_Frequency:** 0.0%
- **Exposure:** 0.0% (categorical)

Sentinel value: `"nd"` = no data. Must use `pd.to_numeric(col, errors='coerce')` on all feature columns.

---

### AIMS LTMP Manta Tow (Validation Only)
- **File:** `data/raw/aims/manta-tow-by-reef.csv`
- **Source:** AIMS Long-Term Monitoring Program
- **Role:** Post-2020 inference validation only — NOT a training target
- **Row count:** 2,646
- **Columns:** 22
- **Year range:** 1993–2023 (REPORT_YEAR)
- **Bleaching column:** NONE — this is why it's validation-only
- **Post-2020 rows:** 411 (2020: 86, 2021: 127, 2022: 87, 2023: 111)

**Key columns:** MEAN_LIVE_CORAL, MEAN_DEAD_CORAL, LATITUDE, LONGITUDE, REPORT_YEAR, SECTOR, SHELF, REEF_NAME, REEF_ID, SAMPLE_DATE

**SECTOR codes:** CA, CB, CG, CL, CU, IN, PC, PO, SW, TO, WH (AIMS internal codes — different from our Northern/Central/Southern latitude-based scheme)

**Validation use:** After model predicts bleaching severity at reef locations using NOAA CRW satellite features, cross-reference whether reefs flagged as high-bleaching-risk show drops in MEAN_LIVE_CORAL or spikes in MEAN_DEAD_CORAL.

---

## Feature Availability Matrix

| Feature | In GCBD? | Column Name(s) | NaN | Notes |
|---------|----------|----------------|-----|-------|
| SST (temperature metrics) | YES | Temperature_Kelvin/Mean/Min/Max/StdDev, ClimSST | 0.1% | CoRTAD v6 pre-matched |
| SSTA (SST Anomaly) | YES | SSTA/StdDev/Mean/Min/Max | 0.1% | CoRTAD v6 |
| SSTA Frequency | YES | SSTA_Frequency/StdDev/Max/Mean | 0.1% | CoRTAD v6 |
| SSTA DHW | YES | SSTA_DHW/StdDev/Max/Mean | 0.1% | CoRTAD v6 |
| TSA | YES | TSA/StdDev/Min/Max/Mean | 0.1% | CoRTAD v6 |
| TSA Frequency | YES | TSA_Frequency/StdDev/Max/Mean | 0.1% | CoRTAD v6 |
| TSA DHW | YES | TSA_DHW/StdDev/Max/Mean | 0.1% | CoRTAD v6 — primary DHW metric |
| Windspeed | YES | Windspeed | 0.1% | Pre-matched |
| Distance to Shore | YES | Distance_to_Shore | 0.0% | Pre-matched |
| Turbidity | YES | Turbidity | 0.0% | Pre-matched |
| Cyclone Frequency | YES | Cyclone_Frequency | 0.0% | Pre-matched |
| Exposure | YES | Exposure | 0.0% | Categorical (Sheltered/Exposed/Sometimes) |
| Depth | YES | Depth_m | 40.4% | **DROP — exceeds 30% threshold** |
| Percent Cover | YES | Percent_Cover | 44.8% | **DROP — exceeds 30% threshold** |
| Ocean pH | NO | — | — | Would need Copernicus Marine (nice-to-have) |
| Dissolved Oxygen | NO | — | — | Would need Copernicus Marine (nice-to-have) |
| Chlorophyll-a | NO | — | — | Would need Copernicus Marine (nice-to-have) |

**Bottom line:** 37 usable numeric feature columns + 1 categorical (Exposure) available out of the box. No external API sourcing needed for the core model.

---

## Processing Decisions (Sprint 1)

### Architecture: Global Training

The model trains on the **full global GCBD dataset** (not GBR-only) to learn universal thermal-stress → bleaching relationships. Lat/lon are included as features so the model can learn regional differences. Test evaluation remains GBR-only (2016–2017).

### Pipeline: Raw → Processed

| Step | Action | Before → After |
|------|--------|----------------|
| Load | Read `global_bleaching_environmental.csv` | 41,361 rows |
| Coerce target | `pd.to_numeric(Percent_Bleaching, errors='coerce')` | 41,361 → 34,515 (dropped 6,846 with NaN/"nd") |
| Drop high-NaN cols | `Depth_m` (4.9% NaN globally, 40.4% in GBR), `Percent_Cover` (34.3% NaN globally) | 2 columns removed |
| Coerce features | `pd.to_numeric(errors='coerce')` on all 38 numeric columns | "nd" sentinels → NaN |
| One-hot encode | `Exposure` → `Exposure_Exposed`, `Exposure_Sometimes` (Sheltered=baseline) | +2 columns, -1 column |
| Drop NaN rows | Any row with NaN in feature columns | 34,515 → 34,355 (dropped 160 rows, 0.5%) |
| Scale target | `Percent_Bleaching / 100.0`, clipped to [0, 1] | Range: 0–100 → 0.0–1.0 |
| Tag regions | `is_GBR` boolean, `Sector` for GBR rows | +2 metadata columns |

### Processed Dataset: `data/processed/global_training_data.csv`

| Attribute | Value |
|-----------|-------|
| **Total rows** | 34,355 |
| **GBR rows** | 1,949 |
| **Non-GBR rows** | 32,406 |
| **Feature columns** | 40 |
| **Year range** | 1983–2019 |
| **Countries** | 89 |
| **Ecoregions** | 114 |

### Feature Columns (40 total)

| Group | Columns | Count |
|-------|---------|-------|
| CoRTAD v6 thermal | ClimSST, Temperature_*, SSTA_*, TSA_* | 32 |
| Site metadata | Distance_to_Shore, Turbidity, Cyclone_Frequency, Windspeed | 4 |
| Geographic | Latitude_Degrees, Longitude_Degrees | 2 |
| Exposure (one-hot) | Exposure_Exposed, Exposure_Sometimes | 2 |

### Metadata Columns (not features)

| Column | Purpose |
|--------|---------|
| `Date_Year` | Temporal train/test split (Sprint 2: train < 2016, test = GBR 2016+) |
| `is_GBR` | Identifies GBR rows for test set filtering |
| `Sector` | Northern/Central/Southern for GBR rows; "Non-GBR" otherwise |

### Target Distribution

| Stat | Global (n=34,355) | GBR (n=1,949) |
|------|--------------------|----------------|
| Mean | 0.0964 | 0.2004 |
| Median | 0.0025 | 0.0550 |
| Std | 0.2021 | 0.2833 |
| % zeros | 48.1% | 14.2% |
| % > 0.5 | 6.9% | 19.1% |

### GBR Sector Distribution

| Sector | Rows | Mean Bleaching |
|--------|------|----------------|
| Northern | 291 | 0.1644 |
| Central | 1,095 | 0.1633 |
| Southern | 563 | 0.2911 |

### Collinearity Notes

6 CoRTAD feature pairs have |Pearson r| > 0.95:
- `Temperature_Kelvin_Standard_Deviation` ↔ `TSA_Standard_Deviation`: 0.999
- `TSA_FrequencyMean` ↔ `TSA_DHWMean`: 0.975
- `TSA_DHW_Standard_Deviation` ↔ `TSA_DHWMean`: 0.970
- `TSA_Standard_Deviation` ↔ `TSA_Minimum`: 0.964
- `Temperature_Kelvin_Standard_Deviation` ↔ `TSA_Minimum`: 0.963
- `TSA_FrequencyMean` ↔ `TSA_DHW_Standard_Deviation`: 0.963

Not dropped — XGBoost's `colsample_bytree` handles collinearity.

### Top-10 Feature Correlations with Target

| Rank | Global | |r| | GBR | |r| |
|------|--------|-----|-----|-----|
| 1 | SSTA_DHW | 0.272 | SSTA_DHW | 0.332 |
| 2 | TSA_DHW | 0.272 | TSA_Standard_Deviation | 0.294 |
| 3 | SSTA_Frequency | 0.189 | Temp_Kelvin_StdDev | 0.294 |
| 4 | TSA_Frequency | 0.162 | TSA_DHW | 0.290 |
| 5 | Longitude_Degrees | 0.145 | Temperature_Minimum | 0.249 |

DHW (Degree Heating Weeks) dominates both globally and for GBR — confirms thermal stress is the primary driver.
