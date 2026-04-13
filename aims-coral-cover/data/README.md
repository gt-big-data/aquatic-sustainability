# AIMS Coral Cover Data Pipeline

## Sources

### AIMS Long-Term Monitoring Program (LTMP)
- **File:** `raw/aims/manta-tow-by-reef.csv`
- **Source:** AIMS manta-tow reef-level summary data
- **Raw rows:** 2646
- **Reefs:** 296
- **Year range:** 1993-2023
- **Target:** `MEAN_LIVE_CORAL` (0-86.5%, mean 23.6%)

### NOAA Coral Reef Watch (CRW)
- **Dataset:** NOAA_DHW on CoastWatch ERDDAP (daily 5km global)
- **Variables extracted:** `CRW_DHW` (Degree Heating Weeks), `CRW_SSTANOMALY` (SST Anomaly)
- **Method:** Downloaded GBR bounding box (10-25S, 142-154E) NetCDF per year,
  Jan-Apr bleaching season. Extracted annual max at each reef coordinate using
  nearest-neighbor grid cell matching.
- **Rows:** 9176 (296 reefs x 31 years)
- **Cache:** `raw/noaa-crw/reef_dhw_annual.csv`

## Feature Engineering

### Lagged Features (within each reef time series)
| Feature | Description |
|---------|-------------|
| `LIVE_CORAL_LAG1` | Previous observation's mean live coral cover |
| `DEAD_CORAL_LAG1` | Previous observation's mean dead coral |
| `COTS_LAG1` | Previous observation's mean COTS per tow |
| `CORAL_TRAJECTORY` | Lag-1 minus lag-2 coral cover (momentum) |
| `YEAR_GAP` | Years since previous observation at same reef |

### Satellite Features
| Feature | Description |
|---------|-------------|
| `DHW_MAX` | Current year's peak DHW (Jan-Apr) |
| `DHW_MAX_PREV` | Previous year's peak DHW |
| `SSTA_MAX` | Current year's peak SST anomaly |
| `SSTA_MAX_PREV` | Previous year's peak SST anomaly |

### Reef Metadata
| Feature | Description |
|---------|-------------|
| `LATITUDE` | Reef latitude |
| `LONGITUDE` | Reef longitude |
| `SHELF_M` | Mid-shelf indicator (binary) |
| `SHELF_O` | Outer-shelf indicator (binary) |

## NaN Handling

1. Rows with NaN target dropped during ingestion (0 dropped — all 2,646 have target)
2. First observation per reef dropped (no lag available): ~296 rows
3. Rows with year gap > 3 dropped (stale lag): varies
4. Rows with NaN in CORAL_TRAJECTORY dropped (need lag-2): ~296 rows
5. Rows with NaN in satellite features dropped: varies by coverage

## Pipeline Row Counts

- Raw AIMS data: 2646 rows
- After feature engineering + drops: 1821 rows
- Final dataset: `processed/gbr_panel.csv`

## Train/Test Split

- **Train** (< 2020): 1494 rows, mean target = 21.1%
- **Test** (>= 2020): 327 rows, mean target = 32.4%
