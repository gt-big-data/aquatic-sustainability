#!/usr/bin/env python3
"""Sprint 1 pipeline: Data ingestion, satellite extraction, feature engineering, EDA.

Produces: data/processed/gbr_panel.csv
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import (
    TARGET_COL, REEF_ID_COL, YEAR_COL, SECTOR_COL,
    FEATURE_COLS, METADATA_COLS, TEMPORAL_CUTOFF,
    PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR,
)
from src.ingest import load_aims
from src.satellite import extract_satellite_features, sanity_check_hastings
from src.features import engineer_features, validate_no_leakage


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Step 1: Load AIMS data ─────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Load AIMS manta-tow data")
    print("=" * 60)
    aims_df = load_aims()
    print()

    # ── Step 2: Extract satellite features ─────────────────────────────
    print("=" * 60)
    print("STEP 2: Extract NOAA CRW satellite DHW + SSTA")
    print("=" * 60)
    sat_df = extract_satellite_features(aims_df, use_cache=True)
    sanity_check_hastings(sat_df)
    print()

    # ── Step 3: Feature engineering ────────────────────────────────────
    print("=" * 60)
    print("STEP 3: Feature engineering")
    print("=" * 60)
    panel = engineer_features(aims_df, sat_df)
    print()

    # ── Step 4: Validate ───────────────────────────────────────────────
    print("=" * 60)
    print("STEP 4: Validation")
    print("=" * 60)

    # Zero NaN check
    present_features = [c for c in FEATURE_COLS if c in panel.columns]
    nan_count = panel[present_features + [TARGET_COL]].isna().sum().sum()
    print(f"Total NaN in features + target: {nan_count}")
    assert nan_count == 0, "FAIL: NaN values remain in the dataset!"
    print("PASS: Zero NaN in all feature and target columns.\n")

    # No future leakage
    validate_no_leakage(panel)
    print()

    # Top-10 feature correlations with target
    print("Top feature correlations with target:")
    corrs = panel[present_features + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    corrs_sorted = corrs.abs().sort_values(ascending=False)
    for feat in corrs_sorted.index[:10]:
        print(f"  {feat:25s} r = {corrs[feat]:+.3f}")
    print()

    # Target stats by year
    print("Target stats by year:")
    yearly = panel.groupby(YEAR_COL)[TARGET_COL].agg(['mean', 'count'])
    for year, row in yearly.iterrows():
        print(f"  {year}: mean={row['mean']:5.1f}%  n={int(row['count'])}")
    print()

    # Train/test split preview
    train = panel[panel[YEAR_COL] < TEMPORAL_CUTOFF]
    test = panel[panel[YEAR_COL] >= TEMPORAL_CUTOFF]
    print(f"Train (< {TEMPORAL_CUTOFF}): {len(train)} rows, "
          f"mean target = {train[TARGET_COL].mean():.1f}%")
    print(f"Test (>= {TEMPORAL_CUTOFF}): {len(test)} rows, "
          f"mean target = {test[TARGET_COL].mean():.1f}%")
    print()

    # ── Step 5: Save ───────────────────────────────────────────────────
    out_path = os.path.join(PROCESSED_DIR, 'gbr_panel.csv')
    panel.to_csv(out_path, index=False)
    print(f"Saved final panel to {out_path}")
    print(f"  {len(panel)} rows, {panel[REEF_ID_COL].nunique()} reefs, "
          f"{len(present_features)} features\n")

    # ── Step 6: EDA Plots ──────────────────────────────────────────────
    print("=" * 60)
    print("STEP 5: EDA Plots")
    print("=" * 60)
    _plot_coral_timeseries(panel)
    _plot_dhw_vs_coral_change(panel)
    _plot_feature_correlations(panel, corrs)
    _plot_target_distribution(panel)
    print("All plots saved.\n")

    # ── Step 7: Write data README ──────────────────────────────────────
    _write_data_readme(aims_df, sat_df, panel, yearly)

    print("=" * 60)
    print("Sprint 1 COMPLETE")
    print("=" * 60)


# ── EDA Plotting Functions ─────────────────────────────────────────────


def _plot_coral_timeseries(panel: pd.DataFrame):
    """Mean coral cover across all reefs by year."""
    fig, ax = plt.subplots(figsize=(12, 5))
    yearly = panel.groupby(YEAR_COL)[TARGET_COL].mean()
    ax.plot(yearly.index, yearly.values, 'b-o', linewidth=2, markersize=5)

    # Mark known mass bleaching events
    for yr, label in [(1998, '1998'), (2002, '2002'), (2016, '2016'),
                      (2017, '2017'), (2020, '2020'), (2022, '2022')]:
        if yr in yearly.index:
            ax.axvline(yr, color='red', alpha=0.3, linestyle='--')

    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Coral Cover (%)')
    ax.set_title('Mean Coral Cover Across GBR Reefs by Year')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'coral_cover_timeseries.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_dhw_vs_coral_change(panel: pd.DataFrame):
    """Scatter: DHW_MAX_PREV vs coral cover change, colored by sector."""
    if 'DHW_MAX_PREV' not in panel.columns:
        print("  Skipping DHW vs coral change plot (DHW_MAX_PREV not available)")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    panel = panel.copy()
    panel['CORAL_CHANGE'] = panel[TARGET_COL] - panel['LIVE_CORAL_LAG1']

    sectors = panel[SECTOR_COL].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(sectors)))
    for sector, color in zip(sorted(sectors), colors):
        mask = panel[SECTOR_COL] == sector
        ax.scatter(
            panel.loc[mask, 'DHW_MAX_PREV'],
            panel.loc[mask, 'CORAL_CHANGE'],
            c=[color], label=sector, alpha=0.5, s=20, edgecolors='none',
        )

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Previous Year Max DHW (°C-weeks)')
    ax.set_ylabel('Coral Cover Change from Lag-1 (%)')
    ax.set_title('Thermal Stress vs Coral Cover Change')
    ax.legend(fontsize=7, title='Sector', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'dhw_vs_coral_change.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_feature_correlations(panel: pd.DataFrame, corrs: pd.Series):
    """Bar chart of top-10 feature correlations with target."""
    top10 = corrs.abs().sort_values(ascending=True).tail(10)
    colors = ['green' if corrs[f] > 0 else 'red' for f in top10.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top10.index, [corrs[f] for f in top10.index], color=colors, edgecolor='black')
    ax.set_xlabel(f'Correlation with {TARGET_COL}')
    ax.set_title('Top-10 Feature Correlations with Coral Cover')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'feature_correlations.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_target_distribution(panel: pd.DataFrame):
    """Histogram of target for train vs test sets."""
    fig, ax = plt.subplots(figsize=(10, 5))
    train = panel[panel[YEAR_COL] < TEMPORAL_CUTOFF][TARGET_COL]
    test = panel[panel[YEAR_COL] >= TEMPORAL_CUTOFF][TARGET_COL]

    ax.hist(train, bins=30, alpha=0.6, label=f'Train (n={len(train)})', color='blue',
            edgecolor='black')
    ax.hist(test, bins=30, alpha=0.6, label=f'Test (n={len(test)})', color='orange',
            edgecolor='black')
    ax.set_xlabel('Mean Live Coral Cover (%)')
    ax.set_ylabel('Count')
    ax.set_title('Target Distribution: Train vs Test')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, 'target_distribution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


# ── Data README ────────────────────────────────────────────────────────


def _write_data_readme(aims_df, sat_df, panel, yearly):
    """Write data/README.md documenting the pipeline."""
    present_features = [c for c in FEATURE_COLS if c in panel.columns]
    train = panel[panel[YEAR_COL] < TEMPORAL_CUTOFF]
    test = panel[panel[YEAR_COL] >= TEMPORAL_CUTOFF]

    content = f"""# AIMS Coral Cover Data Pipeline

## Sources

### AIMS Long-Term Monitoring Program (LTMP)
- **File:** `raw/aims/manta-tow-by-reef.csv`
- **Source:** AIMS manta-tow reef-level summary data
- **Raw rows:** {len(aims_df)}
- **Reefs:** {aims_df[REEF_ID_COL].nunique()}
- **Year range:** {aims_df[YEAR_COL].min()}-{aims_df[YEAR_COL].max()}
- **Target:** `MEAN_LIVE_CORAL` (0-{aims_df[TARGET_COL].max():.1f}%, mean {aims_df[TARGET_COL].mean():.1f}%)

### NOAA Coral Reef Watch (CRW)
- **Dataset:** NOAA_DHW on CoastWatch ERDDAP (daily 5km global)
- **Variables extracted:** `CRW_DHW` (Degree Heating Weeks), `CRW_SSTANOMALY` (SST Anomaly)
- **Method:** Downloaded GBR bounding box (10-25S, 142-154E) NetCDF per year,
  Jan-Apr bleaching season. Extracted annual max at each reef coordinate using
  nearest-neighbor grid cell matching.
- **Rows:** {len(sat_df)} (296 reefs x 31 years)
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

- Raw AIMS data: {len(aims_df)} rows
- After feature engineering + drops: {len(panel)} rows
- Final dataset: `processed/gbr_panel.csv`

## Train/Test Split

- **Train** (< {TEMPORAL_CUTOFF}): {len(train)} rows, mean target = {train[TARGET_COL].mean():.1f}%
- **Test** (>= {TEMPORAL_CUTOFF}): {len(test)} rows, mean target = {test[TARGET_COL].mean():.1f}%
"""
    readme_path = os.path.join(PROJECT_ROOT, 'data', 'README.md')
    with open(readme_path, 'w') as f:
        f.write(content)
    print(f"Wrote {readme_path}")


if __name__ == '__main__':
    main()
