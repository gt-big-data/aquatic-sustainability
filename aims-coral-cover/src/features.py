"""Feature engineering: lagged reef features + satellite merge + encoding."""

import pandas as pd
import numpy as np

from src.utils import (
    TARGET_COL, REEF_ID_COL, YEAR_COL,
    FEATURE_COLS, METADATA_COLS, MAX_YEAR_GAP,
)


def engineer_features(aims_df: pd.DataFrame, sat_df: pd.DataFrame) -> pd.DataFrame:
    """Build the panel dataset with lagged features and satellite data.

    Args:
        aims_df: cleaned AIMS manta tow data
        sat_df: satellite data with [REEF_ID, YEAR, DHW_MAX, SSTA_MAX]

    Returns:
        Model-ready DataFrame with zero NaN in feature/target columns.
    """
    df = aims_df.copy()
    df = df.sort_values([REEF_ID_COL, YEAR_COL]).reset_index(drop=True)
    print(f"Starting feature engineering: {len(df)} rows")

    # === LAGGED FEATURES (within each reef's time series) ===
    g = df.groupby(REEF_ID_COL)

    df['LIVE_CORAL_LAG1'] = g['MEAN_LIVE_CORAL'].shift(1)
    df['DEAD_CORAL_LAG1'] = g['MEAN_DEAD_CORAL'].shift(1)
    df['COTS_LAG1'] = g['MEAN_COTS_PER_TOW'].shift(1)
    df['YEAR_GAP'] = g[YEAR_COL].diff()

    # Trajectory: lag1 - lag2 (recovery/decline momentum)
    df['LIVE_CORAL_LAG2'] = g['MEAN_LIVE_CORAL'].shift(2)
    df['CORAL_TRAJECTORY'] = df['LIVE_CORAL_LAG1'] - df['LIVE_CORAL_LAG2']

    # === SATELLITE FEATURES ===

    # Current year's DHW and SSTA
    df = df.merge(
        sat_df[[REEF_ID_COL, 'YEAR', 'DHW_MAX', 'SSTA_MAX']],
        left_on=[REEF_ID_COL, YEAR_COL],
        right_on=[REEF_ID_COL, 'YEAR'],
        how='left',
    )
    # Drop the extra YEAR column from the merge
    df = df.drop(columns=['YEAR'], errors='ignore')

    # Previous year's DHW and SSTA
    sat_prev = sat_df[[REEF_ID_COL, 'YEAR', 'DHW_MAX', 'SSTA_MAX']].copy()
    sat_prev['YEAR'] = sat_prev['YEAR'] + 1  # shift so it joins as "previous year"
    sat_prev = sat_prev.rename(columns={
        'DHW_MAX': 'DHW_MAX_PREV',
        'SSTA_MAX': 'SSTA_MAX_PREV',
    })

    df = df.merge(
        sat_prev,
        left_on=[REEF_ID_COL, YEAR_COL],
        right_on=[REEF_ID_COL, 'YEAR'],
        how='left',
    )
    df = df.drop(columns=['YEAR'], errors='ignore')

    # === SHELF ENCODING ===
    if 'SHELF' in df.columns:
        shelf_dummies = pd.get_dummies(df['SHELF'], prefix='SHELF')
        # Keep M and O (inshore I as baseline → drop_first equivalent)
        for col in ['SHELF_M', 'SHELF_O']:
            if col in shelf_dummies.columns:
                df[col] = shelf_dummies[col].astype(int)
            else:
                df[col] = 0

    # === DROP ROWS ===

    # 1. No lag-1 data (first observation per reef)
    before = len(df)
    df = df.dropna(subset=['LIVE_CORAL_LAG1'])
    print(f"  Dropped {before - len(df)} rows without lag-1 (first obs per reef)")

    # 2. Stale lag (year gap > MAX_YEAR_GAP)
    before = len(df)
    df = df[df['YEAR_GAP'] <= MAX_YEAR_GAP]
    print(f"  Dropped {before - len(df)} rows with year gap > {MAX_YEAR_GAP}")

    # 3. NaN audit before final drop
    print("\n  NaN rates before final drop:")
    for col in FEATURE_COLS:
        if col in df.columns:
            rate = df[col].isna().mean() * 100
            if rate > 0:
                print(f"    {col}: {rate:.1f}%")

    # 4. Drop remaining NaN in feature columns
    present_features = [c for c in FEATURE_COLS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=present_features + [TARGET_COL])
    print(f"  Dropped {before - len(df)} rows with NaN in features/target")

    # === SELECT FINAL COLUMNS ===
    keep_cols = [TARGET_COL] + present_features + [
        c for c in METADATA_COLS if c in df.columns
    ]
    df = df[keep_cols].copy()

    # Drop the helper column
    df = df.drop(columns=['LIVE_CORAL_LAG2'], errors='ignore')

    print(f"\nFinal dataset: {len(df)} rows, {df[REEF_ID_COL].nunique()} reefs, "
          f"{df[YEAR_COL].min()}-{df[YEAR_COL].max()}")

    return df


def validate_no_leakage(df: pd.DataFrame):
    """Confirm lagged features come from strictly earlier years."""
    # For each reef, check that the lag-1 value corresponds to an earlier year
    # Since we computed lags via groupby+shift on sorted data, this should hold
    # by construction. But let's verify.
    issues = 0
    for reef_id, group in df.groupby(REEF_ID_COL):
        years = group[YEAR_COL].values
        gaps = group['YEAR_GAP'].values
        for i in range(len(years)):
            if gaps[i] <= 0:
                print(f"  LEAKAGE: {reef_id} at year {years[i]}, gap={gaps[i]}")
                issues += 1

    if issues == 0:
        print("No future leakage detected — all lags from strictly earlier years.")
    else:
        print(f"WARNING: {issues} potential leakage issues found!")
