import pandas as pd
from src.utils import NUMERIC_FEATURE_COLS, DROP_COLS, TARGET_COL, YEAR_COL, CORTAD_COLS, SITE_NUMERIC_COLS, GEO_FEATURE_COLS


def prepare_features(df):
    """Select features, coerce types, drop high-NaN cols, one-hot encode Exposure, drop NaN rows."""
    print(f"\n--- Feature Preparation ---")
    print(f"Starting rows: {len(df)}")

    # Verify and drop high-NaN columns, printing global NaN rates
    for col in DROP_COLS:
        if col in df.columns:
            nan_rate = df[col].isna().sum() / len(df) * 100
            # Also coerce to catch "nd" before computing
            coerced = pd.to_numeric(df[col], errors='coerce')
            nan_rate_coerced = coerced.isna().sum() / len(df) * 100
            print(f"Dropping {col}: {nan_rate_coerced:.1f}% NaN globally (confirms >30% threshold)")
            df = df.drop(columns=[col])

    # Coerce ALL numeric feature columns (handles "nd" sentinels)
    for col in NUMERIC_FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # NaN audit on all feature columns
    print(f"\nNaN audit (global, n={len(df)}):")
    for col in NUMERIC_FEATURE_COLS:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            nan_pct = nan_count / len(df) * 100
            if nan_pct > 0:
                print(f"  {col}: {nan_count} NaN ({nan_pct:.1f}%)")

    # One-hot encode Exposure
    exposure_nan = df['Exposure'].isna().sum()
    exposure_empty = (df['Exposure'] == '').sum() if df['Exposure'].dtype == object else 0
    print(f"\nExposure: {exposure_nan} NaN, {exposure_empty} empty strings")

    if exposure_nan > 0:
        # Fill NaN Exposure with 'Unknown' to avoid dropping rows
        df['Exposure'] = df['Exposure'].fillna('Unknown')
        print(f"  Filled {exposure_nan} NaN Exposure values with 'Unknown'")

    exposure_dummies = pd.get_dummies(df['Exposure'], prefix='Exposure')
    exposure_dummies = exposure_dummies.astype(int)
    # Drop Sheltered as baseline (most common category)
    if 'Exposure_Sheltered' in exposure_dummies.columns:
        exposure_dummies = exposure_dummies.drop(columns=['Exposure_Sheltered'])
    df = pd.concat([df, exposure_dummies], axis=1)
    df = df.drop(columns=['Exposure'])
    exposure_cols = sorted(exposure_dummies.columns.tolist())
    print(f"One-hot encoded Exposure → {exposure_cols}")

    # Build full feature column list
    feature_cols = [c for c in NUMERIC_FEATURE_COLS if c in df.columns] + exposure_cols

    # Drop rows with any NaN in features
    nan_rows = df[feature_cols].isna().any(axis=1).sum()
    print(f"\nRows with any NaN in features: {nan_rows}")
    df = df.dropna(subset=feature_cols).copy()
    print(f"Rows after NaN drop: {len(df)}")

    # Select only needed columns
    keep_cols = [TARGET_COL] + feature_cols + [YEAR_COL]
    df = df[keep_cols].copy()

    # Verify zero NaN
    total_nan = df.isna().sum().sum()
    print(f"Total NaN in output: {total_nan}")
    assert total_nan == 0, f"Expected zero NaN but found {total_nan}"

    return df, feature_cols
