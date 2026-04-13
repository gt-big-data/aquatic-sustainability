"""Load and clean AIMS LTMP manta-tow reef summary data."""

import os
import pandas as pd
from src.utils import RAW_AIMS_DIR, TARGET_COL, REEF_ID_COL, YEAR_COL


def load_aims() -> pd.DataFrame:
    """Load manta-tow-by-reef.csv, parse numerics, drop rows missing target.

    Returns cleaned DataFrame with 2,646 rows (expected).
    """
    path = os.path.join(RAW_AIMS_DIR, 'manta-tow-by-reef.csv')
    df = pd.read_csv(path)

    # Parse numeric columns (some have categorical codes like '3L', '2U')
    numeric_cols = [
        'MEAN_LIVE_CORAL', 'MEAN_DEAD_CORAL', 'MEAN_SOFT_CORAL',
        'MEAN_COTS_PER_TOW', 'MEAN_TROUT_PER_TOW', 'TOWS',
        'LATITUDE', 'LONGITUDE',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[YEAR_COL] = df[YEAR_COL].astype(int)

    # Drop rows missing the target
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with NaN target")

    # Summary
    print(f"Loaded AIMS data: {len(df)} rows, "
          f"{df[REEF_ID_COL].nunique()} reefs, "
          f"{df[YEAR_COL].min()}-{df[YEAR_COL].max()}")
    print(f"  Target ({TARGET_COL}) — "
          f"mean: {df[TARGET_COL].mean():.1f}%, "
          f"min: {df[TARGET_COL].min():.1f}%, "
          f"max: {df[TARGET_COL].max():.1f}%")

    return df
