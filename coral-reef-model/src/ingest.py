import pandas as pd
from src.utils import RAW_GCBD_PATH, GBR_BOUNDS, LAT_COL, LON_COL, TARGET_COL, YEAR_COL


def load_gcbd(path=None):
    """Load full global GCBD CSV, coerce target, drop rows with invalid target."""
    if path is None:
        path = RAW_GCBD_PATH

    df = pd.read_csv(path)
    raw_count = len(df)
    print(f"Raw GCBD rows: {raw_count}")

    # Coerce target — handles "nd" sentinels
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    before = len(df)
    df = df.dropna(subset=[TARGET_COL]).copy()
    valid_count = len(df)
    print(f"Valid target rows: {valid_count} (dropped {before - valid_count} with NaN/nd target)")

    # Print GBR subset info for reference
    gbr_mask = (
        (df[LAT_COL] >= GBR_BOUNDS['lat_min']) &
        (df[LAT_COL] <= GBR_BOUNDS['lat_max']) &
        (df[LON_COL] >= GBR_BOUNDS['lon_min']) &
        (df[LON_COL] <= GBR_BOUNDS['lon_max'])
    )
    gbr_count = gbr_mask.sum()
    print(f"  of which GBR rows: {gbr_count}")

    # Year range
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors='coerce')
    print(f"Year range: {int(df[YEAR_COL].min())}–{int(df[YEAR_COL].max())}")

    return df
