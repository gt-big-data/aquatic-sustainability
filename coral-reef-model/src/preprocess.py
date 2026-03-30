import pandas as pd
from src.utils import TARGET_COL, LAT_COL, LON_COL, SECTOR_COL, GBR_COL, GBR_BOUNDS, assign_sector


def scale_target(df):
    """Scale Percent_Bleaching from [0, 100] to [0.0, 1.0]."""
    df[TARGET_COL] = df[TARGET_COL] / 100.0
    df[TARGET_COL] = df[TARGET_COL].clip(0.0, 1.0)

    print(f"\n--- Target Distribution (scaled [0,1]) ---")
    print(f"Global (n={len(df)}):")
    print(f"  Mean:   {df[TARGET_COL].mean():.4f}")
    print(f"  Median: {df[TARGET_COL].median():.4f}")
    print(f"  Std:    {df[TARGET_COL].std():.4f}")
    print(f"  Min:    {df[TARGET_COL].min():.4f}")
    print(f"  Max:    {df[TARGET_COL].max():.4f}")
    print(f"  % zeros:      {(df[TARGET_COL] == 0).mean() * 100:.1f}%")
    print(f"  % above 0.5:  {(df[TARGET_COL] > 0.5).mean() * 100:.1f}%")

    # GBR-only stats for comparison
    gbr = df[
        (df[LAT_COL] >= GBR_BOUNDS['lat_min']) & (df[LAT_COL] <= GBR_BOUNDS['lat_max']) &
        (df[LON_COL] >= GBR_BOUNDS['lon_min']) & (df[LON_COL] <= GBR_BOUNDS['lon_max'])
    ]
    print(f"\nGBR only (n={len(gbr)}):")
    print(f"  Mean:   {gbr[TARGET_COL].mean():.4f}")
    print(f"  Median: {gbr[TARGET_COL].median():.4f}")
    print(f"  % zeros:      {(gbr[TARGET_COL] == 0).mean() * 100:.1f}%")
    print(f"  % above 0.5:  {(gbr[TARGET_COL] > 0.5).mean() * 100:.1f}%")

    return df


def tag_regions(df):
    """Add is_GBR boolean and Sector column."""
    df[GBR_COL] = (
        (df[LAT_COL] >= GBR_BOUNDS['lat_min']) & (df[LAT_COL] <= GBR_BOUNDS['lat_max']) &
        (df[LON_COL] >= GBR_BOUNDS['lon_min']) & (df[LON_COL] <= GBR_BOUNDS['lon_max'])
    )

    df[SECTOR_COL] = df.apply(
        lambda row: assign_sector(row[LAT_COL]) if row[GBR_COL] else 'Non-GBR',
        axis=1,
    )

    gbr_count = df[GBR_COL].sum()
    non_gbr_count = len(df) - gbr_count
    print(f"\n--- Region Tagging ---")
    print(f"Total rows: {len(df)}")
    print(f"  GBR:     {gbr_count}")
    print(f"  Non-GBR: {non_gbr_count}")

    gbr_sectors = df[df[GBR_COL]][SECTOR_COL].value_counts()
    print(f"\nGBR sector distribution:")
    for sector in ['Northern', 'Central', 'Southern']:
        n = gbr_sectors.get(sector, 0)
        print(f"  {sector}: {n} rows")

    return df
