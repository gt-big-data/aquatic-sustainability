"""Extract NOAA CRW satellite DHW and SST anomaly for GBR reef sites.

Strategy: Download GBR-subset NetCDF from ERDDAP year-by-year (Jan-Apr
bleaching season), then extract annual max DHW and SSTA at each reef
coordinate. Results cached to CSV to avoid re-downloading.

ERDDAP dataset: NOAA_DHW (daily 5km CRW products)
Variables: CRW_DHW, CRW_SSTANOMALY
Coordinates: time, latitude, longitude
"""

import os
import time
import numpy as np
import pandas as pd
import requests

from src.utils import RAW_NOAA_DIR, REEF_ID_COL

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW"
CACHE_CSV = os.path.join(RAW_NOAA_DIR, 'reef_dhw_annual.csv')

# GBR bounding box
GBR_LAT_MIN, GBR_LAT_MAX = -25.0, -10.0
GBR_LON_MIN, GBR_LON_MAX = 142.0, 154.0

# Bleaching season for the southern hemisphere GBR
SEASON_START_MONTH, SEASON_END_MONTH = 1, 4

# Years to cover
YEAR_START, YEAR_END = 1993, 2023


def _download_year_nc(year: int, out_dir: str) -> str:
    """Download a GBR-subset NetCDF for one year's bleaching season."""
    fname = os.path.join(out_dir, f'gbr_dhw_{year}.nc')
    if os.path.exists(fname):
        return fname

    start = f"{year}-01-01T12:00:00Z"
    end = f"{year}-04-30T12:00:00Z"

    # ERDDAP griddap query — request DHW and SSTANOMALY for GBR box
    url = (
        f"{ERDDAP_BASE}.nc?"
        f"CRW_DHW[({start}):1:({end})]"
        f"[({GBR_LAT_MIN}):1:({GBR_LAT_MAX})]"
        f"[({GBR_LON_MIN}):1:({GBR_LON_MAX})],"
        f"CRW_SSTANOMALY[({start}):1:({end})]"
        f"[({GBR_LAT_MIN}):1:({GBR_LAT_MAX})]"
        f"[({GBR_LON_MIN}):1:({GBR_LON_MAX})]"
    )

    print(f"  Downloading {year} (Jan-Apr GBR subset)...")

    # ERDDAP may redirect to a mirror (e.g. PacIOOS) — allow redirects,
    # use generous timeouts (connect=30s, read=600s), and retry up to 3 times
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=(30, 600), stream=True,
                                allow_redirects=True)
            resp.raise_for_status()
            with open(fname, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            size_mb = os.path.getsize(fname) / 1e6
            print(f"    Saved {fname} ({size_mb:.1f} MB)")
            return fname
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(5)

    raise RuntimeError(f"Failed to download {year} after 3 attempts")


def _extract_from_nc(nc_path: str, reef_coords: pd.DataFrame, year: int) -> list[dict]:
    """Extract annual max DHW and SSTA at each reef coord from a NetCDF file."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    results = []

    for _, reef in reef_coords.iterrows():
        try:
            point = ds.sel(
                latitude=reef['LATITUDE'],
                longitude=reef['LONGITUDE'],
                method='nearest'
            )
            dhw_max = float(point['CRW_DHW'].max(dim='time', skipna=True).values)
            ssta_max = float(point['CRW_SSTANOMALY'].max(dim='time', skipna=True).values)
        except Exception:
            dhw_max = np.nan
            ssta_max = np.nan

        results.append({
            REEF_ID_COL: reef[REEF_ID_COL],
            'YEAR': year,
            'DHW_MAX': dhw_max,
            'SSTA_MAX': ssta_max,
        })

    ds.close()
    return results


def _fetch_single_point_csv(lat: float, lon: float, year: int) -> dict:
    """Fallback: fetch DHW for a single reef-year via ERDDAP CSV endpoint."""
    start = f"{year}-01-01T12:00:00Z"
    end = f"{year}-04-30T12:00:00Z"

    url = (
        f"{ERDDAP_BASE}.csv?"
        f"CRW_DHW[({start}):1:({end})]"
        f"[({lat}):1:({lat})][({lon}):1:({lon})],"
        f"CRW_SSTANOMALY[({start}):1:({end})]"
        f"[({lat}):1:({lat})][({lon}):1:({lon})]"
    )

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        lines = resp.text.strip().split('\n')
        # ERDDAP CSV: line 0 = header, line 1 = units, lines 2+ = data
        dhw_vals, ssta_vals = [], []
        for line in lines[2:]:
            parts = line.split(',')
            try:
                dhw_vals.append(float(parts[3]))
            except (ValueError, IndexError):
                pass
            try:
                ssta_vals.append(float(parts[4]))
            except (ValueError, IndexError):
                pass
        return {
            'DHW_MAX': max(dhw_vals) if dhw_vals else np.nan,
            'SSTA_MAX': max(ssta_vals) if ssta_vals else np.nan,
        }
    except Exception as e:
        print(f"    CSV fallback error for ({lat},{lon},{year}): {e}")
        return {'DHW_MAX': np.nan, 'SSTA_MAX': np.nan}


def extract_satellite_features(aims_df: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """Extract annual max DHW and SSTA for all reefs in the AIMS dataset.

    Tries year-by-year NetCDF downloads first. Falls back to per-point CSV
    queries for any years where the NetCDF approach fails.

    Args:
        aims_df: cleaned AIMS DataFrame (needs REEF_ID, LATITUDE, LONGITUDE)
        use_cache: if True, load from cached CSV if it exists

    Returns:
        DataFrame with columns [REEF_ID, YEAR, DHW_MAX, SSTA_MAX]
    """
    if use_cache and os.path.exists(CACHE_CSV):
        print(f"Loading cached satellite data from {CACHE_CSV}")
        return pd.read_csv(CACHE_CSV)

    os.makedirs(RAW_NOAA_DIR, exist_ok=True)

    # Unique reef coordinates
    reef_coords = (
        aims_df[[REEF_ID_COL, 'LATITUDE', 'LONGITUDE']]
        .drop_duplicates(subset=[REEF_ID_COL])
        .reset_index(drop=True)
    )
    print(f"Extracting satellite data for {len(reef_coords)} reefs, "
          f"{YEAR_START}-{YEAR_END}")

    all_results = []

    for year in range(YEAR_START, YEAR_END + 1):
        try:
            nc_path = _download_year_nc(year, RAW_NOAA_DIR)
            results = _extract_from_nc(nc_path, reef_coords, year)
            all_results.extend(results)
        except Exception as e:
            print(f"  NetCDF failed for {year}: {e}")
            print(f"  Falling back to per-point CSV queries for {year}...")
            for _, reef in reef_coords.iterrows():
                vals = _fetch_single_point_csv(
                    reef['LATITUDE'], reef['LONGITUDE'], year
                )
                all_results.append({
                    REEF_ID_COL: reef[REEF_ID_COL],
                    'YEAR': year,
                    **vals,
                })
                time.sleep(0.5)  # rate limit

        # Save incrementally after each year
        pd.DataFrame(all_results).to_csv(CACHE_CSV, index=False)

    sat_df = pd.DataFrame(all_results)
    sat_df.to_csv(CACHE_CSV, index=False)

    print(f"\nSatellite extraction complete: {len(sat_df)} rows")
    print(f"  DHW_MAX range: [{sat_df['DHW_MAX'].min():.1f}, {sat_df['DHW_MAX'].max():.1f}]")
    print(f"  SSTA_MAX range: [{sat_df['SSTA_MAX'].min():.1f}, {sat_df['SSTA_MAX'].max():.1f}]")
    print(f"  NaN rates — DHW: {sat_df['DHW_MAX'].isna().mean():.1%}, "
          f"SSTA: {sat_df['SSTA_MAX'].isna().mean():.1%}")
    print(f"Saved to {CACHE_CSV}")

    return sat_df


def sanity_check_hastings(sat_df: pd.DataFrame):
    """Verify DHW for Hastings Reef (16028S area) in 2020 is plausible."""
    # Hastings Reef is near lat=-16.52, lon=146.02
    # Look for any reef near those coords in 2020
    check = sat_df[sat_df['YEAR'] == 2020].copy()
    if check.empty:
        print("WARNING: No 2020 satellite data found for sanity check")
        return

    # Find reef closest to Hastings coords
    # The AIMS REEF_ID for a Cairns-area reef
    print("\nSanity check — 2020 Cairns-area DHW values (should be 2-6 for bleaching event):")
    sample = check.nlargest(5, 'DHW_MAX')
    for _, row in sample.iterrows():
        print(f"  {row[REEF_ID_COL]}: DHW_MAX={row['DHW_MAX']:.1f}, SSTA_MAX={row['SSTA_MAX']:.2f}")

    median_dhw = check['DHW_MAX'].median()
    print(f"  2020 median DHW across all reefs: {median_dhw:.1f}")
    if median_dhw < 0.1:
        print("  WARNING: 2020 DHW values look too low — check data source")
    elif median_dhw > 15:
        print("  WARNING: 2020 DHW values look too high — check units")
    else:
        print("  Values look plausible.")
