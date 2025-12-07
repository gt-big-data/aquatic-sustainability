# global_smap_loader.py
# Automatic downloader + reprojection for SMAP SPL4SMAU (21 days)
# CRITICAL: SPL4SMAU provides 3-hourly data that must be averaged to daily

import os
import h5py
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import earthaccess
from collections import defaultdict
import re

# Output cache
SMAP_GLOBAL_NPY = "./data/global_smap_21days.npy"

# Target lat/lon grid for model (0.25°)
TARGET_LATS = np.linspace(90, -90, 720)       # descending
TARGET_LONS = np.linspace(-180, 180, 1440)    # ascending


# ================================================================
# AUTO-DOWNLOAD SMAP USING earthaccess
# ================================================================

def ensure_earthdata_auth():
    """Create .netrc on Cloud Run using env vars and authenticate directly."""
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError("Missing Earthdata credentials: EARTHDATA_USERNAME and EARTHDATA_PASSWORD must be set")

    # Cloud Run allows writing inside /root or /tmp
    NETRC_PATH = "/root/.netrc"

    # Standard .netrc format: NO indentation
    content = (
        f"machine urs.earthdata.nasa.gov\n"
        f"login {username}\n"
        f"password {password}\n"
        f"machine data.gesdisc.earthdata.nasa.gov\n"
        f"login {username}\n"
        f"password {password}\n"
    )

    with open(NETRC_PATH, "w") as f:
        f.write(content)

    os.chmod(NETRC_PATH, 0o600)

    print("[AUTH] .netrc file created successfully")
    
    # Also attempt direct login with credentials
    try:
        earthaccess.login(username=username, password=password, persist=False)
        print("[AUTH] Direct earthaccess login successful")
    except Exception as e:
        print(f"[AUTH] Direct earthaccess login failed: {e}")

def download_last_21_smap(out_dir="./data/smap_download/", day_delay=5):
    """
    Downloads the last 21 daily SMAP SPL4SMAU files from NASA.
    
    IMPORTANT: 
    - SPL4SMAU provides 3-hourly data (8 files per day)
    - We need day_delay + 1 to create proper 5-day buffer
    - Returns sorted list of downloaded file paths
    """
    os.makedirs(out_dir, exist_ok=True)

    ensure_earthdata_auth()

    print("[AUTH] Logged in to Earthdata")

    today = datetime.utcnow().date()

    # CRITICAL: Use day_delay + 1 for proper buffer (5 day delay = skip 6 days back)
    latest_available = today - timedelta(days=day_delay + 1)

    print(f"[SMAP] Latest available assumed: {latest_available}")
    print(f"[SMAP] Using {day_delay} day delay (actual: {day_delay + 1} days back)")

    # Query the last 21 days
    start = latest_available - timedelta(days=20)
    end = latest_available

    print(f"[SMAP] Querying SPL4SMAU from {start} to {end}")

    results = earthaccess.search_data(
        short_name="SPL4SMAU",
        version="008",
        temporal=(str(start), str(end)),
    )

    print(f"[SMAP] Found {len(results)} files (includes 3-hourly data).")

    # Download
    print("[SMAP] Downloading files…")
    paths = earthaccess.download(results, local_path=out_dir)

    # Filter to .h5
    smap_files = [
        str(p) for p in paths if str(p).lower().endswith(".h5")
    ]

    print(f"[SMAP] Downloaded {len(smap_files)} 3-hourly files.")
    print(f"[SMAP] Will average to ~{len(smap_files) // 8} daily files.")

    return sorted(smap_files)


# ================================================================
# LOADING AND PROCESSING SMAP FILES
# ================================================================

def load_smap_surface(local_path):
    """
    Load SMAP SPL4SMAU surface soil moisture data.
    
    CRITICAL: Uses Analysis_Data group (not Soil_Moisture_Retrieval_Data_AM)
    """
    with h5py.File(local_path, "r") as f:
        analysis_group = f["Analysis_Data"]
        
        # Try different possible key names for soil moisture
        sm_key = None
        for k in ("sm_surface_analysis", "sm_surface", "sm_surface_analysis_map"):
            if k in analysis_group:
                sm_key = k
                break
        
        if sm_key is None:
            available_keys = list(analysis_group.keys())
            raise KeyError(f"Could not find soil moisture key. Available: {available_keys}")
        
        sm = analysis_group[sm_key][:]
        
        # Set invalid values to NaN
        sm = np.where((sm < 0) | np.isnan(sm), np.nan, sm)

        # Get lat/lon coordinates
        if "cell_lat" in f and "cell_lon" in f:
            lat2d = f["cell_lat"][:]
            lon2d = f["cell_lon"][:]
        elif "Cell_Lat" in f and "Cell_Lon" in f:
            lat2d = f["Cell_Lat"][:]
            lon2d = f["Cell_Lon"][:]
        else:
            raise KeyError("No cell_lat/cell_lon found")

        # Convert 2D coords to 1D
        if lat2d.ndim == 2 and lon2d.ndim == 2:
            lat_1d = lat2d[:, 0]
            lon_1d = lon2d[0, :]
        else:
            lat_1d = np.unique(lat2d).ravel()
            lon_1d = np.unique(lon2d).ravel()

        # Fix latitude orientation (must be descending)
        if np.any(np.diff(lat_1d) < 0):
            lat_1d = lat_1d[::-1]
            sm = sm[::-1, :]

        da = xr.DataArray(
            sm, 
            dims=["lat", "lon"], 
            coords={"lat": lat_1d, "lon": lon_1d}
        )
    
    return da


def group_files_by_date(file_paths):
    """
    Group SMAP files by date.
    SMAP filenames contain date: SMAP_L4_SM_aup_YYYYMMDD_...
    """
    date_to_files = defaultdict(list)
    
    for fp in file_paths:
        match = re.search(r'(\d{8})', os.path.basename(fp))
        if match:
            date_str = match.group(1)
            file_date = datetime.strptime(date_str, "%Y%m%d").date()
            date_to_files[file_date].append(fp)
    
    return date_to_files


def bilinear_resample_to_025deg(da):
    """Resample to 0.25 degree resolution."""
    lat_min = float(np.nanmin(da.lat.values))
    lat_max = float(np.nanmax(da.lat.values))
    lon_min = float(np.nanmin(da.lon.values))
    lon_max = float(np.nanmax(da.lon.values))

    lat_new = np.arange(lat_min, lat_max + 0.25, 0.25)
    lon_new = np.arange(lon_min, lon_max + 0.25, 0.25)

    da025 = da.interp(lat=lat_new, lon=lon_new, method="linear")
    return da025


def reproject_to_025deg(da):
    """
    Reproject to 0.25°×0.25° global grid using xarray interp.
    """
    da_interp = da.interp(
        lat=TARGET_LATS,
        lon=TARGET_LONS,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    return da_interp.values.astype(np.float32)


# ================================================================
# BUILD 21-DAY GLOBAL STACK WITH DAILY AVERAGING
# ================================================================

def build_global_smap_21days(day_delay=5):
    """
    Downloads and constructs the global 21-day stack.
    
    CRITICAL STEPS:
    1. Download 3-hourly SMAP files (8 per day)
    2. Group files by date
    3. Average all 3-hourly files for each day
    4. Reproject each daily average to 0.25° grid
    5. Stack into (21, 720, 1440) array
    """
    print("[GLOBAL SMAP] Auto-building SMAP 21-day stack with daily averaging…")

    smap_files = download_last_21_smap(day_delay=day_delay)
    
    # Group files by date
    date_to_files = group_files_by_date(smap_files)
    
    # Get target dates (21 days)
    today = datetime.utcnow().date()
    latest_available = today - timedelta(days=day_delay + 1)
    target_dates = [latest_available - timedelta(days=i) for i in range(21)]
    target_dates = sorted(target_dates)  # Oldest to newest
    
    print(f"[SMAP] Target date range: {target_dates[0]} to {target_dates[-1]}")

    stack = []
    
    for target_date in target_dates:
        if target_date not in date_to_files:
            print(f"[SMAP] WARNING: Missing data for {target_date}, filling with NaNs")
            # Fill with NaN array
            stack.append(np.full((720, 1440), np.nan, dtype=np.float32))
            continue
        
        files_for_date = date_to_files[target_date]
        print(f"[SMAP] Processing {target_date}: {len(files_for_date)} 3-hourly files")
        
        # Load all 3-hourly files for this date
        daily_grids = []
        for path in files_for_date:
            try:
                da = load_smap_surface(path)
                da_resampled = bilinear_resample_to_025deg(da)
                daily_grids.append(da_resampled.values)
            except Exception as e:
                print(f"[SMAP] Error processing {os.path.basename(path)}: {e}")
                continue
        
        if len(daily_grids) == 0:
            print(f"[SMAP] WARNING: No valid data for {target_date}")
            stack.append(np.full((720, 1440), np.nan, dtype=np.float32))
            continue
        
        # CRITICAL: Average all 3-hourly grids to get daily average
        daily_avg = np.nanmean(daily_grids, axis=0).astype(np.float32)
        
        # Reproject to global 0.25° grid
        da_daily = xr.DataArray(
            daily_avg,
            dims=["lat", "lon"],
            coords={
                "lat": np.linspace(90, -90, daily_avg.shape[0]),
                "lon": np.linspace(-180, 180, daily_avg.shape[1])
            }
        )
        
        interp = reproject_to_025deg(da_daily)
        stack.append(interp)
        
        print(f"[SMAP] ✓ Averaged {len(daily_grids)} files for {target_date}")

    stack = np.stack(stack, axis=0)  # (21, 720, 1440)

    os.makedirs(os.path.dirname(SMAP_GLOBAL_NPY), exist_ok=True)
    np.save(SMAP_GLOBAL_NPY, stack)

    print(f"[SMAP] Saved → {SMAP_GLOBAL_NPY}")
    print(f"[SMAP] Final shape: {stack.shape}")
    
    return stack