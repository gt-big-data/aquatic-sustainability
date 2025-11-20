# global_smap_loader.py
# Automatic downloader + reprojection for SMAP SPL3SMP (21 days)

import os
import h5py
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import earthaccess

# Output cache
SMAP_GLOBAL_NPY = "./data/global_smap_21days.npy"

# Target lat/lon grid for model (0.25°)
TARGET_LATS = np.linspace(90, -90, 720)       # descending
TARGET_LONS = np.linspace(-180, 180, 1440)    # ascending


# ================================================================
# AUTO-DOWNLOAD SMAP USING earthaccess
# ================================================================

def download_last_21_smap(out_dir="./data/smap_download/"):
    """
    Downloads the last 21 daily SMAP SPL3SMP files from NASA.
    Returns a sorted list of downloaded file paths.
    """

    os.makedirs(out_dir, exist_ok=True)

    print("[AUTH] Logging in to Earthdata…")
    earthaccess.login()

    today = datetime.utcnow().date()

    # SMAP is usually available with 2–3 day delay
    latest_available = today - timedelta(days=3)

    print(f"[SMAP] Latest available assumed: {latest_available}")

    # Query the last 21 daily granules
    start = latest_available - timedelta(days=20)
    end = latest_available

    print(f"[SMAP] Querying SPL3SMP from {start} to {end}")

    results = earthaccess.search_data(
        short_name="SPL3SMP",
        version="009",  # adjust if needed
        temporal=(str(start), str(end)),
    )

    print(f"[SMAP] Found {len(results)} files.")

    # Download
    print("[SMAP] Downloading files…")
    paths = earthaccess.download(results, local_path=out_dir)

    # Filter to .h5
    smap_files = [
        str(p) for p in paths if str(p).lower().endswith(".h5")
    ]

    if len(smap_files) < 21:
        raise RuntimeError(
            f"[ERROR] Only {len(smap_files)} SMAP files downloaded — need 21."
        )

    smap_files = sorted(smap_files)[-21:]
    print(f"[SMAP] Using {len(smap_files)} most recent daily files.")

    return smap_files


# ================================================================
# LOADING AND REPROJECTING NATIVE SMAP FILES
# ================================================================
def load_native_smap_file(path):
    """
    Loads a SMAP SPL3SMP file and returns a clean xarray.DataArray with
    strictly monotonically ordered 1D lat/lon coords (no duplicates).
    """
    with h5py.File(path, "r") as f:
        soil = f["Soil_Moisture_Retrieval_Data_AM/soil_moisture"][:].astype(np.float32)
        lat2d = f["Soil_Moisture_Retrieval_Data_AM/latitude"][:]
        lon2d = f["Soil_Moisture_Retrieval_Data_AM/longitude"][:]

    # Convert native SMAP 2D lat/lon → 1D coords (lat varies only by row, lon only by col)
    lat_1d = lat2d[:, 0]
    lon_1d = lon2d[0, :]

    # Force correct orientation:
    # Lat: must be DESCENDING (90 → -90)
    # Lon: must be ASCENDING (-180 → 180)

    # Fix latitude orientation
    if lat_1d[0] < lat_1d[-1]:
        lat_1d = lat_1d[::-1]
        soil = soil[::-1, :]

    # Fix longitude orientation
    if lon_1d[0] > lon_1d[-1]:
        lon_1d = lon_1d[::-1]
        soil = soil[:, ::-1]

    # -------- FIX: Remove duplicate coordinate values --------
    lat_unique, lat_idx = np.unique(lat_1d, return_index=True)
    lon_unique, lon_idx = np.unique(lon_1d, return_index=True)

    # Subset soil grid to match unique coordinates
    soil_unique = soil[np.sort(lat_idx)][:, np.sort(lon_idx)]

    # Construct DataArray
    da = xr.DataArray(
        soil_unique,
        dims=("lat", "lon"),
        coords={"lat": lat_unique[np.argsort(lat_idx)],
                "lon": lon_unique[np.argsort(lon_idx)]},
        name="soil_moisture",
    )

    return da






def reproject_to_025deg(da):
    """
    Reproject to 0.25°×0.25° grid using xarray interp.
    Assumes da has unique sorted lat/lon coords.
    """
    da_interp = da.interp(
        lat=TARGET_LATS,
        lon=TARGET_LONS,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    return da_interp.values.astype(np.float32)





# ================================================================
# BUILD 21-DAY GLOBAL STACK
# ================================================================

def build_global_smap_21days():
    """
    Downloads and constructs (or reconstructs) the global 21-day stack.
    """
    print("[GLOBAL SMAP] Auto-building SMAP 21-day stack…")

    smap_files = download_last_21_smap()

    stack = []
    for path in smap_files:
        print(f"[SMAP] Reprojecting {os.path.basename(path)}")
        native = load_native_smap_file(path)
        interp = reproject_to_025deg(native)
        stack.append(interp)

    stack = np.stack(stack, axis=0)  # (21, 720, 1440)

    os.makedirs(os.path.dirname(SMAP_GLOBAL_NPY), exist_ok=True)
    np.save(SMAP_GLOBAL_NPY, stack)

    print(f"[GLOBAL SMAP] Saved → {SMAP_GLOBAL_NPY}")
    return stack
