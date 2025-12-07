# global_gpm_loader.py
# Automatic downloader for GPM IMERG (4 days, 32 time steps)

import os
import h5py
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
import earthaccess

# Output cache
GPM_GLOBAL_NPY = "./data/global_gpm_4days.npy"

# Target grid for GPM: 0.1° resolution
# GPM native: 1800 lats × 3600 lons (90°N to -90°S, -180°W to 180°E)
TARGET_LATS = np.linspace(90, -90, 1800)       # descending
TARGET_LONS = np.linspace(-180, 180, 3600)     # ascending


# ================================================================
# AUTO-DOWNLOAD GPM USING earthaccess
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

def download_last_4days_gpm(out_dir="./data/gpm_download/"):
    """
    Downloads the last 4 days of GPM 3IMERGHHL files from NASA.
    Returns a sorted list of downloaded file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Ensure .netrc file is created with Earthdata credentials
    ensure_earthdata_auth()

    print("[AUTH] Logged in to Earthdata")

    today = datetime.utcnow().date()

    # GPM is usually available with 1 day delay
    latest_available = today - timedelta(days=2)

    print(f"[GPM] Latest available assumed: {latest_available}")

    # Query the last 4 days
    start = latest_available - timedelta(days=3)
    end = latest_available

    print(f"[GPM] Querying GPM_3IMERGHHL from {start} to {end}")

    results = earthaccess.search_data(
        short_name="GPM_3IMERGHHL",
        version="07",
        temporal=(str(start), str(end)),
    )

    print(f"[GPM] Found {len(results)} files.")

    # Download
    print("[GPM] Downloading files…")
    paths = earthaccess.download(results, local_path=out_dir)

    # Filter to .hdf5
    gpm_files = [
        str(p) for p in paths if str(p).lower().endswith(".hdf5") and "3B-HHR" in str(p)
    ]

    print(f"[GPM] Using {len(gpm_files)} GPM files.")

    return sorted(gpm_files)


# ================================================================
# LOADING GPM FILES
# ================================================================

def load_gpm_file(path):
    """
    Loads a GPM 3IMERGHHL file and returns precipitation as numpy array.
    Returns shape: (1800, 3600) at 0.1° resolution
    """
    with h5py.File(path, "r") as f:
        arr = f["Grid"]["precipitation"][:]
        lat = f["Grid"]["lat"][:]
        lon = f["Grid"]["lon"][:]

    # Convert to 2D (remove time dimension)
    arr = arr[0].T
    
    # Set negative values (missing data) to 0
    arr = np.where(arr < 0, 0.0, arr)
    
    # Convert from mm/hr to mm per 30-min
    precip_30min = arr * 0.5

    return precip_30min.astype(np.float32)


# ================================================================
# BUILD 4-DAY GLOBAL STACK
# ================================================================

def build_global_gpm_4days():
    """
    Downloads and constructs the global 4-day stack.
    Returns shape: (32, 1800, 3600) - 32 time steps of 3-hour blocks
    """
    print("[GLOBAL GPM] Auto-building GPM 4-day stack…")

    gpm_files = download_last_4days_gpm()

    # Group files by date
    from collections import defaultdict
    files_by_date = defaultdict(list)
    
    for path in gpm_files:
        # Extract date from filename (format: 3B-HHR.MS.MRG.3IMERG.YYYYMMDD-SHHMMSS-EHHMMSS.mmmm.V07B.HDF5)
        basename = os.path.basename(path)
        try:
            date_str = basename.split('.')[4].split('-')[0]  # YYYYMMDD
            file_date = datetime.strptime(date_str, "%Y%m%d").date()
            files_by_date[file_date].append(path)
        except Exception as e:
            print(f"[GPM] Warning: Could not parse date from {basename}: {e}")
            continue

    # Sort dates
    sorted_dates = sorted(files_by_date.keys(), reverse=True)[:4]  # Last 4 days
    sorted_dates = sorted(sorted_dates)  # Chronological order

    print(f"[GPM] Processing dates: {sorted_dates}")

    all_blocks = []

    for date in sorted_dates:
        print(f"[GPM] Processing {date}")
        files = sorted(files_by_date[date])

        # Load all 30-min frames for this day
        frames = []
        for path in files:
            try:
                precip = load_gpm_file(path)
                frames.append(precip)
            except Exception as e:
                print(f"[GPM] Error loading {path}: {e}")
                continue

        if len(frames) == 0:
            print(f"[GPM] No frames for {date}, filling with zeros")
            for _ in range(8):
                all_blocks.append(np.zeros((1800, 3600), dtype=np.float32))
            continue

        # Accumulate into 3-hour blocks (6 frames per block)
        blocks_3hr = []
        for i in range(0, len(frames), 6):
            chunk = frames[i:i+6]
            if len(chunk) == 6:
                blocks_3hr.append(np.sum(chunk, axis=0))
            elif len(chunk) > 0:
                # Partial block
                blocks_3hr.append(np.sum(chunk, axis=0))

        # Ensure 8 blocks per day
        while len(blocks_3hr) < 8:
            blocks_3hr.append(np.zeros((1800, 3600), dtype=np.float32))

        all_blocks.extend(blocks_3hr[:8])

    # Ensure exactly 32 time steps
    stack = np.array(all_blocks[:32])

    if len(stack) < 32:
        padding = np.zeros((32 - len(stack), 1800, 3600), dtype=np.float32)
        stack = np.vstack([stack, padding])

    # Save to disk
    os.makedirs(os.path.dirname(GPM_GLOBAL_NPY), exist_ok=True)
    np.save(GPM_GLOBAL_NPY, stack)

    print(f"[GLOBAL GPM] Saved → {GPM_GLOBAL_NPY}")
    print(f"[GLOBAL GPM] Final shape: {stack.shape}")

    return stack