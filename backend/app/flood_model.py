import os
import pickle
from datetime import datetime, timedelta, date

import numpy as np
import torch
import torch.nn as nn
import h5py
import xarray as xr
import earthaccess
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

from . import redis_conn  # optional if you ever want to cache to redis
from app.global_smap_loader import build_global_smap_21days, SMAP_GLOBAL_NPY



# =====================================================================
# CONFIG (adapt from flood_predictor (2).py)
# =====================================================================

SMAP_DAY_DELAY = 5
SMAP_N_DAYS = 21
GPM_DAY_DELAY = 1
GPM_N_DAYS = 4

SOIL_GRID_SIZE = 50
PRECIP_GRID_SIZE = 250

# Base directory = backend/ (one level above app/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Directories for temp data / cache inside backend/
GPM_DIR = os.path.join(BASE_DIR, "gpm_temp")
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

# Model + scaler files inside backend/
MODEL_PATH = os.path.join(BASE_DIR, "DualCNNLSTM_best_model_V3.pt")
SCALER_PATH = os.path.join(BASE_DIR, "DualCNNLSTM_scaler_V3.pkl")

os.makedirs(GPM_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Caching & parallelism (same as flood_predictor (2).py)
USE_CACHE = True
CACHE_DIR = DATA_CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_WORKERS = 4  # Number of parallel download threads


GLOBAL_SMAP = None
GLOBAL_TIMESTAMP = None

# =====================================================================
# MODEL ARCHITECTURE (same as in your script)
# =====================================================================

# =====================================================================
# MODEL ARCHITECTURE (same as in your script)
# =====================================================================

class DualCNNLSTM(nn.Module):
    """
    Updated model architecture matching training code V3.
    Key points:
    - Same CNN architecture for both precip and soil as used in training
    - Dropout rate: 0.5
    - LSTM hidden size: 64
    - AdaptiveAvgPool2d to (8, 8)
    """
    def __init__(self):
        super(DualCNNLSTM, self).__init__()

        # Precipitation CNN (250x250 input, 1 channel)
        self.cnn_precip = nn.Sequential(
            nn.Conv2d(1, 8, 5, 2, 2),           # 1 -> 8 channels, kernel 5x5
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(8, 16, 3, 2, 1),          # 8 -> 16 channels, kernel 3x3
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),       # -> (16, 8, 8)
            nn.Dropout(0.5),
        )

        # Soil moisture CNN (50x50 input, 2 channels: sm + mask)
        self.cnn_soil = nn.Sequential(
            nn.Conv2d(2, 8, 5, 2, 2),           # 2 -> 8 channels, **5x5** kernel (matches checkpoint)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(8, 16, 3, 2, 1),          # 8 -> 16 channels, 3x3 kernel
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),       # -> (16, 8, 8)
            nn.Dropout(0.5),
        )

        # LSTM: takes concatenated features from both CNNs
        # Input size: 2 * 16 * 8 * 8 = 2048
        self.lstm = nn.LSTM(
            input_size=2 * 16 * 8 * 8,
            hidden_size=64,
            batch_first=True,
        )
        self.dropout_lstm = nn.Dropout(0.5)

        # Final classification layer
        self.fc_flood = nn.Linear(64, 1)

    def forward(self, precip, soil):
        """
        Args:
            precip: (B, T_precip, 1, 250, 250) - Precipitation sequence
            soil:   (B, T_soil,   2,  50,  50) - Soil moisture sequence (sm + mask)

        Returns:
            flood_pred: (B, 1) - Flood probability (sigmoid)
        """
        B, Tp, C, H, W = precip.shape
        Ts = soil.shape[1]
        T = min(Tp, Ts)  # Use minimum sequence length

        lstm_in = []

        # Process each time step
        for t in range(T):
            # Precip at time t
            p = self.cnn_precip(precip[:, t])   # (B, 16, 8, 8)

            # Soil at time t
            s = self.cnn_soil(soil[:, t])       # (B, 16, 8, 8)

            # Concatenate and flatten
            combined = torch.cat(
                [p.view(B, -1), s.view(B, -1)],
                dim=1
            )  # (B, 2048)
            lstm_in.append(combined)

        # Stack over time: (B, T, 2048)
        lstm_in = torch.stack(lstm_in, dim=1)

        # LSTM forward
        lstm_out, _ = self.lstm(lstm_in)        # (B, T, 64)

        # Use last timestep
        lstm_last = self.dropout_lstm(lstm_out[:, -1, :])  # (B, 64)

        # Classification
        flood_pred = torch.sigmoid(self.fc_flood(lstm_last))  # (B, 1)

        return flood_pred

# =====================================================================
# SINGLETON MODEL + SCALER
# =====================================================================

_model = None
_scaler = None

def get_model_and_scaler():
    global _model, _scaler
    if _model is None or _scaler is None:
        # Load scaler
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)

        # Load model
        m = DualCNNLSTM().to(device)
        state = torch.load(MODEL_PATH, map_location=device)
        m.load_state_dict(state)
        m.eval()
        _model = m

    return _model, _scaler

# =====================================================================
# SMAP: ensure_global_data_loaded (same as script Option B)
# =====================================================================

def load_global_smap():
    """
    Loads cached global SMAP stack if present, or builds it by
    auto-downloading 21 days of SPL3SMP via Earthaccess (Option B).
    """
    global GLOBAL_SMAP
    if os.path.exists(SMAP_GLOBAL_NPY):
        GLOBAL_SMAP = np.load(SMAP_GLOBAL_NPY)
        print(f"[GLOBAL SMAP] Loaded cached stack: {GLOBAL_SMAP.shape}")
    else:
        print("[GLOBAL SMAP] Cache missing — auto-building 21-day stack…")
        GLOBAL_SMAP = build_global_smap_21days()  # your Option B builder
        print(f"[GLOBAL SMAP] Built stack: {GLOBAL_SMAP.shape}")

def ensure_global_data_loaded():
    """
    Reload global SMAP once per UTC day (so files auto-update daily).
    """
    global GLOBAL_TIMESTAMP
    now = datetime.utcnow()

    if GLOBAL_TIMESTAMP is None or (now - GLOBAL_TIMESTAMP).total_seconds() > 86400:
        print("[GLOBAL CACHE] Reloading global SMAP…")
        load_global_smap()
        GLOBAL_TIMESTAMP = now

# =====================================================================
# GPM + preprocessing (paste from script)
# =====================================================================

# - get_cache_key / load_from_cache / save_to_cache
# - search_and_download_gpm / read_gpm_precip_grid / extract_precip_window
# - accumulate_3hr_grids / process_gpm_file / get_gpm_4days_grids
# - prepare_inference_data(location, scaler)
#
# Paste them here, **unchanged**, except:
#   - Replace SMAP_N_DAYS / GPM_N_DAYS with module-level constants above
#   - In prepare_inference_data() call ensure_global_data_loaded() as in your script
#   - Keep the “scale with scaler” and soil-mask logic exactly as you already have

...
# =====================================================================
# CACHING UTILITIES (SPEED OPTIMIZATION)
# =====================================================================

def get_cache_key(data_type, target_date, lat=None, lon=None):
    """Generate unique cache key for data."""
    if lat is not None and lon is not None:
        key_str = f"{data_type}_{target_date}_{lat}_{lon}"
    else:
        key_str = f"{data_type}_{target_date}"
    return hashlib.md5(key_str.encode()).hexdigest()

def load_from_cache(cache_key):
    """Load data from cache if available."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path, allow_pickle=True)
        except:
            return None
    return None

def save_to_cache(cache_key, data):
    """Save data to cache."""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    try:
        np.save(cache_path, data)
    except Exception as e:
        print(f"[CACHE] Failed to save: {e}")

# =====================================================================
# OPTIMIZED SMAP FUNCTIONS WITH BATCH QUERIES
# =====================================================================

# ============================================================
# GLOBAL SMAP LOADING (Option A)
# ============================================================

def load_global_smap():
    """
    Loads the global 21-day SMAP stack (720 × 1440 × 21)
    from disk, or builds it if missing.
    """
    global GLOBAL_SMAP

    if os.path.exists(SMAP_GLOBAL_NPY):
        print("[GLOBAL SMAP] Loading cached global SMAP stack...")
        GLOBAL_SMAP = np.load(SMAP_GLOBAL_NPY)
    else:
        print("[GLOBAL SMAP] Cache missing — building global SMAP...")
        GLOBAL_SMAP = build_global_smap_21days()

    print(f"[GLOBAL SMAP] Loaded. Shape: {GLOBAL_SMAP.shape}")

def ensure_global_data_loaded():
    """
    Reload global SMAP/IMERG once per day.
    """
    global GLOBAL_TIMESTAMP
    now = datetime.utcnow()

    # First load or stale (>1 day)
    if GLOBAL_TIMESTAMP is None or (now - GLOBAL_TIMESTAMP).total_seconds() > 86400:
        print("[GLOBAL CACHE] Reloading global SMAP (and IMERG in future)...")
        load_global_smap()
        GLOBAL_TIMESTAMP = now


# =====================================================================
# GPM FUNCTIONS (OPTIMIZED WITH CACHING AND PARALLEL PROCESSING)
# =====================================================================

def earthdata_login():
    """
    Ensure we are logged in to Earthdata using the _netrc file.
    """
    print("[GPM] Attempting Earthdata login via netrc…")
    auth = earthaccess.login(strategy="netrc")

    if auth is None or not getattr(auth, "authenticated", False):
        print("[GPM] Earthdata login failed or not authenticated.")
        raise RuntimeError("Earthdata Login failed for GPM (netrc).")

    print("[GPM] Earthdata login OK.")
    return auth
def search_and_download_gpm(gpm_date: date, out_dir: str):
    """Download GPM with regional bounding box."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    start = gpm_date.strftime("%Y-%m-%dT00:00:00")
    end = gpm_date.strftime("%Y-%m-%dT23:59:59")

    # Southeast Asia bounding box for faster search
    bbox = (90, -10, 150, 30)

    results = earthaccess.search_data(
        short_name="GPM_3IMERGHHL",
        version="07",
        temporal=(start, end),
        bounding_box=bbox,
    )

    paths = earthaccess.download(results, local_path=out_dir)

    precip_files = [
        str(fp) for fp in paths
        if ("3B-HHR" in str(fp)) and str(fp).lower().endswith(".hdf5")
    ]

    return sorted(precip_files)

def read_gpm_precip_grid(filepath: str):
    with h5py.File(filepath, "r") as f:
        arr = f["Grid"]["precipitation"][:]
        lats = f["Grid"]["lat"][:]
        lons = f["Grid"]["lon"][:]

    arr = arr[0].T
    arr = np.where(arr < 0, 0.0, arr)
    precip_30min = arr * 0.5

    da = xr.DataArray(
        precip_30min,
        dims=["lat", "lon"],
        coords={"lat": lats, "lon": lons}
    )
    return da

def extract_precip_window(precip_da: xr.DataArray, target_lat: float, target_lon: float, size=250):
    """Extract 250x250 patch from GPM precipitation grid."""
    lat_vals = precip_da.lat.values
    lon_vals = precip_da.lon.values

    iy = int(np.argmin(np.abs(lat_vals - target_lat)))
    ix = int(np.argmin(np.abs(lon_vals - target_lon)))
    half = size // 2

    y0 = iy - half
    y1 = iy + half
    x0 = ix - half
    x1 = ix + half

    out = np.zeros((size, size), dtype=float)

    src_y0 = max(0, y0)
    src_y1 = min(len(lat_vals), y1)
    src_x0 = max(0, x0)
    src_x1 = min(len(lon_vals), x1)

    tgt_y0 = src_y0 - y0
    tgt_y1 = tgt_y0 + (src_y1 - src_y0)
    tgt_x0 = src_x0 - x0
    tgt_x1 = tgt_x0 + (src_x1 - src_x0)

    try:
        src_slice = precip_da.isel(lat=slice(src_y0, src_y1), lon=slice(src_x0, src_x1)).values
        out[int(tgt_y0):int(tgt_y1), int(tgt_x0):int(tgt_x1)] = src_slice
    except Exception as e:
        print(f"[GPM] Warning: {e}")

    return out

def accumulate_3hr_grids(frames: list):
    """Accumulate 30-min grids into 3-hour blocks."""
    blocks = []
    for i in range(0, len(frames), 6):
        chunk = frames[i:i+6]
        if len(chunk) == 6:
            blocks.append(np.sum(chunk, axis=0))
    return blocks

def process_gpm_file(fp, location):
    """Process single GPM file (for parallel processing)."""
    try:
        precip_da = read_gpm_precip_grid(fp)
        patch = extract_precip_window(precip_da, location["lat"], location["lon"], size=PRECIP_GRID_SIZE)
        return patch
    except Exception as e:
        print(f"[GPM] Error processing {fp}: {e}")
        return None

def get_gpm_4days_grids(location, day_delay=1, n_days=4):
    """Download and process GPM with caching and parallel processing."""
    # Check cache first
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).date()
    most_recent_available = today - timedelta(days=day_delay + 1)

    if USE_CACHE:
        cache_key = get_cache_key("gpm", most_recent_available, location["lat"], location["lon"])
        cached_data = load_from_cache(cache_key)
        if cached_data is not None and len(cached_data) >= 32:
            print(f"[GPM CACHE HIT] for {location['name']}")
            return cached_data[:32]

    raw_dates = [most_recent_available - timedelta(days=i) for i in range(n_days)]
    date_list = sorted(raw_dates)

    print(f"\n[GPM] Processing for {location['name']}")
    print(f"[GPM] Date range: {date_list[0]} to {date_list[-1]}")

    all_blocks = []

    for d in date_list:
        files = search_and_download_gpm(d, GPM_DIR)

        # Parallel processing of files
        frames = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {executor.submit(process_gpm_file, fp, location): fp for fp in files}
            for future in as_completed(future_to_file):
                result = future.result()
                if result is not None:
                    frames.append(result)

        if len(frames) == 0:
            print(f"[GPM] No frames for {d}")
            continue

        blocks_3hr = accumulate_3hr_grids(frames)
        all_blocks.extend(blocks_3hr)

    result = np.array(all_blocks)[:32]  # Ensure exactly 32

    # Save to cache
    if USE_CACHE and len(result) >= 32:
        save_to_cache(cache_key, result)

    return result

# =====================================================================
# DATA PROCESSING FOR INFERENCE (OPTIMIZED)
# =====================================================================

def prepare_inference_data(location, scaler):
    """
    Prepare data for one location (optimized with batch queries and caching).

    CRITICAL: Must match training preprocessing exactly:
    1. Fill NaNs with 0 BEFORE scaling
    2. Apply same scaler to both precip and soil
    3. Create mask channel for soil (original NaN locations)
    """
    print(f"\n{'='*60}")
    print(f"Processing location: {location['name']} ({location['lat']}, {location['lon']})")
    print(f"{'='*60}")

    # Get SMAP data (21 days) using BATCH QUERIES
    # --- NEW OPTION A SMAP EXTRACTION ---
    ensure_global_data_loaded()

    lat = location["lat"]
    lon = location["lon"]

    lat_vals = np.linspace(90, -90, GLOBAL_SMAP.shape[1])  # 720
    lon_vals = np.linspace(-180, 180, GLOBAL_SMAP.shape[2])  # 1440

    # Find indices in the 0.25° grid
    iy = int(np.argmin(np.abs(lat_vals - lat)))
    ix = int(np.argmin(np.abs(lon_vals - lon)))

    half = SOIL_GRID_SIZE // 2

    y0 = iy - half
    y1 = iy + half
    x0 = ix - half
    x1 = ix + half

    soil_grids = []
    for day in range(SMAP_N_DAYS):
        sm = GLOBAL_SMAP[day]

        # safe window extraction
        window = np.zeros((SOIL_GRID_SIZE, SOIL_GRID_SIZE), dtype=np.float32)

        src = sm[
            max(0, y0):min(720, y1),
            max(0, x0):min(1440, x1)
        ]

        dy0 = max(0, -y0)
        dx0 = max(0, -x0)

        window[
            dy0:dy0 + src.shape[0],
            dx0:dx0 + src.shape[1]
        ] = src

        soil_grids.append(window)

    # Get GPM data (32 time steps from 4 days)
    print(f"\n[GPM] Extracting 4 days × 8 blocks = 32 time steps")
    precip_grids = get_gpm_4days_grids(location, day_delay=GPM_DAY_DELAY, n_days=GPM_N_DAYS)

    if len(precip_grids) < 32:
        print(f"[WARNING] Only got {len(precip_grids)}/32 precipitation time steps")
        return None

    # Process precipitation (apply scaler)
    precip_seq = []
    for grid in precip_grids[:32]:  # Ensure exactly 32
        scaled = scaler.transform(grid.flatten().reshape(-1, 1)).reshape(grid.shape)
        precip_seq.append(scaled[np.newaxis, ...])  # Add channel dimension

    # Process soil moisture (CRITICAL: must match training!)
    # Training does: 1) Create mask from original, 2) Fill NaNs with 0, 3) Scale, 4) Stack with mask
    soil_seq = []
    for grid in soil_grids[:21]:  # Ensure exactly 21
        # Step 1: Create mask from original (True = valid data)
        mask = ~np.isnan(grid)

        # Step 2: Fill NaNs with 0 BEFORE scaling (matches training!)
        grid_filled = np.nan_to_num(grid, nan=0.0)

        # Step 3: Apply scaler
        scaled_sm = scaler.transform(grid_filled.flatten().reshape(-1, 1)).reshape(grid.shape)

        # Step 4: Stack scaled soil + mask as 2 channels
        stacked = np.stack([scaled_sm, mask.astype(float)], axis=0)
        soil_seq.append(stacked)

    # Convert to tensors
    precip_tensor = torch.tensor(np.stack(precip_seq), dtype=torch.float32)  # (32, 1, 250, 250)
    soil_tensor = torch.tensor(np.stack(soil_seq), dtype=torch.float32)      # (21, 2, 50, 50)

    return precip_tensor, soil_tensor
# =====================================================================
# 56-POINT RADIAL GRID AROUND A CENTER
# =====================================================================

import math

EARTH_R_KM = 6371.0
RINGS_KM = [5, 10, 20, 30, 40, 60, 80]   # 7 rings
BEARINGS_DEG = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 directions
# 7 * 8 = 56 points

def offset_latlon(lat_deg, lon_deg, distance_km, bearing_deg):
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    d_over_r = distance_km / EARTH_R_KM

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d_over_r)
        + math.cos(lat1) * math.sin(d_over_r) * math.cos(brng)
    )
    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(d_over_r) * math.cos(lat1),
        math.cos(d_over_r) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180  # wrap lon

def build_radial_locations(center_lat, center_lon):
    locations = []
    idx = 0
    for r in RINGS_KM:
        for b in BEARINGS_DEG:
            lat, lon = offset_latlon(center_lat, center_lon, r, b)
            locations.append({
                "name": f"pt_{idx}_r{r}_b{b}",
                "lat": lat,
                "lon": lon,
            })
            idx += 1
    return locations  # length 56

# =====================================================================
# PUBLIC ENTRYPOINT USED BY RQ WORKER
# =====================================================================

def run_flood_inference(center_lat, center_lon):
    """
    Run the DualCNNLSTM model on a 56-point radial grid around (center_lat, center_lon).

    Returns:
        {
          "center": {"lat": ..., "lon": ...},
          "points": [
             {"lat": ..., "lon": ..., "prob": ..., "risk": "low|medium|high"},
             ...
          ]
        }
    """
    model, scaler = get_model_and_scaler()
    ensure_global_data_loaded()

    locations = build_radial_locations(center_lat, center_lon)

    batch_precip = []
    batch_soil = []
    valid_locs = []

    for loc in locations:
        result = prepare_inference_data(loc, scaler)
        if result is not None:
            p, s = result
            batch_precip.append(p)
            batch_soil.append(s)
            valid_locs.append(loc)

    if not batch_precip:
        return {
            "center": {"lat": center_lat, "lon": center_lon},
            "points": [],
            "error": "No valid data extracted"
        }

    precip_batch = torch.stack(batch_precip).to(device)  # (N, 32, 1, 250, 250)
    soil_batch   = torch.stack(batch_soil).to(device)    # (N, 21, 2, 50, 50)

    with torch.no_grad():
        preds = model(precip_batch, soil_batch).cpu().numpy().reshape(-1)

    points = []
    for loc, prob in zip(valid_locs, preds):
        p = float(prob)  # 0..1
        if p >= 0.5:
            risk = "high"
        elif p >= 0.3:
            risk = "medium"
        else:
            risk = "low"
        points.append({
            "lat": loc["lat"],
            "lon": loc["lon"],
            "prob": p,
            "risk": risk,
        })

    return {
        "center": {"lat": center_lat, "lon": center_lon},
        "points": points,
    }
