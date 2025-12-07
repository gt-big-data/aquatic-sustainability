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

from . import redis_conn
from app.global_smap_loader import build_global_smap_21days, SMAP_GLOBAL_NPY
from app.global_gpm_loader import build_global_gpm_4days

# =====================================================================
# CONFIG
# =====================================================================

SMAP_DAY_DELAY = 5
SMAP_N_DAYS = 21
GPM_DAY_DELAY = 2  # Skip today + yesterday to ensure complete data
GPM_N_DAYS = 4

SOIL_GRID_SIZE = 50
PRECIP_GRID_SIZE = 250

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GPM_DIR = os.path.join(BASE_DIR, "gpm_temp")
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "DualCNNLSTM_best_model_V3.pt")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "DualCNNLSTM_scaler_V3.pkl")

os.makedirs(GPM_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CACHE = True
CACHE_DIR = DATA_CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_WORKERS = 4

# Global precipitation cache path
GPM_GLOBAL_NPY = os.path.join(DATA_CACHE_DIR, "global_gpm_4days.npy")

GLOBAL_SMAP = None
GLOBAL_GPM = None
GLOBAL_TIMESTAMP = None

# =====================================================================
# MODEL ARCHITECTURE
# =====================================================================

class DualCNNLSTM(nn.Module):
    def __init__(self):
        super(DualCNNLSTM, self).__init__()

        self.cnn_precip = nn.Sequential(
            nn.Conv2d(1, 8, 5, 2, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Dropout(0.5),
        )

        self.cnn_soil = nn.Sequential(
            nn.Conv2d(2, 8, 5, 2, 2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Dropout(0.5),
        )

        self.lstm = nn.LSTM(
            input_size=2 * 16 * 8 * 8,
            hidden_size=64,
            batch_first=True,
        )
        self.dropout_lstm = nn.Dropout(0.5)

        self.fc_flood = nn.Linear(64, 1)

    def forward(self, precip, soil):
        B, Tp, C, H, W = precip.shape
        Ts = soil.shape[1]
        T = min(Tp, Ts)

        lstm_in = []

        for t in range(T):
            p = self.cnn_precip(precip[:, t])
            s = self.cnn_soil(soil[:, t])

            combined = torch.cat(
                [p.view(B, -1), s.view(B, -1)],
                dim=1
            )
            lstm_in.append(combined)

        lstm_in = torch.stack(lstm_in, dim=1)

        lstm_out, _ = self.lstm(lstm_in)

        lstm_last = self.dropout_lstm(lstm_out[:, -1, :])

        flood_pred = torch.sigmoid(self.fc_flood(lstm_last))

        return flood_pred

# =====================================================================
# SINGLETON MODEL + SCALER
# =====================================================================

_model = None
_scaler = None

def get_model_and_scaler():
    global _model, _scaler
    if _model is None or _scaler is None:
        with open(SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)

        m = DualCNNLSTM().to(device)
        state = torch.load(MODEL_PATH, map_location=device)
        m.load_state_dict(state)
        m.eval()
        _model = m

    return _model, _scaler

# =====================================================================
# GLOBAL DATA LOADERS
# =====================================================================

def load_global_smap():
    global GLOBAL_SMAP
    if os.path.exists(SMAP_GLOBAL_NPY):
        GLOBAL_SMAP = np.load(SMAP_GLOBAL_NPY)
        print(f"[GLOBAL SMAP] Loaded cached stack: {GLOBAL_SMAP.shape}")
    else:
        print("[GLOBAL SMAP] Cache missing — auto-building 21-day stack…")
        GLOBAL_SMAP = build_global_smap_21days()
        print(f"[GLOBAL SMAP] Built stack: {GLOBAL_SMAP.shape}")

def load_global_gpm():
    """
    Load or build global GPM precipitation grids for the last 4 days.
    Each day has 8 3-hour blocks, resulting in 32 time steps.
    Grid resolution: 0.1° (1800 x 3600)
    """
    global GLOBAL_GPM
    
    if os.path.exists(GPM_GLOBAL_NPY):
        GLOBAL_GPM = np.load(GPM_GLOBAL_NPY)
        print(f"[GLOBAL GPM] Loaded cached stack: {GLOBAL_GPM.shape}")
    else:
        print("[GLOBAL GPM] Cache missing — building 4-day global stack…")
        GLOBAL_GPM = build_global_gpm_4days(day_delay=GPM_DAY_DELAY)  # Pass the delay parameter
        print(f"[GLOBAL GPM] Built stack: {GLOBAL_GPM.shape}")

def build_global_gpm_4days(day_delay=GPM_DAY_DELAY):
    """
    Download and build global GPM precipitation grids for 4 days.
    
    Args:
        day_delay: Number of days to go back from today (uses GPM_DAY_DELAY constant by default)
    
    Returns shape: (32, 1800, 3600) for 32 time steps at 0.1° resolution
    """
    earthdata_login()
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).date()
    # Use day_delay parameter to ensure consistency
    most_recent_available = today - timedelta(days=day_delay)
    
    raw_dates = [most_recent_available - timedelta(days=i) for i in range(GPM_N_DAYS)]
    date_list = sorted(raw_dates)
    
    print(f"[GLOBAL GPM] Building for dates: {date_list[0]} to {date_list[-1]}")
    
    all_blocks = []
    
    for d in date_list:
        print(f"[GLOBAL GPM] Processing date: {d}")
        files = search_and_download_gpm_global(d, GPM_DIR)
        
        frames = []
        for fp in files:
            try:
                precip_da = read_gpm_precip_grid(fp)
                frames.append(precip_da.values)
            except Exception as e:
                print(f"[GLOBAL GPM] Error processing {fp}: {e}")
                continue
        
        if len(frames) == 0:
            print(f"[GLOBAL GPM] No frames for {d}")
            # Fill with zeros
            for _ in range(8):
                all_blocks.append(np.zeros((1800, 3600), dtype=np.float32))
            continue
        
        blocks_3hr = accumulate_3hr_grids_global(frames)
        all_blocks.extend(blocks_3hr)
    
    stack = np.array(all_blocks[:32])  # Ensure exactly 32
    
    # Pad if necessary
    if len(stack) < 32:
        padding = np.zeros((32 - len(stack), 1800, 3600), dtype=np.float32)
        stack = np.vstack([stack, padding])
    
    os.makedirs(os.path.dirname(GPM_GLOBAL_NPY), exist_ok=True)
    np.save(GPM_GLOBAL_NPY, stack)
    print(f"[GLOBAL GPM] Saved → {GPM_GLOBAL_NPY}")
    
    return stack

def accumulate_3hr_grids_global(frames):
    """Accumulate 30-min grids into 3-hour blocks for global grids."""
    blocks = []
    for i in range(0, len(frames), 6):
        chunk = frames[i:i+6]
        if len(chunk) == 6:
            blocks.append(np.sum(chunk, axis=0))
        elif len(chunk) > 0:
            # Partial block, still sum what we have
            blocks.append(np.sum(chunk, axis=0))
    
    # Ensure we have 8 blocks per day
    while len(blocks) < 8:
        blocks.append(np.zeros_like(blocks[0]) if blocks else np.zeros((1800, 3600), dtype=np.float32))
    
    return blocks[:8]

def ensure_global_data_loaded():
    """
    Reload global SMAP/GPM once per UTC day (so files auto-update daily).
    """
    global GLOBAL_TIMESTAMP
    now = datetime.utcnow()

    if GLOBAL_TIMESTAMP is None or (now - GLOBAL_TIMESTAMP).total_seconds() > 86400:
        print("[GLOBAL CACHE] Reloading global SMAP and GPM…")
        load_global_smap()
        load_global_gpm()
        GLOBAL_TIMESTAMP = now

# =====================================================================
# GPM FUNCTIONS
# =====================================================================

def earthdata_login():
    # Import here to avoid circular imports
    from app.global_gpm_loader import ensure_earthdata_auth
    
    # Ensure .netrc file exists with credentials
    ensure_earthdata_auth()
    
    print("[GPM] Attempting Earthdata login via netrc…")
    auth = earthaccess.login(strategy="netrc")

    if auth is None or not getattr(auth, "authenticated", False):
        print("[GPM] Earthdata login failed or not authenticated.")
        raise RuntimeError("Earthdata Login failed for GPM (netrc).")

    print("[GPM] Earthdata login OK.")
    return auth

def search_and_download_gpm_global(gpm_date: date, out_dir: str):
    """Download GPM without regional bounding box for global coverage."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    start = gpm_date.strftime("%Y-%m-%dT00:00:00")
    end = gpm_date.strftime("%Y-%m-%dT23:59:59")

    results = earthaccess.search_data(
        short_name="GPM_3IMERGHHL",
        version="07",
        temporal=(start, end),
    )

    paths = earthaccess.download(results, local_path=out_dir)

    precip_files = [
        str(fp) for fp in paths
        if ("3B-HHR" in str(fp)) and str(fp).lower().endswith(".hdf5")
    ]

    return sorted(precip_files)

def read_gpm_precip_grid(filepath: str):
    """Read GPM precipitation grid as global DataArray."""
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

def extract_precip_from_global(lat: float, lon: float, size=250):
    """
    Extract precipitation windows from GLOBAL_GPM for a specific location.
    Returns array of shape (32, 250, 250)
    """
    global GLOBAL_GPM
    
    if GLOBAL_GPM is None:
        raise RuntimeError("GLOBAL_GPM not loaded")
    
    # GPM grid: 0.1° resolution, 1800 x 3600
    # Latitude: 90 to -90 (descending)
    # Longitude: -180 to 180 (ascending)
    
    lat_vals = np.linspace(90, -90, 1800)
    lon_vals = np.linspace(-180, 180, 3600)
    
    # Find indices in the 0.1° grid
    iy = int(np.argmin(np.abs(lat_vals - lat)))
    ix = int(np.argmin(np.abs(lon_vals - lon)))
    
    half = size // 2
    
    y0 = iy - half
    y1 = iy + half
    x0 = ix - half
    x1 = ix + half
    
    precip_grids = []
    
    for t in range(32):  # 32 time steps
        window = np.zeros((size, size), dtype=np.float32)
        
        # Extract from global grid
        src_y0 = max(0, y0)
        src_y1 = min(1800, y1)
        src_x0 = max(0, x0)
        src_x1 = min(3600, x1)
        
        # Target coordinates in output window
        tgt_y0 = max(0, -y0)
        tgt_y1 = tgt_y0 + (src_y1 - src_y0)
        tgt_x0 = max(0, -x0)
        tgt_x1 = tgt_x0 + (src_x1 - src_x0)
        
        src = GLOBAL_GPM[t, src_y0:src_y1, src_x0:src_x1]
        window[tgt_y0:tgt_y1, tgt_x0:tgt_x1] = src
        
        precip_grids.append(window)
    
    return np.array(precip_grids)

# =====================================================================
# DATA PROCESSING FOR INFERENCE
# =====================================================================

def prepare_inference_data(location, scaler):
    """
    Prepare data for one location using global cached grids.
    """
    print(f"\n{'='*60}")
    print(f"Processing location: {location['name']} ({location['lat']}, {location['lon']})")
    print(f"{'='*60}")

    ensure_global_data_loaded()

    lat = location["lat"]
    lon = location["lon"]

    # Extract SMAP data from global cache
    lat_vals = np.linspace(90, -90, GLOBAL_SMAP.shape[1])
    lon_vals = np.linspace(-180, 180, GLOBAL_SMAP.shape[2])

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

    # Extract GPM data from global cache
    print(f"[GPM] Extracting from global cache")
    precip_grids = extract_precip_from_global(lat, lon, size=PRECIP_GRID_SIZE)

    if len(precip_grids) < 32:
        print(f"[WARNING] Only got {len(precip_grids)}/32 precipitation time steps")
        return None

    # Process precipitation (apply scaler)
    precip_seq = []
    for grid in precip_grids[:32]:
        scaled = scaler.transform(grid.flatten().reshape(-1, 1)).reshape(grid.shape)
        precip_seq.append(scaled[np.newaxis, ...])

    # Process soil moisture
    soil_seq = []
    for grid in soil_grids[:21]:
        mask = ~np.isnan(grid)
        grid_filled = np.nan_to_num(grid, nan=0.0)
        scaled_sm = scaler.transform(grid_filled.flatten().reshape(-1, 1)).reshape(grid.shape)
        stacked = np.stack([scaled_sm, mask.astype(float)], axis=0)
        soil_seq.append(stacked)

    precip_tensor = torch.tensor(np.stack(precip_seq), dtype=torch.float32)
    soil_tensor = torch.tensor(np.stack(soil_seq), dtype=torch.float32)

    return precip_tensor, soil_tensor

# =====================================================================
# RADIAL GRID GENERATION
# =====================================================================

import math

EARTH_R_KM = 6371.0
RINGS_KM = [5, 10, 20, 30, 40, 60, 80]
BEARINGS_DEG = [0, 45, 90, 135, 180, 225, 270, 315]

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

    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180

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
    return locations

# =====================================================================
# INFERENCE ENTRYPOINT
# =====================================================================

def run_flood_inference(center_lat, center_lon):
    """
    Run the DualCNNLSTM model on a 56-point radial grid around (center_lat, center_lon).
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

    precip_batch = torch.stack(batch_precip).to(device)
    soil_batch   = torch.stack(batch_soil).to(device)

    with torch.no_grad():
        preds = model(precip_batch, soil_batch).cpu().numpy().reshape(-1)

    points = []
    for loc, prob in zip(valid_locs, preds):
        p = float(prob)
        if p >= 0.75:
            risk = "high"
        elif p >= 0.5:
            risk = "warning"
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