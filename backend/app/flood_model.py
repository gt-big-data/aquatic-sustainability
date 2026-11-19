import os
import pickle
import json
import math
from datetime import datetime, timedelta, date

import numpy as np
import torch
import torch.nn as nn
import h5py
import xarray as xr
import earthaccess
import h3  # Uber H3 hexagonal indexing (pip install "h3>=4.0")
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import threading

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

RUNNING_ON_CLOUD_RUN = bool(os.environ.get("K_SERVICE"))
FAST_START_FLOOD_DATA = os.environ.get(
    "FAST_START_FLOOD_DATA",
    "1" if RUNNING_ON_CLOUD_RUN else "0",
).lower() in {"1", "true", "yes", "on"}

# Global precipitation cache path (legacy, built on dev machines)
GPM_GLOBAL_NPY = os.path.join(DATA_CACHE_DIR, "global_gpm_4days.npy")

# Cropped regional inputs built daily by scripts/build_flood_data.py. On Cloud
# Run they are fetched from gs://$FLOOD_DATA_BUCKET/$FLOOD_DATA_BLOB.
REGION_NPZ = os.path.join(DATA_CACHE_DIR, "flood_region.npz")
FLOOD_DATA_BUCKET = os.environ.get("FLOOD_DATA_BUCKET", "")
FLOOD_DATA_BLOB = os.environ.get("FLOOD_DATA_BLOB", "flood/region_latest.npz")

# Global grid coordinates the model's window maths is written against.
# Row 0 is the NORTH edge. Raw GPM files are stored south-first, so GPM data is
# flipped into this orientation when it is built (see orient_gpm_frame).
_SMAP_LAT_VALS = np.linspace(90, -90, 720)
_SMAP_LON_VALS = np.linspace(-180, 180, 1440)
_GPM_LAT_VALS  = np.linspace(90, -90, 1800)
_GPM_LON_VALS  = np.linspace(-180, 180, 3600)

# ---- Region the flood map serves (Southeast Asia) ----
# Data is cropped to this box plus the model's input window, so the server
# holds ~80 MB instead of ~1 GB of global grids. Points outside it still run,
# but their input windows are zero-filled beyond the crop.
REGION_LAT_MIN, REGION_LAT_MAX = -15.0, 30.0
REGION_LON_MIN, REGION_LON_MAX = 90.0, 145.0


def _crop_box(lat_vals, lon_vals, window):
    """Global (row0, row1, col0, col1) covering REGION plus half a window."""
    half = window // 2
    rows = sorted(int(np.argmin(np.abs(lat_vals - v))) for v in (REGION_LAT_MIN, REGION_LAT_MAX))
    cols = sorted(int(np.argmin(np.abs(lon_vals - v))) for v in (REGION_LON_MIN, REGION_LON_MAX))
    return (
        max(0, rows[0] - half - 1), min(len(lat_vals), rows[1] + half + 2),
        max(0, cols[0] - half - 1), min(len(lon_vals), cols[1] + half + 2),
    )


GPM_CROP = _crop_box(_GPM_LAT_VALS, _GPM_LON_VALS, PRECIP_GRID_SIZE)
SMAP_CROP = _crop_box(_SMAP_LAT_VALS, _SMAP_LON_VALS, SOIL_GRID_SIZE)


def orient_gpm_frame(arr):
    """Flip a raw GPM grid (south-first rows) to north-first rows."""
    return arr[::-1, :]


class RegionGrid:
    """A (T, h, w) stack covering a sub-box of a global grid.

    `box` takes GLOBAL row/col indexes and returns a zero-filled (T, ...) array,
    matching how the old global-array code padded windows past the grid edge.
    """

    def __init__(self, data, row0, col0):
        self.data = data
        self.row0 = int(row0)
        self.col0 = int(col0)

    @property
    def shape(self):
        return self.data.shape

    def box(self, y0, y1, x0, x1):
        t, h, w = self.data.shape
        out = np.zeros((t, y1 - y0, x1 - x0), dtype=np.float32)
        sy0, sy1 = max(y0, self.row0), min(y1, self.row0 + h)
        sx0, sx1 = max(x0, self.col0), min(x1, self.col0 + w)
        if sy1 > sy0 and sx1 > sx0:
            out[:, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = self.data[
                :, sy0 - self.row0:sy1 - self.row0, sx0 - self.col0:sx1 - self.col0
            ]
        return out


def crop_to_region(stack, crop):
    """Crop a global (T, H, W) stack (array or memmap) to a RegionGrid."""
    r0, r1, c0, c1 = crop
    return RegionGrid(np.ascontiguousarray(stack[:, r0:r1, c0:c1], dtype=np.float32), r0, c0)


def synthetic_region(value, n_steps, crop):
    r0, r1, c0, c1 = crop
    return RegionGrid(_synthetic_grid(value, (n_steps, r1 - r0, c1 - c0)), r0, c0)

GLOBAL_SMAP = None
GLOBAL_GPM = None
GLOBAL_TIMESTAMP = None
GLOBAL_DATA_LOCK = threading.Lock()
# One inference at a time per worker: concurrent requests (the page preloads
# 10 cities at once) would otherwise stack ~450 MB each and OOM the container.
INFERENCE_LOCK = threading.Lock()


def _synthetic_grid(value, shape):
    """Read-only constant grid backed by a single float (near-zero memory).

    A materialised random grid needs ~1.7 GB per gunicorn worker for GPM and
    OOM-kills Cloud Run. Callers only slice these arrays, so a broadcast view
    is enough.
    """
    return np.broadcast_to(np.float32(value), shape)

# ---- H3 hexagon config ----
H3_RES = 5    # ~8.5 km edge, ~250 km² per cell; matches GPM's ~11 km pixels
H3_K = 4      # 4 rings around the search cell = 61 hexagons

RISK_COLORS = {
    "low": "#2ecc71",
    "medium": "#f1c40f",
    "warning": "#e67e22",
    "high": "#e74c3c",
}

INFERENCE_BATCH = 16  # one sample is ~8 MB; chunking keeps memory sane on Cloud Run

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

def _download_region_npz():
    """Fetch the latest cropped inputs from Cloud Storage into REGION_NPZ."""
    from google.cloud import storage  # imported lazily: only needed on Cloud Run

    tmp = f"{REGION_NPZ}.{os.getpid()}.tmp"
    blob = storage.Client().bucket(FLOOD_DATA_BUCKET).blob(FLOOD_DATA_BLOB)
    blob.download_to_filename(tmp)
    os.replace(tmp, REGION_NPZ)
    print(f"[FLOOD DATA] Downloaded gs://{FLOOD_DATA_BUCKET}/{FLOOD_DATA_BLOB}")


def _load_region_npz(path):
    """Load GPM + SMAP RegionGrids written by scripts/build_flood_data.py."""
    global GLOBAL_GPM, GLOBAL_SMAP
    with np.load(path) as z:
        GLOBAL_GPM = RegionGrid(z["gpm"], z["gpm_row0"], z["gpm_col0"])
        GLOBAL_SMAP = RegionGrid(z["smap"], z["smap_row0"], z["smap_col0"])
        built_at = str(z["built_at"])
    print(f"[FLOOD DATA] Loaded {path} (built {built_at}): "
          f"GPM {GLOBAL_GPM.shape}, SMAP {GLOBAL_SMAP.shape}")
    age_days = (datetime.utcnow() - datetime.fromisoformat(built_at)).days
    if age_days >= 2:
        print(f"[FLOOD DATA] WARNING: inputs are {age_days} days old — is the daily build job running?")


def load_region_data():
    """Load real cropped inputs if available. Returns True on success."""
    if FLOOD_DATA_BUCKET:
        try:
            _download_region_npz()
        except Exception as e:
            print(f"[FLOOD DATA] ERROR downloading from gs://{FLOOD_DATA_BUCKET}/{FLOOD_DATA_BLOB}: {e}")
    if os.path.exists(REGION_NPZ):
        try:
            _load_region_npz(REGION_NPZ)
            return True
        except Exception as e:
            print(f"[FLOOD DATA] ERROR reading {REGION_NPZ}: {e}")
    return False


def load_global_smap():
    global GLOBAL_SMAP
    if os.path.exists(SMAP_GLOBAL_NPY):
        GLOBAL_SMAP = crop_to_region(np.load(SMAP_GLOBAL_NPY, mmap_mode="r"), SMAP_CROP)
        print(f"[GLOBAL SMAP] Loaded cached stack, cropped to {GLOBAL_SMAP.shape}")
    elif FAST_START_FLOOD_DATA:
        print("[GLOBAL SMAP] ERROR: no real soil-moisture data — using constant placeholder.")
        GLOBAL_SMAP = synthetic_region(0.3, SMAP_N_DAYS, SMAP_CROP)
    else:
        print("[GLOBAL SMAP] Cache missing — auto-building 21-day stack…")
        try:
            GLOBAL_SMAP = crop_to_region(build_global_smap_21days(), SMAP_CROP)
            print(f"[GLOBAL SMAP] Built stack, cropped to {GLOBAL_SMAP.shape}")
        except Exception as e:
            print(f"[GLOBAL SMAP] ERROR building stack: {e} — using constant placeholder.")
            GLOBAL_SMAP = synthetic_region(0.3, SMAP_N_DAYS, SMAP_CROP)

def load_global_gpm():
    """
    Load or build GPM precipitation for the last 4 days (32 3-hour blocks),
    cropped to the served region.
    """
    global GLOBAL_GPM

    if os.path.exists(GPM_GLOBAL_NPY):
        # Legacy global caches were saved south-first; flip to north-first.
        stack = np.load(GPM_GLOBAL_NPY, mmap_mode="r")[:, ::-1, :]
        GLOBAL_GPM = crop_to_region(stack, GPM_CROP)
        print(f"[GLOBAL GPM] Loaded cached stack, cropped to {GLOBAL_GPM.shape}")
    elif FAST_START_FLOOD_DATA:
        print("[GLOBAL GPM] ERROR: no real precipitation data — using constant placeholder.")
        GLOBAL_GPM = synthetic_region(2.5, 32, GPM_CROP)
    else:
        print("[GLOBAL GPM] Cache missing — building 4-day global stack…")
        try:
            stack = build_global_gpm_4days(day_delay=GPM_DAY_DELAY)  # south-first, like the cache
            GLOBAL_GPM = crop_to_region(stack[:, ::-1, :], GPM_CROP)
            print(f"[GLOBAL GPM] Built stack, cropped to {GLOBAL_GPM.shape}")
        except Exception as e:
            print(f"[GLOBAL GPM] ERROR building stack: {e} — using constant placeholder.")
            GLOBAL_GPM = synthetic_region(2.5, 32, GPM_CROP)

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

    if GLOBAL_TIMESTAMP is not None and (now - GLOBAL_TIMESTAMP).total_seconds() <= 86400:
        return

    with GLOBAL_DATA_LOCK:
        now = datetime.utcnow()
        if GLOBAL_TIMESTAMP is not None and (now - GLOBAL_TIMESTAMP).total_seconds() <= 86400:
            return

        print("[GLOBAL CACHE] Reloading SMAP and GPM…")
        if not load_region_data():
            load_global_smap()
            load_global_gpm()
        GLOBAL_TIMESTAMP = now

# =====================================================================
# GPM FUNCTIONS
# =====================================================================

def earthdata_login():
    
    from app.global_gpm_loader import ensure_earthdata_auth

    
    ensure_earthdata_auth()

    
    print("[GPM] Logging in to Earthdata via netrc…")
    auth_result = earthaccess.login(strategy="netrc")
    print(f"[GPM] Login result: {auth_result}")
    print(f"[GPM] Authenticated: {getattr(auth_result, 'authenticated', 'unknown')}")
    print("[GPM] Earthdata login OK.")

def search_and_download_gpm_global(gpm_date: date, out_dir: str):
    """Download GPM without regional bounding box for global coverage."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
       
        from app.global_gpm_loader import ensure_earthdata_auth
        ensure_earthdata_auth()
        print("[GPM] Logging in to Earthdata via netrc before download…")
        earthaccess.login(strategy="netrc")

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
    except Exception as e:
        print(f"[GPM] Download failed for {gpm_date}: {e}")
        print(f"[GPM] Returning empty list - will use synthetic data")
        return []

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
    if GLOBAL_GPM is None:
        raise RuntimeError("GLOBAL_GPM not loaded")

    iy = int(np.argmin(np.abs(_GPM_LAT_VALS - lat)))
    ix = int(np.argmin(np.abs(_GPM_LON_VALS - lon)))
    half = size // 2
    return GLOBAL_GPM.box(iy - half, iy + half, ix - half, ix + half)

# =====================================================================
# DATA PROCESSING FOR INFERENCE
# =====================================================================

def prepare_inference_data(location, scaler):
    """
    Prepare data for one location using global cached grids.
    NOTE: ensure_global_data_loaded() must be called before this function.
    """
    lat = location["lat"]
    lon = location["lon"]

    
    iy = int(np.argmin(np.abs(_SMAP_LAT_VALS - lat)))
    ix = int(np.argmin(np.abs(_SMAP_LON_VALS - lon)))

    half = SOIL_GRID_SIZE // 2

    y0 = iy - half
    y1 = iy + half
    x0 = ix - half
    x1 = ix + half

    soil_grids = list(GLOBAL_SMAP.box(y0, y1, x0, x1)[:SMAP_N_DAYS])

  
    precip_grids = extract_precip_from_global(lat, lon, size=PRECIP_GRID_SIZE)

    if len(precip_grids) < 32:
        print(f"[WARNING] Only got {len(precip_grids)}/32 precipitation time steps")
        return None

    # Process precipitation 
    p_arr = precip_grids[:32]  # (32, H, W)
    orig_shape = p_arr.shape
    p_scaled = scaler.transform(p_arr.reshape(-1, 1)).reshape(orig_shape)
    precip_tensor = torch.tensor(p_scaled[:, np.newaxis, :, :], dtype=torch.float32)

    # Process soil moisture 
    s_arr = np.stack(soil_grids[:21])          # (21, H, W)
    masks = ~np.isnan(s_arr)
    s_filled = np.nan_to_num(s_arr, nan=0.0)
    orig_s_shape = s_filled.shape
    s_scaled = scaler.transform(s_filled.reshape(-1, 1)).reshape(orig_s_shape)
    
    soil_tensor = torch.tensor(
        np.stack([s_scaled, masks.astype(np.float32)], axis=1), dtype=torch.float32
    )

    return precip_tensor, soil_tensor

# =====================================================================
# GRID GENERATION
# =====================================================================

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
    """LEGACY: circular 56-point grid. No longer used by run_flood_inference."""
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

def build_h3_locations(center_lat, center_lon, res=H3_RES, k=H3_K):
    """
    Build a hexagonal disk of H3 cells around the search point.
    Returns (center_cell_id, [location dicts]) where each location has the
    cell id plus the lat/lon of the cell center (used to sample the data grids).
    """
    center_cell = h3.latlng_to_cell(center_lat, center_lon, res)
    locations = []
    for cell in h3.grid_disk(center_cell, k):
        lat, lon = h3.cell_to_latlng(cell)
        locations.append({"name": cell, "h3": cell, "lat": lat, "lon": lon})
    return center_cell, locations

# =====================================================================
# INFERENCE ENTRYPOINT
# =====================================================================


_FLOOD_CACHE_TTL_SECONDS = 86400  

def _cell_cache_key(cell: str) -> str:
    """Per-hexagon Redis key scoped to the UTC date so results auto-expire daily."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    return f"flood_cell:{H3_RES}:{cell}:{date_str}"

def _risk_label(p: float) -> str:
    if p >= 0.75:
        return "high"
    if p >= 0.5:
        return "warning"
    if p >= 0.3:
        return "medium"
    return "low"

def _local_rain_mm(lat: float, lon: float) -> float:
    """4-day rainfall at the cell's own GPM pixel (gives real per-cell variation)."""
    iy = int(np.argmin(np.abs(_GPM_LAT_VALS - lat)))
    ix = int(np.argmin(np.abs(_GPM_LON_VALS - lon)))
    return float(GLOBAL_GPM.box(iy, iy + 1, ix, ix + 1).sum())

def _make_point(loc, prob):
    cell = loc["h3"]
    risk = _risk_label(prob)
    return {
        "h3": cell,
        "lat": loc["lat"],
        "lon": loc["lon"],
        "prob": prob,
        "risk": risk,
        "color": RISK_COLORS[risk],
        "local_rain_mm": round(_local_rain_mm(loc["lat"], loc["lon"]), 2),
        # Leaflet order: [lat, lng]
        "boundary": [[lat, lng] for lat, lng in h3.cell_to_boundary(cell)],
    }

def run_flood_inference(center_lat, center_lon):
    """
    Run the DualCNNLSTM model on an H3 hexagon disk around (center_lat, center_lon).

    Each hexagon is cached individually in Redis for the UTC day, so overlapping
    searches only compute hexagons that have not been seen before.
    """
    center_cell, locations = build_h3_locations(center_lat, center_lon)
    results = {}  # cell id -> point dict

  
    if redis_conn is not None:
        try:
            cached = redis_conn.mget([_cell_cache_key(l["h3"]) for l in locations])
            for loc, raw in zip(locations, cached):
                if raw:
                    results[loc["h3"]] = json.loads(raw)
            print(f"[FLOOD CACHE] {len(results)}/{len(locations)} cells cached")
        except Exception as e:
            print(f"[FLOOD CACHE] Redis read error (continuing without cache): {e}")

    misses = [l for l in locations if l["h3"] not in results]


    if misses:
        model, scaler = get_model_and_scaler()
        ensure_global_data_loaded()  

        print(f"[INFERENCE] {len(misses)} new hexagons for ({center_lat:.3f}, {center_lon:.3f})")

        # Prepare inputs one batch at a time: each sample is ~8 MB, so building
        # all 61 up front costs ~500 MB per worker and OOMs Cloud Run.
        valid_locs, preds = [], []
        with INFERENCE_LOCK, torch.no_grad():
            for i in range(0, len(misses), INFERENCE_BATCH):
                batch_locs, batch_precip, batch_soil = [], [], []
                for loc in misses[i:i + INFERENCE_BATCH]:
                    out = prepare_inference_data(loc, scaler)
                    if out is not None:
                        p, s = out
                        batch_precip.append(p)
                        batch_soil.append(s)
                        batch_locs.append(loc)
                if not batch_locs:
                    continue
                pb = torch.stack(batch_precip).to(device)
                sb = torch.stack(batch_soil).to(device)
                del batch_precip, batch_soil
                preds.extend(model(pb, sb).cpu().numpy().reshape(-1).tolist())
                valid_locs.extend(batch_locs)
                del pb, sb

        new_points = {}
        for loc, prob in zip(valid_locs, preds):
            new_points[loc["h3"]] = _make_point(loc, float(prob))
        results.update(new_points)

      
        if redis_conn is not None and new_points:
            try:
                pipe = redis_conn.pipeline()
                for cell, pt in new_points.items():
                    pipe.setex(_cell_cache_key(cell), _FLOOD_CACHE_TTL_SECONDS, json.dumps(pt))
                pipe.execute()
                print(f"[FLOOD CACHE] STORED {len(new_points)} cells")
            except Exception as e:
                print(f"[FLOOD CACHE] Redis write error (result not cached): {e}")

  
    points = [results[l["h3"]] for l in locations if l["h3"] in results]

    if not points:
        return {
            "center": {"lat": center_lat, "lon": center_lon},
            "center_cell": center_cell,
            "points": [],
            "error": "No valid data extracted",
        }

    return {
        "center": {"lat": center_lat, "lon": center_lon},
        "center_cell": center_cell,
        "h3_res": H3_RES,
        "points": points,
    }