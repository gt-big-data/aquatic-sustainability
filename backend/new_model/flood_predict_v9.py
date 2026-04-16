#!/usr/bin/env python3
"""
FLOOD PREDICTION — SOUTHEAST ASIA REAL-TIME INFERENCE (V9 Model)
================================================================
Predicts flood probability across a grid of SE Asia points using
real-time GPM precipitation + SMAP soil moisture + ERA5 temperature
+ elevation data.

Usage:
    1. Place model files in the same directory as this script
    2. pip install earthaccess h5py rasterio xarray cartopy torch cdsapi netCDF4 global-land-mask
    3. python flood_predict_v9.py
"""

import sys
import os

print("=" * 60, flush=True)
print("FLOOD PREDICTION V9 — Starting up ...", flush=True)
print("=" * 60, flush=True)

# ═══════════════════════════════════════════════════════════════
# 0. IMPORTS
# ═══════════════════════════════════════════════════════════════

_missing = []

def _check(name):
    try:
        __import__(name)
    except ImportError:
        _missing.append(name)

import re, glob, pickle
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

for pkg in ["numpy", "pandas", "xarray", "h5py", "torch",
            "matplotlib", "earthaccess", "rasterio"]:
    _check(pkg)

if _missing:
    print(f"\n[FATAL] Missing packages: {', '.join(_missing)}", flush=True)
    print(f"        pip install {' '.join(_missing)}", flush=True)
    sys.exit(1)

import numpy as np
import pandas as pd
import xarray as xr
import h5py
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import earthaccess
import rasterio
import rasterio.windows

HAS_CARTOPY = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    print("[WARN] cartopy not found — plain matplotlib map.", flush=True)

HAS_LAND_MASK = False
try:
    from global_land_mask import globe
    HAS_LAND_MASK = True
    print("[LAND] global-land-mask loaded.", flush=True)
except ImportError:
    print("[WARN] global-land-mask not found — no ocean filter.", flush=True)

ERA5_AVAILABLE = False
try:
    import cdsapi
    ERA5_AVAILABLE = True
    print("[ERA5] cdsapi found.", flush=True)
except ImportError:
    print("[ERA5] cdsapi not installed — temp will be zero-padded.", flush=True)

print("All imports OK.\n", flush=True)

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

EARTHDATA_USER = "albertzheng"
EARTHDATA_PASS = "Alberbruh321$"
CDS_URL = "https://cds.climate.copernicus.eu/api"
CDS_KEY = "25fec36b-e11f-45fd-9dd9-c16fe4a4d4ee"

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

MODEL_PATH         = os.path.join(BASE_DIR, "DualCNNLSTM_best_model_V9.pt")
SCALER_PRECIP_PATH = os.path.join(BASE_DIR, "DualCNNLSTM_scaler_precip_V9.pkl")
SCALER_SOIL_PATH   = os.path.join(BASE_DIR, "DualCNNLSTM_scaler_soil_V9.pkl")
ELEV_TIF_PATH      = os.path.join(BASE_DIR, "elevation_asia_7_5arc.tif")

print(f"[CONFIG] BASE_DIR = {BASE_DIR}", flush=True)
_all_ok = True
for label, path in [("Model weights", MODEL_PATH),
                     ("Precip scaler", SCALER_PRECIP_PATH),
                     ("Soil scaler",   SCALER_SOIL_PATH),
                     ("Elevation TIF", ELEV_TIF_PATH)]:
    exists = os.path.exists(path)
    print(f"  {label:20s}: {'OK' if exists else 'MISSING'}  ({path})", flush=True)
    if not exists:
        _all_ok = False

if not _all_ok:
    print("\n[FATAL] Missing files. Place them next to this script.", flush=True)
    sys.exit(1)
print(flush=True)

THRESHOLD = 0.85
LAT_MIN, LAT_MAX, LAT_STEP = -10, 28, 1.0
LON_MIN, LON_MAX, LON_STEP = 95, 140, 1.0
LAND_FRACTION_THRESHOLD = 0.40

DATA_ROOT = os.path.join(BASE_DIR, "data")
GPM_DIR   = os.path.join(DATA_ROOT, "gpm_download")
SMAP_DIR  = os.path.join(DATA_ROOT, "smap_download")
ERA5_DIR  = os.path.join(DATA_ROOT, "era5_download")
for d in [GPM_DIR, SMAP_DIR, ERA5_DIR]:
    os.makedirs(d, exist_ok=True)

SEQ_LEN = 32; SOIL_SEQ_LEN = 21; TEMP_SEQ_LEN = 32
SOIL_GRID_SIZE = 50; ELEV_GRID_KM = 275; ELEV_PIXEL_SIZE = 8
ELEV_KM_PER_DEG = 111.0; DROPOUT_RATE = 0.5

TODAY       = datetime.utcnow().date()
BEGAN_PROXY = TODAY + timedelta(days=5)
PRECIP_END   = BEGAN_PROXY - timedelta(days=6)
PRECIP_START = PRECIP_END  - timedelta(days=3)
SOIL_END     = BEGAN_PROXY - timedelta(days=11)
SOIL_START   = SOIL_END    - timedelta(days=20)
TEMP_END     = BEGAN_PROXY - timedelta(days=11)
TEMP_START   = TEMP_END    - timedelta(days=7)

_today_str = TODAY.strftime("%Y%m%d")
GPM_CACHE  = os.path.join(DATA_ROOT, f"gpm_blocks_{_today_str}.npy")
SMAP_CACHE = os.path.join(DATA_ROOT, f"smap_daily_{_today_str}.pkl")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH  = os.path.join(OUTPUT_DIR, f"flood_predictions_V9_{TODAY}.csv")
MAP_PATH  = os.path.join(OUTPUT_DIR, f"flood_map_V9_{TODAY}.png")
DIST_PATH = os.path.join(OUTPUT_DIR, f"flood_distribution_V9_{TODAY}.png")

def print_config():
    print("=" * 60, flush=True)
    print("CONFIGURATION", flush=True)
    print("=" * 60, flush=True)
    print(f"  TODAY            : {TODAY}", flush=True)
    print(f"  BEGAN_PROXY      : {BEGAN_PROXY}  (prediction target)", flush=True)
    print(f"  Precip window    : {PRECIP_START} → {PRECIP_END}", flush=True)
    print(f"  Soil window      : {SOIL_START} → {SOIL_END}", flush=True)
    print(f"  Temp window      : {TEMP_START} → {TEMP_END}", flush=True)
    print(f"  Threshold        : {THRESHOLD}", flush=True)
    print("=" * 60 + "\n", flush=True)

# ═══════════════════════════════════════════════════════════════
# 2. CDS API SETUP
# ═══════════════════════════════════════════════════════════════

def setup_cds_credentials():
    rc_path = os.path.join(os.path.expanduser("~"), ".cdsapirc")
    with open(rc_path, "w") as f:
        f.write(f"url: {CDS_URL}\nkey: {CDS_KEY}\n")
    os.chmod(rc_path, 0o600)
    print(f"[CDS] .cdsapirc written to {rc_path}", flush=True)

# ═══════════════════════════════════════════════════════════════
# 3. MODEL DEFINITION (V9)
# ═══════════════════════════════════════════════════════════════

class DualCNNLSTM(nn.Module):
    def __init__(self, dropout=DROPOUT_RATE):
        super().__init__()
        self.cnn_precip = nn.Sequential(
            nn.Conv2d(1,8,5,2,2), nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(8,16,3,2,1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8,8)), nn.Dropout(dropout))
        self.cnn_soil = nn.Sequential(
            nn.Conv2d(2,8,5,2,2), nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(8,16,3,2,1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8,8)), nn.Dropout(dropout))
        self.cnn_temp = nn.Sequential(
            nn.Conv2d(2,8,5,2,2), nn.BatchNorm2d(8), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(dropout),
            nn.Conv2d(8,16,3,2,1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.AdaptiveAvgPool2d((8,8)), nn.Dropout(dropout))
        self.lstm = nn.LSTM(input_size=3*16*8*8, hidden_size=64, batch_first=True)
        self.dropout_lstm = nn.Dropout(dropout)
        self.cnn_elev = nn.Sequential(
            nn.Conv2d(1,8,3,1,1), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Dropout(dropout), nn.AdaptiveAvgPool2d((2,2)))
        self.elev_fc = nn.Sequential(
            nn.Linear(8*2*2, 32), nn.ReLU(), nn.Dropout(dropout))
        self.fc_flood = nn.Linear(64+32, 1)

    def forward(self, precip, soil, temp, elev):
        B, T = precip.size(0), precip.size(1)
        lstm_in = []
        for t in range(T):
            p  = self.cnn_precip(precip[:,t]).view(B,-1)
            s  = self.cnn_soil(soil[:,t]).view(B,-1)
            tp = self.cnn_temp(temp[:,t]).view(B,-1)
            lstm_in.append(torch.cat([p,s,tp], dim=1))
        lstm_in = torch.stack(lstm_in, dim=1)
        lstm_out, _ = self.lstm(lstm_in)
        h_last = self.dropout_lstm(lstm_out[:,-1,:])
        elev_feat = self.cnn_elev(elev).view(B,-1)
        elev_feat = self.elev_fc(elev_feat)
        fused = torch.cat([h_last, elev_feat], dim=1)
        return torch.sigmoid(self.fc_flood(fused))

# ═══════════════════════════════════════════════════════════════
# 4. LOAD MODEL & SCALERS
# ═══════════════════════════════════════════════════════════════

def load_model_and_scalers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MODEL] Device: {device}", flush=True)
    model = DualCNNLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()
    print("[MODEL] V9 model loaded.", flush=True)
    with open(SCALER_PRECIP_PATH,"rb") as f: scaler_precip = pickle.load(f)
    with open(SCALER_SOIL_PATH,"rb") as f:   scaler_soil   = pickle.load(f)
    print("[MODEL] V9 scalers loaded.", flush=True)
    return model, scaler_precip, scaler_soil, device

# ═══════════════════════════════════════════════════════════════
# 5. EARTHDATA AUTH
# ═══════════════════════════════════════════════════════════════

def setup_earthdata_auth():
    os.environ["EARTHDATA_USERNAME"] = EARTHDATA_USER
    os.environ["EARTHDATA_PASSWORD"] = EARTHDATA_PASS
    netrc_path = os.path.expanduser("~/.netrc")
    with open(netrc_path, "w") as f:
        f.write(f"machine urs.earthdata.nasa.gov\nlogin {EARTHDATA_USER}\npassword {EARTHDATA_PASS}\n"
                f"machine data.gesdisc.earthdata.nasa.gov\nlogin {EARTHDATA_USER}\npassword {EARTHDATA_PASS}\n")
    os.chmod(netrc_path, 0o600)
    earthaccess.login(strategy="netrc")
    print("[AUTH] Earthdata login complete.", flush=True)

# ═══════════════════════════════════════════════════════════════
# 6. GPM PRECIPITATION
# ═══════════════════════════════════════════════════════════════

GPM_LATS = np.linspace(90, -90, 1800)
GPM_LONS = np.linspace(-180, 180, 3600)

def download_gpm(start_date, end_date):
    print(f"[GPM] Downloading {start_date} → {end_date} ...", flush=True)
    results = earthaccess.search_data(short_name="GPM_3IMERGHHL", version="07",
                                      temporal=(str(start_date), str(end_date)))
    print(f"[GPM] Found {len(results)} granules.", flush=True)
    paths = earthaccess.download(results, local_path=GPM_DIR)
    files = sorted([str(p) for p in paths if str(p).lower().endswith(".hdf5") and "3B-HHR" in str(p)])
    print(f"[GPM] Downloaded {len(files)} HDF5 files.", flush=True)
    return files

def load_gpm_file(path):
    with h5py.File(path,"r") as f:
        arr = f["Grid"]["precipitation"][0].T
    return (np.where(arr < 0, 0.0, arr) * 0.5).astype(np.float32)

def build_gpm_32blocks(gpm_files):
    by_date = defaultdict(list)
    for path in gpm_files:
        try:
            d = datetime.strptime(os.path.basename(path).split('.')[4].split('-')[0], "%Y%m%d").date()
            by_date[d].append(path)
        except Exception: continue
    all_blocks = []
    for d in sorted(by_date):
        frames = []
        for p in sorted(by_date[d]):
            try: frames.append(load_gpm_file(p))
            except Exception as e: print(f"[GPM] Skip {os.path.basename(p)}: {e}", flush=True)
        for i in range(0, len(frames), 6):
            chunk = frames[i:i+6]
            if chunk: all_blocks.append(np.sum(chunk, axis=0).astype(np.float32))
        while len(all_blocks) % 8 != 0:
            all_blocks.append(np.zeros((1800,3600), dtype=np.float32))
    while len(all_blocks) < SEQ_LEN:
        all_blocks.append(np.zeros((1800,3600), dtype=np.float32))
    return all_blocks[:SEQ_LEN]

def extract_precip_for_point(lat, lon, blocks_32):
    HALF = 10
    li = np.abs(GPM_LATS - lat).argmin(); lo = np.abs(GPM_LONS - lon).argmin()
    r0=max(li-HALF,0); r1=min(li+HALF,1799); c0=max(lo-HALF,0); c1=min(lo+HALF,3599)
    patch = np.stack([b[r0:r1, c0:c1] for b in blocks_32], axis=0)
    out = np.zeros((SEQ_LEN,20,20), dtype=np.float32)
    h,w = patch.shape[1], patch.shape[2]
    out[:, :min(h,20), :min(w,20)] = patch[:, :min(h,20), :min(w,20)]
    return out

def load_or_download_gpm():
    if os.path.exists(GPM_CACHE):
        print(f"[GPM] Loading from cache: {GPM_CACHE}", flush=True)
        blocks = list(np.load(GPM_CACHE))
        print(f"[GPM] Loaded {len(blocks)} cached blocks.", flush=True)
        return blocks
    gpm_files = download_gpm(PRECIP_START, PRECIP_END)
    blocks = build_gpm_32blocks(gpm_files)
    np.save(GPM_CACHE, np.stack(blocks))
    print(f"[GPM] Cached {len(blocks)} blocks → {GPM_CACHE}", flush=True)
    return blocks

# ═══════════════════════════════════════════════════════════════
# 7. SMAP SOIL MOISTURE
# ═══════════════════════════════════════════════════════════════

def download_smap(start_date, end_date):
    print(f"[SMAP] Downloading {start_date} → {end_date} ...", flush=True)
    results = earthaccess.search_data(short_name="SPL4SMAU", version="008",
                                      temporal=(str(start_date), str(end_date)))
    print(f"[SMAP] Found {len(results)} granules.", flush=True)
    paths = earthaccess.download(results, local_path=SMAP_DIR)
    files = sorted([str(p) for p in paths if str(p).lower().endswith(".h5")])
    print(f"[SMAP] Downloaded {len(files)} files.", flush=True)
    return files

def load_smap_file(path):
    with h5py.File(path,"r") as f:
        grp = f["Analysis_Data"]
        sm_key = next((k for k in ("sm_surface_analysis","sm_surface","sm_surface_analysis_map") if k in grp), None)
        if sm_key is None: raise KeyError(f"No SM key in {list(grp.keys())}")
        sm = grp[sm_key][:]
        sm = np.where((sm<0)|np.isnan(sm), np.nan, sm)
        lat2d = f.get("cell_lat", f.get("Cell_Lat", None))[:]
        lon2d = f.get("cell_lon", f.get("Cell_Lon", None))[:]
    lat_1d = lat2d[:,0] if lat2d.ndim==2 else np.unique(lat2d)
    lon_1d = lon2d[0,:] if lon2d.ndim==2 else np.unique(lon2d)
    if np.any(np.diff(lat_1d)<0): lat_1d=lat_1d[::-1]; sm=sm[::-1,:]
    return xr.DataArray(sm, dims=["lat","lon"], coords={"lat":lat_1d,"lon":lon_1d})

def build_smap_daily_stack(smap_files, target_dates):
    by_date = defaultdict(list)
    for fp in smap_files:
        m = re.search(r'(\d{8})', os.path.basename(fp))
        if m: by_date[datetime.strptime(m.group(1),"%Y%m%d").date()].append(fp)
    daily = {}
    for d in target_dates:
        if d not in by_date: daily[d]=None; continue
        grids = []
        for fp in by_date[d]:
            try: grids.append(load_smap_file(fp))
            except Exception as e: print(f"[SMAP] Skip {os.path.basename(fp)}: {e}", flush=True)
        if grids:
            avg = np.nanmean([g.values for g in grids], axis=0).astype(np.float32)
            daily[d] = xr.DataArray(avg, dims=["lat","lon"],
                                    coords={"lat":grids[0].lat.values,"lon":grids[0].lon.values})
        else: daily[d]=None
    return daily

def extract_soil_for_point(lat, lon, daily_stack, target_dates, grid_size=SOIL_GRID_SIZE):
    grids=[]; all_null=True
    for d in target_dates:
        da = daily_stack.get(d)
        if da is None: continue
        half=grid_size//2
        li=np.abs(da.lat.values-lat).argmin(); lo=np.abs(da.lon.values-lon).argmin()
        r0=max(li-half,0); r1=r0+grid_size
        if r1>len(da.lat): r1=len(da.lat); r0=r1-grid_size
        c0=max(lo-half,0); c1=c0+grid_size
        if c1>len(da.lon): c1=len(da.lon); c0=c1-grid_size
        arr=da.values[r0:r1,c0:c1].astype(np.float32)
        if arr.shape!=(grid_size,grid_size): continue
        if np.isnan(arr).all(): continue
        all_null=False
        mask=(~np.isnan(arr)).astype(np.float32)
        arr=np.nan_to_num(arr, nan=0.0)
        grids.append(np.stack([arr,mask], axis=0))
    if all_null or len(grids)==0: return None
    zf=np.zeros((2,grid_size,grid_size), dtype=np.float32)
    while len(grids)<SOIL_SEQ_LEN: grids.append(zf.copy())
    return grids[:SOIL_SEQ_LEN]

def load_or_download_smap():
    smap_dates = pd.date_range(SOIL_START, SOIL_END, freq="D").date.tolist()
    if os.path.exists(SMAP_CACHE):
        print(f"[SMAP] Loading from cache: {SMAP_CACHE}", flush=True)
        with open(SMAP_CACHE,"rb") as f: smap_daily=pickle.load(f)
        ok=sum(v is not None for v in smap_daily.values())
        print(f"[SMAP] Loaded {ok}/{len(smap_dates)} days.", flush=True)
        return smap_daily, smap_dates
    smap_files = download_smap(SOIL_START, SOIL_END)
    smap_daily = build_smap_daily_stack(smap_files, smap_dates)
    with open(SMAP_CACHE,"wb") as f: pickle.dump(smap_daily, f)
    ok=sum(v is not None for v in smap_daily.values())
    print(f"[SMAP] Cached → {SMAP_CACHE} ({ok}/{len(smap_dates)} days)", flush=True)
    return smap_daily, smap_dates

# ═══════════════════════════════════════════════════════════════
# 8. ERA5 TEMPERATURE
# ═══════════════════════════════════════════════════════════════

def download_era5_temp():
    if not ERA5_AVAILABLE:
        print("[ERA5] cdsapi not available — skipping.", flush=True); return None
    fpath = os.path.join(ERA5_DIR, f"era5_t2m_{TEMP_START}_{TEMP_END}.nc")
    if os.path.exists(fpath):
        print(f"[ERA5] Cached: {fpath}", flush=True); return fpath
    print(f"[ERA5] Downloading {TEMP_START} → {TEMP_END} ...", flush=True)
    c = cdsapi.Client()
    dates = pd.date_range(TEMP_START, TEMP_END, freq="D")
    
    c.retrieve("reanalysis-era5-land", {
        "variable":"2m_temperature", "product_type":"reanalysis",
        "year":sorted(set(str(d.year) for d in dates)),
        "month":sorted(set(f"{d.month:02d}" for d in dates)),
        "day":sorted(set(f"{d.day:02d}" for d in dates)),
        "time":["00:00","06:00","12:00","18:00"],
        "data_format":"netcdf",
        "download_format":"unarchived",
        "area":[30,90,-15,145]}, fpath)

    # --- NEW FIX: Handle forced ZIP archives ---
    import zipfile
    if zipfile.is_zipfile(fpath):
        print("[ERA5] API forced a ZIP archive. Extracting...", flush=True)
        extract_dir = os.path.join(ERA5_DIR, "temp_unzip")
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(fpath, 'r') as z:
            z.extractall(extract_dir)
            
        os.remove(fpath) # Delete the zipped version
        
        # The CDS API usually names the unzipped file 'data.nc'
        extracted_files = glob.glob(os.path.join(extract_dir, "*.nc"))
        if extracted_files:
            os.rename(extracted_files[0], fpath)
    # ------------------------------------------

    print(f"[ERA5] Downloaded → {fpath}", flush=True)
    return fpath

def build_temp_stack(era5_path):
    if era5_path is None or not os.path.exists(era5_path):
        print("[ERA5] No temp file — empty stack.", flush=True); return {}
    ds = xr.open_dataset(era5_path, engine="netcdf4")
    if "expver" in ds.dims:
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    var = next((v for v in ("t2m","VAR_2T","2m_temperature") if v in ds), None)
    if var is None: ds.close(); return {}
    stack = {}
    for ts in pd.DatetimeIndex(ds[time_dim].values):
        try: stack[pd.Timestamp(ts)] = ds[var].sel(**{time_dim:pd.Timestamp(ts)}, method="nearest").load()
        except Exception: continue
    ds.close()
    print(f"[ERA5] Loaded {len(stack)} six-hourly frames.", flush=True)
    return stack

def extract_temp_for_point(lat, lon, temp_stack, grid_size=SOIL_GRID_SIZE):
    T_MIN,T_MAX = 230.0, 330.0
    times = [pd.Timestamp(d.year,d.month,d.day,hr)
             for d in pd.date_range(TEMP_START,TEMP_END,freq="D") for hr in [0,6,12,18]]
    grids = []
    for ts in times:
        da = temp_stack.get(ts)
        if da is None: continue
        lat_arr=da.coords.get("latitude",da.coords.get("lat")).values
        lon_arr=da.coords.get("longitude",da.coords.get("lon")).values
        half=grid_size//2
        li=np.abs(lat_arr-lat).argmin(); lo=np.abs(lon_arr-lon).argmin()
        r0=max(li-half,0); r1=r0+grid_size
        if r1>len(lat_arr): r1=len(lat_arr); r0=r1-grid_size
        c0=max(lo-half,0); c1=c0+grid_size
        if c1>len(lon_arr): c1=len(lon_arr); c0=c1-grid_size
        arr=da.values[r0:r1,c0:c1].astype(np.float32)
        if arr.shape!=(grid_size,grid_size): continue
        t_norm=np.clip((arr-T_MIN)/(T_MAX-T_MIN),0.0,1.0)
        mask=(~np.isnan(arr)).astype(np.float32)
        t_norm=np.nan_to_num(t_norm, nan=0.0)
        grids.append(np.stack([t_norm,mask], axis=0))
    zf=np.zeros((2,grid_size,grid_size), dtype=np.float32)
    while len(grids)<TEMP_SEQ_LEN: grids.append(zf.copy())
    return grids[:TEMP_SEQ_LEN]

# ═══════════════════════════════════════════════════════════════
# 9. ELEVATION
# ═══════════════════════════════════════════════════════════════

def extract_elevation_grid(lat, lon, grid_km=ELEV_GRID_KM):
    if not os.path.exists(ELEV_TIF_PATH): return None
    try:
        with rasterio.open(ELEV_TIF_PATH) as src:
            t=src.transform
            hpx=int(np.ceil(grid_km/(2*abs(t.a)*ELEV_KM_PER_DEG)))
            hpy=int(np.ceil(grid_km/(2*abs(t.e)*ELEV_KM_PER_DEG)))
            col,row = ~t*(lon,lat); col,row=int(col),int(row)
            r0=max(row-hpy,0); r1=min(row+hpy,src.height)
            c0=max(col-hpx,0); c1=min(col+hpx,src.width)
            if r0>=r1 or c0>=c1: return None
            arr=src.read(1, window=rasterio.windows.Window(c0,r0,c1-c0,r1-r0)).astype(np.float32)
            nd=src.nodata if src.nodata is not None else -9999
        arr[arr==nd]=np.nan
        return np.nan_to_num(arr, nan=0.0)
    except Exception as e:
        print(f"[ELEV] Error at ({lat},{lon}): {e}", flush=True); return None

def pixelate_elevation(arr, out_size=ELEV_PIXEL_SIZE):
    h,w=arr.shape; ht=(h//out_size)*out_size; wt=(w//out_size)*out_size
    arr=arr[:ht,:wt]; bh,bw=ht//out_size, wt//out_size
    return arr.reshape(out_size,bh,out_size,bw).mean(axis=(1,3)).astype(np.float32)

# ═══════════════════════════════════════════════════════════════
# 10. PREDICTION GRID
# ═══════════════════════════════════════════════════════════════

def is_sufficiently_land(lat, lon, step, n_sub=5, threshold=LAND_FRACTION_THRESHOLD):
    if not HAS_LAND_MASK: return True
    half=step/2.0; offsets=np.linspace(-half*0.9,half*0.9,n_sub)
    lc=tot=0
    for dlat in offsets:
        for dlon in offsets:
            sl,sn=lat+dlat,lon+dlon
            if -90<=sl<=90 and -180<=sn<=180: lc+=int(globe.is_land(sl,sn)); tot+=1
    return (lc/tot)>=threshold if tot>0 else False

def build_grid_points():
    lats=np.arange(LAT_MIN,LAT_MAX+LAT_STEP,LAT_STEP)
    lons=np.arange(LON_MIN,LON_MAX+LON_STEP,LON_STEP)
    print("[GRID] Applying land filter ...", flush=True)
    pts=[]; ncp=0; nnf=0
    for lat in lats:
        for lon in lons:
            if HAS_LAND_MASK and not globe.is_land(lat,lon): continue
            ncp+=1
            if is_sufficiently_land(lat,lon,step=LAT_STEP): pts.append((lat,lon))
            else: nnf+=1
    print(f"[GRID] Total: {len(lats)*len(lons)} | Centre pass: {ncp} | Rejected: {nnf} | Final: {len(pts)}", flush=True)
    return pts

def precompute_elevation(grid_points):
    print("[ELEV] Pre-computing elevation grids ...", flush=True)
    ec={}
    for lat,lon in grid_points:
        raw=extract_elevation_grid(lat,lon)
        if raw is not None:
            try: ec[(lat,lon)]=pixelate_elevation(raw)
            except: pass
    print(f"[ELEV] {len(ec)}/{len(grid_points)} OK.", flush=True)
    return ec

# ═══════════════════════════════════════════════════════════════
# 11. INFERENCE
# ═══════════════════════════════════════════════════════════════

def run_inference(model, scaler_precip, scaler_soil, device,
                  gpm_blocks, smap_daily, smap_dates, temp_stack,
                  grid_points, elev_cache):
    results=[]; nss=nse=nps=0
    print(f"\n[INFERENCE] {len(grid_points)} points | threshold={THRESHOLD}", flush=True)
    for i,(lat,lon) in enumerate(grid_points):
        if (i+1)%20==0:
            pred_n=len([r for r in results if r["prob"] is not None])
            print(f"  [{i+1}/{len(grid_points)}] ({lat:.1f},{lon:.1f}) predicted:{pred_n} skipped:{nss+nse}", flush=True)

        pp=extract_precip_for_point(lat,lon,gpm_blocks)
        ps=scaler_precip.transform(pp.flatten().reshape(-1,1)).reshape(pp.shape).astype(np.float32)
        pt=torch.tensor(ps[:,np.newaxis,:,:], dtype=torch.float32).unsqueeze(0)

        sg=extract_soil_for_point(lat,lon,smap_daily,smap_dates)
        if sg is None:
            nss+=1; results.append({"lat":lat,"lon":lon,"prob":None,"prediction":None,"skip_reason":"soil_all_null"}); continue

        vsd=sum(1 for g in sg if g[1].any())
        if vsd<SOIL_SEQ_LEN: nps+=1
        ss=[]
        for arr in sg:
            sm=scaler_soil.transform(arr[0].flatten().reshape(-1,1)).reshape(arr[0].shape).astype(np.float32)
            ss.append(np.stack([sm,arr[1]], axis=0))
        zs=np.zeros_like(ss[0])
        while len(ss)<SEQ_LEN: ss.append(zs.copy())
        st=torch.tensor(np.stack(ss), dtype=torch.float32).unsqueeze(0)

        tg=extract_temp_for_point(lat,lon,temp_stack)
        tt=torch.tensor(np.stack(tg), dtype=torch.float32).unsqueeze(0)

        ec=elev_cache.get((lat,lon))
        if ec is None:
            nse+=1; results.append({"lat":lat,"lon":lon,"prob":None,"prediction":None,"skip_reason":"no_elev"}); continue
        en=np.clip(ec/5000.0,0.0,1.0).astype(np.float32)
        et=torch.tensor(en[np.newaxis,np.newaxis,:,:], dtype=torch.float32)

        try:
            with torch.no_grad():
                prob=model(pt.to(device),st.to(device),tt.to(device),et.to(device)).item()
            pred="flood" if prob>=THRESHOLD else "non-flood"
            results.append({"lat":lat,"lon":lon,"prob":prob,"prediction":pred,"skip_reason":None,"valid_soil_days":vsd})
        except Exception as e:
            print(f"  [ERR] ({lat},{lon}): {e}", flush=True)
            results.append({"lat":lat,"lon":lon,"prob":None,"prediction":None,"skip_reason":str(e)})

    df=pd.DataFrame(results); vdf=df[df["prob"].notna()].copy()
    print(f"\n[DONE] {len(vdf)}/{len(df)} predicted", flush=True)
    if len(vdf)>0:
        print(f"  Flood: {(vdf['prediction']=='flood').sum()} ({(vdf['prediction']=='flood').mean()*100:.1f}%)", flush=True)
        print(f"  Non-flood: {(vdf['prediction']=='non-flood').sum()} ({(vdf['prediction']=='non-flood').mean()*100:.1f}%)", flush=True)
    print(f"  Skipped soil:{nss} elev:{nse} | Partial soil:{nps}", flush=True)
    return df

# ═══════════════════════════════════════════════════════════════
# 12. PLOTTING
# ═══════════════════════════════════════════════════════════════

def plot_flood_map(df, save_path=None):
    valid=df[df["prob"].notna()].copy(); skipped=df[df["prob"].isna()].copy()
    fp=valid[valid["prediction"]=="flood"]; nfp=valid[valid["prediction"]=="non-flood"]
    nf=len(fp); nnf=len(nfp); nt=len(valid)

    if HAS_CARTOPY:
        fig=plt.figure(figsize=(16,10))
        ax=fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
        ax.set_extent([LON_MIN-2,LON_MAX+2,LAT_MIN-2,LAT_MAX+2],crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN,facecolor="#d0e8f5",zorder=0)
        ax.add_feature(cfeature.LAND,facecolor="#f5f0e8",zorder=1)
        ax.add_feature(cfeature.COASTLINE,linewidth=0.6,zorder=2)
        ax.add_feature(cfeature.BORDERS,linewidth=0.4,linestyle="--",edgecolor="#888888",zorder=2)
        ax.add_feature(cfeature.RIVERS,linewidth=0.3,edgecolor="#aaaaff",zorder=2)
        tr=ccrs.PlateCarree()
    else:
        fig,ax=plt.subplots(figsize=(16,10))
        ax.set_xlim(LON_MIN-2,LON_MAX+2); ax.set_ylim(LAT_MIN-2,LAT_MAX+2)
        ax.set_facecolor("#d0e8f5"); tr=None

    def sc(pts,color,zo,label,sizes,alpha=0.85):
        if len(pts)==0: return
        kw=dict(c=color,s=sizes,alpha=alpha,zorder=zo,edgecolors="white",linewidths=0.4,label=label)
        if HAS_CARTOPY: ax.scatter(pts["lon"],pts["lat"],transform=tr,**kw)
        else: ax.scatter(pts["lon"],pts["lat"],**kw)

    if len(nfp)>0: sc(nfp,"#2ecc71",4,f"Non-flood ({nnf})",40+120*(1-nfp["prob"].values))
    if len(fp)>0:  sc(fp,"#e74c3c",5,f"Flood risk ({nf})",40+160*fp["prob"].values,alpha=0.9)
    if len(skipped)>0:
        kw=dict(c="#aaaaaa",s=25,alpha=0.4,zorder=3,label=f"No data ({len(skipped)})")
        if HAS_CARTOPY: ax.scatter(skipped["lon"],skipped["lat"],transform=tr,**kw)
        else: ax.scatter(skipped["lon"],skipped["lat"],**kw)

    if HAS_CARTOPY:
        gl=ax.gridlines(draw_labels=True,linewidth=0.4,color="gray",alpha=0.5,linestyle="--")
        gl.top_labels=False; gl.right_labels=False

    ax.legend(loc="lower left",fontsize=10,framealpha=0.9)
    sm_bar=plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,norm=plt.Normalize(0,1))
    sm_bar.set_array([]); cbar=plt.colorbar(sm_bar,ax=ax,orientation="vertical",fraction=0.025,pad=0.02)
    cbar.set_label("Flood Probability",fontsize=10)
    fpct=nf/nt*100 if nt>0 else 0
    ax.set_title(f"SE Asia Flood Risk [V9]\nTarget: {BEGAN_PROXY} | Data: {PRECIP_START}→{PRECIP_END}\n"
                 f"Threshold={THRESHOLD:.2f} | {nf}/{nt} flood risk ({fpct:.1f}%)",fontsize=12,pad=10)
    plt.tight_layout()
    if save_path: plt.savefig(save_path,dpi=150,bbox_inches="tight"); print(f"[MAP] Saved → {save_path}",flush=True)
    plt.close()

def plot_probability_distribution(df, save_path=None):
    valid=df[df["prob"].notna()]; fig,axes=plt.subplots(1,2,figsize=(14,5))
    ax=axes[0]
    ax.hist(valid["prob"],bins=30,color="#3498db",edgecolor="white",alpha=0.8)
    ax.axvline(THRESHOLD,color="#e74c3c",linewidth=2,linestyle="--",label=f"Threshold={THRESHOLD}")
    ax.set_xlabel("Flood Probability"); ax.set_ylabel("Grid Points"); ax.set_title("Probability Distribution [V9]"); ax.legend()
    ax=axes[1]
    fn=int((valid["prediction"]=="flood").sum()); nn=int((valid["prediction"]=="non-flood").sum()); sn=int(df["prob"].isna().sum())
    sizes=[fn,nn]+([sn] if sn>0 else []); labels=[f"Flood\n{fn}",f"Non-flood\n{nn}"]+([f"No data\n{sn}"] if sn>0 else [])
    colors=["#e74c3c","#2ecc71"]+([" #aaaaaa"] if sn>0 else [])
    ax.pie(sizes,labels=labels,colors=colors,autopct="%1.1f%%",startangle=90,wedgeprops={"edgecolor":"white","linewidth":1.5})
    ax.set_title(f"Breakdown (threshold={THRESHOLD}) [V9]")
    plt.suptitle(f"Flood Stats — SE Asia — {TODAY} [V9]",fontsize=13,y=1.02); plt.tight_layout()
    if save_path: plt.savefig(save_path,dpi=150,bbox_inches="tight"); print(f"[PLOT] Saved → {save_path}",flush=True)
    plt.close()

# ═══════════════════════════════════════════════════════════════
# 13. SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_summary(results_df):
    valid=results_df[results_df["prob"].notna()]
    print("\n"+"="*60, flush=True)
    print(f"SUMMARY — V9 — {TODAY}", flush=True)
    print("="*60, flush=True)
    print(f"Target: {BEGAN_PROXY} | Precip: {PRECIP_START}→{PRECIP_END} | Soil: {SOIL_START}→{SOIL_END} | Temp: {TEMP_START}→{TEMP_END}", flush=True)
    print(f"Threshold: {THRESHOLD} | Grid: {LAT_STEP}°×{LON_STEP}° | Land filter: ≥{LAND_FRACTION_THRESHOLD:.0%}", flush=True)
    print(f"Total: {len(results_df)} | Predicted: {len(valid)}", flush=True)
    if len(valid)>0:
        print(f"Flood: {(valid['prediction']=='flood').sum()} ({(valid['prediction']=='flood').mean()*100:.1f}%)", flush=True)
        print(f"Non-flood: {(valid['prediction']=='non-flood').sum()} ({(valid['prediction']=='non-flood').mean()*100:.1f}%)", flush=True)
        print(f"Prob stats — mean:{valid['prob'].mean():.4f} std:{valid['prob'].std():.4f} "
              f"median:{valid['prob'].median():.4f} min:{valid['prob'].min():.4f} max:{valid['prob'].max():.4f}", flush=True)
        print("="*60, flush=True)
        print("\nTop 30 highest flood probability:", flush=True)
        cols=["lat","lon","prob","prediction"]
        if "valid_soil_days" in valid.columns: cols.append("valid_soil_days")
        print(valid.nlargest(30,"prob")[cols].to_string(index=False), flush=True)

# ═══════════════════════════════════════════════════════════════
# 14. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print_config()
    setup_cds_credentials()
    setup_earthdata_auth()
    model, scaler_precip, scaler_soil, device = load_model_and_scalers()

    gpm_blocks = load_or_download_gpm()
    smap_daily, smap_dates = load_or_download_smap()
    era5_path = download_era5_temp()
    temp_stack = build_temp_stack(era5_path)

    grid_points = build_grid_points()
    elev_cache = precompute_elevation(grid_points)

    results_df = run_inference(model, scaler_precip, scaler_soil, device,
                               gpm_blocks, smap_daily, smap_dates, temp_stack,
                               grid_points, elev_cache)

    results_df.to_csv(CSV_PATH, index=False)
    print(f"\n[CSV] Saved → {CSV_PATH}", flush=True)

    plot_flood_map(results_df, save_path=MAP_PATH)
    plot_probability_distribution(results_df, save_path=DIST_PATH)
    print_summary(results_df)
    print("\n[COMPLETE] All done!", flush=True)
    return results_df

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[CANCELLED]", flush=True); sys.exit(130)
    except Exception as e:
        print(f"\n[FATAL] {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)
