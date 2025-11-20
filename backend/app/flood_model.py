# app/flood_model.py

import os
import pickle
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn

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

GPM_DIR = "./gpm_temp"
DATA_CACHE_DIR = "./data_cache"
MODEL_PATH = "DualCNNLSTM_best_model_V3.pt"
SCALER_PATH = "DualCNNLSTM_scaler_V3.pkl"

os.makedirs(GPM_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLOBAL_SMAP = None
GLOBAL_TIMESTAMP = None

# =====================================================================
# MODEL ARCHITECTURE (same as in your script)
# =====================================================================

class DualCNNLSTM(nn.Module):
    # -- paste your class from flood_predictor (2).py here unchanged --
    ...

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
