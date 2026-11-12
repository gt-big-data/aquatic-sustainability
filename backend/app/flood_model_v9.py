"""
Adapter: wraps the V9 flood model for use in the Flask API.

Exposes run_flood_inference(center_lat, center_lon) returning the same
dict format as the old V3 model:
    {"center": {"lat": ..., "lon": ...}, "points": [{"lat","lon","prob","risk"}, ...]}

Resilience strategy (mirrors V3's FAST_START_FLOOD_DATA approach):
  GPM  → V9 disk cache → V3 global cache → live download → synthetic zeros
  SMAP → V9 disk cache → live download → synthetic uniform values
  ERA5 → V9 disk cache → live download → empty dict (V9 zero-pads gracefully)

IMPORTANT: Do NOT call v9.setup_earthdata_auth() here.
  entrypoint.sh already writes ~/.netrc from EARTHDATA_USERNAME / EARTHDATA_PASSWORD
  env vars. Calling setup_earthdata_auth() would overwrite those valid credentials
  with the V9 script's hardcoded ones, breaking authentication.
"""
import os
import sys
import threading
from datetime import datetime

import numpy as np
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
_NEW_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "new_model")
)
if _NEW_MODEL_DIR not in sys.path:
    sys.path.insert(0, _NEW_MODEL_DIR)

# Top-level import runs V9's startup checks (missing packages/weights → sys.exit).
# All required packages and weight files must be present before the app starts.
import flood_predict_v9 as v9  # noqa: E402

try:
    import xarray as xr
    _HAS_XR = True
except ImportError:
    _HAS_XR = False

# V3 global GPM cache produced by global_gpm_loader — same (32, 1800, 3600) format as V9
_V3_GPM_CACHE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data_cache/global_gpm_4days.npy")
)

# ── Module-level singletons ──────────────────────────────────────────────────
_lock = threading.Lock()
_model = None
_scaler_precip = None
_scaler_soil = None
_device = None
_gpm_blocks = None
_smap_daily = None
_smap_dates = None
_temp_stack = None
_data_date = None


# ── Data loading helpers with fallbacks ─────────────────────────────────────

def _load_gpm_safe() -> list:
    """GPM blocks: V9 cache → V3 global cache → live download → synthetic zeros."""
    # V9 disk cache (checked internally by load_or_download_gpm)
    if os.path.exists(v9.GPM_CACHE):
        try:
            blocks = list(np.load(v9.GPM_CACHE))
            if len(blocks) == v9.SEQ_LEN:
                print("[V9 ADAPTER] GPM loaded from V9 disk cache", flush=True)
                return blocks
        except Exception as exc:
            print(f"[V9 ADAPTER] V9 GPM cache corrupt: {exc}", flush=True)

    # V3 global cache — identical (32, 1800, 3600) numpy format
    if os.path.exists(_V3_GPM_CACHE):
        try:
            arr = np.load(_V3_GPM_CACHE)
            if arr.shape[0] < v9.SEQ_LEN:
                pad = np.zeros((v9.SEQ_LEN - arr.shape[0], 1800, 3600), dtype=np.float32)
                arr = np.concatenate([arr, pad])
            print("[V9 ADAPTER] GPM loaded from V3 global cache fallback", flush=True)
            return list(arr[:v9.SEQ_LEN].astype(np.float32))
        except Exception as exc:
            print(f"[V9 ADAPTER] V3 GPM cache failed: {exc}", flush=True)

    # Live download (may be slow in Cloud Run but respects the env-var credentials
    # already in ~/.netrc — DO NOT call setup_earthdata_auth() before this)
    try:
        print("[V9 ADAPTER] Attempting live GPM download ...", flush=True)
        return v9.load_or_download_gpm()
    except Exception as exc:
        print(f"[V9 ADAPTER] GPM download failed: {exc} — falling back to synthetic zeros", flush=True)

    return [np.zeros((1800, 3600), dtype=np.float32) for _ in range(v9.SEQ_LEN)]


def _load_smap_safe():
    """SMAP daily stack: V9 cache → live download → synthetic uniform values."""
    smap_dates = pd.date_range(v9.SOIL_START, v9.SOIL_END, freq="D").date.tolist()

    if os.path.exists(v9.SMAP_CACHE):
        try:
            with open(v9.SMAP_CACHE, "rb") as f:
                import pickle
                smap_daily = pickle.load(f)
            ok = sum(v is not None for v in smap_daily.values())
            print(f"[V9 ADAPTER] SMAP loaded from V9 cache ({ok}/{len(smap_dates)} days)", flush=True)
            return smap_daily, smap_dates
        except Exception as exc:
            print(f"[V9 ADAPTER] V9 SMAP cache corrupt: {exc}", flush=True)

    try:
        print("[V9 ADAPTER] Attempting live SMAP download ...", flush=True)
        return v9.load_or_download_smap()
    except Exception as exc:
        print(f"[V9 ADAPTER] SMAP download failed: {exc} — using synthetic soil moisture", flush=True)

    # Synthetic fallback: uniform 0.3 m³/m³, mirrors V3's np.random.uniform approach
    if _HAS_XR:
        smap_lats = np.linspace(90, -90, 720, dtype=np.float32)
        smap_lons = np.linspace(-180, 180, 1440, dtype=np.float32)
        synthetic_sm = np.full((720, 1440), 0.3, dtype=np.float32)
        synthetic_da = xr.DataArray(
            synthetic_sm, dims=["lat", "lon"],
            coords={"lat": smap_lats, "lon": smap_lons}
        )
        smap_daily = {d: synthetic_da for d in smap_dates}
    else:
        smap_daily = {d: None for d in smap_dates}

    print("[V9 ADAPTER] Using synthetic SMAP data", flush=True)
    return smap_daily, smap_dates


def _load_era5_safe():
    """ERA5 temp: disk cache → live download → None (V9 zero-pads gracefully)."""
    try:
        return v9.download_era5_temp()
    except Exception as exc:
        print(f"[V9 ADAPTER] ERA5 unavailable: {exc} — temp will be zero-padded", flush=True)
        return None


# ── Lazy loader ─────────────────────────────────────────────────────────────

def _ensure_loaded() -> None:
    """Load the V9 model and daily satellite data; refresh data once per UTC day."""
    global _model, _scaler_precip, _scaler_soil, _device
    global _gpm_blocks, _smap_daily, _smap_dates, _temp_stack, _data_date

    today = datetime.utcnow().date()

    with _lock:
        if _model is None:
            # entrypoint.sh already wrote ~/.netrc with valid EARTHDATA_* env vars.
            # Skip v9.setup_earthdata_auth() and v9.setup_cds_credentials() here —
            # calling them would overwrite those credentials with V9's hardcoded ones.
            _model, _scaler_precip, _scaler_soil, _device = v9.load_model_and_scalers()

        if _data_date != today:
            _gpm_blocks = _load_gpm_safe()
            _smap_daily, _smap_dates = _load_smap_safe()
            era5_path = _load_era5_safe()
            _temp_stack = v9.build_temp_stack(era5_path) if era5_path else {}
            _data_date = today


# ── Inference ────────────────────────────────────────────────────────────────

def _local_grid(center_lat: float, center_lon: float, radius: int = 3, step: float = 1.0):
    """Return (lat, lon) grid points within `radius` degrees of center at `step` spacing."""
    pts = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            pts.append((round(center_lat + di * step, 4),
                        round(center_lon + dj * step, 4)))
    return pts


def run_flood_inference(center_lat: float, center_lon: float) -> dict:
    """
    Run the V9 DualCNNLSTM flood model over a local grid around (center_lat, center_lon).

    Returns:
        {
            "center": {"lat": float, "lon": float},
            "points": [{"lat": float, "lon": float, "prob": float, "risk": str}, ...]
        }
    """
    _ensure_loaded()

    grid_points = _local_grid(center_lat, center_lon)
    elev_cache = v9.precompute_elevation(grid_points)

    results_df = v9.run_inference(
        _model, _scaler_precip, _scaler_soil, _device,
        _gpm_blocks, _smap_daily, _smap_dates, _temp_stack,
        grid_points, elev_cache,
    )

    points = []
    for _, row in results_df.iterrows():
        if pd.isna(row["prob"]):
            continue
        p = float(row["prob"])
        if p >= 0.75:
            risk = "high"
        elif p >= 0.5:
            risk = "warning"
        elif p >= 0.3:
            risk = "medium"
        else:
            risk = "low"
        points.append({
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "prob": p,
            "risk": risk,
        })

    return {
        "center": {"lat": center_lat, "lon": center_lon},
        "points": points,
    }
