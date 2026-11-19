"""Build the flood model's input data (GPM rainfall + SMAP soil moisture),
cropped to the region the flood map serves, and optionally upload it to GCS.

The web server can't build this itself: the raw NASA downloads are many GB and
take minutes. Run this daily (Cloud Run Job on Cloud Scheduler) instead:

    python scripts/build_flood_data.py --upload          # from backend/

Env:
    EARTHDATA_USERNAME / EARTHDATA_PASSWORD   NASA Earthdata login (required)
    FLOOD_DATA_BUCKET                         GCS bucket (required with --upload)
    FLOOD_DATA_BLOB                           object name (default flood/region_latest.npz)

Output (always): data_cache/flood_region.npz, which the server loads directly.

Processing mirrors app.flood_model.build_global_gpm_4days and
app.global_smap_loader.build_global_smap_21days, except that it works one day
at a time (deleting raw files as it goes) and crops each grid immediately, so
peak disk + memory stays around 1-2 GB instead of ~20 GB.
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_DIR)

import earthaccess  # noqa: E402
import xarray as xr  # noqa: E402

from app import flood_model as fm  # noqa: E402
from app.global_gpm_loader import ensure_earthdata_auth  # noqa: E402
from app.global_smap_loader import (  # noqa: E402
    bilinear_resample_to_025deg,
    load_smap_surface,
    reproject_to_025deg,
)


def _crop(grid, crop):
    r0, r1, c0, c1 = crop
    return np.ascontiguousarray(grid[r0:r1, c0:c1], dtype=np.float32)


def build_gpm(work_dir, today):
    """32 three-hour rainfall blocks (4 days), north-first, cropped to GPM_CROP."""
    latest = today - timedelta(days=fm.GPM_DAY_DELAY)
    dates = sorted(latest - timedelta(days=i) for i in range(fm.GPM_N_DAYS))
    r0, r1, c0, c1 = fm.GPM_CROP
    print(f"[GPM] Dates {dates[0]} → {dates[-1]}", flush=True)

    blocks = []
    for d in dates:
        day_dir = tempfile.mkdtemp(dir=work_dir)
        files = fm.search_and_download_gpm_global(d, day_dir)
        frames = []
        for fp in files:
            try:
                raw = fm.read_gpm_precip_grid(fp).values
                frames.append(_crop(fm.orient_gpm_frame(raw), fm.GPM_CROP))
            except Exception as e:
                print(f"[GPM] Skip {os.path.basename(fp)}: {e}", flush=True)
        shutil.rmtree(day_dir, ignore_errors=True)

        print(f"[GPM] {d}: {len(frames)} half-hour frames", flush=True)
        if not frames:
            blocks.extend(np.zeros((r1 - r0, c1 - c0), np.float32) for _ in range(8))
            continue
        blocks.extend(fm.accumulate_3hr_grids_global(frames))

    stack = np.stack(blocks[:32]).astype(np.float32)
    if len(stack) < 32:
        pad = np.zeros((32 - len(stack), r1 - r0, c1 - c0), np.float32)
        stack = np.concatenate([stack, pad])
    return stack, dates


def build_smap(work_dir, today):
    """21 daily-mean soil-moisture grids, north-first, cropped to SMAP_CROP."""
    latest = today - timedelta(days=fm.SMAP_DAY_DELAY + 1)
    dates = sorted(latest - timedelta(days=i) for i in range(fm.SMAP_N_DAYS))
    r0, r1, c0, c1 = fm.SMAP_CROP
    print(f"[SMAP] Dates {dates[0]} → {dates[-1]}", flush=True)

    stack = []
    for d in dates:
        day_dir = tempfile.mkdtemp(dir=work_dir)
        try:
            results = earthaccess.search_data(
                short_name="SPL4SMAU", version="008",
                temporal=(f"{d}T00:00:00", f"{d}T23:59:59"),
            )
            paths = earthaccess.download(results, local_path=day_dir)
        except Exception as e:
            print(f"[SMAP] Download failed for {d}: {e}", flush=True)
            paths = []

        tag = d.strftime("%Y%m%d")
        files = [str(p) for p in paths
                 if str(p).lower().endswith(".h5")
                 and (m := re.search(r"(\d{8})", os.path.basename(str(p)))) and m.group(1) == tag]

        grids = []
        for fp in files:
            try:
                grids.append(bilinear_resample_to_025deg(load_smap_surface(fp)).values)
            except Exception as e:
                print(f"[SMAP] Skip {os.path.basename(fp)}: {e}", flush=True)
        shutil.rmtree(day_dir, ignore_errors=True)

        print(f"[SMAP] {d}: {len(grids)} three-hour files", flush=True)
        if not grids:
            stack.append(np.full((r1 - r0, c1 - c0), np.nan, np.float32))
            continue

        daily = np.nanmean(grids, axis=0).astype(np.float32)
        da = xr.DataArray(daily, dims=["lat", "lon"], coords={
            "lat": np.linspace(90, -90, daily.shape[0]),
            "lon": np.linspace(-180, 180, daily.shape[1]),
        })
        stack.append(_crop(reproject_to_025deg(da), fm.SMAP_CROP))

    return np.stack(stack).astype(np.float32), dates


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upload", action="store_true", help="upload to gs://$FLOOD_DATA_BUCKET")
    ap.add_argument("--out", default=fm.REGION_NPZ)
    ap.add_argument("--date", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
                    default=datetime.utcnow().date(), help="build as if today were this date (UTC)")
    args = ap.parse_args()

    if args.upload and not fm.FLOOD_DATA_BUCKET:
        sys.exit("--upload needs FLOOD_DATA_BUCKET set")

    ensure_earthdata_auth()
    earthaccess.login(strategy="netrc")

    work_dir = tempfile.mkdtemp(prefix="flood_build_")
    try:
        gpm, gpm_dates = build_gpm(work_dir, args.date)
        smap, smap_dates = build_smap(work_dir, args.date)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    # Refuse to publish a build that's mostly empty: better to keep serving
    # yesterday's data than to overwrite it with zeros.
    gpm_days_ok = sum(bool(gpm[i * 8:(i + 1) * 8].any()) for i in range(4))
    smap_days_ok = int(np.sum(~np.isnan(smap).all(axis=(1, 2))))
    print(f"[BUILD] GPM days with data: {gpm_days_ok}/4, SMAP days with data: {smap_days_ok}/21", flush=True)
    if gpm_days_ok < 3 or smap_days_ok < 15:
        sys.exit("[BUILD] Too much data missing — not publishing.")

    tmp_out = args.out + ".tmp.npz"
    np.savez_compressed(
        tmp_out,
        gpm=gpm, gpm_row0=fm.GPM_CROP[0], gpm_col0=fm.GPM_CROP[2],
        smap=smap, smap_row0=fm.SMAP_CROP[0], smap_col0=fm.SMAP_CROP[2],
        built_at=datetime.utcnow().isoformat(timespec="seconds"),
        gpm_dates=[str(d) for d in gpm_dates],
        smap_dates=[str(d) for d in smap_dates],
    )
    os.replace(tmp_out, args.out)
    print(f"[BUILD] Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB) "
          f"GPM {gpm.shape} SMAP {smap.shape}", flush=True)

    if args.upload:
        from google.cloud import storage
        blob = storage.Client().bucket(fm.FLOOD_DATA_BUCKET).blob(fm.FLOOD_DATA_BLOB)
        blob.upload_from_filename(args.out)
        print(f"[BUILD] Uploaded gs://{fm.FLOOD_DATA_BUCKET}/{fm.FLOOD_DATA_BLOB}", flush=True)


if __name__ == "__main__":
    main()
