#!/usr/bin/env python3
"""
worker_inference.py

Usage example (SLURM/CLI):
python worker_inference.py --scene-csv scene_tasks.csv --row-index 0 \
    --saved-model /path/to/saved_model --results-dir /shared/results --chip-px 400 --stride 200

This script:
 - opens the scene COG via rasterio (VSICURL),
 - iterates tiles (400x400 px) with stride 200,
 - preprocesses to match training (intensity -> dB -> clip -> min-max),
 - runs TF SavedModel classification,
 - writes per-chip small JSONs into results-dir (one file per chip).
"""

import os
import argparse
import csv
import json
import math
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import rasterio
from rasterio.windows import Window
from rasterio.errors import RasterioIOError
from urllib.parse import urlparse

# PARAMETERS
EPS = 1e-6
DB_MIN = -40.0  # dB clip lower bound (tune if your training used a different range)
DB_MAX = 0.0    # dB clip upper bound

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-csv", required=True, help="CSV with scene rows (scene_id,scene_href,datetime)")
    p.add_argument("--row-index", type=int, required=True, help="0-based index into scene CSV")
    p.add_argument("--saved-model", required=True, help="Path to TF SavedModel directory")
    p.add_argument("--results-dir", required=True, help="Directory to write JSON results")
    p.add_argument("--chip-px", type=int, default=400, help="chip width/height in pixels (default 400)")
    p.add_argument("--stride", type=int, default=200, help="chip stride in pixels (default 200)")
    p.add_argument("--batch-size", type=int, default=16, help="how many chips to batch for model inference")
    p.add_argument("--max-tiles", type=int, default=None, help="for testing: max tiles to process from this scene")
    return p.parse_args()

def load_scene_row(csv_path, index):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if index < 0 or index >= len(rows):
        raise IndexError("row-index out of range")
    return rows[index]

def open_raster(href, max_retries=3, backoff=2.0):
    """
    Open a COG URL via rasterio. Uses vsicurl if HTTP(s).
    """
    tries = 0
    last_err = None
    while tries < max_retries:
        try:
            if href.startswith("http://") or href.startswith("https://"):
                # vsicurl prefix allows HTTP range reads via GDAL
                rio_path = "/vsicurl/" + href
            else:
                rio_path = href
            src = rasterio.open(rio_path)
            return src
        except RasterioIOError as e:
            last_err = e
            tries += 1
            time.sleep(backoff ** tries)
    raise last_err

def iter_windows(src, chip_px, stride):
    """Yield windows (col_off, row_off, width, height) that are fully inside raster."""
    width = src.width
    height = src.height
    row = 0
    while row + chip_px <= height:
        col = 0
        while col + chip_px <= width:
            yield Window(col_off=col, row_off=row, width=chip_px, height=chip_px)
            col += stride
        row += stride

def read_chip(src, window, out_shape):
    """Read single band (band 1) and resample to out_shape (H,W)."""
    # read with out_shape to ensure chip_px x chip_px
    arr = src.read(
        1,
        window=window,
        out_shape=out_shape,
        resampling=rasterio.enums.Resampling.bilinear
    )
    return arr

def preprocess_chip(arr):
    """
    Preprocess to match training-style chips:
      - ensure float32
      - replace non-positive / nodata with small eps
      - convert intensity -> dB: 10 * log10(arr)
      - clip to DB_MIN/DB_MAX
      - min-max normalize to [0,1]
      - return shape (H, W, 1), dtype float32
    """
    arr = arr.astype(np.float32)
    arr[arr <= 0.0] = EPS
    db = 10.0 * np.log10(arr + EPS)
    db = np.clip(db, DB_MIN, DB_MAX)
    # Scale to 0-255 to match training
    norm = ((db - DB_MIN) / (DB_MAX - DB_MIN) * 255.0).astype(np.uint8)
    # Repeat to 3 channels for RGB compatibility
    norm = np.stack([norm, norm, norm], axis=-1)  # Shape: (H, W, 3)
    return norm

def batch_predict(model, batch_array):
    """
    model: loaded Keras model
    batch_array: numpy array shape (N,H,W,3)
    returns: numpy array of probabilities for class 1 (oil) (N,)
    """
    logits = model(batch_array, training=False).numpy()
    
    # logits shape: (N, 2) - convert to probabilities
    probs = tf.nn.softmax(logits, axis=1).numpy()
    # Return probability of class 1 (oil)
    return probs[:, 1]

def save_result_json(results_dir, scene_id, tile_index, bbox_geo, datetime_acq, score):
    os.makedirs(results_dir, exist_ok=True)
    fname = f"{scene_id}_tile_{tile_index:08d}.json"
    path = os.path.join(results_dir, fname)
    rec = {
        "scene_id": scene_id,
        "tile_index": tile_index,
        "bbox": bbox_geo,  # keep scene pixel coords or optionally convert to lon/lat
        "datetime": datetime_acq,
        "score": float(score),
        "created_utc": datetime.utcnow().isoformat() + "Z"
    }
    with open(path, "w") as f:
        json.dump(rec, f)
    return path

def main():
    args = parse_args()
    row = load_scene_row(args.scene_csv, args.row_index)
    scene_id = row["scene_id"]
    href = row["scene_href"]
    datetime_acq = row.get("datetime", "")

    print(f"[{scene_id}] Opening scene: {href}")
    src = open_raster(href)

    # load TF model once
    print("Loading Keras model...")
    model = tf.keras.models.load_model(args.saved_model)
    print("Model loaded.")

    chip_px = args.chip_px
    stride = args.stride
    batch_size = args.batch_size

    windows = list(iter_windows(src, chip_px, stride))
    n_tiles = len(windows)
    print(f"Scene dimensions: {src.width}x{src.height}. Will process {n_tiles} tiles (chip {chip_px}, stride {stride}).")

    tile_idx = 0
    batch_images = []
    batch_tileinfo = []

    # iterate windows
    for win in windows:
        if args.max_tiles and tile_idx >= args.max_tiles:
            break
        # read chip (ensure it is chip_px x chip_px)
        try:
            arr = read_chip(src, win, out_shape=(chip_px, chip_px))
        except Exception as e:
            print(f"Read error at tile {tile_idx}: {e}")
            tile_idx += 1
            continue

        # optional quick skip: if arr is all nodata or very low variance, skip
        if np.all(arr == 0) or np.nanstd(arr) < 1e-6:
            # write a small result with score 0 or skip entirely; we'll skip to save compute
            tile_idx += 1
            continue

        proc = preprocess_chip(arr)  # shape (H,W,1)
        batch_images.append(proc)
        # store bbox as pixel coordinates (col_off, row_off, width, height)
        bbox_geo = [int(win.col_off), int(win.row_off), int(win.col_off + win.width), int(win.row_off + win.height)]
        batch_tileinfo.append((tile_idx, bbox_geo))
        tile_idx += 1

        # if batch full, run inference
        if len(batch_images) >= batch_size:
            batch = np.stack(batch_images, axis=0)  # (N,H,W,1)
            preds = batch_predict(model, batch)
            for i, p in enumerate(preds):
                tidx, bbox = batch_tileinfo[i]
                outpath = save_result_json(args.results_dir, scene_id, tidx, bbox, datetime_acq, float(p))
            batch_images = []
            batch_tileinfo = []

    # final partial batch
    if batch_images:
        batch = np.stack(batch_images, axis=0)
        preds = batch_predict(model, batch)
        for i, p in enumerate(preds):
            tidx, bbox = batch_tileinfo[i]
            outpath = save_result_json(args.results_dir, scene_id, tidx, bbox, datetime_acq, float(p))

    print(f"Scene {scene_id} done. Processed {tile_idx} tiles.")
    src.close()

if __name__ == "__main__":
    main()
