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
import planetary_computer
from PIL import Image
from global_land_mask import globe

# PARAMETERS
EPS = 1e-6
DB_MIN = -40.0  # dB clip lower bound (tune if your training used a different range)
DB_MAX = 0.0    # dB clip upper bound

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-csv", required=True, help="CSV with scene rows (scene_id,scene_href,datetime,bbox_*)")
    p.add_argument("--row-index", type=int, required=True, help="0-based index into scene CSV")
    p.add_argument("--saved-model", required=True, help="Path to TF SavedModel directory")
    p.add_argument("--results-dir", required=True, help="Directory to write JSON results")
    p.add_argument("--chip-px", type=int, default=400, help="chip width/height in pixels (default 400)")
    p.add_argument("--stride", type=int, default=200, help="chip stride in pixels (default 200)")
    p.add_argument("--batch-size", type=int, default=16, help="how many chips to batch for model inference")
    p.add_argument("--max-tiles", type=int, default=None, help="for testing: max tiles to process from this scene")
    p.add_argument("--threshold", type=float, default=0.7, help="minimum probability to save result (default 0.7)")
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
    Re-signs URL if it's from Planetary Computer.
    """
    tries = 0
    last_err = None
    while tries < max_retries:
        try:
            # Re-sign URL if it's from Planetary Computer
            if "blob.core.windows.net" in href or "planetarycomputer" in href:
                try:
                    href = planetary_computer.sign(href)
                except Exception as sign_err:
                    print(f"Warning: Could not sign URL: {sign_err}")
            
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
            if "403" in str(e):
                print(f"403 error, attempt {tries}/{max_retries}. Retrying with fresh signature...")
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
    # """
    # Preprocess DN values to match training JPEG distribution.
    # Target: mean≈144, std≈63 on [0, 255] scale
    # """
    # arr = arr.astype(np.float32)
    
    # # Handle invalid values
    # arr[arr <= 0.0] = np.nanmedian(arr) if np.any(arr > 0) else 100.0
    # arr[np.isnan(arr)] = np.nanmedian(arr)
    
    # # Your data stats: mean=99.58, range=[0, 347]
    # # To match training mean=144, we need to scale up
    
    # # Clip to reasonable DN range for ocean
    # arr_clipped = np.clip(arr, 0, 250)  # Avoid extreme outliers
    
    # # Scale to approximate training distribution
    # # Linear mapping: [0, 250] → [0, 255] with shift to match mean
    # norm = (arr_clipped / 250.0 * 255.0).astype(np.uint8)
    
    # # Apply histogram adjustment to match training mean
    # current_mean = np.mean(norm)
    # target_mean = 144.0
    # adjustment = target_mean - current_mean
    # norm = np.clip(norm.astype(np.float32) + adjustment, 0, 255).astype(np.uint8)
    
    # # Convert to 3-channel RGB
    # norm = np.stack([norm, norm, norm], axis=-1)
    
    # return norm
    """
    Preprocess Sentinel-1 GRD DN values to match training data.
    Assumes DN values need calibration to sigma0.
    """
    arr = arr.astype(np.float32)
    
    # Handle invalid values
    arr[arr <= 0.0] = EPS
    arr[np.isnan(arr)] = EPS
    
    # Calibration: DN to sigma0 (linear power)
    # For Sentinel-1 GRD, typical formula is: sigma0 = (DN^2) / calibration_constant
    # Or simpler approximation: sigma0 = DN^2 / 10000.0
    # This converts DN [0-350] to sigma0 [0-12] which is still too high
    
    # Better approach: Use empirical normalization based on your data
    # Your DN range: [0, 347], mean: 99.58
    # Target sigma0 range for ocean: [0.001, 0.1] (linear)
    
    # Empirical calibration (tune these values):
    sigma0_linear = (arr / 100.0) ** 2 / 100.0
    # This maps DN=100 → sigma0≈0.01 which is reasonable for ocean
    
    # Convert to dB
    db = 10.0 * np.log10(sigma0_linear + EPS)
    
    # Clip to typical ocean SAR range
    db = np.clip(db, -30.0, 5.0)
    
    # Normalize to 0-255
    db_min, db_max = -30.0, 5.0
    norm = ((db - db_min) / (db_max - db_min) * 255.0).astype(np.uint8)
    
    # Convert to 3-channel RGB
    norm = np.stack([norm, norm, norm], axis=-1)
    
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

def save_result_json(results_dir, scene_id, tile_index, bbox_pixel, bbox_geo, datetime_acq, score):
    os.makedirs(results_dir, exist_ok=True)
    fname = f"{scene_id}_tile_{tile_index:08d}.json"
    path = os.path.join(results_dir, fname)
    rec = {
        "scene_id": scene_id,
        "tile_index": tile_index,
        "bbox_pixel": bbox_pixel,  # [col_min, row_min, col_max, row_max]
        "bbox_geo": bbox_geo,      # [lon_min, lat_min, lon_max, lat_max]
        "datetime": datetime_acq,
        "score": float(score),
        "created_utc": datetime.utcnow().isoformat() + "Z"
    }
    with open(path, "w") as f:
        json.dump(rec, f)
    return path

def calculate_chip_bbox_geo(scene_bbox, chip_bbox_pixel, raster_width, raster_height):
    """
    Calculate geographic bbox for a chip based on scene bbox and pixel position.
    
    Args:
        scene_bbox: [lon_min, lat_min, lon_max, lat_max] of full scene
        chip_bbox_pixel: [col_min, row_min, col_max, row_max] in pixels
        raster_width: total width of raster in pixels
        raster_height: total height of raster in pixels
    
    Returns:
        [lon_min, lat_min, lon_max, lat_max] for the chip
    """
    scene_lon_min, scene_lat_min, scene_lon_max, scene_lat_max = scene_bbox
    col_min, row_min, col_max, row_max = chip_bbox_pixel
    
    # Calculate pixel size in degrees
    lon_per_pixel = (scene_lon_max - scene_lon_min) / raster_width
    lat_per_pixel = (scene_lat_max - scene_lat_min) / raster_height
    
    # Calculate chip geographic bbox
    chip_lon_min = scene_lon_min + (col_min * lon_per_pixel)
    chip_lon_max = scene_lon_min + (col_max * lon_per_pixel)
    chip_lat_max = scene_lat_max - (row_min * lat_per_pixel)  # Note: rows go down
    chip_lat_min = scene_lat_max - (row_max * lat_per_pixel)
    
    return [chip_lon_min, chip_lat_min, chip_lon_max, chip_lat_max]

def bbox_on_land(bbox_geo):
    """
    Check if bbox intersects with land by checking corners and center.
    """
    chip_lon_min, chip_lat_min, chip_lon_max, chip_lat_max = bbox_geo
    
    # Calculate center
    center_lat = (chip_lat_min + chip_lat_max) / 2
    center_lon = (chip_lon_min + chip_lon_max) / 2
    
    # Check corners and center
    points = [
        (chip_lat_max, chip_lon_min),  # top-left
        (chip_lat_max, chip_lon_max),  # top-right
        (chip_lat_min, chip_lon_min),  # bottom-left
        (chip_lat_min, chip_lon_max),  # bottom-right
        (center_lat, center_lon),      # center
    ]
    
    for lat, lon in points:
        if globe.is_land(lat, lon):
            return True
    
    return False

def main():
    args = parse_args()
    row = load_scene_row(args.scene_csv, args.row_index)
    scene_id = row["scene_id"]
    href = row["scene_href"]
    datetime_acq = row.get("datetime", "")
    
    # Load scene bounding box from CSV
    scene_bbox = [
        float(row["bbox_lon_min"]),
        float(row["bbox_lat_min"]),
        float(row["bbox_lon_max"]),
        float(row["bbox_lat_max"])
    ]

    print(f"[{scene_id}] Opening scene: {href}")
    print(f"Scene bbox: {scene_bbox}")
    print(f"Using threshold: {args.threshold}")
    src = open_raster(href)

    # load TF model once
    print("Loading Keras model...")
    model = tf.keras.models.load_model(args.saved_model)
    print("Model loaded.")

    chip_px = args.chip_px
    stride = args.stride
    batch_size = args.batch_size
    
    raster_width = src.width
    raster_height = src.height

    windows = list(iter_windows(src, chip_px, stride))
    n_tiles = len(windows)
    print(f"Scene dimensions: {raster_width}x{raster_height}. Will process {n_tiles} tiles (chip {chip_px}, stride {stride}).")

    tile_idx = 0
    batch_images = []
    batch_tileinfo = []
    saved_count = 0  # Track how many were saved

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
            tile_idx += 1
            continue
        
        # Calculate bbox BEFORE preprocessing to save computation
        bbox_pixel = [
            int(win.col_off), 
            int(win.row_off), 
            int(win.col_off + win.width), 
            int(win.row_off + win.height)
        ]
        
        bbox_geo = calculate_chip_bbox_geo(
            scene_bbox, 
            bbox_pixel, 
            raster_width, 
            raster_height
        )

        # Skip if on land (before expensive preprocessing)
        if bbox_on_land(bbox_geo):
            tile_idx += 1
            continue

        # Now do the expensive preprocessing
        proc = preprocess_chip(arr)  # shape (H,W,3)
        batch_images.append(proc)
        batch_tileinfo.append((tile_idx, bbox_pixel, bbox_geo))
        tile_idx += 1

        # if batch full, run inference
        if len(batch_images) >= batch_size:
            batch = np.stack(batch_images, axis=0)  # (N,H,W,3)
            preds = batch_predict(model, batch)
            for i, p in enumerate(preds):
                tidx, bbox_pix, bbox_g = batch_tileinfo[i]
                # Only save if above threshold
                if p >= args.threshold:
                    outpath = save_result_json(
                        args.results_dir, 
                        scene_id, 
                        tidx, 
                        bbox_pix, 
                        bbox_g, 
                        datetime_acq, 
                        float(p)
                    )
                    sample_path = os.path.join(args.results_dir, f"sample_chip_inference_{tidx}.jpg")
                    Image.fromarray(proc[:,:,0]).save(sample_path)
                    print(f"Saved sample chip to {sample_path}")
                    saved_count += 1
            batch_images = []
            batch_tileinfo = []

    # final partial batch
    if batch_images:
        batch = np.stack(batch_images, axis=0)
        preds = batch_predict(model, batch)
        for i, p in enumerate(preds):
            tidx, bbox_pix, bbox_g = batch_tileinfo[i]
            # Only save if above threshold
            if p >= args.threshold:
                outpath = save_result_json(
                    args.results_dir, 
                    scene_id, 
                    tidx, 
                    bbox_pix, 
                    bbox_g, 
                    datetime_acq, 
                    float(p)
                )
                sample_path = os.path.join(args.results_dir, f"sample_chip_inference_{tidx}.jpg")
                Image.fromarray(proc[:,:,0]).save(sample_path)
                print(f"Saved sample chip to {sample_path}")
                saved_count += 1

    print(f"Scene {scene_id} done. Processed {tile_idx} tiles, saved {saved_count} detections above threshold {args.threshold}.")
    src.close()

if __name__ == "__main__":
    main()
