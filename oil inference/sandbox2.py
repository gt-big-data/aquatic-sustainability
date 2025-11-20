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

def pixel_to_lonlat(src, col, row):
    """
    Convert pixel coordinates to lon/lat using the raster's transform.
    
    Args:
        src: rasterio dataset
        col: column (x) pixel coordinate
        row: row (y) pixel coordinate
    
    Returns:
        (lon, lat) tuple
    """
    lon, lat = src.xy(row, col)
    return lon, lat

def bbox_to_lonlat(src, bbox):
    """
    Convert pixel bbox to geographic bbox.
    
    Args:
        src: rasterio dataset
        bbox: [col_min, row_min, col_max, row_max]
    
    Returns:
        [lon_min, lat_min, lon_max, lat_max]
    """
    col_min, row_min, col_max, row_max = bbox
    
    # Use transform to get corner coordinates (not pixel centers)
    transform = src.transform
    
    # Top-left corner
    lon_min, lat_max = transform * (col_min, row_min)
    
    # Bottom-right corner
    lon_max, lat_min = transform * (col_max, row_max)
    
    return [lon_min, lat_min, lon_max, lat_max]

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

href = "https://sentinel1euwest.blob.core.windows.net/s1-grd/GRD/2025/10/1/IW/DV/S1A_IW_GRDH_1SDV_20251001T235418_20251001T235443_061239_07A347_D037/measurement/iw-vv.tiff"

href = planetary_computer.sign(href)

src = open_raster(href)

# print(src)

print(pixel_to_lonlat(src, 0, 0))

print(src.transform * (0, 0))

print(f"CRS: {src.crs}")
print(f"Transform: {src.transform}")

# scene_bbox = [-89.439354,29.780401,-86.495453,31.694811]
# chip_bbox_pixel = [8600, 20400, 9000, 20800]
# raster_width = 25559
# raster_height = 16714

# print(calculate_chip_bbox_geo(scene_bbox, chip_bbox_pixel, raster_width, raster_height))