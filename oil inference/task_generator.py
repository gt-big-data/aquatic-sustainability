#!/usr/bin/env python3
"""
task_generator.py
Query MS Planetary Computer STAC for Sentinel-1 GRD (VV) scenes intersecting the oceans,
and write a CSV with one scene per row including geographic bounding box.

Outputs: scene_tasks.csv, scene_coverage_map.png
"""

import csv
from datetime import datetime
from pystac_client import Client
import planetary_computer
import geopandas as gpd
from shapely.geometry import box, mapping
from shapely.ops import unary_union
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mplPolygon

# PARAMETERS - adjust as needed
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
START_DATE = "2025-09-01"
END_DATE = "2025-11-01"
OUT_CSV = "scene_tasks.csv"
OUT_MAP = "scene_coverage_map.png"
MAX_ITEMS_PER_MONTH = None  # set to int to limit for testing

def build_ocean_polygon():
    """
    Build a simplified ocean polygon by subtracting naturalearth land from global bbox.
    Requires geopandas.
    """
    world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
    land_union = unary_union(world.geometry)
    # global bbox (lon/lat)
    global_box = box(-180.0, -90.0, 180.0, 90.0)
    ocean = global_box.difference(land_union)
    # optionally simplify
    ocean = gpd.GeoSeries([ocean]).simplify(0.05).iloc[0]
    return ocean

def main():
    print("Building ocean polygon (this may take a few seconds)...")
    ocean = build_ocean_polygon()
    stac = Client.open(STAC_URL)
    bbox = ocean.bounds  # (minx, miny, maxx, maxy)

    # Write header to the CSV file first with bbox columns
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_id", "scene_href", "datetime", "bbox_lon_min", "bbox_lat_min", "bbox_lon_max", "bbox_lat_max"])

    total_scenes = 0
    all_scene_geometries = []  # Collect all scene geometries for visualization
    
    # Generate monthly date ranges to process in chunks
    date_ranges = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')

    for i in range(len(date_ranges) - 1):
        start = date_ranges[i]
        end = date_ranges[i+1]
        date_range_str = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        print(f"\nSearching STAC for date range: {date_range_str}...")

        search = stac.search(
            collections=["sentinel-1-grd"],
            intersects=mapping(ocean),
            datetime=date_range_str,
            max_items=1000,
        )
        
        # Check how many items were found by the API for this query
        matched_count = search.matched()
        print(f"API found {matched_count} matching items for this period.")
        if matched_count == 0:
            continue

        rows = []
        count = 0
        for item in search.items():
            # Client-side filter: check for VV polarization
            pols = item.properties.get("sar:polarizations", [])
            if "VV" not in pols:
                continue

            # pick VV asset if present (names vary); try common keys
            assets = item.assets
            href = None
            for key in ("vv", "VV", "VV_db", "VH", "vh"):
                if key in assets:
                    href = assets[key].href
                    break
            # fallback: pick first asset URL
            if href is None:
                # find first asset that is a tiff / COG-like
                for a in assets.values():
                    if a.href.endswith(".tif") or a.href.endswith(".tiff"):
                        href = a.href
                        break
            if href is None:
                continue

            # sign the asset href for Planetary Computer (if needed)
            try:
                signed = planetary_computer.sign(href)
            except Exception:
                signed = href

            dt = item.properties.get("datetime", "")
            
            # Get geographic bounding box from STAC metadata
            scene_bbox = item.bbox  # [lon_min, lat_min, lon_max, lat_max]
            
            rows.append((
                item.id, 
                href,  # Use unsigned URL
                dt,
                scene_bbox[0],  # lon_min
                scene_bbox[1],  # lat_min
                scene_bbox[2],  # lon_max
                scene_bbox[3]   # lat_max
            ))
            
            # Collect geometry for visualization
            if item.geometry:
                all_scene_geometries.append(item.geometry)
            
            count += 1
            if MAX_ITEMS_PER_MONTH and count >= MAX_ITEMS_PER_MONTH:
                break
        
        total_scenes += len(rows)
        print(f"Found {len(rows)} scenes for this period. Appending to {OUT_CSV}")
        # Append results for the current chunk to the CSV
        with open(OUT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    print(f"\nDone. Wrote a total of {total_scenes} scene rows to {OUT_CSV}")
    
    # Generate visualization
    if all_scene_geometries:
        print(f"\nGenerating coverage map with {len(all_scene_geometries)} scenes...")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Plot ocean polygon as background
        ocean_gdf = gpd.GeoDataFrame([1], geometry=[ocean], crs="EPSG:4326")
        ocean_gdf.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.3, linewidth=0.5)
        
        # Plot each scene footprint
        for geom in all_scene_geometries:
            if geom['type'] == 'Polygon':
                coords = geom['coordinates'][0]
                polygon = mplPolygon(coords, fill=False, edgecolor='red', linewidth=0.5, alpha=0.7)
                ax.add_patch(polygon)
            elif geom['type'] == 'MultiPolygon':
                for poly_coords in geom['coordinates']:
                    coords = poly_coords[0]
                    polygon = mplPolygon(coords, fill=False, edgecolor='red', linewidth=0.5, alpha=0.7)
                    ax.add_patch(polygon)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'Sentinel-1 Scene Coverage Map\n{START_DATE} to {END_DATE} ({total_scenes} scenes)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        
        plt.tight_layout()
        plt.savefig(OUT_MAP, dpi=150, bbox_inches='tight')
        print(f"Coverage map saved as '{OUT_MAP}'")
    else:
        print("\nNo scenes found, skipping map generation.")

if __name__ == "__main__":
    main()
