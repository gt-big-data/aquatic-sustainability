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

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

def build_ocean_polygon():
    """
    Build a simplified ocean polygon by subtracting naturalearth land from global bbox.
    Requires geopandas.
    """
    # world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    # world = gpd.read_file("naturalearth_lowres.shp")
    world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
    land_union = unary_union(world.geometry)
    # global bbox (lon/lat)
    global_box = box(-180.0, -90.0, 180.0, 90.0)
    ocean = global_box.difference(land_union)
    # optionally simplify
    ocean = gpd.GeoSeries([ocean]).simplify(0.05).iloc[0]
    return ocean

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

time_range = "2020-12-01/2020-12-31"
area_of_interest = {
    "type": "Polygon",
    "coordinates": [
        [
            [-122.2751, 47.5469],
            [-121.9613, 47.9613],
            [-121.9613, 47.9613],
            [-122.2751, 47.9613],
            [-122.2751, 47.5469],
        ]
    ],
}

ocean = build_ocean_polygon()
stac = Client.open(STAC_URL)
bbox = ocean.bounds

print(ocean)
print(bbox)

# # Visualize the ocean polygon
# fig, ax = plt.subplots(figsize=(15, 8))

# # Convert ocean polygon to GeoDataFrame for easy plotting
# ocean_gdf = gpd.GeoDataFrame([1], geometry=[ocean], crs="EPSG:4326")

# # Plot the ocean polygon
# ocean_gdf.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.5)

# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# ax.set_title('Ocean Polygon (Land Removed from Global Bounding Box)')
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('ocean_polygon.png', dpi=150, bbox_inches='tight')
# plt.show()

# print("Ocean polygon visualization saved as 'ocean_polygon.png'")

time_range = "2020-12-01/2020-12-31"

# search = catalog.search(
#     collections=["sentinel-1-grd"], intersects=ocean, datetime=time_range, max_items=5000
# )

search = stac.search(
    collections=["sentinel-1-grd"],
    intersects=ocean,
    datetime=time_range,
    # query={"sar:polarizations": {"contains": "VV"}},
    query={"s1:polarizations": {"in": ["VV", "VH", "VV,VH"]}},
    max_items=500
)

items = list(search.items())
print(f"Found {len(items)} scenes")

# Extract scene geometries
scene_geometries = []
for item in items:
    geom = item.geometry
    if geom:
        scene_geometries.append(geom)

# Create visualization
fig, ax = plt.subplots(figsize=(20, 10))

# Plot ocean polygon as background
ocean_gdf = gpd.GeoDataFrame([1], geometry=[ocean], crs="EPSG:4326")
ocean_gdf.plot(ax=ax, color='lightblue', edgecolor='blue', alpha=0.3, linewidth=0.5)

# Plot each scene footprint
for i, geom in enumerate(scene_geometries):
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
ax.set_title(f'Sentinel-1 Scene Footprints ({time_range})\n{len(items)} scenes found', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)

plt.tight_layout()
plt.savefig('scene_footprints.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Visualization saved as 'scene_footprints.png'")
print(f"Total scenes plotted: {len(scene_geometries)}")