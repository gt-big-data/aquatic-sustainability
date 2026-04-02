"""
Build LSTM sequences for coral bleaching prediction.

For each bleaching event in the GCBD, extracts a 52-week lookback 
sequence of thermal stress features from local CoRTAD v6 NetCDF files.

Inputs:
  - cortadv6_FilledSST.nc  (50 GB)  -> FilledSST
  - cortadv6_SSTA.nc       (111 GB) -> SSTA, SSTA_DHW, SSTA_Frequency
  - cortadv6_TSA.nc        (83 GB)  -> TSA, TSA_DHW, TSA_Frequency
  - global_bleaching_environmental.csv (GCBD)

Outputs:
  - sequences.npz: X (N, 52, 7), y (N,), metadata (N, 4)
"""

import os
import xarray as xr
import pandas as pd
import numpy as np
import time

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
LOOKBACK_WEEKS = 16  # 16-week lookback
GCBD_PATH = "datasets/global_bleaching_environmental.csv"
SST_PATH = "datasets/cortadv6_FilledSST.nc"
SSTA_PATH = "datasets/cortadv6_SSTA.nc"
TSA_PATH = "datasets/cortadv6_TSA.nc"
OUTPUT_PATH = "datasets/sequences.npz"

# # Bleaching severity bins: 0=none, 1=low, 2=moderate, 3=severe
# BLEACH_BINS = [-1, 1, 10, 50, 100]
# BLEACH_LABELS = [0, 1, 2, 3]

# None (0%), Moderate (1-50%), Severe (>50%)
BLEACH_BINS = [-1, 1, 50, 100]
BLEACH_LABELS = [0, 1, 2]

# Features to extract (4 total)
FEATURE_NAMES = [
    "FilledSST",      # Raw SST
    # "SSTA",           # SST Anomaly (vs weekly climatology)
    # "SSTA_DHW",       # SSTA-based Degree Heating Weeks
    # "SSTA_Frequency", # SSTA frequency (times SSTA>=1 in past 52 weeks)
    "TSA",            # Thermal Stress Anomaly (vs max monthly mean)
    "TSA_DHW",        # TSA-based Degree Heating Weeks (standard DHW)
    "TSA_Frequency",  # TSA frequency (times TSA>=1 in past 52 weeks)
]

# ──────────────────────────────────────────────────────────────
# STEP 1: Load GCBD and prepare events
# ──────────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Loading GCBD")
print("=" * 70)

gcbd = pd.read_csv(GCBD_PATH)
print(f"  Total records: {len(gcbd)}")

# Filter to records that have enough info for a label and a date
gcbd = gcbd.dropna(subset=["Percent_Bleaching", "Date_Year", "Latitude_Degrees", "Longitude_Degrees"])
gcbd = gcbd[gcbd["Date_Year"] >= 1983]  # Need 1 year of lookback from 1982 start
print(f"  After filtering: {len(gcbd)}")

# Coerce non-numeric Percent_Bleaching entries to NaN
gcbd["Percent_Bleaching"] = pd.to_numeric(gcbd["Percent_Bleaching"], errors="coerce")

# Drop rows where Percent_Bleaching couldn't be parsed to a number
gcbd = gcbd.dropna(subset=["Percent_Bleaching"])

# Create ordinal bleaching class
# Clip to bin range to avoid NaN from pd.cut on out-of-range values
gcbd["bleach_class"] = pd.cut(
    gcbd["Percent_Bleaching"].clip(0, 100),
    bins=BLEACH_BINS,
    labels=BLEACH_LABELS,
    include_lowest=True,
).astype(int)

# Create approximate date (use middle of month, default to June if month missing)
gcbd["event_month"] = gcbd["Date_Month"].fillna(6).astype(int).clip(1, 12)
gcbd["event_date"] = pd.to_datetime(
    gcbd["Date_Year"].astype(int).astype(str) + "-" +
    gcbd["event_month"].astype(str).str.zfill(2) + "-15"
)

print(f"  After dropping non-numeric bleaching values: {len(gcbd)}")
print(f"  Date range: {gcbd['event_date'].min()} to {gcbd['event_date'].max()}")
print(f"  Class distribution:")
for c in BLEACH_LABELS:
    n = (gcbd["bleach_class"] == c).sum()
    print(f"    Class {c}: {n} ({n/len(gcbd)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# STEP 2: Open CoRTAD files and extract unique site time series
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Opening local CoRTAD files")
print("=" * 70)

t0 = time.time()

# Use chunks to avoid loading everything into memory
sst_ds = xr.open_dataset(SST_PATH, chunks={"time": 104})
# ssta_ds = xr.open_dataset(SSTA_PATH, chunks={"time": 104})
tsa_ds = xr.open_dataset(TSA_PATH, chunks={"time": 104})

print(f"  Opened in {time.time()-t0:.1f}s")

# xarray already decodes CoRTAD time into datetime64 values
dates = pd.DatetimeIndex(sst_ds["time"].values)
print(f"  Time range: {dates[0]} to {dates[-1]} ({len(dates)} weeks)")

# ──────────────────────────────────────────────────────────────
# STEP 3: Extract unique site locations
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Extracting unique site time series")
print("=" * 70)

unique_sites = gcbd[["Latitude_Degrees", "Longitude_Degrees"]].drop_duplicates().reset_index(drop=True)
print(f"  Unique sites: {len(unique_sites)}")

lats = xr.DataArray(unique_sites["Latitude_Degrees"].values, dims="site")
lons = xr.DataArray(unique_sites["Longitude_Degrees"].values, dims="site")

# Extract all sites at once for each variable
# This is the slow part — reads from disk but only the needed grid cells
print("  Extracting FilledSST...", end=" ", flush=True)
t0 = time.time()
sst_all = sst_ds["FilledSST"].sel(lat=lats, lon=lons, method="nearest").load()
print(f"done ({time.time()-t0:.1f}s)")

# print("  Extracting SSTA...", end=" ", flush=True)
# t0 = time.time()
# ssta_all = ssta_ds["SSTA"].sel(lat=lats, lon=lons, method="nearest").load()
# print(f"done ({time.time()-t0:.1f}s)")

# print("  Extracting SSTA_DHW...", end=" ", flush=True)
# t0 = time.time()
# ssta_dhw_all = ssta_ds["SSTA_DHW"].sel(lat=lats, lon=lons, method="nearest").load()
# print(f"done ({time.time()-t0:.1f}s)")

# print("  Extracting SSTA_Frequency...", end=" ", flush=True)
# t0 = time.time()
# ssta_freq_all = ssta_ds["SSTA_Frequency"].sel(lat=lats, lon=lons, method="nearest").load()
# print(f"done ({time.time()-t0:.1f}s)")

print("  Extracting TSA...", end=" ", flush=True)
t0 = time.time()
tsa_all = tsa_ds["TSA"].sel(lat=lats, lon=lons, method="nearest").load()
print(f"done ({time.time()-t0:.1f}s)")

print("  Extracting TSA_DHW...", end=" ", flush=True)
t0 = time.time()
tsa_dhw_all = tsa_ds["TSA_DHW"].sel(lat=lats, lon=lons, method="nearest").load()
print(f"done ({time.time()-t0:.1f}s)")

print("  Extracting TSA_Frequency...", end=" ", flush=True)
t0 = time.time()
tsa_freq_all = tsa_ds["TSA_Frequency"].sel(lat=lats, lon=lons, method="nearest").load()
print(f"done ({time.time()-t0:.1f}s)")

# Clip TSA to >= 0 to match CRW HotSpot behavior
tsa_all = tsa_all.clip(min=0)

# Stack into a single array: (time, site, features)
all_features = np.stack([
    sst_all.values,
    # ssta_all.values,
    # ssta_dhw_all.values,
    # ssta_freq_all.values,
    tsa_all.values,
    tsa_dhw_all.values,
    tsa_freq_all.values,
], axis=-1)  # shape: (num_weeks, num_sites, 4)

print(f"\n  Combined feature array shape: {all_features.shape}")
print(f"  (time_steps, sites, features)")

# Build a lookup: (lat, lon) -> site index
site_lookup = {}
for idx, row in unique_sites.iterrows():
    key = (round(row["Latitude_Degrees"], 4), round(row["Longitude_Degrees"], 4))
    site_lookup[key] = idx

# ──────────────────────────────────────────────────────────────
# STEP 4: Build sequences for each bleaching event
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Building sequences")
print("=" * 70)

X_list = []
y_list = []
meta_list = []  # lat, lon, year, month
skipped = 0

for i, row in gcbd.iterrows():
    lat = row["Latitude_Degrees"]
    lon = row["Longitude_Degrees"]
    event_date = row["event_date"]
    label = row["bleach_class"]
    
    # Find site index
    key = (round(lat, 4), round(lon, 4))
    site_idx = site_lookup.get(key)
    if site_idx is None:
        skipped += 1
        continue
    
    # Find the time index closest to the event date
    end_idx = int(np.argmin(np.abs(dates - event_date)))
    start_idx = end_idx - LOOKBACK_WEEKS
    
    # Check bounds
    if start_idx < 0:
        skipped += 1
        continue
    
    # Extract sequence: (52, 4)
    seq = all_features[start_idx:end_idx, site_idx, :]
    
    if seq.shape[0] != LOOKBACK_WEEKS:
        skipped += 1
        continue
    
    # Check for all-NaN sequences (land pixels etc.)
    if np.all(np.isnan(seq)):
        skipped += 1
        continue
    
    X_list.append(seq)
    y_list.append(label)
    meta_list.append([lat, lon, row["Date_Year"], row["event_month"]])
    
    if len(X_list) % 5000 == 0:
        print(f"  Built {len(X_list)} sequences ({skipped} skipped)...")

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
meta = np.array(meta_list, dtype=np.float32)

print(f"\n  Final dataset:")
print(f"    X shape: {X.shape}  (samples, timesteps, features)")
print(f"    y shape: {y.shape}")
print(f"    meta shape: {meta.shape}  (lat, lon, year, month)")
print(f"    Skipped: {skipped}")
print(f"    NaN fraction in X: {np.isnan(X).mean():.4f}")

print(f"\n  Class distribution in final dataset:")
for c in BLEACH_LABELS:
    n = (y == c).sum()
    print(f"    Class {c}: {n} ({n/len(y)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────
# STEP 5: Handle NaNs and save
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Saving")
print("=" * 70)

# Forward-fill NaNs within each sequence, then fill remaining with 0
# This handles occasional missing weeks in the satellite data
for i in range(X.shape[0]):
    for f in range(X.shape[2]):
        col = X[i, :, f]
        # Forward fill
        mask = np.isnan(col)
        if mask.any() and not mask.all():
            idx_arr = np.where(~mask, np.arange(len(col)), 0)
            np.maximum.accumulate(idx_arr, out=idx_arr)
            col[mask] = col[idx_arr[mask]]
        X[i, :, f] = col

# Any remaining NaNs become 0
remaining_nans = np.isnan(X).mean()
X = np.nan_to_num(X, nan=0.0)

print(f"  NaNs after forward-fill (replaced with 0): {remaining_nans:.4f}")

np.savez_compressed(
    OUTPUT_PATH,
    X=X,
    y=y,
    meta=meta,
    feature_names=FEATURE_NAMES,
    bleach_bins=BLEACH_BINS,
)

file_size = os.path.getsize(OUTPUT_PATH) / (1024**2) if os.path.exists(OUTPUT_PATH) else 0
print(f"  Saved to {OUTPUT_PATH} ({file_size:.1f} MB)")
print(f"\n  Features per timestep: {FEATURE_NAMES}")
print(f"  To load:")
print(f"    data = np.load('{OUTPUT_PATH}')")
print(f"    X, y = data['X'], data['y']")

print("\nDone!")