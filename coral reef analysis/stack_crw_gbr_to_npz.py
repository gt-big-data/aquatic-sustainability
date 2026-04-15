#!/usr/bin/env python3
"""
Stack NOAA CRW GBR daily NetCDF files into a sequence NPZ matching
sequences_reduced_16.npz structure.

Output keys:
  - X: (N, lookback_weeks, 4) where features are
       [FilledSST, TSA, TSA_DHW, TSA_Frequency]
  - y: (N,) placeholder labels (-1, unlabeled inference data)
  - meta: (N, 4) -> [lat, lon, year, month] for sequence end week
  - end_time: (N,) exact sequence end timestamp (weekly)
  - feature_names
  - bleach_bins

Data mapping from CRW files:
  - FilledSST    <- analysed_sst
  - TSA          <- hotspot
  - TSA_DHW      <- degree_heating_week
  - TSA_Frequency <- rolling 52-week count of weekly TSA >= 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


TARGET_FEATURES = ["FilledSST", "TSA", "TSA_DHW", "TSA_Frequency"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CRW GBR daily NetCDF data to weekly sequence NPZ."
    )
    parser.add_argument(
        "--input-dir",
        default="crw_gbr_nc_yearly",
        help="Root directory containing sst/hotspot/dhw subfolders with .nc files.",
    )
    parser.add_argument(
        "--output-path",
        default="datasets/crw_gbr_sequences_reduced_16.npz",
        help="Path to output NPZ.",
    )
    parser.add_argument(
        "--reference-path",
        default="datasets/sequences_reduced_16.npz",
        help="Reference NPZ for lookback length and bleach_bins.",
    )
    parser.add_argument(
        "--lookback-weeks",
        type=int,
        default=-1,
        help="Lookback window length. If <1, inferred from reference X.shape[1].",
    )
    parser.add_argument(
        "--week-freq",
        default="W-SUN",
        help="Weekly resample frequency (default: W-SUN).",
    )
    parser.add_argument(
        "--agg",
        choices=["mean", "max"],
        default="mean",
        help="Aggregation for daily->weekly conversion (default: mean).",
    )
    parser.add_argument(
        "--tsa-threshold",
        type=float,
        default=1.0,
        help="Threshold for TSA_Frequency indicator (TSA >= threshold).",
    )
    parser.add_argument(
        "--rolling-weeks",
        type=int,
        default=52,
        help="Rolling window size for TSA_Frequency (default: 52).",
    )
    parser.add_argument(
        "--drop-all-nan-sites",
        action="store_true",
        help="Drop sites that are all-NaN across time/features.",
    )
    parser.add_argument(
        "--max-weeks",
        type=int,
        default=0,
        help="Debug: limit number of weekly timesteps after resampling (0 = all).",
    )
    parser.add_argument(
        "--max-sites",
        type=int,
        default=0,
        help="Debug: limit number of spatial sites after flattening (0 = all).",
    )
    return parser.parse_args()


def load_var_from_monthlies(files: list[Path], var_name: str) -> xr.DataArray:
    if not files:
        raise FileNotFoundError(f"No files found for variable '{var_name}'")

    arrays = []
    for fp in files:
        ds = xr.open_dataset(fp)
        if var_name not in ds:
            raise KeyError(f"Variable '{var_name}' not found in {fp}")
        da = ds[var_name]
        arrays.append(da)

    out = xr.concat(arrays, dim="time").sortby("time")
    _, unique_idx = np.unique(out["time"].values, return_index=True)
    out = out.isel(time=np.sort(unique_idx))
    return out


def resample_weekly(da: xr.DataArray, freq: str, agg: str) -> xr.DataArray:
    if agg == "mean":
        return da.resample(time=freq).mean(skipna=True)
    if agg == "max":
        return da.resample(time=freq).max(skipna=True)
    raise ValueError(f"Unsupported agg: {agg}")


def forward_fill_over_time(arr: np.ndarray) -> np.ndarray:
    """
    Forward-fill NaNs along axis=0 (time) for arr shape (T, S, F),
    then replace remaining NaNs with 0.
    """
    mask = np.isnan(arr)
    if not mask.any():
        return arr

    t_idx = np.arange(arr.shape[0], dtype=np.int32).reshape(-1, 1, 1)
    idx = np.where(~mask, t_idx, 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    ff = np.take_along_axis(arr, idx, axis=0)
    arr = np.where(mask, ff, arr)
    return np.nan_to_num(arr, nan=0.0)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ref_path = Path(args.reference_path).resolve()
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    ref = np.load(ref_path, allow_pickle=True)
    if "X" not in ref.files:
        raise KeyError(f"Reference file missing X: {ref_path}")
    lookback = args.lookback_weeks if args.lookback_weeks > 0 else int(ref["X"].shape[1])
    bleach_bins = ref["bleach_bins"] if "bleach_bins" in ref.files else np.array(
        [-1, 1, 50, 100], dtype=np.float32
    )

    sst_files = sorted((input_dir / "sst").glob("*.nc"))
    hotspot_files = sorted((input_dir / "hotspot").glob("*.nc"))
    dhw_files = sorted((input_dir / "dhw").glob("*.nc"))

    if not (sst_files and hotspot_files and dhw_files):
        raise FileNotFoundError(
            "Missing product files. Expected .nc files under input_dir/{sst,hotspot,dhw}."
        )

    print("=" * 70)
    print("Loading daily CRW files")
    print("=" * 70)
    print(f"Input dir: {input_dir}")
    print(f"SST files: {len(sst_files)}")
    print(f"Hotspot files: {len(hotspot_files)}")
    print(f"DHW files: {len(dhw_files)}")

    sst_daily = load_var_from_monthlies(sst_files, "analysed_sst")
    hotspot_daily = load_var_from_monthlies(hotspot_files, "hotspot")
    dhw_daily = load_var_from_monthlies(dhw_files, "degree_heating_week")

    sst_daily, hotspot_daily, dhw_daily = xr.align(
        sst_daily, hotspot_daily, dhw_daily, join="inner"
    )
    print(f"Daily aligned shape: {sst_daily.shape} (time, lat, lon)")
    print(
        f"Daily range: {str(np.datetime64(sst_daily.time.values[0]))} -> "
        f"{str(np.datetime64(sst_daily.time.values[-1]))}"
    )

    print("\nConverting daily data to weekly...")
    sst_w = resample_weekly(sst_daily, args.week_freq, args.agg)
    tsa_w = resample_weekly(hotspot_daily, args.week_freq, args.agg)
    dhw_w = resample_weekly(dhw_daily, args.week_freq, args.agg)

    sst_w, tsa_w, dhw_w = xr.align(sst_w, tsa_w, dhw_w, join="inner")
    tsa_indicator = (tsa_w >= args.tsa_threshold).astype(np.float32)
    tsa_freq_w = tsa_indicator.rolling(time=args.rolling_weeks, min_periods=1).sum()

    weekly = xr.Dataset(
        {
            "FilledSST": sst_w.astype(np.float32),
            "TSA": tsa_w.astype(np.float32),
            "TSA_DHW": dhw_w.astype(np.float32),
            "TSA_Frequency": tsa_freq_w.astype(np.float32),
        }
    )

    if args.max_weeks > 0:
        weekly = weekly.isel(time=slice(0, args.max_weeks))

    times = weekly["time"].values
    lats = weekly["latitude"].values
    lons = weekly["longitude"].values
    print(
        f"Weekly shape: time={weekly.sizes['time']}, "
        f"lat={weekly.sizes['latitude']}, lon={weekly.sizes['longitude']}"
    )
    print(
        f"Weekly range: {str(np.datetime64(times[0]))} -> {str(np.datetime64(times[-1]))}"
    )
    print(f"Lookback weeks: {lookback}")

    if weekly.sizes["time"] < lookback:
        raise ValueError(
            f"Not enough weekly timesteps ({weekly.sizes['time']}) for lookback={lookback}"
        )

    print("\nStacking features and building rolling sequences...")
    feat_arr = np.stack([weekly[f].values for f in TARGET_FEATURES], axis=-1).astype(np.float32)
    # (T, lat, lon, F) -> (T, S, F)
    T, nlat, nlon, F = feat_arr.shape
    feat_arr = feat_arr.reshape(T, nlat * nlon, F)

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    site_lat = lat_grid.reshape(-1).astype(np.float32)
    site_lon = lon_grid.reshape(-1).astype(np.float32)

    if args.drop_all_nan_sites:
        valid = ~np.all(np.isnan(feat_arr), axis=(0, 2))
        feat_arr = feat_arr[:, valid, :]
        site_lat = site_lat[valid]
        site_lon = site_lon[valid]
        print(f"Dropped all-NaN sites. Remaining sites: {feat_arr.shape[1]}")

    if args.max_sites > 0:
        feat_arr = feat_arr[:, : args.max_sites, :]
        site_lat = site_lat[: args.max_sites]
        site_lon = site_lon[: args.max_sites]

    feat_arr = forward_fill_over_time(feat_arr)

    S = feat_arr.shape[1]
    num_windows = T - lookback + 1
    N = num_windows * S

    print(f"Flattened sites: {S}")
    print(f"Weekly timesteps: {T}")
    print(f"Window count: {num_windows}")
    print(f"Total sequences N: {N}")

    X = np.empty((N, lookback, F), dtype=np.float32)
    y = np.full((N,), -1, dtype=np.int32)  # unlabeled inference data
    meta = np.empty((N, 4), dtype=np.float32)
    end_time = np.empty((N,), dtype="datetime64[ns]")

    cursor = 0
    for end_idx in range(lookback - 1, T):
        seq = feat_arr[end_idx - lookback + 1 : end_idx + 1, :, :]  # (lookback, S, F)
        block = np.transpose(seq, (1, 0, 2))  # (S, lookback, F)

        next_cursor = cursor + S
        X[cursor:next_cursor] = block

        ts = np.datetime64(times[end_idx], "D")
        year = float(str(ts)[:4])
        month = float(str(ts)[5:7])
        meta[cursor:next_cursor, 0] = site_lat
        meta[cursor:next_cursor, 1] = site_lon
        meta[cursor:next_cursor, 2] = year
        meta[cursor:next_cursor, 3] = month
        end_time[cursor:next_cursor] = np.datetime64(times[end_idx], "ns")
        cursor = next_cursor

        w = end_idx - (lookback - 1) + 1
        if w % 10 == 0 or w == num_windows:
            print(f"  Built windows: {w}/{num_windows}")

    print("\nSaving NPZ...")
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        meta=meta,
        end_time=end_time,
        feature_names=np.array(TARGET_FEATURES, dtype=object),
        bleach_bins=bleach_bins,
    )

    print("=" * 70)
    print("Done")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape} (all -1 for unlabeled data)")
    print(f"meta shape: {meta.shape}")
    print(f"end_time shape: {end_time.shape}")
    print(f"feature_names: {TARGET_FEATURES}")


if __name__ == "__main__":
    main()
