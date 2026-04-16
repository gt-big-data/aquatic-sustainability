#!/usr/bin/env python3
"""
Stack NOAA CRW GBR daily NetCDF files into a sequence NPZ matching
sequences_reduced_16.npz structure.

Output keys:
  - X: (N, lookback_weeks, 4) where features are
       [FilledSST, SSTA, TSA, TSA_DHW]
  - y: (N,) placeholder labels (-1, unlabeled inference data)
  - meta: (N, 4) -> [lat, lon, year, month] for sequence end week
  - end_time: (N,) exact sequence end timestamp (weekly)
  - feature_names
  - bleach_bins

Data mapping from CRW files:
  - FilledSST <- crw_sst / analysed_sst (converted from degree_C to Kelvin)
  - SSTA      <- crw_sstanomaly / sea_surface_temperature_anomaly
  - TSA       <- crw_hotspot / hotspot (clipped to >= 0)
  - TSA_DHW   <- crw_dhw / degree_heating_week
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import xarray as xr


CANONICAL_FEATURES = ["filled_sst", "ssta", "tsa", "tsa_dhw"]
DEFAULT_OUTPUT_FEATURE_NAMES = ["FilledSST", "SSTA", "TSA", "TSA_DHW"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CRW GBR daily NetCDF data to weekly sequence NPZ."
    )
    parser.add_argument(
        "--input-dir",
        default="crw_gbr_nc_yearly",
        help="Root directory containing sst/sst_anomaly/hotspot/dhw subfolders with .nc files.",
    )
    parser.add_argument(
        "--output-path",
        default="datasets/crw_gbr_sequences_reduced_16.npz",
        help="Path to output NPZ.",
    )
    parser.add_argument(
        "--reference-path",
        default="datasets/sequences_reduced_16.npz",
        help="Reference NPZ for lookback length, bleach_bins, and optional feature_names.",
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
        default=0.0,
        help="Lower clip bound for TSA (hotspot), default 0.0.",
    )
    parser.add_argument(
        "--rolling-weeks",
        type=int,
        default=52,
        help="Deprecated (unused). Retained for CLI compatibility.",
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


def load_var_from_monthlies(files: list[Path], var_names: list[str]) -> tuple[xr.DataArray, str]:
    if not files:
        raise FileNotFoundError(f"No files found for variable candidates {var_names}")

    arrays = []
    resolved_name: str | None = None
    for fp in files:
        ds = xr.open_dataset(fp)
        found_name = next((name for name in var_names if name in ds), None)
        if found_name is None:
            available = list(ds.data_vars)
            raise KeyError(
                f"None of variables {var_names} found in {fp}. Available data vars: {available}"
            )
        if resolved_name is None:
            resolved_name = found_name
        da = ds[found_name]
        arrays.append(da)

    out = xr.concat(arrays, dim="time").sortby("time")
    _, unique_idx = np.unique(out["time"].values, return_index=True)
    out = out.isel(time=np.sort(unique_idx))
    if resolved_name is None:
        raise RuntimeError(f"Could not resolve variable from candidates: {var_names}")
    return out, resolved_name


def resample_weekly(da: xr.DataArray, freq: str, agg: str) -> xr.DataArray:
    if agg == "mean":
        return da.resample(time=freq).mean(skipna=True)
    if agg == "max":
        return da.resample(time=freq).max(skipna=True)
    raise ValueError(f"Unsupported agg: {agg}")


def find_product_files(input_dir: Path, subdirs: list[str]) -> tuple[list[Path], str | None]:
    for sub in subdirs:
        files = sorted((input_dir / sub).glob("*.nc"))
        if files:
            return files, sub
    return [], None


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_output_feature_names(ref: np.lib.npyio.NpzFile) -> list[str]:
    if "feature_names" not in ref.files:
        return DEFAULT_OUTPUT_FEATURE_NAMES

    ref_names = [str(x) for x in np.asarray(ref["feature_names"]).reshape(-1).tolist()]
    if len(ref_names) != 4:
        return DEFAULT_OUTPUT_FEATURE_NAMES

    expected = [normalize_name(x) for x in CANONICAL_FEATURES]
    got = [normalize_name(x) for x in ref_names]
    if got == expected:
        return ref_names
    return DEFAULT_OUTPUT_FEATURE_NAMES


def maybe_celsius_to_kelvin(da: xr.DataArray) -> tuple[xr.DataArray, str]:
    units = str(da.attrs.get("units", "")).strip().lower()
    looks_kelvin = ("kelvin" in units) or units == "k"
    looks_celsius = any(token in units for token in ["degree_c", "degrees_c", "celsius", "degc"])

    sample_mean = float(da.isel(time=slice(0, min(8, da.sizes["time"]))).mean(skipna=True).values)
    if looks_kelvin or sample_mean > 200.0:
        return da, "already_kelvin"

    if looks_celsius or sample_mean < 120.0:
        out = (da + np.float32(273.15)).astype(np.float32)
        out.attrs = dict(da.attrs)
        out.attrs["units"] = "kelvin"
        return out, "converted_celsius_to_kelvin"

    return da, "left_unchanged"


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
    output_feature_names = resolve_output_feature_names(ref)
    bleach_bins = ref["bleach_bins"] if "bleach_bins" in ref.files else np.array(
        [-1, 1, 50, 100], dtype=np.float32
    )

    sst_files, sst_dir = find_product_files(input_dir, ["sst", "crw_sst"])
    ssta_files, ssta_dir = find_product_files(
        input_dir, ["sst_anomaly", "sstanomaly", "ssta", "crw_sstanomaly"]
    )
    hotspot_files, hotspot_dir = find_product_files(input_dir, ["hotspot", "crw_hotspot"])
    dhw_files, dhw_dir = find_product_files(input_dir, ["dhw", "crw_dhw"])

    if not (sst_files and ssta_files and hotspot_files and dhw_files):
        raise FileNotFoundError(
            "Missing product files. Expected .nc files under "
            "input_dir/{sst, sst_anomaly, hotspot, dhw} (or alias dirs)."
        )

    print("=" * 70)
    print("Loading daily CRW files")
    print("=" * 70)
    print(f"Input dir: {input_dir}")
    print(f"SST files: {len(sst_files)} (dir='{sst_dir}')")
    print(f"SSTA files: {len(ssta_files)} (dir='{ssta_dir}')")
    print(f"Hotspot files: {len(hotspot_files)} (dir='{hotspot_dir}')")
    print(f"DHW files: {len(dhw_files)} (dir='{dhw_dir}')")

    sst_daily, sst_var = load_var_from_monthlies(
        sst_files, ["crw_sst", "analysed_sst", "sea_surface_temperature"]
    )
    ssta_daily, ssta_var = load_var_from_monthlies(
        ssta_files, ["crw_sstanomaly", "sea_surface_temperature_anomaly", "sst_anomaly", "ssta"]
    )
    hotspot_daily, hotspot_var = load_var_from_monthlies(hotspot_files, ["crw_hotspot", "hotspot"])
    dhw_daily, dhw_var = load_var_from_monthlies(
        dhw_files, ["crw_dhw", "degree_heating_week", "dhw"]
    )
    print(f"Resolved vars: sst='{sst_var}', ssta='{ssta_var}', hotspot='{hotspot_var}', dhw='{dhw_var}'")

    sst_daily, ssta_daily, hotspot_daily, dhw_daily = xr.align(
        sst_daily, ssta_daily, hotspot_daily, dhw_daily, join="inner"
    )
    print(f"Daily aligned shape: {sst_daily.shape} (time, lat, lon)")
    print(
        f"Daily range: {str(np.datetime64(sst_daily.time.values[0]))} -> "
        f"{str(np.datetime64(sst_daily.time.values[-1]))}"
    )

    sst_daily, sst_conversion = maybe_celsius_to_kelvin(sst_daily)
    hotspot_daily = hotspot_daily.clip(min=args.tsa_threshold)
    print(f"SST unit normalization: {sst_conversion}")
    print(f"TSA clipping: hotspot >= {args.tsa_threshold}")

    print("\nConverting daily data to weekly...")
    sst_w = resample_weekly(sst_daily, args.week_freq, args.agg)
    ssta_w = resample_weekly(ssta_daily, args.week_freq, args.agg)
    tsa_w = resample_weekly(hotspot_daily, args.week_freq, args.agg)
    dhw_w = resample_weekly(dhw_daily, args.week_freq, args.agg)

    sst_w, ssta_w, tsa_w, dhw_w = xr.align(sst_w, ssta_w, tsa_w, dhw_w, join="inner")

    weekly = xr.Dataset(
        {
            "filled_sst": sst_w.astype(np.float32),
            "ssta": ssta_w.astype(np.float32),
            "tsa": tsa_w.astype(np.float32),
            "tsa_dhw": dhw_w.astype(np.float32),
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
    feat_arr = np.stack([weekly[f].values for f in CANONICAL_FEATURES], axis=-1).astype(np.float32)
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
        feature_names=np.array(output_feature_names, dtype=object),
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
    print(f"feature_names: {output_feature_names}")


if __name__ == "__main__":
    main()
