"""
Different flattening strategies for ablation experiments.

All functions take (X_seq, meta, feature_names) and return a pd.DataFrame.
X_seq: (N, 16, 4), meta: (N, 4), feature_names: list of 4 strings.
"""

import numpy as np
import pandas as pd


def flatten_summary(X_seq, meta, feature_names):
    """4 features x 8 stats + 4 metadata = 36 features."""
    flat = {}
    for i, fname in enumerate(feature_names):
        series = X_seq[:, :, i]
        flat[f"{fname}_mean"] = np.nanmean(series, axis=1)
        flat[f"{fname}_max"] = np.nanmax(series, axis=1)
        flat[f"{fname}_min"] = np.nanmin(series, axis=1)
        flat[f"{fname}_std"] = np.nanstd(series, axis=1)
        flat[f"{fname}_last"] = series[:, -1]
        flat[f"{fname}_first"] = series[:, 0]
        flat[f"{fname}_trend"] = series[:, -1] - series[:, 0]
        x_time = np.arange(series.shape[1])
        x_mean = x_time.mean()
        x_var = ((x_time - x_mean) ** 2).sum()
        slopes = (
            (series - series.mean(axis=1, keepdims=True)) * (x_time - x_mean)
        ).sum(axis=1) / x_var
        flat[f"{fname}_slope"] = slopes

    flat["latitude"] = meta[:, 0]
    flat["longitude"] = meta[:, 1]
    flat["year"] = meta[:, 2]
    flat["month"] = meta[:, 3]
    return pd.DataFrame(flat).fillna(0)


def flatten_raw(X_seq, meta, feature_names):
    """16 weeks x 4 features unrolled + 4 metadata = 68 features."""
    flat = {}
    for w in range(X_seq.shape[1]):
        for i, fname in enumerate(feature_names):
            flat[f"{fname}_week{w:02d}"] = X_seq[:, w, i]
    flat["latitude"] = meta[:, 0]
    flat["longitude"] = meta[:, 1]
    flat["year"] = meta[:, 2]
    flat["month"] = meta[:, 3]
    return pd.DataFrame(flat).fillna(0)


def flatten_hybrid(X_seq, meta, feature_names):
    """Summary stats + raw weekly values + metadata = 100 features."""
    summary_df = flatten_summary(X_seq, meta, feature_names)
    raw_df = flatten_raw(X_seq, meta, feature_names)
    raw_df = raw_df.drop(columns=["latitude", "longitude", "year", "month"])
    return pd.concat([summary_df, raw_df], axis=1)


def flatten_summary_no_year(X_seq, meta, feature_names):
    """Same as summary but without year. Tests temporal trend dependence."""
    df = flatten_summary(X_seq, meta, feature_names)
    return df.drop(columns=["year"])


def flatten_summary_interactions(X_seq, meta, feature_names):
    """Summary stats + interaction terms = 42 features."""
    df = flatten_summary(X_seq, meta, feature_names)
    df["DHW_x_SST"] = df["TSA_DHW_max"] * df["FilledSST_max"]
    df["stress_acceleration"] = df["TSA_Frequency_slope"] * df["TSA_DHW_slope"]
    df["sustained_stress"] = df["TSA_mean"] / (df["TSA_std"] + 0.01)
    df["recent_stress_ratio"] = (df["TSA_last"] + 0.01) / (df["TSA_first"] + 0.01)
    df["dhw_freq_interaction"] = df["TSA_DHW_last"] * df["TSA_Frequency_slope"]
    df["sst_above_threshold"] = np.clip(df["FilledSST_max"] - 301.0, 0, None)
    return df


def flatten_hybrid_no_year(X_seq, meta, feature_names):
    """Hybrid minus year column = 99 features."""
    df = flatten_hybrid(X_seq, meta, feature_names)
    return df.drop(columns=["year"])
