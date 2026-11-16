"""Per-sample driver features (tides, river flow, weather) built from timeseries_v2/timeseries.npz.

Groups (choose any):
  tide    : mean daily tidal range over the 7 days before the last input, and over last input -> target
            (tides are predictable, so 'ahead' is legitimate at forecast time)
  river   : log1p mean flow (Knik, Matanuska, Ship Creek) over the 7 days before the last input
  weather : rain sum, snowfall sum, mean air temp over the 7 days before the last input
  oracle  : flow/rain/snow/temp DURING the forecast window (last input -> target).
            NOT knowable in advance. Diagnostic upper bound only.
long_window=L (days) adds an L-day version of the river/weather features ending at the last input.
"""
import numpy as np, pandas as pd

RIVERS = ['knik', 'matanuska', 'shipcreek']


def _pos(d):
    days = pd.DatetimeIndex(d['days'])
    return pd.Series(np.arange(len(days)), index=days)


def _agg(daily, a, b, how):
    """aggregate daily[a:b]; NaN if the span is empty or has no data"""
    seg = daily[max(a, 0):b]
    if len(seg) == 0 or np.all(np.isnan(seg)):
        return np.nan
    return np.nansum(seg) if how == 'sum' else np.nanmean(seg)


def driver_features(d, groups, long_window=0):
    W = d['windows']
    dates = pd.DatetimeIndex(d['dates']).normalize()
    pos = _pos(d)
    p_last = pos.reindex(dates[W[:, -2]]).values.astype(int)    # day index of the last input
    p_tgt = pos.reindex(dates[W[:, -1]]).values.astype(int)     # day index of the target
    last = W[:, -2]
    out = {}

    def span(daily, a, b, how):
        return np.array([_agg(daily, x, y, how) for x, y in zip(a, b)])

    if 'tide' in groups:
        out['tide_before'] = d['tide_range_before'][last]
        out['tide_ahead'] = d['tide_range_ahead']
    if 'river' in groups:
        for r in RIVERS:
            out[f'{r}_7d'] = np.log1p(d[f'{r}_flow_before'][last])
            if long_window:
                out[f'{r}_{long_window}d'] = np.log1p(span(d[f'{r}_flow_daily'], p_last - long_window, p_last, 'mean'))
    if 'weather' in groups:
        out['rain_7d'] = d['rain_before'][last]
        out['snow_7d'] = d['snowfall_before'][last]
        out['temp_7d'] = d['temp_before'][last]
        if long_window:
            L = long_window
            out[f'rain_{L}d'] = span(d['rain_daily'], p_last - L, p_last, 'sum')
            out[f'snow_{L}d'] = span(d['snowfall_daily'], p_last - L, p_last, 'sum')
            out[f'temp_{L}d'] = span(d['temp_daily'], p_last - L, p_last, 'mean')
    if 'oracle' in groups:
        for r in RIVERS:
            out[f'{r}_horizon'] = np.log1p(span(d[f'{r}_flow_daily'], p_last, p_tgt, 'mean'))
        out['rain_horizon'] = span(d['rain_daily'], p_last, p_tgt, 'sum')
        out['snow_horizon'] = span(d['snowfall_daily'], p_last, p_tgt, 'sum')
        out['temp_horizon'] = span(d['temp_daily'], p_last, p_tgt, 'mean')
    return out