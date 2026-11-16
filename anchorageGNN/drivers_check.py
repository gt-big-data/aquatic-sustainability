"""Sanity-check the v2 drivers and figure out how much the channel shoals?

Usage: python drivers_check.py --data path/to/timeseries_v2/timeseries.npz
Predicts the SAMPLE-LEVEL average depth change
(and the share of cells shoaling > 1 ft) with a small ridge regression, scored leave-one-year-out.
"""
import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from anchorage_drivers import driver_features, RIVERS

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--alpha', type=float, default=10.0)
ap.add_argument('--long_window', type=int, default=28)
a = ap.parse_args()
d = np.load(a.data, allow_pickle=True)

# ------------------------------------------------------------------ 1. inventory + alignment
print('== 1. Data check ==')
for k in ['tide_range_before', 'tide_range_ahead'] + [f'{r}_flow_before' for r in RIVERS] + \
         ['rain_before', 'snowfall_before', 'temp_before']:
    v = d[k]
    print(f'{k:22s} n={len(v):4d}  missing={np.isnan(v).sum():3d}  min={np.nanmin(v):9.2f}  max={np.nanmax(v):9.2f}')

dates = pd.DatetimeIndex(d['dates']).normalize()
cal = pd.DatetimeIndex(d['days'])
print(f'daily calendar {cal.min().date()} -> {cal.max().date()}  ({len(cal)} days);  '
      f'tide calendar {pd.DatetimeIndex(d["tide_days"]).min().date()} -> {pd.DatetimeIndex(d["tide_days"]).max().date()}')
print(f'tide days that are predictions, not measurements: {int(d["tide_is_pred"].sum())}')


def recompute(series, how):
    out = np.full(len(dates), np.nan)
    for p, day in enumerate(dates):
        w = series[(series.index >= day - pd.Timedelta(days=7)) & (series.index < day)]
        if len(w) and w.notna().any():
            out[p] = w.sum() if how == 'sum' else w.mean()
    return out

checks = {f'{r}_flow_before': (pd.Series(d[f'{r}_flow_daily'], index=cal), 'mean') for r in RIVERS}
checks.update({'rain_before': (pd.Series(d['rain_daily'], index=cal), 'sum'),
               'snowfall_before': (pd.Series(d['snowfall_daily'], index=cal), 'sum'),
               'temp_before': (pd.Series(d['temp_daily'], index=cal), 'mean'),
               'tide_range_before': (pd.Series(d['tide_range_daily'], index=pd.DatetimeIndex(d['tide_days'])), 'mean')})
print('\nAlignment: recompute each *_before from its daily series and compare to the stored array')
for k, (s, how) in checks.items():
    r, st = recompute(s, how), d[k]
    both = ~np.isnan(r) & ~np.isnan(st)
    print(f'  {k:22s} max abs diff {np.abs(r[both] - st[both]).max():.4g}   '
          f'NaN pattern differs on {int((np.isnan(r) != np.isnan(st)).sum())} maps')

# ------------------------------------------------------------------ 2. do drivers explain shoaling?
print('\n== 2. Do the drivers explain shoaling? (leave-one-year-out ridge, sample level) ==')
W, maps, masks = d['windows'], d['maps'], d['masks']
delta = maps[W[:, -1]] - maps[W[:, -2]]
n = masks.reshape(len(W), -1).sum(1)
y_mean = np.where(masks, delta, 0).reshape(len(W), -1).sum(1) / n            # avg change over graded cells
y_hot = ((delta < -1) & masks).reshape(len(W), -1).sum(1) / n                 # share of cells shoaling > 1 ft
year = pd.DatetimeIndex(d['dates'])[W[:, -1]].year.values
doy = pd.DatetimeIndex(d['dates'])[W[:, -2]].dayofyear.values / 365.0
season = {'season_sin': np.sin(2 * np.pi * doy), 'season_cos': np.cos(2 * np.pi * doy)}
print(f'{len(W)} samples, {len(set(year))} target years;  avg change per sample: mean {y_mean.mean():+.3f} ft, sd {y_mean.std():.3f}')

L = a.long_window
sets = {
    'season only (control)': season,
    '+ tides': {**season, **driver_features(d, ['tide'])},
    '+ rivers (7d)': {**season, **driver_features(d, ['river'])},
    '+ weather (7d)': {**season, **driver_features(d, ['weather'])},
    '+ all, 7d (deployable)': {**season, **driver_features(d, ['tide', 'river', 'weather'])},
    f'+ all, 7d+{L}d (deployable)': {**season, **driver_features(d, ['tide', 'river', 'weather'], L)},
    '+ all + during-horizon (ORACLE)': {**season, **driver_features(d, ['tide', 'river', 'weather', 'oracle'])},
}


def loyo(feats, y):
    X = np.column_stack(list(feats.values()))
    pred, base = np.zeros(len(y)), np.zeros(len(y))
    for yr in np.unique(year):
        te, tr = year == yr, year != yr
        med = np.nanmedian(X[tr], 0); med = np.where(np.isnan(med), 0, med)
        Xf = np.where(np.isnan(X), med, X)
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0); sd[sd == 0] = 1
        Z = (Xf - mu) / sd
        pred[te] = Ridge(alpha=a.alpha).fit(Z[tr], y[tr]).predict(Z[te])
        base[te] = y[tr].mean()
    return 1 - ((y - pred) ** 2).sum() / ((y - base) ** 2).sum()

print(f'\n{"features":<36}{"R2 avg change":>16}{"R2 hot share":>16}   (0 = no better than predicting the training mean)')
for name, f in sets.items():
    print(f'{name:<36}{loyo(f, y_mean):16.3f}{loyo(f, y_hot):16.3f}')

print('\nSpearman correlation of each driver with the sample-level average change:')
allf = driver_features(d, ['tide', 'river', 'weather', 'oracle'], L)
for k, v in allf.items():
    ok = ~np.isnan(v)
    print(f'  {k:22s} {pd.Series(v[ok]).corr(pd.Series(y_mean[ok]), method="spearman"):+.3f}  (n={ok.sum()})')