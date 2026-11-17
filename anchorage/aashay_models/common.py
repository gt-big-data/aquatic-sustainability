import json
import os
import time

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'timeseries_v2', 'timeseries.npz')
RESULTS = os.path.join(HERE, 'results')
DRIVERS = ['tide_range', 'knik_flow', 'matanuska_flow', 'shipcreek_flow', 'rain', 'temp']
LOG = {'knik_flow', 'matanuska_flow', 'shipcreek_flow', 'rain'}
TRAIN_LAST_YEAR = 2022


def season_bucket(dates):
    m = dates.astype('datetime64[M]').astype(int) % 12 + 1
    return np.digitize(m, [7, 9])


class Data:
    def __init__(self, loyo=True):
        self.loyo = loyo
        d = np.load(DATA)
        self.d = d
        self.maps, self.windows, self.masks, self.split = d['maps'], d['windows'], d['masks'], d['split']
        self.dates, self.dredged, self.core = d['dates'], d['dredged'], d['core']
        self.S, self.H, self.W = self.masks.shape
        w = self.windows
        self.inputs = self.maps[w[:, :-1]]
        self.last = self.inputs[:, -1]
        self.target = self.maps[w[:, -1]]
        self.change = np.where(self.masks, self.target - self.last, 0).astype(np.float32)
        days = self.dates.astype('datetime64[D]').astype(np.int64)
        self.days = days
        self.in_days = days[w[:, :-1]]
        self.tgt_days = days[w[:, -1]]
        self.horizon = (self.tgt_days - self.in_days[:, -1]).astype(np.float32)
        self.tgt_year = self.dates[w[:, -1]].astype('datetime64[Y]').astype(int) + 1970
        doy = (self.dates[w[:, -1]] - self.dates[w[:, -1]].astype('datetime64[Y]')).astype(int)
        self.doy_sin, self.doy_cos = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        self.idx = {k: np.where(self.split == k)[0] for k in ['train', 'val', 'test']}
        self._rates()
        self._drivers()
        self._hot_cells()

    def _rates(self):
        maps, days, dredged = self.maps, self.days, self.dredged
        year = self.dates.astype('datetime64[Y]').astype(int) + 1970
        bucket = season_bucket(self.dates)
        years = np.arange(2014, TRAIN_LAST_YEAR + 1)
        tc = np.zeros((len(years), 3, self.H, self.W))
        td = np.zeros((len(years), 3, self.H, self.W))
        for p in range(len(maps) - 1):
            gap = days[p + 1] - days[p]
            if gap < 60 and year[p + 1] <= TRAIN_LAST_YEAR:
                step = maps[p + 1] - maps[p]
                ok = ~np.isnan(step) & ~dredged[p + 1]
                yi, b = year[p + 1] - 2014, bucket[p + 1]
                tc[yi, b][ok] += step[ok]
                td[yi, b][ok] += gap
        self._harmonic(maps, days, dredged, year, loyo=self.loyo)
        tgt_bucket = season_bucket(self.dates[self.windows[:, -1]])
        self.rate_season = np.zeros((self.S, self.H, self.W), np.float32)
        self.rate_all = np.zeros((self.S, self.H, self.W), np.float32)
        TC, TD = tc.sum(0), td.sum(0)
        for s in range(self.S):
            y, b = self.tgt_year[s], tgt_bucket[s]
            c, dd = TC.copy(), TD.copy()
            if y <= TRAIN_LAST_YEAR:
                c -= tc[y - 2014]
                dd -= td[y - 2014]
            self.rate_season[s] = np.where(dd[b] > 0, c[b] / np.maximum(dd[b], 1), 0)
            self.rate_all[s] = np.where(dd.sum(0) > 0, c.sum(0) / np.maximum(dd.sum(0), 1), 0)

    def _harmonic(self, maps, days, dredged, year, lam=60.0, loyo=True):
        years = np.arange(2014, TRAIN_LAST_YEAR + 1)
        A = np.zeros((len(years), self.H, self.W, 3, 3))
        b = np.zeros((len(years), self.H, self.W, 3))
        for p in range(len(maps) - 1):
            gap = days[p + 1] - days[p]
            if gap < 60 and year[p + 1] <= TRAIN_LAST_YEAR:
                step = maps[p + 1] - maps[p]
                ok = ~np.isnan(step) & ~dredged[p + 1]
                mid = self.dates[p] + gap // 2
                th = 2 * np.pi * (mid - mid.astype('datetime64[Y]')).astype(int) / 365.25
                x = np.array([1, np.sin(th), np.cos(th)])
                yi = year[p + 1] - 2014
                A[yi][ok] += gap * np.outer(x, x)
                b[yi][ok] += step[ok][:, None] * x
        mid = self.dates[self.windows[:, -2]] + (self.horizon // 2).astype('timedelta64[D]')
        th = 2 * np.pi * (mid - mid.astype('datetime64[Y]')).astype(int) / 365.25
        X = np.stack([np.ones_like(th), np.sin(th), np.cos(th)], 1)
        coef = {}
        for y in list(range(2014, TRAIN_LAST_YEAR + 1)) + [None]:
            keep = years != y
            AA, bb = A[keep].sum(0), b[keep].sum(0)
            g = np.linalg.solve(AA.sum((0, 1)), bb.sum((0, 1)))
            coef[y] = np.linalg.solve(AA + lam * np.eye(3), (bb + lam * g)[..., None])[..., 0]
        self.rate_harm = np.zeros((self.S, self.H, self.W), np.float32)
        for s in range(self.S):
            y = self.tgt_year[s] if (loyo and self.tgt_year[s] <= TRAIN_LAST_YEAR) else None
            self.rate_harm[s] = coef[y] @ X[s]

    def _drivers(self):
        before = np.stack([self.d[f'{n}_before'] for n in DRIVERS], 1).astype(np.float64)
        for j, n in enumerate(DRIVERS):
            if n in LOG:
                before[:, j] = np.log1p(before[:, j])
        drv = before[self.windows[:, :-1]]
        tr = self.idx['train']
        mu, sd = np.nanmean(drv[tr], axis=(0, 1)), np.nanstd(drv[tr], axis=(0, 1))
        self.drv = np.nan_to_num((drv - mu) / sd).astype(np.float32)
        a = self.d['tide_range_ahead']
        self.ahead = ((a - a[tr].mean()) / a[tr].std()).astype(np.float32)

    def _hot_cells(self):
        tr = self.idx['train']
        per28 = self.change[tr] / self.horizon[tr, None, None] * 28
        m = self.masks[tr]
        cnt = m.sum(0)
        mean = np.where(cnt > 0, (per28 * m).sum(0) / np.maximum(cnt, 1), 0)
        self.hot = (mean < -0.75) & self.core


def nan_box(a, size):
    ok = ~np.isnan(a)
    num = uniform_filter(np.where(ok, a, 0), size=size, axes=(-2, -1), mode='nearest')
    den = uniform_filter(ok.astype(np.float64), size=size, axes=(-2, -1), mode='nearest')
    return np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)


def ls_slope(vals, t):
    ok = ~np.isnan(vals)
    n = ok.sum(1)
    tt = np.where(ok, t, 0)
    vv = np.where(ok, vals, 0)
    tm = tt.sum(1) / np.maximum(n, 1)
    vm = vv.sum(1) / np.maximum(n, 1)
    cov = (np.where(ok, (t - tm[:, None]) * (vals - vm[:, None]), 0)).sum(1)
    var = (np.where(ok, (t - tm[:, None]) ** 2, 0)).sum(1)
    return np.where((n >= 2) & (var > 0), cov / np.maximum(var, 1e-9), np.nan)


def tabular(D, which=('train', 'val', 'test')):
    out = {}
    rr, cc = np.meshgrid(np.arange(D.H), np.arange(D.W), indexing='ij')
    for name in which:
        idx = D.idx[name]
        m = D.masks[idx]
        inp = D.inputs[idx].astype(np.float64)
        last = inp[:, -1]
        t = (D.in_days[idx] - D.in_days[idx, -1:]).astype(np.float64)[:, :, None, None] * np.ones_like(inp)
        feats, names = [], []

        def add(nm, arr):
            names.append(nm)
            feats.append(np.asarray(arr, np.float32)[m] if np.ndim(arr) == 3 else
                         np.broadcast_to(np.asarray(arr, np.float32)[:, None, None], m.shape)[m])

        for k in range(4):
            add(f'd{k}', inp[:, k] - last if k < 3 else last)
            add(f'obs{k}', ~np.isnan(inp[:, k]))
            add(f'dredged{k}', D.dredged[D.windows[idx, k]])
        slope = ls_slope(inp, t)
        add('slope', slope)
        add('slope_h', slope * D.horizon[idx][:, None, None])
        add('d3_d0', last - inp[:, 0])
        add('d3_d2', last - inp[:, 2])
        for s in (3, 5, 9, 15):
            add(f'nb{s}', nan_box(last, s) - last)
        add('nb_trend5', nan_box(last - inp[:, 0], 5))
        add('nb_trend15', nan_box(last - inp[:, 0], 15))
        gy, gx = np.gradient(np.nan_to_num(nan_box(last, 3)), axis=(-2, -1))
        add('grad', np.hypot(gx, gy))
        for k in range(3):
            add(f'gap{k}', D.in_days[idx, k + 1] - D.in_days[idx, k])
        h = D.horizon[idx]
        add('horizon', h)
        add('doy_sin', D.doy_sin[idx])
        add('doy_cos', D.doy_cos[idx])
        add('row', np.broadcast_to(rr, m.shape))
        add('col', np.broadcast_to(cc, m.shape))
        add('clim_season', D.rate_season[idx] * h[:, None, None])
        add('clim_all', D.rate_all[idx] * h[:, None, None])
        add('clim_harm', D.rate_harm[idx] * h[:, None, None])
        add('clim_harm_nb', nan_box(D.rate_harm[idx] * h[:, None, None], 5))
        add('hot', np.broadcast_to(D.hot, m.shape))
        for j, nm in enumerate(DRIVERS):
            add(f'{nm}_last', D.drv[idx, -1, j])
            add(f'{nm}_mean', D.drv[idx, :, j].mean(1))
        add('tide_ahead', D.ahead[idx])
        out[name] = (np.stack(feats, 1), D.change[idx][m], m)
    return out, names


def unflatten(D, name, flat):
    m = D.masks[D.idx[name]]
    out = np.zeros(m.shape, np.float32)
    out[m] = flat
    return out


def evaluate(D, preds, label, extra=None, t0=None):
    res = {'model': label}
    base = {}
    for name in ['val', 'test']:
        idx = D.idx[name]
        m = D.masks[idx]
        p, y = preds[name][m], D.change[idx][m]
        hot = np.broadcast_to(D.hot, m.shape)[m]
        base[name] = np.abs(y).mean()
        res[f'{name}_mae'] = float(np.abs(p - y).mean())
        res[f'{name}_hot_mae'] = float(np.abs(p - y)[hot].mean())
        if name == 'test':
            res['test_rmse'] = float(np.sqrt(((p - y) ** 2).mean()))
            res['test_median_ae'] = float(np.median(np.abs(p - y)))
            res['test_bias'] = float((p - y).mean())
            lab = y < -1
            res['test_auroc'] = float(roc_auc_score(lab, -p))
            res['test_ap'] = float(average_precision_score(lab, -p))
    res['test_vs_persistence_pct'] = float(100 * (1 - res['test_mae'] / base['test']))
    if t0 is not None:
        res['seconds'] = round(time.time() - t0)
    if extra:
        res.update(extra)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f'{label}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return res
