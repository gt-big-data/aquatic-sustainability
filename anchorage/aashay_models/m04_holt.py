import itertools
import time

import numpy as np

from common import DATA, Data, evaluate

t0 = time.time()
D = Data()
L = 12
series = np.loadtxt(DATA.replace('timeseries.npz', 'timeseries_surveys.csv'), delimiter=',', skiprows=1,
                    usecols=11, dtype=str).astype(int)


def history(idx):
    last = D.windows[idx, 3]
    pos = last[:, None] - np.arange(L - 1, -1, -1)[None]
    ok = (pos >= 0) & (series[np.clip(pos, 0, None)] == series[last][:, None])
    pos = np.clip(pos, 0, None)
    x = D.maps[pos].astype(np.float64)
    x[~ok] = np.nan
    dred = D.dredged[pos] & ok[:, :, None, None]
    after = np.flip(np.cumsum(np.flip(dred, 1), 1), 1) > dred
    x[after] = np.nan
    return x, np.where(ok, D.days[pos], D.days[last][:, None]).astype(np.float64)


def holt(x, t, idx, alpha, beta, phi, w, kappa):
    level = np.full(x.shape[:1] + x.shape[2:], np.nan)
    trend = np.zeros_like(level)
    for k in range(L):
        dt = (t[:, k] - t[:, k - 1])[:, None, None] if k else np.ones((len(t), 1, 1))
        xk = x[:, k]
        ok = ~np.isnan(xk)
        seen = ~np.isnan(level)
        fc = np.where(seen, level + trend * dt, np.nan)
        nl = np.where(seen, alpha * xk + (1 - alpha) * fc, xk)
        nt = np.where(seen, beta * (nl - level) / dt + (1 - beta) * trend, 0)
        level = np.where(ok, nl, fc)
        trend = np.where(ok, nt, trend)
    h = D.horizon[idx][:, None, None]
    damp = phi * (1 - phi ** (h / 7)) / (1 - phi) * 7 / np.maximum(h, 1) if phi < 1 else 1.0
    return np.nan_to_num((level - D.last[idx]) + kappa * trend * h * damp + w * D.rate_harm[idx] * h)


tr = D.idx['train']
xt, tt = history(tr)
m = D.masks[tr]
grid = itertools.product([0.2, 0.35, 0.5, 0.7, 1.0], [0.05, 0.2, 0.5], [0.5, 0.9], [0.9, 1.0, 1.15], [-0.3, 0.0, 0.3])
best = None
for g in grid:
    s = np.abs(holt(xt, tt, tr, *g) - D.change[tr])[m].mean()
    if best is None or s < best[0]:
        best = (s, g)
print('train best', best, flush=True)
g = best[1]
preds = {}
for k in ['val', 'test']:
    x, t = history(D.idx[k])
    preds[k] = holt(x, t, D.idx[k], *g).astype(np.float32)
evaluate(D, preds, 'm04_damped_holt', dict(zip(['alpha', 'beta', 'phi', 'clim_weight', 'trend_weight'], g),
                                           history_maps=L), t0)
