import itertools
import time

import numpy as np

from common import Data, evaluate

t0 = time.time()
D = Data()


def holt(idx, alpha, beta, phi, w, kappa):
    inp = D.inputs[idx].astype(np.float64)
    t = D.in_days[idx].astype(np.float64)
    level = inp[:, 0].copy()
    trend = np.zeros_like(level)
    seen = ~np.isnan(level)
    level = np.where(seen, level, 0)
    for k in range(1, 4):
        dt = (t[:, k] - t[:, k - 1])[:, None, None]
        x = inp[:, k]
        ok = ~np.isnan(x)
        fc = level + trend * dt
        new_level = np.where(seen, alpha * x + (1 - alpha) * fc, x)
        new_trend = np.where(seen, beta * (new_level - level) / dt + (1 - beta) * trend, 0)
        level = np.where(ok, new_level, np.where(seen, fc, level))
        trend = np.where(ok, new_trend, trend)
        seen |= ok
    h = D.horizon[idx][:, None, None]
    damp = phi * (1 - phi ** (h / 7)) / (1 - phi) * 7 / np.maximum(h, 1) if phi < 1 else 1.0
    clim = D.rate_harm[idx] * h
    return (level - D.last[idx]) + kappa * trend * h * damp + w * clim


def mae(p, idx):
    m = D.masks[idx]
    return np.abs(np.where(m, p, 0) - D.change[idx])[m].mean()


tr = D.idx['train']
grid = itertools.product([0.5, 0.8, 1.0], [0.1, 0.3, 0.6, 1.0], [0.5, 0.8, 0.95], [0.8, 1.0, 1.2], [-0.6, -0.3, 0.0, 0.3])
best = None
for a, b, ph, w, kp in grid:
    s = mae(holt(tr, a, b, ph, w, kp), tr)
    if best is None or s < best[0]:
        best = (s, a, b, ph, w, kp)
print('train best', best)
_, a, b, ph, w, kp = best
preds = {k: np.nan_to_num(holt(D.idx[k], a, b, ph, w, kp)).astype(np.float32) for k in ['val', 'test']}
evaluate(D, preds, 'm04_damped_holt', {'alpha': a, 'beta': b, 'phi': ph, 'clim_weight': w, 'trend_weight': kp}, t0)
