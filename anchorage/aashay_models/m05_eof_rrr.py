import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from common import Data, TRAIN_LAST_YEAR, evaluate, nan_box

t0 = time.time()
D = Data(augment=True)
core = D.core
tr = D.idx['train']


def clim(idx):
    return (D.rate_harm[idx] * D.horizon[idx, None, None])[:, core]


def design(idx):
    inp = D.inputs[idx]
    last = inp[:, -1]
    blocks = [last - nan_box(last, 15), last - inp[:, 0], last - inp[:, 2]]
    X = np.concatenate([np.nan_to_num(b)[:, core] for b in blocks] + [clim(idx)], 1)
    g = np.c_[D.doy_sin[idx], D.doy_cos[idx], D.horizon[idx] / 28]
    return X, g


def target(idx):
    return np.where(D.masks[idx], D.change[idx], 0)[:, core] - clim(idx) * D.masks[idx][:, core]


X_all, g_all = design(tr)
Y_all, M_all = target(tr), D.masks[tr][:, core]
yr = D.tgt_year[tr]


class EOF:
    def __init__(self, X, g, Y, kx=32, ky=16):
        self.mx, self.sx, self.my = X.mean(0), X.std(0) + 1e-6, Y.mean(0)
        self.px = PCA(kx, svd_solver='randomized', random_state=0).fit((X - self.mx) / self.sx)
        self.py = PCA(ky, svd_solver='randomized', random_state=0).fit(Y - self.my)
        self.Zx = self.zx(X)
        self.Zy = self.py.transform(Y - self.my)

    def zx(self, X):
        return self.px.transform((X - self.mx) / self.sx)

    def fit(self, g, kx, ky, a):
        self.k = (kx, ky)
        self.r = Ridge(alpha=a).fit(np.c_[self.Zx[:, :kx], g], self.Zy[:, :ky])
        return self

    def predict(self, X, g):
        kx, ky = self.k
        zy = self.r.predict(np.c_[self.zx(X)[:, :kx], g])
        return zy @ self.py.components_[:ky] + self.my


grid = [(kx, ky, a) for kx in [4, 8, 16, 32] for ky in [2, 4, 8, 16] for a in [10, 100, 1000]]
cv = {g: [] for g in grid}
for y in range(2014, TRAIN_LAST_YEAR + 1):
    fit_rows, hold = yr != y, (yr == y) & ~D.aug[tr]
    if hold.sum() == 0:
        continue
    e = EOF(X_all[fit_rows], g_all[fit_rows], Y_all[fit_rows])
    for g in grid:
        p = e.fit(g_all[fit_rows], *g).predict(X_all[hold], g_all[hold])
        cv[g].append(np.abs(p - Y_all[hold])[M_all[hold]].mean())
best = min(grid, key=lambda g: np.mean(cv[g]))
print('cv best', best, round(float(np.mean(cv[best])), 4), flush=True)
model = EOF(X_all, g_all, Y_all).fit(g_all, *best)
preds = {}
for k in ['val', 'test']:
    idx = D.idx[k]
    out = np.zeros((len(idx), D.H, D.W), np.float32)
    X, g = design(idx)
    out[:, core] = model.predict(X, g) + clim(idx)
    preds[k] = out
evaluate(D, preds, 'm05_eof_reduced_rank', {'kx': best[0], 'ky': best[1], 'alpha': best[2],
                                            'selection': 'leave-one-year-out CV on train', 'depth_input': 'local (15x15) relative depth'}, t0)
