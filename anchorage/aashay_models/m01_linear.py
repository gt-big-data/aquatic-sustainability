import time

import numpy as np
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler

from common import Data, evaluate, tabular, unflatten

t0 = time.time()
D = Data()
tab, names = tabular(D, max_train_rows=1_200_000)
SPL = ['clim_harm', 'clim_harm_nb', 'd3', 'nb5', 'nb15', 'nb_trend15', 'grad', 'row', 'col', 'slope_h', 'd3_d0']
INT = ['shipcreek_flow_last', 'knik_flow_last', 'matanuska_flow_last', 'tide_ahead', 'doy_sin', 'doy_cos', 'horizon']
ci = names.index('clim_harm')

X0 = np.nan_to_num(tab['train'][0])
sc = StandardScaler().fit(X0)
lo, hi = np.percentile(X0, 1, 0), np.percentile(X0, 99, 0)
spl = SplineTransformer(n_knots=8, degree=3).fit(np.clip(X0, lo, hi)[:, [names.index(n) for n in SPL]])


def design(X, kind):
    X = np.nan_to_num(X)
    Z = [sc.transform(X)]
    if kind != 'linear':
        Z.append(spl.transform(np.clip(X, lo, hi)[:, [names.index(n) for n in SPL]]))
        Zs = sc.transform(X)
        Z.append(Zs[:, [ci]] * Zs[:, [names.index(n) for n in INT]])
    return np.hstack(Z)


ytr = tab['train'][1] - tab['train'][0][:, ci]
yv = tab['val'][1] - tab['val'][0][:, ci]
sub = np.random.default_rng(0).choice(len(ytr), 300_000, replace=False)
cands = []
for kind in ['linear', 'gam']:
    Ztr, Zv = design(tab['train'][0], kind), design(tab['val'][0], kind)
    for a in [1, 100, 10000]:
        cands.append((f'{kind}_ridge_a{a}', kind, Ridge(alpha=a).fit(Ztr, ytr)))
    for e in [1.1, 1.5]:
        cands.append((f'{kind}_huber_e{e}', kind, HuberRegressor(epsilon=e, alpha=1e-3, max_iter=500).fit(Ztr[sub], ytr[sub])))
    for n, k, m in cands[-5:]:
        print(n, round(float(np.abs(m.predict(Zv) - yv).mean()), 4), flush=True)
scores = [(np.abs(m.predict(design(tab['val'][0], k)) - yv).mean(), n, k, m) for n, k, m in cands]
_, name, kind, model = min(scores, key=lambda x: x[0])
preds = {s: unflatten(D, s, model.predict(design(tab[s][0], kind)) + tab[s][0][:, ci]) for s in ['val', 'test']}
evaluate(D, preds, 'm01_linear', {'chosen': name}, t0)
