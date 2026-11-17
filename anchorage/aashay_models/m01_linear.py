import time

import numpy as np
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler

from common import Data, evaluate, tabular, unflatten

t0 = time.time()
D = Data()
tab, names = tabular(D)
Xtr, ytr, _ = tab['train']
sc = StandardScaler().fit(np.nan_to_num(Xtr))
prep = lambda X: np.nan_to_num(sc.transform(np.nan_to_num(X)))
sub = np.random.default_rng(0).choice(len(ytr), 300_000, replace=False)

cands = [(f'ridge_a{a}', Ridge(alpha=a).fit(prep(Xtr), ytr)) for a in [10, 1000, 10000]]
cands += [(f'huber_e{e}', HuberRegressor(epsilon=e, alpha=1e-3, max_iter=300).fit(prep(Xtr[sub]), ytr[sub]))
          for e in [1.1, 1.35, 2.0]]
scores = [(np.abs(m.predict(prep(tab['val'][0])) - tab['val'][1]).mean(), n, m) for n, m in cands]
for s, n, _ in scores:
    print(n, round(s, 4))
_, name, model = min(scores, key=lambda x: x[0])
preds = {k: unflatten(D, k, model.predict(prep(tab[k][0]))) for k in ['val', 'test']}
evaluate(D, preds, 'm01_linear', {'chosen': name}, t0)
