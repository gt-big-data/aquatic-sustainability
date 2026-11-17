import time

import lightgbm as lgb
import numpy as np

from common import Data, evaluate, tabular, unflatten

t0 = time.time()
D = Data()
tab, names = tabular(D)
rng = np.random.default_rng(0)
Xtr, ytr, _ = tab['train']
sub = rng.choice(len(ytr), min(800_000, len(ytr)), replace=False)
X, y = Xtr[sub], ytr[sub]
cls = np.digitize(y, [-1, 1])

base = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=2, seed=0)
clf = lgb.train({**base, 'objective': 'multiclass', 'num_class': 3}, lgb.Dataset(X, cls), 400)
regs = [lgb.train({**base, 'objective': 'l1'}, lgb.Dataset(X[cls == c], y[cls == c]), 400) for c in range(3)]


def predict(Xq, mode):
    p = clf.predict(Xq)
    r = np.stack([g.predict(Xq) for g in regs], 1)
    if mode == 'expected':
        return (p * r).sum(1)
    return r[np.arange(len(r)), p.argmax(1)]


best = min(['expected', 'argmax'], key=lambda md: np.abs(predict(tab['val'][0], md) - tab['val'][1]).mean())
preds = {k: unflatten(D, k, predict(tab[k][0], best)) for k in ['val', 'test']}
evaluate(D, preds, 'm03_hurdle_lgbm', {'combine': best}, t0)
