import json
import os
import time

import lightgbm as lgb
import numpy as np

import gbm
from common import RESULTS, Data, evaluate, tabular, unflatten

t0 = time.time()
D = Data()
tab, names = tabular(D, max_train_rows=1_200_000)
X, y, _ = tab['train']
rng = np.random.default_rng(0)
v = rng.choice(len(tab['val'][1]), 200_000, replace=False)
Xv, yv = tab['val'][0][v], tab['val'][1][v]
best = json.load(open(os.path.join(RESULTS, 'm02_lightgbm.json')))['params']
best = {k: (int(val) if k in ('num_leaves', 'min_data_in_leaf') else val) for k, val in best.items()}
cp = {k: val for k, val in best.items() if k not in ('objective', 'target')}
cp.update(gbm.BASE, objective='multiclass', num_class=3, learning_rate=0.05, seed=0)


def build(tau):
    c, cv = np.digitize(y, [-tau, tau]), np.digitize(yv, [-tau, tau])
    clf = lgb.train(cp, lgb.Dataset(X, c), 3000, valid_sets=[lgb.Dataset(Xv, cv)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    regs = [gbm.fit(best, X[c == k], y[c == k], names, Xv[cv == k], yv[cv == k], lr=0.05) for k in range(3)]
    return clf, regs


def predict(model, Z, mode):
    clf, regs = model
    p = clf.predict(Z, num_iteration=clf.best_iteration)
    r = np.stack([gbm.predict(g, best, Z, names) for g in regs], 1)
    if mode == 'expected':
        return (p * r).sum(1)
    if mode == 'argmax':
        return r[np.arange(len(r)), p.argmax(1)]
    return np.where(p.max(1) > 0.6, r[np.arange(len(r)), p.argmax(1)], (p * r).sum(1))


trials = []
for tau in [0.75, 1.0, 1.5]:
    model = build(tau)
    for mode in ['expected', 'argmax', 'confident']:
        s = float(np.abs(predict(model, Xv, mode) - yv).mean())
        print(tau, mode, round(s, 4), flush=True)
        trials.append((s, tau, mode, model))
_, tau, mode, model = min(trials, key=lambda t: t[0])
preds = {k: unflatten(D, k, predict(model, tab[k][0], mode)) for k in ['val', 'test']}
evaluate(D, preds, 'm03_hurdle_lgbm', {'tau': tau, 'combine': mode, 'reg_params': best,
                                       'clf_iter': model[0].best_iteration}, t0)
