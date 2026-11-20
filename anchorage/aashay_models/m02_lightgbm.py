import json
import os
import time

import numpy as np

import gbm
from common import HERE, Data, evaluate, tabular, unflatten

t0 = time.time()
D = Data()
tab, names = tabular(D, max_train_rows=1_500_000)
X, y, _ = tab['train']
rng = np.random.default_rng(0)
s = rng.choice(len(y), 500_000, replace=False)
v = rng.choice(len(tab['val'][1]), 200_000, replace=False)
Xv, yv = tab['val'][0][v], tab['val'][1][v]

trials = gbm.search(X[s], y[s], names, Xv, yv, n=14)
top = [t[1] for t in trials[:3]]
models = [gbm.fit(p, X, y, names, Xv, yv, lr=0.03, seed=k) for k, p in enumerate(top)]
pred = lambda Z: np.mean([gbm.predict(m, p, Z, names) for m, p in zip(models, top)], 0)
preds = {k: unflatten(D, k, pred(tab[k][0])) for k in ['val', 'test']}
imp = np.mean([m.feature_importance('gain') for m in models], 0)
clean = lambda p: {k: (v if isinstance(v, str) else float(v)) for k, v in p.items()}
evaluate(D, preds, 'm02_lightgbm', {'params': clean(top[0]), 'ensemble': [clean(p) for p in top],
                                    'iters': [m.best_iteration for m in models],
                                    'top_features': [names[i] for i in np.argsort(-imp)[:10]]}, t0)

art = os.path.join(HERE, 'artifacts')
os.makedirs(art, exist_ok=True)
for k, m in enumerate(models):
    m.save_model(os.path.join(art, f'm02_lightgbm_{k}.txt'), num_iteration=m.best_iteration)
json.dump({'params': [clean(p) for p in top], 'features': names}, open(os.path.join(art, 'm02_lightgbm.json'), 'w'), indent=1)
