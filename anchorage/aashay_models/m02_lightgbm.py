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
vsub = rng.choice(len(tab['val'][1]), 200_000, replace=False)

params = dict(objective='l1', learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=2, seed=0)
dtr = lgb.Dataset(Xtr[sub], ytr[sub], feature_name=names)
dva = lgb.Dataset(tab['val'][0][vsub], tab['val'][1][vsub], reference=dtr)
model = lgb.train(params, dtr, 3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

imp = sorted(zip(names, model.feature_importance('gain')), key=lambda x: -x[1])[:10]
preds = {k: unflatten(D, k, model.predict(tab[k][0], num_iteration=model.best_iteration)) for k in ['val', 'test']}
evaluate(D, preds, 'm02_lightgbm', {'best_iter': model.best_iteration,
                                    'top_features': [n for n, _ in imp]}, t0)
