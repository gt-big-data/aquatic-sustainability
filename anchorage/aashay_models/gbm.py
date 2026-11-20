import lightgbm as lgb
import numpy as np

BASE = dict(verbose=-1, num_threads=2, bagging_freq=1)
SPACE = dict(num_leaves=[31, 63, 127, 255], min_data_in_leaf=[100, 400, 1500], feature_fraction=[0.5, 0.7, 0.9],
             bagging_fraction=[0.6, 0.8, 1.0], lambda_l2=[0, 1, 10], objective=['l1', 'huber'], target=['raw', 'resid'])


def offset(X, names, target):
    return X[:, names.index('clim_harm')] if target == 'resid' else 0


def fit(params, X, y, names, Xv, yv, lr=0.05, seed=0, rounds=5000, patience=150):
    p = {k: v for k, v in params.items() if k != 'target'}
    p.update(BASE, learning_rate=lr, seed=seed)
    if p['objective'] == 'huber':
        p['alpha'] = 1.0
    o, ov = offset(X, names, params['target']), offset(Xv, names, params['target'])
    model = lgb.train(p, lgb.Dataset(X, y - o), rounds, valid_sets=[lgb.Dataset(Xv, yv - ov)],
                      feval=lambda pr, ds: ('mae', float(np.abs(pr - ds.get_label()).mean()), False),
                      callbacks=[lgb.early_stopping(patience, verbose=False)])
    return model


def predict(model, params, X, names):
    return model.predict(X, num_iteration=model.best_iteration) + offset(X, names, params['target'])


def search(X, y, names, Xv, yv, n=12, seed=0):
    rng = np.random.default_rng(seed)
    trials = []
    for i in range(n):
        params = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
        m = fit(params, X, y, names, Xv, yv, lr=0.08, patience=80)
        s = float(np.abs(predict(m, params, Xv, names) - yv).mean())
        trials.append((s, params, m.best_iteration))
        print(i, round(s, 4), m.best_iteration, params, flush=True)
    return sorted(trials, key=lambda t: t[0])
