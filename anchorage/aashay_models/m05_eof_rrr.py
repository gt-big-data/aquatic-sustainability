import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from common import Data, evaluate

t0 = time.time()
D = Data()
core = D.core
mean_depth = np.nanmean(D.inputs[D.idx['train']][:, -1], axis=0)


def design(idx):
    inp = D.inputs[idx]
    last = inp[:, -1]
    h = D.horizon[idx][:, None, None]
    blocks = [last - mean_depth, last - inp[:, 0], last - inp[:, 2], D.rate_harm[idx] * h]
    X = np.concatenate([np.nan_to_num(b)[:, core] for b in blocks], 1)
    g = np.c_[D.doy_sin[idx], D.doy_cos[idx], D.horizon[idx] / 28, D.drv[idx, -1], D.ahead[idx]]
    return X, g


tr = D.idx['train']
Xtr, gtr = design(tr)
Ytr = D.change[tr][:, core] - (D.rate_harm[tr] * D.horizon[tr, None, None])[:, core]
mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-6
mu_y = Ytr.mean(0)

best = None
for kx in [8, 16, 32, 64]:
    px = PCA(kx, random_state=0).fit((Xtr - mu_x) / sd_x)
    for ky in [4, 8, 16, 32]:
        py = PCA(ky, random_state=0).fit(Ytr - mu_y)
        for a in [1, 10, 100]:
            Z = lambda X, g, px=px: np.c_[px.transform((X - mu_x) / sd_x), g]
            r = Ridge(alpha=a).fit(Z(Xtr, gtr), py.transform(Ytr - mu_y))

            def predict(idx, r=r, py=py, Z=Z):
                X, g = design(idx)
                out = np.zeros((len(idx), D.H, D.W), np.float32)
                out[:, core] = py.inverse_transform(r.predict(Z(X, g))) + mu_y
                return out + D.rate_harm[idx] * D.horizon[idx, None, None]

            v = D.idx['val']
            s = np.abs(predict(v) - D.change[v])[D.masks[v]].mean()
            if best is None or s < best[0]:
                best = (s, kx, ky, a, predict)
print('val best', best[:4])
preds = {k: best[4](D.idx[k]) for k in ['val', 'test']}
evaluate(D, preds, 'm05_eof_reduced_rank', {'kx': best[1], 'ky': best[2], 'alpha': best[3]}, t0)
