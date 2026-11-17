import time

import numpy as np

from common import Data, evaluate

D = Data()
t0 = time.time()
zero = {k: np.zeros_like(D.change[D.idx[k]]) for k in ['val', 'test']}
evaluate(D, zero, 'baseline_persistence', t0=t0)
clim = {k: D.rate_season[D.idx[k]] * D.horizon[D.idx[k], None, None] for k in ['val', 'test']}
evaluate(D, clim, 'baseline_seasonal_rate', t0=t0)
harm = {k: D.rate_harm[D.idx[k]] * D.horizon[D.idx[k], None, None] for k in ['val', 'test']}
evaluate(D, harm, 'baseline_harmonic_rate', t0=t0)
