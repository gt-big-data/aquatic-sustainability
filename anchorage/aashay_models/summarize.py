import glob
import json
import os

import pandas as pd

from common import RESULTS

rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RESULTS, '*.json')))]
cols = ['model', 'val_mae', 'val_hot_mae', 'test_mae', 'test_hot_mae', 'test_rmse', 'test_median_ae', 'test_bias',
        'test_auroc', 'test_ap', 'test_vs_persistence_pct', 'seconds']
df = pd.DataFrame(rows)[cols].sort_values('test_mae')
df.to_csv(os.path.join(RESULTS, 'summary.csv'), index=False, float_format='%.3f')
print(df.to_markdown(index=False, floatfmt='.3f'))
