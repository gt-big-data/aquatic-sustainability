"""Add the LightGBM and TimeSformer forecasts to the dredging map.

Run after tools/anchorage/publish_overlays.py. Reuses its geometry, footprint and file contract,
leaves the existing models' files untouched, and is safe to re-run.
"""
import hashlib
import json
import os
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import torch

from common import DATA, HERE, Data, tabular

ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
TOOLS = os.path.join(ROOT, 'tools', 'anchorage')
STATIC = os.path.join(ROOT, 'backend', 'static')
MANIFEST = os.path.join(STATIC, 'data', 'anchorage', 'manifest.json')
ART = os.path.join(HERE, 'artifacts')
MODELS = {'lightgbm': 'LightGBM ensemble (Aashay)', 'timesformer': 'TimeSformer space-time transformer (Aashay)'}
sys.path.insert(0, TOOLS)
from forecast import FEET_TO_METRES, read_archive  # noqa: E402
from geojson_export import feature_collection, reference_footprint  # noqa: E402


def lightgbm_predict(idx, cells):
    D = Data()
    meta = json.load(open(os.path.join(ART, 'm02_lightgbm.json')))
    boosters = [lgb.Booster(model_file=os.path.join(ART, f'm02_lightgbm_{k}.txt')) for k in range(len(meta['params']))]
    D.masks = D.masks.copy()
    D.masks[idx] = cells[idx]
    D.idx['publish'] = idx
    tab, names = tabular(D, ('publish',))
    assert names == meta['features']
    X, _, m = tab['publish']
    ci = names.index('clim_harm')
    flat = np.mean([b.predict(X) + (X[:, ci] if p['target'] == 'resid' else 0)
                    for b, p in zip(boosters, meta['params'])], 0)
    out = np.full(m.shape, np.nan)
    out[m] = flat
    return out


def timesformer_predict(idx):
    import deep
    from m10_timesformer import TimeSformer
    torch.set_num_threads(2)
    D = Data(augment=True)
    T = deep.Tensors(D)
    model = TimeSformer(T.c_seq, T.c_stat)
    model.load_state_dict(torch.load(os.path.join(ART, 'm10_timesformer.pt')))
    return deep.predict(model, T, idx, 'seq').astype(np.float64)


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n')
    os.replace(tmp, path)


def main():
    src = np.load(DATA)
    maps, windows, split = src['maps'], src['windows'], src['split']
    archive = read_archive(DATA)
    train = windows[split == 'train']
    core = np.isfinite(maps[np.unique(train)]).mean(0) >= 0.8
    test = np.where(split == 'test')[0]
    origin_ok = np.isfinite(maps[windows[:, -2]])
    cells = core[None] & origin_ok

    preds = {'lightgbm': lightgbm_predict(test, cells), 'timesformer': timesformer_predict(test)}

    scores = {}
    for name, p in [('persistence', None)] + list(preds.items()):
        errs = []
        for j, i in enumerate(test):
            truth = maps[windows[i, -1]] - maps[windows[i, -2]]
            ok = core & np.isfinite(truth) & origin_ok[i]
            pr = np.zeros_like(truth) if p is None else p[j]
            assert np.isfinite(pr[ok]).all()
            errs.append(np.abs(pr[ok] - truth[ok]).mean())
        scores[name] = float(np.mean(errs))
    print('test MAE, per-survey average on the platform core (ft):', json.dumps(scores, indent=1))

    manifest = json.load(open(MANIFEST))
    manifest['models'] = [m for m in manifest['models'] if m['id'] not in MODELS]
    for f in manifest['frames']:
        for k in MODELS:
            f['outputs'].pop(k, None)
    base = manifest['models'][0]['id']

    collections = {}
    for f in manifest['frames']:
        o, t = np.datetime64(f['from_survey']), np.datetime64(f['predicted_for'])
        hit = [j for j, i in enumerate(test) if archive.dates[windows[i, -2]] == o and archive.dates[windows[i, -1]] == t]
        assert len(hit) == 1, f'no test window for frame {f["id"]}'
        j = hit[0]
        footprint = set(reference_footprint(os.path.join(STATIC, f['outputs'][base].removeprefix('/static/')), archive.grid))
        mask = np.zeros(core.shape, bool)
        for r, c in footprint:
            mask[r, c] = True
        history = windows[test[j], :-1]
        for k in MODELS:
            change_m = preds[k][j][mask] * FEET_TO_METRES
            collections[(f['id'], k)] = feature_collection({'core': mask}, archive, history, t, change_m, footprint)

    blob = json.dumps({key[0] + key[1]: c for key, c in sorted(collections.items())}, sort_keys=True).encode()
    generation = 'aashay-' + hashlib.sha256(blob).hexdigest()[:12]
    for f in manifest['frames']:
        for k in MODELS:
            rel = f'data/anchorage/predictions/{generation}/{k}-{f["id"]}.geojson'
            write_json(os.path.join(STATIC, rel), collections[(f['id'], k)])
            f['outputs'][k] = '/static/' + rel
    manifest['models'] += [{'id': k, 'label': v, 'mae_ft': scores[k]} for k, v in MODELS.items()]

    ev_path = os.path.join(os.path.dirname(MANIFEST), 'evaluation.json')
    ev = json.load(open(ev_path))
    for k, script in [('lightgbm', 'm02_lightgbm.py'), ('timesformer', 'm10_timesformer.py')]:
        ev['models'][k] = {'training_script': f'anchorage/aashay_models/{script}',
                           'evaluation': {'test': {'mae_ft': scores[k], 'persistence_mae_ft': scores['persistence'],
                                                   'windows': int(len(test))}}}
    write_json(ev_path, ev)
    write_json(MANIFEST, manifest)
    subprocess.run([sys.executable, os.path.join(TOOLS, 'export_comparisons.py')], check=True, cwd=ROOT)
    subprocess.run([sys.executable, os.path.join(TOOLS, 'verify_overlays.py')], check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
