"""Leave-one-season-out check of the hot-spot results.

For each target year h (2014..2026): test = year h, validation = the neighbouring year (h-1, or h+1 for the
first year), train = all other years. Features, per-cell statistics and normalisation are rebuilt from the
training years of that fold only. Early stopping and the F1 cutoff use the validation year.

Compares: GNN, no-graph MLP, recent-trend rule, per-cell history, on the hot-spot task (shoals > 1 ft), plus depth-change MAE.
Usage:  python loso.py --data ../anchorage/data/timeseries_v1/timeseries.npz        (~40 min on an M-series Mac)
Writes loso_results.csv as it goes. Needs anchorage_gnn.py and hotspots.py in the same folder.
"""
import argparse, csv, time
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
import anchorage_gnn as G
from hotspots import prf, pick_threshold

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--epochs', type=int, default=30)
ap.add_argument('--patience', type=int, default=8)
ap.add_argument('--seeds', type=int, default=2)
ap.add_argument('--hidden', type=int, default=64)
ap.add_argument('--layers', type=int, default=3)
ap.add_argument('--only', default='', help='comma list of held-out years, e.g. 2019,2024 (for quick tests)')
ap.add_argument('--out', default='loso_results.csv')
a = ap.parse_args()

dev = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
d = np.load(a.data, allow_pickle=True)
W, core = d['windows'], d['core']
tyear = np.array(d['dates'])[W[:, -1]].astype('datetime64[Y]').astype(int) + 1970
all_years = sorted(set(tyear))
todo = [int(y) for y in a.only.split(',')] if a.only else all_years
Dm = d['maps'][W[:, :-1]][:, :, core]
trend = -np.nan_to_num(Dm[:, -1] - Dm[:, 0])             # higher = was shoaling more across the input window
print('device', dev, '| samples per target year:', {int(y): int((tyear == y).sum()) for y in all_years})

MODELS = ['GNN', 'MLP', 'trend rule', 'cell history']
rows = []
for h in todo:
    t0 = time.time()
    v = h - 1 if (h - 1) in all_years else h + 1
    sp = np.where(tyear == h, 'test', np.where(tyear == v, 'val', 'train'))
    X, Y, M, split, graph, dsd, years = G.build(a.data, split_override=sp)
    tr, va, te = (np.where(split == s)[0] for s in ('train', 'val', 'test'))
    Lab = (Y < G.HOT) & M

    def ens(layers):
        runs = [G.train_gnn(X, Y, M, split, years, graph, dsd, dev, a.epochs, s, a.hidden, layers, patience=a.patience)[1]
                for s in range(a.seeds)]
        return {k: (np.mean([r[k][0] for r in runs], 0), np.mean([r[k][1] for r in runs], 0)) for k in ('val', 'test')}

    print(f'\n=== held-out {h} (val {v}, train {len(tr)} samples, test {len(te)}) ===', flush=True)
    g_out = ens(a.layers); m_out = ens(0)
    rate = Lab[tr].sum(0) / np.maximum(M[tr].sum(0), 1)
    scores = {
        'GNN': {k: g_out[k][1] for k in ('val', 'test')},
        'MLP': {k: m_out[k][1] for k in ('val', 'test')},
        'trend rule': {'val': trend[va], 'test': trend[te]},
        'cell history': {k: np.broadcast_to(rate, Y[i].shape) for k, i in (('val', va), ('test', te))},
    }
    yt = Lab[te][M[te]]
    row = {'year': h, 'n_test': len(te), 'hot_rate': yt.mean()}
    for name, s in scores.items():
        thr = pick_threshold(s['val'][M[va]], Lab[va][M[va]])
        ss = s['test'][M[te]]
        p, r, f, _ = prf(ss >= thr, yt)
        p10, _, _, _ = prf(ss >= np.quantile(ss, 0.9), yt)
        row.update({f'{name}|auroc': roc_auc_score(yt, ss), f'{name}|ap': average_precision_score(yt, ss),
                    f'{name}|f1': f, f'{name}|p10': p10})
    row['GNN|mae'] = np.abs(g_out['test'][0] - Y[te])[M[te]].mean()
    row['persistence|mae'] = np.abs(Y[te])[M[te]].mean()
    rows.append(row)
    print(f'  hot-spot rate {row["hot_rate"]:.3f} | AUROC  ' + '  '.join(f'{m} {row[m + "|auroc"]:.3f}' for m in MODELS) +
          f' | MAE GNN {row["GNN|mae"]:.3f} vs persistence {row["persistence|mae"]:.3f}   ({time.time() - t0:.0f}s)', flush=True)
    with open(a.out, 'w', newline='') as f_:
        w_ = csv.DictWriter(f_, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
    del X, Y, M

# ------------------------------------------------------------------ summary
def table(metric, title):
    print(f'\n{title}')
    print(f'{"year":<6}{"hot%":>6}' + ''.join(f'{m:>13}' for m in MODELS))
    for r in rows:
        print(f'{r["year"]:<6}{r["hot_rate"]:6.2f}' + ''.join(f'{r[m + "|" + metric]:13.3f}' for m in MODELS))
    print(f'{"mean":<12}' + ''.join(f'{np.mean([r[m + "|" + metric] for r in rows]):13.3f}' for m in MODELS))

table('auroc', 'AUROC by held-out season (ranking quality; 0.5 = chance)')
table('ap', 'Average precision by held-out season (chance = hot%)')
table('f1', 'F1 at the cutoff chosen on the validation year')
table('p10', 'Precision when flagging the top 10% of cells')

print('\nHead-to-head across held-out seasons (paired differences)')
print(f'{"comparison":<28}{"metric":<8}{"mean diff":>10}{"wins":>8}{"  per-season diffs"}')
for A, B in (('GNN', 'trend rule'), ('GNN', 'MLP'), ('MLP', 'trend rule')):
    for met in ('auroc', 'ap', 'f1', 'p10'):
        diff = np.array([r[f'{A}|{met}'] - r[f'{B}|{met}'] for r in rows])
        print(f'{A + " - " + B:<28}{met:<8}{diff.mean():+10.3f}{f"{(diff > 0).sum()}/{len(diff)}":>8}   ' +
              ' '.join(f'{x:+.2f}' for x in diff))
diff = np.array([r['persistence|mae'] - r['GNN|mae'] for r in rows])
print(f'\nDepth-change MAE: GNN beats persistence in {(diff > 0).sum()}/{len(diff)} seasons, mean improvement {diff.mean():+.3f} ft')
print(f'\nsaved per-season results -> {a.out}')