"""Hot-spot evaluation: turn the hot-spot head into a yes/no map and measure how good it is.

A hot spot = a graded cell whose depth falls by more than 1 ft over the forecast window.
Trains the GNN and the no-graph MLP (3 seeds each), picks a probability cutoff on the VALIDATION years,
then reports precision / recall on the TEST years, plus a map of predicted vs actual hot spots.

Usage: python hotspots.py --data ../anchorage/data/timeseries_v1/timeseries.npz
Needs: pip install matplotlib   (anchorage_gnn.py must be in the same folder)
"""
import argparse
import numpy as np, torch
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import anchorage_gnn as G


def prf(flag, lab):
    tp, fp, fn = (flag & lab).sum(), (flag & ~lab).sum(), (~flag & lab).sum()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9), flag.mean()


def pick_threshold(score, lab):
    p, r, t = precision_recall_curve(lab, score)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-9)
    return t[np.argmax(f)]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--out', default='hotspot_map.png')
    ap.add_argument('--calib', choices=['platt', 'isotonic'], default='platt', help='probability calibration fitted on validation')
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    d = np.load(a.data, allow_pickle=True)
    core = d['core']; rr, cc = np.where(core)
    tdates = np.array(d['dates'])[d['windows'][:, -1]].astype('datetime64[D]')
    X, Y, M, split, graph, dsd, years = G.build(a.data)
    tr, va, te = (np.where(split == s)[0] for s in ('train', 'val', 'test'))
    L = lambda ids: (Y[ids] < G.HOT) & M[ids]                  # true hot-spot labels
    print(f'hot-spot rate among graded cells: train {L(tr).sum() / M[tr].sum():.3f}  '
          f'val {L(va).sum() / M[va].sum():.3f}  test {L(te).sum() / M[te].sum():.3f}')

    def ensemble(layers):
        runs = [G.train_gnn(X, Y, M, split, years, graph, dsd, dev, a.epochs, s, a.hidden, layers)[1] for s in range(a.seeds)]
        return {sp: np.mean([r[sp][1] for r in runs], 0) for sp in ('val', 'test')}   # mean hot-spot logit

    scores = {}
    print('training GNN ...'); scores['GNN'] = ensemble(a.layers)
    print('training MLP (no graph) ...'); scores['MLP (no graph)'] = ensemble(0)
    rate = L(tr).sum(0) / np.maximum(M[tr].sum(0), 1)           # each cell's historical hot-spot rate
    scores['per-cell history'] = {sp: np.broadcast_to(rate, Y[ids].shape) for sp, ids in (('val', va), ('test', te))}

    # ---- simple non-graph rivals built from the depth maps themselves
    Dm = d['maps'][d['windows'][:, :-1]][:, :, core]                    # (S, K, N) the input depths
    trend = np.nan_to_num(Dm[:, -1] - Dm[:, 0])                         # depth change across the input window (negative = shoaling)
    step = np.nan_to_num(Dm[:, -1] - Dm[:, -2])                         # change over the last interval
    with np.errstate(all='ignore'):
        cmean = np.nanmean(Dm[tr][:, -1], 0)
    rel = np.nan_to_num(Dm[:, -1] - np.where(np.isnan(cmean), np.nanmean(cmean), cmean))
    scores['recent trend'] = {sp: -trend[ids] for sp, ids in (('val', va), ('test', te))}   # "already shoaling -> keeps shoaling"
    F = np.stack([trend, step, np.broadcast_to(rate, trend.shape), rel], -1).astype(np.float32)
    ii = np.argwhere(M & (split == 'train')[:, None])
    ii = ii[np.random.default_rng(0).choice(len(ii), min(400_000, len(ii)), replace=False)]
    sc = StandardScaler().fit(F[ii[:, 0], ii[:, 1]])
    lr = LogisticRegression(max_iter=500).fit(sc.transform(F[ii[:, 0], ii[:, 1]]), ((Y < G.HOT) & M)[ii[:, 0], ii[:, 1]])
    scores['logistic: trend+history'] = {sp: lr.decision_function(sc.transform(F[ids].reshape(-1, 4))).reshape(Y[ids].shape)
                                         for sp, ids in (('val', va), ('test', te))}

    # ---- precision / recall at a cutoff chosen on validation
    print('\nCutoff chosen on VALIDATION (maximises F1), applied to TEST')
    print(f'{"model":<26}{"flagged":>9}{"precision":>11}{"recall":>9}{"F1":>7}{"AUROC":>8}{"AP":>7}')
    thr = {}
    for name, s in scores.items():
        thr[name] = pick_threshold(s['val'][M[va]], L(va)[M[va]])
        p, r, f, fl = prf((s['test'] >= thr[name])[M[te]], L(te)[M[te]])
        yy = L(te)[M[te]]; ss = s['test'][M[te]]
        print(f'{name:<26}{fl:9.1%}{p:11.2f}{r:9.2f}{f:7.2f}{roc_auc_score(yy, ss):8.3f}{average_precision_score(yy, ss):7.3f}')
    print(f'(a model with no skill flagging the same share would have precision ~ {L(te).sum() / M[te].sum():.2f})')

    print('\nIf you flag only the top X% highest-scoring test cells')
    print(f'{"model":<26}' + ''.join(f'{f"top {k}%":>16}' for k in (5, 10, 20)) + '      (precision / recall)')
    for name, s in scores.items():
        row = f'{name:<26}'
        for k in (5, 10, 20):
            t = np.quantile(s['test'][M[te]], 1 - k / 100)
            p, r, _, _ = prf((s['test'] >= t)[M[te]], L(te)[M[te]])
            row += f'{p:9.2f} /{r:5.2f} '
        print(row)

    # ---- calibration: fit on VALIDATION, check on TEST
    lv, lt = scores['GNN']['val'][M[va]], scores['GNN']['test'][M[te]]
    yv, yt = L(va)[M[va]], L(te)[M[te]]
    sig = lambda z: 1 / (1 + np.exp(-z))
    platt = LogisticRegression(C=1e6, max_iter=500).fit(lv[:, None], yv)
    iso = IsotonicRegression(out_of_bounds='clip').fit(sig(lv), yv)
    cands = {'raw': sig(lt), 'platt': platt.predict_proba(lt[:, None])[:, 1], 'isotonic': iso.predict(sig(lt))}
    def ece(p, y):
        b_ = np.minimum((p * 10).astype(int), 9)
        return sum(abs(p[b_ == k].mean() - y[b_ == k].mean()) * (b_ == k).mean() for k in range(10) if (b_ == k).any())
    print('\nCalibration on TEST (lower = better; Brier of always predicting the base rate = '
          f'{((yt.mean() - yt) ** 2).mean():.3f})')
    for k_, p_ in cands.items():
        print(f'  {k_:<9} Brier {((p_ - yt) ** 2).mean():.4f}   mean predicted {p_.mean():.3f} (actual {yt.mean():.3f})   ECE {ece(p_, yt):.3f}')
    fn = (lambda z: platt.predict_proba(z.reshape(-1, 1))[:, 1].reshape(z.shape)) if a.calib == 'platt' else \
         (lambda z: iso.predict(sig(z).ravel()).reshape(z.shape))
    pr = fn(scores['GNN']['test'])                                      # calibrated test probabilities, used below and in the map
    print(f'\nGNN probabilities ({a.calib}-calibrated): when it says X%, how often is it a hot spot? (test)')
    edges = [0, .05, .1, .2, .35, .5, .75, 1.0001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = M[te] & (pr >= lo) & (pr < hi)
        if m.sum():
            print(f'  predicted {lo:4.0%}-{min(hi, 1):4.0%}   cells {m.sum():8d}   actual hot-spot rate {L(te)[m].mean():5.1%}')
    print('\nUsing the calibrated probability directly as the cutoff (test)')
    print(f'{"flag if P >=":<14}{"flagged":>9}{"precision":>11}{"recall":>9}')
    for t_ in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        p_, r_, _, fl_ = prf((pr >= t_)[M[te]], yt)
        print(f'{t_:<14.0%}{fl_:9.1%}{p_:11.2f}{r_:9.2f}')

    # ---- per-year at the chosen cutoff
    print('\nGNN by test year (same cutoff)')
    yrs = years[te]
    for yr in sorted(set(yrs)):
        k = yrs == yr
        p, r, f, fl = prf((scores['GNN']['test'][k] >= thr['GNN'])[M[te][k]], L(te)[k][M[te][k]])
        print(f'  {yr}  hot-spot rate {L(te)[k][M[te][k]].mean():.3f}  flagged {fl:.1%}  precision {p:.2f}  recall {r:.2f}  F1 {f:.2f}')

    # ---- maps: one typical test window per year (median F1, not cherry-picked)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    flag_all = scores['GNN']['test'] >= thr['GNN']
    picks = []
    for yr in sorted(set(yrs)):
        idx = np.where(yrs == yr)[0]
        idx = [i for i in idx if L(te)[i].sum() >= 50] or list(idx)
        f1 = [prf(flag_all[i][M[te][i]], L(te)[i][M[te][i]])[2] for i in idx]
        picks.append(idx[np.argsort(f1)[len(f1) // 2]])
    def raster(v):
        img = np.full(core.shape, np.nan); img[rr, cc] = v; return img
    fig, ax = plt.subplots(len(picks), 3, figsize=(9, 5.5 * len(picks)), squeeze=False)
    cat = ListedColormap(['#e8e8e8', '#2ca02c', '#ff7f0e', '#d62728'])
    for r_, i in enumerate(picks):
        g = M[te][i]; lab, fl = L(te)[i], flag_all[i]
        ax[r_, 0].imshow(raster(np.where(g, lab, np.nan)), cmap=ListedColormap(['#e8e8e8', '#d62728']), vmin=0, vmax=1)
        ax[r_, 1].imshow(raster(np.where(g, pr[i], np.nan)), cmap='viridis', vmin=0, vmax=1)
        ax[r_, 2].imshow(raster(np.where(g, (fl & lab) * 1 + (fl & ~lab) * 2 + (~fl & lab) * 3, np.nan)), cmap=cat, vmin=0, vmax=3)
        p, rc, f, _ = prf(fl[g], lab[g])
        ax[r_, 0].set_title(f'Actual: shoaled >1 ft\n(window ending {tdates[te][i]})', fontsize=9)
        ax[r_, 1].set_title('Model probability', fontsize=9)
        ax[r_, 2].set_title(f'Flagged vs actual\nprecision {p:.2f}  recall {rc:.2f}', fontsize=9)
        for c in range(3): ax[r_, c].axis('off')
    fig.legend(handles=[Patch(color='#2ca02c', label='hit (flagged, real)'), Patch(color='#ff7f0e', label='false alarm'),
                        Patch(color='#d62728', label='missed'), Patch(color='#e8e8e8', label='not a hot spot')],
               loc='lower center', ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(a.out, dpi=130)
    print(f'\nsaved map -> {a.out}  (windows shown: {[str(tdates[te][i]) for i in picks]})')