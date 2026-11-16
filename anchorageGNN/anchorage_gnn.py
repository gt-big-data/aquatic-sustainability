"""Anchorage shoaling: baselines + grid-graph GNN.

From K depth maps (+ gaps, season, static cell info) predict the change in depth
(target - last input) over the horizon, on core-channel cells, graded only where masks[n] is True.
Also reports the hot-spot task: will the cell shoal by more than 1 ft (delta < -1)?

Usage:  python anchorage_gnn.py --data data/timeseries_v1/timeseries.npz --epochs 60 --seeds 3
"""
import argparse, time
import numpy as np
import torch, torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score

HOT = -1.0   # ft: delta below this = hot spot (cell got >1 ft shallower)


# ----------------------------------------------------------------------------- data
def build(path, groups=(), long_window=0):
    d = np.load(path, allow_pickle=True)
    maps, dred, core = d['maps'], d['dredged'], d['core']
    W, masks, split, dates = d['windows'], d['masks'], d['split'], d['dates']
    K = int(d['K']); S = len(W); rr, cc = np.where(core); N = len(rr)

    cm, dm = maps[:, core], dred[:, core]                    # (surveys, N)
    D = cm[W[:, :K]]                                         # (S, K, N) inputs
    Y = cm[W[:, K]] - D[:, -1]                               # (S, N) change to predict
    M = masks[:, core] & ~np.isnan(Y)
    Y = np.where(M, Y, 0).astype(np.float32)

    # was the cell dredged since the previous input (k=0: since the survey before it)?
    DR = np.zeros((S, K, N), np.float32)
    for n, w in enumerate(W):
        for k in range(K):
            lo = w[k - 1] + 1 if k else w[0]
            DR[n, k] = dm[lo:w[k] + 1].any(0)

    tr = split == 'train'                                    # statistics from TRAIN only
    mu, sd = np.nanmean(D[tr]), np.nanstd(D[tr])
    last = D[:, -1]
    with np.errstate(all='ignore'):
        cell_depth = np.nanmean(last[tr], 0)
    cell_depth = np.where(np.isnan(cell_depth), mu, cell_depth)
    cell_delta = (Y * M)[tr].sum(0) / np.maximum(M[tr].sum(0), 1)
    dsd = float(Y[M & tr[:, None]].std())                    # delta scale

    feats = []
    for k in range(K):
        feats += [np.nan_to_num((D[:, k] - mu) / sd), (~np.isnan(D[:, k])).astype(np.float32), DR[:, k]]
    for k in range(1, K):
        feats.append(np.nan_to_num((D[:, k] - D[:, k - 1]) / dsd))
    # graph-level scalars broadcast to every node
    dts = dates[W].astype('datetime64[D]')
    gaps = np.diff(dts, axis=1).astype(float) / 30.0         # (S, K) incl. last input -> target
    doy = (dts[:, K - 1] - dts[:, K - 1].astype('datetime64[Y]')).astype(float) / 365.0
    pre18 = (dts[:, K - 1].astype('datetime64[Y]').astype(int) + 1970 < 2018).astype(float)
    scal = np.column_stack([gaps, np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy), pre18])
    feats += [np.repeat(scal[:, i:i + 1], N, 1) for i in range(scal.shape[1])]
    # outside drivers (tides / rivers / weather), standardised with TRAIN statistics; NaN -> 0 plus a flag
    names = []
    if groups:
        from anchorage_drivers import driver_features
        for name, v in driver_features(d, groups, long_window).items():
            v = np.asarray(v, float); nan = np.isnan(v)
            m, sdv = np.nanmean(v[tr]), np.nanstd(v[tr])
            z = np.where(nan, 0.0, (v - m) / (sdv if sdv > 0 else 1.0))
            feats.append(np.repeat(z[:, None], N, 1)); names.append(name)
            if nan.any():
                feats.append(np.repeat(nan[:, None].astype(np.float32), N, 1)); names.append(name + '_missing')
        print('driver features:', names)
    # static per-cell
    static = [rr / rr.max(), cc / cc.max(), (cell_depth - mu) / sd, cell_delta / dsd]
    feats += [np.broadcast_to(s[None], (S, N)) for s in static]
    feats.append(np.nan_to_num((last - cell_depth) / sd))
    X = np.stack(feats, -1).astype(np.float32)               # (S, N, F)

    # graph: 8-neighbour grid restricted to core cells, fixed for all samples
    idx = -np.ones(core.shape, int); idx[core] = np.arange(N)
    E = []
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        r2, c2 = rr + dr, cc + dc
        ok = (r2 < core.shape[0]) & (c2 >= 0) & (c2 < core.shape[1])
        ok[ok] &= core[r2[ok], c2[ok]]
        E.append(np.stack([idx[rr[ok], cc[ok]], idx[r2[ok], c2[ok]], np.full(ok.sum(), dr), np.full(ok.sum(), dc)]))
    E = np.concatenate(E, 1)
    src = np.concatenate([E[0], E[1]]); dst = np.concatenate([E[1], E[0]])
    dr_, dc_ = np.concatenate([E[2], -E[2]]), np.concatenate([E[3], -E[3]])
    ea = np.column_stack([dr_, dc_, np.hypot(dr_, dc_)]).astype(np.float32)
    graph = dict(src=src, dst=dst, ea=ea, N=N)
    years = dts[:, K].astype('datetime64[Y]').astype(int) + 1970   # year of the target
    return X, Y, M, split, graph, dsd, years


# ----------------------------------------------------------------------------- metrics
def metrics(pred, Y, M, score=None):
    p, y = pred[M], Y[M]
    out = dict(MAE=np.abs(p - y).mean(), RMSE=np.sqrt(((p - y) ** 2).mean()))
    lab = y < HOT
    s = -p if score is None else score[M]
    out['AUROC'] = roc_auc_score(lab, s) if 0 < lab.sum() < len(lab) else np.nan
    out['AP'] = average_precision_score(lab, s) if lab.sum() else np.nan
    out['hot_rate'] = lab.mean()
    out['bias'] = (p - y).mean()
    return out


# ----------------------------------------------------------------------------- baselines
def run_baselines(X, Y, M, split, dsd, rows, seed=0):
    tr, va, te = (split == s for s in ('train', 'val', 'test'))
    out = {}
    glob = Y[tr][M[tr]].mean()
    cd = X[tr][:, :, -2].mean(0) * dsd                        # per-cell mean delta (train)
    preds = {
        'persistence (0)': lambda s: np.zeros_like(Y[s]),
        'train-mean delta': lambda s: np.full_like(Y[s], glob),
        'per-cell mean delta': lambda s: np.broadcast_to(cd, Y[s].shape),
    }
    for name, f in preds.items():
        out[name] = {sp: metrics(f(m), Y[m], M[m]) for sp, m in (('val', va), ('test', te))}

    # per-cell gradient boosting, no graph: same node features, rows subsampled
    rng = np.random.default_rng(seed)
    ii = np.argwhere(M & tr[:, None]); ii = ii[rng.choice(len(ii), min(rows, len(ii)), replace=False)]
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                        l2_regularization=1.0, random_state=seed)
    hgb.fit(X[ii[:, 0], ii[:, 1]], Y[ii[:, 0], ii[:, 1]])
    def hp(m):
        return hgb.predict(X[m].reshape(-1, X.shape[2])).reshape(Y[m].shape).astype(np.float32)
    out['HistGB (no graph)'] = {sp: metrics(hp(m), Y[m], M[m]) for sp, m in (('val', va), ('test', te))}
    return out


# ----------------------------------------------------------------------------- GNN
class Layer(nn.Module):
    def __init__(s, H, de=3, p=0.1):
        super().__init__()
        s.self_, s.msg = nn.Linear(H, H), nn.Linear(H, H)
        s.edge = nn.Linear(de, H, bias=False)
        s.norm, s.drop = nn.LayerNorm(H), nn.Dropout(p)

    def forward(s, h, src, dst, ea, deg):                     # h: (B, N, H)
        m = torch.relu(s.msg(h)[:, src] + s.edge(ea))         # (B, E, H) messages
        agg = torch.zeros_like(h).index_add_(1, dst, m) / deg
        return h + s.drop(torch.relu(s.norm(s.self_(h) + agg)))


class GNN(nn.Module):
    def __init__(s, F, H=64, L=3, p=0.1):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(F, H), nn.ReLU())
        s.layers = nn.ModuleList([Layer(H, p=p) for _ in range(L)])
        s.head = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 2))   # [delta, hot-spot logit]

    def forward(s, x, g):
        h = s.enc(x)
        for l in s.layers:
            h = l(h, g['src'], g['dst'], g['ea'], g['deg'])
        o = s.head(h)
        return o[..., 0], o[..., 1]


def train_gnn(X, Y, M, split, years, graph, dsd, dev, epochs, seed, H=64, L=3, bs=8, lr=5e-4, wd=5e-2,
              patience=20, clip=1.0, loss_kind='l1', ema=0.99, min_year=0, p_drop=0.2):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = np.where((split == 'train') & (years >= min_year))[0]
    va, te = np.where(split == 'val')[0], np.where(split == 'test')[0]
    g = dict(src=torch.tensor(graph['src'], device=dev), dst=torch.tensor(graph['dst'], device=dev),
             ea=torch.tensor(graph['ea'], device=dev))
    deg = torch.zeros(graph['N']).index_add_(0, torch.tensor(graph['dst']), torch.ones(len(graph['dst'])))
    g['deg'] = deg.clamp(min=1).view(1, -1, 1).to(dev)
    Xt, Yt, Mt = (torch.tensor(a) for a in (X, Y / dsd, M.astype(np.float32)))
    model = GNN(X.shape[2], H, L, p_drop).to(dev)
    if ema > 0:                                              # smoothed copy of the weights, used for eval
        emod = GNN(X.shape[2], H, L, p_drop).to(dev); emod.load_state_dict(model.state_dict())
    else:
        emod = model
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def predict(ids):
        emod.eval(); P, Q = [], []
        with torch.no_grad():
            for i in range(0, len(ids), 16):
                b = ids[i:i + 16]
                p, q = emod(Xt[b].to(dev), g)
                P.append(p.cpu().numpy() * dsd); Q.append(q.cpu().numpy())
        return np.concatenate(P), np.concatenate(Q)

    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr); tot = 0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x, y, m = Xt[b].to(dev), Yt[b].to(dev), Mt[b].to(dev)
            p, q = model(x, g)
            reg = (p - y).abs() if loss_kind == 'l1' else nn.functional.huber_loss(p, y, reduction='none', delta=0.3)
            bce = nn.functional.binary_cross_entropy_with_logits(q, (y * dsd < HOT).float(), reduction='none')
            loss = ((reg + 0.5 * bce) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward()
            if clip: nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step(); tot += loss.item()
            if ema > 0:
                with torch.no_grad():
                    for pe, pm in zip(emod.parameters(), model.parameters()):
                        pe.mul_(ema).add_(pm.detach(), alpha=1 - ema)
        sched.step()
        pv, _ = predict(va); vmae = np.abs(pv - Y[va])[M[va]].mean()
        if vmae < best:
            best, bad = vmae, 0
            best_state = {k: v.detach().clone() for k, v in emod.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            print(f'  ep {ep:3d} train {tot / max(1, len(perm) // bs):.4f}  val MAE {vmae:.4f}  best {best:.4f}')
        if bad >= patience:
            break
    emod.load_state_dict(best_state)
    out, preds = {}, {}
    for sp, ids in (('val', va), ('test', te)):
        p, q = predict(ids); preds[sp] = (p, q)
        out[sp] = metrics(p, Y[ids], M[ids], score=q)
    return out, preds


# ----------------------------------------------------------------------------- main
def show(title, res):
    print(f'\n{title}')
    print(f'{"model":<26}{"split":<6}{"MAE":>8}{"RMSE":>8}{"AUROC":>8}{"AP":>8}{"bias":>8}')
    for name, r in res.items():
        for sp in ('val', 'test'):
            m = r[sp]
            print(f'{name:<26}{sp:<6}{m["MAE"]:8.3f}{m["RMSE"]:8.3f}{m["AUROC"]:8.3f}{m["AP"]:8.3f}{m["bias"]:8.3f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/timeseries_v1/timeseries.npz')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--hgb_rows', type=int, default=400_000)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=3, help='0 = same model with no message passing (MLP ablation)')
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=5e-2)
    ap.add_argument('--loss', choices=['l1', 'huber'], default='l1')
    ap.add_argument('--ema', type=float, default=0.99, help='weight EMA decay, 0 = off')
    ap.add_argument('--min_year', type=int, default=0, help='train only on target years >= this (e.g. 2018)')
    ap.add_argument('--drivers', default='', help='comma list of: tide,river,weather,oracle (needs timeseries_v2)')
    ap.add_argument('--long_window', type=int, default=0, help='also add L-day river/weather aggregates')
    ap.add_argument('--skip_baselines', action='store_true')
    a = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')
    print('device', dev)
    X, Y, M, split, graph, dsd, years = build(a.data, [g for g in a.drivers.split(',') if g], a.long_window)
    print(f'X {X.shape}  edges {len(graph["src"])}  delta std {dsd:.3f} ft  '
          f'hot-spot rate (train) {(Y[(split == "train")][M[split == "train"]] < HOT).mean():.3f}')

    res = {} if a.skip_baselines else run_baselines(X, Y, M, split, dsd, a.hgb_rows)
    runs, allp = [], []
    for s in range(a.seeds):
        t = time.time(); print(f'GNN seed {s}')
        r, pr = train_gnn(X, Y, M, split, years, graph, dsd, dev, a.epochs, s, a.hidden, a.layers, lr=a.lr, wd=a.wd,
                          loss_kind=a.loss, ema=a.ema, min_year=a.min_year)
        runs.append(r); allp.append(pr); print(f'  done in {time.time() - t:.0f}s')
    tag = f'GNN L={a.layers}' if a.layers else 'MLP (no graph)'
    res[tag + ' (mean seeds)'] = {sp: {k: float(np.mean([r[sp][k] for r in runs])) for k in runs[0][sp]}
                                  for sp in ('val', 'test')}
    ens = {}                                                  # average the predictions, then score
    for sp in ('val', 'test'):
        ids = np.where(split == sp)[0]
        p = np.mean([pr[sp][0] for pr in allp], 0); q = np.mean([pr[sp][1] for pr in allp], 0)
        ens[sp] = metrics(p, Y[ids], M[ids], score=q)
    res[tag + ' (ensemble)'] = ens
    show('Results (graded cells only; change = target - last input, ft)', res)
    print('\nGNN test MAE per seed:', [round(float(r['test']['MAE']), 3) for r in runs])

    # per-year breakdown on val/test: is the gap one bad season?
    print('\nMAE by target year (ensemble vs persistence)  [n_samples]')
    for sp in ('val', 'test'):
        ids = np.where(split == sp)[0]; p = np.mean([pr[sp][0] for pr in allp], 0)
        for yr in sorted(set(years[ids])):
            k = years[ids] == yr; mk = M[ids][k]
            print(f'  {yr} [{k.sum():3d}]  model {np.abs(p[k] - Y[ids][k])[mk].mean():.3f}   '
                  f'persistence {np.abs(Y[ids][k])[mk].mean():.3f}   mean true change {Y[ids][k][mk].mean():+.3f}')