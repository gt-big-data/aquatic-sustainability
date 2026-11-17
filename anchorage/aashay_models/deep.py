import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Data, evaluate

HP, WP = 176, 96


def pad(a):
    h, w = a.shape[-2:]
    return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, HP - h), (0, WP - w)])


class Tensors:
    def __init__(self, D):
        self.D = D
        tr = D.idx['train']
        inp = D.inputs
        obs = ~np.isnan(inp)
        mu, sd = np.nanmean(inp[tr]), np.nanstd(inp[tr])
        dep = np.where(obs, (inp - mu) / sd, 0).astype(np.float32)
        dred = D.dredged[D.windows[:, :-1]]
        self.maps = pad(np.stack([dep, obs, dred], 2).astype(np.float32))
        del dep
        self.vec = np.concatenate([((D.in_days - D.tgt_days[:, None]) / 28.0)[:, :, None], D.drv], 2).astype(np.float32)
        h = D.horizon[:, None, None]
        self.cs = float(D.change[tr][D.masks[tr]].std())
        self.clim = (D.rate_harm * h).astype(np.float32)
        ones = np.ones((D.S, D.H, D.W), np.float32)
        stat = [self.clim / self.cs, D.rate_season * h / self.cs, ones * D.core, ones * D.hot,
                ones * D.doy_sin[:, None, None], ones * D.doy_cos[:, None, None], ones * (D.horizon[:, None, None] / 28),
                ones * D.ahead[:, None, None]]
        self.stat = pad(np.stack(stat, 1).astype(np.float32))
        self.y = pad(((D.change - np.where(D.masks, self.clim, 0)) / self.cs).astype(np.float32))
        self.m = pad(D.masks.astype(np.float32))
        self.c_seq, self.c_stat = self.maps.shape[2] + self.vec.shape[2], self.stat.shape[1]

    def seq(self, idx):
        v = np.broadcast_to(self.vec[idx][:, :, :, None, None], (len(idx), 4, self.vec.shape[2], HP, WP))
        return np.concatenate([self.maps[idx], v], 2)

    def flat(self, idx):
        s = self.seq(idx)
        return np.concatenate([s.reshape(len(idx), -1, HP, WP), self.stat[idx]], 1)

    @property
    def c_flat(self):
        return 4 * self.c_seq + self.c_stat


def batch(T, idx, mode):
    if mode == 'flat':
        return (torch.from_numpy(T.flat(idx)),)
    return torch.from_numpy(T.seq(idx)), torch.from_numpy(T.stat[idx])


def predict(model, T, idx, mode, bs=8):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            out.append(model(*batch(T, idx[s:s + bs], mode))[:, :T.D.H, :T.D.W].numpy())
    return np.concatenate(out) * T.cs + T.clim[idx]


def val_mae(T, p, idx):
    D = T.D
    return float(np.abs(p - D.change[idx])[D.masks[idx]].mean())


def run(label, build, mode, epochs=10, lr=5e-4, bs=8, seed=0, crop=None, wd=0.05, loyo=True, trace=False, extra=None):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    D = Data(loyo)
    T = Tensors(D)
    model = build(T)
    n_par = sum(p.numel() for p in model.parameters())
    print(label, f'{n_par:,} params')
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    tr, va = D.idx['train'], D.idx['val']
    steps = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.15)
    v0 = val_mae(T, predict(model, T, va, mode), va)
    print(f'ep  0 val {v0:.4f}')
    best = (np.inf, None, -1)
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(tr)
        losses = []
        te = time.time()
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            xs = batch(T, b, mode)
            y, m = torch.from_numpy(T.y[b]), torch.from_numpy(T.m[b])
            if crop:
                r = np.random.randint(0, T.D.H - crop + 1)
                xs = [x[..., r:r + crop, :] for x in xs]
                y, m = y[..., r:r + crop, :], m[..., r:r + crop, :]
            pred = model(*xs)
            loss = (torch.abs(pred - y) * m).sum() / m.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        v = val_mae(T, predict(model, T, va, mode), va)
        if trace:
            te_ = D.idx['test']
            print(f'   test {val_mae(T, predict(model, T, te_, mode), te_):.4f}', end='')
        if v < best[0]:
            best = (v, copy.deepcopy(model.state_dict()), ep)
        print(f'ep {ep + 1:2d} loss {np.mean(losses):.4f} val {v:.4f}{" *" if best[2] == ep else ""} ({time.time() - te:.0f}s)', flush=True)
    model.load_state_dict(best[1])
    preds = {k: predict(model, T, D.idx[k], mode) for k in ['val', 'test']}
    info = {'params': n_par, 'best_epoch': best[2] + 1, 'epochs': epochs, 'seed': seed, 'crop': crop, 'lr': lr, 'wd': wd, 'loyo_clim': loyo, 'val_mae_untrained': v0}
    info.update(extra or {})
    return evaluate(D, preds, label, info, t0)


class Block(nn.Module):
    def __init__(self, cin, cout, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GELU(),
                                 nn.Dropout2d(drop), nn.Conv2d(cout, cout, 3, padding=1), nn.GELU())
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


def zero_head(cin):
    head = nn.Conv2d(cin, 1, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head
