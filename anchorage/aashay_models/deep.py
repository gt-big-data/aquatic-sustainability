import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Data, evaluate, nan_box

HP, WP = 176, 96


def pad(a):
    h, w = a.shape[-2:]
    return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, HP - h), (0, WP - w)])


class Tensors:
    def __init__(self, D, depth='local', chunk=200):
        self.D = D
        tr = D.idx['train']
        S = D.S
        h = D.horizon[:, None, None]
        self.cs = float(D.change[tr][D.masks[tr]].std())
        self.clim = (D.rate_harm * h).astype(np.float32)
        if depth == 'local':
            sub = tr[::5]
            sd = float(np.nanstd(D.inputs[sub] - nan_box(D.last[sub], 15)[:, None]))
        else:
            mu, sd = float(np.nanmean(D.inputs[tr])), float(np.nanstd(D.inputs[tr]))
        self.maps = np.zeros((S, 4, 3, HP, WP), np.float16)
        self.stat = np.zeros((S, 8, HP, WP), np.float16)
        for s0 in range(0, S, chunk):
            i = np.arange(s0, min(S, s0 + chunk))
            inp = D.inputs[i]
            ref = nan_box(D.last[i], 15)[:, None] if depth == 'local' else mu
            rel = (inp - ref) / sd
            ok = ~np.isnan(rel)
            self.maps[i, :, 0, :D.H, :D.W] = np.where(ok, rel, 0)
            self.maps[i, :, 1, :D.H, :D.W] = ~np.isnan(inp)
            self.maps[i, :, 2, :D.H, :D.W] = D.dredged[D.windows[i, :-1]]
            st = self.stat[i]
            st[:, 0, :D.H, :D.W] = self.clim[i] / self.cs
            st[:, 1, :D.H, :D.W] = D.rate_season[i] * h[i] / self.cs
            st[:, 2, :D.H, :D.W] = D.core
            st[:, 3, :D.H, :D.W] = D.hot
            for c, v in enumerate([D.doy_sin[i], D.doy_cos[i], D.horizon[i] / 28, D.ahead[i]]):
                st[:, 4 + c, :D.H, :D.W] = v[:, None, None]
            self.stat[i] = st
        self.vec = np.concatenate([((D.in_days - D.tgt_days[:, None]) / 28.0)[:, :, None], D.drv], 2).astype(np.float32)
        self.y = pad(((D.change - np.where(D.masks, self.clim, 0)) / self.cs).astype(np.float32))
        self.m = pad(D.masks)
        self.c_seq, self.c_stat = self.maps.shape[2] + self.vec.shape[2], self.stat.shape[1]

    def seq(self, idx):
        v = np.broadcast_to(self.vec[idx][:, :, :, None, None], (len(idx), 4, self.vec.shape[2], HP, WP))
        return np.concatenate([self.maps[idx].astype(np.float32), v], 2)

    def flat(self, idx):
        s = self.seq(idx)
        return np.concatenate([s.reshape(len(idx), -1, HP, WP), self.stat[idx].astype(np.float32)], 1)

    @property
    def c_flat(self):
        return 4 * self.c_seq + self.c_stat


def batch(T, idx, mode):
    if mode == 'flat':
        return (torch.from_numpy(T.flat(idx)),)
    return torch.from_numpy(T.seq(idx)), torch.from_numpy(T.stat[idx].astype(np.float32))


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


def run(label, build, mode, steps=1500, eval_every=100, lr=5e-4, bs=8, seed=0, wd=0.05, ema=0.995, augment=True,
        depth='local', extra=None):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    D = Data(augment=augment)
    T = Tensors(D, depth)
    model = build(T)
    avg = copy.deepcopy(model)
    n_par = sum(p.numel() for p in model.parameters())
    print(label, f'{n_par:,} params, {len(D.idx["train"])} train maps')
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    tr, va = D.idx['train'], D.idx['val']
    order, pos, losses, te = np.random.permutation(tr), 0, [], time.time()
    best = (np.inf, None, -1)
    for step in range(1, steps + 1):
        model.train()
        if pos + bs > len(order):
            order, pos = np.random.permutation(tr), 0
        b = order[pos:pos + bs]
        pos += bs
        y, m = torch.from_numpy(T.y[b]), torch.from_numpy(T.m[b]).float()
        loss = (torch.abs(model(*batch(T, b, mode)) - y) * m).sum() / m.sum()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pa, pm in zip(avg.parameters(), model.parameters()):
                pa.mul_(ema).add_(pm, alpha=1 - ema)
        losses.append(loss.item())
        if step % eval_every == 0:
            v = val_mae(T, predict(avg, T, va, mode), va)
            if v < best[0]:
                best = (v, copy.deepcopy(avg.state_dict()), step)
            print(f'step {step:5d} loss {np.mean(losses):.4f} val {v:.4f}{" *" if best[2] == step else ""} '
                  f'({time.time() - te:.0f}s)', flush=True)
            losses, te = [], time.time()
    avg.load_state_dict(best[1])
    preds = {k: predict(avg, T, D.idx[k], mode) for k in ['val', 'test']}
    info = {'params': n_par, 'best_step': best[2], 'steps': steps, 'seed': seed, 'lr': lr, 'wd': wd, 'ema': ema,
            'augment': augment, 'depth': depth, 'train_maps': int(len(tr))}
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
