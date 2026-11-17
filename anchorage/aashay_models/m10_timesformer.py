import torch
import torch.nn as nn
import torch.nn.functional as F

from deep import Block, run, zero_head


class DividedBlock(nn.Module):
    def __init__(self, d, heads, drop=0.1):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.time = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.space = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, N, D = x.shape
        t = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        q = self.n1(t)
        t = t + self.time(q, q, q, need_weights=False)[0]
        x = t.reshape(B, N, T, D).permute(0, 2, 1, 3)
        s = x.reshape(B * T, N, D)
        q = self.n2(s)
        s = s + self.space(q, q, q, need_weights=False)[0]
        s = s + self.mlp(self.n3(s))
        return s.reshape(B, T, N, D)


class TimeSformer(nn.Module):
    def __init__(self, c_seq, c_stat, H=176, W=96, p=8, d=64, depth=3, heads=4, steps=4):
        super().__init__()
        self.p, self.gh, self.gw = p, H // p, W // p
        n = self.gh * self.gw
        self.embed = nn.Conv2d(c_seq, d, p, stride=p)
        self.embed_stat = nn.Conv2d(c_stat, d, p, stride=p)
        self.pos = nn.Parameter(torch.zeros(1, 1, n, d))
        self.tpos = nn.Parameter(torch.zeros(1, steps, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.tpos, std=0.02)
        self.blocks = nn.ModuleList([DividedBlock(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.unpatch = nn.Linear(d, 16 * p * p)
        self.refine = Block(16 + c_stat + c_seq, 32)
        self.head = zero_head(32)

    def forward(self, seq, stat):
        B, T, C, H, W = seq.shape
        tok = self.embed(seq.flatten(0, 1)).flatten(2).transpose(1, 2).reshape(B, T, -1, self.pos.shape[-1])
        st = self.embed_stat(stat).flatten(2).transpose(1, 2)[:, None]
        x = tok + st + self.pos + self.tpos
        for blk in self.blocks:
            x = blk(x)
        z = self.unpatch(self.norm(x[:, -1]))
        z = z.reshape(B, self.gh, self.gw, 16, self.p, self.p).permute(0, 3, 1, 4, 2, 5).reshape(B, 16, H, W)
        return self.head(self.refine(torch.cat([z, stat, seq[:, -1]], 1)))[:, 0]


run('m10_timesformer', lambda T: TimeSformer(T.c_seq, T.c_stat), 'seq')
