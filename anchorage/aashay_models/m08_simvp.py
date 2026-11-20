import torch
import torch.nn as nn
import torch.nn.functional as F

from deep import Block, run, zero_head


class Inception(nn.Module):
    def __init__(self, c, hid, groups=4):
        super().__init__()
        self.inp = nn.Conv2d(c, hid, 1)
        self.branches = nn.ModuleList([nn.Sequential(nn.Conv2d(hid, c, k, padding=k // 2, groups=groups),
                                                     nn.LeakyReLU(0.2)) for k in (3, 5)])

    def forward(self, x):
        h = self.inp(x)
        return x + sum(b(h) for b in self.branches)


class SimVP(nn.Module):
    def __init__(self, c_seq, c_stat, ch=16, steps=4, n_trans=3):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(c_seq + c_stat, ch, 3, padding=1), nn.LeakyReLU(0.2))
        self.e2 = nn.Sequential(nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
                                nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.trans = nn.Sequential(*[Inception(steps * ch, 2 * ch) for _ in range(n_trans)])
        self.up = nn.Sequential(nn.ConvTranspose2d(steps * ch, ch, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
                                nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1), nn.LeakyReLU(0.2))
        self.dec = Block(2 * ch, ch)
        self.head = zero_head(ch)

    def forward(self, seq, stat):
        B, T, C, H, W = seq.shape
        x = torch.cat([seq, stat[:, None].expand(-1, T, -1, -1, -1)], 2).flatten(0, 1)
        skip = self.e1(x)
        z = self.e2(skip)
        z = self.trans(z.reshape(B, -1, H // 4, W // 4))
        out = self.up(z)
        skip_last = skip.reshape(B, T, -1, H, W)[:, -1]
        return self.head(self.dec(torch.cat([out, skip_last], 1)))[:, 0]


run('m08_simvp', lambda T: SimVP(T.c_seq, T.c_stat), 'seq')
