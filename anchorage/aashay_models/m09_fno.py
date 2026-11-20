import torch
import torch.nn as nn
import torch.nn.functional as F

from deep import run, zero_head


class Spectral(nn.Module):
    def __init__(self, c, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        s = 1 / (c * c)
        self.w1 = nn.Parameter(s * torch.randn(c, c, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(s * torch.randn(c, c, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        f = torch.fft.rfft2(x)
        out = torch.zeros(B, C, H, W // 2 + 1, dtype=torch.cfloat)
        out[:, :, :self.m1, :self.m2] = torch.einsum('bixy,ioxy->boxy', f[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = torch.einsum('bixy,ioxy->boxy', f[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(out, s=(H, W))


class FNO(nn.Module):
    def __init__(self, cin, width=32, modes=(20, 12), layers=4):
        super().__init__()
        self.lift = nn.Conv2d(cin, width, 1)
        self.spec = nn.ModuleList([Spectral(width, *modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in range(layers)])
        self.proj = nn.Sequential(nn.Conv2d(width, 64, 1), nn.GELU())
        self.head = zero_head(64)

    def forward(self, x):
        h = self.lift(x)
        for s, l in zip(self.spec, self.local):
            h = F.gelu(s(h) + l(h))
        return self.head(self.proj(h))[:, 0]


run('m09_fno', lambda T: FNO(T.c_flat), 'flat')
