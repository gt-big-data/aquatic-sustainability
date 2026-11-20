import torch
import torch.nn as nn
import torch.nn.functional as F

from deep import Block, run, zero_head


class UNet(nn.Module):
    def __init__(self, cin, base=24):
        super().__init__()
        self.e1 = Block(cin, base, 0.1)
        self.e2 = Block(base, base * 2, 0.1)
        self.e3 = Block(base * 2, base * 4, 0.1)
        self.d2 = Block(base * 6, base * 2)
        self.d1 = Block(base * 3, base)
        self.head = zero_head(base)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        b = self.d2(torch.cat([F.interpolate(c, scale_factor=2), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, scale_factor=2), a], 1))
        return self.head(a)[:, 0]


run('m06_unet', lambda T: UNet(T.c_flat), 'flat')
