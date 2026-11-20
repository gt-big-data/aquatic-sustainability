import torch
import torch.nn as nn
import torch.nn.functional as F

from deep import Block, run, zero_head


class ConvGRUCell(nn.Module):
    def __init__(self, cin, hid):
        super().__init__()
        self.hid = hid
        self.gates = nn.Conv2d(cin + hid, 2 * hid, 3, padding=1)
        self.cand = nn.Conv2d(cin + hid, hid, 3, padding=1)

    def forward(self, x, h):
        z, r = torch.sigmoid(self.gates(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.cand(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class ConvGRU(nn.Module):
    def __init__(self, c_seq, c_stat, hid=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(c_seq + c_stat, hid, 3, stride=2, padding=1), nn.GELU())
        self.cell1 = ConvGRUCell(hid, hid)
        self.cell2 = ConvGRUCell(hid, hid)
        self.dec = Block(hid + c_stat + c_seq, hid)
        self.head = zero_head(hid)

    def forward(self, seq, stat):
        B, T, _, H, W = seq.shape
        h1 = seq.new_zeros(B, self.cell1.hid, H // 2, W // 2)
        h2 = torch.zeros_like(h1)
        for t in range(T):
            e = self.enc(torch.cat([seq[:, t], stat], 1))
            h1 = self.cell1(e, h1)
            h2 = self.cell2(h1, h2)
        up = F.interpolate(h2, scale_factor=2, mode='bilinear', align_corners=False)
        return self.head(self.dec(torch.cat([up, stat, seq[:, -1]], 1)))[:, 0]


run('m07_convgru', lambda T: ConvGRU(T.c_seq, T.c_stat), 'seq')
