"""TSLANet-style temporal encoder per player + attention across players (a learned, 'deep' GTW).

Temporal block (after TSLANet, Eldele et al. 2024): an adaptive spectral block (FFT along time, learned complex
filters, a learned threshold that suppresses high-frequency noise) and an interactive convolution block (two
convolution branches that gate each other). Spatial block: attention across the players of one time step, with a
team embedding, so each player's representation is conditioned on who it reacts to. Output: H / A / D per time step.
"""
import torch, torch.nn as nn, torch.nn.functional as Fn


class SpectralBlock(nn.Module):
    def __init__(self, d, L):
        super().__init__()
        nf = L // 2 + 1
        self.w = nn.Parameter(torch.randn(nf, d, 2) * 0.02)
        self.w_hi = nn.Parameter(torch.randn(nf, d, 2) * 0.02)
        self.thr = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):                       # x [M, L, d]
        X = torch.fft.rfft(x, dim=1)
        e = X.abs().pow(2).mean(-1, keepdim=True)                        # energy per frequency
        cut = torch.quantile(e.squeeze(-1), self.thr.clamp(0.05, 0.95).item(), dim=1, keepdim=True)[..., None]
        m = (e > cut).float()                                              # adaptive high-energy mask
        Y = X * torch.view_as_complex(self.w) + X * m * torch.view_as_complex(self.w_hi)
        return torch.fft.irfft(Y, n=x.shape[1], dim=1)


class ConvBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Conv1d(d, d, 3, padding=1); self.b = nn.Conv1d(d, d, 5, padding=2); self.o = nn.Conv1d(d, d, 1)

    def forward(self, x):                       # [M, L, d]
        z = x.transpose(1, 2); a, b = self.a(z), self.b(z)
        return self.o(Fn.gelu(a) * b + Fn.gelu(b) * a).transpose(1, 2)


class TSLA(nn.Module):
    def __init__(self, d, L):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.asb, self.icb = SpectralBlock(d, L), ConvBlock(d)

    def forward(self, x):
        x = x + self.asb(self.n1(x))
        return x + self.icb(self.n2(x))


class PlayerAttn(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.n = nn.LayerNorm(d); self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, mask):                 # x [B*L, N, d], mask [B*L, N] True = present
        h = self.n(x)
        a, _ = self.att(h, h, h, key_padding_mask=~mask, need_weights=False)
        x = x + a
        return x + self.ff(x)


class DeepGTW(nn.Module):
    def __init__(self, C=7, d=48, L=50, depth=2):
        super().__init__()
        self.inp = nn.Linear(C, d)
        self.team = nn.Embedding(2, d)
        self.temporal = nn.ModuleList([TSLA(d, L) for _ in range(depth)])
        self.spatial = nn.ModuleList([PlayerAttn(d) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 3))

    def forward(self, X, mask):                 # X [B, N, L, C], mask [B, N]
        B, N, L, C = X.shape
        is_away = (X[..., 6].sum(-1) < 0).long()                          # [B, N]
        h = self.inp(X) + self.team(is_away)[:, :, None]
        m2 = mask[:, None].expand(B, L, N).reshape(B * L, N).clone()
        m2[:, 0] = m2[:, 0] | ~m2.any(1)                                  # never an all-masked row
        for tb, sb in zip(self.temporal, self.spatial):
            h = tb(h.reshape(B * N, L, -1)).reshape(B, N, L, -1)
            h = sb(h.transpose(1, 2).reshape(B * L, N, -1), m2).reshape(B, L, N, -1).transpose(1, 2)
        w = mask.float()[:, :, None, None]
        home = (w * (1 - is_away.float())[:, :, None, None])
        away = (w * is_away.float()[:, :, None, None])
        pool = lambda wt: (h * wt).sum(1) / wt.sum(1).clamp(min=1)
        z = torch.cat([pool(home), pool(away), pool(w)], -1)              # [B, L, 3d]
        return self.head(z)                                               # [B, L, 3]
