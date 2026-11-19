"""Small epsilon-prediction UNet for 32x32 sketches, plus the cosine VP schedule and a DDIM sampler.

Built only from ops that port directly to TF.js (Section 3 runs it in the browser):
conv2d 3x3 / 1x1, GroupNorm, SiLU, dense, nearest 2x upsampling, channel concat. No attention.

    32x32: stem -> RB(c0) RB(c0) ------------------------------ concat -> RB(c0) -> out
    16x16:   down -> RB(c1) RB(c1) ---------------- concat -> RB(c1) -> up
     8x8:      down -> RB(c2) RB(c2) -> mid RB RB -> concat -> RB(c2) -> up

Schedule (continuous t in [0, 1]): alpha_bar(t) = cos^2(((t + s) / (1 + s)) * pi / 2) / cos^2((s / (1 + s)) * pi / 2),
s = 0.008, clipped to [1e-5, 1 - 1e-5]. x_t = sqrt(alpha_bar) x0 + sqrt(1 - alpha_bar) eps. The network sees 1000 t.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

S_COS = 0.008


def alpha_bar(t):
    f = lambda u: torch.cos(((u + S_COS) / (1 + S_COS)) * math.pi / 2) ** 2  # noqa: E731
    return (f(t) / f(torch.zeros_like(t))).clamp(1e-5, 1 - 1e-5)


def alpha_sigma(t):
    ab = alpha_bar(t)
    return ab.sqrt(), (1 - ab).sqrt()


def timestep_embedding(t, dim=128):
    """t in [0, 1] -> sinusoidal features of 1000 t."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = 1000 * t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout=0.1):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(temb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class SketchUNet(nn.Module):
    def __init__(self, ch=(32, 64, 128), tdim=128, dropout=0.1):
        super().__init__()
        c0, c1, c2 = ch
        self.config = {"ch": list(ch), "tdim": tdim, "dropout": dropout}
        self.temb = nn.Sequential(nn.Linear(128, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.stem = nn.Conv2d(1, c0, 3, padding=1)
        rb = lambda a, b: ResBlock(a, b, tdim, dropout)  # noqa: E731
        self.d0 = nn.ModuleList([rb(c0, c0), rb(c0, c0)])
        self.down0 = nn.Conv2d(c0, c0, 3, stride=2, padding=1)
        self.d1 = nn.ModuleList([rb(c0, c1), rb(c1, c1)])
        self.down1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)
        self.d2 = nn.ModuleList([rb(c1, c2), rb(c2, c2)])
        self.mid = nn.ModuleList([rb(c2, c2), rb(c2, c2)])
        self.u2 = rb(2 * c2, c2)
        self.up2 = nn.Conv2d(c2, c2, 3, padding=1)
        self.u1 = rb(c2 + c1, c1)
        self.up1 = nn.Conv2d(c1, c1, 3, padding=1)
        self.u0 = rb(c1 + c0, c0)
        self.out_norm, self.out = nn.GroupNorm(8, c0), nn.Conv2d(c0, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        """x: (B, 1, 32, 32) in [-1, 1] space; t: (B,) in [0, 1]. Returns predicted eps."""
        temb = self.temb(timestep_embedding(t))
        h = self.stem(x)
        for b in self.d0:
            h = b(h, temb)
        s0 = h
        h = self.down0(h)
        for b in self.d1:
            h = b(h, temb)
        s1 = h
        h = self.down1(h)
        for b in self.d2:
            h = b(h, temb)
        s2 = h
        for b in self.mid:
            h = b(h, temb)
        h = self.u2(torch.cat([h, s2], 1), temb)
        h = self.up2(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.u1(torch.cat([h, s1], 1), temb)
        h = self.up1(F.interpolate(h, scale_factor=2, mode="nearest"))
        h = self.u0(torch.cat([h, s0], 1), temb)
        return self.out(F.silu(self.out_norm(h)))


@torch.no_grad()
def ddim_sample(model, n, steps=50, device="cuda", generator=None, clip_x0=True):
    """Deterministic DDIM from t = 1 to t = 0. Returns x0 in [-1, 1], shape (n, 1, 32, 32)."""
    x = torch.randn(n, 1, 32, 32, device=device, generator=generator)
    ts = torch.linspace(1, 0, steps + 1, device=device)
    for i in range(steps):
        t, tp = ts[i].expand(n), ts[i + 1].expand(n)
        a, s = alpha_sigma(t)
        ap, sp = alpha_sigma(tp)
        a, s, ap, sp = (v[:, None, None, None] for v in (a, s, ap, sp))
        eps = model(x, t)
        x0 = (x - s * eps) / a
        if clip_x0:
            x0 = x0.clamp(-1, 1)
            eps = (x - a * x0) / s
        x = ap * x0 + sp * eps if i < steps - 1 else x0
    return x
