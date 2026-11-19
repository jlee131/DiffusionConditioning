"""Small epsilon-prediction UNet for sketches (32x32 or 64x64), plus the cosine VP schedule and a DDIM sampler.

Built only from ops that port directly to TF.js (Section 3 runs it in the browser):
conv2d 3x3 / 1x1, GroupNorm, SiLU, dense, nearest 2x upsampling, channel concat. No attention.

With L = len(ch) levels (3 for 32x32, 4 for 64x64), e.g. L = 3:
    32x32: stem -> RB(c0) RB(c0) ------------------------------ concat -> RB(c0) -> out
    16x16:   down -> RB(c1) RB(c1) ---------------- concat -> RB(c1) -> up
     8x8:      down -> RB(c2) RB(c2) -> mid RB RB -> concat -> RB(c2) -> up
Parameter names: d{i}.{0,1}, down{i} (i < L-1), mid.{0,1}, u{i}, up{i} (i > 0), so 3-level checkpoints keep
their names.

Schedule (continuous t in [0, 1]): alpha_bar(t) = cos^2(((t + s) / (1 + s)) * pi / 2) / cos^2((s / (1 + s)) * pi / 2),
s = 0.008, clipped to [1e-5, 1 - 1e-5]. x_t = sqrt(alpha_bar) x0 + sqrt(1 - alpha_bar) eps. The network sees 1000 t.
Resolution shift (Hoogeboom et al., simple diffusion): for size > 32 the log-SNR is shifted by 2 log(32 / size),
i.e. logit(alpha_bar) -> logit(alpha_bar) + 2 log(32 / size), so a 64x64 image gets the same effective noise level
per unit of structure as 32x32. size = 32 leaves the schedule unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

S_COS = 0.008


def alpha_bar(t, size=32):
    f = lambda u: torch.cos(((u + S_COS) / (1 + S_COS)) * math.pi / 2) ** 2  # noqa: E731
    ab = (f(t) / f(torch.zeros_like(t))).clamp(1e-5, 1 - 1e-5)
    if size != 32:
        ab = torch.sigmoid(torch.logit(ab) + 2 * math.log(32 / size))
    return ab.clamp(1e-5, 1 - 1e-5)


def alpha_sigma(t, size=32):
    ab = alpha_bar(t, size)
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
    def __init__(self, ch=(32, 64, 128), tdim=128, dropout=0.1, size=32):
        super().__init__()
        ch = tuple(ch)
        L = self.levels = len(ch)
        self.config = {"ch": list(ch), "tdim": tdim, "dropout": dropout, "size": size}
        self.temb = nn.Sequential(nn.Linear(128, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.stem = nn.Conv2d(1, ch[0], 3, padding=1)
        rb = lambda a, b: ResBlock(a, b, tdim, dropout)  # noqa: E731
        for i in range(L):
            cin = ch[i - 1] if i > 0 else ch[0]
            self.add_module(f"d{i}", nn.ModuleList([rb(cin, ch[i]), rb(ch[i], ch[i])]))
            if i < L - 1:
                self.add_module(f"down{i}", nn.Conv2d(ch[i], ch[i], 3, stride=2, padding=1))
        self.mid = nn.ModuleList([rb(ch[-1], ch[-1]), rb(ch[-1], ch[-1])])
        for i in reversed(range(L)):
            below = ch[i + 1] if i < L - 1 else ch[-1]       # channels arriving from the coarser level
            self.add_module(f"u{i}", rb(below + ch[i], ch[i]))
            if i > 0:
                self.add_module(f"up{i}", nn.Conv2d(ch[i], ch[i], 3, padding=1))
        self.out_norm, self.out = nn.GroupNorm(8, ch[0]), nn.Conv2d(ch[0], 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        """x: (B, 1, size, size) in [-1, 1] space; t: (B,) in [0, 1]. Returns predicted eps."""
        temb = self.temb(timestep_embedding(t))
        h = self.stem(x)
        skips = []
        for i in range(self.levels):
            for b in getattr(self, f"d{i}"):
                h = b(h, temb)
            skips.append(h)
            if i < self.levels - 1:
                h = getattr(self, f"down{i}")(h)
        for b in self.mid:
            h = b(h, temb)
        for i in reversed(range(self.levels)):
            h = getattr(self, f"u{i}")(torch.cat([h, skips[i]], 1), temb)
            if i > 0:
                h = getattr(self, f"up{i}")(F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.out(F.silu(self.out_norm(h)))


def load_sketch_model(path, device="cuda", ema=True):
    """Build a SketchUNet from a checkpoint's config and load its (EMA) weights."""
    ck = torch.load(path, map_location=device, weights_only=True)
    cfg = dict(ck["config"])
    cfg["ch"] = tuple(cfg["ch"])
    model = SketchUNet(**cfg).to(device)
    model.load_state_dict(ck["ema" if ema else "model"])
    return model.eval().requires_grad_(False), ck


@torch.no_grad()
def ddim_sample(model, n, steps=50, device="cuda", generator=None, clip_x0=True):
    """Deterministic DDIM from t = 1 to t = 0. Returns x0 in [-1, 1], shape (n, 1, size, size)."""
    size = model.config.get("size", 32)
    x = torch.randn(n, 1, size, size, device=device, generator=generator)
    ts = torch.linspace(1, 0, steps + 1, device=device)
    for i in range(steps):
        t, tp = ts[i].expand(n), ts[i + 1].expand(n)
        a, s = alpha_sigma(t, size)
        ap, sp = alpha_sigma(tp, size)
        a, s, ap, sp = (v[:, None, None, None] for v in (a, s, ap, sp))
        eps = model(x, t)
        x0 = (x - s * eps) / a
        if clip_x0:
            x0 = x0.clamp(-1, 1)
            eps = (x - a * x0) / s
        x = ap * x0 + sp * eps if i < steps - 1 else x0
    return x
