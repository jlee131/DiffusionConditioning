"""Time-conditioned noisy latent classifier for classifier guidance (CG).

Input: noisy SD latent z_t (1x4x64x64, scaled latent space) and integer timestep t in [0, 999].
Output (forward): one logit = log p(pencil | z_t) - log p(photo | z_t) (equal priors).
    n_classes = 1 : binary head (photo_bw, if used, is folded into photo)
    n_classes = 3 : softmax over CLASSES = (photo, photo_bw, pencil); forward returns f_pencil - f_photo, which for
                    an oracle is exactly log p_t(z|pencil) - log p_t(z|photo) (+ const), so oracle CG = CFG still
                    holds. photo_bw only shapes the shared features (it forces texture, not color, cues).
CG uses grad_z of this logit (see PLAN.md). SiLU everywhere and no max-pooling keep the input gradient smooth.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import DEVICE, ROOT

CKPT = ROOT / "runs" / "cg_classifier" / "classifier.pt"
CLASSES = ("photo", "photo_bw", "pencil")  # index order of the 3-class head


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(temb)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class NoisyClassifier(nn.Module):
    def __init__(self, widths=(64, 128, 256, 256), tdim=256, n_classes=1):
        super().__init__()
        self.tdim, self.n_classes = tdim, n_classes
        self.temb = nn.Sequential(nn.Linear(128, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.stem = nn.Conv2d(4, widths[0], 3, padding=1)
        blocks, downs, c = [], [], widths[0]
        for i, w in enumerate(widths):
            blocks.append(nn.ModuleList([ResBlock(c, w, tdim), ResBlock(w, w, tdim)]))
            c = w
            downs.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(widths) - 1 else nn.Identity())
        self.blocks, self.downs = nn.ModuleList(blocks), nn.ModuleList(downs)
        self.out_norm, self.head = nn.GroupNorm(8, c), nn.Linear(c, n_classes)

    def forward(self, z, t):
        """pencil-vs-photo logit (used by CG)."""
        f = self.logits(z, t)
        if self.n_classes == 1:
            return f.squeeze(-1)
        return f[:, CLASSES.index("pencil")] - f[:, CLASSES.index("photo")]

    def logits(self, z, t):
        """raw head output, shape (B, n_classes)."""
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=z.device)
        t = t.expand(z.shape[0]) if t.numel() == 1 else t
        temb = self.temb(timestep_embedding(t, 128))
        h = self.stem(z.float())
        for (b1, b2), down in zip(self.blocks, self.downs):
            h = down(b2(b1(h, temb), temb))
        h = F.silu(self.out_norm(h)).mean(dim=(2, 3))
        return self.head(h)


def load_classifier(path: Path = CKPT):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = NoisyClassifier(**ck.get("config", {}))
    model.load_state_dict(ck["model"])
    return model.to(DEVICE).eval().requires_grad_(False)
