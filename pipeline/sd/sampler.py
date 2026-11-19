"""Deterministic DDIM (eta = 0) with style guidance, recording every step.

Unified guidance form (see .documents/PLAN.md):
    eps_tilde = eps_base - s * sigma_t * g_t,   g_t ~ grad log p_t(c | z_t)

    base : eps_base = eps_null + w_cat * (eps_cat - eps_null)       (cat baseline)
    CFG  : -sigma_t g_t = d_cfg = w_cat * (eps_pencil - eps_cat)     -> s = 1 is the pencil-cat prompt
    CG   : -sigma_t g_t = -w_cat * sigma_t * grad_z logit_phi(z_t, t), logit = log p(pencil|z_t) - log p(photo|z_t);
           no calibration. The w_cat factor matches the CFG target p(z) p(pencil|z)^w_cat, so an oracle logit
           gives exactly the CFG direction at the same s
    TFG  : g_t = -grad_z loss(Decode(x0_hat_base(z_t))), backprop through x0_hat (and the UNet)
"""
from dataclasses import dataclass, field

import torch

from .models import DEVICE, DTYPE


@dataclass
class StyleSetup:
    neg_prompt: str = ""
    base_prompt: str = "a photo of a cat"
    style_prompt: str = "a pencil drawing of a cat"
    w_cat: float = 5.0
    n_steps: int = 30


@dataclass
class RunSpec:
    method: str = "none"          # none | cfg | cg | tfg
    scale: float = 0.0
    start_step: int = 0           # guidance active for step index >= start_step
    record_every: int = 1


@dataclass
class RunResult:
    final: torch.Tensor = None                      # decoded final image [-1, 1], 1x3x512x512
    x0_base: list = field(default_factory=list)     # decoded base clean predictions (256 px)
    x0_guided: list = field(default_factory=list)   # decoded guided clean predictions (256 px)
    zt_preview: list = field(default_factory=list)  # latent preview of z_t (64 px)
    steps: list = field(default_factory=list)       # recorded step indices
    metrics: dict = field(default_factory=dict)


def _rms(x):
    return x.float().pow(2).mean().sqrt().item()


def _cos(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


class GuidedSampler:
    def __init__(self, sd, setup: StyleSetup, scorer=None, tfg_lambda=1.0, task_loss=None, classifier=None):
        """scorer: CLIP style scorer used for metrics; task_loss(x0_latent) -> scalar for TFG
        (defaults to the CLIP loss on the TAESD-decoded clean prediction);
        classifier: NoisyClassifier for CG (if given, its direction is also logged in every run)."""
        self.sd, self.setup, self.scorer = sd, setup, scorer
        self.classifier = classifier
        self.task_loss = task_loss or (lambda x0: scorer.loss(scorer.decode_tiny(x0)))
        self.final_latent = None
        self.emb = sd.encode_text([setup.neg_prompt, setup.base_prompt, setup.style_prompt])
        self.tfg_lambda = tfg_lambda  # calibration: maps TFG scale onto CFG-comparable magnitude

    def initial_noise(self, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(1, 4, 64, 64, generator=g).to(DEVICE, DTYPE)

    def _eps3(self, z, t):
        e = self.sd.eps(z.expand(3, -1, -1, -1), t, self.emb)
        return e[0:1], e[1:2], e[2:3]

    def _tfg_grad(self, z, t, alpha, sigma):
        """Returns (eps_null, eps_cat, grad_z loss, loss) with the gradient taken through x0_hat(z)."""
        w = self.setup.w_cat
        z32 = z.detach().float().requires_grad_(True)
        with torch.enable_grad():
            zz = z32.to(DTYPE)
            e = self.sd.eps(zz.expand(2, -1, -1, -1), t, self.emb[:2])
            eps_base = e[0:1] + w * (e[1:2] - e[0:1])
            x0 = (zz - sigma * eps_base) / alpha
            loss = self.task_loss(x0).float()
            (grad,) = torch.autograd.grad(loss, z32)
        return e[0:1].detach(), e[1:2].detach(), grad.to(DTYPE), loss.item()

    def _cg_grad(self, z, t):
        """grad_z of the classifier logit log p(pencil|z_t) - log p(photo|z_t)."""
        z32 = z.detach().float().requires_grad_(True)
        with torch.enable_grad():
            logit = self.classifier(z32, t).sum()
            (grad,) = torch.autograd.grad(logit, z32)
        return grad.to(DTYPE), logit.item()

    @torch.no_grad()
    def run(self, seed, spec: RunSpec, probe_tfg=False):
        sd, w = self.sd, self.setup.w_cat
        z = self.initial_noise(seed)
        res = RunResult()
        m = {k: [] for k in ["t", "guid_rms", "cfg_rms", "tfg_rms", "cos_cfg_tfg", "loss", "cg_rms", "cos_cfg_cg",
                             "cg_logit", "clip_x0_base", "clip_x0_guided"]}
        sched = sd.timesteps(self.setup.n_steps)
        for i, (t, t_prev) in enumerate(sched):
            a, s = sd.alpha_sigma(t)
            ap, sp = sd.alpha_sigma(t_prev)
            active = spec.method != "none" and i >= spec.start_step and spec.scale != 0
            need_grad = (spec.method == "tfg" and active) or probe_tfg

            tfg_dir, loss = None, float("nan")
            if need_grad:
                e_null, e_cat, grad, loss = self._tfg_grad(z, t, a, s)
                tfg_dir = self.tfg_lambda * s * grad  # = -sigma * lambda * g_t, in eps units
                e_pc = sd.eps(z, t, self.emb[2:3])
            else:
                e_null, e_cat, e_pc = self._eps3(z, t)
            eps_base = e_null + w * (e_cat - e_null)
            d_cfg = w * (e_pc - e_cat)

            cg_dir, logit = None, float("nan")
            if self.classifier is not None:
                grad, logit = self._cg_grad(z, t)
                # oracle: -sigma * grad logit = eps_pencil - eps_photo, so w_cat * (.) equals d_cfg
                cg_dir = -w * s * grad

            eps_g = eps_base
            if active and spec.method == "cfg":
                eps_g = eps_base + spec.scale * d_cfg
            elif active and spec.method == "cg":
                eps_g = eps_base + spec.scale * cg_dir
            elif active and spec.method == "tfg":
                eps_g = eps_base + spec.scale * tfg_dir

            x0_base = (z - s * eps_base) / a
            x0_g = (z - s * eps_g) / a

            m["t"].append(t)
            m["guid_rms"].append(_rms(eps_g - eps_base))
            m["cfg_rms"].append(_rms(d_cfg))
            m["tfg_rms"].append(_rms(tfg_dir) if tfg_dir is not None else float("nan"))
            m["cos_cfg_tfg"].append(_cos(d_cfg, tfg_dir) if tfg_dir is not None else float("nan"))
            m["loss"].append(loss)
            m["cg_rms"].append(_rms(cg_dir) if cg_dir is not None else float("nan"))
            m["cos_cfg_cg"].append(_cos(d_cfg, cg_dir) if cg_dir is not None else float("nan"))
            m["cg_logit"].append(logit)

            if i % spec.record_every == 0 or i == len(sched) - 1:
                imgs = sd.decode(torch.cat([x0_base, x0_g]))
                small = torch.nn.functional.interpolate(imgs, size=(256, 256), mode="bilinear", antialias=True)
                res.x0_base.append(small[0:1].cpu())
                res.x0_guided.append(small[1:2].cpu())
                res.zt_preview.append(sd.latent_preview(z))
                res.steps.append(i)
                if self.scorer is not None:
                    p = self.scorer.prob(imgs)
                    m["clip_x0_base"].append(p[0].item())
                    m["clip_x0_guided"].append(p[1].item())

            z = ap * x0_g + sp * eps_g

        self.final_latent = z
        res.final = sd.decode(z).cpu()
        if self.scorer is not None:
            m["clip_final"] = self.scorer.prob(res.final.to(DEVICE)).item()
        res.metrics = m
        return res
