# Implementation Plan

Companion to [GOAL.md](GOAL.md). Deadline: **2026-12-09**. Hardware: RTX 3060 (12 GB), conda env `diffcon` (Python 3.11, torch 2.5.1+cu121, diffusers 0.35.2).

Working rule: long-running jobs (data generation, training, precompute) are handed to the user as commands and run by the user in the background. Claude writes code and runs only short checks.

## Repository Layout

```
.documents/        GOAL.md, PLAN.md, proposal, announcement
pipeline/          offline Python (never served)
  sd/              SD sampler, guidance, CG classifier, precompute, export
  sketch/          Quick, Draw! prep, sketch-model training, TF.js weight export
site/              static website (deployed to GitHub Pages)
  index.html
  js/              toy2d/, sd/, sketch/, shared/
  assets/sd/       exported sprite sheets + metrics JSON
  assets/sketch/   TF.js weights + manifest
models/            SD 1.5 weights (gitignored)
data/, runs/       datasets, checkpoints, raw precompute output (gitignored)
```

Frontend: plain HTML + ES modules, no build step. Libraries from CDN with pinned versions: KaTeX (equations), TensorFlow.js (Section 3), uPlot or custom canvas for metric plots. Deployment through a GitHub Actions Pages workflow that publishes `site/`.

## Shared Conventions

- Variance-preserving forward process $x_t = \alpha_t x_0 + \sigma_t \varepsilon$, $\alpha_t^2 + \sigma_t^2 = 1$, in all three sections.
- Deterministic DDIM ($\eta = 0$) everywhere, so identical initial noise gives a fully fair comparison.
- Clean prediction (Tweedie): $\hat x_0 = (x_t - \sigma_t \hat\varepsilon)/\alpha_t$.
- Unified guidance form: $\tilde\varepsilon = \varepsilon_\text{base} - s\,\sigma_t\, g_t$, where $g_t$ approximates $\nabla \log p_t(c \mid x_t)$.
  - CFG: $g_t = -(\varepsilon_c - \varepsilon_\text{base})/\sigma_t$, i.e. $\tilde\varepsilon = \varepsilon_\text{base} + s(\varepsilon_c - \varepsilon_\text{base})$.
  - CG: $g_t = \nabla_{x_t} \log p_\phi(c \mid x_t, t)$.
  - TFG: $g_t = -\nabla_{x_t}\, \ell(\hat x_0(x_t))$, gradient taken through $\hat x_0$.
- $s = 0$ is the base model, $s = 1$ the (oracle) conditional, $s > 1$ extrapolation.
- Guidance start time: guidance is applied only from step $k_\text{start}$ onward; earlier steps use the base field.

## Section 1 — Toy 2D (browser only, no training)

**Model.** $p_0 = \sum_k \pi_k \mathcal N(\mu_k, s_k^2 I)$ in 2D, each component with a label probability $p(c \mid k)$. Then $p_t$ is a GMM with means $\alpha_t \mu_k$ and variances $\alpha_t^2 s_k^2 + \sigma_t^2$, so everything is closed form:

- Score $\nabla \log p_t(x)$ from posterior responsibilities $r_k(x)$.
- Classifier $p_t(c \mid x) = \sum_k r_k(x)\, p(c \mid k)$ and its gradient (oracle CG).
- Conditional density $p_t(x \mid c)$ as a reweighted GMM (oracle CFG). The two fields agree identically; the UI also plots their difference (zero).
- Tweedie $\hat x_0(x)$ as a responsibility-weighted posterior mean, with Jacobian $\partial \hat x_0 / \partial x_t = (\alpha_t/\sigma_t^2)\,\mathrm{Cov}[x_0 \mid x_t]$ (analytic), used for TFG.
- "Estimated" mode: CG uses a mis-specified classifier (temperature-smoothed, or a linear/logistic one); CFG uses a conditional model with perturbed means and weights. An error slider controls how far they deviate, and the difference heatmap lights up.
- TFG loss: $\ell(\hat x_0) = \lVert \hat x_0 - x^\star \rVert^2$ (or a soft region) for a user-placed target $x^\star$.

**Views.** One main canvas: background heatmap (density $p_t$, guided target density, classifier $p_t(c\mid x)$, or the CG−CFG difference); arrow layers (unconditional score, guidance, total) on a grid; 400 particles from shared noise flowing along DDIM, colored by final class; $\hat x_0$ rings (every 3rd particle) linked to their particles; focusing a particle (hover, or click to pin; Shift+click in TFG mode) dims the others and shows its $x_t$ trail up to the current step (fading with age, in its class color), its current $\hat x_0$ ring labeled "estimate", and a ★ at its actual endpoint, so the gap between estimate and endpoint is visible. The $\hat x_0$ path (dashed) is off by default, behind a checkbox. Controls: time slider + play, method, scale $s$, start time, oracle/estimated toggle + error slider, layer toggles. Side panel: KaTeX equations with live values at the hovered point; a small plot of the fraction of particles ending in the pencil class vs $s$.

**Implementation (done, v1).** Vanilla JS + Canvas 2D: math in `site/js/toy2d/gmm.js`, UI in `toy2d.js`. Checks, run with Node from conda env `webdev`: oracle CG and CFG agree to about 1e-15 relative to the field scale; the score and the $\hat x_0$ Jacobian match finite differences to about 1e-9; at $s=1$, CG and CFG give the same pencil fraction. URL parameters (`?method=cfg&s=2&frame=50&model=estimated&kappa=0.6&bg=classifier`) reproduce a given state. Local preview: `python -m http.server 8765 -d site`.

## Section 2 — Stable Diffusion (offline precompute)

**Fixed settings (confirmed in the pilot).** SD 1.5 fp16, 512×512, DDIM 30 steps, $\eta = 0$.

- Base field: $\varepsilon_\text{base} = \varepsilon_\varnothing + w_\text{cat}(\varepsilon_\text{cat} - \varepsilon_\varnothing)$ with a fixed $w_\text{cat}$ (around 5). Raw $w = 1$ samples are poor, so the base is itself CFG toward "cat"; the README states this.
- Prompts: "a photo of a cat" (base) vs "a pencil drawing of a cat" (condition), with exact wording finalized in the pilot.
- CFG direction: $\varepsilon_\text{pencil-cat} - \varepsilon_\text{cat}$, both from the same guided-base construction.

**CG classifier.**
1. Dataset (`pipeline/sd/make_cg_dataset.py`): 1000 final latents per class, DDIM 20 steps. Class "photo" uses the base field (cat prompt, $w_\text{cat}$); class "pencil" uses the pencil prompt with the same $w_\text{cat}$, which is exactly the CFG $s=1$ target. The prompts match the experiment exactly. Only $z_0$ is stored (4×64×64 fp16).
2. Training (`pipeline/sd/train_cg_classifier.py`, model in `cg_classifier.py`): a time-conditioned ResNet (about 6.4M params, GroupNorm + SiLU, no max-pooling so input gradients stay smooth) on $z_t = \alpha_t z_0 + \sigma_t \varepsilon$ with $t \sim U\{1..999\}$ and horizontal-flip augmentation, using binary cross-entropy with equal class priors. Report validation accuracy vs $t$.
3. Use the **logit** gradient: $g_t = \nabla_z [\log p_\phi(\text{pencil} \mid z_t) - \log p_\phi(\text{photo} \mid z_t)]$. With equal priors, an oracle logit equals $\log p_t(z \mid \text{pencil}) - \log p_t(z \mid \text{photo})$, so its gradient is exactly the CFG direction $-d_\text{cfg}/\sigma_t$. Oracle CG and CFG then coincide at the same $s$, and any measured gap is estimation error. For this reason CG uses no scale calibration (unlike TFG).
4. Check (`pipeline/sd/eval_cg.py`): CFG vs CG finals at equal $s$, and per-step cos(CFG direction, CG direction) and RMS ratio.

**TFG loss (decided in the cold test, see RESULTS.md).** VGG16 Gram-matrix style loss (relu1_2–relu4_3) between the TAESD-decoded clean prediction and a reference pencil drawing (the CFG $s=1$ sample of held-out seed 7). Gradients flow through the decoder, $\hat x_0$, and the UNet back to $z_t$. CLIP-based losses were tried and rejected because they were adversarial. The TFG scale is calibrated so that $s=1$ matches the median CFG guidance RMS; the usable range is $s\in[1,2]$. The result is a grayscale pencil *tone* rather than line strokes, which is accepted.

**Recorded per step** (custom DDIM loop in `pipeline/sd/sampler.py`):
- $z_t$ preview (cheap latent→RGB linear projection, 64 px)
- $\hat x_0$ from the base denoiser and from the guided $\tilde\varepsilon$ (full VAE decode → 256 px)
- $\lVert g_t \rVert$; cos(CG direction, CFG direction), computed in every run since both are cheap
- CLIP pencil score of $\hat x_0$; L2 distance to the unguided run's $\hat x_0$
- final image (512 px)

**Precompute grid** (adjust after the pilot): seeds × {none} ∪ {CG, CFG, TFG} × scales × start times. Initial target: 4–6 seeds, 3–4 scales, 3 start times (steps 0, 10, 20), about 110–220 runs.

**Export.** `pipeline/sd/export.py` packs each run into WebP sprite sheets (one per image stream) plus one metrics JSON per seed, and writes an `index.json` describing the grid.

**Frontend.** Seed picker; method checkboxes (side-by-side columns); scale and start-time sliders that snap to the grid; one shared time slider + play. Each column shows guided $\hat x_0$ (large), base $\hat x_0$ and $z_t$ preview (small), all clearly labeled. Metric plots below share the time axis, and hovering one sets the slider.

## Section 3 — Draw an Outline (TF.js, live)

**Data.** Quick, Draw! "cat" simplified drawings (CC BY 4.0, about 120k), rasterized to 32×32 grayscale with antialiased strokes, stored as a uint8 `.npy`.

**Model.** Small ε-prediction UNet (about 2–4M params): conv, GroupNorm, SiLU, sinusoidal time embedding, 2 down/up levels, optional one attention block at 8×8. Cosine VP schedule. Trained in PyTorch (a few hours in the background on the 3060).

**Port to TF.js.** Re-implement the same architecture with TF.js ops (GroupNorm written manually) and load weights from an exported binary + JSON manifest. Parity test: export input/output test vectors from PyTorch and require max abs error below 1e-3 in the browser before building any UI.

**Guided sampling.** DDIM, user-selectable 20–100 steps. Outline loss on $\hat x_0$ against the user's rasterized strokes $M$: blurred MSE between ink maps, $\ell = \lVert G * \mathrm{ink}(\hat x_0) - G * M \rVert^2$ (Gaussian blur $G$ gives a tolerance band; optionally multi-scale). The gradient is taken with `tf.grad` through the UNet and $\hat x_0$. Batch of 4 seeds generated together.

**Views.** Drawing canvas with clear/undo/preset outlines; generation panel animating $x_t$ and $\hat x_0$ (upscaled) per step; gradient heatmap overlay (add ink / remove ink); per-step loss curve; guided vs unguided from the same seed. Controls: strength, start time, steps, seed.

## Step Order

| # | Dates | Step | Claude | User |
|---|---|---|---|---|
| 0 | 11/18–19 | Repo cleanup, env check, skeleton site, Pages workflow | all | review |
| 1 | 11/19–22 | SD pilot: custom DDIM, CFG + TFG, small grid | code | run pilot (~15 min), pick prompts / $w_\text{cat}$ / scale ranges |
| 1b | 11/19–22 | Quick, Draw! prep + sketch model training | code | run prep + training (background, a few hours) |
| 2 | 11/22–25 | Section 1 toy module | all | try it, give feedback |
| 3 | 11/19–22 | CG dataset + classifier | code | run generation (~1.2 h) and training (~30 min) |
| 4 | 11/27–30 | Full SD precompute + export; Section 2 frontend | code | run precompute (~1.5–2 h) and export |
| 5 | 11/30–12/4 | TF.js port + parity test; Section 3 UI | all | try it, give feedback |
| 6 | 12/4–12/7 | Explanations and equations, README, deploy | draft | review observations, check the deployed site |
| — | 12/8–12/9 | Buffer, presentation prep | — | — |

## Risks and Fallbacks

- **CG/TFG gradients on SD look adversarial rather than stylistic.** Tune the scale range, apply guidance from a start time, and normalize the gradient. If results stay weak, present them honestly as estimation error relative to CFG.
- **Out of memory in TFG backprop.** Use the TAESD gradient path, lower-resolution decoding for the loss, and gradient checkpointing.
- **Site asset size.** Shrink the grid (seeds/scales), use 192 px frames, or subsample frames for the base $\hat x_0$ stream. Commit exported assets only at final export to keep git history small.
- **TF.js speed.** Keep 32×32, fewer steps by default, and the WebGL backend; fall back to fewer seeds per batch.
