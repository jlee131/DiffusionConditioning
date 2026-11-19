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
  css/             main.css (v1 design), theme-sky.css (default design, layered on top)
  js/              toy2d/, sd/, sketch/
  assets/sd/       exported sprite sheets + metrics JSON
  assets/sketch/   TF.js weights + manifest + parity vectors
  tests/           sketch_parity.html, tfjs_grad_ops.html (browser checks)
results/           small figures referenced by RESULTS.md (committed)
models/            SD 1.5 weights (gitignored)
data/, runs/       datasets, checkpoints, raw precompute output (gitignored)
```

Frontend: plain HTML + ES modules, no build step. Libraries from CDN with pinned versions: KaTeX 0.16.11 (equations), TensorFlow.js 4.22.0 + WASM backend (Section 3). All charts are custom Canvas 2D. Deployment through a GitHub Actions Pages workflow that publishes `site/`.

**Design.** The default design follows `.documents/ui-style-preferences.SKILL.md`:
- Sky/peach/coral palette in `theme-sky.css`, with a dark mode that follows the system setting. A ☾/☀ toggle (remembered) and `?theme=dark|light` override it.
- Font: Source Sans 3 (400/600).
- Method colors: CFG sky, CG coral, TFG lavender.
- Spacing: large hero spacing, and sections separated by a divider with 88 px above and below.

`?style=default` shows the saved v1 design (`main.css` alone). Canvas modules redraw on a `themechange` event and after web fonts load.

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

**Implementation (done, v1).** Vanilla JS + Canvas 2D: math in `site/js/toy2d/gmm.js`, UI in `toy2d.js`.
- Node checks (conda env `webdev`): oracle CG and CFG agree to about 1e-15 relative to the field scale; the score and the $\hat x_0$ Jacobian match finite differences to about 1e-9; at $s=1$, CG and CFG give the same pencil fraction.
- URL parameters (`?method=cfg&s=2&frame=50&model=estimated&kappa=0.6&bg=classifier`) reproduce a given state.
- Local preview: `python -m http.server 8765 -d site`.

**UI revisions after feedback (11/19).** These supersede the Views paragraph above where they differ.
- Removed: the guidance start-time slider (guidance is always on), the "guided target" and "CG−CFG difference" backgrounds, and the guided/base $\hat x_0$ selector (always guided).
- Background is a two-way segment: density $p_t(x)$ or classifier $p_t(\text{pencil}\mid x)$.
- The six layer checkboxes sit behind a "Display settings" fold, and the hover readout plus the CG=CFG badge sit behind a "Field values" fold.
- With CG or CFG selected, the classifier contours $p_t(\text{pencil}\mid x)=0.25/0.5/0.75$ are drawn as soft coral bands (4/6/8 px, growing with the level), so the guidance arrows visibly cross them at right angles. With any guidance on, the base-score arrows fade to 30% opacity.
- Defaults: method CG, Estimated model ($\kappa=0.5$), $s=1.5$, $\hat x_0$ rings off (the legend entry hides with them), and starting frame 50/80 ($t\approx0.38$), where contours and guidance are visible.
- The chart is titled "Pencil fraction vs. guidance scale" (TFG mode: "Distance to TFG target vs. guidance scale"). Its series and captions use the method colors.

## Section 2 — Stable Diffusion (offline precompute)

**Fixed settings (confirmed in the pilot).** SD 1.5 fp16, 512×512, DDIM 30 steps, $\eta = 0$.

- Base field: $\varepsilon_\text{base} = \varepsilon_\varnothing + w_\text{cat}(\varepsilon_\text{cat} - \varepsilon_\varnothing)$ with a fixed $w_\text{cat}$ (around 5). Raw $w = 1$ samples are poor, so the base is itself CFG toward "cat"; the README states this.
- Prompts: "a photo of a cat" (base) vs "a pencil drawing of a cat" (condition), with exact wording finalized in the pilot.
- CFG direction: $\varepsilon_\text{pencil-cat} - \varepsilon_\text{cat}$, both from the same guided-base construction.

**CG classifier.**
1. Dataset (`pipeline/sd/make_cg_dataset.py`): 1000 final latents per class, DDIM 20 steps. Class "photo" uses the base field (cat prompt, $w_\text{cat}$); class "pencil" uses the pencil prompt with the same $w_\text{cat}$, which is exactly the CFG $s=1$ target. The prompts match the experiment exactly. Only $z_0$ is stored (4×64×64 fp16).
   - v2 (after v1 learned only a color cue, see RESULTS.md): 2000 per class, plus 1000 "a black and white photo of a cat" latents labelled photo (hard negatives that remove the color shortcut). Classifier v2 is **3-class** (photo / photo_bw / pencil, softmax CE with equal effective priors), and CG uses $f_\text{pencil}-f_\text{photo}$. For an oracle this is exactly $\log p_t(z\mid\text{pencil})-\log p_t(z\mid\text{photo})$, so oracle CG = CFG still holds; photo_bw only forces texture features in the shared trunk. (`--mode binary` folds photo_bw into photo; oracle equality is then only approximate.)
2. Training (`pipeline/sd/train_cg_classifier.py`, model in `cg_classifier.py`): a time-conditioned ResNet (about 6.4M params, GroupNorm + SiLU, no max-pooling so input gradients stay smooth) on $z_t = \alpha_t z_0 + \sigma_t \varepsilon$ with $t \sim U\{1..999\}$ and horizontal-flip augmentation. v1 is binary (BCE, 6000 iterations, `classifier_v1.pt`); v2 is 3-class (CE, 12000 iterations, `classifier_v2.pt`, used everywhere). Both have equal class priors. Report validation accuracy vs $t$.
3. Use the **logit** gradient: $g_t = \nabla_z [\log p_\phi(\text{pencil} \mid z_t) - \log p_\phi(\text{photo} \mid z_t)]$. With equal priors, an oracle logit equals $\log p_t(z \mid \text{pencil}) - \log p_t(z \mid \text{photo})$, so $-\sigma_t g_t = \varepsilon_\text{pencil}-\varepsilon_\text{photo}$, and the CG direction is $-w_\text{cat}\,\sigma_t g_t$ (the CFG target is $p(z)\,p(\text{pencil}\mid z)^{w_\text{cat}}$), which equals $d_\text{cfg}$. Oracle CG and CFG then coincide at the same $s$, and any measured gap is estimation error. For this reason CG uses no scale calibration (unlike TFG).
4. Check (`pipeline/sd/eval_cg.py`): CFG vs CG finals at equal $s$, and per-step cos(CFG direction, CG direction) and RMS ratio.

**TFG loss (decided in the cold test, see RESULTS.md).** VGG16 Gram-matrix style loss (relu1_2–relu4_3) between the TAESD-decoded clean prediction and a reference pencil drawing (the CFG $s=1$ sample of held-out seed 7). Gradients flow through the decoder, $\hat x_0$, and the UNet back to $z_t$. CLIP-based losses were tried and rejected because they were adversarial. The TFG scale is calibrated so that $s=1$ matches the median CFG guidance RMS; the usable range is $s\in[1,2]$. The result is a grayscale pencil *tone* rather than line strokes, which is accepted.

**Recorded per step** (custom DDIM loop in `pipeline/sd/sampler.py`):
- $z_t$ preview (cheap latent→RGB linear projection, 64 px)
- $\hat x_0$ from the base denoiser and from the guided $\tilde\varepsilon$ (full VAE decode → 256 px)
- $\lVert g_t \rVert$; cos(CG direction, CFG direction), computed in every run since both are cheap
- CLIP pencil score of $\hat x_0$; L2 distance to the unguided run's $\hat x_0$
- final image (512 px)

**Precompute grid** (`pipeline/sd/precompute.py`, decided): seeds 0–5 (7 is the TFG reference) × {none} ∪ {CFG, CG, TFG} × $s\in\{0.5,1,2\}$ × start step $\in\{0,10,20\}$, which is 28 runs per seed and 168 in total, about 55 min. The slider is $s\in\{0,0.5,1,2\}$ ($s=0$ = none) with default $s=1$, the point where oracle CG = CFG. Every step records the guided and base $\hat x_0$ (256 px) and the $z_t$ preview. CG uses classifier v2. Raw output goes to `runs/sd_grid/` (about 220 MB).

**Status (11/19): done.** The full grid ran (168 runs, 49 min) and was exported to `site/assets/sd/` (35.9 MB, not committed until the final export).

**Export.** `pipeline/sd/export.py` (1–2 min, CPU) writes the following and skips incomplete runs:
- Per run: three WebP sprite sheets, each an 8×4 grid where frames 0–29 are the steps and frame 30 is the final image. Guided $\hat x_0$ at 192 px, base $\hat x_0$ at 128 px, $z_t$ at 64 px.
- Per seed: a metrics JSON and a thumbnail.
- Global: `index.json`.

**Frontend (`site/js/sd/sd.js`).**
- Controls: seed thumbnails (64 px, unselected ones dimmed; default seed 2) and an $s$ segment {0, 0.5, 1, 2} (default 1). The start step {0, 10, 20} sits behind an "Advanced: guidance start" fold.
- Four columns: No guidance (reference card, coral header), then CFG / CG / TFG (badges A/B/C). Each shows guided $\hat x_0$ large, base $\hat x_0$ and $z_t$ small, and a pencil-score bar.
- One shared step slider + play; frame 30 is the final image.
- Three titled step charts: "Guidance strength", "Direction match with CFG" (on the unguided path), and "Pencil score of $\hat x_0$". Clicking or dragging a chart scrubs the step.
- URL state: `?sdseed=3&sds=2&sdstart=10&sdframe=12`.

## Section 3 — Draw an Outline (TF.js, live)

**Data.** Quick, Draw! "cat" simplified drawings (CC BY 4.0, about 120k), rasterized to 32×32 grayscale with antialiased strokes, stored as a uint8 `.npy`.

**Model.** Small ε-prediction UNet (about 2–4M params): conv, GroupNorm, SiLU, sinusoidal time embedding, 2 down/up levels, optional one attention block at 8×8. Cosine VP schedule. Trained in PyTorch (a few hours in the background on the 3060).

**Status (11/19).**
- Data: `pipeline/sketch/prep_quickdraw.py` keeps 103,031 recognized cats out of 123,202, at 32×32 (`data/sketch/cat32.npy`, 45 s). Preview: `results/sketch/data_preview.png`.
- Model: `pipeline/sketch/model.py` (`SketchUNet`, widths 32/64/128, no attention, 2.35M params). It uses only TF.js-portable ops. A width of 48/96/192 (5.3M) was too heavy for in-browser gradients.
- Schedule: continuous cosine with $t\in[0,1]$, and the network sees $1000t$. DDIM uses $\hat x_0$ clipping.
- Training: `pipeline/sketch/train.py`, ε-MSE, batch 256, EMA 0.9995, 60k iterations, about 1.6 h. Samples go to `results/sketch/samples_*.png`. A 1k-iteration smoke test already gives cat-like sketches.

**Port to TF.js.** Re-implement the same architecture with TF.js ops (GroupNorm written manually) and load weights from an exported binary + JSON manifest. Parity test: export input/output test vectors from PyTorch and require max abs error below 1e-3 in the browser before building any UI.

**Port status (11/19).** Implemented and passing:
- `pipeline/sketch/export_tfjs.py` writes `site/assets/sketch/{weights.bin (float32, TF.js layout, 9.4 MB), manifest.json, parity.json}` from the EMA weights.
- `site/js/sketch/unet.js` is the TF.js UNet and schedule.
- `site/tests/sketch_parity.html` passes on WebGL, WASM, and CPU: eps error about 2e-6, guidance-gradient relative error about 1e-6, and $\bar\alpha$ error about 5e-8. It also prints a WebGL renderer diagnostic and timings.

Three tfjs 4.22 pitfalls, all handled and checked by `site/tests/tfjs_grad_ops.html`:
1. `resizeNearestNeighbor` has a wrong gradient on CPU for batch > 1, and `tile` has no rank-6 gradient. Upsampling therefore uses concat + reshape.
2. Stride-2 conv with 'same' pads asymmetrically, so it uses an explicit symmetric pad followed by 'valid'.
3. `tf.where`/`greater` have no gradient, so the loss weight's step function is built from `sign` (zero gradient).

Re-run the export after training finishes (`python -m pipeline.sketch.export_tfjs`, seconds).

**Guidance design (decided 11/19, `pipeline/sketch/guided_test.py`).**
- Loss: weighted MSE between blurred ink maps (Gaussian σ = 1). Extra ink away from the outline has weight 0.2, so the model can still add eyes and whiskers.
- Rejected alternatives: plain MSE only copies the outline, and a coverage-only loss scribbles ink everywhere.
- Guidance term: $\tilde\varepsilon=\varepsilon+s\,c\,\sigma_t\nabla_x\ell(\hat x_0)$ with a fixed $c=10^4$; $s\in[0,1]$, default 0.25.

**UI v1 (`site/js/sketch/sketch.js`).**
- TF.js, weights, and the model load only when §3 scrolls into view.
- Backend order: WebGL → WASM → CPU. A speed probe at load (one guided step at batch 2 above 400 ms) switches to light mode: batch 2 instead of 3, plus a notice.
- Defaults: 20 steps (slider 10–50), guidance on for the first 60% of steps.
- Display: live $\hat x_0$ per step as crisp 32×32 pixels, an unguided twin from the same noise, an outline overlay, an optional add/remove-ink overlay ($\partial\ell/\partial\hat x_0$), a loss-per-step chart, and a scrub slider after generation.
- Pad: the card is exactly as wide as the pad (about 656 px), so the whole card is drawable. Border: 2 px dashed (brighter on hover). Strokes are shown 30% thinner than the rasterized target, which keeps the training stroke width (1.6 px at 32×32).
- Presets: Head, Oval, Ears.
- Measured guided step at batch 4: about 99 ms on WebGL (laptop GPU) and about 1.1 s on WASM.

(The earlier plan of 20–100 steps, batch 4, and a plain blurred MSE is superseded by the decisions above.)

## Step Order

| # | Planned | Step | Status (11/19) |
|---|---|---|---|
| 0 | 11/18–19 | Repo cleanup, env check, skeleton site | done (Pages workflow still to add) |
| 1 | 11/19–22 | SD pilot: custom DDIM, CFG + TFG, small grid | done |
| 1b | 11/19–22 | Quick, Draw! prep + sketch model training | prep done; training running (60k iterations) |
| 2 | 11/22–25 | Section 1 toy module | done, revised with feedback |
| 3 | 11/19–22 | CG dataset + classifier | done (v1 and v2) |
| 4 | 11/27–30 | Full SD precompute + export; Section 2 frontend | done |
| 5 | 11/30–12/4 | TF.js port + parity test; Section 3 UI | done v1; re-export final weights, then feedback |
| 6 | 12/4–12/7 | Explanations and equations, README, deploy | next |
| — | 12/8–12/9 | Buffer, presentation prep | — |

Remaining:
- Final sketch export once training finishes.
- §3 feedback pass.
- Text pass over all three sections.
- README (including the $w_\text{cat}$ base, Quick, Draw! CC BY 4.0, and SD licenses).
- GitHub Pages workflow.
- Single final commit of `site/assets/`.

## Risks and Fallbacks

- **CG/TFG gradients on SD look adversarial rather than stylistic.** Tune the scale range, apply guidance from a start time, and normalize the gradient. If results stay weak, present them honestly as estimation error relative to CFG.
- **Out of memory in TFG backprop.** Use the TAESD gradient path, lower-resolution decoding for the loss, and gradient checkpointing.
- **Site asset size.** Shrink the grid (seeds/scales), use 192 px frames, or subsample frames for the base $\hat x_0$ stream. Commit exported assets only at final export to keep git history small.
- **TF.js speed.** Handled: 32×32 with 2.35M params, 20 steps by default, guidance only on the first 60% of steps, a WASM fallback, and an automatic light mode (batch 2) when no GPU is available.
