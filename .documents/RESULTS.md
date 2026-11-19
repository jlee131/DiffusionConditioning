# Results Log

Short notes per experiment. Figures live in `results/`, raw outputs in `runs/` (gitignored).

## 2026-11-19 — SD cold test (CFG + TFG, no CG yet)

**Setup.** SD 1.5 fp16, 512 px, DDIM 30 steps ($\eta=0$). Base = "a photo of a cat" with $w_\text{cat}=5$ vs empty prompt. Condition = "a pencil drawing of a cat". Seeds 0–2, same initial noise for all methods. Code: `pipeline/sd/` (`python -m pipeline.sd.cold_test`). About 9 s per CFG run and 17 s per TFG run; peak memory 5 GB.

**Figures.** `results/sd_cold_test/finals_seed{0,1,2}.png` (final images), `filmstrip_seed{0,1,2}.png` ($z_t$ and $\hat x_0$ over steps), `tfg_explore_seed0.png` / `tfg_explore2_seed0.png` (TFG loss search), `tfg_reference.png`.

**CFG** ($\tilde\varepsilon=\varepsilon_\text{base}+s\,w_\text{cat}(\varepsilon_\text{pencil}-\varepsilon_\text{cat})$)
- Works reliably. The switch to pencil happens between $s=0$ and $s\approx0.5$ (seeds 1, 2) or $0.5$–$1$ (seed 0), so the final grid should be dense on $[0,1]$.
- It can change composition entirely: in seed 0, $s\ge1$ produces a drawing on paper lying on a table.
- Start at step 10: for seed 1 the style still changes late; for seeds 0 and 2 the change is almost none.

**TFG**
- CLIP-based losses (2-way softmax, text cosine, reference-image cosine) are **adversarial**: colors saturate, silhouettes appear, or the image breaks, while CLIP scores barely move. Rejected.
- VGG Gram style loss against a reference pencil drawing (CFG sample of seed 7), with the gradient through $\hat x_0$ and the UNet, gives a smooth effect: desaturated, graphite-like photos that keep the cat. The usable range is $s\in[1,2]$, and $s=4$ breaks. It captures tone and texture statistics rather than line strokes. Early guidance still shifts the pose.
- The CLIP pencil score is a poor metric for TFG. A Gram-distance or saturation metric is needed.

**Clean predictions.** Composition is visible in $\hat x_0$ by steps 6–9, while $z_t$ looks like noise until about step 15. This confirms $\hat x_0$ as the main view. The TFG-guided $\hat x_0$ is wild in early steps (steps 0–9) while the base $\hat x_0$ stays plausible, a useful "estimate, not endpoint" illustration.

**Decision.** TFG = VGG Gram loss; the grayscale-pencil-tone result is accepted.

**Takeaway (cold test).** The direction is visually viable. Contrast to show on the site: CFG uses the model's learned joint, so it changes style *and* scene, while a TFG style loss changes appearance statistics while mostly keeping the scene.

## 2026-11-19 — CG classifier and CG vs CFG

**Classifier.** Noisy latent classifier (6.4M params) on 1000 + 1000 base-model latents (`make_cg_dataset`, 20-step DDIM). 6000 iterations, 16 min. Validation accuracy is 0.97 for $t\le550$, 0.94 at $t=750$, 0.91 at 850, and 0.81 at 950 (`results/cg_classifier/acc_vs_t.png`).

**Bug fixed.** The CG direction was missing the factor $w_\text{cat}$. The CFG target is $p(z)\,p(\text{pencil}\mid z)^{w_\text{cat}}$, so the direction is $-w_\text{cat}\sigma_t\nabla_z\text{logit}$. Before the fix, CG ran at an effective $s/5$.

**CG vs CFG** (`python -m pipeline.sd.eval_cg`, `results/cg_eval/`). Same noise, same $s\in\{0.5,1,2,4\}$.
- CFG turns every seed into a pencil drawing (CLIP pencil score about 1.0 for $s\ge1$).
- CG mostly **desaturates** the photo (gray, high contrast) but does not produce pencil texture (CLIP score 0.0–0.55).
- Direction: cos(CFG, CG) is 0.75 at step 0, about 0.3 by step 4, and close to 0 after step 15.
- Magnitude: rms(CG)/rms(CFG) is 2–3 in the first steps and then decays to about 0 after step 10.
- Reading: the classifier is accurate, but its input gradient carries a low-frequency color cue. Its strength at low noise does not scale like the true $1/\sigma_t$ score gap, and the texture that makes "pencil" forms late. Accuracy does not imply a useful gradient.

**Decision.** Keep this as the SD result for CG. It is the practical counterpart of the toy "Estimated" mode: CG equals CFG only with an exact classifier.

## 2026-11-19 — CG classifier v2 (3-class, black-and-white hard negatives)

**Data/training.** 2000 photo + 2000 pencil + 1000 "a black and white photo of a cat" latents. 3-class softmax (photo / photo_bw / pencil, equal effective priors); CG uses $f_\text{pencil}-f_\text{photo}$, so oracle CG = CFG still holds. 12000 iterations, 34 min. Pencil-vs-photo validation accuracy is 0.99 for $t\le350$ and 0.80 at $t=950$; 3-way accuracy is 0.97 → 0.68. Checkpoints: `runs/cg_classifier/classifier_v1.pt`, `classifier_v2.pt`. Figures: `results/cg_eval/` (v2) and `results/cg_eval_v1/`.

**CG vs CFG (v2).** Mean CLIP pencil score over seeds 0–2:

| $s$ | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|
| CFG | 0.67 | 1.00 | 1.00 | 1.00 |
| CG v1 | 0.16 | 0.09 | 0.20 | 0.06 |
| CG v2 | 0.21 | 0.20 | 0.36 | 0.04 |

- v2 is better at $s=1$–2: higher-contrast, sketch-like shading appears, but the image is still a photo, with no paper or stroke texture.
- The direction stays aligned longer: on the unguided path, cos is about 0.3 until step 10, compared with step 4 for v1. The magnitude ratio decays more slowly but still reaches about 0 by step 20.
- At $s=4$ the early overshoot (ratio of about 2 in the first steps) washes the image out.
- The remaining gap is structural, not a color shortcut: the learned gradient does not grow like $1/\sigma_t$ at low noise, where pencil texture forms.

**Decision.** Use v2 for the SD section. Optionally show v1 vs v2 as "a better classifier narrows but does not close the gap".

## 2026-11-19 — Full SD grid (Section 2)

`python -m pipeline.sd.precompute`: 168 runs (seeds 0–5 × none ∪ {CFG, CG v2, TFG} × $s\in\{0.5,1,2\}$ × start step $\in\{0,10,20\}$), 49 min. Exported to `site/assets/sd/` (35.9 MB). Mean final CLIP pencil score over 6 seeds (none = 0.01):

| start | method | $s=0.5$ | $s=1$ | $s=2$ |
|---|---|---|---|---|
| 0 | CFG | 0.73 | 1.00 | 1.00 |
| 0 | CG | 0.22 | 0.28 | 0.38 |
| 0 | TFG | 0.06 | 0.08 | 0.20 |
| 10 | CFG | 0.27 | 0.53 | 0.66 |
| 10 | CG / TFG | ≤ 0.02 | ≤ 0.01 | ≤ 0.01 |
| 20 | CFG | 0.12 | 0.14 | 0.14 |
| 20 | CG / TFG | 0.01 | 0.01 | 0.01 |

- Late start weakens every method, but CFG keeps a partial effect at step 10 while CG and TFG do nothing. This matches the per-step curves: the CG and TFG directions only matter in the first ~10 steps.
- CG varies by seed: seed 3 at $s=2$ reaches 0.99, while the other seeds stay low.
- TFG's CLIP score underrates it (it changes tone and texture statistics, not "drawing-ness"), as noted in the cold test.

## 2026-11-19 — Sketch model (Section 3): data, guidance loss, browser port

**Data.** Quick, Draw! "cat", simplified (CC BY 4.0). 103,031 of 123,202 drawings were recognized by the game and kept. They are centered, drawn with 1.6 px antialiased strokes, and stored at 32×32 (`results/sketch/data_preview.png`).

**Model and training.** `SketchUNet`, 2.35M params (widths 32/64/128), with a continuous cosine schedule. A 5.3M variant (48/96/192) trained fine, but was dropped to keep in-browser gradients fast. Training uses ε-MSE, batch 256, and EMA, for 60k iterations (about 1.6 h, still running at the time of writing). Samples are already cat-like at 1k iterations and clean by 10k (`results/sketch/samples_*.png`).

**Outline guidance loss** (`python -m pipeline.sketch.guided_test`, checkpoint at 10k, 20 DDIM steps, guidance on the first 60%, 8 seeds; `results/sketch/guided_{head,body}_{mse,mse_w,coverage}.png`):

| loss on blurred ink maps | result |
|---|---|
| MSE | follows the outline from $s\approx0.25$, but only copies it (no face details) |
| coverage only (missing ink on the outline) | ink everywhere; fails |
| **weighted MSE** (extra ink off the outline ×0.2) | follows the outline and still adds eyes and whiskers; an oval outline gets ears and a face |

The best range is $s\approx0.1$–$0.25$ (with $c=10^4$); $s\ge0.5$ mostly reproduces the outline. Decision: weighted MSE, default $s=0.25$.

**TF.js port.** The parity test passes on WebGL, WASM, and CPU: ε error about 2e-6 and guidance-gradient relative error about 1e-6. It found a tfjs 4.22 CPU bug in the `resizeNearestNeighbor` gradient for batch > 1, so upsampling uses concat + reshape. Speed per guided step at batch 4 (forward + backward through the UNet):

| environment | plain step | guided step |
|---|---|---|
| laptop, hardware acceleration on (WebGL, AMD Radeon) | 37 ms | 99 ms |
| laptop, hardware acceleration off (WASM) | 113 ms | 1128 ms |

At default settings this means about 2 s per generation with a GPU, and about 7 s in light mode (batch 2) without one.
