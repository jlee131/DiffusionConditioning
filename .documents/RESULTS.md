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

**Takeaway.** The direction is visually viable. Contrast to show on the site: CFG uses the model's learned joint, so it changes style *and* scene, while a TFG style loss changes appearance statistics while mostly keeping the scene.
