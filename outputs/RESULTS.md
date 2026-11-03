# Generation Results

Date: 2026-10-22. These are single-seed exploratory runs, not aggregate evaluations.

## Shared Settings

- Model: `stable-diffusion-v1-5/stable-diffusion-v1-5`, revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, FP16.
- GPU: RTX 3060, 12 GB. PyTorch 2.8.0+cu128; Diffusers 0.35.2.
- Resolution: 512 x 512. Seed: 42. DDIM: 20 steps, eta 0; no additional sampling noise after initialization.
- All four saved runs have identical initial latents and timesteps, verified directly.
- Contact sheets show decoded clean predictions at steps 1, 4, 7, 11, 14, 17, and 20, followed by the final image. Predictions are estimates, not noisy trajectory states.

## Conditions and Outputs

| Run | Initial condition | Later condition | Final image | Clean predictions |
| --- | --- | --- | --- | --- |
| Smoke test (CFG) | Pencil-cat prompt, CFG 7.5 | Same | [Image](smoke_test/cat.png) | [Sheet](smoke_test/clean_predictions.png) |
| Baseline | Empty prompt, no CFG | Same | [Image](baseline/final.png) | [Sheet](baseline/clean_predictions.png) |
| Cat baseline | Cat prompt, CFG 7.5 | Same | [Image](cat_baseline/cat.png) | [Sheet](cat_baseline/clean_predictions.png) |
| Delayed CFG (30/70; `half_cfg`) | Steps 1-6: cat prompt, CFG 7.5 | Steps 7-20: pencil-cat prompt, CFG 7.5 | [Image](half_cfg/cat.png) | [Sheet](half_cfg/clean_predictions.png) |

Pencil-cat prompt:

> a graphite pencil drawing of a single sitting cat, full body, white paper background, detailed pencil strokes

Cat prompt:

> a single sitting cat, full body, white paper background

The baseline uses the model's empty-text branch; its pipeline guidance scale is 1. Delayed CFG keeps cat conditioning active throughout and switches from the cat prompt to the pencil-cat prompt at step 7 (training timestep 651). CFG stays at 7.5 throughout. The 30/70 split refers to sampling-step counts. The existing `half_cfg` directory and flag are retained for compatibility; this run replaces the previous 50/50 result.

## Observations

- Smoke test: a seated pencil-drawn cat with a gray background.
- Baseline: a brown decorative pattern; no explicit cat or style condition.
- Cat baseline: a white seated cat with a brown background and foreground details. Its pose differs from the pencil-cat result.
- Delayed CFG (30/70): a white cat on a brown textured surface. The face, front-leg arrangement, and foreground differ from the cat baseline; visible fine line texture develops, but the result does not match the strong pencil appearance of full-run pencil conditioning.
- The first six clean prediction tensors exactly match the cat baseline; the prediction changes at step 7. The initial latent and timestep schedule also match the cat baseline.

## Reproduce

Run from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py --save-predictions
.\.venv\Scripts\python.exe scripts\smoke_test.py --unconditional --save-predictions
.\.venv\Scripts\python.exe scripts\smoke_test.py --prompt "a single sitting cat, full body, white paper background" --output-dir outputs/cat_baseline --save-predictions
.\.venv\Scripts\python.exe scripts\smoke_test.py --half-cfg --cat-fraction 0.3 --save-predictions
```

Each local run directory contains `report.json`, individual images under `predictions/`, and `predictions.pt` with the initial latent and all 20 clean latent predictions. The Half CFG schedule is implemented through local U-Net hooks; installed Diffusers source is unchanged. Git tracks this summary, contact sheets, final images, and run reports; individual prediction images and tensor files remain local.
