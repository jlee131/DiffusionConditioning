# Steering Diffusion: CG, CFG and Training-Free Guidance

An interactive, browser-only explainer of three ways to steer a diffusion model: **classifier guidance (CG)**, **classifier-free guidance (CFG)** and **training-free guidance (TFG)**. Final project for *Mathematics of Diffusion Models* (SNU, Fall 2026).

**Live site:** https://jlee131.github.io/DiffusionConditioning/

All three methods are written as one update on the DDIM noise prediction,

$$\tilde\varepsilon = \varepsilon_\text{base} - s\,\sigma_t\, g_t ,$$

and differ only in where the guidance direction $g_t$ comes from: a trained classifier (CG), the difference of two conditional predictions (CFG), or the gradient of a loss on the clean estimate $\hat x_0$ (TFG).

## The three sections

| Section | What it shows | How it runs |
|---|---|---|
| 1. Toy 2D | A 2D Gaussian mixture ("photo" vs "pencil" cats) with exact scores. Guidance arrows, $p_t(\text{pencil}\mid x)$ contours, sample paths, and per-step charts. | Live in the browser (exact formulas, no network). |
| 2. Stable Diffusion | "A cat" pushed toward "a pencil drawing of a cat" by CFG, CG and TFG on SD 1.5. Scrub every DDIM step: guided $\hat x_0$, unguided $\hat x_0$, $z_t$, and three charts. | Precomputed offline (6 seeds × 3 scales × 3 start steps), served as WebP sprites. |
| 3. Draw an Outline | Draw a shape; a small sketch diffusion model trained on Quick, Draw! cats fills it, guided through $\hat x_0$ by an outline loss. | Live in the browser with TensorFlow.js (WebGL, with WASM/CPU fallback). |

## Repository layout

```
site/                 the static website (published to GitHub Pages as-is)
  index.html
  css/                main.css (base), theme-sky.css (default theme, light/dark)
  js/                 toy2d/, sd/, sketch/
  assets/sd/          SD sprite sheets + metrics (from pipeline/sd/export.py)
  assets/sketch64/    TF.js weights of the 64 px sketch model (default)
  assets/sketch/      TF.js weights of the 32 px sketch model (?skmodel=sketch)
  tests/              browser checks: sketch_parity.html, tfjs_grad_ops.html
pipeline/             offline Python (never served)
  sd/                 custom DDIM sampler, guidance, CG classifier, precompute, export
  sketch/             Quick, Draw! prep, sketch UNet training, TF.js export
results/              figures referenced by the project notes
.documents/           goal, plan, results log, proposal
models/, data/, runs/ weights, datasets, checkpoints (gitignored)
```

## View the site locally

No build step. Any static server works:

```
python -m http.server 8765 -d site
```

Then open http://localhost:8765. Useful URL parameters: `?theme=dark|light`, `?style=default` (the plain base design), `?skmodel=sketch` (32 px sketch model).


## Notes on the setup

- **Base field.** Plain SD 1.5 samples of "a cat" at $w = 1$ are poor, so the base in Section 2 is itself CFG toward "a cat": $\varepsilon_\text{base} = \varepsilon_\varnothing + w_\text{cat}(\varepsilon_\text{cat} - \varepsilon_\varnothing)$ with $w_\text{cat} = 5$. Every method then adds its guidance on top of this base.
- **CFG** uses $w(\varepsilon_\text{pencil} - \varepsilon_\text{cat})$.
- **CG** uses a 3-class latent classifier (photo / black-and-white photo / pencil) trained on noisy SD latents. Its guidance is the gradient of $f_\text{pencil} - f_\text{photo}$, scaled by $w_\text{cat}$ so that CG with a perfect classifier equals CFG.
- **TFG** uses a VGG Gram-matrix style loss on the TAESD decode of $\hat x_0$, against a reference pencil drawing (seed 7, CFG $s = 1$).
- **Section 3** uses a weighted MSE between blurred ink maps of $\hat x_0$ and the drawn outline; ink outside the outline costs less than missing ink.

## Credits and licenses

- **Stable Diffusion 1.5** (Runway, CompVis, Stability AI): CreativeML Open RAIL-M license. The images in `site/assets/sd/` were generated with it and are subject to its use restrictions.
- **Quick, Draw! dataset** (Google): CC BY 4.0. The sketch model in Section 3 is trained on its "cat" drawings.
- **TAESD** (madebyollin): MIT. **CLIP** (OpenAI): MIT. **VGG16** ImageNet weights via torchvision.
- **TensorFlow.js** (Apache 2.0) and **KaTeX** (MIT), loaded from jsDelivr. **Source Sans 3** (SIL OFL), from Google Fonts.
