# Project Goal

Build an interactive, browser-only website for the final project of Mathematics of Diffusion Models, based on The Principles of Diffusion Models v3, Chapter 8.

The site explains how classifier guidance (CG), classifier-free guidance (CFG), and training-free guidance (TFG) steer generation, moving from exact principles to a real large model and then to a hands-on experiment:

1. **Toy 2D** — principles, computed exactly and live.
2. **Stable Diffusion** — the same methods at scale, precomputed.
3. **Draw an Outline** — TFG live in the browser.

Visualization and interaction are the primary goals. Every section should let the user change something and immediately see how the guided behavior changes.

## 1. Toy 2D: Guidance as a Vector Field

A 2D mixture-of-Gaussians "dataset" with a cat / pencil-cat label, where every quantity is analytic and computed live in the browser.

- Show the density $p_t$, the unconditional score, the guidance direction, and their sum as layered vector fields over time.
- Animate particles along the deterministic sampler and show each particle's clean prediction $\hat x_0$ (Tweedie) linked to it.
- Let the user vary the guidance scale, guidance start time, and method (none / CG / CFG / TFG).
- Show that **oracle CG and oracle CFG give the identical field** at matching scales, and that switching to imperfect estimates (noisy classifier, imperfect conditional model) makes them differ.
- TFG: the user places a target on the canvas; a loss on $\hat x_0$ pulls particles, with the gradient taken through $\hat x_0$.

## 2. Stable Diffusion: CG vs CFG vs TFG on Cats

Pretrained Stable Diffusion 1.5 generating cats, with pencil-drawing style as the condition. All runs are precomputed offline and explored in the browser.

- All methods start from identical initial noise and use the same deterministic DDIM ($\eta = 0$) schedule.
- A single shared time slider synchronizes, for every method side by side: the clean prediction $\hat x_0$ (main view), a small preview of the current state $z_t$, and linked metric plots.
- Clearly label whether a clean prediction comes from the base denoiser or includes guidance. A clean prediction is an estimate, not the actual endpoint of the remaining trajectory.
- The user explores seeds, guidance scale, and guidance start time over a grid dense enough that sliders feel continuous.
- Metric plots are interpretable: cosine similarity between CG and CFG guidance directions per step, guidance norm, distance from the unguided run, and pencil-style score of $\hat x_0$. Hovering a plot jumps to that step.
- Observations to surface: coarse structure (pose, composition) is fixed early while style can still change late; practical CG/CFG differences reflect estimation error.

## 3. Draw an Outline: Live TFG

A small diffusion model trained on cat sketches (Quick, Draw!) runs in the browser with TensorFlow.js.

- The user draws a cat outline with the mouse and generates sketches guided toward it by a differentiable outline-matching loss on $\hat x_0$, with gradients backpropagated through the network's clean prediction.
- Generation is animated step by step, showing $x_t$, $\hat x_0$, and a heatmap of where the guidance gradient adds or removes ink.
- The user controls guidance strength and start time and compares guided vs unguided samples from the same seed, seeing the trade-off between outline agreement and plausibility.

## Mathematical Focus

- The conditional score as the unconditional score plus a guidance direction: $\nabla \log p_t(x \mid c) = \nabla \log p_t(x) + \nabla \log p_t(c \mid x)$.
- Oracle CG and CFG give the same field at matching guidance scales; practical differences reflect estimation errors.
- TFG builds a surrogate guidance direction from a task loss on the clean prediction, including the gradient through $\hat x_0$.

## Deliverables

- Deployed website (GitHub Pages), accessible without installation or login.
- Source code repository.
- README: topic and mathematical background, how to use the site, key implementation details, main observations, references.
