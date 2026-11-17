# Project Goal

Build an interactive HTML demo for the final project of Mathematics of Diffusion Models, based on The Principles of Diffusion Models v3, Chapter 8.

Explain how classifier guidance (CG), classifier-free guidance (CFG), and training-free guidance (TFG) steer generation through trajectories and clean predictions.

## 1. Generation Trajectories

Compare CG, CFG, and TFG while generating cats with pencil-drawing style conditioning. Show intermediate sample states and how guidance changes the generation path.

## 2. Clean Predictions

Show the predicted clean sample at each step using the same data and sampling runs as Section 1. Start all methods from identical initial noise and synchronize the time controls. Match sampling settings and share per-step noise when using stochastic sampling.

Clearly label whether a clean prediction comes from the base denoiser or includes guidance. A clean prediction is an estimate, not the actual endpoint of the remaining trajectory.

## 3. Interactive Outline Guidance

Let the user draw a cat outline with the mouse and use TFG to steer generation toward that outline through a differentiable outline-matching loss.

## Mathematical Focus

- Explain the unconditional score and the added guidance direction.
- Show that oracle CG and CFG give the same field at matching guidance scales; practical differences can reflect estimation errors.
- Explain how TFG uses a task loss, including gradients through the clean prediction when applicable.

## Decisions Still Needed

- Shared model, cat data, and compute budget.
- Noisy style classifier for CG, style-conditioned model for CFG, and style loss for TFG.
- Trajectory display, clean-prediction convention, and outline loss.
