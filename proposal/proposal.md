# Conditioning: How Guidance Steers Diffusion Generation
 
**Project proposal — Fall 2026**

2022-18121 Joonha Lee


## Topic and Objective

This project will develop an interactive website explaining classifier guidance (**CG**), classifier-free guidance (**CFG**), and training-free guidance (**TFG**). Using cat generation and pencil-drawing style as a visual example, it will show how conditioning changes both the sampling trajectory and the predicted clean output. 

The backend will use pretrained Stable Diffusion 1.5 with deterministic DDIM sampling ($\eta=0$) to eliminate the randomness in the diffusion process, better displaying the effect of guidance.


## Mathematical Concepts

The central identity is the Bayesian decomposition of the conditional score:

$$
\nabla_{z_t}\log p_t(z_t\mid c)
=\nabla_{z_t}\log p_t(z_t)
+\nabla_{z_t}\log p_t(c\mid z_t).
$$

CG estimates the second term with a noise-dependent classifier; CFG obtains it from the difference between conditional and unconditional score estimates. With exact estimates, they produce the same guided field at matching scales. TFG instead constructs a surrogate direction from a task loss, for example

$$
G_t=-\nabla_{z_t}\ell\!\left(D(\hat z_0(z_t,t)),c\right),
$$

where ($D$) is the image decoder. The visualization will distinguish the current noisy latent ($z_t$), its clean prediction ($\hat z_0$), and the final generated image. It will also explain that guidance may change pose and composition.


## Planned Interactive Features

1. **Sampling trajectories.** Compare guidance methods from identical initial noise using synchronized playback. Users will select seeds, guidance strengths, and conditioning start times. Intermediate image previews will accompany a clearly labeled shared projection of latent trajectories.
2. **Clean predictions.** Using the same recorded runs, a linked time slider will display decoded clean predictions beside the current-state previews. Cat-only and pencil-cat conditions will demonstrate how predictions change when style conditioning begins immediately or partway through sampling.
3. **Draw an outline.** Users will sketch a cat outline and explore TFG with a differentiable outline-matching loss applied to the decoded clean prediction. The interface will show how guidance strength affects outline agreement and visual plausibility.
