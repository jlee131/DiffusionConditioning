# Diffusion Conditioning Demo

Local SD 1.5 environment for comparing CG, CFG, and TFG with DDIM sampling.

## Environment

- Windows, NVIDIA RTX 3060 (12 GB VRAM).
- Project-local Windows CPython under `.tools/python`; `.venv` uses this interpreter.
- CUDA PyTorch 2.8.0 (cu128); Diffusers dependencies in `requirements.txt`.
- Model: `stable-diffusion-v1-5/stable-diffusion-v1-5` (community mirror).
- Local clone: `models/stable-diffusion-v1-5`; FP16 Safetensors weights only.
- Model source and license are included in the clone. Downloaded weights replace selected Git LFS pointers, so the clone can show local modifications.

## Run

Use the interpreter directly; PowerShell activation is optional:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py
```

The smoke test runs offline at 512 x 512, with 20 DDIM steps, eta 0, CFG 7.5, and seed 42. Results are saved in `outputs/smoke_test/`. This verifies local generation; trajectory recording and CG/TFG are not implemented yet.

In VS Code, select `.venv/Scripts/python.exe` as the Python interpreter.

To inspect clean predictions from the same generation run:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py --save-predictions
```

This saves seven decoded clean predictions and a final-image comparison in `outputs/smoke_test/clean_predictions.png`. Individual images are in `predictions/`; `predictions.pt` contains the initial latent and all clean latent predictions. These predictions include CFG and are estimates of the clean latent, not the noisy states along the trajectory. No new noise is injected with DDIM eta 0. Snapshot decoding adds time beyond the generation time in the report.

## Reinstall dependencies

```powershell
.\.tools\uv\uv.exe pip install --python .venv\Scripts\python.exe --cache-dir .tools\uv-cache torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.tools\uv\uv.exe pip install --python .venv\Scripts\python.exe --cache-dir .tools\uv-cache -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_model.py
```

The exact installed versions are recorded in `requirements-lock.txt` after setup.

## Unconditional baseline

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py --unconditional --save-predictions
```

Uses empty text with CFG disabled (pipeline guidance scale 1), the same seed 42 and DDIM schedule as the conditioned run. Outputs go to `outputs/baseline/`. This removes the cat condition as well as the pencil-style condition. Use `--prompt "a single sitting cat" --output-dir outputs/cat_baseline --save-predictions` for a separate cat-conditioned style baseline.

## Delayed conditioning

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py --half-cfg --cat-fraction 0.3 --save-predictions
```

Runs the first 6 of 20 steps (30%) with the cat-baseline prompt, then switches to the pencil-cat prompt for the remaining 14 steps (70%). CFG stays at 7.5 throughout; only the prompt changes. `--cat-fraction` defaults to 0.3; use 0.5 for the previous 50/50 schedule. The existing `--half-cfg` flag and `outputs/half_cfg/` directory are retained. See [experiment results](outputs/RESULTS.md) for all four runs.
