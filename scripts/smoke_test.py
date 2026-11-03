"""Run a local, offline SD 1.5 + deterministic DDIM CUDA smoke test."""

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

ROOT = Path(__file__).resolve().parents[1]


class RecordingDDIMScheduler(DDIMScheduler):
    """Record the actual scheduler outputs without changing sampling."""

    def step(self, model_output, timestep, sample, eta=0.0, generator=None, **kwargs):
        if not hasattr(self, "predictions"):
            self.predictions = []
            self.initial_latent = sample.detach().cpu().clone()
        result = super().step(
            model_output, timestep, sample, eta=eta, generator=generator, **kwargs
        )
        clean = result[1] if isinstance(result, tuple) else result.pred_original_sample
        self.predictions.append((int(timestep), clean.detach().cpu().clone()))
        return result


def save_predictions(pipe, final_image, output):
    from PIL import Image, ImageDraw

    snapshots = output / "predictions"
    snapshots.mkdir(parents=True, exist_ok=True)
    predictions = pipe.scheduler.predictions
    count = len(predictions)
    indices = sorted({round(i * (count - 1) / 6) for i in range(7)})
    panels = []
    records = []
    for index in indices:
        timestep, clean = predictions[index]
        clean = clean.to(device=pipe._execution_device, dtype=pipe.vae.dtype)
        with torch.inference_mode():
            decoded = pipe.vae.decode(
                clean / pipe.vae.config.scaling_factor, return_dict=False
            )[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        name = f"step_{index + 1:03d}_t_{timestep:04d}.png"
        image.save(snapshots / name)
        panels.append((f"Clean prediction {index + 1}/{count} | t={timestep}", image))
        records.append({"step": index + 1, "timestep": timestep, "file": name})
    panels.append(("Final generated image", final_image))
    size, caption = 256, 34
    columns = 4
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * size, rows * (size + caption)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, image) in enumerate(panels):
        x, y = (i % columns) * size, (i // columns) * (size + caption)
        draw.text((x + 8, y + 10), label, fill="black")
        sheet.paste(image.resize((size, size)), (x, y + caption))
    sheet.save(output / "clean_predictions.png")
    torch.save({
        "initial_latent": pipe.scheduler.initial_latent,
        "timesteps": [t for t, _ in predictions],
        "pred_original_samples": torch.cat([z for _, z in predictions]),
    }, output / "predictions.pt")
    (snapshots / "index.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional", action="store_true",
                        help="Use the empty-text branch without CFG; overrides --prompt and --guidance-scale.")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--half-cfg", action="store_true",
                        help="Switch from the cat prompt to --prompt; CFG stays enabled.")
    parser.add_argument("--cat-fraction", type=float, default=0.3,
                        help="Fraction of cat-only steps with --half-cfg (default: 0.3).")
    parser.add_argument("--output-dir", type=Path,
                        help="Output directory; defaults to outputs/baseline for unconditional runs.")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save clean predictions and a contact sheet from this sampling run.")
    parser.add_argument("--prompt", default=(
        "a graphite pencil drawing of a single sitting cat, full body, "
        "white paper background, detailed pencil strokes"
    ))
    args = parser.parse_args()
    if args.unconditional:
        args.prompt = ""
        args.guidance_scale = 1.0
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.half_cfg and (args.unconditional or args.guidance_scale <= 1 or args.steps < 2):
        parser.error("--half-cfg requires at least two steps, guidance scale > 1, and a conditional run")
    if not 0 < args.cat_fraction < 1:
        parser.error("--cat-fraction must be between 0 and 1, exclusively")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and CUDA PyTorch installation.")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        ROOT / "models" / "stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
    ).to("cuda")
    scheduler_class = RecordingDDIMScheduler if args.save_predictions else DDIMScheduler
    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    switch_step = max(1, min(args.steps - 1, round(args.steps * args.cat_fraction))) if args.half_cfg else 0
    step_number = 0
    cat_prompt = "a single sitting cat, full body, white paper background"
    initial_prompt_embeds = None
    if args.half_cfg:
        with torch.inference_mode():
            cat_embeds, empty_embeds = pipe.encode_prompt(
                cat_prompt, pipe._execution_device, 1, True
            )
            initial_prompt_embeds = torch.cat([empty_embeds, cat_embeds])

    def gate_prompt(module, inputs, kwargs):
        nonlocal step_number
        if step_number < switch_step:
            # Keep the same unconditional branch and CFG strength; change text only.
            kwargs = dict(kwargs)
            kwargs["encoder_hidden_states"] = initial_prompt_embeds
        step_number += 1
        return inputs, kwargs

    hooks = []
    if args.half_cfg:
        hooks = [pipe.unet.register_forward_pre_hook(gate_prompt, with_kwargs=True)]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt, height=512, width=512,
            num_inference_steps=args.steps, guidance_scale=args.guidance_scale, eta=0.0,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        )
    torch.cuda.synchronize()
    for hook in hooks:
        hook.remove()
    default_output = "half_cfg" if args.half_cfg else "baseline" if args.unconditional else "smoke_test"
    output = args.output_dir or ROOT / "outputs" / default_output
    output.mkdir(parents=True, exist_ok=True)
    image_name = "final.png" if args.unconditional else "cat.png"
    result.images[0].save(output / image_name)
    report = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "seed": args.seed,
        "steps": args.steps, "scheduler": "DDIM", "eta": 0.0,
        "guidance_scale": args.guidance_scale, "prompt": args.prompt,
        "unconditional": args.unconditional,
        "half_cfg": args.half_cfg,
        "unconditional_steps": args.steps if args.unconditional else 0,
        "conditioning_start_step": None if args.unconditional else 1,
        "initial_prompt": cat_prompt if args.half_cfg else args.prompt,
        "prompt_switch_step": switch_step + 1 if args.half_cfg else None,
        "cat_only_steps": switch_step if args.half_cfg else None,
        "cat_fraction": switch_step / args.steps if args.half_cfg else None,
        "resolution": [512, 512],
        "generation_seconds": round(time.perf_counter() - started, 2),
        "peak_allocated_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "nsfw_content_detected": result.nsfw_content_detected,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output / image_name}")
    if args.save_predictions:
        save_predictions(pipe, result.images[0], output)
        print(f"Saved: {output / 'clean_predictions.png'}")


if __name__ == "__main__":
    main()
