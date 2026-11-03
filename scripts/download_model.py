"""Fetch only the FP16 Diffusers files into the local SD 1.5 clone."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "stable-diffusion-v1-5"


def main():
    import subprocess

    # Pin downloads to the commit checked out by git, rather than a moving main.
    revision = subprocess.check_output(
        ["git", "-C", str(MODEL), "rev-parse", "HEAD"], text=True
    ).strip()
    snapshot_download(
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        revision=revision,
        local_dir=MODEL,
        allow_patterns=[
            "model_index.json", "README.md", "LICENSE*",
            "*/config.json", "scheduler/*.json", "tokenizer/*",
            "feature_extractor/*.json", "*/*.fp16.safetensors",
        ],
        max_workers=3,
    )
    manifest = {"repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
                "revision": revision, "variant": "fp16"}
    (MODEL / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
