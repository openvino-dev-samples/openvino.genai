#!/usr/bin/env python3
"""Smoke test for the FLUX.2-klein C++ pipeline exposed via openvino_genai bindings.

Runs a text-to-image generation and (optionally) an image-editing pass, then
saves the resulting PNGs. Intended for quick local validation of the new
``Flux2KleinPipeline`` C++ implementation.

Usage (PowerShell):
    $env:PYTHONPATH = "D:\\openvino.genai\\build"
    D:\\optimum-intel\\py_env\\Scripts\\python.exe \
        D:\\openvino.genai\\tools\\flux2_klein_smoke_test.py \
        --model D:\\models\\FLUX.2-klein-4B-OV-INT4 --device CPU
"""

import argparse
from pathlib import Path

import numpy as np
import openvino_genai as ov_genai
from PIL import Image


def tensor_to_image(image_tensor) -> Image.Image:
    arr = image_tensor.data
    # Expected layout: (B, H, W, C) uint8
    if arr.ndim == 4:
        arr = arr[0]
    return Image.fromarray(arr.astype(np.uint8))


def run_text2image(args) -> None:
    print(f"[t2i] loading Text2ImagePipeline from {args.model} on {args.device}")
    pipe = ov_genai.Text2ImagePipeline(args.model, args.device)

    print(f"[t2i] generating: {args.prompt!r}")
    result = pipe.generate(
        args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        rng_seed=args.seed,
    )
    image = tensor_to_image(result)
    out_path = Path(args.output_dir) / "flux2_klein_t2i.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"[t2i] saved {out_path} ({image.size})")


def run_image_edit(args) -> None:
    ref_path = Path(args.edit_image)
    if not ref_path.exists():
        print(f"[edit] reference image {ref_path} not found, skipping edit test")
        return

    print(f"[edit] loading Image2ImagePipeline from {args.model} on {args.device}")
    pipe = ov_genai.Image2ImagePipeline(args.model, args.device)

    ref = Image.open(ref_path).convert("RGB")
    ref_arr = np.array(ref)[None]  # (1, H, W, C)
    ref_tensor = ov_genai.openvino.Tensor(ref_arr) if hasattr(ov_genai, "openvino") else None
    import openvino as ov

    ref_tensor = ov.Tensor(ref_arr)

    print(f"[edit] editing with prompt: {args.edit_prompt!r}")
    result = pipe.generate(
        args.edit_prompt,
        ref_tensor,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        rng_seed=args.seed,
    )
    image = tensor_to_image(result)
    out_path = Path(args.output_dir) / "flux2_klein_edit.png"
    image.save(out_path)
    print(f"[edit] saved {out_path} ({image.size})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the OV FLUX.2-klein model dir")
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--prompt", default="A cat holding a sign that says hello world")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--edit-image", default="", help="Optional reference image for editing test")
    parser.add_argument("--edit-prompt", default="Make the background a sunny beach")
    args = parser.parse_args()

    run_text2image(args)
    if args.edit_image:
        run_image_edit(args)


if __name__ == "__main__":
    main()
