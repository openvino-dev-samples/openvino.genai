"""Standalone Python reference for FLUX.2-klein using the OV IR models directly.

Replicates diffusers Flux2KleinPipeline.__call__ (text2image path) to produce a
golden image and dump per-stage latent statistics, so the C++ pipeline can be
validated/debugged against it.
"""
import argparse
import sys

import numpy as np
import openvino as ov
import torch
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, r"D:\diffusers\src")
from diffusers.pipelines.flux2.pipeline_flux2_klein import (  # noqa: E402
    Flux2KleinPipeline,
    compute_empirical_mu,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (  # noqa: E402
    FlowMatchEulerDiscreteScheduler,
)


def stats(name, t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    print(f"  [{name}] shape={a.shape} min={a.min():.4f} max={a.max():.4f} "
          f"mean={a.mean():.4f} std={a.std():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="A cat holding a sign that says hello world")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=r"D:\openvino.genai\tools\flux2_klein_ref.png")
    ap.add_argument("--pt-transformer", default=None,
                    help="Path to original PyTorch transformer dir; if set, use it instead of OV INT4.")
    args = ap.parse_args()

    core = ov.Core()
    m = args.model
    print("Loading OV models...")
    text_encoder = core.compile_model(rf"{m}\text_encoder\openvino_model.xml", "CPU")
    pt_tr = None
    if args.pt_transformer:
        from diffusers import Flux2Transformer2DModel
        print("Loading PyTorch transformer (fp32)...")
        pt_tr = Flux2Transformer2DModel.from_pretrained(args.pt_transformer, torch_dtype=torch.float32).eval()
        transformer = None
    else:
        transformer = core.compile_model(rf"{m}\transformer\openvino_model.xml", "CPU")
    vae_decoder = core.compile_model(rf"{m}\vae_decoder\openvino_model.xml", "CPU")
    tokenizer = AutoTokenizer.from_pretrained(rf"{m}\tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(rf"{m}\scheduler")

    bn = np.load(rf"{m}\vae_bn_stats.npz")
    bn_mean = torch.tensor(bn["running_mean"], dtype=torch.float32).view(1, -1, 1, 1)
    bn_std = torch.sqrt(torch.tensor(bn["running_var"], dtype=torch.float32).view(1, -1, 1, 1) + 1e-4)

    # ---- encode prompt ----
    def encode(prompt):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inp = tokenizer(text, return_tensors="np", padding="max_length",
                        truncation=True, max_length=512)
        out = text_encoder({"input_ids": inp["input_ids"].astype(np.int64),
                            "attention_mask": inp["attention_mask"].astype(np.int64)})
        return torch.tensor(out[0])

    prompt_embeds = encode(args.prompt)
    stats("prompt_embeds", prompt_embeds)
    text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).float()[0]  # (L,4)

    do_cfg = args.guidance > 1.0
    if do_cfg:
        neg_embeds = encode("")
        neg_text_ids = Flux2KleinPipeline._prepare_text_ids(neg_embeds).float()[0]

    # ---- prepare latents ----
    vae_scale_factor = 8
    num_channels_latents = 128 // 4  # 32
    lh = 2 * (args.height // (vae_scale_factor * 2))
    lw = 2 * (args.width // (vae_scale_factor * 2))
    shape = (1, num_channels_latents * 4, lh // 2, lw // 2)
    gen = torch.Generator().manual_seed(args.seed)
    latents4d = torch.randn(shape, generator=gen, dtype=torch.float32)
    latent_ids = Flux2KleinPipeline._prepare_latent_ids(latents4d).float()[0]  # (H*W,4)
    latents = Flux2KleinPipeline._pack_latents(latents4d)  # (1, H*W, 128)
    stats("init_latents", latents)

    # ---- timesteps ----
    image_seq_len = latents.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=args.steps)
    sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
    scheduler.set_timesteps(sigmas=sigmas, mu=mu, device="cpu")
    timesteps = scheduler.timesteps
    print(f"  mu={mu:.5f} sigmas={list(scheduler.sigmas.numpy())}")
    print(f"  timesteps={list(timesteps.numpy())}")

    # ---- denoise ----
    scheduler.set_begin_index(0)
    for i, t in enumerate(timesteps):
        ts = t.expand(latents.shape[0]).float() / 1000.0
        if pt_tr is not None:
            with torch.no_grad():
                noise_pred = pt_tr(hidden_states=latents, timestep=ts, guidance=None,
                                   encoder_hidden_states=prompt_embeds,
                                   txt_ids=text_ids, img_ids=latent_ids, return_dict=False)[0]
            noise_pred = noise_pred[:, : latents.shape[1], :]
        else:
            out = transformer({
                "hidden_states": latents.numpy(),
                "timestep": ts.numpy(),
                "encoder_hidden_states": prompt_embeds.numpy(),
                "txt_ids": text_ids.numpy(),
                "img_ids": latent_ids.numpy(),
            })
            noise_pred = torch.tensor(out[0])[:, : latents.shape[1], :]
        if do_cfg:
            out_n = transformer({
                "hidden_states": latents.numpy(),
                "timestep": ts.numpy(),
                "encoder_hidden_states": neg_embeds.numpy(),
                "txt_ids": neg_text_ids.numpy(),
                "img_ids": latent_ids.numpy(),
            })
            neg = torch.tensor(out_n[0])[:, : latents.shape[1], :]
            noise_pred = neg + args.guidance * (noise_pred - neg)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        stats(f"latents_step{i}", latents)

    # ---- decode ----
    latents = Flux2KleinPipeline._unpack_latents_with_ids(
        latents, latent_ids.unsqueeze(0), lh // 2, lw // 2)
    stats("unpacked", latents)
    latents = latents * bn_std + bn_mean
    stats("bn_denorm", latents)
    latents = Flux2KleinPipeline._unpatchify_latents(latents)
    stats("unpatchified", latents)
    img = vae_decoder({"latent_sample": latents.numpy()})[0]
    img = torch.tensor(img)
    stats("vae_out", img)

    img = (img / 2 + 0.5).clamp(0, 1)
    img = (img[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(img).save(args.output)
    print(f"saved {args.output}  pixel mean={img.mean():.2f} std={img.std():.2f}")


if __name__ == "__main__":
    main()
