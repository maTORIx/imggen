"""LayerDiffuse: native transparent image generation (SDXL).

Generates true RGBA images with a real alpha channel using latent-transparency
(lllyasviel's LayerDiffuse). A transparency LoRA is merged onto the SDXL UNet
and a special ``TransparentVAEDecoder`` turns the latents into RGBA pixels,
recovering soft/semi-transparent edges (hair, glass, glow) that background
matting cannot. The pure-diffusers port by rootonchair is vendored under
``imggen.vendor.layer_diffuse`` (MIT).

Only SDXL-family base models are supported (the LoRA and transparent decoder
are SDXL-specific).
"""

from __future__ import annotations

import torch
import typer

from ..config import hf_token
from ..params import GenRequest
from ..registry import resolve_model
from . import common

# LayerDiffuse assets (fp16). The VAE fp16-fix is required for stable decoding.
_VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"
_DECODER_REPO = "LayerDiffusion/layerdiffusion-v1"
_DECODER_FILE = "vae_transparent_decoder.safetensors"
_LORA_REPO = "rootonchair/diffuser_layerdiffuse"
_LORA_FILE = "diffuser_layer_xl_transparent_attn.safetensors"

DEFAULTS = {"steps": 30, "cfg": 7.0}


def _build_pipeline(model_ref: str, token: str | None):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from diffusers import StableDiffusionXLPipeline

    from ..vendor.layer_diffuse.models.modules import TransparentVAEDecoder

    dtype = torch.float16  # LayerDiffuse weights are trained/released in fp16.

    typer.echo("loading LayerDiffuse transparent VAE decoder")
    vae = TransparentVAEDecoder.from_pretrained(_VAE_REPO, torch_dtype=dtype)
    decoder_sd = load_file(hf_hub_download(_DECODER_REPO, _DECODER_FILE, token=token))
    vae.set_transparent_decoder(decoder_sd)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_ref,
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    typer.echo(f"merging transparency LoRA {_LORA_REPO}/{_LORA_FILE}")
    pipe.load_lora_weights(_LORA_REPO, weight_name=_LORA_FILE)
    return pipe


def generate(req: GenRequest, on_step=None):
    if not req.prompt:
        raise ValueError("see-through --method layerdiffuse requires a --prompt")

    device, _, seeds = common.prepare(req)
    resolved = resolve_model("see-through", req.model)

    pipe = _build_pipeline(resolved.ref, hf_token(req.hf_token))
    # Custom transparent decoder: skip the vae tiling/slicing from common.place.
    if req.offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    steps = req.steps or DEFAULTS["steps"]
    cfg = req.cfg if req.cfg is not None else DEFAULTS["cfg"]
    prompt = common.compose_prompt(req.prompt, req.prompt_prefix, req.prompt_suffix)
    negative = req.negative or ""
    common.set_scheduler(pipe, req.sampler)
    gens = common.generators(seeds, device)

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative,
        width=req.width or 1024,
        height=req.height or 1024,
        num_inference_steps=steps,
        guidance_scale=cfg,
        # Always 1: the transparent VAE decoder conditions per-sample on the
        # latents, so its condition batch must match the sample batch (see
        # vendor/layer_diffuse/loaders.py). `--batch-size` is therefore not
        # applied here; layerdiffuse stays one image per forward pass.
        num_images_per_prompt=1,
        return_dict=False,
    )
    # SDXL-based, so `(word:1.2)` weighting works here too.
    common.apply_weighting(pipe, kwargs, prompt, negative)

    results = []
    for i, (seed, gen) in enumerate(zip(seeds, gens)):
        call_kwargs = dict(kwargs)
        common.attach_progress(call_kwargs, pipe, on_step, i, len(seeds))
        images = pipe(generator=gen, **call_kwargs)[0]
        image = images[0].convert("RGBA")
        results.append(
            (
                image,
                {
                    "kind": req.kind,
                    "mode": "transparent",
                    "method": "layerdiffuse",
                    "prompt": prompt,
                    "negative": req.negative,
                    "model": resolved.ref,
                    "steps": steps,
                    "cfg": cfg,
                    "seed": seed,
                    "size": f"{image.width}x{image.height}",
                    **({"sampler": req.sampler} if req.sampler else {}),
                },
                None,
            )
        )
    return results
