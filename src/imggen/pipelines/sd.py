"""Stable Diffusion backend (SD1.5 / SDXL / SD3.5, auto-detected).

Uses diffusers' ``AutoPipeline`` classes, which read ``model_index.json`` to
pick the correct pipeline for the checkpoint. When ``--init`` is given we run
image-to-image instead of text-to-image.
"""

from __future__ import annotations

import os

from ..config import hf_token
from ..params import GenRequest
from ..registry import resolve_model
from . import common

DEFAULTS = {"steps": 30, "cfg": 7.0}

# diffusers' ``infer_diffusers_model_type`` result -> concrete pipeline class
# names for (text2img, img2img). ``AutoPipeline*`` has no ``from_single_file``
# (it needs a repo's ``model_index.json``), so a raw ``.safetensors`` / ``.ckpt``
# must be matched to a concrete SD / SDXL / SD3 class.
_SINGLE_FILE_PIPELINES = {
    "xl_base": ("StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline"),
    "xl_refiner": ("StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline"),
    "xl_inpaint": ("StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline"),
    "playground-v2-5": ("StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline"),
    "v1": ("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline"),
    "v2": ("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline"),
    "inpainting": ("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline"),
    "inpainting_v2": ("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline"),
    "sd3": ("StableDiffusion3Pipeline", "StableDiffusion3Img2ImgPipeline"),
    "sd35_large": ("StableDiffusion3Pipeline", "StableDiffusion3Img2ImgPipeline"),
    "sd35_medium": ("StableDiffusion3Pipeline", "StableDiffusion3Img2ImgPipeline"),
}


def _single_file_pipeline_cls(path: str, img2img: bool):
    """Pick the concrete pipeline class for a single-file checkpoint.

    Reads the checkpoint header (safetensors is mmapped, so this is cheap), lets
    diffusers detect the architecture, and maps it to a concrete SD / SDXL / SD3
    pipeline. Raises a clear error for architectures this backend doesn't handle
    (Flux, SANA, video models, ...).
    """
    import diffusers
    from diffusers.loaders.single_file_utils import (
        infer_diffusers_model_type,
        load_single_file_checkpoint,
    )

    checkpoint = load_single_file_checkpoint(path)
    model_type = infer_diffusers_model_type(checkpoint)
    del checkpoint

    mapping = _SINGLE_FILE_PIPELINES.get(model_type)
    if mapping is None:
        raise ValueError(
            f"unsupported single-file checkpoint {os.path.basename(path)!r} "
            f"(diffusers detected model type {model_type!r}); the sd backend "
            f"loads SD1.5 / SD2 / SDXL / SD3.5 single files"
        )
    return getattr(diffusers, mapping[1] if img2img else mapping[0])


def _build(resolved, dtype, token, img2img: bool):
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

    kwargs = dict(torch_dtype=dtype, token=token)
    if resolved.is_local and resolved.ref.endswith((".safetensors", ".ckpt")):
        cls = _single_file_pipeline_cls(resolved.ref, img2img)
        return cls.from_single_file(resolved.ref, **kwargs)
    cls = AutoPipelineForImage2Image if img2img else AutoPipelineForText2Image
    return cls.from_pretrained(resolved.ref, **kwargs)


def generate(req: GenRequest, on_step=None):
    device, dtype, seeds = common.prepare(req)
    img2img = req.init is not None
    token = hf_token(req.hf_token)
    resolved = resolve_model(req.kind, req.model, token)
    key = ("sd", resolved.ref, img2img, str(dtype), device, bool(req.offload))
    pipe = common.cached_pipeline(
        key, lambda: common.place(_build(resolved, dtype, token, img2img), device, req.offload)
    )

    d = resolved.defaults
    steps = common.setting(req.steps, "steps", d, DEFAULTS["steps"])
    cfg = common.setting(req.cfg, "cfg", d, DEFAULTS["cfg"])
    negative = common.setting(req.negative, "negative", d, None)
    prefix = common.setting(req.prompt_prefix, "prompt_prefix", d, None)
    suffix = common.setting(req.prompt_suffix, "prompt_suffix", d, None)
    prompt = common.compose_prompt(req.prompt, prefix, suffix)
    width = common.setting(req.width, "width", d, None)
    height = common.setting(req.height, "height", d, None)
    strength = common.setting(req.strength, "strength", d, 0.8)
    sampler = common.setting(req.sampler, "sampler", d, None)
    common.set_scheduler(pipe, sampler)
    gens = common.generators(seeds, device)

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=cfg,
    )
    # ComfyUI/A1111 `(word:1.2)` weighting: swap in weighted embeds when markup is
    # present (SD1.5/SDXL/SD3.5). A no-op — kwargs keep prompt/negative — otherwise.
    weighted = common.apply_weighting(pipe, kwargs, prompt, negative)
    if img2img:
        kwargs["image"] = common.load_init_image(req.init)
        kwargs["strength"] = strength
    else:
        if width:
            kwargs["width"] = width
        if height:
            kwargs["height"] = height

    results = []
    for base, seed_chunk, gen_chunk in common.batches(seeds, gens, req.batch_size):
        call_kwargs = dict(kwargs)
        call_kwargs["num_images_per_prompt"] = len(seed_chunk)
        common.attach_progress(call_kwargs, pipe, on_step, base, len(seeds))
        images = pipe(generator=gen_chunk, **call_kwargs).images
        for seed, image in zip(seed_chunk, images):
            meta = {
                "kind": req.kind,
                "prompt": prompt,
                "negative": negative,
                "model": resolved.ref,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "size": f"{image.width}x{image.height}",
            }
            if sampler:
                meta["sampler"] = sampler
            if weighted:
                meta["weighted"] = True
            if img2img:
                meta["init"] = req.init
                meta["strength"] = strength
            results.append((image, meta, None))
    return results
