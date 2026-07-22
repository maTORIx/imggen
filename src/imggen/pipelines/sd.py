"""Stable Diffusion backend (SD1.5 / SDXL / SD3.5, auto-detected).

Uses diffusers' ``AutoPipeline`` classes, which read ``model_index.json`` to
pick the correct pipeline for the checkpoint. When ``--init`` is given we run
image-to-image instead of text-to-image; adding ``--mask`` switches to
inpainting, redrawing only the white region and restoring everything else from
the original in pixel space (see :func:`common.composite_masked`).
"""

from __future__ import annotations

import os

from ..config import hf_token
from ..params import GenRequest
from ..registry import resolve_model
from . import common

DEFAULTS = {"steps": 30, "cfg": 7.0}

# diffusers' ``infer_diffusers_model_type`` result -> concrete pipeline class
# names per task. ``AutoPipeline*`` has no ``from_single_file`` (it needs a
# repo's ``model_index.json``), so a raw ``.safetensors`` / ``.ckpt`` must be
# matched to a concrete SD / SDXL / SD3 class.
#
# The inpaint entries are the *ordinary* checkpoints' inpaint pipelines, not
# inpainting-specific models: diffusers' inpaint pipelines detect a 4-channel
# UNet and fall back to masked latent blending (what ComfyUI's
# SetLatentNoiseMask does), which is what makes an anime SDXL checkpoint usable
# for clothing swaps without a dedicated inpainting variant.
_SINGLE_FILE_PIPELINES = {
    "xl_base": ("StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLInpaintPipeline"),
    "xl_refiner": ("StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLInpaintPipeline"),
    "xl_inpaint": ("StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLInpaintPipeline"),
    "playground-v2-5": ("StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLInpaintPipeline"),
    "v1": ("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline", "StableDiffusionInpaintPipeline"),
    "v2": ("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline", "StableDiffusionInpaintPipeline"),
    "inpainting": ("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline", "StableDiffusionInpaintPipeline"),
    "inpainting_v2": ("StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline", "StableDiffusionInpaintPipeline"),
    "sd3": ("StableDiffusion3Pipeline", "StableDiffusion3Img2ImgPipeline", "StableDiffusion3InpaintPipeline"),
    "sd35_large": ("StableDiffusion3Pipeline", "StableDiffusion3Img2ImgPipeline", "StableDiffusion3InpaintPipeline"),
    "sd35_medium": ("StableDiffusion3Pipeline", "StableDiffusion3Img2ImgPipeline", "StableDiffusion3InpaintPipeline"),
}

#: Index into the tuples above, by task.
_TASK_SLOT = {"text2img": 0, "img2img": 1, "inpaint": 2}


def _single_file_pipeline_cls(path: str, task: str):
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
    return getattr(diffusers, mapping[_TASK_SLOT[task]])


def _build(resolved, dtype, token, task: str):
    from diffusers import (
        AutoPipelineForImage2Image,
        AutoPipelineForInpainting,
        AutoPipelineForText2Image,
    )

    kwargs = dict(torch_dtype=dtype, token=token)
    if resolved.is_local and resolved.ref.endswith((".safetensors", ".ckpt")):
        cls = _single_file_pipeline_cls(resolved.ref, task)
        return cls.from_single_file(resolved.ref, **kwargs)
    cls = {
        "text2img": AutoPipelineForText2Image,
        "img2img": AutoPipelineForImage2Image,
        "inpaint": AutoPipelineForInpainting,
    }[task]
    return cls.from_pretrained(resolved.ref, **kwargs)


def _task(req: GenRequest) -> str:
    if req.init is None:
        if req.mask:
            raise ValueError("--mask needs an input image to edit (--init/-i)")
        return "text2img"
    return "inpaint" if req.mask else "img2img"


def generate(req: GenRequest, on_step=None):
    device, dtype, seeds = common.prepare(req)
    task = _task(req)
    img2img = task != "text2img"
    token = hf_token(req.hf_token)
    resolved = resolve_model(req.kind, req.model, token)
    key = ("sd", resolved.ref, task, str(dtype), device, bool(req.offload))
    pipe = common.cached_pipeline(
        key, lambda: common.place(_build(resolved, dtype, token, task), device, req.offload)
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
    # Inpainting defaults to a full redraw inside the mask: the region is being
    # replaced (bare skin -> a garment), not nudged, and anything outside it is
    # restored from the original anyway.
    strength = common.setting(req.strength, "strength", d, 1.0 if task == "inpaint" else 0.8)
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
    init_image = mask = None
    if img2img:
        init_image = common.load_init_image(req.init)
        kwargs["image"] = init_image
        kwargs["strength"] = strength
    else:
        if width:
            kwargs["width"] = width
        if height:
            kwargs["height"] = height
    if task == "inpaint":
        grow, blur = common.mask_settings(req, d)
        mask = common.load_mask(req.mask, init_image.size, grow=grow, blur=blur)
        kwargs["mask_image"] = mask
        # Match the init image exactly; the default would re-derive a size from
        # the pipeline config and the composite back would then be a resize.
        kwargs["width"], kwargs["height"] = init_image.size

    results = []
    for base, seed_chunk, gen_chunk in common.batches(seeds, gens, req.batch_size):
        call_kwargs = dict(kwargs)
        call_kwargs["num_images_per_prompt"] = len(seed_chunk)
        common.attach_progress(call_kwargs, pipe, on_step, base, len(seeds))
        images = pipe(generator=gen_chunk, **call_kwargs).images
        if task == "inpaint":
            # The pipeline round-trips the whole canvas through the VAE, so even
            # "untouched" pixels come back a few levels off. Put the original
            # back everywhere the mask is black — that byte-exactness is what
            # lets separately-made layers (a face patch, a garment) be reused
            # across variants of the same base image.
            images = [common.composite_masked(init_image, img, mask) for img in images]
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
            if task == "inpaint":
                meta["mask"] = req.mask
            results.append((image, meta, None))
    return results
