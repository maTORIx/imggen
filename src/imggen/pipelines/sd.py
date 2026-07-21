"""Stable Diffusion backend (SD1.5 / SDXL / SD3.5, auto-detected).

Uses diffusers' ``AutoPipeline`` classes, which read ``model_index.json`` to
pick the correct pipeline for the checkpoint. When ``--init`` is given we run
image-to-image instead of text-to-image.
"""

from __future__ import annotations

from ..config import hf_token
from ..params import GenRequest
from ..registry import resolve_model
from . import common

DEFAULTS = {"steps": 30, "cfg": 7.0}


def _load(req: GenRequest, dtype, img2img: bool):
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image

    token = hf_token(req.hf_token)
    resolved = resolve_model(req.kind, req.model, token)
    cls = AutoPipelineForImage2Image if img2img else AutoPipelineForText2Image
    kwargs = dict(torch_dtype=dtype, token=token)
    if resolved.is_local and resolved.ref.endswith((".safetensors", ".ckpt")):
        return cls.from_single_file(resolved.ref, **kwargs), resolved
    return cls.from_pretrained(resolved.ref, **kwargs), resolved


def generate(req: GenRequest):
    device, dtype, seeds = common.prepare(req)
    img2img = req.init is not None
    pipe, resolved = _load(req, dtype, img2img)
    pipe = common.place(pipe, device, req.offload)

    d = resolved.defaults
    steps = common.setting(req.steps, "steps", d, DEFAULTS["steps"])
    cfg = common.setting(req.cfg, "cfg", d, DEFAULTS["cfg"])
    negative = common.setting(req.negative, "negative", d, None)
    width = common.setting(req.width, "width", d, None)
    height = common.setting(req.height, "height", d, None)
    gens = common.generators(seeds, device)

    kwargs = dict(
        prompt=req.prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=cfg,
    )
    if img2img:
        kwargs["image"] = common.load_init_image(req.init)
        kwargs["strength"] = req.strength
    else:
        if width:
            kwargs["width"] = width
        if height:
            kwargs["height"] = height

    results = []
    for seed, gen in zip(seeds, gens):
        image = pipe(generator=gen, **kwargs).images[0]
        meta = {
            "kind": req.kind,
            "prompt": req.prompt,
            "negative": negative,
            "model": resolved.ref,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "size": f"{image.width}x{image.height}",
        }
        if img2img:
            meta["init"] = req.init
            meta["strength"] = req.strength
        results.append((image, meta, None))
    return results
