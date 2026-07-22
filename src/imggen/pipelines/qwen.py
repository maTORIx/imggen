"""Qwen-Image and Qwen-Image-Edit backends.

By default these use GGUF-quantized transformers (the community conversions
ComfyUI uses): only the 20B transformer is quantized, while the text encoder,
VAE and scheduler come from the base repo at full precision. Pass a
``-fp16`` alias (or a plain repo id) to use unquantized weights instead.

Qwen's diffusion pipelines use ``true_cfg_scale`` rather than the classic
``guidance_scale``, so ``--cfg`` is mapped onto that.
"""

from __future__ import annotations

import typer

from ..config import hf_token
from ..params import GenRequest
from ..registry import ResolvedModel, resolve_model
from . import common

DEFAULTS = {"steps": 50, "cfg": 4.0}

#: Long side (px) the masked crop is worked at when ``--width``/``--height`` are
#: not given. Qwen-Image-Edit was trained around ~1M-pixel inputs, so a small
#: face crop has to be scaled up or it comes back soft.
_MASK_WORK_PX = 768

# All-in-one (ComfyUI) checkpoints bundle the transformer under this prefix
# alongside the VAE and text encoder in the same ``.safetensors`` file.
_AIO_PREFIX = "model.diffusion_model."


def _safetensors_keys(path) -> list[str]:
    """Tensor names in a ``.safetensors`` file (header only, no weights read)."""
    import json
    import struct

    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return [k for k in header if k != "__metadata__"]


def _load_transformer(path, is_gguf, dtype, config_ref, subfolder):
    """Load a ``QwenImageTransformer2DModel`` from a single file.

    Handles three shapes: GGUF-quantized files (a ``GGUFQuantizationConfig`` is
    applied), all-in-one ComfyUI checkpoints (transformer under a
    ``model.diffusion_model.`` prefix -> loaded directly, see
    :func:`_load_aio_transformer`), and plain ``.safetensors`` transformers in
    diffusers naming (``from_single_file``). The config comes from
    ``config_ref``/``subfolder``.
    """
    from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel

    if not is_gguf and str(path).lower().endswith(".safetensors"):
        if any(k.startswith(_AIO_PREFIX) for k in _safetensors_keys(path)):
            return _load_aio_transformer(path, dtype, config_ref, subfolder)

    kwargs = dict(torch_dtype=dtype, config=config_ref, subfolder=subfolder)
    if is_gguf:
        kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=dtype)
    return QwenImageTransformer2DModel.from_single_file(path, **kwargs)


def _load_aio_transformer(path, dtype, config_ref, subfolder):
    """Load the transformer from an all-in-one (ComfyUI) checkpoint.

    These files store the transformer under a ``model.diffusion_model.`` prefix
    together with the VAE and text encoder. diffusers' single-file converter
    does not recognise that prefix — it leaves every transformer weight on the
    meta device, which then fails with "Cannot copy out of meta tensor". With
    the prefix stripped, the keys match the diffusers ``QwenImageTransformer2DModel``
    checkpoint 1:1, so we hand the stripped dict to ``from_single_file`` (which
    also accepts a state dict) and let diffusers build the model normally —
    including the non-persistent complex RoPE tables that a manual meta load
    would leave uninitialised. Only the transformer keys are kept; the bundled
    VAE / text-encoder weights come from ``config_ref``.
    """
    from diffusers import QwenImageTransformer2DModel
    from safetensors import safe_open

    state = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(_AIO_PREFIX):
                state[key[len(_AIO_PREFIX):]] = f.get_tensor(key)
    state.pop("__index_timestep_zero__", None)  # AIO sentinel, not a real weight

    return QwenImageTransformer2DModel.from_single_file(
        state, config=config_ref, subfolder=subfolder, torch_dtype=dtype
    )


def _build_pipeline(pipeline_cls, resolved: ResolvedModel, dtype, token):
    """Build a Qwen pipeline from a single-file transformer or a full repo."""
    if resolved.single_file is not None:
        sf = resolved.single_file
        label = "GGUF" if sf.is_gguf else "single-file"
        typer.echo(f"loading {label} transformer {sf.path}")
        transformer = _load_transformer(
            sf.path, sf.is_gguf, dtype, sf.config_ref, sf.config_subfolder
        )
        return pipeline_cls.from_pretrained(
            sf.base_repo, transformer=transformer, torch_dtype=dtype, token=token
        )
    return pipeline_cls.from_pretrained(resolved.ref, torch_dtype=dtype, token=token)


def _model_meta(resolved: ResolvedModel) -> dict:
    if resolved.single_file is not None:
        return {"model": resolved.single_file.base_repo, "checkpoint": resolved.single_file.path}
    return {"model": resolved.ref}


def _cache_key(kind: str, resolved: ResolvedModel, dtype, device: str, offload: bool):
    """Warm-cache key: the checkpoint file distinguishes GGUF vs fp16 variants."""
    tag = resolved.single_file.path if resolved.single_file is not None else resolved.ref
    return (kind, resolved.ref, tag, str(dtype), device, bool(offload))


def _strip_weights(prompt, negative):
    """Drop ComfyUI/A1111 `(word:weight)` markup for the (LLM-encoder) Qwen path.

    Token-level emphasis has no standard meaning for Qwen's Qwen2.5-VL text
    encoder, so rather than feed the literal parentheses to the model we strip
    the markup and warn (matching the documented no-weighting behaviour).
    """
    if common.has_weight_markup(prompt) or common.has_weight_markup(negative):
        typer.secho(
            "note: (word:weight) emphasis is unsupported for qwen (LLM text "
            "encoder); generating without weights",
            fg=typer.colors.YELLOW,
        )
        prompt = common.strip_weight_markup(prompt)
        negative = common.strip_weight_markup(negative)
    return prompt, negative


def generate(req: GenRequest, on_step=None):
    from diffusers import QwenImagePipeline

    device, dtype, seeds = common.prepare(req)
    token = hf_token(req.hf_token)
    resolved = resolve_model(req.kind, req.model, token)
    key = _cache_key("qwen-image", resolved, dtype, device, req.offload)
    pipe = common.cached_pipeline(
        key,
        lambda: common.place(
            _build_pipeline(QwenImagePipeline, resolved, dtype, token), device, req.offload
        ),
    )

    d = resolved.defaults
    steps = common.setting(req.steps, "steps", d, DEFAULTS["steps"])
    cfg = common.setting(req.cfg, "cfg", d, DEFAULTS["cfg"])
    negative = common.setting(req.negative, "negative", d, " ")
    prefix = common.setting(req.prompt_prefix, "prompt_prefix", d, None)
    suffix = common.setting(req.prompt_suffix, "prompt_suffix", d, None)
    prompt = common.compose_prompt(req.prompt, prefix, suffix)
    prompt, negative = _strip_weights(prompt, negative)
    width = common.setting(req.width, "width", d, None)
    height = common.setting(req.height, "height", d, None)
    sampler = common.setting(req.sampler, "sampler", d, None)
    common.set_scheduler(pipe, sampler)
    gens = common.generators(seeds, device)

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative or " ",
        num_inference_steps=steps,
        true_cfg_scale=cfg,
    )
    if width:
        kwargs["width"] = width
    if height:
        kwargs["height"] = height

    model_meta = _model_meta(resolved)
    results = []
    for base, seed_chunk, gen_chunk in common.batches(seeds, gens, req.batch_size):
        call_kwargs = dict(kwargs)
        call_kwargs["num_images_per_prompt"] = len(seed_chunk)
        common.attach_progress(call_kwargs, pipe, on_step, base, len(seeds))
        images = pipe(generator=gen_chunk, **call_kwargs).images
        for seed, image in zip(seed_chunk, images):
            results.append(
                (
                    image,
                    {
                        "kind": req.kind,
                        "prompt": prompt,
                        "negative": negative,
                        **model_meta,
                        "steps": steps,
                        "cfg": cfg,
                        "seed": seed,
                        "size": f"{image.width}x{image.height}",
                        **({"sampler": sampler} if sampler else {}),
                    },
                    None,
                )
            )
    return results


def _work_size(crop_size, width, height, align: int = 16):
    """Resolution to run a masked crop at, snapped to *align*.

    ``--width``/``--height`` win when given; otherwise the crop is scaled so its
    long side is :data:`_MASK_WORK_PX`.
    """
    cw, ch = crop_size
    if width or height:
        w = width or round(cw * height / ch)
        h = height or round(ch * width / cw)
    else:
        scale = _MASK_WORK_PX / max(cw, ch)
        w, h = round(cw * scale), round(ch * scale)
    snap = lambda v: max(align, int(round(v / align)) * align)  # noqa: E731
    return snap(w), snap(h)


def _masked_crop(req: GenRequest, init_image, defaults, width, height):
    """Set up the region-locked edit: ``(mask, box, work_image, work_size)``.

    Qwen-Image-Edit re-frames whatever it is given — on a full-body image it
    moves the head by several pixels and rescales it a couple of percent, which
    is enough to make an expression patch built against the base body unusable
    on an outfit variant. Editing a fixed crop instead pins the geometry to the
    crop box, and the surrounding canvas never reaches the model at all.
    """
    grow, blur = common.mask_settings(req, defaults)
    mask = common.load_mask(req.mask, init_image.size, grow=grow, blur=blur)
    box = common.mask_bbox(mask, pad=common.MASK_CROP_PAD, align=16)
    crop = init_image.crop(box)
    size = _work_size(crop.size, width, height)
    from PIL import Image

    return mask, box, crop.resize(size, Image.LANCZOS), size


def generate_edit(req: GenRequest, on_step=None):
    from diffusers import QwenImageEditPlusPipeline

    if not req.init:
        raise ValueError("qwen-image-edit requires an input image (--init/-i)")

    device, dtype, seeds = common.prepare(req)
    token = hf_token(req.hf_token)
    resolved = resolve_model(req.kind, req.model, token)
    key = _cache_key("qwen-image-edit", resolved, dtype, device, req.offload)
    pipe = common.cached_pipeline(
        key,
        lambda: common.place(
            _build_pipeline(QwenImageEditPlusPipeline, resolved, dtype, token), device, req.offload
        ),
    )

    d = resolved.defaults
    steps = common.setting(req.steps, "steps", d, DEFAULTS["steps"])
    cfg = common.setting(req.cfg, "cfg", d, DEFAULTS["cfg"])
    negative = common.setting(req.negative, "negative", d, " ")
    prefix = common.setting(req.prompt_prefix, "prompt_prefix", d, None)
    suffix = common.setting(req.prompt_suffix, "prompt_suffix", d, None)
    prompt = common.compose_prompt(req.prompt, prefix, suffix)
    prompt, negative = _strip_weights(prompt, negative)
    width = common.setting(req.width, "width", d, None)
    height = common.setting(req.height, "height", d, None)
    sampler = common.setting(req.sampler, "sampler", d, None)
    common.set_scheduler(pipe, sampler)
    init_image = common.load_init_image(req.init)
    gens = common.generators(seeds, device)

    mask = box = None
    if req.mask:
        mask, box, edit_image, (width, height) = _masked_crop(req, init_image, d, width, height)
        typer.secho(
            f"masked edit: crop {box[2] - box[0]}x{box[3] - box[1]} at "
            f"({box[0]},{box[1]}) -> working at {width}x{height}",
            fg=typer.colors.BLUE,
        )
        # The crop is what pins the geometry. Once it covers most of the canvas
        # there is nothing left to pin against and the model is as free to
        # re-frame inside it as it would be on the full image.
        area = (box[2] - box[0]) * (box[3] - box[1]) / (init_image.width * init_image.height)
        if area > 0.8:
            typer.secho(
                f"note: the mask spans {area * 100:.0f}% of the image, so the crop "
                "barely constrains the framing — expect the drift an unmasked edit "
                "has (the mask still decides which pixels are kept).",
                fg=typer.colors.YELLOW,
            )
    else:
        edit_image = init_image

    kwargs = dict(
        image=edit_image,
        prompt=prompt,
        negative_prompt=negative or " ",
        num_inference_steps=steps,
        true_cfg_scale=cfg,
    )
    # Explicit output resolution; when unset the pipeline derives it from the
    # input image (Qwen-Image-Edit auto-picks a size close to the init).
    if width:
        kwargs["width"] = width
    if height:
        kwargs["height"] = height

    model_meta = _model_meta(resolved)
    results = []
    for base, seed_chunk, gen_chunk in common.batches(seeds, gens, req.batch_size):
        call_kwargs = dict(kwargs)
        call_kwargs["num_images_per_prompt"] = len(seed_chunk)
        common.attach_progress(call_kwargs, pipe, on_step, base, len(seeds))
        images = pipe(generator=gen_chunk, **call_kwargs).images
        if mask is not None:
            images = [_paste_back(init_image, img, box, mask) for img in images]
        for seed, image in zip(seed_chunk, images):
            results.append(
                (
                    image,
                    {
                        "kind": req.kind,
                        "prompt": prompt,
                        "negative": negative,
                        **model_meta,
                        "init": req.init,
                        "steps": steps,
                        "cfg": cfg,
                        "seed": seed,
                        "size": f"{image.width}x{image.height}",
                        **({"sampler": sampler} if sampler else {}),
                        **({"mask": req.mask} if mask is not None else {}),
                    },
                    None,
                )
            )
    return results


def _paste_back(init_image, edited, box, mask):
    """Scale *edited* back into *box* on the original canvas and blend by *mask*."""
    from PIL import Image

    crop_size = (box[2] - box[0], box[3] - box[1])
    patched = init_image.copy()
    patched.paste(edited.convert("RGB").resize(crop_size, Image.LANCZOS), box[:2])
    return common.composite_masked(init_image, patched, mask)
