"""Helpers shared across generation backends."""

from __future__ import annotations

import os
import re

import torch
import typer
from PIL import Image

from ..config import make_seeds
from ..device import resolve_device, resolve_dtype
from ..params import GenRequest


def prepare(req: GenRequest):
    """Resolve device, dtype and per-image seeds/generators for a request."""
    device = resolve_device(req.device)
    dtype = resolve_dtype(device, req.dtype)
    seeds = make_seeds(req.seed, req.num)
    return device, dtype, seeds


# --- warm pipeline cache (server mode) ----------------------------------
# A one-shot CLI run loads a model once and exits, so caching would be pointless
# there and is disabled by default. ``imggen serve`` sets IMGGEN_PIPELINE_CACHE
# to keep the last-used, already-placed pipeline resident (LRU=1) so repeated
# requests for the same model skip the (tens-of-seconds) load. A request for a
# different model unloads the previous pipeline and frees CUDA memory first.
_PIPE_CACHE: dict = {}


def cache_enabled() -> bool:
    return os.environ.get("IMGGEN_PIPELINE_CACHE", "").lower() not in ("", "0", "false", "no")


def cached_pipeline(key, build):
    """Return the warm pipeline for ``key``, building it (evicting any other) on miss.

    ``build`` must return a fully-placed pipeline (i.e. wrap :func:`place`), since
    it runs only on a cache miss. The scheduler and per-call kwargs are applied by
    the caller on every request, so only the load+placement is cached.
    """
    if not cache_enabled():
        return build()
    cached = _PIPE_CACHE.get(key)
    if cached is not None:
        return cached
    if _PIPE_CACHE:  # LRU=1: drop the resident pipeline before loading another
        _PIPE_CACHE.clear()
        _free_memory()
    pipe = build()
    _PIPE_CACHE[key] = pipe
    return pipe


def _free_memory() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generators(seeds: list[int], device: str) -> list[torch.Generator]:
    # CUDA generators are fine; for MPS fall back to CPU generators.
    gen_device = "cpu" if device == "mps" else device
    return [torch.Generator(device=gen_device).manual_seed(s) for s in seeds]


def batches(seeds: list[int], gens: list, batch_size: int | None):
    """Split per-image seeds/generators into forward-pass batches.

    Yields ``(base_index, seed_chunk, gen_chunk)`` so a backend can run
    ``len(chunk)`` images in a single ``pipe()`` call (``num_images_per_prompt``)
    while still giving each image its own consecutive seed via ``gen_chunk`` (a
    list of that many generators). ``batch_size`` is clamped to at least 1, so the
    default (``1``) yields one image per batch — identical to the old per-image
    loop (diffusers treats a single-element generator list the same as one
    generator, so the plain path stays bit-for-bit).
    """
    step = max(int(batch_size or 1), 1)
    for base in range(0, len(seeds), step):
        yield base, seeds[base:base + step], gens[base:base + step]


def place(pipe, device: str, offload: bool):
    """Move a pipeline onto the device, or enable CPU offload.

    On unified-memory hosts (e.g. GB10) where a full-precision model would push
    the CUDA pool near the physical RAM ceiling, set ``IMGGEN_SEQUENTIAL_OFFLOAD``
    to keep weights in CPU RAM and stream one submodule to the GPU at a time —
    slower, but a much lower peak than model-granular offload.
    """
    if device == "cuda" and os.environ.get("IMGGEN_SEQUENTIAL_OFFLOAD"):
        pipe.enable_sequential_cpu_offload()
    elif offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    # Memory-friendly defaults; harmless when unsupported.
    for meth in ("enable_vae_tiling", "enable_attention_slicing"):
        fn = getattr(pipe, meth, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
    return pipe


def set_scheduler(pipe, name: str | None) -> None:
    """Swap ``pipe.scheduler`` to the sampler named ``name`` (``--sampler``).

    A no-op when ``name`` is falsy. Unknown names and schedulers incompatible
    with the loaded pipeline (notably classic samplers on a flow-matching Qwen /
    SD3.5 model) warn and keep the pipeline's default scheduler rather than
    aborting the run.
    """
    if not name:
        return
    from .. import schedulers

    spec = schedulers.resolve(name)
    if spec is None:
        typer.secho(
            f"unknown sampler '{name}'; keeping the model default "
            f"(known: {', '.join(schedulers.NAMES)})",
            fg=typer.colors.YELLOW,
        )
        return
    cls_name, extra = spec
    import diffusers

    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        typer.secho(
            f"sampler '{name}' needs {cls_name}, unavailable in this diffusers "
            f"build; keeping the model default",
            fg=typer.colors.YELLOW,
        )
        return
    try:
        pipe.scheduler = cls.from_config(pipe.scheduler.config, **extra)
        typer.secho(f"sampler: {name} ({cls_name})", fg=typer.colors.BLUE)
    except Exception as exc:  # incompatible config (e.g. flow-matching model)
        typer.secho(
            f"sampler '{name}' is incompatible with this model ({exc}); "
            f"keeping the model default",
            fg=typer.colors.YELLOW,
        )


def load_init_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# --- region-locked editing (--mask) -------------------------------------
#
# A mask answers one question: which pixels is this run allowed to change?
# White = editable, black = keep. Two things make it actually hold:
#
# 1. Every masked path ends with :func:`composite_masked`, which restores the
#    untouched pixels *in pixel space*. Latent-level masking alone is not
#    enough — the VAE round-trip shifts "unchanged" pixels by a few levels
#    across the whole canvas, which is exactly what stops one expression patch
#    from being reusable across outfit variants.
# 2. Backends whose model re-frames the subject (Qwen-Image-Edit rescales and
#    shifts the head by a few px even when told not to) edit a *crop* of the
#    mask region instead of the full canvas, so the geometry is pinned by the
#    crop box rather than by the model's goodwill.

def load_mask(path: str, size=None, grow: int = 0, blur: float = 0.0) -> Image.Image:
    """Load an ``L`` mask, optionally resized to *size*, grown and feathered.

    ``grow`` dilates (positive) or erodes (negative) by that many pixels before
    feathering — handy to give a garment room to spill past the silhouette, or
    to pull a face mask in off the hairline.
    """
    mask = Image.open(path).convert("L")
    if size is not None and mask.size != tuple(size):
        mask = mask.resize(tuple(size), Image.LANCZOS)
    if grow:
        import cv2  # rank filters in PIL are O(r^2) per pixel; cv2 is separable
        import numpy as np

        r = abs(int(grow))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
        op = cv2.dilate if grow > 0 else cv2.erode
        mask = Image.fromarray(op(np.asarray(mask), kernel))
    if blur:
        from PIL import ImageFilter

        mask = mask.filter(ImageFilter.GaussianBlur(float(blur)))
    return mask


def composite_masked(original: Image.Image, result: Image.Image, mask: Image.Image) -> Image.Image:
    """Blend *result* over *original* through *mask*, in pixel space.

    Where the mask is black the output is *original* byte-for-byte, which is the
    whole point: it is what lets a face patch made against one variant be
    dropped onto another without the rest of the image drifting.
    """
    base = original.convert("RGB")
    top = result.convert("RGB")
    if top.size != base.size:
        top = top.resize(base.size, Image.LANCZOS)
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.LANCZOS)
    return Image.composite(top, base, mask)


#: Padding around a mask's bounding box when a backend crops to it. Generous on
#: purpose: the model needs the whole head in frame to draw a coherent
#: expression, even though only the masked pixels of what it returns are kept.
#: ``imggen parts masks`` reports the box computed with this same value, so the
#: two must not drift apart.
MASK_CROP_PAD = 0.6


def mask_bbox(mask: Image.Image, pad: float = MASK_CROP_PAD, align: int = 8, threshold: int = 8):
    """Square-ish crop box around the white region of *mask*, padded by *pad*.

    Returned as a PIL ``(left, top, right, bottom)`` clamped to the mask bounds,
    with each side snapped to a multiple of *align*. Raises when the mask is
    empty — a mask with nothing in it is a mistake worth reporting, not a
    silently-skipped edit.
    """
    box = mask.point(lambda v: 255 if v > threshold else 0).getbbox()
    if box is None:
        raise ValueError("mask is empty (no pixels above the threshold)")
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) / 2 * (1.0 + max(pad, 0.0))
    w, h = mask.size
    left = max(0, int(cx - half) // align * align)
    top = max(0, int(cy - half) // align * align)
    right = min(w, -(-int(cx + half) // align) * align)
    bottom = min(h, -(-int(cy + half) // align) * align)
    return left, top, right, bottom


def mask_settings(req: GenRequest, defaults: dict, grow: int = 0, blur: float = 4.0):
    """Resolve ``--mask-grow`` / ``--mask-blur`` through the usual precedence."""
    return (
        int(setting(req.mask_grow, "mask_grow", defaults, grow)),
        float(setting(req.mask_blur, "mask_blur", defaults, blur)),
    )


def setting(cli_value, key: str, defaults: dict, fallback):
    """Resolve one generation setting by precedence.

    Explicit CLI value (non-``None``) wins, then the model manifest ``defaults``,
    then the pipeline ``fallback``.
    """
    if cli_value is not None:
        return cli_value
    if key in defaults and defaults[key] is not None:
        return defaults[key]
    return fallback


def compose_prompt(prompt, prefix=None, suffix=None):
    """Wrap the positive prompt with a preset's template tags.

    ``prompt_prefix`` / ``prompt_suffix`` come from a model's manifest
    ``defaults`` (or ``--prompt-prefix`` / ``--prompt-suffix``) and carry the
    model's recommended quality boilerplate — e.g. Pony's ``score_9,
    score_8_up, ...`` that belongs *before* the user's description, or trailing
    style tags. Segments are joined with ``", "`` and stray leading/trailing
    commas trimmed. A no-op when neither template is set or the prompt is empty.
    """
    if not prompt or (not prefix and not suffix):
        return prompt
    segments = []
    for part in (prefix, prompt, suffix):
        if not part:
            continue
        part = part.strip().strip(",").strip()
        if part:
            segments.append(part)
    return ", ".join(segments)


# --- prompt weighting (ComfyUI / A1111 `(word:1.2)` emphasis) ------------
# An *unescaped* opening bracket is the only way to introduce a weighted group,
# so its presence is what flips a prompt onto the weighted-embedding path. Plain
# prompts skip it entirely and stay bit-for-bit on the normal encoder path.
_WEIGHT_OPEN = re.compile(r"(?<!\\)[(\[]")


def has_weight_markup(text: str | None) -> bool:
    """True if *text* uses ComfyUI/A1111 weighting markup — ``(w:1.2)``, ``(w)``, ``[w]``."""
    return bool(text) and _WEIGHT_OPEN.search(text) is not None


def strip_weight_markup(text: str | None) -> str | None:
    """Return *text* with weighting markup removed, keeping the words.

    ``(word:1.3)`` -> ``word``, ``[word]`` -> ``word``, ``\\(`` -> literal ``(``.
    Used by backends whose text encoder is an LLM (Qwen), where token-level
    weighting has no standard meaning: the markup is dropped and a note printed.
    """
    if not has_weight_markup(text):
        return text
    from ..vendor.sd_embed.prompt_parser import parse_prompt_attention

    parts = [seg for seg, _w in parse_prompt_attention(text) if seg != "BREAK"]
    return "".join(parts).strip()


def weighted_embeddings(pipe, prompt: str | None, negative: str | None):
    """Compute weighted prompt embeddings for a CLIP-based Stable Diffusion pipe.

    Returns a kwargs dict to pass to the pipeline instead of ``prompt`` /
    ``negative_prompt`` (``prompt_embeds`` + ``negative_prompt_embeds``, plus the
    pooled variants for SDXL / SD3), or ``None`` if the pipeline family has no
    supported weighting path (the caller then falls back to the plain prompt).

    The embeddings are produced with the pipeline's own text encoders (via the
    vendored ``sd_embed``), so device placement / CPU-offload hooks are honoured.
    """
    from ..vendor.sd_embed import embedding_funcs as ef

    name = type(pipe).__name__
    pos, neg = prompt or "", negative or ""
    if "StableDiffusion3" in name:
        pe, ne, pp, npp = ef.get_weighted_text_embeddings_sd3(pipe, pos, neg)
        return dict(
            prompt_embeds=pe, negative_prompt_embeds=ne,
            pooled_prompt_embeds=pp, negative_pooled_prompt_embeds=npp,
        )
    if "StableDiffusionXL" in name:
        pe, ne, pp, npp = ef.get_weighted_text_embeddings_sdxl(pipe, pos, neg)
        return dict(
            prompt_embeds=pe, negative_prompt_embeds=ne,
            pooled_prompt_embeds=pp, negative_pooled_prompt_embeds=npp,
        )
    if name.startswith("StableDiffusion"):  # SD1.5 / SD2
        pe, ne = ef.get_weighted_text_embeddings_sd15(pipe, pos, neg)
        return dict(prompt_embeds=pe, negative_prompt_embeds=ne)
    return None


def apply_weighting(pipe, kwargs: dict, prompt: str | None, negative: str | None) -> bool:
    """If *prompt*/*negative* use weighting markup, swap embeds into *kwargs*.

    Mutates *kwargs* in place: drops ``prompt`` / ``negative_prompt`` and adds the
    weighted embedding tensors. Returns ``True`` when weighting was applied. A
    no-op (returns ``False``) for plain prompts or pipelines without a supported
    weighting path — so callers keep the ordinary prompt kwargs untouched.
    """
    if not (has_weight_markup(prompt) or has_weight_markup(negative)):
        return False
    embeds = weighted_embeddings(pipe, prompt, negative)
    if embeds is None:
        return False
    kwargs.pop("prompt", None)
    kwargs.pop("negative_prompt", None)
    kwargs.update(embeds)
    return True


# --- per-step progress (server -> remote client) -------------------------

def _step_callback(on_step, img_index: int, num_images: int):
    """Adapt ``on_step(img, imgs, step, total)`` to a diffusers step callback."""
    state = {"total": None}

    def _cb(pipe, step, timestep, cb_kwargs):
        if state["total"] is None:
            try:
                state["total"] = len(pipe.scheduler.timesteps)
            except Exception:
                state["total"] = None
        try:
            on_step(img_index, num_images, step + 1, state["total"])
        except Exception:
            pass
        return cb_kwargs

    return _cb


def attach_progress(kwargs: dict, pipe, on_step, img_index: int, num_images: int) -> None:
    """Add a ``callback_on_step_end`` to *kwargs* (in place) when progress is wanted.

    ``on_step(img_index, num_images, step, total)`` fires at the end of each
    denoising step. No-op when *on_step* is ``None`` or the pipeline predates the
    ``callback_on_step_end`` API. The server uses this to stream progress to a
    remote client; local runs pass ``on_step=None`` and rely on diffusers' tqdm.
    """
    if on_step is None:
        return
    import inspect

    try:
        supported = "callback_on_step_end" in inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        supported = False
    if supported:
        kwargs["callback_on_step_end"] = _step_callback(on_step, img_index, num_images)
