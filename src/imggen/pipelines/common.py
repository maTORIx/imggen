"""Helpers shared across generation backends."""

from __future__ import annotations

import os

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


def generators(seeds: list[int], device: str) -> list[torch.Generator]:
    # CUDA generators are fine; for MPS fall back to CPU generators.
    gen_device = "cpu" if device == "mps" else device
    return [torch.Generator(device=gen_device).manual_seed(s) for s in seeds]


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
