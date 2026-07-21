"""Helpers shared across generation backends."""

from __future__ import annotations

import os

import torch
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
