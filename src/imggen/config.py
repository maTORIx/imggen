"""Small runtime helpers: HF token discovery and seed generation."""

from __future__ import annotations

import os
import random


def hf_token(explicit: str | None = None) -> str | None:
    """Return an HF token from ``--hf-token`` or the usual environment vars.

    A logged-in ``huggingface-cli login`` credential is still used automatically
    by ``huggingface_hub`` even when this returns ``None``.
    """
    if explicit:
        return explicit
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val
    return None


def make_seeds(seed: int | None, num: int) -> list[int]:
    """Produce ``num`` seeds. If ``seed`` is given, they are consecutive."""
    if seed is None:
        return [random.randint(0, 2**32 - 1) for _ in range(num)]
    return [seed + i for i in range(num)]
