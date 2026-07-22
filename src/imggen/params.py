"""Shared request parameters passed from the CLI to the pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenRequest:
    kind: str
    prompt: str | None = None
    negative: str | None = None
    prompt_prefix: str | None = None  # positive-prompt template prepended to prompt
    prompt_suffix: str | None = None  # positive-prompt template appended to prompt
    model: str | None = None

    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg: float | None = None
    strength: float | None = None  # img2img / edit denoising strength (default 0.8)
    sampler: str | None = None  # scheduler alias, e.g. euler / ddim / flowmatch

    seed: int | None = None
    num: int = 1
    batch_size: int = 1  # images per forward pass (num_images_per_prompt); each still gets its own seed

    init: str | None = None  # input image path (img2img / edit / see-through)

    # region-locked editing: only the white pixels of `mask` may change
    mask: str | None = None  # mask image path (white = editable, black = keep)
    mask_grow: int | None = None  # dilate (+) / erode (-) the mask, px
    mask_blur: float | None = None  # feather the mask edge, px

    # see-through specific
    mode: str = "transparent"  # "transparent" | "layers"
    method: str = "auto"  # "auto" | "layerdiffuse" | "matte" | "decompose"
    parts: str = "face"  # --method decompose: "face" (head split, body merged) | "all"

    device: str | None = None
    dtype: str | None = None
    offload: bool = False
    hf_token: str | None = None

    extra: dict = field(default_factory=dict)
