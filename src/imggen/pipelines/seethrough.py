"""see-through backend: transparent images and layer decomposition.

Two engines:

* **LayerDiffuse** (``--method layerdiffuse``): native transparent generation
  with a real alpha channel, recovering soft edges (hair, glass, glow). Used by
  default for ``transparent`` mode when generating from a prompt.
* **matting** (``--method matte``): a base image (from ``--init`` or generated
  with the Stable Diffusion backend) is run through a BiRefNet matting model to
  produce the foreground alpha. Used for ``--init`` decomposition and for the
  ``layers`` mode.

Modes:

* ``transparent`` (default): one RGBA PNG with the background removed.
* ``layers``: a transparent foreground layer plus an inpainted background layer
  (always matting-based, since it needs the opaque pixels behind the subject).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
import typer
from PIL import Image

from ..config import hf_token
from ..params import GenRequest
from . import common, layerdiffuse, sd

_MATTE_SIZE = 1024
# Fixed matting model for background removal / layer decomposition.
_BIREFNET_REPO = "ZhengPeng7/BiRefNet"


def generate(req: GenRequest, on_step=None):
    method = _resolve_method(req)
    if method == "layerdiffuse":
        return layerdiffuse.generate(req, on_step=on_step)
    return _generate_matte(req, on_step=on_step)


def _resolve_method(req: GenRequest) -> str:
    """Pick the engine: explicit ``--method`` wins, else auto-select."""
    method = req.method or "auto"
    if method == "auto":
        # LayerDiffuse only makes sense when generating a fresh transparent
        # image; decomposing an existing image or building layers needs matting.
        method = "matte" if (req.init or req.mode == "layers") else "layerdiffuse"
    if method == "layerdiffuse" and req.mode == "layers":
        typer.secho(
            "note: layers mode uses matting (LayerDiffuse has no background layer)",
            fg=typer.colors.YELLOW,
        )
        method = "matte"
    if method == "layerdiffuse" and req.init:
        typer.secho(
            "note: --init decomposition uses matting (falling back from layerdiffuse)",
            fg=typer.colors.YELLOW,
        )
        method = "matte"
    return method


def _generate_matte(req: GenRequest, on_step=None):
    device, _, _ = common.prepare(req)
    bases = _base_images(req, on_step=on_step)
    matte_model = _load_birefnet(device, req)

    results = []
    for img, base_meta in bases:
        alpha = _matte(matte_model, img, device)
        meta = {**base_meta, "kind": req.kind, "mode": req.mode}

        if req.mode == "layers":
            fg = _apply_alpha(img, alpha)
            bg = _inpaint_background(img, alpha)
            results.append((fg, {**meta, "layer": "foreground"}, "fg"))
            results.append((bg, {**meta, "layer": "background"}, "bg"))
        else:
            results.append((_apply_alpha(img, alpha), meta, None))
    return results


def _base_images(req: GenRequest, on_step=None):
    """Return a list of ``(PIL.Image, metadata)`` base images."""
    if req.init:
        img = common.load_init_image(req.init)
        return [(img, {"prompt": req.prompt, "model": req.init, "seed": None})]

    # Generate with the SD backend (see-through's default model is sdxl). This is
    # the only step-based stage, so it carries the progress callback; matting is a
    # single forward pass with nothing to report.
    sub = replace(req, kind="sd", mode="transparent")
    return [(img, meta) for img, meta, _ in sd.generate(sub, on_step=on_step)]


def _load_birefnet(device: str, req: GenRequest):
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        _BIREFNET_REPO, trust_remote_code=True, token=hf_token(req.hf_token)
    )
    # Matting runs in float32 for numerical stability; it is cheap.
    model.to(device=device, dtype=torch.float32).eval()
    return model


@torch.no_grad()
def _matte(model, img: Image.Image, device: str) -> Image.Image:
    """Return an ``L`` alpha mask the same size as ``img``."""
    import torchvision.transforms as T

    transform = T.Compose(
        [
            T.Resize((_MATTE_SIZE, _MATTE_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    x = transform(img.convert("RGB")).unsqueeze(0).to(device)
    preds = model(x)[-1].sigmoid().cpu()[0].squeeze()
    mask = Image.fromarray((preds.numpy() * 255).astype(np.uint8)).resize(img.size)
    return mask


def _apply_alpha(img: Image.Image, alpha: Image.Image) -> Image.Image:
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def _inpaint_background(img: Image.Image, alpha: Image.Image) -> Image.Image:
    """Fill the foreground region so the background layer has no hole."""
    import cv2

    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    mask = np.array(alpha)
    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    binary = cv2.dilate(binary, np.ones((7, 7), np.uint8), iterations=2)
    filled = cv2.inpaint(bgr, binary, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB))
