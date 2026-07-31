"""background-removal backend: one image in, its subject cut out.

A single forward pass through a matting network — no prompt, no seed, no
denoising steps. ``--mode transparent`` (default) returns the input as RGBA with
the predicted alpha; ``--mode mask`` returns the alpha matte on its own, which is
what feeds ``--mask`` elsewhere in imggen.

Three presets ship with it, and the *pre/post-processing belongs to the model's
architecture*, not to imggen — mixing them up produces a plausible-looking but
wrong matte, so the recipe is picked from the loaded model's class
(:func:`_family`) rather than from the preset name:

* ``lucida`` (default, ``egeorcun/lucida``, MIT) — a BiRefNet-HR fine-tune aimed
  at soft alpha: hair, glass, glow, text/logos and illustrations.
* ``birefnet-hr`` (``ZhengPeng7/BiRefNet_HR``, MIT) — the high-resolution
  BiRefNet Lucida is derived from. Same recipe, but its detail only shows up at
  its native **2048 px**, which is what its preset asks for.
* ``rmbg-1.4`` (``briaai/RMBG-1.4``) — BRIA's U²-Net-derived model. A different
  architecture: different input normalization, a different output shape, plus a
  min-max stretch upstream applies and the BiRefNet family does not. Its weights
  are non-commercial (bria-rmbg-1.4 license).

``--width``/``--height`` set the resolution the *model* runs at (its input is
always resized to a square-ish fixed size and the alpha resized back), so they
trade edge detail against memory; they never change the output size, which is
always the input image's.

Related: ``see-through --method matte`` also removes a background (with
BiRefNet), but as one stage of transparent generation / layer splitting. This
kind is the standalone tool, and the one to add matting models to.
"""

from __future__ import annotations

import numpy as np
import torch
import typer
from PIL import Image

from ..config import hf_token
from ..device import resolve_device, resolve_dtype
from ..params import GenRequest
from . import common

# Fallback model input size when neither the preset nor the CLI names one.
DEFAULTS = {"width": 1024, "height": 1024}

#: Model classes that use BRIA's U²-Net recipe. Everything else is treated as
#: BiRefNet-family — which covers Lucida, BiRefNet, BiRefNet-HR and RMBG-2.0
#: (2.0 is a BiRefNet despite the name, so match on the class, not the repo).
_RMBG_CLASSES = ("BriaRMBG",)

#: ``(mean, std)`` used to normalize the model input, per family.
_NORM = {
    "birefnet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet
    "rmbg": ([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
}


def generate(req: GenRequest, on_step=None):
    """Return ``[(image, metadata, None)]`` — the cutout (or the matte).

    ``on_step`` is accepted for the backend contract and ignored: there are no
    denoising steps to report, so a remote client sees only the server's single
    "running" status line.
    """
    from ..registry import resolve_model

    if not req.init:
        raise ValueError("background-removal needs an input image (--init/-i)")

    token = hf_token(req.hf_token)
    device = resolve_device(req.device)
    # Matting is one cheap forward pass, so it runs in fp32 by default (bf16
    # batch-norm statistics visibly grain the matte). An explicit --dtype still
    # wins — worth it for BiRefNet-HR at 2048 px on a smaller card.
    dtype = resolve_dtype(device, req.dtype) if req.dtype else torch.float32

    resolved = resolve_model("background-removal", req.model, token)
    defaults = resolved.defaults
    size = (
        int(common.setting(req.width, "width", defaults, DEFAULTS["width"])),
        int(common.setting(req.height, "height", defaults, DEFAULTS["height"])),
    )

    img = common.load_init_image(req.init)
    model = _load_model(resolved.ref, device, dtype, token)
    family = _family(model)
    typer.secho(
        f"matting: {resolved.ref} ({family}) at {size[0]}x{size[1]}",
        fg=typer.colors.BLUE,
    )

    alpha = matte(model, img, size, device, dtype)
    meta = {
        "kind": req.kind,
        "mode": req.mode,
        "model": resolved.ref,
        "family": family,
        "init": req.init,
        "size": f"{img.width}x{img.height}",
        "matte_size": f"{size[0]}x{size[1]}",
    }
    if req.mode == "mask":
        return [(alpha, meta, None)]
    return [(_apply_alpha(img, alpha), meta, None)]


# --- model loading ------------------------------------------------------


def _load_model(ref: str, device: str, dtype, token: str | None):
    """Load a matting model (warm-cached under ``imggen serve``)."""
    key = ("background-removal", ref, device, str(dtype))
    return common.cached_pipeline(key, lambda: _build_model(ref, device, dtype, token))


def _build_model(ref: str, device: str, dtype, token: str | None):
    """Instantiate the model from ``ref`` (an HF repo id or a local folder).

    Every model here ships its architecture as repository code, so loading goes
    through ``trust_remote_code``. We resolve the class ourselves — exactly what
    the ``AutoModelForImageSegmentation`` factory does internally — because
    RMBG-1.4's code was written for transformers 4.38 and never calls
    ``post_init()``; transformers >= 5 then trips over the missing
    ``all_tied_weights_keys`` while loading the checkpoint. Supplying the empty
    default it would have received keeps that model loadable without pinning an
    old transformers (the BiRefNet-family code is unaffected, and the shim is a
    no-op there).
    """
    from transformers import AutoConfig, AutoModelForImageSegmentation

    config = AutoConfig.from_pretrained(ref, trust_remote_code=True, token=token)
    class_ref = (getattr(config, "auto_map", None) or {}).get("AutoModelForImageSegmentation")
    if class_ref:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module(class_ref, ref, token=token)
        if not hasattr(cls, "all_tied_weights_keys"):
            cls.all_tied_weights_keys = {}
        model = cls.from_pretrained(ref, config=config, token=token)
    else:  # an ordinary transformers segmentation model
        model = AutoModelForImageSegmentation.from_pretrained(
            ref, trust_remote_code=True, token=token
        )
    return model.to(device=device, dtype=dtype).eval()


def _family(model) -> str:
    """Which pre/post-processing recipe the loaded model needs.

    Keyed off the model class, the same way :func:`common.weighted_embeddings`
    picks a prompt-weighting path from the pipeline class.
    """
    name = type(model).__name__
    return "rmbg" if name in _RMBG_CLASSES else "birefnet"


# --- inference ----------------------------------------------------------


@torch.no_grad()
def matte(model, img: Image.Image, size, device: str, dtype) -> Image.Image:
    """Return the subject's alpha for *img* as an ``L`` mask of the same size."""
    import torchvision.transforms as T

    family = _family(model)
    mean, std = _NORM[family]
    # Resizing the PIL image (antialiased) rather than the raw tensor the way
    # BRIA's own script does: same recipe, slightly cleaner downscale.
    transform = T.Compose(
        [T.Resize((size[1], size[0])), T.ToTensor(), T.Normalize(mean, std)]
    )
    x = transform(img.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)

    out = model(x)
    if family == "rmbg":
        # BriaRMBG returns ``([sigmoid(d1..d6)], [features])``: d1 is the
        # full-resolution side output and is already a probability map. BRIA's
        # postprocess then min-max stretches it, so we do too.
        pred = _stretch(out[0][0])
    else:
        # BiRefNet and its fine-tunes return a list of logit maps, finest last.
        pred = out[-1].sigmoid()

    arr = (pred[0, 0].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize(img.size, Image.BILINEAR)


def _stretch(pred: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a prediction to 0..1 (BRIA's ``postprocess_image``)."""
    lo, hi = pred.min(), pred.max()
    if hi <= lo:
        return torch.zeros_like(pred)
    return (pred - lo) / (hi - lo)


def _apply_alpha(img: Image.Image, alpha: Image.Image) -> Image.Image:
    rgba = img.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba
