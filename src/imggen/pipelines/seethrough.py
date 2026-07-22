"""see-through backend: transparent images and layer decomposition.

Three engines:

* **LayerDiffuse** (``--method layerdiffuse``): native transparent generation
  with a real alpha channel, recovering soft edges (hair, glass, glow). Used by
  default for ``transparent`` mode when generating from a prompt.
* **matting** (``--method matte``): a base image (from ``--init`` or generated
  with the Stable Diffusion backend) is run through a BiRefNet matting model to
  produce the foreground alpha. Used for ``--init`` decomposition and for the
  ``layers`` mode.
* **decompose** (``--method decompose``): full anime part-layer decomposition
  (hair / eyes / mouth / nose / ears / face / clothing, with occluded regions
  inpainted) exported as a single layered ``.psd``. Powered by the See-through
  model (Lin et al., SIGGRAPH 2026), which pins an incompatible diffusers/
  transformers stack, so it runs in an **isolated venv driven as a subprocess**.
  That venv is provisioned automatically on first use (see
  :mod:`imggen.seethrough_env`), like any other first-run download. Works over
  the remote daemon too — the PSD travels as bytes in the result payload — in
  which case it is the *server* that provisions and holds the install.

Modes:

* ``transparent`` (default): one RGBA PNG with the background removed.
* ``layers``: a transparent foreground layer plus an inpainted background layer
  (always matting-based, since it needs the opaque pixels behind the subject).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import typer
from PIL import Image

from ..config import hf_token, make_seeds
from ..params import GenRequest
from . import common, layerdiffuse, sd

_MATTE_SIZE = 1024
# Fixed matting model for background removal / layer decomposition.
_BIREFNET_REPO = "ZhengPeng7/BiRefNet"


def generate(req: GenRequest, on_step=None):
    method = _resolve_method(req)
    if method == "layerdiffuse":
        return layerdiffuse.generate(req, on_step=on_step)
    if method == "decompose":
        return _generate_decompose(req, on_step=on_step)
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


# --- decompose (See-through part-layer -> PSD) ----------------------------
#
# See-through pins diffusers==0.37 / transformers==5.0 and vendors deep
# subclasses of diffusers internals, incompatible with imggen's own stack, so
# it cannot share this process. Instead we drive its ``inference_psd.py`` in a
# separate venv as a subprocess and hand the produced ``.psd`` back to the
# runner. Install location defaults under the imggen cache; override with
# ``$IMGGEN_SEETHROUGH_REPO`` / ``$IMGGEN_SEETHROUGH_PYTHON``.


def _seethrough_env() -> tuple[Path, Path]:
    """Return ``(repo_dir, python_exe)``, provisioning the install if it is absent.

    The isolated venv is imggen's own plumbing, not something the user should
    have to assemble; the first decompose run builds it the same way a first
    ``sd`` run downloads weights. Set ``$IMGGEN_SEETHROUGH_AUTOSETUP=0`` to make
    a missing environment an error instead (for a locked-down host, or when the
    checkout is managed by something else).
    """
    from .. import seethrough_env as st_env

    if st_env.is_ready() or os.environ.get("IMGGEN_SEETHROUGH_AUTOSETUP", "1") != "0":
        return st_env.ensure()
    info = st_env.describe()
    raise RuntimeError(
        "see-through --method decompose needs the isolated See-through install, "
        "and $IMGGEN_SEETHROUGH_AUTOSETUP=0 disables provisioning it.\n"
        f"  expected repo:   {info['repo']}\n"
        f"  expected python: {info['python']}\n"
        "Run `imggen setup see-through` (or unset the variable) to build it."
    )


#: Tags kept as their own PSD layer under ``--parts face``. Everything else —
#: neck, clothing, hands, feet, tail, props — collapses into one ``body`` layer.
#: Word boundaries matter: a bare ``ear`` would otherwise match ``handwear``,
#: ``footwear`` and every other ``*wear`` tag.
_FACE_TAGS_RE = r"\b(?:hair|head\w*|face|eye\w*|ear\w*|nose|mouth|brow\w*|irid\w*|iris)\b"


def _parts_mode(req: GenRequest) -> str:
    """``face`` (split the head only) or ``all`` (every See-through part)."""
    return (req.parts or "face").lower()


def _merge_body_layers(py: Path, repo: Path, psd: Path) -> None:
    """Collapse the non-face layers of *psd* into a single ``body`` layer, in place.

    Runs in the isolated venv: rewriting the PSD needs ``psd_tools`` and
    See-through's own writer, neither of which lives in imggen's interpreter.
    A failure here is not fatal — the full-split PSD is still a usable result —
    so it warns and leaves the file alone.
    """
    script = Path(__file__).resolve().parent.parent / "data" / "seethrough_merge.py"
    cmd = [str(py), str(script), "--psd", str(psd), "--keep-re", _FACE_TAGS_RE]
    proc = subprocess.run(cmd, cwd=str(repo))
    if proc.returncode != 0:
        typer.secho(
            f"warning: merging body layers failed (exit {proc.returncode}); "
            "keeping the full part split. Use --parts all to silence this.",
            fg=typer.colors.YELLOW,
        )


def _generate_decompose(req: GenRequest, on_step=None):
    """Decompose a character into semantic part layers, exported as one PSD.

    The base image is either ``--init`` or freshly generated with the ``sd``
    backend; See-through then splits it into up to ~23 inpainted RGBA layers
    (hair, eyes, mouth, nose, ears, face, clothing) ordered by inferred depth.
    By default (``--parts face``) only the head stays split and everything from
    the neck down is merged back into a single ``body`` layer, which is what
    character editing actually wants; ``--parts all`` keeps every part.
    Returns a single result whose ``meta['_psd_path']`` the runner moves to the
    user's ``--out`` as a ``.psd`` (the preview image is the base, for the inline
    terminal preview only — the layers live in the PSD). Under ``imggen serve``
    that path is server-local, so the server ships the bytes instead and the
    client re-stashes them (see :func:`imggen.server._encode_result`).
    """
    repo, py = _seethrough_env()
    seed = make_seeds(req.seed, 1)[0]
    if req.num > 1:
        typer.secho("note: decompose emits one PSD per run; ignoring --num > 1", fg=typer.colors.YELLOW)

    if req.init:
        base = common.load_init_image(req.init).convert("RGB")
        base_meta = {"prompt": req.prompt, "model": req.init}
    else:
        sub = replace(req, kind="sd", mode="transparent", method="auto", num=1, batch_size=1)
        img, meta, _ = sd.generate(sub, on_step=on_step)[0]
        base = img.convert("RGB")
        base_meta = {"prompt": meta.get("prompt"), "model": meta.get("model")}

    resolution = int(req.width or req.height or 1280)

    with tempfile.TemporaryDirectory(prefix="imggen-st-") as td:
        tdp = Path(td)
        in_path = tdp / "input.png"
        base.save(in_path)
        out_dir = tdp / "out"
        cmd = [
            str(py), "inference/scripts/inference_psd.py",
            "--srcp", str(in_path),
            "--save_dir", str(out_dir),
            "--save_to_psd",
            "--resolution", str(resolution),
            "--seed", str(seed),
        ]
        typer.secho(
            f"decomposing into part layers via See-through (~several min, res {resolution}) ...",
            fg=typer.colors.CYAN,
        )
        # Inherit stdout/stderr so the user sees See-through's own progress bars.
        proc = subprocess.run(cmd, cwd=str(repo))
        if proc.returncode != 0:
            raise RuntimeError(
                f"See-through decomposition failed (exit {proc.returncode}); see the output above"
            )
        psd_src = out_dir / "input.psd"
        if not psd_src.exists():
            raise RuntimeError(f"See-through produced no PSD (expected {psd_src})")
        if _parts_mode(req) == "face":
            _merge_body_layers(py, repo, psd_src)
        # The sidecar's ``parts`` maps 1:1 to the PSD's layers (both come from the
        # same dict), so it is the accurate layer count — unlike the per-part PNG
        # dir, which also holds pre-split/merged fragments.
        n_layers = None
        psd_json = out_dir / "input.psd.json"
        if psd_json.exists():
            import json as _json

            try:
                n_layers = len(_json.loads(psd_json.read_text()).get("parts", {}))
            except Exception:
                n_layers = None
        # Copy the PSD out before the temp dir is cleaned; the runner moves this
        # stashed file to --out (and thereby removes it), so nothing leaks.
        fd, stash = tempfile.mkstemp(prefix="imggen-st-", suffix=".psd")
        os.close(fd)
        shutil.copyfile(psd_src, stash)

    if n_layers:
        typer.secho(f"  {n_layers} layers", fg=typer.colors.GREEN)
    meta = {
        "kind": req.kind,
        "mode": "decompose",
        "method": "decompose",
        "prompt": base_meta.get("prompt"),
        "model": base_meta.get("model"),
        "seed": seed,
        "size": f"{base.width}x{base.height}",
        "resolution": resolution,
        "parts": _parts_mode(req),
        "_psd_path": stash,
    }
    if n_layers:
        meta["layers"] = n_layers
    return [(base, meta, None)]
