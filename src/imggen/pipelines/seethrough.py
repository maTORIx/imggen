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
  transformers stack, so it runs in an **isolated venv driven as a subprocess**
  (see :func:`_seethrough_env`) — local execution only, never the remote daemon.

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


def _seethrough_home() -> Path:
    root = os.environ.get("IMGGEN_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "imggen"
    return base / "seethrough"


def _seethrough_env() -> tuple[Path, Path]:
    """Return ``(repo_dir, python_exe)`` for the isolated See-through install."""
    home = _seethrough_home()
    repo = Path(os.environ.get("IMGGEN_SEETHROUGH_REPO") or home / "repo")
    py = Path(os.environ.get("IMGGEN_SEETHROUGH_PYTHON") or home / "venv" / "bin" / "python")
    script = repo / "inference" / "scripts" / "inference_psd.py"
    if not script.exists() or not py.exists():
        raise RuntimeError(
            "see-through --method decompose needs the isolated See-through install.\n"
            f"  expected repo:   {repo}\n"
            f"  expected python: {py}\n"
            "Set $IMGGEN_SEETHROUGH_REPO / $IMGGEN_SEETHROUGH_PYTHON to point at a\n"
            "See-through checkout (https://github.com/shitagaki-lab/see-through) and\n"
            "its inference venv."
        )
    return repo, py


def _generate_decompose(req: GenRequest, on_step=None):
    """Decompose a character into semantic part layers, exported as one PSD.

    The base image is either ``--init`` or freshly generated with the ``sd``
    backend; See-through then splits it into up to ~23 inpainted RGBA layers
    (hair, eyes, mouth, nose, ears, face, clothing) ordered by inferred depth.
    Returns a single result whose ``meta['_psd_path']`` the runner moves to the
    user's ``--out`` as a ``.psd`` (the preview image is the base, for the inline
    terminal preview only — the layers live in the PSD).
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
        "_psd_path": stash,
    }
    if n_layers:
        meta["layers"] = n_layers
    return [(base, meta, None)]
