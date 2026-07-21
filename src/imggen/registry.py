"""Model resolution for ``--model``.

Every model is described by a **manifest** — a JSON file under
``~/.config/imggen/settings/`` (see :mod:`imggen.manifest`). Built-in models are
seeded there on first use; ``imggen <kind> --save --alias`` adds new ones.
``--model`` accepts these forms:

* a **manifest name** (``sdxl``, ``qwen-image``, ``qwen-image-edit``,
  ``sd3.5`` ...) -> resolved from ``<config>/settings/<kind>/<name>.json``
* a **Hugging Face repo id** (``stabilityai/stable-diffusion-xl-base-1.0``)
* a **local path** to a diffusers folder or single-file checkpoint

The heavy Qwen-Image / Qwen-Image-Edit transformers (20B params) default to a
GGUF quantization (the same community files ComfyUI uses): only the transformer
is quantized, while the text encoder, VAE and scheduler come from the base repo.
The ``*-fp16`` manifests load the full unquantized weights instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Default manifest name per kind, used when ``--model`` is omitted.
DEFAULT_MODEL: dict[str, str] = {
    "sd": "sdxl",
    "qwen-image": "qwen-image",
    "qwen-image-edit": "qwen-image-edit",
    "see-through": "sdxl",  # base generator; matting model is fixed (BiRefNet)
}

# Base diffusers repo used for the non-transformer components (text encoder,
# VAE, scheduler) when a Qwen manifest ships only a single-file transformer and
# does not name a ``load.base_repo`` explicitly.
_QWEN_BASE_REPO: dict[str, str] = {
    "qwen-image": "Qwen/Qwen-Image",
    "qwen-image-edit": "Qwen/Qwen-Image-Edit-2509",
}


@dataclass(frozen=True)
class SingleFileTransformer:
    """A Qwen transformer loaded from a single file (``.safetensors`` or GGUF).

    The transformer is read with ``from_single_file`` and the remaining pipeline
    components come from ``base_repo``. ``is_gguf`` selects the GGUF
    quantization path.
    """

    path: str  # local file path to the transformer weights
    base_repo: str
    is_gguf: bool = False
    config_repo: str | None = None
    config_subfolder: str = "transformer"

    @property
    def config_ref(self) -> str:
        return self.config_repo or self.base_repo


@dataclass
class ResolvedModel:
    ref: str  # repo id or local path passed to from_pretrained
    is_local: bool
    is_gated: bool
    single_file: SingleFileTransformer | None = None  # set by a Qwen manifest
    defaults: dict = field(default_factory=dict)  # manifest generation defaults


def models_info() -> dict:
    """A serializable listing of installed presets plus the per-kind defaults.

    Returns ``{"models": {kind: [{name, summary, gated, description}, ...]},
    "defaults": {kind: name}}``. Shared by the ``imggen models`` command and the
    ``/models`` server endpoint so a remote client renders the *server's* presets
    exactly as it would render local ones (same shape, no manifest objects on the
    wire). Reads manifests only — no torch, so it stays light on the CLI path.
    """
    from . import manifest as mf

    models: dict[str, list[dict]] = {}
    for kind, mans in mf.list_manifests().items():
        models[kind] = [
            {
                "name": m.name,
                "summary": m.summary(),
                "gated": m.gated,
                "description": m.description,
            }
            for m in mans
        ]
    return {"models": models, "defaults": dict(DEFAULT_MODEL)}


def resolve_model(
    kind: str, model_arg: str | None, token: str | None = None
) -> ResolvedModel:
    """Resolve ``--model`` (or the per-kind default) to a concrete reference."""
    arg = model_arg or DEFAULT_MODEL[kind]

    # Local path (diffusers folder or single-file checkpoint) wins if it exists.
    if os.path.exists(arg):
        return ResolvedModel(ref=os.path.abspath(arg), is_local=True, is_gated=False)

    # A manifest at <config>/settings/<kind>/<arg>.json.
    resolved = _resolve_manifest(kind, arg, token)
    if resolved is not None:
        return resolved

    # Otherwise treat ``arg`` as a raw Hugging Face repo id.
    return ResolvedModel(ref=arg, is_local=False, is_gated=False)


def _resolve_manifest(
    kind: str, arg: str, token: str | None
) -> ResolvedModel | None:
    """Resolve ``arg`` against a ``settings/<kind>/<arg>.json`` manifest."""
    from . import manifest as mf

    man = mf.load_manifest(kind, arg)
    if man is None:
        return None

    mat = mf.materialize(man, token)
    defaults = dict(man.defaults)

    # A bare repo folder: load with from_pretrained.
    if mat.is_repo:
        return ResolvedModel(
            ref=mat.ref, is_local=False, is_gated=man.gated, defaults=defaults
        )

    # A single downloaded file. Qwen kinds load it as a transformer paired with
    # a base repo; other kinds treat it as a single-file checkpoint (sd.py).
    if kind in _QWEN_BASE_REPO:
        base_repo = man.load.base_repo or _QWEN_BASE_REPO[kind]
        is_gguf = mat.ref.lower().endswith(".gguf")
        return ResolvedModel(
            ref=base_repo,
            is_local=False,
            is_gated=man.gated,
            single_file=SingleFileTransformer(
                path=mat.ref,
                base_repo=base_repo,
                is_gguf=is_gguf,
                config_repo=man.load.config_repo,
                config_subfolder=man.load.config_subfolder,
            ),
            defaults=defaults,
        )

    return ResolvedModel(
        ref=mat.ref, is_local=True, is_gated=man.gated, defaults=defaults
    )
