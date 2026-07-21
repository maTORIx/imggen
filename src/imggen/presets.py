"""Saving CLI options as reusable model presets (``--save --alias``).

``imggen <kind> --model <src> --steps N --sampler S ... --save --alias NAME``
writes ``<config>/settings/<kind>/NAME.json`` — a normal manifest (see
:mod:`imggen.manifest`) so it is recalled later with ``--model NAME``.

The manifest ``source`` is inferred from ``--model``:

* omitted            -> the kind's built-in default model (inherit its source + defaults)
* an existing preset -> that preset's ``source`` / ``load`` / ``defaults`` are inherited
* an ``http(s)://`` URL -> ``{"url": ...}`` (an HF ``resolve`` URL covers repo + file + subfolder)
* an ``org/repo`` id -> ``{"hf_repo": ...}`` (a full diffusers repo folder)

Only the generation flags the user *typed* are recorded on top of the inherited
defaults, so a preset captures exactly the tuning that was asked for.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import manifest as mf

# Generation flags that may be recorded into a preset's ``defaults``.
SAVEABLE = (
    "steps", "cfg", "width", "height", "negative", "strength", "sampler",
    "prompt_prefix", "prompt_suffix",
)

_REPO_ID = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")


def _base_from_model(kind: str, model_arg: str | None) -> tuple[dict, dict | None, dict]:
    """Infer ``(source, load, base_defaults)`` for a preset from ``--model``."""
    from .registry import DEFAULT_MODEL

    if not model_arg:
        model_arg = DEFAULT_MODEL.get(kind)
        if not model_arg:
            raise ValueError(f"no default model for kind '{kind}'; pass --model")

    # An existing preset/manifest: inherit its source, load and defaults.
    path = mf.manifest_path(kind, model_arg)
    if path is not None:
        data = json.loads(path.read_text())
        source = data.get("source")
        if not source:
            raise ValueError(f"preset '{model_arg}' has no source to inherit")
        return source, data.get("load"), dict(data.get("defaults") or {})

    if model_arg.startswith(("http://", "https://")):
        return {"url": model_arg}, None, {}

    if os.path.exists(model_arg):
        raise ValueError(
            f"cannot save a preset for the local path '{model_arg}': local files "
            "are not portable. Recall it directly with --model <path>, or pass an "
            "HF repo id / URL to --model when saving."
        )

    if _REPO_ID.match(model_arg):
        return {"hf_repo": model_arg}, None, {}

    raise ValueError(
        f"cannot infer a model source from --model '{model_arg}'; expected a "
        "preset name, an HF repo id (org/repo), or an http(s) URL"
    )


def save_preset(
    kind: str,
    model_arg: str | None,
    overrides: dict,
    alias: str,
    description: str | None = None,
) -> tuple[Path, bool, dict]:
    """Write ``<config>/settings/<kind>/<alias>.json``.

    ``overrides`` are the explicitly-typed generation flags (a subset of
    :data:`SAVEABLE`); they are layered over any inherited base defaults.
    Returns ``(path, already_existed, merged_defaults)``.
    """
    source, load, base_defaults = _base_from_model(kind, model_arg)
    defaults = {**base_defaults, **{k: v for k, v in overrides.items() if v is not None}}

    doc: dict = {}
    if description:
        doc["description"] = description
    doc["source"] = source
    if load:
        doc["load"] = load
    if defaults:
        doc["defaults"] = defaults

    path = mf.user_settings_dir() / kind / f"{alias}.json"
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    return path, existed, defaults
