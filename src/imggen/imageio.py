"""Output path handling and image saving with embedded metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".psd"}


def build_output_paths(out: str | None, kind: str, seeds: list[int]) -> list[Path]:
    """Compute one output path per image.

    ``out`` may be:

    * ``None``       -> ``<kind>_<seed>.png`` in the current directory
    * a directory    -> ``<dir>/<kind>_<seed>.png``
    * a file path     -> used as-is for a single image; for multiple images the
      index is inserted before the extension (``foo.png`` -> ``foo_00.png`` ...)
    * a template with ``{i}`` / ``{seed}`` / ``{kind}`` placeholders
    """
    n = len(seeds)

    if out is None:
        return [Path(f"{kind}_{seed}.png") for seed in seeds]

    if any(tok in out for tok in ("{i}", "{seed}", "{kind}")):
        return [
            Path(out.format(i=i, seed=seed, kind=kind)) for i, seed in enumerate(seeds)
        ]

    p = Path(out)
    is_dir = out.endswith(("/", os.sep)) or p.is_dir() or p.suffix.lower() not in _IMG_EXTS
    if is_dir:
        return [p / f"{kind}_{seed}.png" for seed in seeds]

    if n == 1:
        return [p]
    stem, suffix = p.stem, p.suffix
    width = max(2, len(str(n - 1)))
    return [p.with_name(f"{stem}_{i:0{width}d}{suffix}") for i in range(n)]


def save_image(
    image: Image.Image,
    path: Path,
    metadata: dict | None = None,
    embed_metadata: bool = True,
) -> Path:
    """Save ``image`` to ``path``, embedding ``metadata`` for PNG outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext == ".png" and embed_metadata and metadata:
        info = PngInfo()
        info.add_text("imggen", json.dumps(metadata, ensure_ascii=False))
        # A/1111-style "parameters" field for interoperability with viewers.
        info.add_text("parameters", _format_parameters(metadata))
        image.save(path, pnginfo=info)
    elif ext in (".jpg", ".jpeg"):
        image.convert("RGB").save(path, quality=95)
    else:
        image.save(path)
    return path


def _format_parameters(m: dict) -> str:
    parts = [str(m.get("prompt", ""))]
    if m.get("negative"):
        parts.append(f"Negative prompt: {m['negative']}")
    tags = []
    for key in ("steps", "cfg", "seed", "model", "size", "kind"):
        if m.get(key) is not None:
            label = {"cfg": "CFG scale", "steps": "Steps"}.get(key, key.capitalize())
            tags.append(f"{label}: {m[key]}")
    if tags:
        parts.append(", ".join(tags))
    return "\n".join(parts)
