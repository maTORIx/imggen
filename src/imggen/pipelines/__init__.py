"""Generation backends, one entry point per kind."""

from __future__ import annotations

from ..params import GenRequest


def run(req: GenRequest):
    """Dispatch a request to the backend for ``req.kind``.

    Returns a list of ``(PIL.Image, metadata dict, path_hint)`` tuples, where
    ``path_hint`` is an optional filename suffix (used by see-through to emit
    ``_fg`` / ``_bg`` layers). ``path_hint`` is ``None`` for a plain image.
    """
    if req.kind == "sd":
        from . import sd

        return sd.generate(req)
    if req.kind == "qwen-image":
        from . import qwen

        return qwen.generate(req)
    if req.kind == "qwen-image-edit":
        from . import qwen

        return qwen.generate_edit(req)
    if req.kind == "see-through":
        from . import seethrough

        return seethrough.generate(req)
    raise ValueError(f"unknown kind: {req.kind}")
