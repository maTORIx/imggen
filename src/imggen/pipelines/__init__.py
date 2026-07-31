"""Generation backends, one entry point per kind."""

from __future__ import annotations

from ..params import GenRequest


def run(req: GenRequest, on_step=None):
    """Dispatch a request to the backend for ``req.kind``.

    Returns a list of ``(PIL.Image, metadata dict, path_hint)`` tuples, where
    ``path_hint`` is an optional filename suffix (used by see-through to emit
    ``_fg`` / ``_bg`` layers). ``path_hint`` is ``None`` for a plain image.

    ``on_step(img_index, num_images, step, total)`` is an optional progress
    callback fired at the end of each denoising step; the ``imggen serve`` daemon
    passes one to stream progress to a remote client. It is *not* part of
    :class:`GenRequest` (it cannot cross the wire) — local runs leave it ``None``
    and rely on diffusers' own tqdm bar.
    """
    if req.kind == "sd":
        from . import sd

        return sd.generate(req, on_step=on_step)
    if req.kind == "qwen-image":
        from . import qwen

        return qwen.generate(req, on_step=on_step)
    if req.kind == "qwen-image-edit":
        from . import qwen

        return qwen.generate_edit(req, on_step=on_step)
    if req.kind == "see-through":
        from . import seethrough

        return seethrough.generate(req, on_step=on_step)
    if req.kind == "background-removal":
        from . import bgremove

        return bgremove.generate(req, on_step=on_step)
    raise ValueError(f"unknown kind: {req.kind}")
