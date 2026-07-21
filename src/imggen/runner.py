"""Glue between the CLI and the backends: run, name, and save results."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from . import pipelines
from .imageio import build_output_paths, save_image
from .params import GenRequest


def _progress_reporter():
    """A terminal progress bar for remote generation (fed by streamed events).

    Local runs show diffusers' own tqdm bar; a remote client sees nothing until
    the daemon streams per-step progress, so we render one here. Returns a
    callable ``report(event)`` with a ``.close()`` that finishes the line. Stays
    silent when stderr is not a TTY (piped/redirected) to avoid log spam.
    """
    stream = sys.stderr
    tty = stream.isatty()
    state = {"active": False}

    def report(evt: dict) -> None:
        if evt.get("event") != "progress":
            return
        step, total = evt.get("step"), evt.get("total")
        imgs, img = evt.get("images") or 1, (evt.get("image") or 0) + 1
        if total:
            frac = min(max(step / total, 0.0), 1.0)
            filled = int(frac * 24)
            label = f"step {step}/{total} [{'#' * filled}{'-' * (24 - filled)}] {int(frac * 100):3d}%"
        else:
            label = f"step {step}"
        if imgs > 1:
            label += f"  (image {img}/{imgs})"
        if tty:
            stream.write(f"\r  {label}\033[K")
            stream.flush()
            state["active"] = True

    def close() -> None:
        if state["active"]:
            stream.write("\n")
            stream.flush()
            state["active"] = False

    report.close = close
    return report


def run_and_save(
    req: GenRequest,
    out: str | None,
    embed_metadata: bool = True,
    use_remote: bool = False,
    preview: bool = True,
) -> list[Path]:
    # A configured remote runs the backend on another host and returns the same
    # (image, meta, hint) tuples; saving stays local. No fallback: remote.run
    # raises RemoteError (propagated to the CLI) if the server is unreachable.
    if use_remote:
        from . import remote

        reporter = _progress_reporter()
        try:
            results = remote.run(req, on_progress=reporter)
        finally:
            reporter.close()
    else:
        results = pipelines.run(req)

    # Group results into base images. A new base starts on a plain result
    # (hint None) or on the foreground of a layer pair (hint "fg"); the
    # background ("bg") shares its foreground's base name.
    base_seeds: list = []
    base_of: list[int] = []
    for _, meta, hint in results:
        if hint in (None, "fg"):
            base_seeds.append(meta.get("seed"))
        base_of.append(len(base_seeds) - 1)

    base_seeds = [s if s is not None else i for i, s in enumerate(base_seeds)]
    base_paths = build_output_paths(out, req.kind, base_seeds)

    saved: list[Path] = []
    for (image, meta, hint), base_idx in zip(results, base_of):
        path = base_paths[base_idx]
        if hint:
            path = path.with_name(f"{path.stem}_{hint}{path.suffix or '.png'}")
        # RGBA needs a format that keeps the alpha channel.
        if image.mode == "RGBA" and path.suffix.lower() not in (".png", ".webp"):
            path = path.with_suffix(".png")
        saved.append(save_image(image, path, meta, embed_metadata))

    # Draw the result inline in graphics-capable terminals (Ghostty/Kitty/iTerm2);
    # a no-op elsewhere. Batches are gridded. Never fails a completed run.
    if preview:
        from . import termimg

        termimg.preview([image for image, _, _ in results])

    return saved


def echo_saved(paths: list[Path]) -> None:
    for p in paths:
        typer.secho(f"  saved {p}", fg=typer.colors.GREEN)
