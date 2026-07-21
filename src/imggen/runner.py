"""Glue between the CLI and the backends: run, name, and save results."""

from __future__ import annotations

from pathlib import Path

import typer

from . import pipelines
from .imageio import build_output_paths, save_image
from .params import GenRequest


def run_and_save(
    req: GenRequest,
    out: str | None,
    embed_metadata: bool = True,
    use_remote: bool = False,
) -> list[Path]:
    # A configured remote runs the backend on another host and returns the same
    # (image, meta, hint) tuples; saving stays local. No fallback: remote.run
    # raises RemoteError (propagated to the CLI) if the server is unreachable.
    if use_remote:
        from . import remote

        results = remote.run(req)
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

    return saved


def echo_saved(paths: list[Path]) -> None:
    for p in paths:
        typer.secho(f"  saved {p}", fg=typer.colors.GREEN)
