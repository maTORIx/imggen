"""imggen command-line interface.

Usage: ``imggen <kind> --prompt "..." [options]`` where kind is one of
``sd``, ``qwen-image``, ``qwen-image-edit``, ``see-through``.
"""

from __future__ import annotations

from typing import Optional

import typer

from . import __version__
from .device import describe
from .params import GenRequest
from .runner import echo_saved, run_and_save

app = typer.Typer(
    help="Command-line image generation with automatic model download.",
    no_args_is_help=True,
    add_completion=False,
)

# --- reusable option definitions ----------------------------------------
Prompt = typer.Option(..., "--prompt", "-p", help="Text prompt.")
PromptOpt = typer.Option(None, "--prompt", "-p", help="Text/instruction prompt.")
Negative = typer.Option(None, "--negative", help="Negative prompt.")
Model = typer.Option(None, "--model", "-m", help="Alias, HF repo id, or local path.")
Out = typer.Option(None, "--out", "-o", help="Output file, directory, or {seed}/{i} template.")
Width = typer.Option(None, "--width", "-W", help="Image width (px).")
Height = typer.Option(None, "--height", "-H", help="Image height (px).")
Steps = typer.Option(None, "--steps", "-s", help="Inference steps.")
Cfg = typer.Option(None, "--cfg", "-g", help="Guidance / true-CFG scale.")
Seed = typer.Option(None, "--seed", help="Base seed (consecutive across a batch).")
Num = typer.Option(1, "--num", "-n", min=1, help="Number of images.")
Init = typer.Option(None, "--init", "-i", help="Input image (img2img / edit / see-through base).")
Strength = typer.Option(0.8, "--strength", min=0.0, max=1.0, help="img2img denoising strength.")
Device = typer.Option(None, "--device", help="cuda / mps / cpu (auto if unset).")
Dtype = typer.Option(None, "--dtype", help="bf16 / fp16 / fp32 (auto if unset).")
Offload = typer.Option(False, "--offload/--no-offload", help="CPU-offload to save VRAM.")
HfToken = typer.Option(None, "--hf-token", help="Hugging Face token for gated models.")
Meta = typer.Option(True, "--metadata/--no-metadata", help="Embed parameters in PNG.")


def _go(req: GenRequest, out, embed_metadata):
    typer.secho(f"device: {describe()}", fg=typer.colors.BLUE)
    typer.secho(f"generating [{req.kind}] ...", fg=typer.colors.CYAN)
    paths = run_and_save(req, out, embed_metadata)
    echo_saved(paths)


@app.command()
def sd(
    prompt: str = Prompt,
    negative: Optional[str] = Negative,
    model: Optional[str] = Model,
    out: Optional[str] = Out,
    width: Optional[int] = Width,
    height: Optional[int] = Height,
    steps: Optional[int] = Steps,
    cfg: Optional[float] = Cfg,
    seed: Optional[int] = Seed,
    num: int = Num,
    init: Optional[str] = Init,
    strength: float = Strength,
    device: Optional[str] = Device,
    dtype: Optional[str] = Dtype,
    offload: bool = Offload,
    hf_token: Optional[str] = HfToken,
    metadata: bool = Meta,
):
    """Stable Diffusion (SD1.5 / SDXL / SD3.5, auto-detected)."""
    _go(
        GenRequest(
            kind="sd", prompt=prompt, negative=negative, model=model, width=width,
            height=height, steps=steps, cfg=cfg, seed=seed, num=num, init=init,
            strength=strength, device=device, dtype=dtype, offload=offload,
            hf_token=hf_token,
        ),
        out, metadata,
    )


@app.command("qwen-image")
def qwen_image(
    prompt: str = Prompt,
    negative: Optional[str] = Negative,
    model: Optional[str] = Model,
    out: Optional[str] = Out,
    width: Optional[int] = Width,
    height: Optional[int] = Height,
    steps: Optional[int] = Steps,
    cfg: Optional[float] = Cfg,
    seed: Optional[int] = Seed,
    num: int = Num,
    device: Optional[str] = Device,
    dtype: Optional[str] = Dtype,
    offload: bool = Offload,
    hf_token: Optional[str] = HfToken,
    metadata: bool = Meta,
):
    """Qwen-Image text-to-image."""
    _go(
        GenRequest(
            kind="qwen-image", prompt=prompt, negative=negative, model=model,
            width=width, height=height, steps=steps, cfg=cfg, seed=seed, num=num,
            device=device, dtype=dtype, offload=offload, hf_token=hf_token,
        ),
        out, metadata,
    )


@app.command("qwen-image-edit")
def qwen_image_edit(
    prompt: str = Prompt,
    init: str = typer.Option(..., "--init", "-i", help="Input image to edit."),
    negative: Optional[str] = Negative,
    model: Optional[str] = Model,
    out: Optional[str] = Out,
    steps: Optional[int] = Steps,
    cfg: Optional[float] = Cfg,
    seed: Optional[int] = Seed,
    num: int = Num,
    device: Optional[str] = Device,
    dtype: Optional[str] = Dtype,
    offload: bool = Offload,
    hf_token: Optional[str] = HfToken,
    metadata: bool = Meta,
):
    """Qwen-Image-Edit: instruction-based editing of an input image."""
    _go(
        GenRequest(
            kind="qwen-image-edit", prompt=prompt, init=init, negative=negative,
            model=model, steps=steps, cfg=cfg, seed=seed, num=num, device=device,
            dtype=dtype, offload=offload, hf_token=hf_token,
        ),
        out, metadata,
    )


@app.command("see-through")
def see_through(
    prompt: Optional[str] = PromptOpt,
    mode: str = typer.Option("transparent", "--mode", help="transparent | layers"),
    method: str = typer.Option(
        "auto", "--method",
        help="auto | layerdiffuse (native transparent) | matte (BiRefNet)",
    ),
    init: Optional[str] = Init,
    negative: Optional[str] = Negative,
    model: Optional[str] = Model,
    out: Optional[str] = Out,
    width: Optional[int] = Width,
    height: Optional[int] = Height,
    steps: Optional[int] = Steps,
    cfg: Optional[float] = Cfg,
    seed: Optional[int] = Seed,
    num: int = Num,
    device: Optional[str] = Device,
    dtype: Optional[str] = Dtype,
    offload: bool = Offload,
    hf_token: Optional[str] = HfToken,
    metadata: bool = Meta,
):
    """Transparent generation (LayerDiffuse) and layer decomposition (matting)."""
    if mode not in ("transparent", "layers"):
        raise typer.BadParameter("mode must be 'transparent' or 'layers'")
    if method not in ("auto", "layerdiffuse", "matte"):
        raise typer.BadParameter("method must be 'auto', 'layerdiffuse', or 'matte'")
    if not prompt and not init:
        raise typer.BadParameter("provide --prompt to generate, or --init for an existing image")
    _go(
        GenRequest(
            kind="see-through", prompt=prompt, mode=mode, method=method, init=init,
            negative=negative, model=model, width=width, height=height, steps=steps,
            cfg=cfg, seed=seed, num=num, device=device, dtype=dtype, offload=offload,
            hf_token=hf_token,
        ),
        out, metadata,
    )


@app.command("models")
def models():
    """List available model manifests (bundled + ./settings)."""
    from .manifest import list_manifests
    from .registry import DEFAULT_MODEL

    typer.secho("Models (settings/<kind>/<name>.json):", bold=True)
    for kind, mans in list_manifests().items():
        typer.secho(f"  {kind}:", fg=typer.colors.CYAN)
        for man in mans:
            gated = "  [gated]" if man.gated else ""
            desc = f"  — {man.description}" if man.description else ""
            typer.echo(f"    {man.name:22s} -> {man.summary()}{gated}{desc}")

    typer.secho("\nDefault per kind:", bold=True)
    for kind, name in DEFAULT_MODEL.items():
        typer.echo(f"  {kind:16s} -> {name}")

    typer.secho(
        "\nAlso accepts a Hugging Face repo id or a local path as --model.",
        fg=typer.colors.BRIGHT_BLACK,
    )


@app.command()
def version():
    """Show version and device info."""
    typer.echo(f"imggen {__version__}")
    typer.echo(describe())


if __name__ == "__main__":
    app()
