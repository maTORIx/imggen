"""imggen command-line interface.

Usage: ``imggen <kind> --prompt "..." [options]`` where kind is one of
``sd``, ``qwen-image``, ``qwen-image-edit``, ``see-through``.

Any invocation can be stored as a reusable preset with ``--save --alias NAME``
(writes ``~/.config/imggen/settings/<kind>/NAME.json``) and recalled later with
``--model NAME``.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import typer

from . import __version__
from . import schedulers
from .device import describe
from .params import GenRequest
from .remote import DEFAULT_PORT
from .runner import echo_saved, run_and_save

app = typer.Typer(
    help="Command-line image generation with automatic model download.",
    no_args_is_help=True,
    add_completion=True,  # enables `imggen --install-completion` for the shell
)


# --- shell completion callbacks -----------------------------------------
# Typer classifies a completer's params by type: ``typer.Context`` -> ctx,
# ``str`` -> the incomplete word. Keep these fast (no torch, short timeouts):
# they run on every <TAB>.

def _complete_kind(incomplete: str):
    from .manifest import KINDS

    return [k for k in KINDS if k.startswith(incomplete)]


def _complete_sampler(incomplete: str):
    return [n for n in schedulers.NAMES if n.startswith(incomplete)]


def _model_names_for_kind(kind: str, local: bool = False) -> list[str]:
    """Preset names available for ``kind`` — the remote's when one is configured.

    With a remote set, completion should offer the *server's* presets (that is
    where ``--model`` is resolved), so we query ``/models`` with a short timeout.
    ``local=True`` (the caller saw ``--local`` on the line) forces the local
    presets instead, mirroring what ``--local`` does at generation time. Must
    never raise: on any failure return ``[]`` so <TAB> stays silent rather than
    erroring. No local fallback when a remote is used — that would suggest names
    the server may not have (matches the no-fallback design).
    """
    from . import remote

    if not local and remote.endpoint():
        try:
            info = remote.models(timeout=2)
        except Exception:
            return []
        return [m["name"] for m in info.get("models", {}).get(kind, [])]
    from .manifest import list_manifests

    try:
        return [m.name for m in list_manifests().get(kind, [])]
    except Exception:
        return []


def _complete_model(ctx: typer.Context, incomplete: str):
    """Complete --model with the presets installed for this command's kind.

    Honours a ``--local`` already typed on the line (``ctx.params``): completion
    then lists this machine's presets, matching the run it would produce.
    """
    cmd = getattr(ctx.command, "name", None) or ""
    kind = "sd" if cmd == "see-through" else cmd
    local = bool(ctx.params.get("local"))
    return [n for n in _model_names_for_kind(kind, local) if n.startswith(incomplete)]


def _complete_catalog_alias(ctx: typer.Context, incomplete: str):
    """Complete the pull alias with catalog names for the already-typed kind."""
    from . import catalog
    from .manifest import KINDS

    kind = ctx.params.get("kind")
    if kind not in KINDS:
        return []
    return [n for n in catalog.list_names(kind, timeout=4) if n.startswith(incomplete)]


# --- reusable option definitions ----------------------------------------
PromptOpt = typer.Option(None, "--prompt", "-p", help="Text/instruction prompt.")
Negative = typer.Option(None, "--negative", help="Negative prompt.")
PromptPrefix = typer.Option(
    None, "--prompt-prefix",
    help="Positive-prompt template prepended to --prompt (e.g. quality tags "
         "like Pony's `score_9, ...`). Usually set in a preset's defaults.",
)
PromptSuffix = typer.Option(
    None, "--prompt-suffix", help="Positive-prompt template appended to --prompt.",
)
Model = typer.Option(
    None, "--model", "-m", help="Alias, HF repo id, or local path.",
    autocompletion=_complete_model,
)
Out = typer.Option(None, "--out", "-o", help="Output file, directory, or {seed}/{i} template.")
Width = typer.Option(None, "--width", "-W", help="Image width (px).")
Height = typer.Option(None, "--height", "-H", help="Image height (px).")
Steps = typer.Option(None, "--steps", "-s", help="Inference steps.")
Cfg = typer.Option(None, "--cfg", "-g", help="Guidance / true-CFG scale.")
Sampler = typer.Option(
    None, "--sampler",
    help="Sampler/scheduler: euler[_a], heun, dpm_2[_a], dpm++_2m[_sde], "
         "dpm++_3m_sde, dpm++_2s_a, dpm++_sde, unipc[_bh2], deis, ddim, ddpm, "
         "ipndm, lms, lcm, tcd, flowmatch — plus _karras/_exponential/_beta "
         "sigma variants (see `imggen samplers`). Classic samplers are for "
         "SD1.5/SDXL; Qwen-Image & SD3.5 are flow-matching (use flowmatch).",
    autocompletion=_complete_sampler,
)
Seed = typer.Option(None, "--seed", help="Base seed (consecutive across a batch).")
Num = typer.Option(1, "--num", "-n", min=1, help="Number of images (total).")
BatchSize = typer.Option(
    1, "--batch-size", "-b", min=1,
    help="Images per forward pass (num_images_per_prompt). Faster than -n but "
         "uses ~batch-size× the VRAM; each image still gets its own seed. "
         "-n is split into ceil(n/batch-size) passes. Not applied to see-through "
         "layerdiffuse.",
)
Init = typer.Option(None, "--init", "-i", help="Input image (img2img / edit / see-through base).")
Strength = typer.Option(None, "--strength", min=0.0, max=1.0, help="img2img denoising strength (default 0.8).")
Device = typer.Option(None, "--device", help="cuda / mps / cpu (auto if unset).")
Dtype = typer.Option(None, "--dtype", help="bf16 / fp16 / fp32 (auto if unset).")
Offload = typer.Option(False, "--offload/--no-offload", help="CPU-offload to save VRAM.")
HfToken = typer.Option(None, "--hf-token", help="Hugging Face token for gated models.")
Meta = typer.Option(True, "--metadata/--no-metadata", help="Embed parameters in PNG.")
Save = typer.Option(False, "--save", help="Save these options as a reusable preset (does not generate).")
Alias = typer.Option(None, "--alias", help="Preset name to save under (settings/<kind>/<alias>.json).")
Desc = typer.Option(None, "--desc", help="Optional description stored in the preset.")
LocalOpt = typer.Option(
    False, "--local",
    help="Run on this machine even if a remote is configured (imggen remote set).",
)
Preview = typer.Option(
    True, "--preview/--no-preview",
    help="Show the result inline in graphics-capable terminals "
         "(Ghostty/Kitty/iTerm2); batches as a grid. No-op elsewhere.",
)


def _go(req: GenRequest, out, embed_metadata, local: bool = False, preview: bool = True):
    from . import remote

    # A configured remote wins unless --local is passed. On the remote path we
    # never import torch here (describe() would): the client just ships the
    # request and saves what comes back.
    ep = None if local else remote.endpoint()
    if ep:
        typer.secho(f"remote: {ep}", fg=typer.colors.BLUE)
    else:
        typer.secho(f"device: {describe()}", fg=typer.colors.BLUE)
    typer.secho(f"generating [{req.kind}] ...", fg=typer.colors.CYAN)
    try:
        paths = run_and_save(req, out, embed_metadata, use_remote=bool(ep), preview=preview)
    except remote.RemoteError as exc:
        # No fallback by design: a configured-but-unreachable remote is an error.
        typer.secho(f"remote error: {exc}", fg=typer.colors.RED)
        typer.secho(
            "  (no local fallback; run with --local to generate here, "
            "or `imggen remote clear`)",
            fg=typer.colors.BRIGHT_BLACK,
        )
        raise typer.Exit(1)
    echo_saved(paths)


# --- preset saving ------------------------------------------------------

def _preset_kind(kind: str) -> str:
    """The settings kind a preset is stored under.

    ``see-through`` resolves its base model through the ``sd`` backend, so its
    presets live under ``sd`` and recall via ``see-through --model <alias>``.
    """
    return "sd" if kind == "see-through" else kind


def _explicit(ctx: typer.Context, names) -> dict:
    """The subset of ``names`` the user actually typed on the command line.

    Compared by enum *name* rather than identity: Typer vendors its own Click
    (``typer._click``), so the ``ParameterSource`` returned here is not the same
    class as ``click.core.ParameterSource``.
    """
    out = {}
    for name in names:
        source = ctx.get_parameter_source(name)
        if getattr(source, "name", None) == "COMMANDLINE":
            out[name] = ctx.params.get(name)
    return out


def _save_preset(ctx: typer.Context, kind: str, model, alias, desc) -> None:
    from .presets import SAVEABLE, save_preset

    if not alias:
        raise typer.BadParameter("--save requires --alias <name>", param_hint="--alias")
    overrides = _explicit(ctx, SAVEABLE)
    pkind = _preset_kind(kind)
    try:
        path, existed, defaults = save_preset(pkind, model, overrides, alias, desc)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--model")

    verb = "updated" if existed else "saved"
    typer.secho(f"{verb} preset: {path}", fg=typer.colors.GREEN)
    typer.echo(f"  model source: {model or f'built-in default ({pkind})'}")
    if defaults:
        typer.echo("  defaults: " + ", ".join(f"{k}={v}" for k, v in defaults.items()))
    typer.secho(
        f'  recall: imggen {ctx.command.name} --model {alias} -p "..."',
        fg=typer.colors.BRIGHT_BLACK,
    )


def _require(value, flag: str):
    if not value:
        raise typer.BadParameter(
            f"{flag} is required (or use --save --alias NAME to store a preset)",
            param_hint=flag,
        )


@app.command()
def sd(
    ctx: typer.Context,
    prompt: Optional[str] = PromptOpt,
    negative: Optional[str] = Negative,
    prompt_prefix: Optional[str] = PromptPrefix,
    prompt_suffix: Optional[str] = PromptSuffix,
    model: Optional[str] = Model,
    out: Optional[str] = Out,
    width: Optional[int] = Width,
    height: Optional[int] = Height,
    steps: Optional[int] = Steps,
    cfg: Optional[float] = Cfg,
    sampler: Optional[str] = Sampler,
    seed: Optional[int] = Seed,
    num: int = Num,
    batch_size: int = BatchSize,
    init: Optional[str] = Init,
    strength: Optional[float] = Strength,
    device: Optional[str] = Device,
    dtype: Optional[str] = Dtype,
    offload: bool = Offload,
    hf_token: Optional[str] = HfToken,
    metadata: bool = Meta,
    local: bool = LocalOpt,
    preview: bool = Preview,
    save: bool = Save,
    alias: Optional[str] = Alias,
    desc: Optional[str] = Desc,
):
    """Stable Diffusion (SD1.5 / SDXL / SD3.5, auto-detected)."""
    if save or alias is not None:
        return _save_preset(ctx, "sd", model, alias, desc)
    _require(prompt, "--prompt")
    _go(
        GenRequest(
            kind="sd", prompt=prompt, negative=negative, prompt_prefix=prompt_prefix,
            prompt_suffix=prompt_suffix, model=model, width=width,
            height=height, steps=steps, cfg=cfg, sampler=sampler, seed=seed, num=num,
            batch_size=batch_size, init=init, strength=strength, device=device,
            dtype=dtype, offload=offload, hf_token=hf_token,
        ),
        out, metadata, local, preview,
    )


@app.command("qwen-image")
def qwen_image(
    ctx: typer.Context,
    prompt: Optional[str] = PromptOpt,
    negative: Optional[str] = Negative,
    prompt_prefix: Optional[str] = PromptPrefix,
    prompt_suffix: Optional[str] = PromptSuffix,
    model: Optional[str] = Model,
    out: Optional[str] = Out,
    width: Optional[int] = Width,
    height: Optional[int] = Height,
    steps: Optional[int] = Steps,
    cfg: Optional[float] = Cfg,
    sampler: Optional[str] = Sampler,
    seed: Optional[int] = Seed,
    num: int = Num,
    batch_size: int = BatchSize,
    device: Optional[str] = Device,
    dtype: Optional[str] = Dtype,
    offload: bool = Offload,
    hf_token: Optional[str] = HfToken,
    metadata: bool = Meta,
    local: bool = LocalOpt,
    preview: bool = Preview,
    save: bool = Save,
    alias: Optional[str] = Alias,
    desc: Optional[str] = Desc,
):
    """Qwen-Image text-to-image."""
    if save or alias is not None:
        return _save_preset(ctx, "qwen-image", model, alias, desc)
    _require(prompt, "--prompt")
    _go(
        GenRequest(
            kind="qwen-image", prompt=prompt, negative=negative,
            prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix, model=model,
            width=width, height=height, steps=steps, cfg=cfg, sampler=sampler,
            seed=seed, num=num, batch_size=batch_size, device=device, dtype=dtype,
            offload=offload, hf_token=hf_token,
        ),
        out, metadata, local, preview,
    )


@app.command("qwen-image-edit")
def qwen_image_edit(
    ctx: typer.Context,
    prompt: Optional[str] = PromptOpt,
    init: Optional[str] = Init,
    negative: Optional[str] = Negative,
    prompt_prefix: Optional[str] = PromptPrefix,
    prompt_suffix: Optional[str] = PromptSuffix,
    model: Optional[str] = Model,
    out: Optional[str] = Out,
    width: Optional[int] = Width,
    height: Optional[int] = Height,
    steps: Optional[int] = Steps,
    cfg: Optional[float] = Cfg,
    sampler: Optional[str] = Sampler,
    seed: Optional[int] = Seed,
    num: int = Num,
    batch_size: int = BatchSize,
    device: Optional[str] = Device,
    dtype: Optional[str] = Dtype,
    offload: bool = Offload,
    hf_token: Optional[str] = HfToken,
    metadata: bool = Meta,
    local: bool = LocalOpt,
    preview: bool = Preview,
    save: bool = Save,
    alias: Optional[str] = Alias,
    desc: Optional[str] = Desc,
):
    """Qwen-Image-Edit: instruction-based editing of an input image."""
    if save or alias is not None:
        return _save_preset(ctx, "qwen-image-edit", model, alias, desc)
    _require(prompt, "--prompt")
    _require(init, "--init")
    _go(
        GenRequest(
            kind="qwen-image-edit", prompt=prompt, init=init, negative=negative,
            prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix,
            model=model, width=width, height=height, steps=steps, cfg=cfg,
            sampler=sampler, seed=seed, num=num,
            batch_size=batch_size, device=device, dtype=dtype, offload=offload,
            hf_token=hf_token,
        ),
        out, metadata, local, preview,
    )


@app.command("see-through")
def see_through(
    ctx: typer.Context,
    prompt: Optional[str] = PromptOpt,
    mode: str = typer.Option("transparent", "--mode", help="transparent | layers"),
    method: str = typer.Option(
        "auto", "--method",
        help="auto | layerdiffuse (native transparent) | matte (BiRefNet) | "
             "decompose (part layers → PSD)",
    ),
    parts: str = typer.Option(
        "face", "--parts",
        help="--method decompose only: face = split the head (hair/eyes/brows/mouth/...) "
             "and merge everything from the neck down into one 'body' layer; "
             "all = keep every See-through part as its own layer.",
    ),
    init: Optional[str] = Init,
    negative: Optional[str] = Negative,
    prompt_prefix: Optional[str] = PromptPrefix,
    prompt_suffix: Optional[str] = PromptSuffix,
    model: Optional[str] = Model,
    out: Optional[str] = Out,
    width: Optional[int] = Width,
    height: Optional[int] = Height,
    steps: Optional[int] = Steps,
    cfg: Optional[float] = Cfg,
    sampler: Optional[str] = Sampler,
    seed: Optional[int] = Seed,
    num: int = Num,
    batch_size: int = BatchSize,
    device: Optional[str] = Device,
    dtype: Optional[str] = Dtype,
    offload: bool = Offload,
    hf_token: Optional[str] = HfToken,
    metadata: bool = Meta,
    local: bool = LocalOpt,
    preview: bool = Preview,
    save: bool = Save,
    alias: Optional[str] = Alias,
    desc: Optional[str] = Desc,
):
    """Transparent generation (LayerDiffuse) and layer decomposition (matting)."""
    if save or alias is not None:
        return _save_preset(ctx, "see-through", model, alias, desc)
    if mode not in ("transparent", "layers"):
        raise typer.BadParameter("mode must be 'transparent' or 'layers'")
    if method not in ("auto", "layerdiffuse", "matte", "decompose"):
        raise typer.BadParameter("method must be 'auto', 'layerdiffuse', 'matte', or 'decompose'")
    if parts not in ("face", "all"):
        raise typer.BadParameter("parts must be 'face' or 'all'")
    if not prompt and not init:
        raise typer.BadParameter("provide --prompt to generate, or --init for an existing image")
    _go(
        GenRequest(
            kind="see-through", prompt=prompt, mode=mode, method=method, parts=parts, init=init,
            negative=negative, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix,
            model=model, width=width, height=height, steps=steps,
            cfg=cfg, sampler=sampler, seed=seed, num=num, batch_size=batch_size,
            device=device, dtype=dtype, offload=offload, hf_token=hf_token,
        ),
        out, metadata, local, preview,
    )


# --- catalog install (`imggen pull`) ------------------------------------

def _recall_cmd(kind: str) -> str:
    """The subcommand used to recall a pulled preset (kind == command name)."""
    return kind


def _pull_list(catalog, kinds, kind: str | None) -> None:
    if kind is not None and kind not in kinds:
        raise typer.BadParameter(
            f"unknown kind '{kind}'; expected one of {', '.join(kinds)}",
            param_hint="KIND",
        )
    wanted = [kind] if kind else list(kinds)
    try:
        found = catalog.list_all(wanted)
    except catalog.CatalogError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)

    source = catalog.source_desc()
    if not found:
        typer.secho(f"Catalog · {source}", bold=True)
        typer.secho("  (no presets found)", fg=typer.colors.YELLOW)
        return
    _render_catalog(source, found)


def _render_catalog(source: str, found: dict) -> None:
    """Print the catalog as aligned, wrapping columns (rich, with a plain fallback)."""
    try:
        from rich.console import Console
        from rich.padding import Padding
        from rich.table import Table
    except ImportError:
        return _render_catalog_plain(source, found)

    console = Console()
    console.print(f"[bold]Catalog[/]  [dim]{source}[/]")
    for k, items in found.items():
        console.print(f"\n[bold cyan]{k}[/]")
        table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 3, 0, 0))
        table.add_column(style="green", no_wrap=True)     # preset name
        table.add_column(style="dim", overflow="fold")    # description (wraps)
        for name, desc in items:
            table.add_row(name, desc or "—")
        console.print(Padding(table, (0, 0, 0, 4)))
    console.print("\n[dim]Install (downloads weights):[/]  imggen pull <kind> <name>")


def _render_catalog_plain(source: str, found: dict) -> None:
    import shutil
    import textwrap

    width = max(48, shutil.get_terminal_size((80, 20)).columns)
    typer.secho(f"Catalog  {source}", bold=True)
    namew = max((len(n) for items in found.values() for n, _ in items), default=0)
    for k, items in found.items():
        typer.secho(f"\n{k}", fg=typer.colors.CYAN, bold=True)
        for name, desc in items:
            prefix = f"    {name.ljust(namew)}   "
            hang = " " * len(prefix)
            typer.echo(
                textwrap.fill(desc, width=width, initial_indent=prefix, subsequent_indent=hang)
                if desc else prefix.rstrip()
            )
    typer.secho(
        "\nInstall (downloads weights):  imggen pull <kind> <name>",
        fg=typer.colors.BRIGHT_BLACK,
    )


def _rollback(target, existed: bool, old_text: str | None) -> None:
    """Undo the installed preset file after a failed download."""
    try:
        if existed and old_text is not None:
            target.write_text(old_text)
        else:
            target.unlink(missing_ok=True)
    except OSError:
        pass


def _download_weights(mf, man, token):
    """Materialize a manifest's weights, returning the local path (or repo dir)."""
    mat = mf.materialize(man, token)
    if mat.is_repo:  # bare hf_repo: pull the whole folder so it is really present
        from huggingface_hub import snapshot_download

        return snapshot_download(man.source.hf_repo, token=token)
    if not os.path.exists(mat.ref):
        raise RuntimeError(f"expected downloaded file is missing: {mat.ref}")
    return mat.ref


def _pull_install(catalog, kind: str, alias: str, force: bool, hf_token_arg) -> None:
    from . import manifest as mf
    from .config import hf_token as resolve_hf_token

    # 1. Fetch and validate the published preset.
    try:
        data = catalog.fetch_preset(kind, alias)
    except catalog.CatalogNotFound as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        typer.secho(f"  see: imggen pull {kind} --list", fg=typer.colors.BRIGHT_BLACK)
        raise typer.Exit(1)
    except catalog.CatalogError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(1)
    try:
        man = mf._parse(kind, alias, data)
    except ValueError as exc:
        typer.secho(f"catalog preset {kind}/{alias} is malformed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    # 2. Install the preset file (seed built-ins first so we don't suppress them),
    #    remembering the prior state so a failed download can be rolled back.
    mf.ensure_seeded()
    target = mf.user_settings_dir() / kind / f"{alias}.json"
    existed = target.exists()
    old_text = target.read_text() if existed else None
    if existed and not force:
        typer.secho(f"preset already installed: {target}", fg=typer.colors.YELLOW)
        typer.secho("  re-download / overwrite with --force", fg=typer.colors.BRIGHT_BLACK)
        raise typer.Exit(1)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    typer.secho(f"installed preset: {target}", fg=typer.colors.GREEN)

    # 3. Download the weights now; cancel (roll back the preset) on any failure.
    token = resolve_hf_token(hf_token_arg)
    typer.secho(f"downloading weights for {kind}/{alias} (this can be large) ...", fg=typer.colors.CYAN)
    try:
        local = _download_weights(mf, man, token)
    except (Exception, KeyboardInterrupt) as exc:
        _rollback(target, existed, old_text)
        if isinstance(exc, KeyboardInterrupt):
            typer.secho(f"\ncancelled; rolled back {kind}/{alias}", fg=typer.colors.RED)
            raise typer.Exit(130)
        typer.secho(f"download failed; rolled back {kind}/{alias}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)

    typer.secho(f"weights ready: {local}", fg=typer.colors.GREEN)
    typer.secho(
        f'  recall: imggen {_recall_cmd(kind)} --model {alias} -p "..."',
        fg=typer.colors.BRIGHT_BLACK,
    )


@app.command()
def pull(
    kind: Optional[str] = typer.Argument(
        None, help="Model kind: sd | qwen-image | qwen-image-edit.",
        autocompletion=_complete_kind,
    ),
    alias: Optional[str] = typer.Argument(
        None, help="Catalog preset name to install (see --list).",
        autocompletion=_complete_catalog_alias,
    ),
    list_models: bool = typer.Option(False, "--list", help="List catalog presets instead of installing."),
    force: bool = typer.Option(False, "--force", help="Overwrite an already-installed preset of the same name."),
    hf_token: Optional[str] = HfToken,
):
    """Install a model preset from the public catalog and download its weights.

    ``imggen pull sd wai-illustrious-sdxl`` copies the published preset into
    ~/.config/imggen/settings/sd/ and downloads the checkpoint straight away; if
    the download fails the preset is rolled back so nothing half-installed is
    left behind. ``imggen pull [kind] --list`` shows what is available.
    """
    from . import catalog
    from .manifest import KINDS

    if list_models:
        return _pull_list(catalog, KINDS, kind)

    if not kind or not alias:
        raise typer.BadParameter(
            "usage: imggen pull <kind> <alias>   (or: imggen pull [kind] --list)"
        )
    if kind not in KINDS:
        raise typer.BadParameter(
            f"unknown kind '{kind}'; expected one of {', '.join(KINDS)} "
            "(see-through models install under 'sd')",
            param_hint="KIND",
        )
    _pull_install(catalog, kind, alias, force, hf_token)


@app.command()
def init(
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing built-in presets (keeps your own)."
    ),
):
    """Seed ~/.config/imggen/settings with the built-in model presets."""
    from .manifest import seed_settings, user_settings_dir

    dst = user_settings_dir()
    copied = seed_settings(force=force)
    if copied:
        typer.secho(f"seeded {len(copied)} preset(s) into {dst}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"{dst} already up to date", fg=typer.colors.BLUE)


@app.command()
def samplers():
    """List the sampler/scheduler names accepted by --sampler."""
    suffixes = tuple(f"_{s}" for s in schedulers._SIGMA_SUFFIX)
    base = [n for n in schedulers.NAMES if not n.endswith(suffixes)]
    variants = [n for n in schedulers.NAMES if n.endswith(suffixes)]

    typer.secho("Samplers (--sampler <name>):", bold=True)
    for name in base:
        cls_name, extra = schedulers.resolve(name)
        note = f"  ({', '.join(f'{k}={v}' for k, v in extra.items())})" if extra else ""
        typer.echo(f"  {name:16s} -> {cls_name}{note}")

    if variants:
        typer.secho(
            "\nSigma-schedule variants (ComfyUI's karras/exponential/beta):",
            bold=True,
        )
        typer.echo("  " + ", ".join(variants))

    typer.secho(
        "\nClassic samplers apply to SD1.5 / SDXL. Qwen-Image and SD3.5 are "
        "flow-matching models — use `flowmatch`. Distilled models: `lcm` / `tcd`.",
        fg=typer.colors.BRIGHT_BLACK,
    )


def _render_models(source: str, models_by_kind: dict, defaults: dict, only: str | None) -> None:
    """Print the ``imggen models`` listing from a :func:`registry.models_info` dict.

    ``models_by_kind`` maps kind -> list of ``{name, summary, gated, description}``
    (local or remote — same shape). ``only`` restricts the output to one kind.
    """
    typer.secho(f"Presets ({source}):", bold=True)
    kinds = [only] if only else list(models_by_kind.keys())
    shown = False
    for kind in kinds:
        mans = models_by_kind.get(kind) or []
        if not mans:
            continue
        shown = True
        typer.secho(f"  {kind}:", fg=typer.colors.CYAN)
        for m in mans:
            gated = "  [gated]" if m.get("gated") else ""
            desc = f"  — {m['description']}" if m.get("description") else ""
            typer.echo(f"    {m['name']:22s} -> {m.get('summary', '')}{gated}{desc}")
    if only and not shown:
        typer.secho(f"  (no presets installed for '{only}')", fg=typer.colors.YELLOW)

    typer.secho("\nDefault per kind:", bold=True)
    for kind, name in defaults.items():
        if only and kind != only:
            continue
        typer.echo(f"  {kind:16s} -> {name}")

    typer.secho(
        "\nAlso accepts a Hugging Face repo id or a local path as --model. "
        "Create a preset with `imggen <kind> ... --save --alias NAME`.",
        fg=typer.colors.BRIGHT_BLACK,
    )


@app.command("models")
def models(
    kind: Optional[str] = typer.Argument(
        None, help="Only list presets for this kind (sd | qwen-image | qwen-image-edit).",
        autocompletion=_complete_kind,
    ),
    local: bool = LocalOpt,
):
    """List the model presets (--model aliases) available for each kind.

    With a remote configured (imggen remote set), this lists the remote's presets
    — that is where --model is resolved — so pass --local to list this machine's
    presets instead. Give a KIND to narrow the output to one kind.
    """
    from . import remote
    from .manifest import KINDS

    only = None
    if kind is not None:
        only = _preset_kind(kind)  # see-through -> sd (its presets live under sd)
        if only not in KINDS:
            raise typer.BadParameter(
                f"unknown kind '{kind}'; expected one of {', '.join(KINDS)} "
                "(see-through presets live under 'sd')",
                param_hint="KIND",
            )

    ep = None if local else remote.endpoint()
    if ep:
        try:
            info = remote.models(timeout=5)
        except remote.RemoteError as exc:
            typer.secho(f"remote error: {exc}", fg=typer.colors.RED)
            typer.secho(
                "  (no local fallback; run with --local to list this machine's "
                "presets, or `imggen remote clear`)",
                fg=typer.colors.BRIGHT_BLACK,
            )
            raise typer.Exit(1)
        source = f"remote {ep}"
    else:
        from .manifest import user_settings_dir
        from .registry import models_info

        info = models_info()
        source = f"{user_settings_dir()}/<kind>/<name>.json"

    _render_models(source, info.get("models", {}), info.get("defaults", {}), only)


@app.command()
def version():
    """Show version and device info."""
    typer.echo(f"imggen {__version__}")
    typer.echo(describe())


# --- remote execution: serve + client config ----------------------------

@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind address (0.0.0.0 = all interfaces)."),
    port: int = typer.Option(DEFAULT_PORT, "--port", help=f"Port to listen on (default {DEFAULT_PORT})."),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", envvar="IMGGEN_API_KEY",
        help="Require this bearer token on /generate (recommended when exposed).",
    ),
):
    """Serve generation over HTTP so other machines can `imggen remote set` here.

    Runs in the foreground until Ctrl-C. One generation runs at a time and the
    last-used model is kept warm in memory between requests. Background it with
    your own supervisor (systemd, tmux, `&`) as you like.
    """
    # Keep the last-used pipeline resident across requests (see common.py).
    os.environ.setdefault("IMGGEN_PIPELINE_CACHE", "1")
    from .server import serve as run_server

    run_server(host, port, api_key)


setup_app = typer.Typer(
    help="Provision backends that need their own isolated environment.",
    no_args_is_help=True,
)
app.add_typer(setup_app, name="setup")


@setup_app.command("see-through")
def setup_see_through(
    force: bool = typer.Option(False, "--force", help="Delete and rebuild the checkout and venv."),
    status: bool = typer.Option(False, "--status", help="Only report where it lives and whether it is ready."),
):
    """Build the isolated env `see-through --method decompose` runs in.

    Clones the See-through checkout, creates a venv, and installs its pinned
    dependencies under `~/.cache/imggen/seethrough/` (a few GB, mostly torch).
    The first decompose run does this automatically; use this command to warm it
    up ahead of time, retry after a failed install, or `--force` a rebuild.
    """
    from . import seethrough_env as st_env

    if status:
        info = st_env.describe()
        mark = lambda ok: typer.style("ok" if ok else "missing",  # noqa: E731
                                      fg=typer.colors.GREEN if ok else typer.colors.RED)
        typer.echo(f"repo:   {info['repo']}  [{mark(info['repo_ok'])}]")
        typer.echo(f"python: {info['python']}  [{mark(info['python_ok'])}]")
        if not info["ready"]:
            typer.secho("not ready — run `imggen setup see-through`", fg=typer.colors.YELLOW)
        raise typer.Exit(0 if info["ready"] else 1)

    try:
        st_env.ensure(force=force)
    except st_env.SetupError as exc:
        typer.secho(f"setup failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc


remote_app = typer.Typer(
    help="Configure a remote imggen serve endpoint (set/show/status/clear).",
    no_args_is_help=True,
)
app.add_typer(remote_app, name="remote")


@remote_app.command("set")
def remote_set(
    endpoint: str = typer.Argument(
        ..., help=f"HOST[:PORT] of an `imggen serve` daemon (PORT defaults to {DEFAULT_PORT})."
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", envvar="IMGGEN_API_KEY",
        help="Bearer token, if the server requires one.",
    ),
):
    """Route generation to a remote imggen serve daemon (no local fallback)."""
    from . import remote

    try:
        path = remote.save(endpoint, api_key)
    except ValueError as exc:
        typer.secho(f"invalid endpoint {endpoint!r}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)
    stored = remote.endpoint()
    typer.secho(f"remote set: {stored}", fg=typer.colors.GREEN)
    typer.secho(f"  saved to {path}", fg=typer.colors.BRIGHT_BLACK)
    try:
        info = remote.ping(timeout=5)
    except remote.RemoteError as exc:
        typer.secho(f"  warning: not reachable yet — {exc}", fg=typer.colors.YELLOW)
        typer.secho(
            "  generation will error until the server is up (there is no local fallback; "
            "use --local to override per-run, or `imggen remote clear`).",
            fg=typer.colors.BRIGHT_BLACK,
        )
        return
    typer.secho(
        f"  reachable — server device: {info.get('device', '?')} "
        f"(imggen {info.get('version', '?')})",
        fg=typer.colors.BLUE,
    )


@remote_app.command("show")
def remote_show():
    """Show the currently configured remote (if any)."""
    from . import remote

    cfg = remote.load()
    if not cfg:
        typer.secho("no remote configured — running locally", fg=typer.colors.BLUE)
        return
    typer.secho(f"remote: {cfg['endpoint']}", fg=typer.colors.GREEN)
    if cfg.get("api_key"):
        typer.echo("  api-key: set")


@remote_app.command("status")
def remote_status():
    """Ping the configured remote and print its device/version."""
    from . import remote

    cfg = remote.load()
    if not cfg:
        typer.secho("no remote configured — running locally", fg=typer.colors.BLUE)
        return
    try:
        info = remote.ping(timeout=5)
    except remote.RemoteError as exc:
        typer.secho(f"unreachable: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho(f"reachable: {cfg['endpoint']}", fg=typer.colors.GREEN)
    typer.echo(f"  device:  {info.get('device', '?')}")
    typer.echo(f"  version: imggen {info.get('version', '?')}")
    typer.echo(f"  kinds:   {', '.join(info.get('kinds', []))}")


@remote_app.command("clear")
def remote_clear():
    """Remove the configured remote and go back to local execution."""
    from . import remote

    if remote.clear():
        typer.secho("remote cleared — running locally", fg=typer.colors.GREEN)
    else:
        typer.secho("no remote was configured", fg=typer.colors.BLUE)


if __name__ == "__main__":
    app()
