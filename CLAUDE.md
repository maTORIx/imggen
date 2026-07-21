# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`imggen` is a Typer-based CLI wrapping Hugging Face `diffusers` for local image generation. It ships several backends ("kinds"): `sd` (SD1.5/SDXL/SD3.5), `qwen-image` and `qwen-image-edit` (Qwen-Image 20B, GGUF-quantizable), and `see-through` (transparent/RGBA output via LayerDiffuse or BiRefNet matting).

## Commands

The project uses **uv** (there is a `uv.lock`; `pip` alone will not pick up the pinned CUDA wheel index). There is **no test suite, linter, formatter, or CI configured** — do not invent `pytest`/`ruff`/`mypy` commands.

```bash
uv sync                              # install into .venv (pulls torch from the cu130 index)
uv run imggen version                # smoke test: prints version + device
uv run imggen <kind> -p "a red fox" --out fox.png   # generate (sd|qwen-image|qwen-image-edit|see-through)
uv run imggen qwen-image-edit -i in.png -p "make it snow"   # img2img/edit kinds need --init/-i
uv run imggen models                 # list discovered manifests + per-kind defaults
uv run imggen pull [kind] --list     # list catalog presets published under models/ on GitHub
uv run imggen pull sd pony-diffusion-v6-xl   # install a catalog preset + download its weights now
uv run imggen samplers               # list --sampler aliases
uv run imggen init [--force]         # (re)seed ~/.config/imggen/settings with built-in presets
```

Installed as a tool (`uv tool install git+...`), drop the `uv run` prefix. If not on CUDA 13.0, swap `pytorch-cu130` → the matching index in `pyproject.toml`.

## Architecture

End-to-end flow — the CLI never imports a backend; `GenRequest` is the only contract crossing the boundary:

```
cli.py (Typer)  ->  GenRequest (params.py)  ->  runner.run_and_save
  ->  pipelines/__init__.py::run   # single kind->backend switch, lazy imports
        ->  <backend>.generate(req) -> list[(PIL.Image, metadata, path_hint)]
              ->  registry.resolve_model(kind, --model, token) -> ResolvedModel
                    ->  manifest.load_manifest / materialize   # JSON manifest -> download weights
              ->  common.py helpers (device, dtype, seeds, scheduler, placement)
  ->  imageio.build_output_paths + save_image   # PNG/JPG/WebP with embedded metadata
```

Key seams to understand before changing anything:

- **Backend contract is duck-typed, not an ABC.** Each `pipelines/<name>.py` exposes a module-level `generate(req) -> list[(image, dict, path_hint)]`. `path_hint` is `None` for a plain image, or `"fg"`/`"bg"` for see-through layer pairs (drives `_fg`/`_bg` output suffixes and forces `.png` for RGBA). Dispatch lives only in `pipelines/__init__.py::run`.

- **`registry.py` is a *model* resolver, not a plugin registry.** `resolve_model` turns `--model` into a `ResolvedModel` with precedence: existing local path → manifest name (`settings/<kind>/<name>.json`) → raw HF repo id. `DEFAULT_MODEL` gives the per-kind default. Qwen single-file/GGUF transformers are wrapped in `SingleFileTransformer` (transformer loaded via `from_single_file`; text encoder/VAE/scheduler come from `base_repo`).

- **`catalog.py` + `imggen pull` is the *install* path, distinct from resolution.** The repo-root `models/<kind>/<name>.json` tree is a public catalog served over HTTPS from GitHub (raw for a preset, the contents API for `--list`); `IMGGEN_CATALOG_DIR` reads a local tree instead (dev/testing), `IMGGEN_CATALOG_REPO`/`IMGGEN_CATALOG_REF` retarget the remote. `pull` copies the preset into the user settings dir **then downloads the weights via `manifest.materialize`; a download failure rolls the preset back** (nothing half-installed). Note `models/` is NOT packaged into the wheel — it's a GitHub-hosted catalog, only reachable once committed/pushed.

- **Manifests and presets are the same JSON schema** (`manifest.py` reads, `presets.py` writes). A manifest pins a `source` (`hf_repo` [+ optional `hf_file`], or a `url` — HF `resolve` URLs route through `hf_hub_download`, other URLs stream via `wget`) plus generation `defaults`. Saving a preset (`--save --alias`) writes an ordinary manifest, so recall is symmetric with any other `--model NAME`.

- **Setting precedence is centralized** in `common.py::setting(cli_value, key, defaults, fallback)`: explicit CLI flag > manifest `defaults[key]` > backend hardcoded `DEFAULTS`. Most `GenRequest` numeric fields default to `None` precisely so this rule works. Respect it — do not read CLI values directly in a backend. Positive-prompt templates (`defaults.prompt_prefix` / `prompt_suffix`, or `--prompt-prefix`/`--prompt-suffix`) are resolved the same way and applied via `common.compose_prompt` — the composed string is what's sent *and* recorded in metadata. Note: do **not** add a clip-skip setting for SDXL/Pony/Illustrious — diffusers already reads the penultimate CLIP layer for SDXL by default (== A1111 "clip skip 2"); passing `clip_skip=2` would wrongly index `hidden_states[-4]`.

- **`see-through` composes the SD backend.** Its `matte` path re-dispatches through `sd.generate` via `replace(req, kind="sd")`; its presets are stored under the `sd` kind (`_preset_kind`); default model is `sdxl`. The `layerdiffuse` path uses vendored code under `src/imggen/vendor/layer_diffuse/` and hardcodes fp16.

- **Schedulers** (`schedulers.py`): short ComfyUI/A1111 names → diffusers scheduler classes, with auto-generated `_karras`/`_exponential`/`_beta` variants. `common.set_scheduler` warns and keeps the model default on an unknown/incompatible sampler rather than aborting. Flow-matching models (Qwen-Image, SD3.5) require `flowmatch`; classic samplers are for SD1.5/SDXL.

## Runtime state (outside the repo)

- Presets/manifests: `~/.config/imggen/settings/<kind>/<name>.json` (override with `$IMGGEN_HOME` or `$XDG_CONFIG_HOME`). Package-bundled built-ins live at `src/imggen/settings/` and are copied into the user dir on first use, never clobbering user edits.
- Weight cache for `url`-sourced models: `~/.cache/imggen/models/<kind>/` (`$IMGGEN_CACHE`). HF-sourced weights use the standard HF cache.
- Gated models: `--hf-token` / `$HF_TOKEN` (also `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN`).

## Hardware notes

Targets aarch64 + NVIDIA GB10 (Blackwell, sm_121), CUDA 13.0. `device.py` picks CUDA > MPS > CPU; default dtype is **bf16 on CUDA** (fp16 MPS, fp32 CPU). Only the ~20B Qwen transformer is GGUF-quantized — the rest of the pipeline stays full precision. On unified-memory hosts set `IMGGEN_SEQUENTIAL_OFFLOAD=1` for sequential CPU offload; `--offload` on CUDA uses model CPU offload.

## Bundled settings docs

`src/imggen/settings/README.md` is the authoritative reference for the manifest/preset JSON schema — consult it when adding or changing a preset shape.
