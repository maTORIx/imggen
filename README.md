# imggen

Command-line image generation with automatic model download.

```
imggen <kind> --prompt "..." [--model NAME] [--out PATH] [options]
```

Backends (`kind`):

| kind              | what it does                                              | default model |
| ----------------- | -------------------------------------------------------- | ------------- |
| `sd`              | Stable Diffusion — SD1.5 / SDXL / SD3.5 (auto-detected)   | `sdxl`        |
| `qwen-image`      | Qwen-Image text-to-image                                 | `qwen-image` (GGUF Q4) |
| `qwen-image-edit` | Qwen-Image-Edit (2509) — instruction editing of an image | `qwen-image-edit` (GGUF Q4) |
| `see-through`     | transparent PNG generation and layer decomposition       | `sdxl` + LayerDiffuse |

Every model is described by a **preset** — a JSON manifest at
`~/.config/imggen/settings/<kind>/<name>.json`. Built-in models are seeded there
on first use; create your own from the command line with `--save --alias NAME`.
`imggen models` lists them; `--model` also accepts a raw Hugging Face repo id or
a local path. Weights are downloaded on first use.

### Quantized Qwen models (GGUF)

The Qwen-Image / Qwen-Image-Edit transformers are 20B parameters (~40 GB in
bf16). By default `imggen` loads a **GGUF-quantized** transformer — the same
community conversions ComfyUI uses — so only the transformer is quantized while
the text encoder and VAE stay full precision. `Q4_K_M` (~13 GB) is the default
balance point; `*-fp16` loads the full unquantized weights:

```bash
imggen qwen-image -p "..."                      # GGUF Q4_K_M (default)
imggen qwen-image -p "..." -m qwen-image-fp16   # full unquantized weights
```

To use a different quant level, save a preset pointing at the desired `.gguf`
file — e.g. `imggen qwen-image -m <Q8 resolve URL> --steps 40 --save --alias qwen-q8`.

### Model presets (`~/.config/imggen/settings/`)

Models are registered once in a JSON manifest that records the download source
**and** the preferred settings, so the same definition reproduces on another
machine. All presets live in a single directory —
`~/.config/imggen/settings/<kind>/<name>.json` (override the root with
`$IMGGEN_HOME` / `$XDG_CONFIG_HOME`). The built-in models are copied there on
first use; run `imggen init` to re-seed them.

`<kind>` is a backend (`sd`, `qwen-image`, `qwen-image-edit`) and `<name>`
becomes the `--model` value.

**Create a preset from the CLI** — run a command with the options you want plus
`--save --alias NAME`. It writes the preset instead of generating:

```bash
# an 8-step Euler SDXL preset, then recall it by name
imggen sd -m stabilityai/stable-diffusion-xl-base-1.0 \
          --steps 8 --cfg 2.0 --sampler euler --save --alias sdxl-fast
imggen sd -m sdxl-fast -p "a red fox"        # applies steps=8, cfg=2.0, sampler=euler

# derive from an existing preset (inherits its source + defaults, overlays your flags)
imggen qwen-image -m qwen-image --steps 8 --save --alias qwen-fast

# an explicit CLI flag always overrides a preset's default
imggen qwen-image-edit -m qwen-rapid-aio-v23 -i photo.png -p "..." --steps 8
```

Only the flags you explicitly type are recorded (as `defaults`); the `--model`
source is inferred from a preset name (inherited), an `http(s)://` URL, or an
`org/repo` id. A hand-written preset needs only `source`:

```jsonc
// ~/.config/imggen/settings/qwen-image-edit/qwen-rapid-aio-v23.json
{
  "description": "Phr00t Qwen-Image-Edit Rapid AIO SFW v23 (distilled, ~4-step)",
  "source": { "url": "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v23/Qwen-Rapid-AIO-SFW-v23.safetensors" },
  "defaults": { "steps": 4, "cfg": 1.0, "negative": " " }
}
```

`source` is one of:

- `{ "url": "..." }` — any download URL. A `huggingface.co/.../resolve/...` URL
  is fetched via the Hub (cached, authenticated, resumable); any other URL is
  downloaded with `wget` into `~/.cache/imggen/models` (`$IMGGEN_CACHE` to
  override). Optional `filename`, `sha256` (verified after download), and
  `token_env` (name of an env var sent as `Authorization: Bearer`, e.g. a
  Civitai token).
- `{ "hf_repo": "...", "hf_file": "...", "revision": "main", "subfolder": null }` — an explicit file in a Hub repo.
- `{ "hf_repo": "..." }` — a full diffusers repo folder (loaded with `from_pretrained`).

For `qwen-image` / `qwen-image-edit`, a single-file source (`.safetensors` or
`.gguf`, auto-detected) is loaded as the transformer while the text encoder, VAE
and scheduler come from the base repo (`Qwen/Qwen-Image` /
`Qwen/Qwen-Image-Edit-2509`); override with `"load": { "base_repo": "..." }`.
`defaults` accepts `steps`, `cfg`, `width`, `height`, `negative`, `strength`,
`sampler`; an explicit CLI flag always wins. Add `"gated": true` for repos that
need an accepted license / HF token. `imggen models` lists every discovered
preset.

See [`src/imggen/settings/README.md`](src/imggen/settings/README.md) for the
full manifest schema and more examples.

### Prompt weighting (`(word:1.2)`)

Emphasise or de-emphasise parts of a prompt with ComfyUI/A1111 syntax, on the
`sd` backend (SD1.5 / SDXL / SD3.5) and `see-through --method layerdiffuse`:

```bash
imggen sd -p "a (red:1.3) fox, (blurry background:0.4), (masterpiece)"
imggen sd -p "a fox" --negative "(worst quality:1.4), [watermark]"
```

- `(word:1.3)` scales attention by 1.3; `(word)` = 1.1, `[word]` = 1/1.1.
  Escape literal brackets as `\(` `\)` `\[` `\]`.
- It kicks in **only when the prompt contains an unescaped `(` or `[`** — plain
  prompts are encoded exactly as before. Works in the negative prompt too.
- **Qwen-Image / Qwen-Image-Edit** use an LLM text encoder with no standard
  token weighting, so the markup is stripped (with a note) and the words kept.

### Transparent generation (LayerDiffuse)

`see-through --method layerdiffuse` (the default when generating from a prompt)
produces a **native transparent RGBA image** with a real alpha channel via
[LayerDiffuse](https://github.com/lllyasviel/sd-forge-layerdiffuse), recovering
soft/semi-transparent edges (hair, glass, glow) that background removal cannot.
`--method matte` uses BiRefNet background removal instead; it is used
automatically for `--init` decomposition and for `--mode layers`.

### Remote execution (`imggen serve` / `imggen remote`)

Run the heavy generation on a GPU box and drive it from any other machine. The
GPU host serves; the client ships the request and saves the returned image
locally — output paths, `--out`, and metadata stay on the client.

```bash
# On the GPU host — run a foreground daemon (Ctrl-C to stop):
imggen serve --host 0.0.0.0 --port 7863 --api-key SECRET

# On the client — point it at the host, then use imggen exactly as before:
imggen remote set 192.168.1.10:7863 --api-key SECRET
imggen remote status                       # ping: prints the server's device/version
imggen sd -p "a red fox" -o fox.png        # runs on the host, saves fox.png here
imggen qwen-image-edit -i in.png -p "make it snow" -o out.png   # --init is uploaded

imggen sd -p "..." --local                 # run on THIS machine for one command
imggen remote clear                        # forget the remote; back to local
```

Details:

- **No fallback.** Once a remote is set, generation goes there; if the server is
  unreachable the command errors and exits non-zero (it does *not* silently run
  locally). Use `--local` for a one-off local run, or `imggen remote clear`.
- **`--api-key`** is optional. When set on `serve`, clients must present the same
  token (`Authorization: Bearer`). `/health` needs no auth so `remote status`
  can probe. Plain HTTP — put it behind a VPN/SSH tunnel or reverse proxy with
  TLS if it crosses an untrusted network.
- **Models resolve on the server.** `--model` names a preset / repo / path *on
  the host*; a client-only local path won't exist there. `imggen pull` and your
  presets live on the host. The last-used model is kept warm in memory between
  requests; one generation runs at a time.
- **Live progress.** The client shows a per-step progress bar for remote runs
  (the server streams step counts as it denoises), just as a local run shows
  diffusers' own bar. Works with any current `imggen serve`; against an older
  server it simply falls back to no bar.
- No new dependencies (standard-library HTTP). `$IMGGEN_REMOTE` /
  `$IMGGEN_API_KEY` override the stored config; the endpoint is saved to
  `~/.config/imggen/remote.json`.

## Install

Built and tested on **aarch64 + NVIDIA GB10 (Blackwell)** with CUDA 13.0 wheels.

```bash
uv tool install git+https://github.com/matorix/imggen
```

This installs the `imggen` command globally. PyTorch is pulled from the
`cu130` index (declared in `pyproject.toml`). If you need the CUDA 13.2 build
instead, change `pytorch-cu130` to `pytorch-cu132` in `pyproject.toml`, or
install torch yourself first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Development

```bash
git clone https://github.com/matorix/imggen && cd imggen
uv sync
uv run imggen version
```

### Shell completion

```bash
imggen --install-completion    # then restart your shell (bash/zsh/fish)
```

Once installed, <kbd>Tab</kbd> completes subcommands and, where it helps, their
values: `imggen pull <Tab>` → kinds, `imggen pull sd <Tab>` → catalog preset
names, `imggen sd -m <Tab>` → installed presets, and `--sampler <Tab>` → sampler
aliases.

## Examples

```bash
# Stable Diffusion (SDXL by default)
imggen sd --prompt "a red fox in a snowy forest" --out fox.png

# Pick a model by alias, id, or local path
imggen sd -p "portrait, studio light" -m sd3.5 --out p.png     # gated: needs HF token
imggen sd -p "cat" -m stabilityai/stable-diffusion-xl-base-1.0
imggen sd -p "cat" -m ./checkpoints/my_model.safetensors

# Batch + reproducible seed; PNG carries the generation parameters
imggen sd -p "cyberpunk city" -n 4 --seed 1234 --out out/

# Qwen-Image
imggen qwen-image -p "泼墨山水画" --out ink.png

# Qwen-Image-Edit (input image required)
imggen qwen-image-edit -i photo.png -p "make it snow" --out edited.png

# Native transparent PNG (LayerDiffuse, real alpha with soft edges)
imggen see-through -p "a cute robot mascot" --out robot.png

# Background removal on an existing image (BiRefNet matting)
imggen see-through -i photo.png --out cutout.png

# Layer decomposition -> scene_fg.png (transparent) + scene_bg.png (filled)
imggen see-through --mode layers -i scene.png --out scene.png
```

List aliases: `imggen models`.

## Common options

| option | meaning |
| --- | --- |
| `-p, --prompt` | text prompt |
| `--negative` | negative prompt |
| `-m, --model` | alias / HF repo id / local path |
| `-o, --out` | file, directory, or `{seed}`/`{i}` template |
| `-W, --width` / `-H, --height` | size in px |
| `-s, --steps` | inference steps |
| `-g, --cfg` | guidance / true-CFG scale |
| `--sampler` | scheduler: `euler`, `dpm_2`, `dpm++_2m[_sde]`, `dpm++_3m_sde`, `unipc`, `deis`, `ddim`, `lcm`, `tcd`, `flowmatch`, plus `_karras`/`_exponential`/`_beta` variants (`imggen samplers`) |
| `--seed` | base seed (consecutive across a batch) |
| `-n, --num` | number of images |
| `-i, --init` | input image (img2img / edit / see-through base) |
| `--strength` | img2img denoising strength |
| `--device` / `--dtype` | `cuda`/`mps`/`cpu`, `bf16`/`fp16`/`fp32` (auto if unset) |
| `--offload` | CPU-offload to save VRAM |
| `--hf-token` | token for gated models (or set `HF_TOKEN`) |
| `--no-metadata` | do not embed parameters in the PNG |
| `--local` | run on this machine even if a remote is configured |
| `--save --alias NAME` | save these options as a reusable preset (no generation) |

Utility commands: `imggen models` (list presets), `imggen samplers` (list
sampler names), `imggen init` (re-seed the built-in presets), `imggen serve`
(run a generation daemon), `imggen remote set/status/clear` (drive a remote
daemon from this machine).

## Gated models

SD3.5 requires accepting its license on the Hub. Provide a token via
`--hf-token`, the `HF_TOKEN` environment variable, or `huggingface-cli login`.
Defaults (`sdxl`, Qwen-Image) are ungated and need no token.
