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

Every model is described by a **manifest** (`settings/<kind>/<name>.json`).
Built-in models ship bundled with the package; add or override models with your
own `./settings` directory. `imggen models` lists them; `--model` also accepts a
raw Hugging Face repo id or a local path. Weights are downloaded on first use.

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

To use a different quant level, add a manifest pointing at the desired `.gguf`
file (see below) — e.g. `settings/qwen-image/qwen-image-q8.json` with
`"hf_file": "Qwen_Image-Q8_0.gguf"`.

### Model definitions (`settings/`)

Models are registered once in a JSON manifest that records the download source
**and** the preferred settings, so the same definition reproduces on another
machine. This is how the built-ins are defined (bundled with the package) and
how you add your own — e.g. a Qwen-Image-Edit all-in-one checkpoint or a Civitai
SDXL fine-tune. Manifests are discovered in this order (first wins):

1. `./settings/<kind>/<name>.json` — your project's models (commit for reproducibility)
2. the package-bundled `settings/` — the built-in models

`<kind>` is a backend (`sd`, `qwen-image`, `qwen-image-edit`) and `<name>`
becomes the `--model` value. Only `source` is required:

```jsonc
// settings/qwen-image-edit/qwen-rapid-aio-v23.json
{
  "description": "Phr00t Qwen-Image-Edit Rapid AIO SFW v23 (distilled, ~4-step)",
  "source": { "url": "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v23/Qwen-Rapid-AIO-SFW-v23.safetensors" },
  "defaults": { "steps": 4, "cfg": 1.0, "negative": " " }
}
```

```bash
imggen qwen-image-edit -m qwen-rapid-aio-v23 -i photo.png -p "make it snow"
# downloads the checkpoint on first use; applies steps=4 / cfg=1.0 automatically
imggen qwen-image-edit -m qwen-rapid-aio-v23 -i photo.png -p "..." --steps 8   # flag overrides the default
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
`defaults` accepts `steps`, `cfg`, `width`, `height`, `negative`; an explicit
CLI flag always wins. Add `"gated": true` for repos that need an accepted
license / HF token. `imggen models` lists every discovered manifest.

See [`settings/README.md`](settings/README.md) for the full manifest schema and
more examples.

### Transparent generation (LayerDiffuse)

`see-through --method layerdiffuse` (the default when generating from a prompt)
produces a **native transparent RGBA image** with a real alpha channel via
[LayerDiffuse](https://github.com/lllyasviel/sd-forge-layerdiffuse), recovering
soft/semi-transparent edges (hair, glass, glow) that background removal cannot.
`--method matte` uses BiRefNet background removal instead; it is used
automatically for `--init` decomposition and for `--mode layers`.

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
| `--seed` | base seed (consecutive across a batch) |
| `-n, --num` | number of images |
| `-i, --init` | input image (img2img / edit / see-through base) |
| `--strength` | img2img denoising strength |
| `--device` / `--dtype` | `cuda`/`mps`/`cpu`, `bf16`/`fp16`/`fp32` (auto if unset) |
| `--offload` | CPU-offload to save VRAM |
| `--hf-token` | token for gated models (or set `HF_TOKEN`) |
| `--no-metadata` | do not embed parameters in the PNG |

## Gated models

SD3.5 requires accepting its license on the Hub. Provide a token via
`--hf-token`, the `HF_TOKEN` environment variable, or `huggingface-cli login`.
Defaults (`sdxl`, Qwen-Image) are ungated and need no token.
