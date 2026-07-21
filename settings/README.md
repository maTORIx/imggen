# `settings/` — model manifests

A **manifest** is a small JSON file that defines one model: **where to download
it** and **its preferred generation settings**. Because the download source and
the defaults live together in a committed file, the same `settings/` directory
reproduces a model on any machine — `imggen <kind> -m <name>` fetches the
weights on first use and applies the recorded defaults automatically.

## Layout

```
settings/<kind>/<name>.json
```

- `<kind>` — the backend the model plugs into: `sd`, `qwen-image`, or
  `qwen-image-edit`. (The `see-through` backend builds on `sd`.)
- `<name>` — the file stem; this is the value you pass to `--model` / `-m`.

Example: `settings/qwen-image-edit/qwen-rapid-aio-v23.json` is used with
`imggen qwen-image-edit -m qwen-rapid-aio-v23 ...`.

### Discovery & precedence

Manifests are looked up in this order — **first match wins**:

1. `./settings/<kind>/<name>.json` — this directory, relative to your current
   working dir. Add models here, or shadow a built-in by reusing its name.
2. The package-bundled `settings/` shipped inside `imggen` — the built-in
   models (`sdxl`, `sd3.5`, `qwen-image`, `qwen-image-edit`, …), always
   available regardless of where you run from.

`imggen models` lists every discovered manifest and its source.

> `-m` also accepts a raw Hugging Face repo id (`stabilityai/...`) or a local
> path (`./checkpoints/model.safetensors`) with no manifest at all — a manifest
> is only needed to pin a source *and* attach defaults under a short name.

## Schema

Only `source` is required; every other key is optional.

```jsonc
{
  "description": "human-readable note (shown by `imggen models`)",
  "gated": false,          // true → repo needs an accepted license / HF token
  "source": { ... },       // where to download from — see below
  "load":   { ... },       // how to load it (optional; usually inferred)
  "defaults": {            // applied only when the matching CLI flag is unset
    "steps": 30,
    "cfg": 7.0,
    "width": 1024,
    "height": 1024,
    "negative": " ",
    "strength": 0.8
  }
}
```

An explicit CLI flag **always** overrides the corresponding `defaults` value.

### `source`

Pick exactly one shape:

| shape | meaning |
| --- | --- |
| `{ "hf_repo": "org/repo" }` | a full diffusers repo folder, loaded with `from_pretrained`. |
| `{ "hf_repo": "org/repo", "hf_file": "model.gguf" }` | one explicit file inside a Hub repo. Optional `revision` (default `"main"`) and `subfolder`. |
| `{ "url": "https://…" }` | any download URL. A `huggingface.co/<repo>/resolve/<rev>/<path>` URL is auto-routed through the Hub (cached, authenticated, resumable). Any other URL (e.g. Civitai) is streamed into the imggen cache with `wget`. |

Extra keys accepted on a `{ "url": ... }` source:

- `filename` — local filename for a plain-URL download (defaults to the URL's basename).
- `sha256` — checksum verified after download; a mismatch aborts.
- `token_env` — name of an environment variable whose value is sent as
  `Authorization: Bearer <value>` (for private/gated hosts such as Civitai).

### `load`

Usually inferred and omitted. Relevant mainly for single-file Qwen transformers
(`.safetensors` / `.gguf`), where the file is loaded as the transformer and the
text encoder, VAE and scheduler come from a base repo:

- `base_repo` — the base diffusers repo (default `Qwen/Qwen-Image` or
  `Qwen/Qwen-Image-Edit-2509` depending on `<kind>`).
- `config_repo` / `config_subfolder` — override where the transformer config is
  read from (`config_subfolder` defaults to `"transformer"`).

## Examples

Full diffusers repo with defaults (built-in `sdxl`):

```jsonc
// settings/sd/sdxl.json
{
  "description": "Stable Diffusion XL base 1.0 (default `sd` model)",
  "source": { "hf_repo": "stabilityai/stable-diffusion-xl-base-1.0" },
  "defaults": { "steps": 30, "cfg": 7.0 }
}
```

Gated repo (needs an HF token / accepted license):

```jsonc
// settings/sd/sd3.5.json
{
  "description": "Stable Diffusion 3.5 Large",
  "source": { "hf_repo": "stabilityai/stable-diffusion-3.5-large" },
  "gated": true,
  "defaults": { "steps": 30, "cfg": 7.0 }
}
```

GGUF-quantized Qwen transformer + full-precision base (built-in `qwen-image`):

```jsonc
// settings/qwen-image/qwen-image.json
{
  "description": "Qwen-Image — GGUF Q4_K_M transformer + Qwen/Qwen-Image base",
  "source": { "hf_repo": "QuantStack/Qwen-Image-GGUF", "hf_file": "Qwen_Image-Q4_K_M.gguf" },
  "load": { "base_repo": "Qwen/Qwen-Image" },
  "defaults": { "steps": 50, "cfg": 4.0 }
}
```

Single-file checkpoint from a download URL, with distilled-model defaults:

```jsonc
// settings/qwen-image-edit/qwen-rapid-aio-v23.json
{
  "description": "Phr00t Qwen-Image-Edit Rapid AIO SFW v23 (distilled, ~4-step)",
  "source": {
    "url": "https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v23/Qwen-Rapid-AIO-SFW-v23.safetensors"
  },
  "load": { "base_repo": "Qwen/Qwen-Image-Edit-2509" },
  "defaults": { "steps": 4, "cfg": 1.0, "negative": " " }
}
```

```bash
imggen qwen-image-edit -m qwen-rapid-aio-v23 -i photo.png -p "make it snow"
# downloads on first use; applies steps=4 / cfg=1.0 automatically
imggen qwen-image-edit -m qwen-rapid-aio-v23 -i photo.png -p "…" --steps 8
# an explicit --steps overrides the default
```

## Where weights are cached

- Hub sources (`hf_repo` / HF `resolve` URLs) use the standard Hugging Face
  cache (`~/.cache/huggingface`, or `$HF_HOME`).
- Plain-URL downloads go to `~/.cache/imggen/models` (override with
  `$IMGGEN_CACHE`).
