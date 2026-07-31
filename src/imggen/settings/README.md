# imggen presets (model manifests)

A **preset** (a.k.a. manifest) is a small JSON file that defines one model:
**where to download it** and **its preferred generation settings**. Because the
download source and the defaults live together, a preset reproduces a model on
any machine — `imggen <kind> -m <name>` fetches the weights on first use and
applies the recorded defaults automatically.

## Location — a single directory

```
~/.config/imggen/settings/<kind>/<name>.json
```

This is the **one** place imggen reads models from (override the root with
`$IMGGEN_HOME`, or `$XDG_CONFIG_HOME`). The built-in models shipped with the
package are copied here **once on first use**; after that the directory is fully
yours to edit, add to, or delete from.

- `<kind>` — the backend the model plugs into: `sd`, `qwen-image`,
  `qwen-image-edit`, or `background-removal`. (The `see-through` backend resolves
  its base model through `sd`, so its presets live under `sd/`.)
- `<name>` — the file stem; this is the value you pass to `--model` / `-m`.

Re-seed the built-ins at any time with `imggen init` (add `--force` to overwrite
built-ins you have edited; your own presets are never touched). `imggen models`
lists every discovered preset and its source.

> `-m` also accepts a raw Hugging Face repo id (`stabilityai/...`) or a local
> path (`./checkpoints/model.safetensors`) with no preset at all — a preset is
> only needed to pin a source *and* attach defaults under a short name.

## Creating a preset from the CLI

Instead of hand-writing JSON, run a command with the options you want and add
`--save --alias NAME`. Nothing is generated; the preset is written to
`~/.config/imggen/settings/<kind>/NAME.json`.

```bash
# Save an 8-step Euler SDXL preset, then use it by name
imggen sd -m stabilityai/stable-diffusion-xl-base-1.0 \
          --steps 8 --cfg 2.0 --sampler euler --save --alias sdxl-fast
imggen sd -m sdxl-fast -p "a red fox"        # steps=8, cfg=2.0, sampler=euler

# Derive a preset from an existing one (inherits its source + defaults)
imggen qwen-image -m qwen-image --steps 8 --save --alias qwen-fast

# Pin a GGUF quant via its HF resolve URL (covers repo + file + subfolder)
imggen qwen-image \
  -m https://huggingface.co/QuantStack/Qwen-Image-GGUF/resolve/main/Qwen_Image-Q8_0.gguf \
  --steps 40 --save --alias qwen-q8
```

Only the flags you **explicitly typed** are recorded into `defaults`; when
`--model` names an existing preset (or is omitted, meaning the kind's default),
that preset's `source`, `load` and `defaults` are inherited and your flags are
layered on top. `--desc "note"` stores a description.

The `--model` source is inferred as: an existing preset name → inherit it; an
`http(s)://` URL → `{"url": ...}`; an `org/repo` id → `{"hf_repo": ...}`. Local
paths are not portable and are rejected when saving (recall them directly with
`-m <path>`).

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
    "strength": 0.8,
    "sampler": "euler",
    "mask_grow": 0,          // --mask: dilate (+) / erode (-) the mask, px
    "mask_blur": 4.0,        // --mask: feather the mask edge, px
    "prompt_prefix": "masterpiece, best quality",  // prepended to --prompt
    "prompt_suffix": "cinematic lighting"          // appended to --prompt
  }
}
```

An explicit CLI flag **always** overrides the corresponding `defaults` value.
`sampler` accepts any name from `imggen samplers` (e.g. `euler`, `ddim`,
`dpm++_2m_karras`, `unipc`, `flowmatch`).

Only the keys a backend reads have any effect. A `background-removal` preset,
for instance, uses just `width` / `height` — the resolution its matting model
runs at (1024 for `lucida` / `rmbg-1.4`, 2048 for `birefnet-hr`), which is
independent of the output size:

```jsonc
// ~/.config/imggen/settings/background-removal/lucida.json
{
  "description": "Lucida — BiRefNet-HR fine-tune for soft alpha",
  "source": { "hf_repo": "egeorcun/lucida" },
  "defaults": { "width": 1024, "height": 1024 }
}
```

### Positive-prompt templates (`prompt_prefix` / `prompt_suffix`)

Many community checkpoints expect a fixed block of quality tags around your
prompt — Pony's `score_9, score_8_up, ...`, an Illustrious model's
`masterpiece, best quality`, and so on. Store that boilerplate in the preset and
it is wrapped around every `--prompt` automatically:

```
final positive prompt = prompt_prefix + ", " + <your --prompt> + ", " + prompt_suffix
```

Empty segments and stray commas are trimmed, so a template that already ends in a
comma still joins cleanly. Override per-run with `--prompt-prefix` /
`--prompt-suffix` (both are also recorded by `--save`). `negative` is the
matching default for the negative prompt (overridden by `--negative`).

> **Clip skip on SDXL:** you do *not* need a clip-skip setting for SDXL/Pony/
> Illustrious models. diffusers already reads the **penultimate** CLIP layer for
> SDXL by default, which is exactly what A1111/ComfyUI call "clip skip 2".

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

GGUF-quantized Qwen transformer + full-precision base (built-in `qwen-image`):

```jsonc
// ~/.config/imggen/settings/qwen-image/qwen-image.json
{
  "description": "Qwen-Image — GGUF Q4_K_M transformer + Qwen/Qwen-Image base",
  "source": { "hf_repo": "QuantStack/Qwen-Image-GGUF", "hf_file": "Qwen_Image-Q4_K_M.gguf" },
  "load": { "base_repo": "Qwen/Qwen-Image" },
  "defaults": { "steps": 50, "cfg": 4.0 }
}
```

Single-file checkpoint from a download URL, with distilled-model defaults
(built-in `qwen-rapid-aio-v23`):

```jsonc
// ~/.config/imggen/settings/qwen-image-edit/qwen-rapid-aio-v23.json
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

Civitai SDXL checkpoint with its recommended positive/negative templates (the
`models/sd/` catalog ships ready-to-use WAI-illustrious and Pony presets —
`imggen pull sd pony-diffusion-v6-xl`):

```jsonc
// ~/.config/imggen/settings/sd/wai-illustrious-sdxl.json
{
  "description": "WAI-illustrious-SDXL (Illustrious). VAE baked in.",
  "source": {
    "url": "https://civitai.com/api/download/models/2883731",
    "filename": "wai-illustrious-sdxl.safetensors"  // give the cached file a .safetensors name
  },
  "defaults": {
    "steps": 28, "cfg": 6.0, "sampler": "euler_a", "width": 1024, "height": 1344,
    "prompt_prefix": "masterpiece, best quality, amazing quality",
    "negative": "bad quality, worst quality, worst detail, sketch, censor"
  }
}
```

```bash
imggen sd -m wai-illustrious-sdxl -p "1girl, city street at night"
# actual prompt sent: "masterpiece, best quality, amazing quality, 1girl, city street at night"
```

A plain Civitai `api/download` URL has no file extension, so set `filename` to a
`.safetensors` name — the `sd` backend needs that extension to load it as a
single-file checkpoint. If the download host needs an API key, add
`"token_env": "CIVITAI_TOKEN"` and export that variable.

## Where weights are cached

- Hub sources (`hf_repo` / HF `resolve` URLs) use the standard Hugging Face
  cache (`~/.cache/huggingface`, or `$HF_HOME`).
- Plain-URL downloads go to `~/.cache/imggen/models/<kind>/` (override the root
  with `$IMGGEN_CACHE`).
