# Model catalog

The presets in this directory (`models/<kind>/<name>.json`) make up the public
catalog behind **`imggen pull`**. Each one pins where to download a model *and*
its author-recommended generation settings, so a single command installs the
preset and fetches the weights.

| kind | name | model |
| --- | --- | --- |
| `sd` | `wai-illustrious-sdxl` | WAI-illustrious-SDXL v17 (anime/illustration, Illustrious) |
| `sd` | `pony-diffusion-v6-xl` | Pony Diffusion V6 XL (versatile SDXL) |
| `qwen-image-edit` | `qwen-image-edit-rapid-AIO` | Phr00t Qwen-Image-Edit Rapid AIO NSFW v23 (distilled, ~4-step) |

See `../src/imggen/settings/README.md` for the preset/manifest schema.

## Using the catalog

```bash
imggen pull --list                       # everything in the catalog
imggen pull sd --list                    # just the sd presets
imggen pull sd pony-diffusion-v6-xl      # install preset + download weights now
imggen sd -m pony-diffusion-v6-xl -p "a knight in a snowy forest"
```

`pull` copies the preset into `~/.config/imggen/settings/<kind>/` and downloads
the checkpoint immediately. **If the download fails the preset is rolled back**,
so you never end up with a preset whose weights are missing. Re-run with
`--force` to overwrite an already-installed preset.

## Where `pull` reads from

By default the catalog is fetched over HTTPS from this repo on GitHub
(`https://github.com/matorix/imggen`, branch `main`). Override with:

- `IMGGEN_CATALOG_REPO` — a different `owner/repo` (default `matorix/imggen`)
- `IMGGEN_CATALOG_REF` — a branch or tag (default `main`)
- `IMGGEN_CATALOG_DIR` — read presets from a **local** directory laid out as
  `<dir>/<kind>/<name>.json` instead of GitHub (handy for testing a preset
  before pushing it):

  ```bash
  IMGGEN_CATALOG_DIR="$PWD/models" imggen pull sd wai-illustrious-sdxl
  ```

> New presets are only reachable by `pull` once they are committed and pushed to
> the catalog repo/branch (or exposed via `IMGGEN_CATALOG_DIR`).

## Notes

- **Downloads:** the `sd` presets pull single-file `.safetensors` from Civitai
  via `wget`; the Qwen preset streams from Hugging Face. If a Civitai download
  needs an API key, add `"token_env": "CIVITAI_TOKEN"` to the preset `source`
  and export that variable — an auth failure then cancels the pull cleanly.
- **Clip skip:** nothing to set — for SDXL, diffusers already reads the
  penultimate CLIP layer (= "clip skip 2"), which is what these models want.
- **NSFW:** the WAI and Rapid-AIO presets point at NSFW-capable checkpoints. WAI
  filters by adding `nsfw` to the negative prompt if you want SFW output.
