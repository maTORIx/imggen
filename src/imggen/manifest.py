"""Model manifests: reproducible model definitions under the imggen config dir.

A manifest is a small JSON file at ``<config>/settings/<kind>/<name>.json``
(``<config>`` = ``~/.config/imggen`` by default) that records **where to get a
model** and **its default generation settings**. It is the single source of
truth for models: the package-bundled built-ins are copied there once on first
use (see :func:`ensure_seeded`), and ``imggen <kind> --save --alias <name>``
writes new presets there too.

The directory ``<kind>`` (``sd`` | ``qwen-image`` | ``qwen-image-edit``) is
authoritative; ``<name>`` (the file stem) becomes the ``--model`` value.

Schema (all keys but ``source`` optional)::

    {
      "description": "human-readable note",
      "source": { ... },           # where to download from (see Source)
      "load":    { ... },          # how to load it (optional; inferred otherwise)
      "defaults": {                # applied when the matching CLI flag is unset
        "steps": 4, "cfg": 1.0, "width": 1024, "height": 1024,
        "negative": " ", "strength": 0.8
      }
    }

``source`` is one of:

* ``{"url": "..."}`` — any download URL. A ``huggingface.co/<repo>/resolve/...``
  URL is auto-routed through :func:`huggingface_hub.hf_hub_download` (cache,
  auth and resume come for free; ``?download=true`` is stripped). Any other URL
  is streamed to the imggen cache. Optional ``filename``, ``sha256`` (integrity
  check) and ``token_env`` (name of an env var whose value is sent as
  ``Authorization: Bearer``, e.g. a Civitai token).
* ``{"hf_repo": "...", "hf_file": "...", "revision": "main", "subfolder": null}``
  — an explicit file inside a Hugging Face repo.
* ``{"hf_repo": "..."}`` — a full diffusers repo folder (loaded with
  ``from_pretrained``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# ``<kind>`` directory names that may hold manifests.
KINDS = ("sd", "qwen-image", "qwen-image-edit")


def config_home() -> Path:
    """The imggen config directory (``$IMGGEN_HOME`` / ``$XDG_CONFIG_HOME`` aware).

    Defaults to ``~/.config/imggen``. ``$IMGGEN_HOME`` overrides it wholesale
    (useful for a project-local set of presets); otherwise ``$XDG_CONFIG_HOME``
    is honoured.
    """
    root = os.environ.get("IMGGEN_HOME")
    if root:
        return Path(root)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "imggen"


def user_settings_dir() -> Path:
    """The single manifest root: ``<config_home>/settings``."""
    return config_home() / "settings"


def _bundled_settings_dir() -> Path:
    """The built-in presets shipped inside the package (the seed source)."""
    return Path(__file__).parent / "settings"


def seed_settings(force: bool = False) -> list[Path]:
    """Copy the package-bundled built-in presets into :func:`user_settings_dir`.

    Existing files are left untouched unless ``force`` is set, so a user's own
    edits and presets are never clobbered. Returns the list of files written.
    """
    src = _bundled_settings_dir()
    dst = user_settings_dir()
    copied: list[Path] = []
    if not src.is_dir():
        return copied
    for jf in sorted(src.rglob("*.json")):
        target = dst / jf.relative_to(src)
        if target.exists() and not force:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(jf.read_text())
        copied.append(target)
    return copied


def ensure_seeded() -> None:
    """Seed the built-in presets on first use (when the root does not exist)."""
    if not user_settings_dir().exists():
        seed_settings(force=False)


def settings_dirs() -> list[Path]:
    """The manifest search roots (a single, user-owned directory).

    Seeding runs lazily so the built-in models are always available even on a
    fresh machine.
    """
    ensure_seeded()
    return [user_settings_dir()]


def cache_dir(kind: str | None = None) -> Path:
    """Where URL-downloaded weights are stored (``$IMGGEN_CACHE`` overrides).

    Plain-URL downloads are namespaced by ``kind`` so each backend keeps its
    checkpoints separate: ``~/.cache/imggen/models/<kind>/``.
    """
    root = os.environ.get("IMGGEN_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "imggen"
    models = base / "models"
    return models / kind if kind else models


@dataclass
class Source:
    url: str | None = None
    hf_repo: str | None = None
    hf_file: str | None = None
    revision: str = "main"
    subfolder: str | None = None
    filename: str | None = None  # local name for a plain-URL download
    sha256: str | None = None
    token_env: str | None = None  # env var -> "Authorization: Bearer <value>"


@dataclass
class LoadSpec:
    base_repo: str | None = None
    config_repo: str | None = None
    config_subfolder: str = "transformer"


@dataclass
class ModelManifest:
    kind: str
    name: str
    source: Source
    description: str | None = None
    gated: bool = False  # repo needs an accepted license / HF token
    load: LoadSpec = field(default_factory=LoadSpec)
    defaults: dict = field(default_factory=dict)

    def summary(self) -> str:
        """One-line description of where this model comes from."""
        s = self.source
        if s.url:
            return s.url
        if s.hf_repo and s.hf_file:
            sub = f"{s.subfolder}/" if s.subfolder else ""
            return f"{s.hf_repo}/{sub}{s.hf_file}"
        if s.hf_repo:
            return s.hf_repo
        return "<no source>"


# --- discovery ----------------------------------------------------------


def manifest_path(kind: str, name: str) -> Path | None:
    """First existing ``<root>/<kind>/<name>.json`` across the search roots."""
    for root in settings_dirs():
        path = root / kind / f"{name}.json"
        if path.is_file():
            return path
    return None


def load_manifest(kind: str, name: str | None) -> ModelManifest | None:
    """Read ``settings/<kind>/<name>.json`` if it exists (no download)."""
    if not name:
        return None
    path = manifest_path(kind, name)
    if path is None:
        return None
    return _parse(kind, name, json.loads(path.read_text()))


def list_manifests() -> dict[str, list[ModelManifest]]:
    """All discoverable manifests, grouped by kind."""
    seen: dict[tuple[str, str], ModelManifest] = {}
    for root in settings_dirs():
        if not root.is_dir():
            continue
        for kind_dir in sorted(root.iterdir()):
            if not kind_dir.is_dir():
                continue
            for jf in sorted(kind_dir.glob("*.json")):
                key = (kind_dir.name, jf.stem)
                if key in seen:
                    continue  # already provided by a higher-precedence root
                try:
                    seen[key] = _parse(kind_dir.name, jf.stem, json.loads(jf.read_text()))
                except (json.JSONDecodeError, ValueError):
                    continue  # skip malformed files rather than abort the listing

    out: dict[str, list[ModelManifest]] = {}
    for (kind, _), man in sorted(seen.items()):
        out.setdefault(kind, []).append(man)
    return out


def _parse(kind: str, name: str, data: dict) -> ModelManifest:
    src = data.get("source") or {}
    if not isinstance(src, dict) or not (src.get("url") or src.get("hf_repo")):
        raise ValueError(f"manifest {kind}/{name}: 'source' needs a 'url' or 'hf_repo'")
    load = data.get("load") or {}
    return ModelManifest(
        kind=kind,
        name=name,
        description=data.get("description"),
        gated=bool(data.get("gated", False)),
        source=Source(
            url=src.get("url"),
            hf_repo=src.get("hf_repo"),
            hf_file=src.get("hf_file"),
            revision=src.get("revision", "main"),
            subfolder=src.get("subfolder"),
            filename=src.get("filename"),
            sha256=src.get("sha256"),
            token_env=src.get("token_env"),
        ),
        load=LoadSpec(
            base_repo=load.get("base_repo"),
            config_repo=load.get("config_repo"),
            config_subfolder=load.get("config_subfolder", "transformer"),
        ),
        defaults=data.get("defaults") or {},
    )


# --- materialization (download) -----------------------------------------

# ``https://huggingface.co/<org>/<repo>/resolve/<revision>/<path...>[?query]``
_HF_URL = re.compile(
    r"^https?://huggingface\.co/(?P<repo>[^/]+/[^/]+)/resolve/"
    r"(?P<rev>[^/]+)/(?P<path>.+?)(?:\?.*)?$"
)


def parse_hf_url(url: str) -> tuple[str, str, str] | None:
    """Return ``(repo, revision, path)`` for an HF ``resolve`` URL, else None."""
    m = _HF_URL.match(url)
    if not m:
        return None
    return m.group("repo"), m.group("rev"), m.group("path")


@dataclass
class Materialized:
    ref: str  # local file path, or a repo id when ``is_repo``
    is_repo: bool  # True -> ``ref`` is an HF repo folder for from_pretrained


def materialize(manifest: ModelManifest, token: str | None = None) -> Materialized:
    """Ensure the model's weights are present locally; return how to reference them."""
    from huggingface_hub import hf_hub_download

    s = manifest.source

    if s.url:
        hf = parse_hf_url(s.url)
        if hf:  # HF resolve URL -> cached/auth/resumable download
            repo, revision, path = hf
            subfolder, filename = os.path.split(path)
            local = hf_hub_download(
                repo, filename, subfolder=subfolder or None,
                revision=revision, token=token,
            )
            return Materialized(ref=local, is_repo=False)
        # Plain URL (Civitai etc.)
        local = _download_url(s.url, s.filename, s.token_env, s.sha256, manifest.kind)
        return Materialized(ref=local, is_repo=False)

    if s.hf_repo and s.hf_file:
        local = hf_hub_download(
            s.hf_repo, s.hf_file, subfolder=s.subfolder,
            revision=s.revision, token=token,
        )
        return Materialized(ref=local, is_repo=False)

    # Bare repo folder -> loaded with from_pretrained.
    return Materialized(ref=s.hf_repo, is_repo=True)


def _download_url(
    url: str,
    filename: str | None,
    token_env: str | None,
    sha256: str | None,
    kind: str | None = None,
) -> str:
    """Download ``url`` into the imggen cache with ``wget``; skip if already valid.

    ``wget`` (assumed present) gives resumable transfers (``-c``) and a progress
    bar for free; a ``token_env`` becomes an ``Authorization: Bearer`` header for
    private/gated hosts such as Civitai. The file lands under the ``kind``'s
    cache subdirectory (see :func:`cache_dir`).
    """
    import shutil
    import subprocess

    name = filename or os.path.basename(url.split("?", 1)[0]) or "model.bin"
    dest = cache_dir(kind) / name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and (sha256 is None or _sha256(dest) == sha256.lower()):
        return str(dest)

    if shutil.which("wget") is None:
        raise RuntimeError("wget is required to download URL-sourced models but was not found")

    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = ["wget", "--continue", "--tries=3", "--show-progress", "-O", str(tmp)]
    if token_env:
        tok = os.environ.get(token_env)
        if not tok:
            raise ValueError(f"token_env '{token_env}' is set in the manifest but empty")
        cmd.append(f"--header=Authorization: Bearer {tok}")
    cmd.append(url)
    subprocess.run(cmd, check=True)

    if sha256 is not None:
        got = _sha256(tmp)
        if got != sha256.lower():
            tmp.unlink(missing_ok=True)
            raise ValueError(f"sha256 mismatch for {name}: expected {sha256}, got {got}")

    tmp.replace(dest)
    return str(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
