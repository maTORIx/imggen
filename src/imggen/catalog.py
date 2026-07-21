"""The public model catalog behind ``imggen pull``.

Presets are published under ``models/<kind>/<name>.json`` in the project's
public GitHub repo. ``pull`` fetches one over HTTPS, installs it into the user
settings dir, and downloads its weights (see :mod:`imggen.manifest`).

The catalog location is configurable:

* ``IMGGEN_CATALOG_DIR`` — read presets from a local directory laid out as
  ``<dir>/<kind>/<name>.json`` (used for development, before pushing to GitHub).
* ``IMGGEN_CATALOG_REPO`` / ``IMGGEN_CATALOG_REF`` — override the GitHub
  ``owner/repo`` and branch/tag (defaults below).

Only the standard library is used (``urllib``), so listing/fetching the catalog
never pulls in extra dependencies.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

# Public catalog defaults (see https://github.com/matorix/imggen).
DEFAULT_REPO = "matorix/imggen"
DEFAULT_REF = "main"
_SUBDIR = "models"
_UA = "imggen-pull"


class CatalogError(Exception):
    """A catalog lookup or download failed."""


class CatalogNotFound(CatalogError):
    """The requested preset does not exist in the catalog."""


def _repo() -> str:
    return os.environ.get("IMGGEN_CATALOG_REPO", DEFAULT_REPO)


def _ref() -> str:
    return os.environ.get("IMGGEN_CATALOG_REF", DEFAULT_REF)


def local_dir() -> Path | None:
    """A local catalog directory (``IMGGEN_CATALOG_DIR``), or ``None`` for GitHub."""
    d = os.environ.get("IMGGEN_CATALOG_DIR")
    return Path(d) if d else None


def source_desc() -> str:
    """Human-readable description of where the catalog is being read from."""
    d = local_dir()
    return str(d) if d else f"github:{_repo()}@{_ref()}/{_SUBDIR}"


def _raw_url(kind: str, name: str) -> str:
    return f"https://raw.githubusercontent.com/{_repo()}/{_ref()}/{_SUBDIR}/{kind}/{name}.json"


def _contents_url(kind: str) -> str:
    return f"https://api.github.com/repos/{_repo()}/contents/{_SUBDIR}/{kind}?ref={_ref()}"


def _get(url: str, accept: str | None = None, timeout: float = 30) -> bytes:
    headers = {"User-Agent": _UA}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError:
        raise  # caller distinguishes 404 from other statuses
    except urllib.error.URLError as exc:
        raise CatalogError(f"could not reach the catalog ({url}): {exc.reason}") from exc


def fetch_preset(kind: str, name: str) -> dict:
    """Return the catalog preset ``models/<kind>/<name>.json`` as a dict."""
    d = local_dir()
    if d is not None:
        path = d / kind / f"{name}.json"
        if not path.is_file():
            raise CatalogNotFound(f"{kind}/{name} not found in catalog {d}")
        return json.loads(path.read_text())

    try:
        raw = _get(_raw_url(kind, name))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise CatalogNotFound(
                f"{kind}/{name} not found in catalog {source_desc()}"
            ) from exc
        raise CatalogError(f"catalog fetch failed ({exc.code}) for {kind}/{name}") from exc
    return json.loads(raw)


def list_kind(kind: str) -> list[tuple[str, str | None]]:
    """List ``(name, description)`` for every preset published under ``kind``."""
    d = local_dir()
    if d is not None:
        out: list[tuple[str, str | None]] = []
        kdir = d / kind
        if kdir.is_dir():
            for jf in sorted(kdir.glob("*.json")):
                try:
                    data = json.loads(jf.read_text())
                except json.JSONDecodeError:
                    continue
                out.append((jf.stem, data.get("description")))
        return out

    try:
        raw = _get(_contents_url(kind), accept="application/vnd.github+json")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return []  # kind has no directory in the catalog
        raise CatalogError(
            f"catalog listing failed ({exc.code}) for {kind}; GitHub API may be "
            f"rate-limited — try again shortly"
        ) from exc

    entries = json.loads(raw)
    out = []
    for entry in entries:
        name = entry.get("name", "")
        if entry.get("type") != "file" or not name.endswith(".json"):
            continue
        stem = name[: -len(".json")]
        desc = None
        dl = entry.get("download_url")
        if dl:  # best-effort description (raw file, not rate-limited)
            try:
                desc = json.loads(_get(dl)).get("description")
            except Exception:
                desc = None
        out.append((stem, desc))
    return out


def list_names(kind: str, timeout: float = 30) -> list[str]:
    """Just the preset names published under ``kind`` — one request, for tab completion.

    Unlike :func:`list_kind` this never fetches descriptions, and it swallows all
    errors (returning ``[]``) so completion stays fast and never raises into the
    shell. Pass a short ``timeout`` when driving interactive completion.
    """
    d = local_dir()
    if d is not None:
        kdir = d / kind
        return sorted(p.stem for p in kdir.glob("*.json")) if kdir.is_dir() else []
    try:
        raw = _get(_contents_url(kind), accept="application/vnd.github+json", timeout=timeout)
        entries = json.loads(raw)
    except Exception:
        return []
    return sorted(
        entry["name"][: -len(".json")]
        for entry in entries
        if entry.get("type") == "file" and entry.get("name", "").endswith(".json")
    )


def list_all(kinds) -> dict[str, list[tuple[str, str | None]]]:
    """Map each kind to its catalog presets (kinds with none are omitted)."""
    result = {}
    for kind in kinds:
        items = list_kind(kind)
        if items:
            result[kind] = items
    return result
