"""Client-side remote execution: config + HTTP transport.

When a remote endpoint is configured (``imggen remote set HOST:PORT``), the
generate commands ship the :class:`~imggen.params.GenRequest` to an
``imggen serve`` daemon on a GPU host instead of running a backend locally. The
daemon returns the images and metadata; the client saves them, so output paths
stay a client-side concern (see :func:`imggen.runner.run_and_save`).

Only the standard library is used here (no torch / diffusers / PIL at import
time) so this module stays cheap to import — like :mod:`imggen.device`, keep it
that way, or shell completion and the light commands regress. ``PIL`` is
imported lazily inside :func:`run`, on the actual generation path only.

There is deliberately **no fallback**: when a remote is configured and cannot be
reached, generation raises :class:`RemoteError` and the CLI exits non-zero.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path

from .manifest import config_home
from .params import GenRequest


class RemoteError(RuntimeError):
    """Remote generation failed: unreachable, auth rejected, or a server error."""


# --- configuration ------------------------------------------------------

def config_path() -> Path:
    """Where the remote endpoint is stored (``<config_home>/remote.json``)."""
    return config_home() / "remote.json"


def load() -> dict | None:
    """Return the configured remote ``{"endpoint", "api_key"?}`` or ``None``.

    ``$IMGGEN_REMOTE`` (with optional ``$IMGGEN_API_KEY``) overrides the file,
    which is handy for one-off runs and CI.
    """
    env_ep = os.environ.get("IMGGEN_REMOTE")
    if env_ep:
        cfg = {"endpoint": env_ep.strip()}
        key = os.environ.get("IMGGEN_API_KEY")
        if key:
            cfg["api_key"] = key
        return cfg
    path = config_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not data.get("endpoint"):
        return None
    return data


def endpoint() -> str | None:
    """The configured endpoint string (``host:port``), or ``None`` if unset."""
    cfg = load()
    return cfg["endpoint"] if cfg else None


def save(ep: str, api_key: str | None) -> Path:
    """Persist ``ep`` (and optional ``api_key``) as the active remote."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"endpoint": ep.strip()}
    if api_key:
        data["api_key"] = api_key
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def clear() -> bool:
    """Remove the stored remote. Returns ``True`` if one existed."""
    path = config_path()
    if path.is_file():
        path.unlink()
        return True
    return False


# --- transport ----------------------------------------------------------

def _base_url(ep: str) -> str:
    ep = ep.strip().rstrip("/")
    if "://" not in ep:
        ep = "http://" + ep
    return ep


def _headers(cfg: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    key = cfg.get("api_key")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _resolve_cfg(ep: str | None, api_key: str | None) -> dict:
    if ep:
        cfg = {"endpoint": ep}
        if api_key:
            cfg["api_key"] = api_key
        return cfg
    cfg = load()
    if not cfg:
        raise RemoteError("no remote endpoint configured (imggen remote set HOST:PORT)")
    return cfg


def ping(ep: str | None = None, api_key: str | None = None, timeout: float = 5.0) -> dict:
    """Probe ``/health`` and return the server info dict; raise on failure."""
    cfg = _resolve_cfg(ep, api_key)
    url = _base_url(cfg["endpoint"]) + "/health"
    request = urllib.request.Request(url, headers=_headers(cfg), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RemoteError(f"cannot reach imggen server at {cfg['endpoint']}: {exc}") from exc


def _gen_timeout() -> float | None:
    raw = os.environ.get("IMGGEN_REMOTE_TIMEOUT", "3600")
    try:
        val = float(raw)
    except ValueError:
        return 3600.0
    return None if val <= 0 else val


def run(req: GenRequest):
    """Run ``req`` on the configured remote and return the backend result list.

    Returns the same shape as :func:`imggen.pipelines.run`:
    ``[(PIL.Image, metadata dict, path_hint), ...]``. Raises :class:`RemoteError`
    on any transport / auth / server failure — there is no local fallback.
    """
    cfg = _resolve_cfg(None, None)

    # Fast reachability check so a misconfigured / down server fails quickly
    # rather than blocking on the (possibly long) generation request.
    ping(timeout=5)

    payload = _encode_request(req)
    body = json.dumps(payload).encode()
    url = _base_url(cfg["endpoint"]) + "/generate"
    request = urllib.request.Request(url, data=body, headers=_headers(cfg), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=_gen_timeout()) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise RemoteError(
            f"remote generation failed ({exc.code}): {_http_error_detail(exc)}"
        ) from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RemoteError(f"remote generation failed: {exc}") from exc
    return _decode_results(req, data)


def _encode_request(req: GenRequest) -> dict:
    """Serialize ``req`` for the wire, shipping the ``--init`` image inline."""
    data = asdict(req)
    payload: dict = {"request": data}
    init = data.get("init")
    if init:
        p = Path(init)
        if not p.is_file():
            raise RemoteError(f"init image not found: {init}")
        payload["init_image"] = base64.b64encode(p.read_bytes()).decode()
        payload["init_name"] = p.name
        data["init"] = None  # the server writes a temp file and fills this in
    return payload


def _decode_results(req: GenRequest, data: dict):
    from io import BytesIO

    from PIL import Image

    results = []
    for item in data.get("results", []):
        image = Image.open(BytesIO(base64.b64decode(item["image"])))
        image.load()
        meta = item.get("meta") or {}
        # The server records its own temp path for the init image; show the
        # client's original path in the saved metadata instead.
        if req.init and meta.get("init"):
            meta["init"] = req.init
        results.append((image, meta, item.get("hint")))
    return results


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode())
        return body.get("error") or body.get("detail") or str(exc)
    except Exception:
        return str(exc)
