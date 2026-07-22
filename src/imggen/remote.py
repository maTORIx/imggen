"""Client-side remote execution: config + HTTP transport.

When a remote endpoint is configured (``imggen remote set HOST[:PORT]``), the
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
import socket
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

from .manifest import config_home
from .params import GenRequest

#: Port ``imggen serve`` listens on by default; assumed when an endpoint omits
#: one, so ``imggen remote set 192.168.1.50`` targets the daemon directly.
DEFAULT_PORT = 7863

# Invisible characters that survive ``str.strip()`` but make ``getaddrinfo``
# fail with ``[Errno -2]`` when they ride along on a copy-pasted address (a
# classic "curl works but imggen doesn't" cause). Stripped from the endpoint.
_INVISIBLE = (
    "\u200b\u200c\u200d\u2060\ufeff\u200e\u200f\u00a0"  # zero-width/joiners, BOM, marks, NBSP
)


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
    """Persist a normalized ``ep`` (and optional ``api_key``) as the active remote.

    Raises :class:`ValueError` (via :func:`normalize_endpoint`) if ``ep`` has no
    parseable host, so a typo is caught at ``imggen remote set`` time.
    """
    endpoint = normalize_endpoint(ep)
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"endpoint": endpoint}
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

def normalize_endpoint(ep: str) -> str:
    """Return a clean ``scheme://host:port`` base URL for *ep*.

    Accepts ``host``, ``host:port``, ``http://host:port``, or a full URL with a
    path. A missing scheme defaults to ``http``; a missing port on an ``http``
    endpoint defaults to :data:`DEFAULT_PORT` (the ``imggen serve`` default), so
    the port can be omitted for a default deployment. Any path/query/fragment is
    dropped so ``/health`` and ``/generate`` append cleanly, and stray
    copy-paste characters (spaces, zero-width marks, wrapping quotes) are
    removed — those are a common cause of ``getaddrinfo`` ``[Errno -2]`` even
    when the same address works under ``curl``.

    Raises :class:`ValueError` if no host can be parsed.
    """
    ep = "".join(c for c in ep if c not in _INVISIBLE).strip().strip("<>\"'").strip()
    if not ep:
        raise ValueError("empty remote endpoint")
    if "://" not in ep:
        ep = "http://" + ep
    parts = urlsplit(ep)
    host = parts.hostname
    if not host:
        raise ValueError(f"could not parse a host from remote endpoint {ep!r}")
    try:
        port = parts.port
    except ValueError as exc:  # e.g. non-numeric port
        raise ValueError(f"invalid port in remote endpoint {ep!r}: {exc}") from exc
    scheme = parts.scheme or "http"
    if port is None and scheme == "http":
        port = DEFAULT_PORT
    hostpart = f"[{host}]" if ":" in host else host  # re-bracket IPv6 literals
    return f"{scheme}://{hostpart}" if port is None else f"{scheme}://{hostpart}:{port}"


def _base_url(ep: str) -> str:
    return normalize_endpoint(ep)


def _headers(cfg: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    key = cfg.get("api_key")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _explain(exc: BaseException) -> str:
    """A human-friendly transport error, with a hint when the host won't resolve."""
    reason = getattr(exc, "reason", None) or exc
    if isinstance(reason, socket.gaierror):
        return (
            f"cannot resolve the server host ({reason}). Check the address — if "
            "you used a hostname, try the server's IP — and note an "
            "http_proxy/https_proxy env var can break this even when curl works."
        )
    return str(exc)


def _resolve_cfg(ep: str | None, api_key: str | None) -> dict:
    if ep:
        cfg = {"endpoint": ep}
        if api_key:
            cfg["api_key"] = api_key
        return cfg
    cfg = load()
    if not cfg:
        raise RemoteError("no remote endpoint configured (imggen remote set HOST[:PORT])")
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
        raise RemoteError(f"cannot reach imggen server at {url}: {_explain(exc)}") from exc


def models(ep: str | None = None, api_key: str | None = None, timeout: float = 5.0) -> dict:
    """Fetch the server's installed presets + defaults from ``/models``.

    Returns the same structure as :func:`imggen.registry.models_info`
    (``{"models": {kind: [...]}, "defaults": {kind: name}}``), so the client can
    render a remote model list identically to a local one. Raises
    :class:`RemoteError` on any transport / auth / server failure (including a
    server too old to expose ``/models``) — there is no local fallback.
    """
    cfg = _resolve_cfg(ep, api_key)
    url = _base_url(cfg["endpoint"]) + "/models"
    request = urllib.request.Request(url, headers=_headers(cfg), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise RemoteError(
            f"cannot list models at {url} ({exc.code}): {_http_error_detail(exc)}"
        ) from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RemoteError(f"cannot list models at {url}: {_explain(exc)}") from exc


def _gen_timeout() -> float | None:
    raw = os.environ.get("IMGGEN_REMOTE_TIMEOUT", "3600")
    try:
        val = float(raw)
    except ValueError:
        return 3600.0
    return None if val <= 0 else val


def run(req: GenRequest, on_progress=None):
    """Run ``req`` on the configured remote and return the backend result list.

    Returns the same shape as :func:`imggen.pipelines.run`:
    ``[(PIL.Image, metadata dict, path_hint), ...]``. Raises :class:`RemoteError`
    on any transport / auth / server failure — there is no local fallback. A
    result carrying a layered document (``see-through --method decompose``)
    arrives as bytes and is written to a client temp file, with
    ``meta['_psd_path']`` repointed at it so saving is identical to a local run.

    ``on_progress(event)`` (optional) is called with each streamed progress /
    status event so the caller can render a live bar; the request asks the server
    to stream NDJSON. A server too old to stream returns a single JSON object,
    which is handled transparently (no progress, same result).
    """
    cfg = _resolve_cfg(None, None)

    # Fast reachability check so a misconfigured / down server fails quickly
    # rather than blocking on the (possibly long) generation request.
    ping(timeout=5)

    payload = _encode_request(req)
    payload["stream"] = True
    body = json.dumps(payload).encode()
    url = _base_url(cfg["endpoint"]) + "/generate"
    request = urllib.request.Request(url, data=body, headers=_headers(cfg), method="POST")
    try:
        resp = urllib.request.urlopen(request, timeout=_gen_timeout())
    except urllib.error.HTTPError as exc:
        raise RemoteError(
            f"remote generation failed ({exc.code}): {_http_error_detail(exc)}"
        ) from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RemoteError(f"remote generation failed: {_explain(exc)}") from exc

    ctype = (resp.headers.get("Content-Type") or "").lower()
    try:
        if "ndjson" in ctype:
            data = _read_stream(resp, on_progress)
        else:  # older server: one JSON object, no streaming
            data = json.loads(resp.read().decode())
    except RemoteError:
        raise
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RemoteError(f"remote generation failed: {_explain(exc)}") from exc
    finally:
        resp.close()
    return _decode_results(req, data)


def _read_stream(resp, on_progress) -> dict:
    """Consume an NDJSON ``/generate`` stream, returning the final result object.

    Progress / status events are forwarded to ``on_progress``; an ``error`` event
    becomes a :class:`RemoteError`. Raises if the stream ends with no result.
    """
    final = None
    for raw in resp:
        line = raw.strip()
        if not line:
            continue
        try:
            evt = json.loads(line.decode())
        except ValueError:
            continue
        kind = evt.get("event")
        if kind in ("progress", "status"):
            if on_progress:
                on_progress(evt)
        elif kind == "error":
            raise RemoteError(evt.get("error") or "remote generation failed")
        elif kind == "result" or "results" in evt:
            final = evt
    if final is None:
        raise RemoteError("remote stream ended without a result")
    return final


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
        # ``see-through --method decompose`` ships a whole layered document
        # alongside its preview image. Rehydrate it to a local temp file and
        # point ``_psd_path`` at it, so the runner saves it exactly as it does
        # for a local run (it moves the file to --out, consuming the stash).
        if item.get("psd"):
            meta["_psd_path"] = _stash_psd(item["psd"])
        results.append((image, meta, item.get("hint")))
    return results


def _stash_psd(b64: str) -> str:
    """Write a base64 PSD payload to a temp file and return its path."""
    import tempfile

    fd, path = tempfile.mkstemp(prefix="imggen-st-", suffix=".psd")
    with os.fdopen(fd, "wb") as fh:
        fh.write(base64.b64decode(b64))
    return path


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(exc.read().decode())
        return body.get("error") or body.get("detail") or str(exc)
    except Exception:
        return str(exc)
