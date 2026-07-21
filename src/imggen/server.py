"""``imggen serve``: an HTTP daemon that runs the generation backends on a GPU
host so other machines can reach them with ``imggen remote set``.

This is the server counterpart to the client transport in :mod:`imggen.remote`.
It uses only the standard library (:mod:`http.server`); one generation runs at a
time behind :data:`_GEN_LOCK` (the GPU is a single resource), while ``/health``
stays responsive so reachability probes never block behind a long render.

Loaded lazily by the ``serve`` CLI command, so torch / diffusers are imported
only when actually serving — never on the light command / completion path.

Endpoints:

* ``GET  /health``   — liveness + device/version info (no auth; used to probe).
* ``POST /generate`` — a JSON ``{request, init_image?, init_name?}`` body; runs
  the backend and returns ``{results: [{image, meta, hint}, ...]}`` with each
  image as a base64 PNG. Requires the bearer token when one is configured.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .params import GenRequest

# Only one generation at a time: the GPU is a single shared resource and the
# warm-model cache holds a single pipeline (see common.cached_pipeline).
_GEN_LOCK = threading.Lock()


def serve(host: str, port: int, api_key: str | None) -> None:
    """Serve generation requests until interrupted (Ctrl-C)."""
    from .device import describe

    httpd = ThreadingHTTPServer((host, port), _make_handler(api_key))
    auth = "api-key required" if api_key else "no auth"
    print(f"imggen serve {__version__} — http://{host}:{port}  ({auth})")
    print(f"device: {describe()}")
    print("one generation at a time; Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


def _make_handler(api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"imggen/{__version__}"

        # --- helpers ---
        def _send_json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self) -> bool:
            if not api_key:
                return True
            return self.headers.get("Authorization", "") == f"Bearer {api_key}"

        # --- routes ---
        def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
            path = self.path.rstrip("/")
            if path in ("/health", ""):
                from .device import describe
                from .manifest import KINDS

                self._send_json(200, {
                    "ok": True,
                    "service": "imggen",
                    "version": __version__,
                    "device": describe(),
                    "kinds": list(KINDS) + ["see-through"],
                })
            elif path == "/models":
                # The presets this host would resolve --model against, so a remote
                # client lists/completes the server's models, not its own. Light
                # (reads manifests only); gated like /generate when a key is set.
                if not self._authed():
                    self._send_json(401, {"error": "missing or invalid api key"})
                    return
                from .registry import models_info

                self._send_json(200, models_info())
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/generate":
                self._send_json(404, {"error": "not found"})
                return
            if not self._authed():
                self._send_json(401, {"error": "missing or invalid api key"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length).decode())
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": f"bad request: {exc}"})
                return
            try:
                with _GEN_LOCK:
                    results = _run_payload(payload)
            except Exception as exc:  # surface the backend error to the client
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._send_json(200, {"results": results})

        def log_message(self, fmt: str, *args) -> None:  # to stdout, one line
            print(f"{self.address_string()} - {fmt % args}")

    return Handler


def _run_payload(payload: dict) -> list[dict]:
    from . import pipelines

    data = dict(payload.get("request") or {})
    tmp_init: str | None = None
    try:
        if payload.get("init_image"):
            suffix = os.path.splitext(payload.get("init_name") or "")[1] or ".png"
            fd, tmp_init = tempfile.mkstemp(prefix="imggen_init_", suffix=suffix)
            with os.fdopen(fd, "wb") as fh:
                fh.write(base64.b64decode(payload["init_image"]))
            data["init"] = tmp_init
        req = _build_request(data)
        results = pipelines.run(req)
        return [_encode_result(img, meta, hint) for img, meta, hint in results]
    finally:
        if tmp_init and os.path.exists(tmp_init):
            os.unlink(tmp_init)


def _build_request(data: dict) -> GenRequest:
    """Build a GenRequest, ignoring unknown keys (client/server skew tolerant)."""
    fields = {f.name for f in dataclasses.fields(GenRequest)}
    return GenRequest(**{k: v for k, v in data.items() if k in fields})


def _encode_result(image, meta: dict, hint) -> dict:
    buf = io.BytesIO()
    image.save(buf, format="PNG")  # lossless; keeps RGBA for see-through layers
    return {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "meta": meta,
        "hint": hint,
    }
