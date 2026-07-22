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
* ``POST /generate`` — a JSON ``{request, init_image?, init_name?,
  mask_image?, mask_name?, stream?}``
  body; runs the backend and returns ``{results: [{image, meta, hint, psd?}, ...]}``
  with each image as a base64 PNG (and, for ``see-through --method decompose``,
  the layered document as a base64 ``psd``). Requires the bearer token when
  configured.
  With ``stream: true`` the response is instead an NDJSON stream of per-step
  ``{"event": "progress", ...}`` lines followed by a final ``{"event":
  "result", "results": [...]}`` (or ``{"event": "error"}``) line, so a remote
  client can render a live progress bar (see :mod:`imggen.remote`).
"""

from __future__ import annotations

import base64
import contextlib
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
                    # Request features this build understands. A client checks
                    # this before sending something an older server would drop
                    # silently (see remote._require_features).
                    "features": ["mask"],
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
            if payload.get("stream"):
                self._generate_stream(payload)
            else:
                self._generate_once(payload)

        def _generate_once(self, payload: dict) -> None:
            try:
                with _GEN_LOCK:
                    results = _run_payload(payload)
            except Exception as exc:  # surface the backend error to the client
                self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
                return
            self._send_json(200, {"results": results})

        def _generate_stream(self, payload: dict) -> None:
            """Run generation and stream NDJSON progress + a final result line.

            Each line is one JSON object: ``{"event": "progress", step, total,
            image, images}`` per denoising step, a ``{"event": "status"}`` when the
            run actually starts (after any model load / lock wait), then either
            ``{"event": "result", "results": [...]}`` or ``{"event": "error"}``.
            HTTP/1.0 + connection-close framing: the client reads lines until EOF.
            """
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()

            def emit(obj: dict) -> None:
                self.wfile.write((json.dumps(obj) + "\n").encode())
                self.wfile.flush()

            def on_step(img: int, imgs: int, step: int, total) -> None:
                emit({"event": "progress", "image": img, "images": imgs,
                      "step": step, "total": total})

            try:
                with _GEN_LOCK:
                    emit({"event": "status", "stage": "running"})
                    results = _run_payload(payload, on_step=on_step)
                emit({"event": "result", "results": results})
            except Exception as exc:
                emit({"event": "error", "error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *args) -> None:  # to stdout, one line
            print(f"{self.address_string()} - {fmt % args}")

    return Handler


def _run_payload(payload: dict, on_step=None) -> list[dict]:
    from . import pipelines

    data = dict(payload.get("request") or {})
    tmp_files: list[str] = []
    try:
        # Client-side image paths (--init, --mask) arrive as base64; rehydrate
        # each to a temp file and point the request at it.
        for field in ("init", "mask"):
            blob = payload.get(f"{field}_image")
            if not blob:
                continue
            suffix = os.path.splitext(payload.get(f"{field}_name") or "")[1] or ".png"
            fd, tmp = tempfile.mkstemp(prefix=f"imggen_{field}_", suffix=suffix)
            with os.fdopen(fd, "wb") as fh:
                fh.write(base64.b64decode(blob))
            tmp_files.append(tmp)
            data[field] = tmp
        req = _build_request(data)
        results = pipelines.run(req, on_step=on_step)
        return [_encode_result(img, meta, hint) for img, meta, hint in results]
    finally:
        for tmp in tmp_files:
            if os.path.exists(tmp):
                os.unlink(tmp)


def _build_request(data: dict) -> GenRequest:
    """Build a GenRequest, ignoring unknown keys (client/server skew tolerant)."""
    fields = {f.name for f in dataclasses.fields(GenRequest)}
    return GenRequest(**{k: v for k, v in data.items() if k in fields})


def _encode_result(image, meta: dict, hint) -> dict:
    # A backend that produced a real multi-layer document (see-through --method
    # decompose) returns a *server-local* path in meta['_psd_path'], which is
    # meaningless to the client. Drop the path and ship the bytes instead; the
    # client rehydrates it into its own temp file (remote._decode_results).
    psd_path = meta.pop("_psd_path", None)
    buf = io.BytesIO()
    image.save(buf, format="PNG")  # lossless; keeps RGBA for see-through layers
    item = {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "meta": meta,
        "hint": hint,
    }
    if psd_path:
        try:
            with open(psd_path, "rb") as fh:
                item["psd"] = base64.b64encode(fh.read()).decode()
        finally:
            # The client owns the copy now; never leave the stash behind, even
            # if the read failed.
            with contextlib.suppress(OSError):
                os.unlink(psd_path)
    return item
