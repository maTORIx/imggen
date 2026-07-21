"""Render generated images inline in graphics-capable terminals.

A best-effort *preview*: after a run we draw the result straight into the
terminal when it speaks a known inline-image protocol — the Kitty graphics
protocol (Ghostty, Kitty, WezTerm, Konsole) or the iTerm2 protocol (iTerm2,
VS Code) — and otherwise stay completely silent. A batch of several images is
composited into a near-square grid first, so ``--num 4`` shows one 2x2 tile.

This is display-only and never fails a run: any error (unsupported terminal,
odd geometry, write error) is swallowed and we simply print nothing. It lives
off the ``cli.py`` import path (imported lazily by the runner, only when a
generation actually happens), so it may use PIL freely — but stays torch-free.
"""

from __future__ import annotations

import base64
import io
import math
import os
import shutil
import sys

from PIL import Image

# Vertical/horizontal cell margin kept free so the image never crowds the edge
# or the "saved ..." lines printed just below it.
_COL_MARGIN = 1
_ROW_MARGIN = 2
# Fallback cell pixel size (width x height) when the terminal does not report
# its pixel geometry via TIOCGWINSZ. A monospace cell is ~1:2 (twice as tall).
_FALLBACK_CELL = (10.0, 20.0)


def preview(images: list[Image.Image]) -> bool:
    """Draw ``images`` inline in the terminal; return whether anything was shown.

    Safe to call unconditionally: returns ``False`` (drawing nothing) when the
    terminal has no known inline-image support, stdout is redirected, or the
    list is empty. Multiple images are gridded into a single tile.
    """
    try:
        if not images:
            return False
        proto = detect()
        if proto is None:
            return False
        canvas = _make_grid(images)
        return _show(canvas, proto)
    except Exception:
        # A preview must never break a successful generation.
        return False


def detect() -> str | None:
    """Return the inline-image protocol for this terminal: ``"kitty"``,
    ``"iterm2"``, or ``None`` when unsupported / not a TTY.

    Env-based detection (no terminal round-trip) — conservative: only known-good
    terminals qualify. ``IMGGEN_PREVIEW=0`` forces it off globally.
    """
    out = sys.stdout
    if not hasattr(out, "isatty") or not out.isatty():
        return None
    if os.environ.get("IMGGEN_PREVIEW") == "0":
        return None

    env = os.environ
    term = env.get("TERM", "")
    prog = env.get("TERM_PROGRAM", "")

    # Kitty graphics protocol.
    if env.get("KITTY_WINDOW_ID") or "kitty" in term:
        return "kitty"
    if term == "xterm-ghostty" or env.get("GHOSTTY_RESOURCES_DIR") or env.get("GHOSTTY_BIN_DIR"):
        return "kitty"
    if prog == "WezTerm" or env.get("WEZTERM_PANE"):
        return "kitty"
    if "konsole" in term or env.get("KONSOLE_VERSION"):
        return "kitty"

    # iTerm2 inline-image protocol.
    if prog == "iTerm.app" or env.get("LC_TERMINAL") == "iTerm2":
        return "iterm2"
    if prog == "vscode":
        return "iterm2"

    return None


# --- grid compositing ----------------------------------------------------

def _make_grid(images: list[Image.Image], gap: int = 8) -> Image.Image:
    """Composite ``images`` into a near-square grid (single image passes through).

    Cells are sized to the largest image and each picture is centred in its
    cell; the canvas is transparent so gaps and any RGBA alpha show through the
    terminal background.
    """
    imgs = [im.convert("RGBA") for im in images]
    n = len(imgs)
    if n == 1:
        return imgs[0]

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_w = max(im.width for im in imgs)
    cell_h = max(im.height for im in imgs)

    width = cols * cell_w + (cols - 1) * gap
    height = rows * cell_h + (rows - 1) * gap
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        x = c * (cell_w + gap) + (cell_w - im.width) // 2
        y = r * (cell_h + gap) + (cell_h - im.height) // 2
        canvas.alpha_composite(im, (x, y))
    return canvas


# --- geometry ------------------------------------------------------------

def _cell_pixels() -> tuple[float, float]:
    """Best estimate of one cell's pixel size (width, height).

    Uses the terminal's reported pixel geometry (TIOCGWINSZ ws_xpixel/ws_ypixel)
    when available — no terminal round-trip needed — and falls back to a 1:2
    monospace assumption otherwise.
    """
    cw, ch = _FALLBACK_CELL
    try:
        import fcntl
        import struct
        import termios

        buf = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
        rows, cols, xpix, ypix = struct.unpack("HHHH", buf)
        if cols and xpix:
            cw = xpix / cols
        if rows and ypix:
            ch = ypix / rows
    except Exception:
        pass
    return cw, ch


def _fit_cells(img_w: int, img_h: int) -> tuple[int, int]:
    """How many (columns, rows) to display an ``img_w`` x ``img_h`` image over,
    fitting the terminal's width and height while preserving aspect ratio and
    never enlarging past the native pixel size."""
    term_cols, term_rows = shutil.get_terminal_size((80, 24))
    cell_w, cell_h = _cell_pixels()

    avail_cols = max(1, term_cols - _COL_MARGIN)
    avail_rows = max(1, term_rows - _ROW_MARGIN)

    native_cols = img_w / cell_w
    native_rows = img_h / cell_h
    scale = min(1.0, avail_cols / native_cols, avail_rows / native_rows)
    return max(1, round(native_cols * scale)), max(1, round(native_rows * scale))


# --- emitting ------------------------------------------------------------

def _png_bytes(image: Image.Image, cols: int, rows: int) -> bytes:
    """PNG-encode ``image``, downscaling to the pixels the cell box will occupy
    so we never ship (or ask the terminal to shrink) more than we display."""
    cell_w, cell_h = _cell_pixels()
    target = (max(1, round(cols * cell_w)), max(1, round(rows * cell_h)))
    if target[0] < image.width or target[1] < image.height:
        image = image.resize(target, Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _in_tmux() -> bool:
    """True when running inside tmux (its multiplexer swallows graphics APCs)."""
    return bool(os.environ.get("TMUX"))


def _tmux_wrap(esc: bytes) -> bytes:
    """Wrap one escape sequence in tmux's passthrough DCS so it reaches the outer
    terminal. tmux forwards the inner bytes verbatim with each ESC un-doubled;
    if ``allow-passthrough`` is off it strips the whole thing (no garbage)."""
    return b"\x1bPtmux;" + esc.replace(b"\x1b", b"\x1b\x1b") + b"\x1b\\"


def _kitty_payload(png: bytes, cols: int, rows: int, tmux: bool = False) -> bytes:
    """Kitty graphics escape(s) for ``png``, chunked at 4096 base64 bytes.

    ``C=1`` keeps the cursor put so the caller controls placement; ``c``/``r``
    scale the image into that many cells; ``q=2`` suppresses the terminal's OK/
    error replies (which would otherwise print as stray text). Under tmux each
    chunk is wrapped in a passthrough envelope so it is not swallowed.
    """
    b64 = base64.standard_b64encode(png)
    ctrl = f"a=T,f=100,C=1,q=2,c={cols},r={rows}"
    chunk = 4096
    parts: list[bytes] = []
    i, total, first = 0, len(b64), True
    while i < total:
        piece = b64[i : i + chunk]
        i += chunk
        more = 1 if i < total else 0
        head = (f"{ctrl},m={more}" if first else f"m={more}").encode()
        esc = b"\x1b_G" + head + b";" + piece + b"\x1b\\"
        parts.append(_tmux_wrap(esc) if tmux else esc)
        first = False
    return b"".join(parts)


def _show(image: Image.Image, proto: str) -> bool:
    """Write ``image`` to the terminal using ``proto``; return ``True`` on write."""
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        return False
    cols, rows = _fit_cells(image.width, image.height)
    png = _png_bytes(image, cols, rows)
    tmux = _in_tmux()

    if proto == "kitty":
        # Reserve `rows` lines (scrolling if near the bottom), jump back to the
        # top of that block, draw without moving the cursor (C=1), then step
        # below the image so the following "saved ..." lines land underneath.
        # Cursor moves and newlines are handled by tmux natively; only the
        # graphics APC needs passthrough wrapping.
        out = (
            b"\n"
            + b"\n" * rows
            + f"\x1b[{rows}A".encode()
            + _kitty_payload(png, cols, rows, tmux)
            + f"\x1b[{rows}B".encode()
        )
    else:  # iterm2
        b64 = base64.standard_b64encode(png).decode()
        seq = (
            f"\x1b]1337;File=inline=1;size={len(png)};width={cols};height={rows};"
            f"preserveAspectRatio=1:{b64}\x07"
        ).encode()
        if tmux:
            seq = _tmux_wrap(seq)
        out = b"\n" + seq + b"\n"

    buffer.write(out)
    buffer.flush()
    return True
