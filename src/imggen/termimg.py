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

# The Unicode-placeholder scheme: an image is transmitted once, then "painted"
# by printing U+10EEEE cells whose row/column is encoded by combining diacritics
# and whose image id is the cell's foreground colour. Because the cells are
# ordinary text, the terminal (and any tmux in between) tracks and scrolls them
# like text — which is why this places correctly where direct placement, sent
# blind through tmux, lands at the top-left corner. Diacritic list is kitty's
# `rowcolumn-diacritics.txt` (index i encodes value i).
_PLACEHOLDER = "\U0010eeee"
_DIACRITICS = (
    0x0305, 0x030D, 0x030E, 0x0310, 0x0312, 0x033D, 0x033E, 0x033F, 0x0346, 0x034A, 0x034B, 0x034C,
    0x0350, 0x0351, 0x0352, 0x0357, 0x035B, 0x0363, 0x0364, 0x0365, 0x0366, 0x0367, 0x0368, 0x0369,
    0x036A, 0x036B, 0x036C, 0x036D, 0x036E, 0x036F, 0x0483, 0x0484, 0x0485, 0x0486, 0x0487, 0x0592,
    0x0593, 0x0594, 0x0595, 0x0597, 0x0598, 0x0599, 0x059C, 0x059D, 0x059E, 0x059F, 0x05A0, 0x05A1,
    0x05A8, 0x05A9, 0x05AB, 0x05AC, 0x05AF, 0x05C4, 0x0610, 0x0611, 0x0612, 0x0613, 0x0614, 0x0615,
    0x0616, 0x0617, 0x0657, 0x0658, 0x0659, 0x065A, 0x065B, 0x065D, 0x065E, 0x06D6, 0x06D7, 0x06D8,
    0x06D9, 0x06DA, 0x06DB, 0x06DC, 0x06DF, 0x06E0, 0x06E1, 0x06E2, 0x06E4, 0x06E7, 0x06E8, 0x06EB,
    0x06EC, 0x0730, 0x0732, 0x0733, 0x0735, 0x0736, 0x073A, 0x073D, 0x073F, 0x0740, 0x0741, 0x0743,
    0x0745, 0x0747, 0x0749, 0x074A, 0x07EB, 0x07EC, 0x07ED, 0x07EE, 0x07EF, 0x07F0, 0x07F1, 0x07F3,
    0x0816, 0x0817, 0x0818, 0x0819, 0x081B, 0x081C, 0x081D, 0x081E, 0x081F, 0x0820, 0x0821, 0x0822,
    0x0823, 0x0825, 0x0826, 0x0827, 0x0829, 0x082A, 0x082B, 0x082C, 0x082D, 0x0951, 0x0953, 0x0954,
    0x0F82, 0x0F83, 0x0F86, 0x0F87, 0x135D, 0x135E, 0x135F, 0x17DD, 0x193A, 0x1A17, 0x1A75, 0x1A76,
    0x1A77, 0x1A78, 0x1A79, 0x1A7A, 0x1A7B, 0x1A7C, 0x1B6B, 0x1B6D, 0x1B6E, 0x1B6F, 0x1B70, 0x1B71,
    0x1B72, 0x1B73, 0x1CD0, 0x1CD1, 0x1CD2, 0x1CDA, 0x1CDB, 0x1CE0, 0x1DC0, 0x1DC1, 0x1DC3, 0x1DC4,
    0x1DC5, 0x1DC6, 0x1DC7, 0x1DC8, 0x1DC9, 0x1DCB, 0x1DCC, 0x1DD1, 0x1DD2, 0x1DD3, 0x1DD4, 0x1DD5,
    0x1DD6, 0x1DD7, 0x1DD8, 0x1DD9, 0x1DDA, 0x1DDB, 0x1DDC, 0x1DDD, 0x1DDE, 0x1DDF, 0x1DE0, 0x1DE1,
    0x1DE2, 0x1DE3, 0x1DE4, 0x1DE5, 0x1DE6, 0x1DFE, 0x20D0, 0x20D1, 0x20D4, 0x20D5, 0x20D6, 0x20D7,
    0x20DB, 0x20DC, 0x20E1, 0x20E7, 0x20E9, 0x20F0, 0x2CEF, 0x2CF0, 0x2CF1, 0x2DE0, 0x2DE1, 0x2DE2,
    0x2DE3, 0x2DE4, 0x2DE5, 0x2DE6, 0x2DE7, 0x2DE8, 0x2DE9, 0x2DEA, 0x2DEB, 0x2DEC, 0x2DED, 0x2DEE,
    0x2DEF, 0x2DF0, 0x2DF1, 0x2DF2, 0x2DF3, 0x2DF4, 0x2DF5, 0x2DF6, 0x2DF7, 0x2DF8, 0x2DF9, 0x2DFA,
    0x2DFB, 0x2DFC, 0x2DFD, 0x2DFE, 0x2DFF, 0xA66F, 0xA67C, 0xA67D, 0xA6F0, 0xA6F1, 0xA8E0, 0xA8E1,
    0xA8E2, 0xA8E3, 0xA8E4, 0xA8E5, 0xA8E6, 0xA8E7, 0xA8E8, 0xA8E9, 0xA8EA, 0xA8EB, 0xA8EC, 0xA8ED,
    0xA8EE, 0xA8EF, 0xA8F0, 0xA8F1, 0xAAB0, 0xAAB2, 0xAAB3, 0xAAB7, 0xAAB8, 0xAABE, 0xAABF, 0xAAC1,
    0xFE20, 0xFE21, 0xFE22, 0xFE23, 0xFE24, 0xFE25, 0xFE26, 0x10A0F, 0x10A38, 0x1D185, 0x1D186, 0x1D187,
    0x1D188, 0x1D189, 0x1D1AA, 0x1D1AB, 0x1D1AC, 0x1D1AD, 0x1D242, 0x1D243, 0x1D244,
)


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


def _kitty_transmit(png: bytes, img_id: int, cols: int, rows: int, tmux: bool) -> bytes:
    """Transmit ``png`` under ``img_id`` and create a virtual placement of
    ``cols`` x ``rows`` cells (``U=1``) to be painted with placeholders.

    Data is chunked at 4096 base64 bytes; ``q=2`` suppresses OK/error replies.
    Under tmux each APC is wrapped in a passthrough envelope so it is forwarded
    to the outer terminal instead of being swallowed.
    """
    b64 = base64.standard_b64encode(png)
    chunk = 4096
    parts: list[bytes] = []
    i, total, first = 0, len(b64), True
    while i < total:
        piece = b64[i : i + chunk]
        i += chunk
        more = 1 if i < total else 0
        if first:
            head = f"a=t,f=100,i={img_id},q=2,m={more}".encode()
            first = False
        else:
            head = f"m={more}".encode()
        esc = b"\x1b_G" + head + b";" + piece + b"\x1b\\"
        parts.append(_tmux_wrap(esc) if tmux else esc)

    place = f"\x1b_Ga=p,U=1,i={img_id},c={cols},r={rows},q=2\x1b\\".encode()
    parts.append(_tmux_wrap(place) if tmux else place)
    return b"".join(parts)


def _placeholder_grid(img_id: int, cols: int, rows: int) -> bytes:
    """The grid of U+10EEEE cells that paints the virtual placement.

    Ordinary text (no escapes beyond the per-row SGR that carries the image id
    as a 256-colour foreground), so it needs no tmux passthrough and scrolls
    naturally. Each cell carries an explicit row then column diacritic.
    """
    ph = _PLACEHOLDER
    lines = []
    for r in range(rows):
        row_d = chr(_DIACRITICS[r])
        cells = "".join(ph + row_d + chr(_DIACRITICS[c]) for c in range(cols))
        lines.append(f"\x1b[38;5;{img_id}m{cells}\x1b[39m")
    return "\n".join(lines).encode()


def _show(image: Image.Image, proto: str) -> bool:
    """Write ``image`` to the terminal using ``proto``; return ``True`` on write."""
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        return False
    cols, rows = _fit_cells(image.width, image.height)
    # Row/column numbers are encoded by diacritics, so the grid can be at most
    # len(_DIACRITICS) cells in each direction.
    cols = min(cols, len(_DIACRITICS))
    rows = min(rows, len(_DIACRITICS))
    png = _png_bytes(image, cols, rows)
    tmux = _in_tmux()

    if proto == "kitty":
        # Transmit once (passthrough-wrapped under tmux), then paint the image
        # with a grid of Unicode placeholder cells — ordinary text the terminal
        # positions and scrolls itself, so it lands under the cursor rather than
        # at the top-left. A trailing newline drops below for the "saved" lines.
        img_id = (os.getpid() % 255) + 1
        out = (
            b"\n"
            + _kitty_transmit(png, img_id, cols, rows, tmux)
            + _placeholder_grid(img_id, cols, rows)
            + b"\n"
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
