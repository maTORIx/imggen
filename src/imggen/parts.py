"""Layer-based character variants: masks, garment layers, compositing.

The workflow this supports, and why each piece exists:

1. Generate a **base body** once — a plain figure, neutral expression. It is the
   only thing that fixes the geometry, so everything else is expressed relative
   to it.
2. ``see-through --method decompose`` splits it into part layers. Those layers
   are **not** stacked back up (See-through's own loader drops any part living
   only in the bottom/right 10% of the canvas, so a recomposition loses the
   shoes); they are used purely as *regions*. :func:`build_masks` turns them
   into the two masks the rest of the flow needs.
3. Clothes are painted with ``sd --init base.png --mask cloth.png``. Because the
   inpaint path restores everything outside the mask byte-for-byte, the head is
   structurally unable to move.
4. Expressions are made with ``qwen-image-edit --init … --mask face.png``, which
   edits a *crop* so Qwen cannot re-frame the character.
5. :func:`extract_layer` recovers the garment as an RGBA layer by differencing a
   variant against the base, so outfits become stackable rather than baked in.

The masks are emitted in the coordinate system of the **original image**, not
the PSD: See-through letterboxes its output into a ``--resolution`` square, and
undoing that here means every downstream step works in one coordinate system.
"""

from __future__ import annotations

import json
from pathlib import Path

# See-through part tags, grouped by the role they play when building masks.
#
# ``face`` is the *head silhouette* — it reaches under the hair, so it is never
# used as-is for a face mask. ``back hair`` is occlusion-completed: it is filled
# in behind the whole torso, so subtracting it wholesale erases the body. Only
# the part of it that no body layer covers is really visible hair.
HEAD_TAGS = frozenset({
    "face", "front hair", "ears", "headwear", "eyewear", "earwear",
    "eyewhite", "irides", "eyelash", "eyebrow", "nose", "mouth",
})
BEHIND_TAGS = frozenset({"back hair"})
FEATURE_TAGS = frozenset({"eyewhite", "irides", "eyelash", "eyebrow", "nose", "mouth"})
FACE_TAGS = frozenset({"face"})
OVER_TAGS = frozenset({"front hair"})

#: Defaults measured on 832x1216 full-body art decomposed at ``--resolution 1280``.
CLOTH_DILATE = 26     # how far a tight cloth mask may spill past the silhouette
HEAD_GUARD = 6        # keep-out margin around the head and visible hair
FACE_ERODE = 16       # pull the head silhouette in, off the hairline
FEATURE_DILATE = 10   # ... then add the eyes/nose/mouth back, which erosion ate
FACE_FEATHER = 2.0


class PartsError(RuntimeError):
    """A PSD that does not look like a See-through decomposition."""


# --- PSD -> per-role alpha ----------------------------------------------

def _require_psd_tools():
    try:
        from psd_tools import PSDImage
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise PartsError(
            "reading a decompose PSD needs psd-tools; reinstall imggen "
            "(`uv sync` / `uv tool upgrade imggen`)"
        ) from exc
    return PSDImage


def _role_alphas(psd_path):
    """Accumulate per-role alpha maps over the PSD canvas.

    Returns ``(roles, (w, h), layer_names)`` where *roles* maps ``head`` /
    ``behind`` / ``body`` / ``face`` / ``features`` / ``over`` to a float array
    in ``[0, 1]`` over the PSD canvas. Works for both ``--parts face`` PSDs
    (where the neck-down is a single ``body`` layer) and ``--parts all`` ones:
    anything not tagged as head or behind-hair *is* the body.
    """
    import numpy as np

    PSDImage = _require_psd_tools()
    try:
        psd = PSDImage.open(str(psd_path))
    except Exception as exc:  # psd_tools raises assorted parse errors on non-PSDs
        raise PartsError(
            f"{psd_path}: cannot read as a PSD ({type(exc).__name__}: {exc}). "
            "Pass the .psd from `see-through --method decompose`, not the image."
        ) from exc
    w, h = psd.width, psd.height
    roles = {k: np.zeros((h, w), dtype=np.float32) for k in
             ("head", "behind", "body", "face", "features", "over")}
    names = []

    for layer in psd:
        img = layer.composite()
        if img is None:
            continue
        names.append(layer.name)
        a = np.asarray(img.getchannel("A"), dtype=np.float32) / 255.0
        buf = np.zeros((h, w), dtype=np.float32)
        buf[layer.top:layer.top + a.shape[0], layer.left:layer.left + a.shape[1]] = a
        name = layer.name
        if name in HEAD_TAGS:
            roles["head"] = np.maximum(roles["head"], buf)
        elif name in BEHIND_TAGS:
            roles["behind"] = np.maximum(roles["behind"], buf)
        else:
            roles["body"] = np.maximum(roles["body"], buf)
        if name in FACE_TAGS:
            roles["face"] = np.maximum(roles["face"], buf)
        if name in FEATURE_TAGS:
            roles["features"] = np.maximum(roles["features"], buf)
        if name in OVER_TAGS:
            roles["over"] = np.maximum(roles["over"], buf)

    if not names:
        raise PartsError(f"{psd_path}: no readable layers")
    if roles["face"].max() <= 0:
        raise PartsError(
            f"{psd_path}: no 'face' layer — this does not look like a "
            f"see-through decomposition (layers: {', '.join(names)})"
        )
    return roles, (w, h), names


# --- geometry: PSD square letterbox -> original image ---------------------

def letterbox_box(canvas, target):
    """Where a *target*-aspect image sits inside See-through's square canvas."""
    cw, ch = canvas
    tw, th = target
    if tw * ch >= th * cw:  # content is wider than the canvas: full width
        content_w, content_h = cw, round(cw * th / tw)
    else:
        content_w, content_h = round(ch * tw / th), ch
    return ((cw - content_w) // 2, (ch - content_h) // 2,
            (cw - content_w) // 2 + content_w, (ch - content_h) // 2 + content_h)


def _to_target(arr_u8, canvas, target):
    """Crop the letterbox out of a canvas-sized mask and scale it to *target*."""
    from PIL import Image

    im = Image.fromarray(arr_u8)
    if target is None or tuple(target) == tuple(canvas):
        return im
    return im.crop(letterbox_box(canvas, target)).resize(tuple(target), Image.LANCZOS)


def _grow(mask_bool, radius: int):
    """Dilate (positive) / erode (negative) a boolean mask, as uint8 0/255."""
    import cv2
    import numpy as np

    m = (mask_bool.astype(np.uint8)) * 255
    if not radius:
        return m
    r = abs(int(radius))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
    return (cv2.dilate if radius > 0 else cv2.erode)(m, k)


# --- the two masks -------------------------------------------------------

def build_masks(psd_path, size=None, fit: str = "wide", dilate: int = CLOTH_DILATE,
                guard: int = HEAD_GUARD):
    """Build the clothing and face masks from a decompose PSD.

    ``size`` is the original image's ``(w, h)``; the masks come back in that
    coordinate system (omit it to stay in PSD coordinates).

    ``fit`` picks how much room a garment gets:

    * ``wide`` (default) — everything from the top of the neck down, background
      included. Bell skirts, flared coats and capes need to extend well past the
      base silhouette, and on a plain background repainting it costs nothing:
      ``parts extract`` drops the background again when it lifts the layer.
    * ``tight`` — the silhouette dilated by ``dilate`` px. Leaves the background
      untouched, at the price of clamping voluminous outfits to roughly the
      shape of the body underneath.

    Returns ``(cloth_mask, face_mask, info)`` with PIL ``L`` images.
    """
    import numpy as np

    if fit not in ("wide", "tight"):
        raise ValueError("fit must be 'wide' or 'tight'")
    roles, canvas, names = _role_alphas(psd_path)
    w, h = canvas
    body, head, behind = roles["body"], roles["head"], roles["behind"]

    # Hair that is actually on top of the character, as opposed to the part of
    # the occlusion-completed back-hair layer that the body hides.
    visible_hair = (behind > 0.05) & (body <= 0.05)
    keep = (head > 0.05) | visible_hair

    rows = np.where(body.max(axis=1) > 0.05)[0]
    if rows.size == 0:
        raise PartsError(f"{psd_path}: no body layers below the head")
    body_top = int(rows.min())

    if fit == "wide":
        region = np.zeros((h, w), dtype=np.uint8)
        region[body_top:, :] = 255
    else:
        region = _grow(body > 0.05, dilate)
    cloth = np.clip(region.astype(np.int32) - _grow(keep, guard).astype(np.int32), 0, 255)
    cloth = cloth.astype(np.uint8)

    face = _face_mask(roles)

    cloth_img = _to_target(cloth, canvas, size)
    face_img = _to_target(face, canvas, size)
    cloth_img = cloth_img.point(lambda v: 255 if v > 127 else 0)

    # Same helper the masked-edit backends use, so the box reported here is
    # exactly the crop `qwen-image-edit --mask` will take. (Pulls in torch; this
    # command is not on the latency-sensitive path.)
    from .pipelines import common

    info = {
        "psd": str(psd_path),
        "canvas": [w, h],
        "size": list(size) if size else [w, h],
        "layers": names,
        "cloth": {"fit": fit, "dilate": dilate if fit == "tight" else None, "guard": guard},
        "face_box": list(common.mask_bbox(face_img, align=16)),
        "body_top": body_top,
    }
    return cloth_img, face_img, info


def _face_mask(roles):
    """The visible face: what an expression edit is allowed to repaint.

    Not the ``face`` layer. That one is the whole head silhouette and reaches
    under the hair, so blending through it smears a grey halo along the
    hairline. Shrink it well inside the face, add the eyes/nose/mouth back
    (erosion would have eaten them), subtract the hair in front, and finally
    keep only the islands that actually contain a facial feature — erosion
    leaves stray slivers out at the temples otherwise.
    """
    import cv2
    import numpy as np

    face, features, over = roles["face"], roles["features"], roles["over"]
    inner = _grow(face > 0.05, -FACE_ERODE).astype(np.int32)
    feats = _grow(features > 0.05, FEATURE_DILATE).astype(np.int32)
    in_front = (over > 0.05).astype(np.int32) * 255
    mask = np.clip(np.maximum(inner, feats) - in_front, 0, 255).astype(np.uint8)

    n, labels = cv2.connectedComponents((mask > 127).astype(np.uint8), connectivity=8)
    alive = {int(v) for v in np.unique(labels[features > 0.05]) if v != 0}
    if alive:
        mask = np.where(np.isin(labels, list(alive)), 255, 0).astype(np.uint8)

    from PIL import Image, ImageFilter

    return np.asarray(
        Image.fromarray(mask).filter(ImageFilter.GaussianBlur(FACE_FEATHER))
    )


def write_masks(psd_path, out_dir, size=None, prefix=None, **kwargs):
    """:func:`build_masks`, written out as ``<prefix>_cloth/_face.png`` + JSON."""
    cloth, face, info = build_masks(psd_path, size=size, **kwargs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or Path(psd_path).stem
    paths = {
        "cloth": out_dir / f"{stem}_cloth.png",
        "face": out_dir / f"{stem}_face.png",
        "info": out_dir / f"{stem}_masks.json",
    }
    cloth.save(paths["cloth"])
    face.save(paths["face"])
    info["files"] = {k: str(v) for k, v in paths.items()}
    paths["info"].write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return paths, info


# --- garment layer: variant - base -> RGBA -------------------------------

def extract_layer(base_path, variant_path, out_path, recon_path=None,
                  threshold: int = 8, close: int = 7, min_area: int = 400,
                  matte: bool = True, mask_path=None, device=None, token=None):
    """Lift what changed between *base* and *variant* into one RGBA layer.

    Give ``mask_path`` — the same mask the variant was generated with — and it
    is used *as* the alpha. Do that whenever the edit has soft edges, an
    expression patch above all: re-deriving the alpha from a difference
    threshold produces a hard 0/255 cutout, which then overwrites at full
    strength where the original edit was a 40% blend. With the mask it is the
    same blend, so re-stacking reproduces the variant.

    Without it, the alpha is derived from the difference. A variant produced by
    the masked inpaint path is byte-identical to the base outside the mask, so
    "what changed" is exactly the garment — no second decomposition needed, and
    an opaque garment loses nothing by being cut out. Two cleanups make that
    hold up:

    * **Matting.** A ``wide`` cloth mask repaints the background too, which
      shows up as a faint tinted rectangle. Intersecting with the subject's
      alpha discards it, and the base's own background survives the composite
      untouched. ``matte=False`` skips the model when the variant came from a
      ``tight`` mask and the background was never touched.
    * **Hole filling.** Where a garment happens to match the skin underneath the
      difference vanishes; any fully-enclosed gap is filled back in.

    Returns a stats dict worth reading as a check. ``recon_max`` is how far
    re-stacking the layer lands from the variant *inside the garment* — it
    should be near zero. ``base_max`` is how far the result drifts from the base
    everywhere the layer is transparent, which is what confirms the base's own
    background and head came through untouched.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageFilter

    base = Image.open(base_path).convert("RGB")
    variant = Image.open(variant_path).convert("RGB")
    if variant.size != base.size:
        variant = variant.resize(base.size, Image.LANCZOS)

    b = np.asarray(base, dtype=np.int32)
    v = np.asarray(variant, dtype=np.int32)

    if mask_path:
        m = Image.open(mask_path).convert("L")
        if m.size != base.size:
            m = m.resize(base.size, Image.LANCZOS)
        alpha = np.asarray(m)
        keep, holes = [], 0
    else:
        changed = np.abs(v - b).max(axis=2) > threshold

        if matte:
            from .pipelines.seethrough import foreground_alpha

            fg = np.asarray(foreground_alpha(variant, device, token), dtype=np.float32) / 255.0
            changed &= fg > 0.5

        alpha = (changed.astype(np.uint8)) * 255
        if close:
            k = np.ones((close * 2 + 1, close * 2 + 1), np.uint8)
            alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k)

        n, labels, stats, _ = cv2.connectedComponentsWithStats((alpha > 127).astype(np.uint8), 8)
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        alpha = np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)
        alpha, holes = _fill_holes(alpha)
        alpha = np.asarray(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(0.8)))

    layer = variant.convert("RGBA")
    layer.putalpha(Image.fromarray(alpha))
    layer.save(out_path)

    recon = base.convert("RGBA")
    recon.alpha_composite(layer)
    recon = recon.convert("RGB")
    if recon_path:
        recon.save(recon_path)

    # Judge the two halves separately. Comparing the whole canvas against the
    # variant would flag the discarded background as error, when dropping it is
    # the point.
    r = np.asarray(recon, dtype=np.int32)
    inside, outside = alpha > 200, alpha < 10
    to_variant = np.abs(r - v).max(axis=2)
    to_base = np.abs(r - b).max(axis=2)
    stat = lambda arr, sel: (int(arr[sel].max()), float(arr[sel].mean())) if sel.any() else (0, 0.0)  # noqa: E731
    recon_max, recon_mean = stat(to_variant, inside)
    base_max, base_mean = stat(to_base, outside)
    return {
        "coverage": float((alpha > 0).mean()),
        "components": len(keep),
        "holes_filled": holes,
        "recon_max": recon_max,
        "recon_mean": recon_mean,
        "base_max": base_max,
        "base_mean": base_mean,
    }


def _fill_holes(alpha):
    """Fill regions of zero alpha fully enclosed by the layer."""
    import cv2
    import numpy as np

    n, labels, _, _ = cv2.connectedComponentsWithStats((alpha < 128).astype(np.uint8), 4)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    holes = [i for i in range(1, n) if i not in border]
    if holes:
        alpha = np.where(np.isin(labels, holes), 255, alpha)
    return alpha, len(holes)


# --- stacking ------------------------------------------------------------

def compose(base_path, layer_paths, out_path):
    """Stack RGBA layers over a base image, bottom-up, and save the result."""
    from PIL import Image

    out = Image.open(base_path).convert("RGBA")
    for p in layer_paths:
        layer = Image.open(p).convert("RGBA")
        if layer.size != out.size:
            layer = layer.resize(out.size, Image.LANCZOS)
        out.alpha_composite(layer)
    if str(out_path).lower().endswith((".jpg", ".jpeg")):
        out = out.convert("RGB")
    out.save(out_path)
    return out
