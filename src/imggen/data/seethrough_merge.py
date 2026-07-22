"""Merge a See-through PSD's non-face layers into one flat layer.

Run inside the isolated See-through venv (it needs ``psd_tools`` and the
checkout's ``utils`` package), not in imggen's own interpreter::

    <venv>/bin/python seethrough_merge.py --psd out/input.psd --keep-re '<regex>'

See-through splits a character into ~20 parts. For most editing work only the
face needs to be separable — hair, eyes, brows, mouth — while the body below the
neck is one thing you move as a unit. This rewrites the PSD in place with the
face layers untouched and everything else composited into a single ``body``
layer.

The round-trip goes through See-through's own helpers (``psd2partdicts`` reads
the PSD + its ``_depth.psd`` + ``.json`` sidecar back into part dicts,
``dump_parts_psd`` writes all three out again), so layer placement, depth
ordering and the sidecar stay exactly as the upstream pipeline would produce
them — this only changes *which* parts exist.
"""

import argparse
import re

import numpy as np

from utils.inference_utils import dump_parts_psd, psd2partdicts

#: Name of the single layer the non-face parts collapse into.
MERGED_TAG = "body"


def _alpha_over(dst_rgba, dst_depth, part, frame_h, frame_w):
    """Composite one part dict onto full-canvas RGBA + depth buffers."""
    img = part["img"]
    x1, y1 = 0, 0
    if "xyxy" in part:
        x1, y1 = int(part["xyxy"][0]), int(part["xyxy"][1])
    h, w = img.shape[:2]
    # Guard against a part whose bbox runs past the canvas (rounding upstream).
    h, w = min(h, frame_h - y1), min(w, frame_w - x1)
    if h <= 0 or w <= 0:
        return
    src = img[:h, :w].astype(np.float32)
    dst = dst_rgba[y1:y1 + h, x1:x1 + w].astype(np.float32)

    sa = src[..., 3:4] / 255.0
    da = dst[..., 3:4] / 255.0
    out_a = sa + da * (1 - sa)
    with np.errstate(invalid="ignore", divide="ignore"):
        out_rgb = np.where(
            out_a > 0,
            (src[..., :3] * sa + dst[..., :3] * da * (1 - sa)) / np.maximum(out_a, 1e-6),
            0,
        )
    dst_rgba[y1:y1 + h, x1:x1 + w] = np.concatenate(
        [out_rgb, out_a * 255.0], axis=-1
    ).round().clip(0, 255).astype(np.uint8)

    # Depth follows the visible pixel: wherever this part is opaque enough to
    # show, it owns the depth. Parts arrive far-to-near, so later wins.
    depth = part.get("depth")
    if depth is not None:
        vis = (src[..., 3] > 10)
        region = dst_depth[y1:y1 + h, x1:x1 + w]
        region[vis] = depth[:h, :w][vis]


def merge(psd_path, keep_re, out_path=None):
    data = psd2partdicts(psd_path)
    parts, frame = data["parts"], data["frame_size"]
    frame_h, frame_w = int(frame[0]), int(frame[1])

    keep = re.compile(keep_re, re.IGNORECASE)
    kept = {tag: p for tag, p in parts.items() if keep.search(tag)}
    for tag, part in kept.items():
        part.setdefault("tag", tag)  # save_psd names the layer from this
    victims = [p for tag, p in parts.items() if tag not in kept]
    if not victims:
        print("seethrough_merge: nothing to merge (all layers are face parts)")
        return psd_path

    # Far to near, so the composite matches how the PSD stacks them.
    victims.sort(key=lambda p: p["depth_median"], reverse=True)
    canvas = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    depth = np.zeros((frame_h, frame_w), dtype=np.float32)
    for part in victims:
        _alpha_over(canvas, depth, part, frame_h, frame_w)

    ys, xs = np.nonzero(canvas[..., 3] > 0)
    if len(ys) == 0:
        print("seethrough_merge: merged layers are fully transparent; keeping the original")
        return psd_path
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = canvas[y1:y2, x1:x2]
    crop_depth = depth[y1:y2, x1:x2]

    merged = {
        "tag": MERGED_TAG,
        "layer_name": MERGED_TAG,
        "img": crop,
        "depth": crop_depth,
        "xyxy": [x1, y1, x2, y2],
        # Depth decides where the merged layer lands in the stack (dump sorts
        # far-to-near). Take the *nearest* member's depth, not the mean or the
        # median: the merged layer has to sit where its front-most part sat, or
        # anything that was behind the body -- back hair especially -- pops out
        # in front of it. It also keeps a hand raised over the face on top of
        # the face, which is where it belongs.
        "depth_median": float(min(p["depth_median"] for p in victims)),
    }
    kept[MERGED_TAG] = merged

    out = out_path or psd_path
    dump_parts_psd(kept, [frame_h, frame_w], out)
    print(f"seethrough_merge: {len(parts)} -> {len(kept)} layers "
          f"({len(victims)} merged into '{MERGED_TAG}')")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd", required=True, help="PSD produced by inference_psd.py")
    ap.add_argument("--out", default=None, help="output PSD (default: overwrite --psd)")
    ap.add_argument("--keep-re", required=True,
                    help="regex; layers whose tag matches are kept as separate layers")
    args = ap.parse_args()
    merge(args.psd, args.keep_re, args.out)
