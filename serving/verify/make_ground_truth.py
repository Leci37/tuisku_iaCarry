"""Generate demo frames whose contents ARE the fixture's ground truth.

Each detection in sample_upload_response.json above the UI threshold is
painted as a filled rectangle at its true position. An overlay box that lands on
the paint is therefore verifiably correct, rather than merely looking plausible.

Three shapes are produced so box alignment can be checked independently of
aspect ratio: landscape, square and portrait. The frames the shop actually uses
are 640x640; these exist only to make misalignment visible if it appears.
"""
import json
import os
import re
import sys

from PIL import Image, ImageDraw

MIN_SCORE = 0.45          # must match DETECTION_MIN_SCORE in the template

SHAPES = [
    ("ziacarry_eval_img_1.png", 1280, 720),   # landscape
    ("ziacarry_eval_img_2.png", 800, 800),    # square
    ("ziacarry_eval_img_3.png", 640, 960),    # portrait
    ("ziacarry_eval_img_4.png", 1280, 720),
]


def catalogue_tags(template_path):
    """The product tags the UI knows about, read from CATALOG in the template.

    Read rather than duplicated: a tag added to the catalogue must not silently
    fall out of the ground truth and turn a real regression into a pass.
    """
    src = open(template_path, encoding="utf-8").read()
    block = re.search(r"const CATALOG=\[(.*?)\n\];", src, re.S).group(1)
    return set(re.findall(r'tag:"([A-Za-z0-9_]+)"', block))


def geometry(p):
    """Normalised (left, top, width, height), or None if it cannot be drawn.

    boundingBox names are misleading at the source: 'width' and 'height' hold
    the right and bottom edges, so sizes need subtracting.
    """
    b = p["boundingBox"]
    w = b["width"] - b["left"]
    h = b["height"] - b["top"]
    if w <= 0 or h <= 0:                # degenerate: counted, never drawn
        return None
    return {"tag": p["tagName"], "l": b["left"], "t": b["top"], "w": w, "h": h}


def truth_boxes(fixture, known):
    """What the UI is expected to draw: above threshold, drawable, known class."""
    out = []
    for p in fixture["predictions"]:
        if p["probability"] < MIN_SCORE:
            continue
        g = geometry(p)
        if g and g["tag"] in known:
            out.append(g)
    return out


def painted_boxes(fixture):
    """Everything drawable in the response, including classes the UI skips.

    Painting the unknown class too keeps the image faithful to the response —
    and makes the skip visible: that rectangle should end up with no overlay.
    """
    return [g for p in fixture["predictions"] if p["probability"] >= MIN_SCORE
            for g in [geometry(p)] if g]


def main(out_dir, fixture_path, template_path=None):
    os.makedirs(out_dir, exist_ok=True)
    fixture = json.load(open(fixture_path))
    template_path = template_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(fixture_path))),
        "serving", "templates", "iacarry_checkout.html")
    if not os.path.exists(template_path):
        template_path = os.path.join(os.path.dirname(os.path.abspath(fixture_path)), "templates",
                                     "iacarry_checkout.html")
    known = catalogue_tags(template_path)
    truth = truth_boxes(fixture, known)
    painted = painted_boxes(fixture)
    for name, W, H in SHAPES:
        im = Image.new("RGB", (W, H), (238, 234, 249))
        d = ImageDraw.Draw(im)
        for b in painted:
            d.rectangle([b["l"] * W, b["t"] * H, (b["l"] + b["w"]) * W, (b["t"] + b["h"]) * H],
                        fill=(40, 40, 40), outline=(255, 0, 0), width=2)
        im.save(os.path.join(out_dir, name))
        print("wrote %s  %dx%d" % (name, W, H))
    print("ground truth: %d boxes expected on screen, %d painted (%d skipped as unknown classes)"
          % (len(truth), len(painted), len(painted) - len(truth)))
    return truth


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
