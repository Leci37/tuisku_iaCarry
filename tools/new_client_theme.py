"""Add a retailer (client) to the checkout screen's dropdown.

Makes the logo file, derives the palette from one brand colour, and adds the
dropdown option and the THEMES entry to the template. See "Adding a client" in
README_4_FRONTEND.md for what each piece is and how to check the result.

    python tools/new_client_theme.py <key> "<Name>" <logo> [--color "#RRGGBB"] [--dry-run]

    python tools/new_client_theme.py mercadona "Mercadona" ~/mercadona.png --color "#009660"

<key>    lowercase letters and digits; becomes the file name <key>-logo.png
<logo>   any image Pillow can open, ideally the horizontal mark on transparent
         or white; it is cropped to its content and fitted into 600x180 on white
--color  the brand's primary colour. Take it from the brand manual when there
         is one. Without it, the most common saturated colour in the logo is used
         and printed, so check it
--dry-run  print what would change, write nothing
"""
import argparse
import colorsys
import os
import re
import sys
from collections import Counter

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "serving", "templates", "iacarry_checkout.html")
LOGOS = os.path.join(ROOT, "serving", "static", "assets", "logos")

# Canvas every client logo shares; the page sizes its slots for this ratio.
CANVAS = (600, 180)
MARK_MAX = (540, 120)

# How much of the brand colour each background tint keeps, mixed toward white.
# These reproduce the strength of the existing themes: "ps"/"pt" are the soft
# fills behind selected items, "bg" the page, "sb" the strongest tint.
TINTS = {"ps": .08, "pt": .07, "bg": .04, "mt": .075, "seg": .09, "sa": .075, "sb": .13}


def fit_logo(src, dst):
    im = Image.open(src).convert("RGBA")
    bbox = im.getbbox()
    if bbox:
        im = im.crop(bbox)
    s = min(MARK_MAX[0] / im.width, MARK_MAX[1] / im.height)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    out = Image.new("RGBA", CANVAS, (255, 255, 255, 255))
    out.alpha_composite(im, ((CANVAS[0] - im.width) // 2, (CANVAS[1] - im.height) // 2))
    out.save(dst, optimize=True)


def guess_color(src):
    """Most common opaque, saturated, not-too-light colour in the logo."""
    im = Image.open(src).convert("RGBA")
    im.thumbnail((400, 400))
    c = Counter()
    px = im.get_flattened_data() if hasattr(im, "get_flattened_data") else im.getdata()
    for r, g, b, a in px:
        if a < 250:
            continue
        h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        if s > .35 and .15 < l < .75:
            c[(r >> 3 << 3, g >> 3 << 3, b >> 3 << 3)] += 1
    if not c:
        sys.exit("no saturated colour found in the logo; pass --color")
    return "#%02X%02X%02X" % c.most_common(1)[0][0]


def tint(hex_color, amount):
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02X%02X%02X" % tuple(round(255 * (1 - amount) + v * amount) for v in rgb)


def contrast_with_white(hex_color):
    def lin(c):
        c /= 255
        return c / 12.92 if c <= .03928 else ((c + .055) / 1.055) ** 2.4
    r, g, b = (lin(int(hex_color[i:i + 2], 16)) for i in (1, 3, 5))
    return 1.05 / (.2126 * r + .7152 * g + .0722 * b + .05)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("key")
    ap.add_argument("name")
    ap.add_argument("logo")
    ap.add_argument("--color")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not re.fullmatch(r"[a-z][a-z0-9]*", a.key):
        sys.exit("key must be lowercase letters and digits, starting with a letter")
    html = open(TEMPLATE, encoding="utf-8").read()
    if re.search(r'<option value="%s">' % a.key, html) or re.search(r"\n  %s:\{name:" % a.key, html):
        sys.exit("'%s' is already in the template" % a.key)

    color = (a.color or guess_color(a.logo)).upper()
    if not re.fullmatch(r"#[0-9A-F]{6}", color):
        sys.exit("colour must look like #RRGGBB")
    if not a.color:
        print("colour taken from the logo: %s  (pass --color to override)" % color)

    # The primary colour is the fill of the pay button and the selected chips,
    # with white text on it. Below 3:1 that text is hard to read.
    ratio = contrast_with_white(color)
    if ratio < 3:
        print("WARNING: white text on %s has contrast %.1f:1 (want 3:1 or more). "
              "Use a darker shade of the brand colour." % (color, ratio))

    logo_url = "/static/assets/logos/%s-logo.png" % a.key
    fields = ",".join('%s:"%s"' % (k, tint(color, v)) for k, v in TINTS.items())
    theme_line = '  %s:{name:"%s",pri:"%s",%s,logo:"%s"}' % (a.key, a.name, color, fields, logo_url)
    option_line = '        <option value="%s">%s</option>' % (a.key, a.name)

    # Append after the last existing entry of each list.
    opts = list(re.finditer(r'\n        <option value="[a-z0-9]+">[^<]*</option>(?=\n      </select>)', html))
    themes_end = re.search(r'(logo:"[^"]*"\})\n\};', html)
    if not opts or not themes_end:
        sys.exit("could not find the client <select> or the THEMES table; add the lines by hand:\n%s\n%s"
                 % (option_line, theme_line))

    print("option: " + option_line.strip())
    print("theme:  " + theme_line.strip())
    print("logo:   serving/static/assets/logos/%s-logo.png" % a.key)
    if a.dry_run:
        return

    end = opts[-1].end()
    html = html[:end] + "\n" + option_line + html[end:]
    themes_end = re.search(r'(logo:"[^"]*"\})\n\};', html)
    html = html[:themes_end.end(1)] + ",\n" + theme_line + html[themes_end.end(1):]
    open(TEMPLATE, "w", encoding="utf-8").write(html)
    fit_logo(a.logo, os.path.join(LOGOS, "%s-logo.png" % a.key))
    print("done. Now check it: python3 serving/verify/verify.py  (sections E and F cover every theme)")


if __name__ == "__main__":
    main()
