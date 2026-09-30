"""Make the checkout screens for the station renders in this folder.

For each station (basket, trolley) it composes a schematic top-down "camera"
frame from the catalogue thumbnails, with exactly the products in the Sora
drafts, derives exact box labels from where each product is pasted, and renders
the real checkout page (serving/templates/iacarry_checkout.html) in large-text
mode, English, iaCarry theme. The frame and the /upload answer are swapped in
the browser, so the template and the demo assets are not touched.

    python3 presentation/sora/make_station_screens.py

Writes frame_<station>.png, screen_<station>.png (landscape) and
screen_<station>_portrait.png (1080x1920, for a vertical signage screen) next
to this file.
Change BASKET / TROLLEY (rows of products) to change what is in the cart, and
SPREAD to make the pile tidier or messier. Products overlap as they do in a
real cart, but each stays at least MIN_VISIBLE in view, and a pile is only kept
when the page draws every name pill clear of the others.
"""
import glob
import io
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

from PIL import Image, ImageDraw, ImageFilter
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SERVING = os.path.join(ROOT, "serving")
PORT = 8128
BASE = "http://127.0.0.1:%d/" % PORT
from PIL import Image, ImageDraw, ImageFilter

PROD = os.path.join(ROOT, "serving", "static", "assets", "products", "%s_300.png")
S = 640

def basket_bg():
    im = Image.new("RGB", (S, S), (22, 70, 40))
    d = ImageDraw.Draw(im)
    # basket floor: rounded rectangle with slot holes, rim around it
    d.rounded_rectangle([18, 18, S - 18, S - 18], radius=40, fill=(18, 98, 52))
    d.rounded_rectangle([46, 46, S - 46, S - 46], radius=28, fill=(14, 84, 44))
    for y in range(70, S - 70, 34):
        for x in range(70, S - 70, 26):
            d.rounded_rectangle([x, y, x + 14, y + 22], radius=5, fill=(225, 228, 222))
    return im.filter(ImageFilter.GaussianBlur(0.6))

def trolley_bg():
    im = Image.new("RGB", (S, S), (58, 60, 64))
    d = ImageDraw.Draw(im)
    for k in range(0, S + 1, 22):
        d.line([(k, 0), (k, S)], fill=(176, 180, 186), width=3)
        d.line([(0, k), (S, k)], fill=(150, 154, 160), width=2)
    d.rectangle([0, 0, S - 1, S - 1], outline=(200, 204, 210), width=10)
    return im.filter(ImageFilter.GaussianBlur(0.5))

MARGIN = 30
LABEL = 28          # room above the top row for the name pills the page draws
MIN_VISIBLE = .60   # every product must stay at least this much in view
# The label the page draws for each tag (shortOf() in the template), used to
# keep the name pills from colliding.
SHORT = {"cocacola_33cl": "CocaCola", "fanta_33cl": "Fanta", "chipsahoy_300g": "Chips Ahoy!",
         "colacao_383g": "ColaCao", "aguila_33cl": "Águila", "estrellag_33cl": "Estrella",
         "mahou5_33cl": "Mahou", "mahou00_33cl": "Mahou", "smacks_330g": "Smacks",
         "tostarica_570g": "Tostarica", "axedrak_150ml": "Axe", "colgate_75ml": "Colgate",
         "hysori_300ml": "H&S"}


def _label_rect(tag, l, t, r):
    """Where the page's name pill lands, in frame pixels (measured on renders)."""
    w = max(70, 12 * len(SHORT.get(tag, tag)) + 34)
    cx = (l + r) / 2
    return (cx - w / 2, t - 28, cx + w / 2, t + 20)


def _hit(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _try_layout(rows, spread, rng):
    """One random pile: rows that overlap each other, products a little larger
    than their cell, jittered and rotated. Later rows are pasted on top."""
    row_h = (S - 2 * MARGIN - LABEL) / len(rows)
    placed = []
    for ri, row in enumerate(rows):
        cell_w = (S - 2 * MARGIN) / len(row)
        for ci, (tag, ang) in enumerate(row):
            ang += rng.uniform(-spread["rot"], spread["rot"])
            p = Image.open(PROD % tag).convert("RGBA").rotate(ang, expand=True, resample=Image.BICUBIC)
            p = p.crop(p.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox())
            k = min(cell_w * spread["fill"] / p.width, row_h * spread["fill_h"] / p.height)
            p = p.resize((round(p.width * k), round(p.height * k)), Image.LANCZOS)
            cx = MARGIN + cell_w * (ci + .5) + rng.uniform(-spread["jit"], spread["jit"])
            cy = MARGIN + LABEL + row_h * (ri + .5) + rng.uniform(-spread["jit"], spread["jit"])
            x = min(max(round(cx - p.width / 2), 4), S - 4 - p.width)
            y = min(max(round(cy - p.height / 2), LABEL), S - 4 - p.height)
            placed.append((tag, p, x, y))
    # visibility: who owns each pixel once everything is pasted in order
    owner = Image.new("L", (S, S), 0)
    for i, (tag, p, x, y) in enumerate(placed):
        owner.paste(i + 1, (x, y), p.getchannel("A").point(lambda a: 255 if a > 40 else 0))
    hist = owner.histogram()
    for i, (tag, p, x, y) in enumerate(placed):
        area = p.getchannel("A").point(lambda a: 255 if a > 40 else 0).histogram()[255]
        if hist[i + 1] < MIN_VISIBLE * area:
            return None
    labels = [_label_rect(tag, x, y, x + p.width) for tag, p, x, y in placed]
    for i, a in enumerate(labels):
        if any(_hit(a, b) for b in labels[i + 1:]):
            return None
    return placed


def place(bg, rows, seed, spread):
    """Pile products the way they lie in a real cart: overlapping, but each one
    recognisable and each name pill readable. Retries random piles until one
    passes both checks, and derives the exact box of every product."""
    rng = random.Random(seed)
    for _ in range(4000):
        placed = _try_layout(rows, spread, rng)
        if placed:
            break
    else:
        raise SystemExit("no layout keeps every product %d%% visible with readable labels; "
                         "lower the spread" % (MIN_VISIBLE * 100))
    preds = []
    for tag, p, x, y in placed:
        sh = Image.new("RGBA", p.size, (0, 0, 0, 0))   # soft shadow on what lies below
        sh.putalpha(p.getchannel("A").point(lambda a: int(a * .5)))
        sh = sh.filter(ImageFilter.GaussianBlur(7))
        bg.paste(sh, (x + 6, y + 8), sh)
        bg.paste(p, (x, y), p)
        preds.append({"probability": round(rng.uniform(.93, .99), 2), "tagName": tag,
                      "boundingBox": {"left": round(x / S, 4), "top": round(y / S, 4),
                                      "width": round((x + p.width) / S, 4),
                                      "height": round((y + p.height) / S, 4)}})
    tags = sorted({p["tagName"] for p in preds})
    for p in preds:
        p["tagInt"] = tags.index(p["tagName"]) + 1
    return bg, preds

# How messy each cart is: fill > 1 makes neighbours overlap, fill_h > 1 makes a
# row lie over the one before it; jit and rot are random shift (px) and turn (deg).
SPREAD = {"basket": {"fill": 1.02, "fill_h": 1.12, "jit": 10, "rot": 10},
          "trolley": {"fill": 1.18, "fill_h": 1.30, "jit": 14, "rot": 14}}

BASKET = [  # rows of (tag, base rotation in degrees), bottom layer first
    [("cocacola_33cl", 6), ("cocacola_33cl", -5), ("fanta_33cl", 4), ("mahou5_33cl", -6)],
    [("colgate_75ml", -8), ("hysori_300ml", 5), ("chipsahoy_300g", -4)],
]
TROLLEY = [
    [("chipsahoy_300g", 3), ("tostarica_570g", -3), ("smacks_330g", 4), ("colacao_383g", -4)],
    [("cocacola_33cl", 5), ("cocacola_33cl", -4), ("fanta_33cl", 3), ("estrellag_33cl", -3), ("mahou5_33cl", 4)],
    [("mahou00_33cl", -5), ("aguila_33cl", 4), ("hysori_300ml", -3), ("colgate_75ml", 5), ("axedrak_150ml", -4)],
]


def _serve_frame(frame):
    def handler(route):
        route.fulfill(status=200, content_type="image/png", body=frame)
    return handler


def _serve_labels(labels):
    def handler(route):
        route.fulfill(status=200, content_type="application/json", body=json.dumps(labels),
                      headers={"X-Request-Id": route.request.headers.get("x-request-id", "r"),
                               "Access-Control-Expose-Headers": "X-Request-Id"})
    return handler


# The page's name pills, as drawn: any two that intersect make the render fail.
PILL_CLASH_JS = """() => {
  const r=[...document.querySelectorAll('#boxes .box b, .cam .box b')].map(b=>b.getBoundingClientRect());
  const out=[];
  for(let i=0;i<r.length;i++) for(let j=i+1;j<r.length;j++){
    const a=r[i],c=r[j];
    if(a.left<c.right-1&&c.left<a.right-1&&a.top<c.bottom-1&&c.top<a.bottom-1) out.push([i,j]);
  }
  return {pills:r.length, clashes:out};
}"""


# Portrait variant for a vertical signage screen (9:16). Render-only CSS: one
# column, a smaller camera view, and nothing a customer at the station does not
# need (retailer and language pickers, search, filters, the fixed accuracy and
# inference figures, the weight panels).
PORTRAIT_CSS = """
body{padding:0!important;background:#fff!important}
.app{max-width:none!important;border:0!important;border-radius:0!important;box-shadow:none!important;min-height:100vh}
.grid{grid-template-columns:1fr!important;gap:14px!important;padding:14px 16px!important}
.top .right > *:not(.avatar){display:none!important}
.det{gap:10px!important}
.cam{width:68%!important;margin:0 auto!important}
.metrics,.facts,.search,.cats{display:none!important}
.cart{height:auto!important}
.list{max-height:250px!important}
"""


def _load(pg, frame, labels, density):
    pg.route("**/static/assets/demo/ziacarry_eval_img_1.png", _serve_frame(frame))
    pg.route("**/upload", _serve_labels(labels))
    pg.goto(BASE, wait_until="networkidle")
    pg.select_option("#lang", "en")
    pg.click("#big")
    pg.select_option("#demo", "demo1")
    pg.wait_for_timeout(300)
    pg.wait_for_function("window.iaCarry.state.detect.status==='ok'", timeout=30000)
    pg.click('#seg button[data-d="%s"]' % density)
    pg.wait_for_timeout(600)


def _render_only_text(pg):
    # Render only: call the source the live camera, and show the station label
    # in English (it is hardcoded Spanish in the page).
    pg.evaluate("""() => {
        const o=document.querySelector('#demo option:checked'); if(o) o.textContent='Live camera';
        document.querySelectorAll('.muted').forEach(e=>{
            if(e.textContent.includes('Estación')) e.textContent='Station 04 · Barakaldo'; }); }""")


def shoot_portrait(browser, name, frame, labels):
    """The same screen laid out for a vertical 1080x1920 signage display."""
    pg = browser.new_page(viewport={"width": 720, "height": 1280}, device_scale_factor=1.5)
    _load(pg, frame, labels, "cols3")
    pg.add_style_tag(content=PORTRAIT_CSS)
    pg.wait_for_timeout(400)
    clashes = pg.evaluate(PILL_CLASH_JS)["clashes"]
    pay = pg.locator("#pay").bounding_box()
    if pay is None or pay["y"] + pay["height"] > 1280:
        raise SystemExit("%s portrait: the pay button falls below the screen" % name)
    _render_only_text(pg)
    pg.screenshot(path=os.path.join(HERE, "screen_%s_portrait.png" % name))
    print("screen_%s_portrait.png" % name)
    pg.close()
    return len(clashes)


def shoot(browser, name, density, frame, labels):
    """Render one station screen; return the number of colliding name pills."""
    pg = browser.new_page(viewport={"width": 1500, "height": 900}, device_scale_factor=2)
    _load(pg, frame, labels, density)
    check = pg.evaluate(PILL_CLASH_JS)
    if check["pills"] != len(labels["predictions"]):
        raise SystemExit("%s: %d name pills drawn for %d products" % (name, check["pills"], len(labels["predictions"])))
    if not check["clashes"]:
        _render_only_text(pg)
        pg.locator(".app").screenshot(path=os.path.join(HERE, "screen_%s.png" % name))
        print("screen_%s.png  total %s  %s" % (name, pg.inner_text("#total"), pg.inner_text("#badgeCount")))
    pg.close()
    return len(check["clashes"])


def main():
    stub = subprocess.Popen([sys.executable, os.path.join(SERVING, "verify", "stub_server.py"), SERVING],
                            env=dict(os.environ, PORT=str(PORT)),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE, timeout=1)
                break
            except Exception:
                time.sleep(.25)
        exe = os.environ.get("CHROME_PATH") or next(
            iter(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")), None)
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
            for name, bgf, items, density in [("basket", basket_bg, BASKET, "comfort"),
                                              ("trolley", trolley_bg, TROLLEY, "cols3")]:
                # Try piles until the page draws every name pill clear of the others.
                for seed in range(1, 41):
                    im, preds = place(bgf(), items, seed, SPREAD[name])
                    buf = io.BytesIO()
                    im.save(buf, "PNG", optimize=True)
                    labels = {"path_server": "presentation/sora", "shape_img": [S, S, 3], "predictions": preds}
                    if shoot(b, name, density, buf.getvalue(), labels) == 0 \
                            and shoot_portrait(b, name, buf.getvalue(), labels) == 0:
                        im.save(os.path.join(HERE, "frame_%s.png" % name), optimize=True)
                        break
                else:
                    raise SystemExit("%s: no pile without colliding labels; lower SPREAD" % name)
            b.close()
    finally:
        stub.kill()


if __name__ == "__main__":
    main()
