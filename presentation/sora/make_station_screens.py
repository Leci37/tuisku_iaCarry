"""Make the checkout screens for the station renders in this folder.

For each station (basket, trolley) it composes a schematic top-down "camera"
frame from the catalogue thumbnails, with exactly the products in the Sora
drafts, derives exact box labels from where each product is pasted, and renders
the real checkout page (serving/templates/iacarry_checkout.html) in large-text
mode, English, iaCarry theme. The frame and the /upload answer are swapped in
the browser, so the template and the demo assets are not touched.

    python3 presentation/sora/make_station_screens.py

Writes frame_<station>.png and screen_<station>.png next to this file.
Change BASKET / TROLLEY to change what is in the cart.
"""
import glob
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

def place(bg, items, seed):
    random.seed(seed)
    preds = []
    for tag, cx, cy, h, ang in items:
        p = Image.open(PROD % tag).convert("RGBA")
        p = p.resize((round(p.width * h / p.height), h), Image.LANCZOS)
        p = p.rotate(ang, expand=True, resample=Image.BICUBIC)
        # soft shadow
        sh = Image.new("RGBA", p.size, (0, 0, 0, 0))
        sh.putalpha(p.getchannel("A").point(lambda a: int(a * .45)))
        sh = sh.filter(ImageFilter.GaussianBlur(6))
        x, y = round(cx - p.width / 2), round(cy - p.height / 2)
        bg.paste(sh, (x + 6, y + 8), sh)
        bg.paste(p, (x, y), p)
        l, t, r, b = p.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox()
        l, t, r, b = max(0, x + l), max(0, y + t), min(S, x + r), min(S, y + b)
        preds.append({"probability": round(random.uniform(.93, .99), 2), "tagName": tag,
                      "boundingBox": {"left": round(l / S, 4), "top": round(t / S, 4),
                                      "width": round(r / S, 4), "height": round(b / S, 4)}})
    tags = sorted({p["tagName"] for p in preds})
    for p in preds:
        p["tagInt"] = tags.index(p["tagName"]) + 1
    return bg, preds

BASKET = [  # tag, centre x, centre y, height px, rotation
    ("cocacola_33cl", 150, 160, 190, 8), ("cocacola_33cl", 300, 150, 190, -6),
    ("fanta_33cl", 460, 165, 190, 5), ("mahou5_33cl", 520, 440, 190, -8),
    ("colgate_75ml", 120, 440, 300, -12), ("hysori_300ml", 250, 450, 260, 6),
    ("chipsahoy_300g", 385, 440, 250, -4),
]
TROLLEY = [
    ("chipsahoy_300g", 95, 120, 200, 4), ("tostarica_570g", 245, 115, 190, -3), ("smacks_330g", 400, 115, 200, 5),
    ("colacao_383g", 545, 125, 190, -5),
    ("cocacola_33cl", 70, 330, 145, 6), ("cocacola_33cl", 175, 330, 145, -5), ("fanta_33cl", 280, 330, 145, 4),
    ("estrellag_33cl", 375, 330, 150, -3), ("mahou5_33cl", 470, 330, 145, 5), ("mahou00_33cl", 570, 335, 145, -6),
    ("aguila_33cl", 95, 520, 150, -4), ("hysori_300ml", 245, 515, 200, 5), ("colgate_75ml", 395, 515, 220, -8),
    ("axedrak_150ml", 540, 515, 190, 6),
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


def render(stations):
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
            for name, density, frame, labels in stations:
                pg = b.new_page(viewport={"width": 1500, "height": 900}, device_scale_factor=2)
                pg.route("**/static/assets/demo/ziacarry_eval_img_1.png", _serve_frame(frame))
                pg.route("**/upload", _serve_labels(labels))
                pg.goto(BASE, wait_until="networkidle")
                pg.select_option("#lang", "en")
                pg.click("#big")
                pg.select_option("#demo", "demo1")
                pg.wait_for_timeout(300)
                pg.wait_for_function("window.iaCarry.state.detect.status==='ok'", timeout=30000)
                pg.click('#seg button[data-d="%s"]' % density)
                pg.wait_for_timeout(800)
                # Render only: call the source the live camera, and show the
                # station label in English (it is hardcoded Spanish in the page).
                pg.evaluate("""() => {
                    const o=document.querySelector('#demo option:checked'); if(o) o.textContent='Live camera';
                    document.querySelectorAll('.muted').forEach(e=>{
                        if(e.textContent.includes('Estación')) e.textContent='Station 04 · Barakaldo'; }); }""")
                pg.locator(".app").screenshot(path=os.path.join(HERE, "screen_%s.png" % name))
                print("screen_%s.png  total %s  %s" % (name, pg.inner_text("#total"), pg.inner_text("#badgeCount")))
                pg.close()
            b.close()
    finally:
        stub.kill()


def main():
    stations = []
    for name, bgf, items, seed, density in [("basket", basket_bg, BASKET, 1, "comfort"),
                                             ("trolley", trolley_bg, TROLLEY, 2, "cols3")]:
        im, preds = place(bgf(), items, seed)
        path = os.path.join(HERE, "frame_%s.png" % name)
        im.save(path, optimize=True)
        labels = {"path_server": "presentation/sora", "shape_img": [S, S, 3], "predictions": preds}
        stations.append((name, density, open(path, "rb").read(), labels))
    render(stations)


if __name__ == "__main__":
    main()
