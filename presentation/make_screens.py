"""Regenerate the screen_*.jpg screenshots in presentation/images/.

Starts the stub server (hand-labelled demo photos, no model needed), drives the
checkout page through six scenes and saves each as a JPEG. Edit SCENES to change
the retailer, language, cart state or buttons of a shot.

    python3 presentation/make_screens.py

Needs flask, pillow and playwright (see serving/verify/README.md).
"""
import glob
import io
import os
import subprocess
import sys
import time
import urllib.request

from PIL import Image
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.join(os.path.dirname(HERE), "serving")
OUT = os.path.join(HERE, "images")
PORT = 8127
BASE = "http://127.0.0.1:%d/" % PORT

# file name, retailer, language, demo photo, cart, density, large text,
# AI confidence, anti-fraud panel open, pay
SCENES = [
    ("screen_1_mercadona_full",        "mercadona", "es", "demo1", "full",  "cols3",   False, False, False, False),
    ("screen_2_eroski_confidence",     "eroski",    "en", "demo2", "full",  "comfort", False, True,  False, False),
    ("screen_3_condis_largetext",      "condis",    "fr", "demo3", "full",  "compact", True,  False, True,  False),
    ("screen_4_ahorramas_paid",        "ahorramas", "es", "demo4", "full",  "cols3",   False, False, False, True),
    ("screen_5_mercadona_empty_large", "mercadona", "es", "demo1", "empty", "cols3",   True,  False, False, False),
    ("screen_6_iacarry_empty",         "iacarry",   "en", "demo3", "empty", "cols3",   False, False, False, False),
]
# Catalan and Basque flags are emoji most fonts lack; they render as "?".


def main():
    env = dict(os.environ, PORT=str(PORT), LABELS="on")
    stub = subprocess.Popen([sys.executable, os.path.join(SERVING, "verify", "stub_server.py"), SERVING],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE, timeout=1)
                break
            except Exception:
                time.sleep(0.25)
        exe = os.environ.get("CHROME_PATH") or next(
            iter(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome")), None)
        with sync_playwright() as p:
            b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
            for name, client, lang, demo, cart, dens, big, conf, fraud, pay in SCENES:
                pg = b.new_page(viewport={"width": 1500, "height": 900}, device_scale_factor=2)
                pg.goto(BASE, wait_until="networkidle")
                pg.select_option("#client", client)
                pg.select_option("#lang", lang)
                pg.select_option("#demo", demo)
                pg.wait_for_timeout(300)
                pg.wait_for_function("window.iaCarry.state.detect.status==='ok'", timeout=30000)
                pg.click('#seg button[data-d="%s"]' % dens)
                if big:
                    pg.click("#big")
                if conf:
                    pg.click("#conf")
                if fraud:
                    pg.click("#fraudBtn")
                if cart == "empty":
                    pg.click("#btnEmpty")
                if pay:
                    pg.click("#pay")
                    pg.wait_for_function("window.iaCarry.state.pay.status==='done'", timeout=20000)
                pg.wait_for_timeout(900)
                im = Image.open(io.BytesIO(pg.locator(".app").screenshot())).convert("RGB")
                im.thumbnail((1920, 1920), Image.LANCZOS)
                im.save(os.path.join(OUT, name + ".jpg"), quality=85, optimize=True, progressive=True)
                pg.close()
                print("saved", name + ".jpg")
            b.close()
    finally:
        stub.kill()


if __name__ == "__main__":
    main()
