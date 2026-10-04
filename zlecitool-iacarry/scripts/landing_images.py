#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Las imágenes del carrusel de la portada (static/img/), todas de 1200×771.

Sin argumentos, la pantalla de la caja (landing-screen.jpg): arranca iaCarry en
una carpeta temporal, con el detector falso y una empresa de prueba sin marca
propia (la caja sale con la de iaCarry: las de los supermercados no van a la
portada pública), analiza la foto de ejemplo 1 en «Probar» y fotografía la
pantalla. Hace falta Chromium (playwright).

Con ``--photo``, una foto de la caja en la tienda (las de
tuisku_iaCarry/presentation/images/), encajada en las mismas medidas con un
fondo desenfocado de sí misma: el carrusel toma la altura de la diapositiva más
alta, y una foto vertical dejaba un hueco blanco debajo de las otras.

    python scripts/landing_images.py
    python scripts/landing_images.py --photo ../presentation/images/station_cesta_2025.jpg station-basket.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
IMG = ROOT / "static" / "img"
OUT = IMG / "landing-screen.jpg"
SIZE = (1200, 771)


def fit(image):
    """La imagen entera, centrada en SIZE, sobre ella misma ampliada y desenfocada."""
    from PIL import Image, ImageFilter, ImageOps

    image = image.convert("RGB")
    back = ImageOps.fit(image, SIZE).filter(ImageFilter.GaussianBlur(28))
    back = Image.blend(back, Image.new("RGB", SIZE, (255, 255, 255)), 0.35)
    front = image.copy()
    front.thumbnail(SIZE, Image.LANCZOS)
    back.paste(front, ((SIZE[0] - front.width) // 2, (SIZE[1] - front.height) // 2))
    return back


def photo(source: Path, name: str) -> int:
    from PIL import Image

    out = IMG / name
    fit(Image.open(source)).save(out, quality=84, optimize=True, progressive=True)
    print(f"{out.relative_to(ROOT)}: {SIZE[0]}×{SIZE[1]}")
    return 0


def main() -> int:
    from PIL import Image
    from playwright.sync_api import sync_playwright
    from werkzeug.serving import make_server

    from zlecitool_core import testing

    with tempfile.TemporaryDirectory() as folder, testing.isolated(folder):
        os.environ["ZLECITOOL_INSECURE_COOKIES"] = "1"
        os.environ.pop("IACARRY_DETECTOR_URL", None)
        from app import create_app
        app = testing.prepare(create_app())
        org = testing.make_org(app, name="Supermercado", credits=50)
        client = testing.signed_in_client(app, org=org, role="owner")
        cookie = client.get_cookie("zlecitool_session").value
        server = make_server("127.0.0.1", 5097, app, threaded=True)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(viewport={"width": 1366, "height": 878}, device_scale_factor=1)
            context.add_cookies([{"name": "zlecitool_session", "value": cookie, "url": "http://127.0.0.1:5097"}])
            page = context.new_page()
            page.goto("http://127.0.0.1:5097/demo?lang=es")
            page.wait_for_selector("#station[data-ready='1']")
            page.select_option("#demo", "demo1")
            page.wait_for_selector("#sema[data-state='stSemaReady']", timeout=20000)
            # Lo de «Probar» que no es de la caja (volver a la trastienda) no sale.
            page.add_style_tag(content=".back{display:none !important}")
            page.wait_for_timeout(400)
            shot = Path(folder) / "shot.png"
            page.screenshot(path=str(shot))
            browser.close()
        server.shutdown()
        image = fit(Image.open(shot))
        image.save(OUT, quality=84, optimize=True, progressive=True)
    print(f"{OUT.relative_to(ROOT)}: {image.size[0]}×{image.size[1]}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Las imágenes del carrusel de la portada.")
    parser.add_argument("--photo", nargs=2, metavar=("FOTO", "NOMBRE"),
                        help="encaja FOTO en static/img/NOMBRE en vez de fotografiar la caja")
    args = parser.parse_args()
    sys.exit(photo(Path(args.photo[0]), args.photo[1]) if args.photo else main())
