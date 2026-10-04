# -*- coding: utf-8 -*-
"""La caja en un navegador de verdad: lo de serving/verify/verify.py que importa
sobre el núcleo (las cajas encima de los productos, las cantidades y el total,
ocupado y en rojo, los idiomas, nada pedido fuera ni nada que falte, el logo del
supermercado y el pago).

Con el detector falso y la app sirviendo en un hilo. Sin Chromium (o sin
playwright), se salta: no la des por probada si se saltó.
"""

import io
import json
import threading
from pathlib import Path

import pytest

from zlecitool_core import testing

from iacarry import detector

from conftest import paired_station

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "iacarry" / "fixtures"
LANGS = ["es", "en", "eu", "ca", "gl", "pt", "fr", "de"]
RED = "rgb(253, 236, 236)"


@pytest.fixture
def live(app, shop, monkeypatch):
    """La app sirviendo en un puerto libre, y una persona de la empresa con sesión."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        testing.unavailable("sin playwright no hay navegador")
    from werkzeug.serving import make_server

    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    person = testing.signed_in_client(app, org=shop, role="owner")
    base = f"http://127.0.0.1:{port}"
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as exc:                # noqa: BLE001 - sin Chromium, se salta
            server.shutdown()
            testing.unavailable(f"sin Chromium: {exc}")

        def page(width=1500, height=1150, lang="es", cookies=None):
            """Una pestaña con la sesión de la persona (o con ``cookies``: las de una caja)."""
            context = browser.new_context(viewport={"width": width, "height": height})
            if cookies is None:
                cookies = {"zlecitool_session": person.get_cookie("zlecitool_session").value}
            context.add_cookies([{"name": name, "value": value, "url": base} for name, value in cookies.items()]
                                + [{"name": "zt_lang", "value": lang, "url": base}])
            tab = context.new_page()
            tab.problems = []
            tab.on("console", lambda m: tab.problems.append(m.text) if m.type == "error" else None)
            tab.on("pageerror", lambda e: tab.problems.append(str(e)))
            return tab

        yield {"base": base, "page": page, "person": person, "app": app}
        browser.close()
    server.shutdown()


def _open(live, **kw):
    tab = live["page"](**kw)
    tab.goto(live["base"] + "/demo", wait_until="networkidle")
    tab.wait_for_selector("#station[data-ready='1']")
    return tab


def _detect(tab, demo="demo1", state="stSemaReady"):
    tab.select_option("#demo", demo)
    tab.wait_for_selector(f"#sema[data-state='{state}']", timeout=20000)
    tab.wait_for_timeout(150)


def _truth(n):
    data = json.loads((FIXTURES / f"ziacarry_eval_img_{n}.json").read_text(encoding="utf-8"))
    return [p for p in data["predictions"] if p["probability"] >= 0.45]


def test_a_boxes_sit_on_the_products(live):
    tab = _open(live)
    _detect(tab)
    assert tab.evaluate("window.iaCarry.state.camAspect") == "640 / 640"
    geo = tab.evaluate("""() => { const c = document.querySelector('#cam').getBoundingClientRect();
        return [...document.querySelectorAll('#cam .box')].map(x => { const r = x.getBoundingClientRect();
          return {l: (r.left - c.left) / c.width, t: (r.top - c.top) / c.height, w: r.width / c.width,
                  h: r.height / c.height}; }); }""")
    truth = [{"l": p["boundingBox"]["left"], "t": p["boundingBox"]["top"],
              "w": p["boundingBox"]["width"] - p["boundingBox"]["left"],
              "h": p["boundingBox"]["height"] - p["boundingBox"]["top"]} for p in _truth(1)]
    assert len(geo) == len(truth) == 12
    worst = max(abs(g[k] - t[k]) for g, t in zip(geo, truth) for k in "ltwh")
    assert worst < 0.01, f"cada caja a menos de un 1 % de la verdad ({worst:.4f})"
    assert not tab.problems, tab.problems


def test_b_quantities_total_and_weight(live):
    tab = _open(live, lang="en")
    _detect(tab)
    qty = tab.evaluate("window.iaCarry.state.live.qty")
    expect = {}
    for p in _truth(1):
        expect[p["tagName"]] = expect.get(p["tagName"], 0) + 1
    assert qty == expect
    assert tab.inner_text("#total").strip() == "€26.31", "los precios del catálogo de la empresa, en céntimos"
    assert tab.inner_text("#badgeCount").strip() == "12 items · 12 ref."
    assert tab.inner_text("#wVal").strip() == "4.19 kg"
    assert tab.inner_text("#mCount").strip() == "12" and tab.inner_text("#mConf").strip() == "100 %"


def test_c_busy_while_the_detector_thinks(live):
    tab = _open(live)
    with detector.fake_detector(delay=1.5):
        tab.select_option("#demo", "demo1")
        tab.wait_for_selector("#sema[data-state='stSemaBusy']", timeout=8000)
        assert tab.is_visible("#cam .busy") and tab.is_disabled("#pay") and tab.is_disabled("#demo")
        tab.wait_for_selector("#sema[data-state='stSemaReady']", timeout=20000)
    assert not tab.is_visible("#cam .busy")


@pytest.mark.parametrize("reply, state", [
    (detector.DetectorError("caído"), "stSemaFail"),
    (detector.DetectorBusy("cola llena"), "stSemaFail"),
    (detector.Detection("v", 640, 640, []), "stSemaNone"),
])
def test_d_failures_are_red_and_never_payable(live, reply, state):
    tab = _open(live, lang="en")
    with detector.fake_detector(default=reply):
        _detect(tab, state=state)
    assert tab.eval_on_selector("#sema", "e => getComputedStyle(e).backgroundColor") == RED
    assert "assistant" in tab.inner_text("#sema").lower()
    assert tab.is_disabled("#pay") and tab.inner_text("#total").strip() == "€0.00"
    tab.click("#pay", force=True)
    assert not tab.is_visible("#success")


def test_d_no_credits_is_red_too(live):
    from zlecitool_core.credits import grant_org
    with live["app"].app_context():
        grant_org(live["person"].org_id, -50, "pruebas: sin saldo")
    tab = _open(live, lang="en")
    _detect(tab, state="stSemaFail")
    assert "errOrgCreditsNeeded" in tab.get_attribute("#sema", "title")


def test_e_every_language_fits(live):
    tab = _open(live, width=1280)
    _detect(tab)
    tab.click("#fraudBtn")
    tab.click("#big")
    clipped = []
    for lang in LANGS:
        tab.select_option("#lang", lang)
        tab.wait_for_timeout(250)
        assert tab.evaluate("document.documentElement.lang") == lang
        bad = tab.evaluate("""() => [...document.querySelectorAll('#pgrid .prod, .btn, .sema, .fact')]
            .filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.textContent.trim().slice(0, 30))""")
        if bad:
            clipped.append((lang, bad[:2]))
    assert not clipped, clipped
    assert tab.inner_text("#payBtnTxt").strip() == "Bezahlen" or tab.inner_text("#payBtnTxt").strip() == "Kontaktlos zahlen"


def test_f_nothing_from_outside_and_nothing_missing(live):
    tab = live["page"](height=1400)
    outbound, missing = [], []
    tab.route("**/*", lambda r, q: r.continue_() if q.url.startswith(live["base"]) else (outbound.append(q.url), r.abort()))
    tab.on("response", lambda r: missing.append(r.url) if r.status >= 400 else None)
    tab.goto(live["base"] + "/demo", wait_until="networkidle")
    tab.wait_for_selector("#station[data-ready='1']")
    _detect(tab)
    tab.evaluate("document.querySelector('.list').scrollTop = 99999")
    tab.wait_for_timeout(800)
    imgs = tab.evaluate("""() => [...document.querySelectorAll('#pgrid img')]
        .map(i => ({src: i.getAttribute('src'), ok: i.complete && i.naturalWidth > 0}))""")
    assert imgs and all(i["ok"] for i in imgs), [i["src"] for i in imgs if not i["ok"]]
    assert all(i["src"].startswith("/static/products/") for i in imgs)
    fonts = tab.evaluate("""async () => { await document.fonts.ready;
        return [...document.fonts].filter(f => f.family.replace(/"/g, '') === 'Ubuntu' && f.status === 'loaded').length; }""")
    assert fonts >= 1, "la Ubuntu del núcleo"
    assert not outbound and not missing, (outbound[:3], missing[:3])
    assert not [p for p in tab.problems if "Content Security Policy" in p], tab.problems


def test_h_each_sample_photo_gets_its_labels(live):
    tab = _open(live)
    for n in (1, 2, 3, 4):
        _detect(tab, demo=f"demo{n}")
        units = tab.evaluate("Object.values(window.iaCarry.state.live.qty).reduce((a, b) => a + b, 0)")
        assert units == len(_truth(n)), n


def test_i_the_supermarkets_logo_is_placed_by_area(live):
    from PIL import Image, ImageDraw
    image = Image.new("RGBA", (500, 120), (255, 255, 255, 255))
    ImageDraw.Draw(image).rounded_rectangle((10, 10, 489, 109), radius=30, fill=(0, 150, 96, 255))
    out = io.BytesIO()
    image.save(out, format="PNG")
    live["person"].post("/branding", data={"name": "Norte", "color": "#009660", "logo": (io.BytesIO(out.getvalue()), "l.png")},
                        content_type="multipart/form-data")
    tab = _open(live)
    tab.wait_for_timeout(400)
    assert tab.is_visible("#clientMark")
    box = tab.evaluate("(() => { const i = document.querySelector('#logoBig'); return [i.clientWidth, i.clientHeight, i.naturalWidth > 0]; })()")
    assert box[2] and (box[0] <= 170 and box[1] <= 40), box
    assert abs(box[0] * box[1] - 4400) < 600 or box[0] == 170, "la misma área para todos los logos (o la caja máxima)"
    pri = tab.evaluate("getComputedStyle(document.body).getPropertyValue('--pri').trim()")
    assert pri == "#009660"
    assert tab.evaluate("document.querySelector('.brand img').naturalWidth") > 0, "la marca de iaCarry, sin cambiar"


def test_paying_shows_the_simulated_confirmation_and_records_it(live):
    tab = _open(live, lang="en")
    _detect(tab)
    tab.click("#pay")
    tab.wait_for_selector("#success:not([hidden])", timeout=8000)
    assert "Simulated payment" in tab.inner_text("#sConfirmed") and tab.is_visible("#sSim")
    assert "12 items" in tab.inner_text("#sTicket")
    tab.wait_for_timeout(500)
    from iacarry.models import Checkout
    with live["app"].app_context():
        row = Checkout.query.one()
        assert row.status == "paid" and row.payment_simulated


def test_a_paired_station_bills_each_frame_in_the_shoppers_language(live):
    """P3: la caja de una tienda (sin persona, emparejada) recibe la foto por su
    costura de la cámara (window.iaCarry.submitFrame), la cobra a la empresa y
    habla el idioma del comprador."""
    from zlecitool_core.credits import balance
    kiosk = paired_station(live["app"], live["person"])
    cookie = kiosk.get_cookie("zt_device_iacarry")
    assert cookie is not None, "la cookie de la caja (núcleo 0.18)"
    tab = live["page"](lang="eu", cookies={"zt_device_iacarry": cookie.value})
    tab.goto(live["base"] + "/station", wait_until="networkidle")
    tab.wait_for_selector("#station[data-ready='1']")
    assert not tab.is_visible("#demo"), "una caja de verdad no enseña las fotos de ejemplo"
    tab.evaluate("""async () => { const r = await fetch('/static/demo/ziacarry_eval_img_1.png');
        await window.iaCarry.submitFrame(await r.blob(), 'ziacarry_eval_img_1.png'); }""")
    tab.wait_for_selector("#sema[data-state='stSemaReady']", timeout=20000)
    labels = json.loads((ROOT / "i18n" / "ui.json").read_text(encoding="utf-8"))
    assert tab.inner_text("#semaTxt").strip() == labels["stSemaReady"]["eu"]
    assert tab.inner_text("#total").strip() == "26,31 €"
    with live["app"].app_context():
        assert balance(org_id=live["person"].org_id) == 49, "una foto, un reconocimiento, del saldo de la empresa"
    assert not tab.problems, tab.problems
