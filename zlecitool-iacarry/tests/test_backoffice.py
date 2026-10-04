# -*- coding: utf-8 -*-
"""La trastienda: el catálogo, la marca del supermercado, las cajas y las
compras. Cada empresa ve lo suyo y nada de las demás; lo cambian sus dueños y
administradores."""

import io
import uuid

from PIL import Image

from zlecitool_core import testing

from iacarry import service
from iacarry.models import Product, RetailerBrand

from conftest import new_screen, paired_station, photo

JSON = {"Sec-Fetch-Dest": "empty"}


def _png(draw) -> bytes:
    image = Image.new("RGBA", (400, 200), (255, 255, 255, 255))
    draw(image)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _logo() -> bytes:
    """Un logo sobre blanco: un óvalo rojo con una «letra» blanca dentro."""
    from PIL import ImageDraw

    def draw(image):
        pen = ImageDraw.Draw(image)
        pen.ellipse((100, 60, 299, 139), fill=(227, 6, 19, 255))
        pen.rectangle((180, 80, 219, 119), fill=(255, 255, 255, 255))
    return _png(draw)


# ── el catálogo ──

def test_the_catalogue_starts_from_the_models_classes(app, shop):
    with app.app_context():
        products = service.catalogue(shop.id)
        assert len(products) == 13
        colgate = next(p for p in products if p.tag == "colgate_75ml")
        assert (colgate.price_cents, colgate.weight_g) == (230, 100), "el peso de la Colgate, arreglado en los datos"


def test_prices_in_cents():
    assert [service.parse_price(v) for v in ("1,25", "2", "1.5", "0,05", "abc", "1,255", "")] == \
        [125, 200, 150, 5, None, None, None]


def test_only_admins_change_the_catalogue(app, owner, member):
    member.get("/catalogue")                     # la primera visita siembra el catálogo
    with app.app_context():
        cola = Product.query.filter_by(tag="cocacola_33cl").one()
    form = {"name": "Coca-Cola lata", "short_name": "Coca-Cola", "price": "0,95", "weight_g": "350",
            "category": "drinks", "color": "#e41e2b", "active": "1"}
    assert member.post(f"/catalogue/{cola.id}", data=form, headers=JSON).status_code == 403
    assert owner.post(f"/catalogue/{cola.id}", data=form).status_code == 302
    bad = owner.post(f"/catalogue/{cola.id}", data={**form, "price": "gratis"}, headers=JSON)
    assert bad.get_json() == {"error": "errProductPrice"}
    with app.app_context():
        cola = Product.query.filter_by(tag="cocacola_33cl").one()
        assert (cola.name, cola.price_cents, cola.weight_g) == ("Coca-Cola lata", 95, 350)
    assert "Coca-Cola lata" in member.get("/catalogue").get_data(as_text=True)


def test_the_catalogue_goes_out_and_comes_back_as_csv(app, owner):
    csv = owner.get("/catalogue.csv").get_data(as_text=True)
    assert csv.splitlines()[0] == "tag,name,short_name,price,weight_g,category,color,active"
    changed = csv.replace("Fanta Limón 33cl,Fanta,0.73", "Fanta Naranja 33cl,Fanta,0.85")
    answer = owner.post("/catalogue/import", data={"file": (io.BytesIO(changed.encode()), "catalogo.csv")},
                        content_type="multipart/form-data", follow_redirects=True)
    assert 'data-i18n="okCatalogueImported"' in answer.get_data(as_text=True)
    bad = "tag,name,price\nno_existe,Algo,1\n"
    answer = owner.post("/catalogue/import", data={"file": (io.BytesIO(bad.encode()), "malo.csv")},
                        content_type="multipart/form-data", follow_redirects=True)
    assert 'data-i18n="errCatalogueCsv"' in answer.get_data(as_text=True)
    with app.app_context():
        assert Product.query.filter_by(tag="fanta_33cl").one().price_cents == 85


def test_a_product_out_of_sale_is_neither_counted_nor_charged(app, shop):
    from zlecitool_core.db import db

    from iacarry import detector
    with app.app_context():
        service.ensure_catalogue(shop.id)
        Product.query.filter_by(org_id=shop.id, tag="cocacola_33cl").one().active = False
        db.session.commit()
        summary = service.summarize(shop.id, detector.detect(photo(1), "ziacarry_eval_img_1.png", "r"))
        assert summary["units"] == 11 and "cocacola_33cl" not in summary["items"]
        assert summary["total_cents"] == 2631 - 80


# ── la marca ──

def test_the_palette_comes_from_one_colour():
    palette = service.derive_palette("#009660")
    assert palette["pri"] == "#009660" and palette["ink"] != "#009660"
    assert service.contrast(palette["ink"], palette["sb"]) >= 4.5, "texto pequeño legible sobre el tinte"
    assert service.brand_color_problem("#ffd200") == ("errBrandContrast", {"n": "1.5"})
    assert service.brand_color_problem("verde")[0] == "errBrandColor"


def test_a_white_backed_logo_comes_out_transparent_and_tight():
    png, width, height = service.clean_logo(_logo())
    image = Image.open(io.BytesIO(png))
    assert (width, height) == (200, 80), "recortado a lo suyo"
    assert image.getpixel((0, 0))[3] == 0, "el blanco de alrededor, transparente"
    assert image.getpixel((100, 40))[3] == 255, "el blanco de dentro, no"


def test_an_admin_brands_the_stations(app, shop, owner, member):
    answer = owner.post("/branding", data={"name": "Norte", "color": "#e30613", "logo": (io.BytesIO(_logo()), "logo.png")},
                        content_type="multipart/form-data", follow_redirects=True)
    assert 'data-i18n="okBrandSaved"' in answer.get_data(as_text=True)
    logo = member.get("/brand/logo")
    assert logo.status_code == 200 and Image.open(io.BytesIO(logo.data)).size == (200, 80)
    station = paired_station(app, owner)
    page = station.get("/station").get_data(as_text=True)
    assert '"name": "Norte"' in page and '"pri": "#e30613"' in page and "/brand/logo?v=" in page
    assert station.get("/brand/logo").status_code == 200, "la caja pide su logo"
    refused = owner.post("/branding", data={"name": "Norte", "color": "#ffd200"}, follow_redirects=True)
    assert 'data-i18n="errBrandContrast"' in refused.get_data(as_text=True)
    assert member.post("/branding", data={"name": "X", "color": "#000000"}, headers=JSON).status_code == 403


# ── las cajas y las compras ──

def test_the_stations_page_lists_the_companys_screens(app, owner, member):
    station = paired_station(app, owner, name="Caja de la entrada")
    station.post("/station/detect", headers={**JSON, "X-Request-Id": uuid.uuid4().hex},
                 data={"file": (io.BytesIO(photo(1)), "ziacarry_eval_img_1.png")}, content_type="multipart/form-data")
    page = owner.get("/stations").get_data(as_text=True)
    assert "Caja de la entrada" in page and 'class="ia-status is-online"' in page
    from iacarry.models import Station
    with app.app_context():
        station_id = Station.query.one().id
    assert member.post(f"/stations/{station_id}", data={"location": "Entrada"}, headers=JSON).status_code == 403
    owner.post(f"/stations/{station_id}", data={"location": "Barakaldo, entrada"})
    assert "Barakaldo, entrada" in owner.get("/stations").get_data(as_text=True)
    assert "Barakaldo, entrada" in owner.get("/checkouts").get_data(as_text=True)


def test_each_company_sees_only_its_own(app, shop, owner):
    station = paired_station(app, owner)
    body = station.post("/station/detect", headers={**JSON, "X-Request-Id": "r1"},
                        data={"file": (io.BytesIO(photo(1)), "ziacarry_eval_img_1.png")},
                        content_type="multipart/form-data").get_json()
    owner.post("/branding", data={"name": "Norte", "color": "#e30613"})
    rival = testing.make_org(app, name="Hiper Sur", credits=10)
    other = testing.signed_in_client(app, org=rival, role="owner")
    assert "26,31" not in other.get("/checkouts").get_data(as_text=True)
    assert other.get(f"/frames/{body['frame_id']}").status_code == 404
    assert other.get("/brand/logo").status_code == 404
    assert "Norte" not in other.get("/branding").get_data(as_text=True)
    their_station = paired_station(app, other, name="Caja Sur")
    paid = their_station.post(f"/station/checkout/{body['checkout_id']}/paid", json={}, headers=JSON)
    assert paid.status_code == 404
    with app.app_context():
        assert RetailerBrand.query.filter_by(org_id=rival.id).count() == 0
    assert owner.get(f"/frames/{body['frame_id']}").status_code == 200


def test_the_dashboard_counts_todays_paid_checkouts(app, owner):
    station = paired_station(app, owner)
    body = station.post("/station/detect", headers={**JSON, "X-Request-Id": "r1"},
                        data={"file": (io.BytesIO(photo(1)), "ziacarry_eval_img_1.png")},
                        content_type="multipart/form-data").get_json()
    station.post(f"/station/checkout/{body['checkout_id']}/paid", json={"simulated": True}, headers=JSON)
    page = owner.get("/").get_data(as_text=True)
    assert "26,31 €" in page and 'data-i18n="iaStatusPaidSim"' in page and "4,19 kg" in page


def test_a_new_screen_is_listed_before_it_is_paired(app, owner):
    new_screen(owner, name="Caja nueva")
    page = owner.get("/stations").get_data(as_text=True)
    assert "Caja nueva" in page and 'data-i18n="iaWaiting"' in page and 'name="location"' in page


# ── la portada ──

def test_the_landing_asks_for_a_demo(anonymous):
    page = anonymous.get("/").get_data(as_text=True)
    assert 'data-i18n="landingRequestDemo"' in page and "/register" not in page
    assert "img/landing-screen.jpg" in page and "mercadona" not in page.lower(), "sin marcas de supermercados"
    for image in ("landing-screen.jpg", "station-cart.jpg", "station-basket.jpg"):
        assert anonymous.get(f"/static/img/{image}").status_code == 200
