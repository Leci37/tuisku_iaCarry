# -*- coding: utf-8 -*-
"""Lo que hace iaCarry, sin Flask: se prueba sin levantar una petición.

Cada función recibe de qué empresa es lo que toca (``org_id``) en vez de leer
la sesión: así no puede mezclar el catálogo, la marca o las compras de dos
supermercados por accidente. Lo que necesita el almacén (guardar un logo) lo
recibe hecho, como una función.
"""

from __future__ import annotations

import colorsys
import csv
import io
import json
import re
from collections import Counter
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from zlecitool_core.db import db, utcnow

from .models import CATEGORIES, Checkout, Frame, OrgSettings, Product, RetailerBrand, Station

MODEL_CLASSES = Path(__file__).resolve().parent / "model_classes.json"
#: El umbral de la caja: lo que no llega no se cuenta ni se cobra al comprador.
#: El mismo que DETECTION_MIN_SCORE en static/js/station.js.
DETECTION_MIN_SCORE = 0.45
#: Una caja que no ha dado señales en este tiempo está «sin conexión».
ONLINE_WINDOW = timedelta(minutes=5)
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


# ── el modelo y el catálogo ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def model_classes() -> dict:
    """Las clases del modelo desplegado: ``{"model_version", "classes": {tag: {...}}}``."""
    data = json.loads(MODEL_CLASSES.read_text(encoding="utf-8"))
    return {"model_version": data["model_version"], "classes": {item["tag"]: item for item in data["classes"]}}


def settings_of(org_id: int) -> OrgSettings:
    row = OrgSettings.query.filter_by(org_id=org_id).first()
    if row is None:
        row = OrgSettings(org_id=org_id, demo=False, keep_frames=True)
        db.session.add(row)
        db.session.commit()
    return row


def ensure_catalogue(org_id: int) -> None:
    """La primera vez, el catálogo de la empresa sale de las clases del modelo
    con sus valores de partida; después es suyo."""
    if Product.query.filter_by(org_id=org_id).first() is not None:
        return
    for item in model_classes()["classes"].values():
        db.session.add(Product(org_id=org_id, tag=item["tag"], name=item["name"], short_name=item.get("short_name"),
                               price_cents=item["price_cents"], weight_g=item["weight_g"],
                               category=item["category"], color=item["color"], active=True))
    db.session.commit()


def catalogue(org_id: int, active_only: bool = False) -> List[Product]:
    ensure_catalogue(org_id)
    query = Product.query.filter_by(org_id=org_id)
    if active_only:
        query = query.filter_by(active=True)
    return query.order_by(Product.category, Product.name).all()


def missing_classes(org_id: int) -> List[dict]:
    """Las clases que conoce el modelo y aún no están en el catálogo de la empresa."""
    have = {row.tag for row in Product.query.filter_by(org_id=org_id)}
    return [item for tag, item in sorted(model_classes()["classes"].items()) if tag not in have]


def for_page(products: List[Product]) -> List[dict]:
    """El catálogo como lo lee la caja (static/js/station.js)."""
    return [{"tag": p.tag, "name": p.name, "short": p.short_name or "", "price": p.price_cents,
             "weight": p.weight_g, "color": p.color, "cat": p.category} for p in products]


def parse_price(raw) -> Optional[int]:
    """«1,25», «1.25» o «2» → céntimos; ``None`` si no es un precio."""
    text = str(raw or "").strip().replace("€", "").replace(" ", "").replace(",", ".")
    if not re.fullmatch(r"\d{1,5}(\.\d{1,2})?", text):
        return None
    euros, _, cents = text.partition(".")
    return int(euros) * 100 + int((cents + "00")[:2])


def _product_problem(name: str, short: str, price, weight, category: str, color: str) -> Optional[str]:
    if not name or len(name) > 120 or len(short) > 40:
        return "errProductName"
    if price is None:
        return "errProductPrice"
    if weight is None or not 0 <= weight <= 50000:
        return "errProductWeight"
    if category not in CATEGORIES:
        return "errProductCategory"
    if not _HEX.match(color or ""):
        return "errProductColor"
    return None


def _weight(raw) -> Optional[int]:
    text = str(raw or "").strip()
    return int(text) if text.isdigit() else None


def update_product(org_id: int, product_id: int, form: dict) -> Optional[str]:
    """Cambia un producto del catálogo de la empresa. Devuelve la clave del
    problema, o ``None``."""
    product = Product.query.filter_by(id=product_id, org_id=org_id).first()
    if product is None:
        return "errNotFound"
    name, short = (form.get("name") or "").strip(), (form.get("short_name") or "").strip()
    price, weight = parse_price(form.get("price")), _weight(form.get("weight_g"))
    category, color = form.get("category") or "", (form.get("color") or "").strip().lower()
    problem = _product_problem(name, short, price, weight, category, color)
    if problem:
        return problem
    product.name, product.short_name, product.price_cents, product.weight_g = name, short or None, price, weight
    product.category, product.color, product.active = category, color, bool(form.get("active"))
    db.session.commit()
    return None


def add_product(org_id: int, tag: str) -> Optional[str]:
    item = model_classes()["classes"].get(tag)
    if item is None:
        return "errProductTag"
    if Product.query.filter_by(org_id=org_id, tag=tag).first() is None:
        db.session.add(Product(org_id=org_id, tag=tag, name=item["name"], short_name=item.get("short_name"),
                               price_cents=item["price_cents"], weight_g=item["weight_g"],
                               category=item["category"], color=item["color"], active=True))
        db.session.commit()
    return None


CSV_COLUMNS = ("tag", "name", "short_name", "price", "weight_g", "category", "color", "active")


def export_csv(org_id: int) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(CSV_COLUMNS)
    for p in catalogue(org_id):
        writer.writerow([p.tag, p.name, p.short_name or "", f"{p.price_cents // 100}.{p.price_cents % 100:02d}",
                         p.weight_g, p.category, p.color, "1" if p.active else "0"])
    return out.getvalue()


def import_csv(org_id: int, text: str) -> Tuple[int, Optional[str], dict]:
    """Actualiza el catálogo desde un CSV (el de ``export_csv``). Todo o nada:
    devuelve ``(filas, problema, variables)``."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if not reader.fieldnames or not {"tag", "name", "price"} <= set(reader.fieldnames):
        return 0, "errCatalogueCsv", {"n": 1}
    classes, rows = model_classes()["classes"], []
    for number, row in enumerate(reader, start=2):
        tag = (row.get("tag") or "").strip()
        if tag not in classes:
            return 0, "errCatalogueCsv", {"n": number}
        name, short = (row.get("name") or "").strip(), (row.get("short_name") or "").strip()
        price = parse_price(row.get("price"))
        weight = _weight(row.get("weight_g")) if (row.get("weight_g") or "").strip() else classes[tag]["weight_g"]
        category = (row.get("category") or classes[tag]["category"]).strip()
        color = (row.get("color") or classes[tag]["color"]).strip().lower()
        if _product_problem(name, short, price, weight, category, color):
            return 0, "errCatalogueCsv", {"n": number}
        active = (row.get("active") or "1").strip().lower() not in ("0", "no", "false", "")
        rows.append((tag, name, short or None, price, weight, category, color, active))
    ensure_catalogue(org_id)
    for tag, name, short, price, weight, category, color, active in rows:
        product = Product.query.filter_by(org_id=org_id, tag=tag).first()
        if product is None:
            product = Product(org_id=org_id, tag=tag)
            db.session.add(product)
        product.name, product.short_name, product.price_cents, product.weight_g = name, short, price, weight
        product.category, product.color, product.active = category, color, active
    db.session.commit()
    return len(rows), None, {}


# ── la marca del supermercado ───────────────────────────────────────────────

#: Cuánto del color de la marca guarda cada tinte de fondo, mezclado hacia el
#: blanco (los de tools/new_client_theme.py, que reproducen los de siempre).
TINTS = {"ps": .08, "pt": .07, "bg": .04, "mt": .075, "seg": .09, "sa": .075, "sb": .13}
#: El texto blanco sobre el color de la marca (el botón de pagar): por debajo,
#: no se lee.
BRAND_MIN_CONTRAST = 3.0
#: El logo, recortado a lo suyo y como mucho de este tamaño.
LOGO_MAX = (900, 160)
WHITE_TOLERANCE = 24


def tint(color: str, amount: float) -> str:
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(255 * (1 - amount) + v * amount) for v in rgb)


def luminance(color: str) -> float:
    def lin(c):
        c /= 255
        return c / 12.92 if c <= .03928 else ((c + .055) / 1.055) ** 2.4
    r, g, b = (lin(int(color[i:i + 2], 16)) for i in (1, 3, 5))
    return .2126 * r + .7152 * g + .0722 * b


def contrast(a: str, b: str = "#ffffff") -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + .05) / (lb + .05)


def ink(color: str) -> str:
    """El color de la marca, oscurecido hasta que el texto pequeño se lea sobre
    su tinte más fuerte (4,5:1): Mercadona rellena con su verde y escribe con éste."""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    for step in range(101):
        factor = 1 - step / 100
        shade = "#%02x%02x%02x" % tuple(round(v * factor) for v in rgb)
        if contrast(shade, tint(color, TINTS["sb"])) >= 4.5:
            return shade
    return "#000000"


def derive_palette(color: str) -> dict:
    """Toda la paleta de la caja desde un color: ``pri``, ``ink`` y los tintes."""
    color = color.lower()
    return {"pri": color, "ink": ink(color), **{key: tint(color, amount) for key, amount in TINTS.items()}}


def brand_color_problem(color: str) -> Tuple[Optional[str], dict]:
    if not _HEX.match(color or ""):
        return "errBrandColor", {}
    ratio = contrast(color.lower())
    if ratio < BRAND_MIN_CONTRAST:
        return "errBrandContrast", {"n": f"{ratio:.1f}"}
    return None, {}


def _clear_background(image):
    """El blanco de alrededor, transparente; el de dentro (las letras de EROSKI),
    no. El borde suavizado se des-mezcla del blanco, o deja un halo pálido."""
    from PIL import Image, ImageDraw, ImageFilter

    w, h = image.size
    px = image.load()
    mask = Image.new("L", image.size, 0)
    mp = mask.load()
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a < 16 or min(r, g, b) >= 255 - WHITE_TOLERANCE:
                mp[x, y] = 255
    edge = [(x, 0) for x in range(w)] + [(x, h - 1) for x in range(w)] \
        + [(0, y) for y in range(h)] + [(w - 1, y) for y in range(h)]
    for xy in edge:
        if mp[xy] == 255:
            ImageDraw.floodfill(mask, xy, 128)
    rim = mask.point(lambda v: 255 if v == 128 else 0).filter(ImageFilter.MaxFilter(5)).load()
    out = image.copy()
    op = out.load()
    for y in range(h):
        for x in range(w):
            if mp[x, y] == 128:
                op[x, y] = (255, 255, 255, 0)
            elif rim[x, y]:
                r, g, b, a = px[x, y]
                alpha = max(255 - r, 255 - g, 255 - b) / 255
                if alpha <= 0:
                    op[x, y] = (255, 255, 255, 0)
                    continue
                un = tuple(max(0, min(255, round((c - 255 * (1 - alpha)) / alpha))) for c in (r, g, b))
                op[x, y] = un + (round(a * alpha),)
    return out


def clean_logo(data: bytes) -> Tuple[bytes, int, int]:
    """Un logo subido, listo para la caja: sin el blanco de alrededor, recortado
    a lo suyo, como mucho 900×160 y en PNG. Lanza ``ValueError("errBrandLogo")``."""
    from PIL import Image, ImageOps

    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGBA")
    except Exception as exc:                    # noqa: BLE001 - cualquier fallo es «no es un logo»
        raise ValueError("errBrandLogo") from exc
    # Grande, primero se reduce: limpiar píxel a píxel una foto de 4000 px no tiene sentido.
    image.thumbnail((1800, 600))
    image = _clear_background(image)
    box = image.split()[3].point(lambda v: 255 if v > 8 else 0).getbbox()
    if not box:
        raise ValueError("errBrandLogo")
    image = image.crop(box)
    scale = min(1, LOGO_MAX[0] / image.width, LOGO_MAX[1] / image.height)
    if scale < 1:
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                             Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue(), image.width, image.height


def guess_color(data: bytes) -> Optional[str]:
    """El color saturado más repetido de un logo: la primera propuesta del selector."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(data)).convert("RGBA")
    except Exception:                           # noqa: BLE001
        return None
    image.thumbnail((400, 400))
    counts: Counter = Counter()
    for r, g, b, a in image.getdata():
        if a < 250:
            continue
        _h, light, sat = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        if sat > .35 and .15 < light < .75:
            counts[(r >> 3 << 3, g >> 3 << 3, b >> 3 << 3)] += 1
    return "#%02x%02x%02x" % counts.most_common(1)[0][0] if counts else None


def brand_of(org_id: int) -> Optional[RetailerBrand]:
    return RetailerBrand.query.filter_by(org_id=org_id).first()


def save_brand(org_id: int, user_id: int, name: str, color: str, logo: Optional[bytes] = None,
               store_logo: Optional[Callable[[bytes], str]] = None) -> Tuple[Optional[str], dict]:
    """La marca de la empresa en sus cajas. Con ``logo``, se limpia y se guarda
    con ``store_logo(png) → clave``. Devuelve ``(problema, variables)``."""
    name = (name or "").strip()
    if not name or len(name) > 80:
        return "errBrandName", {}
    color = (color or "").strip().lower()
    problem, variables = brand_color_problem(color)
    if problem:
        return problem, variables
    png = None
    if logo:
        try:
            png, width, height = clean_logo(logo)
        except ValueError as exc:
            return str(exc), {}
    palette = derive_palette(color)
    row = brand_of(org_id) or RetailerBrand(org_id=org_id)
    row.display_name, row.pri, row.ink = name, palette["pri"], palette["ink"]
    row.palette = json.dumps({key: palette[key] for key in TINTS})
    row.updated_by = user_id
    if png is not None and store_logo is not None:
        row.logo_key, row.logo_w, row.logo_h = store_logo(png), width, height
    db.session.add(row)
    db.session.commit()
    return None, {}


def theme_of(brand: Optional[RetailerBrand], fallback_name: str, logo_url: str = "") -> dict:
    """Lo que la caja necesita de la marca (applyTheme() en station.js). Sin
    marca, la de iaCarry, con su degradado."""
    if brand is None:
        return dict(IACARRY_THEME)
    return {"key": "client", "name": brand.display_name or fallback_name, "pri": brand.pri, "ink": brand.ink,
            **brand.tints, "logo": logo_url if brand.logo_key else ""}


#: La de iaCarry: su violeta y el degradado de su logo. Nunca la de un cliente.
IACARRY_THEME = {"key": "iacarry", "name": "iaCarry", "pri": "#6c5dc7", "ink": "#6a5bc3",
                 "grad": "linear-gradient(115deg,#304cb5 0%,#6c5dc7 55%,#a466d6 100%)", "ps": "#f3f2fb",
                 "pt": "#f5f4fb", "bg": "#f9f9fd", "mt": "#f4f3fb", "seg": "#f2f0fa", "sa": "#f4f3fb",
                 "sb": "#eceaf8", "logo": ""}

#: Los supermercados de la demostración de tuisku (sólo la empresa marcada
#: ``demo``), con sus logos usados con su permiso (static/logos/README.md).
DEMO_BRANDS = (("eroski", "Eroski", "#e30613"), ("ahorramas", "AhorraMas", "#c53842"),
               ("condis", "Condis", "#17398a"), ("mercadona", "Mercadona", "#009660"))


def demo_themes(static_url: Callable[[str], str]) -> List[dict]:
    themes = [dict(IACARRY_THEME)]
    for key, name, color in DEMO_BRANDS:
        themes.append({"key": key, "name": name, **derive_palette(color),
                       "logo": static_url(f"logos/{key}-logo.png")})
    return themes


# ── las cajas ───────────────────────────────────────────────────────────────

def station_for(org_id: int, device_id: int) -> Station:
    """La caja de una pantalla emparejada; la primera vez, se crea."""
    row = Station.query.filter_by(device_id=device_id).first()
    if row is None:
        row = Station(org_id=org_id, device_id=device_id)
        db.session.add(row)
        db.session.commit()
    return row


def stations(org_id: int, devices) -> List[dict]:
    """Las pantallas de la empresa (``Device`` del núcleo, de esta herramienta) con
    lo que iaCarry sabe de cada una."""
    known = {row.device_id: row for row in Station.query.filter_by(org_id=org_id)}
    # Cada pantalla tiene su caja desde que se da de alta: así se le puede decir
    # dónde está antes de emparejarla.
    missing = [d for d in devices if d.id not in known and d.revoked_at is None]
    for device in missing:
        known[device.id] = Station(org_id=org_id, device_id=device.id)
        db.session.add(known[device.id])
    if missing:
        db.session.commit()
    now = utcnow()
    rows = []
    for device in devices:
        station = known.get(device.id)
        if device.revoked_at is not None:
            status = "revoked"
        elif not device.active:
            status = "waiting"
        elif device.last_seen_at is not None and device.last_seen_at > now - ONLINE_WINDOW:
            status = "online"
        else:
            status = "offline"
        rows.append({"device": device, "station": station, "status": status})
    return rows


def set_location(org_id: int, station_id: int, location: str) -> Optional[str]:
    station = Station.query.filter_by(id=station_id, org_id=org_id).first()
    if station is None:
        return "errNotFound"
    location = (location or "").strip()
    if len(location) > 120:
        return "errStationLocation"
    station.location = location or None
    db.session.commit()
    return None


# ── las fotos y las compras ─────────────────────────────────────────────────

def summarize(org_id: int, detection) -> Dict[str, object]:
    """Lo que la caja enseña de una foto, contado en el servidor con el mismo
    umbral y el catálogo de la empresa: las unidades de cada producto, el total
    en céntimos y el peso esperado en gramos."""
    prices = {p.tag: p for p in catalogue(org_id, active_only=True)}
    items: Dict[str, int] = {}
    for prediction in detection.predictions:
        if prediction.probability >= DETECTION_MIN_SCORE and prediction.tag in prices:
            items[prediction.tag] = items.get(prediction.tag, 0) + 1
    return {"items": items, "units": sum(items.values()), "lines": len(items),
            "total_cents": sum(prices[tag].price_cents * qty for tag, qty in items.items()),
            "weight_g": sum(prices[tag].weight_g * qty for tag, qty in items.items())}


def record(org_id: int, user_id: int, detection, *, station_id: Optional[int] = None, demo: bool = False,
           request_id: str = "", checkout_id: Optional[int] = None, storage_key: Optional[str] = None,
           round_trip_ms: Optional[int] = None) -> Tuple[Checkout, Frame]:
    """Apunta una foto analizada y lo que dice del carro. Las fotos de un mismo
    carro (``checkout_id``, mientras no se haya pagado) actualizan su compra."""
    summary = summarize(org_id, detection)
    now = utcnow()
    checkout = None
    if checkout_id:
        checkout = Checkout.query.filter_by(id=checkout_id, org_id=org_id, station_id=station_id,
                                            status="detected").first()
    if checkout is None:
        checkout = Checkout(org_id=org_id, station_id=station_id, user_id=user_id, demo=demo, started_at=now,
                            status="detected")
        db.session.add(checkout)
    checkout.updated_at = now
    checkout.units, checkout.lines = summary["units"], summary["lines"]
    checkout.total_cents, checkout.expected_weight_g = summary["total_cents"], summary["weight_g"]
    checkout.items = json.dumps(summary["items"], sort_keys=True)
    db.session.flush()
    frame = Frame(org_id=org_id, checkout_id=checkout.id, station_id=station_id, request_id=request_id or None,
                  storage_key=storage_key, width=detection.width or None, height=detection.height or None,
                  detections=json.dumps(detection.as_rows()), model_version=detection.model_version or None,
                  inference_ms=detection.inference_ms, round_trip_ms=round_trip_ms, created_at=now)
    db.session.add(frame)
    if station_id is not None:
        station = db.session.get(Station, station_id)
        if station is not None:
            station.last_detection_at = now
    db.session.commit()
    return checkout, frame


def frame_by_request(org_id: int, request_id: str, station_id: Optional[int]) -> Optional[Frame]:
    """La foto ya analizada con ese X-Request-Id en esa caja: un reintento tras un
    corte de red no vuelve a analizarla ni a apuntarla (ni a cobrarla)."""
    if not request_id:
        return None
    return Frame.query.filter_by(org_id=org_id, request_id=request_id, station_id=station_id).first()


def detection_of(frame: Frame):
    """Lo detectado en una foto ya apuntada, como lo dio el detector."""
    from .detector import Detection, Prediction

    rows = json.loads(frame.detections or "[]")
    return Detection(frame.model_version or "", frame.height or 0, frame.width or 0,
                     [Prediction(r["tag"], float(r["p"]), *[float(v) for v in r["box"]]) for r in rows],
                     frame.inference_ms)


def mark_paid(org_id: int, checkout_id: int, station_id: Optional[int], simulated: bool, ref: str
              ) -> Optional[Checkout]:
    """El pago confirmado de un carro (por la pasarela, o simulado mientras no la hay)."""
    checkout = Checkout.query.filter_by(id=checkout_id, org_id=org_id, station_id=station_id).first()
    if checkout is None or checkout.status not in ("detected", "paid"):
        return None
    if checkout.status == "detected":
        checkout.status, checkout.paid_at = "paid", utcnow()
        checkout.payment_simulated, checkout.payment_ref = bool(simulated), (ref or "")[:60] or None
        db.session.commit()
    return checkout


def dashboard(org_id: int, devices, now=None) -> dict:
    """El panel: lo de hoy (las compras de las cajas, no las pruebas) y las cajas."""
    from sqlalchemy import func

    now = now or utcnow()
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today = Checkout.query.filter(Checkout.org_id == org_id, Checkout.demo.is_(False),
                                  Checkout.started_at >= start)
    paid = today.filter(Checkout.status == "paid")
    revenue = db.session.query(func.coalesce(func.sum(Checkout.total_cents), 0)).filter(
        Checkout.org_id == org_id, Checkout.demo.is_(False), Checkout.status == "paid",
        Checkout.started_at >= start).scalar()
    rows = stations(org_id, devices)
    active = [row for row in rows if row["status"] != "revoked"]
    return {"checkouts": today.count(), "paid": paid.count(), "revenue_cents": int(revenue or 0),
            "online": sum(1 for row in active if row["status"] == "online"), "stations": len(active),
            "recent": recent_checkouts(org_id, limit=8)}


def recent_checkouts(org_id: int, limit: int = 50, offset: int = 0) -> List[Checkout]:
    return (Checkout.query.filter_by(org_id=org_id).order_by(Checkout.started_at.desc(), Checkout.id.desc())
            .offset(offset).limit(limit).all())


def last_frames(org_id: int, checkout_ids) -> Dict[int, Frame]:
    """La última foto de cada compra."""
    found: Dict[int, Frame] = {}
    ids = list(checkout_ids) or [0]
    for frame in (Frame.query.filter(Frame.org_id == org_id, Frame.checkout_id.in_(ids))
                  .order_by(Frame.created_at.asc(), Frame.id.asc())):
        found[frame.checkout_id] = frame
    return found
