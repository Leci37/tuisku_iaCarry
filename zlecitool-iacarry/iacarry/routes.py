# -*- coding: utf-8 -*-
"""Las pantallas y la API de iaCarry. Finas: el trabajo, en service.py.

Dos superficies, las dos sobre el núcleo:

* **La trastienda** (personas de la empresa, ``zt/base.html`` con la marca de
  iaCarry): el panel, las cajas, el catálogo, la marca del supermercado, las
  compras y «Probar».
* **La caja** (``/station``, una pantalla emparejada de la empresa, ``@device``,
  sobre ``zt/bare.html``): la pantalla del comprador. Cada foto que analiza se
  cobra (``recognition``) del saldo de la empresa.

Toda ruta pide sesión (el núcleo) y la puerta sólo deja pasar a quien es de una
empresa con iaCarry; cada consulta filtra por ``current_org().id``.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid

from flask import (Blueprint, Response, abort, flash, get_flashed_messages, jsonify, redirect, render_template,
                   request, url_for)

from zlecitool_core import current_tool, retention
from zlecitool_core.accounts import current_user
from zlecitool_core.artifacts import current_storage
from zlecitool_core.artifacts.uploads import UploadError, read_upload
from zlecitool_core.credits import charge
from zlecitool_core.db import end_idle_transaction
from zlecitool_core.devices import Device, current_device, device
from zlecitool_core.http import wants_json
from zlecitool_core.i18n import request_language
from zlecitool_core.landing import with_landing
from zlecitool_core.orgs import current_org, org_admin_required, org_role

from . import detector, service
from .models import CATEGORIES, Frame, Station

bp = Blueprint("iacarry", __name__)
_log = logging.getLogger("iacarry")

#: Lo que se puede subir como foto de un carro.
FRAME_KINDS = ("png", "jpg", "webp")
FRAME_MAX_MB = 8


def _org():
    org = current_org()
    if org is None:                # la puerta ya lo para; esto, por si algo se cuela
        abort(403)
    return org


def _admin() -> bool:
    return org_role() in ("owner", "admin")


def _page(template, **values):
    return render_template(template, notices=get_flashed_messages(with_categories=True), is_admin=_admin(),
                           **values)


def _back(endpoint, key, ok=True, **variables):
    flash({"key": key, "vars": variables or None}, "ok" if ok else "bad")
    return redirect(url_for(endpoint))


def _screens(org):
    """Las pantallas emparejadas de la empresa para iaCarry (``Device``, sólo se lee)."""
    return (Device.query.filter_by(org_id=org.id, tool=current_tool().ficha.slug, kind="browser")
            .order_by(Device.revoked_at.isnot(None), Device.name).all())


def _checkout_context(org, rows) -> dict:
    """Lo que necesita la tabla de compras: la última foto de cada una y el
    nombre y la ubicación de cada caja."""
    stations_ = Station.query.filter_by(org_id=org.id).all()
    names = {d.id: d.name for d in Device.query.filter(Device.id.in_([s.device_id for s in stations_] or [0]))}
    return {"rows": rows, "frames": service.last_frames(org.id, [r.id for r in rows]),
            "station_names": {s.id: names.get(s.device_id, "") for s in stations_},
            "locations": {s.id: s.location for s in stations_}}


# ── la trastienda ───────────────────────────────────────────────────────────

@bp.get("/")
@with_landing("landing.html")
def index():
    org = _org()
    data = service.dashboard(org.id, _screens(org))
    return _page("dashboard.html", org=org, data=data, active="dashboard", **_checkout_context(org, data["recent"]))


@bp.get("/stations")
def stations():
    org = _org()
    return _page("stations.html", org=org, rows=service.stations(org.id, _screens(org)), active="stations")


@bp.post("/stations/<int:station_id>")
@org_admin_required
def station_save(station_id):
    problem = service.set_location(_org().id, station_id, request.form.get("location"))
    return _back("iacarry.stations", problem or "okStationSaved", ok=problem is None)


@bp.get("/catalogue")
def catalogue():
    org = _org()
    return _page("catalogue.html", org=org, products=service.catalogue(org.id), missing=service.missing_classes(org.id),
                 categories=CATEGORIES, active="catalogue")


@bp.post("/catalogue/<int:product_id>")
@org_admin_required
def product_save(product_id):
    problem = service.update_product(_org().id, product_id, request.form)
    if wants_json():
        return (jsonify({"error": problem}), 400) if problem else jsonify({"status": "ok"})
    return _back("iacarry.catalogue", problem or "okProductSaved", ok=problem is None)


@bp.post("/catalogue/add")
@org_admin_required
def product_add():
    problem = service.add_product(_org().id, request.form.get("tag") or "")
    return _back("iacarry.catalogue", problem or "okProductSaved", ok=problem is None)


@bp.get("/catalogue.csv")
def catalogue_csv():
    org = _org()
    return Response(service.export_csv(org.id), mimetype="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="catalogo-{org.slug}.csv"'})


@bp.post("/catalogue/import")
@org_admin_required
def catalogue_import():
    try:
        upload = read_upload(request.files.get("file"), kinds=("csv",), max_mb=2)
        text = upload.data.decode("utf-8")
    except UploadError as exc:
        return _back("iacarry.catalogue", exc.key, ok=False, **exc.vars)
    except UnicodeDecodeError:
        return _back("iacarry.catalogue", "errCatalogueCsv", ok=False, n=1)
    rows, problem, variables = service.import_csv(_org().id, text)
    if problem:
        return _back("iacarry.catalogue", problem, ok=False, **variables)
    return _back("iacarry.catalogue", "okCatalogueImported", n=rows)


def _logo_url(brand):
    if brand is None or not brand.logo_key:
        return ""
    return url_for("iacarry.brand_logo", v=int(brand.updated_at.timestamp()))


@bp.get("/branding")
def branding():
    org = _org()
    brand = service.brand_of(org.id)
    color = brand.pri if brand else service.IACARRY_THEME["pri"]
    return _page("branding.html", org=org, brand=brand, color=color, palette=service.derive_palette(color),
                 contrast=service.contrast(color), logo_url=_logo_url(brand), active="branding")


@bp.post("/branding")
@org_admin_required
def branding_save():
    org = _org()
    logo = None
    file = request.files.get("logo")
    if file is not None and file.filename:
        try:
            logo = read_upload(file, kinds=("png", "jpg", "webp"), max_mb=4).data
        except UploadError as exc:
            return _back("iacarry.branding", exc.key, ok=False, **exc.vars)

    def store(png: bytes) -> str:
        key = f"{current_tool().ficha.slug}/brand/o{org.id}/logo-{uuid.uuid4().hex[:10]}.png"
        current_storage().save(key, png, "image/png")
        return key

    color = (request.form.get("color") or "").strip()
    if not color and logo:
        color = service.guess_color(logo) or ""
    problem, variables = service.save_brand(org.id, current_user.id, request.form.get("name") or org.name, color,
                                            logo, store)
    if problem:
        return _back("iacarry.branding", problem, ok=False, **variables)
    _log.info("Marca guardada | org=%s por=%s logo=%s", org.slug, current_user.id, bool(logo))
    return _back("iacarry.branding", "okBrandSaved")


@bp.get("/brand/logo")
@device
def brand_logo():
    """El logo del supermercado: lo pide su caja y su trastienda, nadie más."""
    brand = service.brand_of(_org().id)
    if brand is None or not brand.logo_key:
        abort(404)
    storage = current_storage()
    if not storage.exists(brand.logo_key):
        abort(404)
    return Response(storage.read(brand.logo_key), mimetype="image/png",
                    headers={"Cache-Control": "private, max-age=86400"})


@bp.get("/checkouts")
def checkouts():
    org = _org()
    page = max(1, request.args.get("page", 1, type=int))
    rows = service.recent_checkouts(org.id, limit=51, offset=(page - 1) * 50)
    more, rows = len(rows) > 50, rows[:50]
    return _page("checkouts.html", org=org, page=page, more=more, active="checkouts", **_checkout_context(org, rows))


@bp.get("/frames/<int:frame_id>")
def frame_image(frame_id):
    """La foto de un carro, mientras no la haya borrado el borrado por antigüedad."""
    frame = Frame.query.filter_by(id=frame_id, org_id=_org().id).first()
    if frame is None or not frame.storage_key:
        abort(404)
    storage = current_storage()
    if not storage.exists(frame.storage_key):
        abort(404)
    kind = frame.storage_key.rsplit(".", 1)[-1]
    return Response(storage.read(frame.storage_key), mimetype="image/jpeg" if kind == "jpg" else f"image/{kind}",
                    headers={"Cache-Control": "private, no-store"})


# ── la caja ─────────────────────────────────────────────────────────────────

def _station_config(org, mode):
    """Lo que la caja necesita, en un atributo data- (nada de JavaScript en línea)."""
    settings = service.settings_of(org.id)
    brand = service.brand_of(org.id)
    themes = [service.theme_of(brand, org.name, _logo_url(brand))]
    demo = mode == "demo" or settings.demo
    if settings.demo:
        themes = service.demo_themes(lambda path: url_for("static", filename=path))
        if brand is not None:
            themes.insert(1, service.theme_of(brand, org.name, _logo_url(brand)))
    machine = current_device()
    station = service.station_for(org.id, machine.id) if machine is not None else None
    label = machine.name if machine is not None else org.name
    if station is not None and station.location:
        label = f"{label} · {station.location}"
    photos = [{"key": f"demo{n}", "n": n, "file": f"ziacarry_eval_img_{n}.png",
               "src": url_for("static", filename=f"demo/ziacarry_eval_img_{n}.png")} for n in range(1, 5)] if demo else []
    detect = "iacarry.demo_detect" if mode == "demo" else "iacarry.station_detect"
    paid = "iacarry.demo_paid" if mode == "demo" else "iacarry.station_paid"
    return {"mode": mode, "detectUrl": url_for(detect), "paidUrl": url_for(paid, checkout_id=0).replace("/0/", "/__ID__/"),
            "catalogue": service.for_page(service.catalogue(org.id, active_only=True)), "themes": themes,
            "photos": photos, "upload": mode == "demo", "label": label,
            "initials": "".join(word[0] for word in label.split()[:2]).upper()[:2] or "IA",
            "minScore": service.DETECTION_MIN_SCORE, "productImg": url_for("static", filename="products/"),
            "backUrl": url_for("iacarry.index") if mode == "demo" else ""}


@bp.get("/station")
@device(home=True)
def station():
    """La pantalla de la caja. Una persona que la abre va a «Probar»: la caja de
    verdad es una pantalla emparejada, y cada foto suya se cobra."""
    org = _org()
    if current_device() is None:
        return redirect(url_for("iacarry.demo"))
    return render_template("station.html", config=_station_config(org, "station"), zt_no_tour=True)


@bp.get("/demo")
def demo():
    """«Probar»: la caja tal cual, con fotos de ejemplo o una propia. Cada foto
    analizada se cobra como prueba (``demo_recognition``)."""
    return render_template("station.html", config=_station_config(_org(), "demo"), zt_no_tour=True)


def _detect(operation: str, demo: bool):
    org = _org()
    try:
        upload = read_upload(request.files.get("file"), kinds=FRAME_KINDS, max_mb=FRAME_MAX_MB)
    except UploadError as exc:
        return jsonify({"error": exc.key, "vars": exc.vars}), 400
    request_id = (request.headers.get("X-Request-Id") or "").strip()[:64] or uuid.uuid4().hex[:16]
    machine = current_device()
    station = service.station_for(org.id, machine.id) if machine is not None and not demo else None
    settings = service.settings_of(org.id)
    checkout_id = request.form.get("checkout", type=int)
    repeated = service.frame_by_request(org.id, request_id, station.id if station else None) \
        if request.headers.get("X-Request-Id") else None
    if repeated is not None:
        _log.info("Foto repetida (mismo X-Request-Id): la de antes | org=%s rid=%s", org.slug, request_id)
        return _answer(service.detection_of(repeated), request_id, repeated.checkout_id, repeated.id)
    started = time.monotonic()
    # El mismo X-Request-Id (un reintento tras un corte de red) se cobra una vez;
    # sin él, el núcleo junta la misma foto repetida en cinco minutos.
    digest = hashlib.sha256(upload.data).hexdigest()
    with charge(operation, inputs=(digest,), subject=machine.name if machine is not None else None) as work:
        if not work.allowed:
            _log.warning("Foto rechazada | org=%s op=%s motivo=%s rid=%s", org.slug, operation, work.reason,
                         request_id)
            return work.refusal()
        end_idle_transaction()       # el detector tarda: sin una conexión cogida mientras
        try:
            found = detector.detect(upload.data, upload.name, request_id)
        except detector.DetectorError as exc:
            _log.error("Detector | org=%s rid=%s %s", org.slug, request_id, exc)
            return jsonify({"error": exc.key, "request_id": request_id}), 503
        key = None
        if settings.keep_frames:
            key = retention.org_key("frames/", f"{uuid.uuid4().hex}.{upload.kind}", org_id=org.id)
            current_storage().save(key, upload.data, f"image/{'jpeg' if upload.kind == 'jpg' else upload.kind}")
        checkout, frame = service.record(org.id, current_user.id, found, station_id=station.id if station else None,
                                         demo=demo, request_id=request_id, checkout_id=checkout_id, storage_key=key,
                                         round_trip_ms=int((time.monotonic() - started) * 1000))
        work.done(reference=f"frame {frame.id}")
    _log.info("Foto analizada | org=%s caja=%s rid=%s unidades=%s total=%s ms=%s", org.slug,
              station.id if station else "-", request_id, checkout.units, checkout.total_cents, frame.round_trip_ms)
    return _answer(found, request_id, checkout.id, frame.id)


def _answer(found, request_id, checkout_id, frame_id):
    """El contrato de siempre de /upload, más la compra y la foto apuntadas."""
    body = found.for_page()
    body.update({"request_id": request_id, "checkout_id": checkout_id, "frame_id": frame_id})
    answer = jsonify(body)
    answer.headers["X-Request-Id"] = request_id
    return answer


@bp.post("/station/detect")
@device
def station_detect():
    if current_device() is None:
        return jsonify({"error": "errStationOnly"}), 403
    return _detect("recognition", demo=False)


@bp.post("/demo/detect")
def demo_detect():
    return _detect("demo_recognition", demo=True)


def _paid(station_only: bool, checkout_id: int):
    org = _org()
    machine = current_device()
    station = service.station_for(org.id, machine.id) if machine is not None and station_only else None
    data = request.get_json(silent=True) or {}
    checkout = service.mark_paid(org.id, checkout_id, station.id if station else None,
                                 simulated=bool(data.get("simulated", True)), ref=str(data.get("ref") or ""))
    if checkout is None:
        return jsonify({"error": "errNotFound"}), 404
    return jsonify({"status": checkout.status, "checkout_id": checkout.id})


@bp.post("/station/checkout/<int:checkout_id>/paid")
@device
def station_paid(checkout_id):
    if current_device() is None:
        return jsonify({"error": "errStationOnly"}), 403
    return _paid(True, checkout_id)


@bp.post("/demo/checkout/<int:checkout_id>/paid")
def demo_paid(checkout_id):
    return _paid(False, checkout_id)


@bp.app_template_filter("euros")
def euros(cents) -> str:
    """1234 → «12,34 €» («€12.34» en inglés), como en la caja."""
    cents = int(cents or 0)
    if request_language() == "en":
        return f"€{cents // 100}.{cents % 100:02d}"
    return f"{cents // 100},{cents % 100:02d} €"


@bp.app_template_filter("kilos")
def kilos(grams) -> str:
    text = f"{int(grams or 0) / 1000:.2f}"
    return f"{text if request_language() == 'en' else text.replace('.', ',')} kg"

