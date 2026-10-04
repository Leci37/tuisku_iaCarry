# -*- coding: utf-8 -*-
"""La caja: cada foto se analiza con el detector, se apunta y se cobra a la
empresa (``recognition``), una vez aunque se reintente, y nunca si el detector
falla. Sólo una pantalla emparejada de la empresa la usa como caja."""

import csv
import io
import json
import uuid

from zlecitool_core import testing
from zlecitool_core.credits import balance

from iacarry import detector
from iacarry.models import Checkout, Frame

from conftest import paired_station, photo

JSON = {"Sec-Fetch-Dest": "empty"}


def _movements(app, monkeypatch, org):
    """Los movimientos del saldo de la empresa, como los ve tuisku en «Clientes» (su CSV)."""
    monkeypatch.setenv("ZLECITOOL_ADMINS", "admin@tuisku.test")
    admin = testing.signed_in_client(app, email="admin@tuisku.test", org=org)
    answer = admin.get(f"/zt/admin/orgs/{org.slug}/movements.csv")
    assert answer.status_code == 200
    return list(csv.DictReader(io.StringIO(answer.get_data(as_text=True))))


def _detect(client, url="/station/detect", n=1, rid=None, **form):
    data = {"file": (io.BytesIO(photo(n)), f"ziacarry_eval_img_{n}.png"), **form}
    headers = {**JSON, "X-Request-Id": rid or uuid.uuid4().hex[:12]}
    return client.post(url, data=data, headers=headers, content_type="multipart/form-data")


def test_a_frame_is_recognised_recorded_and_charged_to_the_company(app, shop, owner, monkeypatch):
    station = paired_station(app, owner)
    answer = _detect(station)
    body = answer.get_json()
    assert answer.status_code == 200 and len(body["predictions"]) == 12, "la verdad de la foto 1"
    assert body["predictions"][0]["boundingBox"].keys() == {"left", "top", "width", "height"}, "el contrato de siempre"
    with app.app_context():
        assert balance(org_id=shop.id) == 49
        checkout = Checkout.query.one()
        assert (checkout.units, checkout.lines, checkout.total_cents, checkout.demo) == (12, 12, 2631, False)
        assert checkout.expected_weight_g == 4193 and checkout.station_id is not None
        frame = Frame.query.one()
        assert frame.storage_key.startswith(f"iacarry/frames/o{shop.id}/"), "en la carpeta con borrado de su empresa"
    charged = [row for row in _movements(app, monkeypatch, shop) if row["reason"] == "recognition"]
    assert len(charged) == 1 and charged[0]["amount"] == "-1" and charged[0]["tool"] == "iacarry"
    assert charged[0]["user_email"] not in ("", owner.email), "la paga la caja (su cuenta técnica), no una persona"


def test_the_same_request_id_is_analysed_and_charged_once(app, shop, owner):
    station = paired_station(app, owner)
    first = _detect(station, rid="r-corte-de-red").get_json()
    with detector.fake_detector() as calls:
        again = _detect(station, rid="r-corte-de-red").get_json()
    assert again["checkout_id"] == first["checkout_id"] and again["frame_id"] == first["frame_id"]
    assert calls == [], "ni se vuelve a analizar"
    assert len(again["predictions"]) == 12
    with app.app_context():
        assert balance(org_id=shop.id) == 49 and Frame.query.count() == 1


def test_frames_of_one_trolley_update_its_checkout(app, owner):
    station = paired_station(app, owner)
    first = _detect(station, n=1).get_json()
    second = _detect(station, n=2, checkout=first["checkout_id"]).get_json()
    assert second["checkout_id"] == first["checkout_id"]
    with app.app_context():
        assert Checkout.query.count() == 1 and Frame.query.count() == 2


def test_a_detector_failure_is_not_charged(app, shop, owner):
    station = paired_station(app, owner)
    with detector.fake_detector(default=detector.DetectorBusy("cola llena")):
        busy = _detect(station)
    with detector.fake_detector(default=detector.DetectorError("caído")):
        down = _detect(station)
    assert (busy.status_code, busy.get_json()["error"]) == (503, "errDetectorBusy")
    assert (down.status_code, down.get_json()["error"]) == (503, "errDetectorDown")
    with app.app_context():
        assert balance(org_id=shop.id) == 50 and Frame.query.count() == 0


def test_without_company_credits_the_station_is_refused(app):
    broke = testing.make_org(app, name="Sin saldo")
    owner = testing.signed_in_client(app, org=broke, role="owner")
    station = paired_station(app, owner)
    refused = _detect(station)
    assert refused.status_code == 402 and refused.get_json()["error"] == "errOrgCreditsNeeded"


def test_only_a_paired_station_is_a_station(app, owner, member):
    assert app.test_client().post("/station/detect", headers=JSON).status_code == 401
    assert member.get("/station").headers["Location"].endswith("/demo"), "una persona, a «Probar»"
    refused = _detect(member)
    assert refused.status_code == 403 and refused.get_json()["error"] == "errStationOnly"
    station = paired_station(app, owner)
    assert station.get("/", headers=JSON).get_json() == {"error": "errDeviceForbidden"}, "la trastienda no es suya"
    assert station.get("/checkouts", headers=JSON).status_code == 403


def test_a_revoked_station_stops_at_once(app, owner):
    from zlecitool_core.devices import Device
    station = paired_station(app, owner)
    with app.app_context():
        device_id = Device.query.one().id
    owner.post(f"/cuenta/empresa/devices/{device_id}/revoke")
    answer = _detect(station)
    assert answer.status_code == 401 and answer.get_json()["error"] == "errDeviceRevoked"


def test_paying_records_the_checkout(app, owner):
    station = paired_station(app, owner)
    checkout = _detect(station).get_json()["checkout_id"]
    paid = station.post(f"/station/checkout/{checkout}/paid", json={"simulated": True, "ref": "DEMO-1"}, headers=JSON)
    assert paid.get_json() == {"status": "paid", "checkout_id": checkout}
    with app.app_context():
        row = Checkout.query.one()
        assert row.status == "paid" and row.payment_simulated and row.payment_ref == "DEMO-1"


def test_the_station_page_carries_the_companys_data(app, owner):
    station = paired_station(app, owner, name="Caja 7")
    page = station.get("/station").get_data(as_text=True)
    config = json.loads(page.split("data-config='", 1)[1].split("'", 1)[0].replace("&#39;", "'"))
    assert config["mode"] == "station" and config["detectUrl"] == "/station/detect" and config["photos"] == []
    assert len(config["catalogue"]) == 13 and config["label"] == "Caja 7" and config["initials"] == "C7"
    assert config["themes"][0]["key"] == "iacarry", "sin marca propia, la de iaCarry"
    assert "<script>" not in page and "onerror=" not in page and 'src="/static/js/station.js"' in page


def test_try_it_charges_a_demo_recognition(app, shop, member, monkeypatch):
    answer = _detect(member, url="/demo/detect", n=3)
    assert answer.status_code == 200 and len(answer.get_json()["predictions"]) == 13
    charged = [row for row in _movements(app, monkeypatch, shop) if row["reason"] == "demo_recognition"]
    assert len(charged) == 1 and charged[0]["user_email"] == member.email
    with app.app_context():
        assert balance(org_id=shop.id) == 49 and Checkout.query.one().demo is True
    page = member.get("/demo").get_data(as_text=True)
    assert '"mode": "demo"' in page and "ziacarry_eval_img_1.png" in page and 'id="uploadFrame"' in page


def test_a_demo_company_shows_the_retailer_themes(app, shop, owner):
    cli = app.test_cli_runner()
    assert "es la empresa de demostración" in cli.invoke(args=["iacarry", "demo", shop.slug]).output
    station = paired_station(app, owner)
    page = station.get("/station").get_data(as_text=True)
    assert "/static/logos/mercadona-logo.png" in page and '"photos": [{' in page
    other = testing.make_org(app, name="Cliente normal", credits=5)
    other_owner = testing.signed_in_client(app, org=other, role="owner")
    assert "mercadona" not in paired_station(app, other_owner, name="Caja B").get("/station").get_data(as_text=True)


def test_a_company_can_choose_not_to_keep_photos(app, shop, owner):
    app.test_cli_runner().invoke(args=["iacarry", "frames", shop.slug, "--off"])
    _detect(paired_station(app, owner))
    with app.app_context():
        frame = Frame.query.one()
        assert frame.storage_key is None and json.loads(frame.detections), "lo detectado sí se queda"
