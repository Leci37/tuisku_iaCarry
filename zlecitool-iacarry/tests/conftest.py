# -*- coding: utf-8 -*-
"""iaCarry, aislado: datos, base de datos y secreto temporales, y el detector
falso (nunca TensorFlow ni la red en una prueba).

testing.isolated() fuerza una carpeta de datos, una base de datos y un secreto
de prueba aunque quien lance las pruebas tenga los de verdad en su entorno.
"""

import re
from pathlib import Path

import pytest

from zlecitool_core import testing

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "static" / "demo"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.delenv("IACARRY_DETECTOR_URL", raising=False)
    with testing.isolated(tmp_path):
        from app import create_app
        yield testing.prepare(create_app())


@pytest.fixture
def anonymous(app):
    return app.test_client()


@pytest.fixture
def shop(app):
    """Un supermercado cliente con iaCarry y saldo para unas pruebas."""
    return testing.make_org(app, name="Supermercados Norte", credits=50)


@pytest.fixture
def owner(app, shop):
    return testing.signed_in_client(app, org=shop, role="owner")


@pytest.fixture
def member(app, shop):
    return testing.signed_in_client(app, org=shop, role="member")


def photo(n: int = 1) -> bytes:
    """Una de las fotos de ejemplo (el detector falso conoce su verdad)."""
    return (DEMO / f"ziacarry_eval_img_{n}.png").read_bytes()


def new_screen(client, kind: str = "browser", name: str = "Caja 1") -> str:
    """Da de alta una máquina desde «Empresa» (del núcleo) y devuelve su código o su ficha."""
    page = client.post("/cuenta/empresa/devices", data={"name": name, "kind": kind}).get_data(as_text=True)
    found = re.search(r'value="([A-Z2-9]{4}-[A-Z2-9]{4}|ztk_[A-Za-z0-9_\-]+)"', page)
    assert found, "no salió el código de la máquina"
    return found.group(1)


def paired_station(app, owner, name: str = "Caja 1"):
    """Un navegador emparejado como caja de la empresa de ``owner``."""
    code = new_screen(owner, name=name)
    kiosk = app.test_client()
    token = testing.csrf_token_of(kiosk.get("/zt/pair"))
    answer = kiosk.post("/zt/pair", data={"code": code, "csrf_token": token})
    assert answer.status_code == 302 and answer.headers["Location"].endswith("/station")
    kiosk.environ_base["HTTP_X_CSRFTOKEN"] = testing.csrf_token_of(kiosk.get("/station"))
    return kiosk
