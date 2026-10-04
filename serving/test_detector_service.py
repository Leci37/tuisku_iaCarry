# -*- coding: utf-8 -*-
"""El servicio del detector con un modelo de mentira: sin TensorFlow, sin
modelo y sin red. Lo que se prueba es el contrato con la herramienta
(``zlecitool-iacarry/iacarry/detector.py``), la ficha, la cola y los errores.

    pip install flask pillow pytest
    pytest serving/test_detector_service.py
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import pytest
from PIL import Image

from detector_service import Found, create_app

DEMO = Path(__file__).resolve().parent / "static" / "assets" / "demo" / "ziacarry_eval_img_1.png"


def _image(kind: str = "PNG", size=(64, 48)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(out, format=kind)
    return out.getvalue()


def _post(client, data: bytes, name: str = "frame.png", **headers):
    return client.post("/v1/detect", data={"file": (io.BytesIO(data), name)}, headers=headers,
                       content_type="multipart/form-data")


def _model(found=None, seen=None):
    def model(image):
        if seen is not None:
            seen.append(image.size)
        return list(found or [Found("cocacola_33cl", 0.97, 0.2, 0.1, 0.35, 0.4), Found("fanta_33cl", 0.1, 0, 0, 1, 1)])
    return model


@pytest.fixture
def client(monkeypatch):
    for name in ("IACARRY_DETECTOR_TOKEN", "IACARRY_DETECTOR_QUEUE", "IACARRY_DETECTOR_MIN_SCORE"):
        monkeypatch.delenv(name, raising=False)
    return create_app(model=_model(), model_version="efi_d1C-54").test_client()


def test_health_says_the_model_is_loaded(client):
    assert client.get("/health").get_json() == {"model_loaded": True, "model_version": "efi_d1C-54"}


def test_a_frame_gets_the_v1_contract_with_honest_corners(client):
    answer = _post(client, _image(size=(64, 48)), **{"X-Request-Id": "r-1"})
    body = answer.get_json()
    assert answer.status_code == 200 and answer.headers["X-Request-Id"] == "r-1"
    assert body["model_version"] == "efi_d1C-54" and body["shape_img"] == [48, 64, 3]
    assert isinstance(body["inference_ms"], int)
    assert body["predictions"] == [{"probability": 0.97, "tagName": "cocacola_33cl",
                                    "box": {"x1": 0.2, "y1": 0.1, "x2": 0.35, "y2": 0.4}}], \
        "lo de debajo de IACARRY_DETECTOR_MIN_SCORE (0,3) no se manda"


def test_the_content_decides_not_the_extension(client):
    """El «jpge» de antes: se mira lo que es la foto, no cómo se llama."""
    assert _post(client, _image("JPEG"), name="foto.jpge").status_code == 200
    assert _post(client, _image("WEBP"), name="sin-extension").status_code == 200
    refused = _post(client, b"esto no es una imagen", name="foto.png")
    assert refused.status_code == 400 and refused.get_json()["error"] == "not_an_image"
    assert client.post("/v1/detect", data={}).get_json()["error"] == "no_file"


def test_with_a_token_only_the_tool_gets_in(monkeypatch):
    monkeypatch.setenv("IACARRY_DETECTOR_TOKEN", "s3creto")
    client = create_app(model=_model(), model_version="v").test_client()
    assert _post(client, _image()).status_code == 401
    assert _post(client, _image(), Authorization="Bearer otro").status_code == 401
    assert _post(client, _image(), Authorization="Bearer s3creto").status_code == 200
    assert client.get("/health").status_code == 200, "la salud no pide ficha (la mira el balanceador)"


def test_a_full_queue_answers_busy_at_once(monkeypatch):
    monkeypatch.setenv("IACARRY_DETECTOR_QUEUE", "1")
    inside, release = threading.Event(), threading.Event()

    def slow(image):
        inside.set()
        release.wait(5)
        return []

    app = create_app(model=slow, model_version="v")
    first = {}
    worker = threading.Thread(target=lambda: first.update(answer=_post(app.test_client(), _image())))
    worker.start()
    assert inside.wait(5)
    busy = _post(app.test_client(), _image(), **{"X-Request-Id": "r-cola"})
    assert busy.status_code == 503 and busy.get_json() == {"error": "busy", "request_id": "r-cola"}
    release.set()
    worker.join(5)
    assert first["answer"].status_code == 200
    assert _post(app.test_client(), _image()).status_code == 200, "con la cola libre, otra vez"


def test_a_model_failure_is_a_500_with_its_request_id(monkeypatch):
    def broken(image):
        raise RuntimeError("tensor roto")

    client = create_app(model=broken, model_version="v").test_client()
    answer = _post(client, _image(), **{"X-Request-Id": "r-roto"})
    assert answer.status_code == 500 and answer.get_json() == {"error": "model_failed", "request_id": "r-roto"}
    assert _post(create_app(model=_model(), model_version="v").test_client(), _image()).status_code == 200


def test_the_demo_photo_reaches_the_model_untouched(monkeypatch):
    """Las fotos de ejemplo son las de la evaluación (640×640): ni se reducen ni
    se recodifican, o la detección se aleja de la grabada."""
    if not DEMO.is_file():
        pytest.skip("sin las fotos de ejemplo")
    seen = []
    client = create_app(model=_model(seen=seen), model_version="v").test_client()
    body = _post(client, DEMO.read_bytes(), name=DEMO.name).get_json()
    assert seen == [(640, 640)] and body["shape_img"] == [640, 640, 3]
