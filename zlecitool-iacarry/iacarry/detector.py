# -*- coding: utf-8 -*-
"""El detector: la red que reconoce los productos de una foto del carro.

TensorFlow no se carga nunca en los procesos web (gunicorn arranca varios, y
cada uno cargaría el modelo): el detector es un servicio interno aparte,
``tuisku_iaCarry/serving/detector_service.py``, y aquí sólo se le llama.

    IACARRY_DETECTOR_URL     http://detector:8600 (sin ella, el falso)
    IACARRY_DETECTOR_TOKEN   la ficha compartida con el servicio
    IACARRY_DETECTOR_TIMEOUT segundos (20)

Su contrato, ``POST /v1/detect`` con la foto (multipart ``file``) y
``X-Request-Id``:

    {"model_version": "…", "shape_img": [alto, ancho, 3], "inference_ms": 840,
     "predictions": [{"probability": 0.97, "tagName": "cocacola_33cl",
                      "box": {"x1": .2, "y1": .1, "x2": .35, "y2": .4}}]}

con las esquinas con su nombre de verdad (``x1, y1, x2, y2``, de 0 a 1). Un
503 es «ocupado» (la cola está llena): se dice en el acto, no se espera.

**El falso** (sin ``IACARRY_DETECTOR_URL``, o ``fake``): contesta las fotos de
ejemplo con su verdad dibujada a mano (``fixtures/<foto>.json``, las de
``serving/demo_labels``) y cualquier otra con ``fixtures/_default.json``. Las
pruebas lo usan siempre, y en local deja usar iaCarry entero sin TensorFlow.
``fake_detector()`` dice qué contesta mientras dure.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

URL_ENV = "IACARRY_DETECTOR_URL"
TOKEN_ENV = "IACARRY_DETECTOR_TOKEN"
TIMEOUT_ENV = "IACARRY_DETECTOR_TIMEOUT"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


class DetectorError(Exception):
    """El detector no contestó bien: la caja se pone en rojo («avisa a un asistente»)."""
    key = "errDetectorDown"


class DetectorBusy(DetectorError):
    key = "errDetectorBusy"


@dataclass
class Prediction:
    tag: str
    probability: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Detection:
    model_version: str
    height: int
    width: int
    predictions: List[Prediction] = field(default_factory=list)
    inference_ms: Optional[int] = None

    def for_page(self) -> dict:
        """Lo que la caja espera: el formato de siempre de /upload, el que lee su
        adaptador (``boundingBox`` con la esquina derecha en ``width`` y la de
        abajo en ``height``, como las mandaba TensorFlow)."""
        return {"model_version": self.model_version, "shape_img": [self.height, self.width, 3],
                "inference_ms": self.inference_ms,
                "predictions": [{"probability": p.probability, "tagName": p.tag,
                                 "boundingBox": {"left": p.x1, "top": p.y1, "width": p.x2, "height": p.y2}}
                                for p in self.predictions]}

    def as_rows(self) -> list:
        return [{"tag": p.tag, "p": round(p.probability, 4), "box": [p.x1, p.y1, p.x2, p.y2]}
                for p in self.predictions]


def _clip(value) -> float:
    return max(0.0, min(1.0, float(value)))


def from_v1(data: dict) -> Detection:
    """Una respuesta del servicio (``/v1/detect``)."""
    shape = data.get("shape_img") or [0, 0, 3]
    predictions = []
    for item in data.get("predictions") or []:
        box = item.get("box") or {}
        try:
            predictions.append(Prediction(str(item["tagName"]), float(item["probability"]), _clip(box["x1"]),
                                          _clip(box["y1"]), _clip(box["x2"]), _clip(box["y2"])))
        except (KeyError, TypeError, ValueError):
            continue          # una predicción rota no tumba la foto entera
    return Detection(str(data.get("model_version") or ""), int(shape[0] or 0), int(shape[1] or 0), predictions,
                     data.get("inference_ms"))


def from_legacy(data: dict, model_version: str = "fixture") -> Detection:
    """Una respuesta con el formato de antes (las verdades de ``fixtures/``)."""
    shape = data.get("shape_img") or [0, 0, 3]
    predictions = []
    for item in data.get("predictions") or []:
        box = item.get("boundingBox") or {}
        try:
            predictions.append(Prediction(str(item["tagName"]), float(item["probability"]), _clip(box["left"]),
                                          _clip(box["top"]), _clip(box["width"]), _clip(box["height"])))
        except (KeyError, TypeError, ValueError):
            continue
    return Detection(model_version, int(shape[0] or 0), int(shape[1] or 0), predictions, 0)


# ── el falso ────────────────────────────────────────────────────────────────

_FAKE: Dict[str, object] = {}


@contextmanager
def fake_detector(replies: Optional[dict] = None, default=None, delay: float = 0.0):
    """Lo que contesta el detector falso mientras dure: por nombre de foto (sin
    extensión) o ``default``, un ``Detection``, un dict con el formato de antes
    o una excepción (``DetectorBusy()``…). Da la lista de las llamadas."""
    previous = dict(_FAKE)
    calls: list = []
    _FAKE.clear()
    _FAKE.update({"replies": dict(replies or {}), "default": default, "delay": delay, "calls": calls})
    try:
        yield calls
    finally:
        _FAKE.clear()
        _FAKE.update(previous)


def _fake(data: bytes, filename: str, request_id: str) -> Detection:
    stem = Path(filename or "").stem
    if _FAKE.get("calls") is not None:
        _FAKE["calls"].append({"filename": filename, "request_id": request_id, "bytes": len(data)})
    if _FAKE.get("delay"):
        time.sleep(float(_FAKE["delay"]))
    reply = (_FAKE.get("replies") or {}).get(stem, _FAKE.get("default"))
    if reply is None:
        path = FIXTURES / f"{stem}.json"
        if not path.is_file():
            path = FIXTURES / "_default.json"
        reply = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(reply, BaseException):
        raise reply
    if isinstance(reply, Detection):
        return reply
    return from_legacy(reply)


# ── el de verdad ────────────────────────────────────────────────────────────

def configured_url() -> Optional[str]:
    url = (os.environ.get(URL_ENV) or "").strip().rstrip("/")
    return None if url in ("", "fake") else url


def is_fake() -> bool:
    return configured_url() is None


def _multipart(data: bytes, filename: str):
    boundary = uuid.uuid4().hex
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n").encode("utf-8")
    return head + data + f"\r\n--{boundary}--\r\n".encode("utf-8"), f"multipart/form-data; boundary={boundary}"


def detect(data: bytes, filename: str, request_id: str) -> Detection:
    """Los productos de una foto. Lanza ``DetectorBusy`` o ``DetectorError``.

    Quien llama suelta antes su transacción (``db.end_idle_transaction()``):
    esperar al detector con una conexión cogida dejaría sin ellas al resto."""
    url = configured_url()
    if url is None:
        return _fake(data, filename, request_id)
    body, content_type = _multipart(data, filename or "frame.jpg")
    request = urllib.request.Request(f"{url}/v1/detect", data=body, method="POST", headers={
        "Content-Type": content_type, "X-Request-Id": request_id,
        "Authorization": f"Bearer {os.environ.get(TOKEN_ENV, '')}"})
    timeout = float(os.environ.get(TIMEOUT_ENV) or 20)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:    # noqa: S310 - URL de configuración
            return from_v1(json.loads(answer.read().decode("utf-8")))
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            raise DetectorBusy(f"detector ocupado ({exc.code})") from exc
        raise DetectorError(f"el detector contestó {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise DetectorError(f"{exc.__class__.__name__}: {exc}") from exc
