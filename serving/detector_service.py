# -*- coding: utf-8 -*-
"""El detector de iaCarry como servicio interno: lo llama la herramienta
``zlecitool-iacarry`` (``iacarry/detector.py``) y nadie más.

TensorFlow no puede vivir en los procesos web de la herramienta (gunicorn
arranca varios y cada uno cargaría el modelo): vive aquí, cargado **una vez**,
en un proceso por CPU o GPU, en la red interna. Del ``app.py`` de antes sólo
queda la inferencia: ni figura con matplotlib, ni CSV, ni un fichero escrito
por petición; la herramienta guarda la foto en su almacén si la empresa lo
quiere, con su borrado.

    POST /v1/detect   la foto en multipart ``file`` (PNG, JPEG o WebP, se
                      mira el contenido y no la extensión) y ``X-Request-Id``
    GET  /health      {"model_loaded": true, "model_version": "…"}

``/v1/detect`` contesta con las esquinas con su nombre de verdad (de 0 a 1):

    {"model_version": "…", "shape_img": [alto, ancho, 3], "inference_ms": 840,
     "predictions": [{"probability": 0.97, "tagName": "cocacola_33cl",
                      "box": {"x1": .2, "y1": .1, "x2": .35, "y2": .4}}]}

Un 503 ``{"error": "busy"}`` es «ocupado»: hay ya ``IACARRY_DETECTOR_QUEUE``
fotos esperando y se dice en el acto en vez de esperar sin fin (la caja se pone
en rojo, «avisa a un asistente»).

Todo sale del entorno, nada de rutas de Windows:

    IACARRY_MODEL_DIR          el saved_model de TensorFlow (obligatoria)
    IACARRY_CATEGORY_INDEX     el P_Category_index.pickle de sus clases (obligatoria)
    IACARRY_MODEL_VERSION      su nombre en las respuestas (por defecto, la carpeta)
    IACARRY_MODEL_SIGNATURE    la firma del modelo (detect)
    IACARRY_LABEL_ID_OFFSET    lo que se suma a cada clase para buscarla en el índice (1)
    IACARRY_DETECTOR_MIN_SCORE lo mínimo que se manda (0.3; la caja filtra otra vez a 0.45)
    IACARRY_DETECTOR_TOKEN     la ficha compartida con la herramienta (Authorization: Bearer)
    IACARRY_DETECTOR_QUEUE     cuántas fotos pueden esperar a la vez (4)
    IACARRY_DETECTOR_MAX_MB    lo que puede pesar una foto (8)
    PORT                       8600

    python detector_service.py                       desarrollo
    gunicorn -w 1 --threads 8 -b 0.0.0.0:8600 "detector_service:create_app()"

Un solo proceso por máquina (``-w 1``): cada proceso carga el modelo entero; los
hilos sólo esperan en la cola, la inferencia va de una en una.
"""

from __future__ import annotations

import hmac
import io
import logging
import os
import pickle
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from flask import Flask, jsonify, request

_log = logging.getLogger("iacarry.detector")

KINDS = {"PNG", "JPEG", "WEBP"}


@dataclass
class Found:
    """Una detección: la clase por su nombre, su probabilidad y su caja (0 a 1)."""
    tag: str
    probability: float
    x1: float
    y1: float
    x2: float
    y2: float


class TensorFlowModel:
    """El saved_model de TensorFlow, cargado una vez. TensorFlow se importa aquí
    dentro: las pruebas del servicio usan un modelo de mentira y no lo necesitan."""

    def __init__(self, model_dir: str, category_index: str, signature: str = "detect", label_offset: int = 1):
        import tensorflow as tf  # noqa: PLC0415 - sólo con el modelo de verdad

        self._tf = tf
        started = time.perf_counter()
        self._detect = tf.saved_model.load(model_dir).signatures[signature]
        with open(category_index, "rb") as handle:
            index = pickle.load(handle)
        self._names = {int(item["id"]): str(item["name"]) for item in index.values()}
        self._offset = label_offset
        _log.info("Modelo cargado en %.1f s | %s | firma=%s | %d clases", time.perf_counter() - started, model_dir,
                  signature, len(self._names))

    def __call__(self, image) -> List[Found]:
        import numpy as np  # noqa: PLC0415

        tensor = self._tf.convert_to_tensor(np.array(image), dtype=self._tf.float32)[self._tf.newaxis, ...]
        raw = self._detect(input_tensor=tensor)
        boxes = raw["detection_boxes"].numpy()[0]
        scores = raw["detection_scores"].numpy()[0]
        classes = raw["detection_classes"].numpy()[0]
        found = []
        for (y1, x1, y2, x2), score, number in zip(boxes, scores, classes):
            # TensorFlow da [ymin, xmin, ymax, xmax]: aquí cada esquina con su
            # nombre. La caja recibe el formato de antes de la herramienta, no de aquí.
            number = int(number) + self._offset
            found.append(Found(self._names.get(number, f"unknown_class_{number}"), float(score),
                               float(x1), float(y1), float(x2), float(y2)))
        return found


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


def load_model_from_env() -> Callable:
    model_dir = os.environ.get("IACARRY_MODEL_DIR", "")
    index = os.environ.get("IACARRY_CATEGORY_INDEX", "")
    if not model_dir or not index:
        raise RuntimeError("Faltan IACARRY_MODEL_DIR y IACARRY_CATEGORY_INDEX (el saved_model y su índice de clases).")
    return TensorFlowModel(model_dir, index, os.environ.get("IACARRY_MODEL_SIGNATURE") or "detect",
                           int(os.environ.get("IACARRY_LABEL_ID_OFFSET") or 1))


def create_app(model: Optional[Callable] = None, model_version: Optional[str] = None) -> Flask:
    """El servicio. ``model`` es lo que convierte una imagen PIL en una lista de
    ``Found``; sin él, el saved_model de ``IACARRY_MODEL_DIR``."""
    app = Flask(__name__)
    if model is None:
        model = load_model_from_env()
    version = model_version or os.environ.get("IACARRY_MODEL_VERSION") \
        or Path(os.environ.get("IACARRY_MODEL_DIR") or "modelo").name
    min_score = _env_float("IACARRY_DETECTOR_MIN_SCORE", 0.3)
    token = os.environ.get("IACARRY_DETECTOR_TOKEN") or ""
    queue = max(1, int(_env_float("IACARRY_DETECTOR_QUEUE", 4)))
    max_bytes = int(_env_float("IACARRY_DETECTOR_MAX_MB", 8) * 1024 * 1024)
    app.config["MAX_CONTENT_LENGTH"] = max_bytes + 64 * 1024

    # La inferencia va de una en una (el modelo no se comparte entre hilos); los
    # que esperan son la cola, y con ella llena se contesta «ocupado» al momento.
    inference = threading.Lock()
    waiting = threading.BoundedSemaphore(queue)

    def _refuse(status: int, key: str, rid: str):
        answer = jsonify({"error": key, "request_id": rid})
        answer.headers["X-Request-Id"] = rid
        return answer, status

    @app.get("/health")
    def health():
        return jsonify({"model_loaded": True, "model_version": version})

    @app.post("/v1/detect")
    def detect():
        rid = (request.headers.get("X-Request-Id") or "").strip()[:64] or uuid.uuid4().hex[:16]
        if token:
            sent = request.headers.get("Authorization", "")
            if not hmac.compare_digest(sent.encode("utf-8"), f"Bearer {token}".encode("utf-8")):
                _log.warning("[%s] ficha que no vale", rid)
                return _refuse(401, "unauthorized", rid)
        upload = request.files.get("file")
        data = upload.read() if upload is not None else b""
        if not data:
            return _refuse(400, "no_file", rid)
        if len(data) > max_bytes:
            return _refuse(413, "too_large", rid)
        from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

        try:
            image = Image.open(io.BytesIO(data))
            kind = image.format
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            kind = None
        if kind not in KINDS:
            _log.warning("[%s] no es una imagen PNG, JPEG o WebP (%s, %d bytes)", rid, kind, len(data))
            return _refuse(400, "not_an_image", rid)

        if not waiting.acquire(blocking=False):
            _log.warning("[%s] ocupado: %d fotos esperando ya", rid, queue)
            return _refuse(503, "busy", rid)
        try:
            queued = time.perf_counter()
            with inference:
                started = time.perf_counter()
                found = model(image)
                done = time.perf_counter()
        except Exception:                       # noqa: BLE001 - el modelo falló: 500 con su request id
            _log.exception("[%s] el modelo falló", rid)
            return _refuse(500, "model_failed", rid)
        finally:
            waiting.release()

        sent = [f for f in found if f.probability >= min_score]
        inference_ms = int((done - started) * 1000)
        _log.info("[%s] %dx%d %s | cola %.0f ms | inferencia %d ms | %d de %d sobre %.2f", rid, image.width,
                  image.height, kind, (started - queued) * 1000, inference_ms, len(sent), len(found), min_score)
        answer = jsonify({
            "model_version": version, "shape_img": [image.height, image.width, 3], "inference_ms": inference_ms,
            "predictions": [{"probability": round(f.probability, 4), "tagName": f.tag,
                             "box": {"x1": f.x1, "y1": f.y1, "x2": f.x2, "y2": f.y2}} for f in sent]})
        answer.headers["X-Request-Id"] = rid
        return answer

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT") or 8600), threaded=True)
