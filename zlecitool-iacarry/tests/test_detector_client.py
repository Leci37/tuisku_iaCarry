# -*- coding: utf-8 -*-
"""El cliente del detector de verdad (``IACARRY_DETECTOR_URL``) contra un
servidor local que habla su contrato (``POST /v1/detect``): la foto y las
cabeceras que le llegan, la traducción al formato de la caja, y «ocupado» o
caído como claves. Sin TensorFlow ni red de fuera."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from iacarry import detector

from conftest import paired_station, photo

V1 = {"model_version": "efi_d1C-54", "shape_img": [640, 640, 3], "inference_ms": 840,
      "predictions": [{"probability": 0.97, "tagName": "cocacola_33cl",
                       "box": {"x1": 0.2, "y1": 0.1, "x2": 0.35, "y2": 0.4}},
                      {"probability": 0.9, "tagName": "fanta_33cl", "box": {"x1": 0.5}}]}


@pytest.fixture
def service(monkeypatch):
    """Un detector de mentira en un puerto libre: contesta ``reply`` (estado, cuerpo)
    y apunta lo que le llega."""
    seen, reply = [], {"status": 200, "body": V1}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):          # noqa: N802 - el nombre lo pone http.server
            length = int(self.headers.get("Content-Length") or 0)
            seen.append({"path": self.path, "headers": dict(self.headers), "body": self.rfile.read(length)})
            data = json.dumps(reply["body"]).encode("utf-8")
            self.send_response(reply["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(detector.URL_ENV, f"http://127.0.0.1:{server.server_port}/")
    monkeypatch.setenv(detector.TOKEN_ENV, "s3creto")
    monkeypatch.setenv(detector.TIMEOUT_ENV, "5")
    yield {"seen": seen, "reply": reply}
    server.shutdown()


def test_the_frame_goes_to_v1_with_its_request_id_and_token(service):
    found = detector.detect(photo(1), "ziacarry_eval_img_1.png", "r-123")
    sent = service["seen"][0]
    assert sent["path"] == "/v1/detect" and sent["headers"]["X-Request-Id"] == "r-123"
    assert sent["headers"]["Authorization"] == "Bearer s3creto"
    assert sent["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
    assert photo(1) in sent["body"] and b'name="file"; filename="ziacarry_eval_img_1.png"' in sent["body"]
    assert (found.model_version, found.width, found.height, found.inference_ms) == ("efi_d1C-54", 640, 640, 840)
    assert [p.tag for p in found.predictions] == ["cocacola_33cl"], "una predicción rota se salta, no tumba la foto"
    assert found.for_page()["predictions"][0]["boundingBox"] == {"left": 0.2, "top": 0.1, "width": 0.35,
                                                                 "height": 0.4}, \
        "la caja recibe el formato de siempre: width y height son el borde derecho e inferior"


@pytest.mark.parametrize("status, error", [(503, detector.DetectorBusy), (500, detector.DetectorError),
                                           (401, detector.DetectorError)])
def test_busy_and_failures_are_errors_with_their_key(service, status, error):
    service["reply"].update(status=status, body={"error": "x"})
    with pytest.raises(error) as raised:
        detector.detect(photo(1), "f.png", "r")
    assert raised.type is error
    assert raised.value.key == ("errDetectorBusy" if status == 503 else "errDetectorDown")


def test_a_detector_that_is_not_there_is_down(monkeypatch):
    monkeypatch.setenv(detector.URL_ENV, "http://127.0.0.1:9")      # el puerto «discard»: nadie escucha
    monkeypatch.setenv(detector.TIMEOUT_ENV, "2")
    with pytest.raises(detector.DetectorError) as raised:
        detector.detect(photo(1), "f.png", "r")
    assert raised.value.key == "errDetectorDown"


def test_a_station_with_the_real_client_charges_and_answers_the_page(app, owner, service):
    station = paired_station(app, owner)
    answer = station.post("/station/detect", data={"file": (io.BytesIO(photo(1)), "f.png")},
                          headers={"Sec-Fetch-Dest": "empty", "X-Request-Id": "r-caja"},
                          content_type="multipart/form-data")
    body = answer.get_json()
    assert answer.status_code == 200 and body["model_version"] == "efi_d1C-54"
    assert body["predictions"][0]["boundingBox"]["width"] == 0.35
    assert service["seen"][0]["headers"]["X-Request-Id"] == "r-caja", "el mismo id en la caja, la web y el detector"
    service["reply"].update(status=503, body={"error": "busy"})
    busy = station.post("/station/detect", data={"file": (io.BytesIO(photo(2)), "g.png")},
                        headers={"Sec-Fetch-Dest": "empty", "X-Request-Id": "r-ocupado"},
                        content_type="multipart/form-data")
    assert busy.status_code == 503 and busy.get_json()["error"] == "errDetectorBusy"
