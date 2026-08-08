"""Stub of RUN_server_py_upload.py, for verifying the checkout UI.

Serves the real template and replays a recorded /upload response, so the whole
front end can be exercised without TensorFlow, the saved model directory or the
Windows-path evaluation folder — none of which exist outside the deployment box.

It is deliberately a stand-in for the Flask app only. It proves nothing about
the detector itself; see README.md for what this can and cannot establish.

Environment:
  MODE      ok | slow | empty | error500 | texterr | hang | square
  DELAY     seconds to sleep before answering /upload (simulates inference)
  PAY_MODE  ok | decline | http500      (for the /payment route)
  STATIC    static folder to serve at /static
  PORT      listen port
"""
import json
import os
import sys
import time

from flask import Flask, Response, render_template, request

SERVER_DIR = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
MODE = os.environ.get("MODE", "ok")
DELAY = float(os.environ.get("DELAY", "0"))
PAY_MODE = os.environ.get("PAY_MODE", "ok")

app = Flask(
    __name__,
    template_folder=os.path.join(SERVER_DIR, "templates"),
    static_folder=os.environ.get("STATIC", os.path.join(SERVER_DIR, "static")),
    # Pinned: Flask otherwise derives the URL prefix from the folder's basename,
    # so serving from a temporary directory would silently move /static.
    static_url_path="/static",
)

with open(os.path.join(SERVER_DIR, "sample_upload_response.json")) as fh:
    FIXTURE = json.load(fh)


@app.route("/")
def home_index():
    return render_template("iacarry_checkout.html")


def _echo_id(resp, rid):
    """Mirror the real route's correlation headers so the UI can be checked."""
    resp = Response(resp) if not isinstance(resp, Response) else resp
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["X-Request-Id"] = rid
    resp.headers["Access-Control-Expose-Headers"] = "X-Request-Id"
    return resp


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    rid = request.headers.get("X-Request-Id") or "stub-no-id"
    if DELAY:
        time.sleep(DELAY)
    f = request.files.get("file")
    print("[stub] /upload MODE=%s rid=%s file=%r bytes=%s"
          % (MODE, rid, getattr(f, "filename", None), len(f.read()) if f else 0), flush=True)
    if MODE == "hang":
        time.sleep(300)
    if MODE == "error500":
        return _echo_id(Response("boom", status=500), rid)
    if MODE == "texterr":
        # The real route answers plain strings on its error paths.
        return _echo_id("File not allowed. Only allowned: png, jpg, jpge", rid)
    if MODE == "empty":
        return _echo_id(json.dumps({**FIXTURE, "predictions": [], "request_id": rid}), rid)
    body = dict(FIXTURE)
    body["request_id"] = rid
    if MODE == "square":
        body["shape_img"] = [800, 800, 3]
    return _echo_id(json.dumps(body), rid)


@app.route("/payment", methods=["POST"])
def payment():
    """Not part of the real app. Exists only to verify the T6 payment seam."""
    order = request.get_json(silent=True) or {}
    print("[stub] /payment order=%r" % order, flush=True)
    if PAY_MODE == "decline":
        return json.dumps({"ok": False, "error": "card declined"})
    if PAY_MODE == "http500":
        return Response("gateway down", status=500)
    return json.dumps({"ok": True, "ref": "TX-9931"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8000")),
            debug=False, threaded=True)
