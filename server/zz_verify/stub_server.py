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
    template_folder=SERVER_DIR,
    static_folder=os.environ.get("STATIC", os.path.join(SERVER_DIR, "static")),
    # Pinned: Flask otherwise derives the URL prefix from the folder's basename,
    # so serving from a temporary directory would silently move /static.
    static_url_path="/static",
)

with open(os.path.join(SERVER_DIR, "zz_sample_upload_response.json")) as fh:
    FIXTURE = json.load(fh)


@app.route("/")
def home_index():
    return render_template("iaCarry_Local_JS_1_Clouding.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if DELAY:
        time.sleep(DELAY)
    f = request.files.get("file")
    print("[stub] /upload MODE=%s file=%r bytes=%s"
          % (MODE, getattr(f, "filename", None), len(f.read()) if f else 0), flush=True)
    if MODE == "hang":
        time.sleep(300)
    if MODE == "error500":
        return Response("boom", status=500)
    if MODE == "texterr":
        # The real route answers plain strings on its error paths.
        return "File not allowed. Only allowned: png, jpg, jpge"
    if MODE == "empty":
        return json.dumps({**FIXTURE, "predictions": []})
    body = dict(FIXTURE)
    if MODE == "square":
        body["shape_img"] = [800, 800, 3]
    return json.dumps(body)


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
