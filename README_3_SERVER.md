# 3 — Real-time server (backend)

The Flask application that holds the model in memory, takes a frame over HTTP and
answers with predictions. Everything in `server/` except the page itself, which
is `README_4_FRONTEND.md`.

`serving/FLOW.md` documents the same ground from the customer's point of view,
with the exact log line each step emits. This file is the backend's own
reference: what loads, what each route does, and where a number can change.

---

## The shape of it

```
browser ──POST /upload (multipart "file", X-Request-Id)──► Flask
                                                            │
                                    save frame to disk ◄────┤
                                                            │
                            Detector_model (singleton) ◄────┤
                              tf.saved_model, "detect"       │
                              serialised by inc_lock         │
                                                            │
                            shape response to client ◄──────┤
                                                            │
browser ◄──── JSON + X-Request-Id ──────────────────────────┘
```

One `X-Request-Id` per frame, minted in the browser, stamped on **every** log
line in all three layers. A whole purchase is one `grep`.

---

## Files

| File | Role |
|---|---|
| `serving/app.py` | The Flask app. Routes, upload handling, response assembly. **Entry point.** |
| `serving/detector.py` | `Detector_model` — loads the SavedModel once, runs inference under a lock |
| `serving/server_utils.py` | Save/sanitise the upload, reshape detections for the client, render + register results |
| `serving/log_utils.py` | Logging setup |
| `serving/visualization_utils.py` | Box drawing, lifted out of the Object Detection API so the server does not depend on it at runtime |
| `serving/sample_upload_response.json` | A recorded real `/upload` response — the fixture the verification suite replays |
| `serving/FLOW.md` | End-to-end narrative, user level and code level |
| `serving/verify/` | The verification suite (see `README_4_FRONTEND.md`) |
| `serving/check_instances.py` | Ops helper — counts live instances |

---

## Startup

`serving/app.py`, in order:

1. **Logging first**, before anything else is imported, so import-time failures
   are captured:
   ```python
   logging.basicConfig(level=DEBUG,
                       filename=r"..\iacarry-evaluation\zlog_" + datetime.now().strftime("%Y_%m_%d"),
                       filemode="a+")
   ```
   plus a console handler. One log file per day.
2. Records its own PID and parent PID (`psutil`) — with the reloader on, there
   are two processes and two model loads, and this is how you tell which is which.
3. `is_running_from_reloader()` prints a restart banner.
4. Template lookup — **the detail that costs the most time when it is wrong**:
   ```python
   TEMPLATE_FOLDER = r"..\iacarry-evaluation"
   app.jinja_loader = ChoiceLoader([FileSystemLoader(SERVER_DIR),
                                    FileSystemLoader(TEMPLATE_FOLDER)])
   ```
   `server/` wins, the legacy folder is the fallback. The app logs
   `serving template from: <path>` on every page load because editing the copy
   Flask is *not* reading is the classic wasted hour.
5. Upload folder: `<TEMPLATE_FOLDER>\_uploads_img_for_test_from_web`,
   subfoldered by day (`DATE_NAME_FOLDER = "%Y_%m_%d"`).
6. `ALLOWED_EXTENSIONS = {"png", "jpg", "jpge"}`.
7. The detector is attached to the app (`current_app.detector1`) so the model is
   loaded once, not per request.

---

## Routes

### `GET /` → `home_index()`

Logs the caller's IP, renders `serving/templates/iacarry_checkout.html` through the
`ChoiceLoader`, logs which file it resolved to.

### `POST /upload` → `upload_file()`

```python
rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
```

The browser's id is honoured if present; otherwise one is minted. It goes back
out on the response:

```python
response.headers["X-Request-Id"] = rid
response.headers["Access-Control-Expose-Headers"] = "X-Request-Id"
```

That second header is not decoration — **without it the browser cannot read
`X-Request-Id` cross-origin**, and the correlation breaks exactly when the server
is on another host, which is the case that matters.

Rejection paths, each logged with its reason (they used to fail silently):

| Condition | Log |
|---|---|
| no `file` part | `REJECTED: no 'file' part in the form (parts=[…])` |
| empty filename | `REJECTED: empty filename` |
| extension not allowed | `REJECTED: extension not allowed name=…` |

Then: `save_img_loaded()` → `do_prediction_from_list_paths()` →
`change_format_dict_json_to_client()` → JSON.

`GET /upload` is accepted and does nothing but say so.

**There is no `/payment` route.** The front end knows this and simulates payment;
see `README_4_FRONTEND.md`.

---

## The model — `serving/detector.py`

```python
SIGNATURE_REF = "detect"
PATH_TO_SAVED_MODEL_INTERFACE_GRAPH = "model_efi_d1C/save_model_sig_54"
PATH_PICKLE_CAT_INDEX = "model_efi_d1C/P_Category_index.pickle"
MIN_SCORE = 0.5
PATH_TO_SAVED = r"..\iacarry-evaluation\_upload_img_bbox_results"
```

On construction: `tf.saved_model.load(...)`, log the available signatures, take
`signatures["detect"]`, unpickle `Category_index`. If the signature is absent the
model was not exported with `exporter_main_v2.py` — the debug line at line 48
says so, because it is the most common way this fails.

`do_prediction_from_list_paths(path_img, rid)`:

```python
np.array(Image.open(path))
with inc_lock:                                   # serialises inference
    tf.convert_to_tensor(float32)[tf.newaxis, ...]
    detections = self.detector(input_tensor=...)
manage_plot_and_save_img_predicted(...)          # matplotlib render to disk
register_MULTI_in_zTelegram_Registers(...)       # CSV append
# tensors → numpy, drop unused keys, class ids +1, map through Category_index
```

**`inc_lock` is the throughput ceiling.** A second customer's frame waits for the
first. From the browser both look like "slow", so the model logs queueing and
compute separately:

```
[SGTON][r…] lock acquired after 0.002s of queueing
[SGTON][r…] inference done in 1.184s (lock held 1.231s)
```

Two things happen after inference that the browser never sees: a matplotlib
render written beside the upload, and a CSV row per detection
(`.../img_<N>/df_test_img.csv`). Both are server-side bookkeeping — the browser
draws its own boxes and never fetches `path_server`.

The `+1` on class ids is `LABEL_ID_OFFSET`: the model emits 0-based ids,
`Category_index` is 1-based. Getting it wrong shifts every label by one product,
which looks like a badly trained model and is not.

---

## Response shaping — `change_format_dict_json_to_client()`

```json
{ "path_server": "..\\…\\z_4_.jpg",
  "shape_img": [720, 1280, 3],
  "request_id": "r…",
  "predictions": [ { "probability": 0.98, "tagInt": 1, "tagName": "cocacola_33cl",
                     "boundingBox": {"left":0.2,"top":0.1,"width":0.35,"height":0.4} } ] }
```

> ⚠️ **`width` and `height` are not sizes.** TensorFlow emits
> `[ymin, xmin, ymax, xmax]`. This function maps them onto keys named
> `left/top/width/height`, so **`width` carries the right edge and `height` the
> bottom edge**, normalised 0–1. The client subtracts. Anything that reads them
> as extents draws boxes that grow steadily wronger toward the bottom-right —
> which reads as a weak model and is not one. Documented here, in `FLOW.md` §6,
> and enforced by the verification suite, because it is the sharpest edge in the
> whole contract.

---

## The three thresholds

A detection has to clear three separate bars, in three different files:

| Threshold | Where | Effect | Log |
|---|---|---|---|
| `MIN_SCORE = 0.5` | `serving/detector.py` | rows in the CSV register and the rendered figure | `RESULT raw=… above MIN_SCORE` |
| `MIN_SCORE_TO_CLIENT = 0.1` | `serving/app.py:88` | what is put on the wire | `shaped response: 100 raw -> 13 sent` |
| `DETECTION_MIN_SCORE = 0.45` | the page | what the customer sees | `detections adapted {uiThreshold: 0.45}` |

Everything between 0.1 and 0.45 is **sent and never shown**. That gap is
invisible unless both numbers are logged together, which is what the
`note:` line exists for. If products go missing from the screen, this table is
where to look first.

---

## Reading the logs

```bash
grep rmsj9wdlsuvqh browser-console.txt zlog_2026_08_07
```

```
browser  [iaCarry][r…] posting frame to the detector {bytes:783149}
Flask    [r…] /upload POST from 10.0.0.7
Flask    [SAVE][r…] wrote …/img_2026_08_07/ziacarry_eval_img_1.png
model    [SGTON][r…] lock acquired after 0.002s of queueing
model    [SGTON][r…] inference done in 1.184s
model    [SGTON][r…] RESULT raw=100 above MIN_SCORE=0.5: 11 | top: …
Flask    [SHAPE][r…] shaped response: 100 raw -> 13 sent, 87 below …
Flask    [r…] /upload DONE predictions=13 detector=1.631s total=1.844s
browser  [iaCarry][r…] detector responded {roundTripMs:1019}
browser  [iaCarry][r…] detections adapted {unitsShown:11, totalEur:22.83}
```

Answerable from that alone: slow because of **queueing** or **inference**? did
the model find less, or did a **threshold** drop it? is a disagreement in the
**model**, the **contract**, or the **screen**?

---

## Running it

```bash
python serving/app.py        # needs TensorFlow + the model folder
python3 serving/verify/verify.py           # 78 checks, no TensorFlow needed
```

The verification suite runs against `verify/stub_server.py`, which serves the
real template and the real static folder but replays
`sample_upload_response.json` instead of inferring. That is how the front end
can be exercised without the model — and equally, why **the suite establishes
nothing about the detector**.

---

## What is missing from the server

1. **It cannot start from a clean clone.** `model_efi_d1C/save_model_sig_54` and
   `P_Category_index.pickle` are not in the repository and there is no script to
   fetch them.
2. **Windows-only paths, hardcoded.** `TEMPLATE_FOLDER = r"..\iacarry-evaluation"`,
   the log path and `PATH_TO_SAVED` are all backslash literals pointing outside
   the repository. **The server does not start on Linux**, and the folder it
   depends on is not version-controlled.
3. **`ALLOWED_EXTENSIONS` contains `"jpge"`** — a typo for `"jpeg"`
   (`serving/app.py:51`). Harmless for the PNG demos; it rejects the
   first real `.jpeg` a camera sends.
4. **`inc_lock` means one customer at a time.** Fine for a demo, not for a shop
   floor with two stations. There is no queue depth limit, no 503 when saturated,
   and no batching — the second customer just waits.
5. **No `/payment` route.** The seam is defined and the client is ready
   (`PAYMENT_BACKEND`), but nothing implements it.
6. **No health or readiness endpoint.** Nothing to ask "is the model loaded?"
   short of posting a frame.
7. **No authentication and no rate limiting** on `/upload`. Anyone who can reach
   the host can queue inference and write files into the upload folder.
8. **Unbounded disk growth.** Every frame, every rendered figure and a daily CSV
   are written and never cleaned up.
9. **No configuration layer.** Model path, thresholds, folders and port are all
   literals in source. Changing the model means editing Python.
10. **Nothing tests the server itself.** `verify` covers the browser↔contract
    seam against a stub; `serving/app.py`, `serving/detector.py`
    and `serving/server_utils.py` have no tests at all.
