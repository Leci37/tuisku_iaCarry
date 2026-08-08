# iaCarry checkout — the complete flow

From an empty station to a price on the screen, at both the user level and the
code level, with the log line each step emits.

Two things in this flow have **no backend today** and are simulated: the
**cart-presence sensor** and the **payment/door**. There is also **no camera
feed**, so frames come from the demo selector. Each is a single named function,
listed in §7. Everything else is real: the frame is genuinely posted to
`/upload`, genuinely run through the model, and every number on screen is
derived from the response.

---

## 1. The short version

```
 ┌────────────── browser ──────────────┐   ┌──────── Flask ────────┐   ┌──── model ────┐
 │ cart present  →  frame  →  POST     │──▶│ save → detect → shape │──▶│  TF saved_model│
 │ (simulated)     (demo)   /upload    │   │                       │◀──│  under a lock  │
 │                                     │◀──│ JSON + X-Request-Id   │   └───────────────┘
 │ adapt → qty/boxes/conf → render     │   └───────────────────────┘
 │ pay (simulated) → confirmation      │
 └─────────────────────────────────────┘
```

One `X-Request-Id` is minted per frame in the browser and stamped on every log
line in all three layers, so a single purchase is one `grep`.

---

## 2. User level

| # | What the customer does | What they see |
|---|---|---|
| 0 | Walks up to the station | Empty cart, *"No image analysed yet"*, `0.00€`, pay button greyed out, grey status strip |
| 1 | Places the trolley | Step 1 turns green, step 2 becomes active |
| 2 | Detection runs | Photo of the trolley appears with a spinner over it, *"Analysing the cart…"*; pay stays disabled |
| 3 | Detection succeeds | Coloured boxes on each product, cart fills with cards (`✓ × N`), total, estimated weight, green *"All correct, you can pay now"* |
| 3b | Detection fails or finds nothing | Red strip, *"Detection failed — please call an assistant"*, payment blocked |
| 4 | Taps pay | Button reads *"Processing payment…"* and is disabled |
| 5 | Payment confirmed | Green tick overlay, thanks, receipt line — plus an amber note: *"Demo mode: no payment gateway or door release is connected."* |
| 6 | Taps *New purchase* | Everything clears back to step 0 |

The customer never edits a quantity: detection is the only source of truth.

---

## 3. Code level, step by step

### Stage 0 — page load

| | |
|---|---|
| Route | `GET /` → `home_index()` — `serving/app.py` |
| Template | `render_template("iacarry_checkout.html")`, resolved by a `ChoiceLoader`: `server/` first, then the legacy `..\iacarry-evaluation` |
| Browser | one `<script>` builds `CATALOG`, `I18N`, `DEMO_SOURCES`, `state`, then `render()` + `fillDemo()` |
| State | `cartFull:false`, `detect.status:"idle"`, `pay.status:"idle"`, `live:null` |

```
Flask   INFO  @app.route(/) Load index.html IP_remote: {'ip': …}
Flask   INFO  serving template from: /…/server/iacarry_checkout.html
browser info  [iaCarry] ready — detection:/upload | payment:SIMULATED (no route)
                | cart sensor:manual stand-in toggle | camera feed:NONE (demo selector only)
```

The *serving template from* line exists because the template is looked up in two
places; editing the copy Flask is not reading is the classic wasted hour.

### Stage 1 — cart presence · **SIMULATED**

Real world: a physical sensor. In this codebase: nothing — no endpoint, no
websocket, no event. The Empty/Full toggle in step 1 stands in for it.

```
setCartPresent(present, source)          ← the only writer of state.cartFull
```

Callers: the Empty/Full toggle, *New purchase*, and a successful detection
(`source:"detection"` — finding products is itself evidence the cart is there).
Setting it false clears the detection, the photo and the payment together.

```
browser info  [iaCarry][-] cart presence changed
                {present: true, via: "detection", sensorBackend: "none (stand-in)"}
```

### Stage 2 — frame acquisition · **SIMULATED**

Real world: overhead camera. In this codebase: `DEMO_SOURCES.camera` is a
placeholder that clears the panel, because no feed endpoint exists. Frames come
from the demo selector.

```
runDemo(key) → fetchDemoBlob(src, rid)
   1. /static/assets/demo/<file>.png        ← local first, works with egress blocked
   2. raw.githubusercontent.com/…           ← fallback only
```

```
browser info  [iaCarry][r…] demo frame requested {demo:"demo1", localFirst:"/static/…"}
browser warn  [iaCarry][r…] frame source unavailable, trying the next one   ← only on fallback
browser info  [iaCarry][r…] frame loaded {from:"/static/…", bytes:783149}
```

The fallback warning matters: a silent slide to GitHub is exactly what breaks
later on a restricted network.

### Stage 3 — submit

```
submitFrame(blob, filename, seq, rid)
   ├ ++detectSeq                    token — a late response cannot overwrite a newer cart
   ├ setCamImage(blob)              objectURL; panel aspect from naturalWidth/Height
   ├ setDetect("busy","analysing")  → busy veil, pay disabled, demo selector locked
   └ fetch("/upload", POST, FormData "file", X-Request-Id, AbortController 60s)
```

The **filename matters**: `allowed_file()` checks the extension, so a blob posted
without one is refused.

```
browser info  [iaCarry][r…] posting frame to the detector
                {filename:"…png", bytes:783149, endpoint:"/upload", timeoutMs:60000}
```

### Stage 4 — Flask route

`upload_file()` — takes `X-Request-Id` from the browser or mints one.

```
Flask INFO     [r…] /upload POST from 127.0.0.1
Flask WARNING  [r…] REJECTED: no 'file' part in the form (parts=[])      ─┐
Flask WARNING  [r…] REJECTED: empty filename                             ├ used to be silent
Flask WARNING  [r…] REJECTED: extension not allowed name=x.jpeg …        ─┘
Flask INFO     [r…] frame received name=… content_type=image/png
Flask INFO     [SAVE][r…] filename sanitised: 'a (1).png' -> 'a_1.png'
Flask INFO     [SAVE][r…] wrote …/img_2026_08_07/a_1.png
Flask INFO     [r…] saved … (783149 bytes)
```

### Stage 5 — the model · **REAL**

`Detector_model.do_prediction_from_list_paths(path, rid)`

```
np.array(Image.open(path))
with inc_lock:                       ← serialises inference; concurrent requests QUEUE
    tf.convert_to_tensor(float32)[tf.newaxis, …]
    detections = self.detector(input_tensor=…)      signature "detect"
manage_plot_and_save_img_predicted(…)   ← matplotlib render to disk
register_MULTI_in_zTelegram_Registers(…) ← CSV append
tensors → numpy → drop unused keys → class ids +1 → Category_index names
```

```
model INFO     [SGTON][r…] START inference img=… shape=(640,640,3) dtype=uint8
model INFO     [SGTON][r…] lock acquired after 0.002s of queueing
model INFO     [SGTON][r…] inference done in 1.184s (lock held 1.231s)
model INFO     [SGTON][r…] figure rendered and saved in 0.402s -> …z_4_.jpg
model INFO     [SGTON][r…] appended 11 row(s) to …/df_test_img.csv
model WARNING  [SGTON][r…] no detection above MIN_SCORE=0.5 - nothing written …
model ERROR    [SGTON][r…] N detection(s) had a class id absent from Category_index …
model INFO     [SGTON][r…] RESULT raw=100 above MIN_SCORE=0.5: 11 |
                 top: cocacola_33cl=0.98, fanta_33cl=0.93, … | total 1.63s
```

**Queueing vs compute is now separable.** The lock means a second customer waits
for the first; from the browser both look like "slow", and the two lines above
tell them apart. The figure render and CSV write are pure server-side
bookkeeping — the browser draws its own boxes and never fetches `path_server`.

### Stage 6 — response shaping

`change_format_dict_json_to_client(detections, img_np_raw, path_img_box, MIN_SCORE_TO_CLIENT=0.1, rid)`

```json
{ "path_server": "..\\…\\z_4_.jpg",
  "shape_img": [720, 1280, 3],
  "request_id": "r…",
  "predictions": [ {"probability":0.98, "tagInt":1, "tagName":"cocacola_33cl",
                    "boundingBox": {"left":0.2,"top":0.1,"width":0.35,"height":0.4}} ] }
```

> ⚠ **`width` and `height` are not sizes.** TensorFlow emits
> `[ymin, xmin, ymax, xmax]`; this function maps them onto keys called
> `left/top/width/height`, so **`width` holds the right edge and `height` the
> bottom edge**, normalised 0–1. The browser subtracts. Reading them directly
> makes boxes grow steadily wronger toward the bottom-right, which looks like a
> weak model and is not.

```
Flask INFO     [SHAPE][r…] shaped response: 100 raw -> 13 sent, 87 below MIN_SCORE_TO_CLIENT=0.1
Flask INFO     [SHAPE][r…] per class sent: chipsahoy_300g x2, cocacola_33cl x5, …
Flask INFO     [SHAPE][r…] note: the browser re-filters these at its own, higher threshold …
Flask WARNING  [SHAPE][r…] NOTHING above MIN_SCORE_TO_CLIENT=0.1 - the screen will show
                 'no products recognised' and block payment
Flask INFO     [r…] /upload DONE predictions=13 shape=(720,1280,3) bytes=1940
                 detector=1.631s total=1.844s
```

**There are two thresholds.** The server keeps everything above
`MIN_SCORE_TO_CLIENT = 0.1`; the browser filters again at
`DETECTION_MIN_SCORE = 0.45`. Anything between them is sent and never shown.
That gap is invisible unless both numbers are logged together, which is what the
*note:* line is for.

### Stage 7 — adapt

`submitFrame` reads the body as **text first** — the route answers plain strings
on its error paths, so `response.json()` would hide the server's own reason.

`adaptDetections(json)`:

| Step | Rule |
|---|---|
| Threshold | drop `probability < DETECTION_MIN_SCORE` (0.45) |
| Unknown class | skip + `console.warn`; **never throw** — an exception here blanks the whole cart |
| Geometry | clamp 0–1, `w = right − left`, `h = bottom − top`; drop degenerate boxes |
| Quantity | one prediction per instance → count per `tagName` |
| Confidence | the **strongest** detection of each tag ("did we recognise this product") |
| Aspect | from `shape_img`, overridden by the image's own dimensions |

```
browser info  [iaCarry][r…] detector responded
                {status:200, bytes:1940, roundTripMs:1019, serverRequestId:"r…"}
browser info  [iaCarry][r…] detections adapted
                {predictionsFromServer:13, uiThreshold:0.45, unitsShown:11, productLines:4,
                 boxesDrawn:10, droppedBelowThresholdOrUnknown:2, skippedUnknownTags:["pringles_200g"],
                 degenerateBoxesNotDrawn:1, panelAspect:"1280 / 720",
                 totalEur:22.83, expectedWeightKg:3.75, adaptMs:0.4}
```

That single line explains any disagreement between what the model found and what
the customer sees.

### Stage 8 — render

`render()` is the only place the DOM is written; call it after any state change.

```
total   = Σ cost × qty                          → pay bar
units   = Σ qty        lines = distinct tags    → header badge, metrics
weight  = Σ weight × qty                        → facts row, badge, anti-fraud panel
boxes   = percentage-positioned divs over the photo (object-fit:fill keeps them aligned)
```

Traffic light, in priority order:

| Condition | Colour | Message |
|---|---|---|
| `pay.status==="error"` | red | payment could not be completed |
| `pay.status==="paying"` | brand | processing payment |
| `detect.status==="busy"` | brand | analysing the cart |
| `detect.status==="error"` | red | detection failed — call an assistant |
| `detect ok` + 0 units | red | no products recognised — call an assistant |
| `detect ok` + cart present + units | green | all correct, you can pay |
| otherwise | grey | place the cart to start |

`payable()` = detection ok **and** cart present **and** units > 0 **and** payment
idle. The button carries `disabled`, so a click cannot land at all.

### Stage 9 — payment and door · **SIMULATED**

```
pay() → requestPayment(order)
   PAYMENT_BACKEND === false  → 900 ms → {ok:true, simulated:true, ref:"DEMO-…"}
   PAYMENT_BACKEND === true   → POST /payment, success only if json.ok === true
```

```
browser info  [iaCarry][r…] payment requested
                {mode:"SIMULATED (no /payment route exists)", endpoint:"/payment", order:{…}}
browser info  [iaCarry][r…] payment SIMULATED — nothing was charged {ref:"DEMO-…", ms:901}
browser error [iaCarry][r…] PAYMENT FAILED: …
```

The overlay appears **only on confirmation**, never on click, and the simulated
path never uses the real wording (which promises a charge and a door opening).

---

## 4. Following one purchase through the logs

```bash
grep rmsj9wdlsuvqh browser-console.txt zlog_2026_08_07
```

```
browser  [iaCarry][rmsj9wdlsuvqh] posting frame to the detector {bytes:783149, …}
Flask    [rmsj9wdlsuvqh] /upload POST from 10.0.0.7
Flask    [SAVE][rmsj9wdlsuvqh] wrote …/img_2026_08_07/ziacarry_eval_img_1.png
model    [SGTON][rmsj9wdlsuvqh] lock acquired after 0.002s of queueing
model    [SGTON][rmsj9wdlsuvqh] inference done in 1.184s
model    [SGTON][rmsj9wdlsuvqh] RESULT raw=100 above MIN_SCORE=0.5: 11 | top: …
Flask    [SHAPE][rmsj9wdlsuvqh] shaped response: 100 raw -> 13 sent, 87 below …
Flask    [rmsj9wdlsuvqh] /upload DONE predictions=13 detector=1.631s total=1.844s
browser  [iaCarry][rmsj9wdlsuvqh] detector responded {roundTripMs:1019, serverRequestId:…}
browser  [iaCarry][rmsj9wdlsuvqh] detections adapted {unitsShown:11, totalEur:22.83, …}
```

Answerable from that alone: was it slow because of **queueing** or **inference**?
did the model find fewer products, or did the **UI threshold** drop them? is a
disagreement in the **model**, the **contract**, or the **screen**?

---

## 5. Where a number can change on its way to the customer

| Boundary | Loses | Visible in |
|---|---|---|
| `MIN_SCORE = 0.5` | rows in the CSV register | `RESULT raw=… above MIN_SCORE` |
| `MIN_SCORE_TO_CLIENT = 0.1` | predictions never sent | `shaped response: … -> … sent` |
| `DETECTION_MIN_SCORE = 0.45` | predictions sent but not shown | `detections adapted {uiThreshold…}` |
| unknown `tagName` | product silently absent | `skippedUnknownTags` + `unknown_class_*` |
| degenerate box | box not drawn, unit still counted | `degenerateBoxesNotDrawn` |
| `state.cartFull` false | the whole cart | `cart presence changed` |

---

## 6. Failure paths

| Failure | Detected | Customer sees | Payment |
|---|---|---|---|
| `/upload` unreachable / server killed | `fetch` rejects | red, call an assistant | blocked |
| Timeout (60 s) | `AbortController` | red, "timed out after 60s" | blocked |
| HTTP 500 | `!res.ok` | red | blocked |
| Plain-text error | `JSON.parse` fails → server's own text kept | red | blocked |
| Model returns nothing | 0 units after adapting | red, "no products recognised" | blocked |
| Unknown class | not in `CATALOG` | that product missing, rest correct | allowed |
| Demo image missing locally | first fetch 404 | falls back to remote, warning logged | allowed |
| Thumbnail missing | `onerror` | one retry, then empty slot | allowed |
| Logo missing | `onerror` | slot hidden, palette still themed | allowed |
| Payment declined | `json.ok !== true` | red, no overlay | blocked |

A failed detection **clears the cart**, so a stale total can never sit under a
red light.

---

## 7. The simulated seams

| Seam | Function | Turn it real |
|---|---|---|
| Cart sensor | `setCartPresent(present, source)` | Have the sensor call it. Nothing else changes; the toggle becomes an override or is deleted. |
| Camera feed | `submitFrame(blob, filename)` | Hand it a frame. The demos already use this exact path. |
| Payment / door | `requestPayment(order)` | Set `PAYMENT_BACKEND = true` and implement `POST /payment` returning `{"ok":true,"ref":"…"}`. |

Each is marked in the boot log line, so a running station states what is real.

---

## 8. What is still fabricated

- **`98% accuracy` and `1.2s inference`** in the metrics row are hardcoded
  constants, not measurements. Inference time is now measurable — the model logs
  it and the browser logs the round trip.
- **`colgate_75ml` weight `0.34 kg`** matches the 33 cl cans exactly and is
  almost certainly a copy-paste error in the original product dictionary.
- **`ALLOWED_EXTENSIONS`** contains `"jpge"`, a typo for `"jpeg"`, so a genuine
  `.jpeg` frame is refused. Harmless for the PNG demos; it will bite a camera.

---

## 9. Verifying

```bash
python3 serving/verify/verify.py     # 78 checks against a stub /upload
```

See `serving/verify/README.md`, including what it cannot establish: nothing
about the detector itself, since responses are replayed rather than inferred.
