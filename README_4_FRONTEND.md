# 4 — Real-time front end (the checkout screen)

The page the customer looks at. One self-contained file — markup, CSS, catalogue,
translations and logic — served by Flask at `GET /`.

`server/iaCarry_Local_JS_1_Clouding.html`

No build step, no framework, no bundler, no external request. It must render
completely with **outbound internet blocked**, because shop floors have
restricted egress and a grid of broken images is not a demo.

---

## Files

| File | Role |
|---|---|
| `server/iaCarry_Local_JS_1_Clouding.html` | **The screen.** Everything below lives here |
| `server/iaCarry_Local_JS_1_Clouding.old.html` | Superseded copy, still committed |
| `server/static/assets/products/` | 13 product thumbnails, `<tag>_300.png` |
| `server/static/assets/demo/` | 4 demo frames, 640×640, posted to `/upload` |
| `server/static/assets/logos/` | 3 client logos, `<client>-logo.png`, 600×180 |
| `server/static/assets/README.md` | What the assets are and where they came from |
| `server/zz_verify/verify.py` | 78 browser checks against a stub server |
| `server/zz_verify/stub_server.py` | Stand-in `/upload`: replays, delays, fails, hangs |
| `server/zz_verify/make_ground_truth.py` | Builds frames with known boxes, so geometry is measured not asserted |
| `iaCarry_azure_JS_1.html` (root) | The older Azure-era front end, superseded |

---

## The state machine

One `state` object; `render()` is the **only** function that writes the DOM. Call
it after any state change and the screen is correct by construction.

```
state = { cartFull, detect:{status, phase, detail}, pay:{status, …},
          live, lang, client, demo, cat, density, big }
```

`detect.status`: `idle → busy → ok | error`
`pay.status`: `idle → paying → done | error`

---

## Stage by stage

### Page load

Builds `CATALOG`, `I18N`, `DEMO_SOURCES`, `THEMES`, then `render()` + `fillDemo()`.
Language is auto-detected (`detectLang()`, Galician folded to Spanish, otherwise
one of es/en/eu/ca/pt/fr/de, defaulting to English).

The boot line states what is real and what is not:

```
[iaCarry] ready — detection:/upload | payment:SIMULATED (no route)
          | cart sensor:manual stand-in toggle | camera feed:NONE (demo selector only)
```

### Cart presence · **SIMULATED**

```js
const CART_SENSOR_BACKEND = false;
setCartPresent(present, source)      // the only writer of state.cartFull
```

Real world: a sensor on the station. Here: nothing — no endpoint, no websocket.
The Empty/Full toggle stands in, and carries `data-stub="cart-sensor"` in the DOM
so the fact is inspectable rather than folklore.

Callers: the toggle, *New purchase*, and **a successful detection** — finding
products is itself evidence the cart is there. Setting it false clears the
detection, the photo and the payment together, so no stale total can survive.

### Frame acquisition · **SIMULATED**

No camera feed exists; `DEMO_SOURCES.camera` is a placeholder that clears the
panel. Frames come from the demo selector.

```
fetchDemoBlob(src, rid)
  1. /static/assets/demo/<file>.png      ← local first, works with egress blocked
  2. raw.githubusercontent.com/…         ← fallback only, logs a warning
```

The warning on fallback matters: **a silent slide to GitHub is exactly what
breaks later on a restricted network.**

### Submit

```js
submitFrame(blob, filename, seq, rid)
  ++detectSeq                      // a late response cannot overwrite a newer cart
  setCamImage(blob)                // objectURL; panel aspect from naturalWidth/Height
  setDetect("busy","analysing")    // veil on, pay disabled, demo selector locked
  fetch("/upload", POST, FormData "file", X-Request-Id, AbortController 60s)
```

`detectSeq` is the guard against out-of-order responses. `DETECTION_TIMEOUT_MS`
is 60 000.

**The filename matters**: the server checks the extension, so a blob posted
without one is refused (`README_3_SERVER.md`).

The body is read as **text first**, then parsed — the route answers plain strings
on its error paths, and `response.json()` would throw away the server's own
reason for refusing.

### Adapt — `adaptDetections(json)`

| Step | Rule |
|---|---|
| Threshold | drop `probability < DETECTION_MIN_SCORE` (**0.45**) |
| Unknown class | skip + `console.warn`; **never throw** — an exception here blanks the whole cart |
| Geometry | `boxFromPrediction()`: clamp 0–1, `w = right − left`, `h = bottom − top`, drop degenerate boxes |
| Quantity | one prediction per instance → count per `tagName` |
| Confidence | the **strongest** detection of each tag — "did we recognise this product" |
| Aspect | from `shape_img`, overridden by the image's own dimensions |

The geometry line is where the server's misnamed `width`/`height` keys are
undone. See the warning in `README_3_SERVER.md` — **the subtraction is not
optional.**

One log line explains any disagreement between model and screen:

```
detections adapted {predictionsFromServer:13, uiThreshold:0.45, unitsShown:11,
  productLines:4, boxesDrawn:10, droppedBelowThresholdOrUnknown:2,
  skippedUnknownTags:["pringles_200g"], degenerateBoxesNotDrawn:1,
  panelAspect:"1280 / 720", totalEur:22.83, expectedWeightKg:3.75}
```

### Render

```
total  = Σ cost × qty                    → pay bar
units  = Σ qty     lines = distinct tags → header badge, metrics
weight = Σ weight × qty                  → facts row, anti-fraud panel
boxes  = percentage-positioned divs over the photo
```

Boxes are positioned in percentages and the photo uses `object-fit: fill`, which
is what keeps them aligned at any panel size.

Traffic light, in priority order:

| Condition | Colour | Message |
|---|---|---|
| `pay.status === "error"` | red | payment could not be completed |
| `pay.status === "paying"` | brand | processing payment |
| `detect.status === "busy"` | brand | analysing the cart |
| `detect.status === "error"` | red | detection failed — call an assistant |
| ok + 0 units | red | no products recognised — call an assistant |
| ok + cart present + units | green | all correct, you can pay |
| otherwise | grey | place the cart to start |

`payable()` = detection ok **and** cart present **and** units > 0 **and** payment
idle. The button carries `disabled`, so a click cannot land at all — the check is
not merely visual.

### Payment and door · **SIMULATED**

```js
const PAYMENT_ENDPOINT = "/payment";   // NOT IMPLEMENTED SERVER-SIDE
const PAYMENT_BACKEND  = false;        // the one line that turns this real
const PAYMENT_TIMEOUT_MS = 30000;

requestPayment(order)
  false → 900 ms → {ok:true, simulated:true, ref:"DEMO-…"}
  true  → POST /payment, success only if json.ok === true
```

The success overlay appears **only on confirmation**, never on click. The
simulated path never uses the real wording — that copy promises a charge and a
door opening, and a demo must not.

---

## Themes and languages

Four themes (`iacarry`, `eroski`, `ahorramas`, `condis`) × seven languages
(es, en, eu, ca, pt, fr, de) × two text sizes × three densities.

Each theme is a palette plus a logo path:

```js
eroski: {name:"Eroski", pri:"#E30613", …, logo:"/static/assets/logos/eroski-logo.png"}
```

`applyTheme()` sets CSS custom properties on `document.body`. The `iacarry` theme
has `logo:""` — the own brand shows no client mark, which is correct.

A logo that fails to load is recorded **once** in `LOGO_MISSING` and its slot is
hidden, so a missing file costs the mark but never the theme, and never leaves a
broken-image icon in the top bar. Recording the miss before re-rendering is what
stops the loop: the next `applyTheme()` sees the entry and does not re-assign
`src`.

Thumbnails have the same discipline — `thumbFallback()` retries once, then leaves
the slot empty.

---

## Offline by construction

```js
const ASSETS   = "/static/assets/";
const PROD_IMG = ASSETS + "products/";
const LOGO_IMG = ASSETS + "logos/";
```

Everything the screen draws comes from the app. `RAW` (raw.githubusercontent.com)
survives only as the demo-frame fallback, and taking it warns. Section F of the
verification suite proves the point the hard way: it aborts every request whose
URL does not start with the local base, and asserts **no outbound request was
even attempted**.

---

## Verification

```bash
python3 server/zz_verify/verify.py     # 78 checks, 0 failed
```

Requires `flask`, `pillow`, `playwright` and a Chromium build. It drives the real
template in a real browser against `stub_server.py`.

| Section | Covers |
|---|---|
| A | **Detection boxes sit on the products** — panel aspect, box count, every box within 1% of ground truth |
| B | **Quantities, total and weight** — counts, sub-threshold dropped, unknown class skipped not thrown, degenerate box counted but not drawn, totals against hand arithmetic |
| C | **Busy state during a slow response** — veil visible, pay and demo selector locked in flight, veil clears on completion |
| D | **Failure states** — unreachable, timeout, HTTP 500, plain-text error, nothing found, server killed mid-request |
| E | **4 themes × 7 languages × 2 text sizes × 3 densities**, checked for clipping |
| F | **Renders completely with outbound traffic blocked** — thumbnails and logos load locally, no outbound request attempted |
| G | **Backend seams are stubs and say so** — payment approved / declined / gateway-500 |

`make_ground_truth.py` is the reason section A is worth anything: it builds
frames with known box positions, so the suite **measures** what the page draws
instead of trusting it — hence "every box within 1% of ground truth" rather than
"boxes appear".

**What it cannot establish**: anything about the detector — responses are
replayed, not inferred. Also, that the logo artwork is *correct*; it confirms the
files decode and are served locally, but no test can tell a real Eroski mark from
a convincing wrong one.

---

## The simulated seams — and how to make each real

| Seam | Function | To wire up |
|---|---|---|
| Cart sensor | `setCartPresent(present, source)` | Have the sensor call it. Nothing else changes; the toggle becomes an override, or is deleted |
| Camera feed | `submitFrame(blob, filename)` | Hand it a frame. The demo selector already uses this exact path |
| Payment / door | `requestPayment(order)` | Set `PAYMENT_BACKEND = true` and implement `POST /payment` returning `{"ok":true,"ref":"…"}` |

Each is named in the boot log, so a running station states what is real.

---

## What is missing from the front end

1. **Three of the four inputs are simulated** — cart sensor, camera feed,
   payment/door. Only detection is real. The seams are clean and named, which is
   the right state for a demo, but nothing is wired.
2. **`98% accuracy` and `1.2s inference` in the metrics row are hardcoded
   constants**, not measurements (`FLOW.md` §8). Inference time is now genuinely
   measurable — the model logs it and the browser logs the round trip — so this
   is fixable today, and until it is, the screen is telling the customer
   something untrue.
3. **`colgate_75ml` weighs `0.34 kg` in `CATALOG`**, identical to the 33 cl cans
   — almost certainly a copy-paste error carried from the original product
   dictionary. It feeds the anti-fraud weight total, so it is wrong in a place
   that is supposed to catch fraud.
4. **The anti-fraud panel reads "Awaiting scale" permanently.** There is no scale
   input, so the expected weight is computed and never compared against anything.
   The feature is presented as present and is inert.
5. **No accessibility pass.** No ARIA roles, no keyboard path through the flow,
   no focus management on the overlay — on a public self-service terminal, that
   matters.
6. **Prices and weights are hardcoded in `CATALOG`.** No POS or SKU integration,
   so a price change means editing the HTML.
7. **The catalogue is 13 products** while the search box advertises "+10.000".
8. **`iaCarry_Local_JS_1_Clouding.old.html` and `iaCarry_azure_JS_1.html` are
   still committed**, with nothing marking which file is live.
9. **Everything is in one file.** It is a deliberate and defensible choice — it
   is what makes the page dependency-free and lets Flask serve it with no build
   step — but at 931 lines of markup, CSS, catalogue, logic and seven languages
   of copy, splitting the translations out is the first thing worth doing if it
   grows again.
