# 4 — Real-time front end (the checkout screen)

The page the customer looks at. One self-contained file — markup, CSS, catalogue,
translations and logic — served by Flask at `GET /`.

`serving/templates/iacarry_checkout.html`

No build step, no framework, no bundler, no external request. It must render
completely with **outbound internet blocked**, because shop floors have
restricted egress and a grid of broken images is not a demo.

---

## Files

| File | Role |
|---|---|
| `serving/templates/iacarry_checkout.html` | **The screen.** Everything below lives here |
| `legacy/iacarry_checkout.old.html` | Superseded copy, still committed |
| `serving/static/assets/products/` | 13 product thumbnails, `<tag>_300.png` |
| `serving/static/assets/demo/` | 4 demo frames, 640×640, posted to `/upload` |
| `serving/static/assets/logos/` | 4 client logos, `<client>-logo.png`, cut to the mark on transparent, at most 160 px tall |
| `serving/static/assets/brand/iacarry-mark.png` | iaCarry's own mark (the cart in the blue→violet blob): top bar and favicon |
| `serving/static/assets/fonts/` | Ubuntu 400/500/700, latin: the face of the iaCarry wordmark, used for the whole screen. Licence in `UFL.txt` |
| `serving/static/assets/README.md` | What the assets are and where they came from |
| `serving/verify/verify.py` | 143 browser checks against a stub server |
| `serving/verify/stub_server.py` | Stand-in `/upload`: replays, delays, fails, hangs |
| `serving/verify/make_ground_truth.py` | Builds frames with known boxes, so geometry is measured not asserted |
| `legacy/iacarry_azure.html` (root) | The older Azure-era front end, superseded |

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

Five themes (`iacarry`, `eroski`, `ahorramas`, `condis`, `mercadona`) × seven languages
(es, en, eu, ca, pt, fr, de) × two text sizes × three densities. To add a client,
see [Adding a client](#adding-a-client) below.

Each theme is a palette plus a logo path:

```js
mercadona:{name:"Mercadona",pri:"#009660",ink:"#007B4F",ps:"#EBF7F2",…,logo:"/static/assets/logos/mercadona-logo.png"}
```

| Key | Used for | Rule |
|---|---|---|
| `pri` | Fills with large white text: the pay button, the avatar, the active step, the dropdown border | The brand colour itself. White on it at least 3:1 |
| `ink` | Small text on the tints (badges, quantity pills, dropdown text and arrow, busy message) and the fill of the selected category chip | `pri` darkened until it reaches 4.5:1 on its strongest tint. Equal to `pri` when `pri` is already dark (Condis) |
| `ps` `pt` `bg` `mt` `seg` `sa` `sb` | Soft backgrounds | `pri` mixed toward white at 4–13 % |
| `grad` | The pay button and active step, iaCarry only | Its logo's blue → violet. Retailers fill with flat `pri` |

`applyTheme()` sets `--pri`, `--pri-ink`, `--pri-fill` (`grad` or `pri`), the tints
and `--chev` (the select arrow, which an SVG data URL cannot read from a CSS
variable) on `document.body`. The `iacarry` theme has `logo:""` — the own brand
shows no client mark, which is correct.

### Two brands on one screen

The screen is iaCarry's product, running for a retailer, and each brand keeps
its own:

- **The iaCarry mark never takes the retailer's colour.** The icon (the cart in
  the blob) and the wordmark, in Ubuntu like the logo, sit top left in their own
  colours under every theme. iaCarry's palette is its logo's: blue `#304CB5` →
  violet `#6C5DC7` → orchid `#DB7DE8`. Its own theme takes `#6C5DC7` as `pri` and
  the gradient for the pay button. Before, the "Carry" half of the wordmark
  turned Mercadona green or Eroski red.
- **Everything else takes the retailer's colour.** Neutrals — greys, borders,
  the disabled pay button, the idle traffic light — are the same under every
  theme. They used to carry iaCarry's violet, which showed as a lilac cast, a
  lilac arrow on every dropdown and a lilac pay button on a Mercadona screen.
  Section I of the verification suite looks for any violet left on a retailer's
  screen.
- **The retailer's logo appears twice**: in the top bar, as "iaCarry | for
  MERCADONA", and by the total, so the customer sees who they are paying. Neither
  has a frame.

### How logos are sized

Logos come in every shape: Mercadona's and AhorraMas's are long wordmarks,
Condis's is a stacked mark. Fitted into one box, as before, the long ones came
out a thin strip a size smaller than the rest, inside a frame that was mostly
empty. `fitLogo()` gives each logo the same **area** instead, within a maximum
box, so they read the same size:

| Slot | Area | Maximum | Eroski | AhorraMas | Condis | Mercadona |
|---|---|---|---|---|---|---|
| Top bar (`#logoBig`) | 4 400 px² | 170 × 40 | 140 × 31 | 170 × 26 | 77 × 40 | 170 × 25 |
| By the total (`#logoSmall`) | 1 900 px² | 124 × 30 | 92 × 21 | 112 × 17 | 58 × 30 | 112 × 17 |

This only works if the file holds the mark and nothing else: a white margin
counts as logo and shrinks that one. The logo files are cut to the mark on
transparent, and `tools/new_client_theme.py` cuts new ones the same way.

The logo by the total is pinned right with `margin-left:auto`; with
`space-between` alone it jumped to the left whenever the "AI-recognised" pill
beside it was hidden, which is every empty cart.

### Brand colours, checked against the logos

| Retailer | `pri` | Taken from | White on `pri` | `ink` |
|---|---|---|---|---|
| iaCarry | `#6C5DC7` | its logo's gradient | 5.2:1 | `#6A5BC3` |
| Eroski | `#E30613` | Eroski red; the logo file reads `#E42219` | 4.9:1 | `#C40511` |
| AhorraMas | `#C53842` | the logo's red wordmark | 5.2:1 | `#BF3640` |
| Condis | `#17398A` | the logo's blue wordmark (`#1C3E95`) | 10.6:1 | `#17398A` |
| Mercadona | `#009660` | the logo's green, exactly | 3.8:1 | `#007B4F` |

AhorraMas was green (`#57A639`, the underline of its logo) until 2026-10: its
buttons did not match its red logo, and white on that green was 3.0:1. Its web
shop prices in a crimson, which agrees with the red.

A logo that fails to load is recorded **once** in `LOGO_MISSING` and its slot is
hidden, so a missing file costs the mark but never the theme, and never leaves a
broken-image icon in the top bar. Recording the miss before re-rendering is what
stops the loop: the next `applyTheme()` sees the entry and does not re-assign
`src`.

Thumbnails have the same discipline — `thumbFallback()` retries once, then leaves
the slot empty.

### Adding a client

A client (retailer) is three things: an entry in the dropdown, a palette in
`THEMES`, and a logo file. `tools/new_client_theme.py` makes all three:

```bash
python tools/new_client_theme.py mercadona "Mercadona" path/to/logo.png --color "#009660"
python3 serving/verify/verify.py
```

| Argument | What to give it |
|---|---|
| key (`mercadona`) | Lowercase letters and digits. It becomes the option value, the `THEMES` key and the logo file name `<key>-logo.png`, which must all match |
| name (`"Mercadona"`) | What the dropdown shows |
| logo | The mark, ideally on a transparent or white background. The script makes the white around it transparent (white inside the mark stays), crops it to the mark and keeps it at most 160 px tall. The page sizes it, so any shape works: long wordmark or stacked |
| `--color` | The brand's primary colour. **Take it from the brand manual when there is one.** Without it, the script picks the most common saturated colour in the logo and prints it, which can be the wrong one (for Mercadona's round icon it picks the orange basket, not the green) |
| `--logo-only` | Only remake the logo file, for a client already in the template, e.g. when the retailer sends a better file |
| `--dry-run` | Print the two lines and the file it would write, and change nothing |

What it derives from the one colour: `pri` is the colour itself; `ink` is it
darkened until small text reaches 4.5:1 on the strongest tint; the seven tints
(`ps`, `pt`, `bg`, `mt`, `seg`, `sa`, `sb`) are it mixed toward white at 4–13 %,
the same strength as the existing themes. See the table in
[Themes and languages](#themes-and-languages) for what each one paints. The
script warns when white text on `pri` falls below 3:1 contrast; use a darker
shade of the brand colour when it does.

Checking the result needs no test edits. Sections E, F and I of the
verification suite read the client list from the dropdown, so the new client is
checked for clipping in every language and text size, its logo for loading
offline and for its size and place, its colours for contrast, and its screen for
any violet left from iaCarry's palette. Then
look at it once in the stub (`python serving/verify/stub_server.py .` from
`serving/`) and pick it in the dropdown.

Two things the script does not do:

- **The trademark note.** Add the client's name to the list in
  `serving/static/assets/README.md` and the root `README.md`: client logos are
  for the demo only.
- **Fine-tuning a tint by hand.** Edit its hex in `THEMES` directly; nothing else
  reads it.

---

## Offline by construction

```js
const ASSETS   = "/static/assets/";
const PROD_IMG = ASSETS + "products/";
const LOGO_IMG = ASSETS + "logos/";
```

Everything the screen draws comes from the app, including the Ubuntu font
(`assets/fonts/`, declared with `@font-face`) and the iaCarry mark, which is also
the favicon. `RAW` (raw.githubusercontent.com)
survives only as the demo-frame fallback, and taking it warns. Section F of the
verification suite proves the point the hard way: it aborts every request whose
URL does not start with the local base, and asserts **no outbound request was
even attempted**.

---

## Verification

```bash
python3 serving/verify/verify.py     # 143 checks, 0 failed
```

Requires `flask`, `pillow`, `playwright` and a Chromium build. It drives the real
template in a real browser against `stub_server.py`.

| Section | Covers |
|---|---|
| A | **Detection boxes sit on the products** — panel aspect, box count, every box within 1% of ground truth |
| B | **Quantities, total and weight** — counts, sub-threshold dropped, unknown class skipped not thrown, degenerate box counted but not drawn, totals against hand arithmetic |
| C | **Busy state during a slow response** — veil visible, pay and demo selector locked in flight, veil clears on completion |
| D | **Failure states** — unreachable, timeout, HTTP 500, plain-text error, nothing found, server killed mid-request |
| E | **every theme × 7 languages × 2 text sizes × 3 densities**, checked for clipping |
| F | **Renders completely with outbound traffic blocked** — thumbnails, logos, the iaCarry mark and the Ubuntu font load locally, no outbound request attempted, nothing answers 404 |
| G | **Backend seams are stubs and say so** — payment approved / declined / gateway-500 |
| H | **Each demo frame gets its own hand-labelled answer** — one box per item, every label a catalogue product |
| I | **Each retailer: logo placed and sized, its own colours, readable text** — logo files cut to the mark; both logos inside their box, proportions kept, the same area within 1.6×, the small one pinned right even on an empty cart; the iaCarry mark unthemed; no violet left on a retailer's screen; the dropdown arrow in the retailer's colour; white on `pri` ≥ 3:1 and `ink` ≥ 4.5:1 for every theme; empty-cart message centred and pay button grey |

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
8. **Two superseded front ends are still in the tree**, now parked in `legacy/`
   (`iacarry_checkout.old.html`, `iacarry_azure.html`) with `legacy/README.md`
   saying which is live. Kept, not deleted — but neither is maintained.
9. **Everything is in one file.** It is a deliberate and defensible choice — it
   is what makes the page dependency-free and lets Flask serve it with no build
   step — but at 1 010 lines of markup, CSS, catalogue, logic and seven languages
   of copy, splitting the translations out is the first thing worth doing if it
   grows again.
10. **No portrait layout of its own.** The page is laid out for a landscape
    screen. The vertical 1080×1920 signage screens in `presentation/sora/` are
    made by injecting extra CSS (`PORTRAIT_CSS` in `make_station_screens.py`);
    a station with a vertical screen needs that as a real media query here.
