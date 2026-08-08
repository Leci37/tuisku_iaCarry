# Verifying the checkout screen

```bash
pip install flask pillow playwright && playwright install chromium
python3 serving/verify/verify.py
```

Exits non-zero if any check fails. One run is about three minutes.

The point of this folder is that claims about the front end are **measured, not
asserted**. Box positions are compared against known ground truth in pixels;
totals against arithmetic done independently of the code under test.

## What it runs

| | Section | Checks |
|---|---|---|
| A | Detection geometry | Panel aspect and box positions on a landscape, a square and a portrait frame |
| B | Quantities, total, weight | Counts, threshold, unknown class, degenerate box, total, header badge, weight, confidence |
| C | Busy state | Veil, disabled pay, locked selector during a slow response, and clearing afterwards |
| D | Failure states | HTTP 500, plain-text error, nothing recognised, server killed mid-request |
| E | Themes and languages | 4 themes × 7 languages × 2 text sizes × 2 widths, plus 3-column density |
| F | Offline | Every non-local request aborted at the browser |
| G | Backend seams | Payment stub marked; the one-line swap to a real backend, approved and refused |

## How the geometry check works

`make_ground_truth.py` paints each detection in the recorded response as a
rectangle at its true position, then reports which of them the UI is expected to
draw — reading the product tags out of `CATALOG` in the template, so a tag added
to the catalogue cannot silently drop out of the ground truth. `verify.py`
measures the rendered overlay boxes against that list.

This is deliberately able to fail. Reintroducing the bug T2 exists to prevent —
reading `boundingBox.width`/`.height` as sizes when they hold the right and
bottom edges — moves boxes by up to 80% of the frame and fails section A loudly.

## What this CANNOT establish

**Nothing about the detector.** The TensorFlow model, its saved-model directory
and the `..\iacarry-evaluation` folder all live outside this repository, so
`stub_server.py` replays a recorded response rather than running inference.
This verifies that the front end reads the `/upload` contract correctly and
behaves correctly around it — not that the model finds the right products.

Specifically still unverified, and only testable on the deployment box:

- Real inference: accuracy, latency, and behaviour under the thread lock that
  serialises concurrent requests.
- The real `path_server` and the matplotlib render written beside it.
- Whether `shape_img` from the live model matches the frames used here.
- That the client logos are the *correct* marks. Section F confirms the three
  files decode and are served locally, not that the artwork is right — no test
  can tell a real Eroski logo from a convincing wrong one.
- `/payment` and the door release, which do not exist; section G verifies the
  seam against a stub route, not a payment terminal.

## Files

- `verify.py` — the suite. Start here.
- `stub_server.py` — stand-in for `RUN_server_py_upload.py`. Serves the real
  template, replays `sample_upload_response.json`, and can be told to be
  slow, to fail, to return nothing, or to hang so it can be killed mid-request.
- `make_ground_truth.py` — generates frames whose contents are the fixture's
  ground truth.
