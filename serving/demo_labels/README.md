# Demo labels

Hand-labelled answers for the four demo photos in
`serving/static/assets/demo/`, one JSON file per photo, in exactly the shape
`/upload` returns.

| File | Items |
|---|---|
| `ziacarry_eval_img_1.json` | 12 |
| `ziacarry_eval_img_2.json` | 9 |
| `ziacarry_eval_img_3.json` | 13 |
| `ziacarry_eval_img_4.json` | 13 |

## Why they exist

No annotations for these photos were ever committed, and the model cannot run
outside the deployment box. The stub server used to answer every photo with
`sample_upload_response.json`, a test fixture whose boxes belong to no real
photo, so the demo drew boxes over the wrong products. `stub_server.py` now
looks here first, by the filename the page posts
(`ziacarry_eval_img_N.png` → `ziacarry_eval_img_N.json`), and falls back to the
fixture only for other files. `LABELS=off` turns this off; the verification
suite does that for every section except H.

## What they are and are not

- **Ground truth drawn by hand, not model output.** Every `probability` is
  `1.0` for that reason. The confidence shown with "AI confidence" on is
  therefore 100 % for every item, and says nothing about the model.
- Boxes are axis-aligned and cover each visible product, within a few percent.
- `boundingBox.width` and `.height` hold the **right and bottom edges**, not
  sizes, exactly as the real server sends them (see `serving/server_utils.py`).

To replace them with what the model really sees, run the real server on the
deployment box, post each photo to `/upload`, and save each response over the
matching file here.
