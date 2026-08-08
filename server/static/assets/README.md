# Static assets for the checkout screen

Served by Flask at `/static/assets/…`. Flask's default static folder is
`server/static`, rooted at the directory holding `RUN_server_py_upload.py`, so
these are reachable with no Python configuration — the directory simply did not
exist before.

The point of this folder is that **the checkout screen renders completely with
outbound internet blocked**. A shop floor with restricted egress must not show a
grid of broken images.

## Contents

| Folder | Files | Used for |
|---|---|---|
| `products/` | `<tag>_300.png` × 13 | Product thumbnails in the cart grid |
| `demo/` | `ziacarry_eval_img_1…4.png` | Demo frames posted to `/upload` |
| `logos/` | `<client>-logo.png` × 3 | Client mark in the top bar and by the total |

`products/` and `demo/` were copied from the `readme_img/` folder of
`Leci37/stocks-prediction-Machine-learning-RealTime-TensorFlow`, which is where
the previous interface loaded them from over the public internet. They are the
same bytes, now version-controlled and served locally.

The demo frames are the 640×640 evaluation images. They are posted to `/upload`
unmodified — do not resize or re-encode them, or detection results will drift
away from the recorded evaluation.

## Missing: the three client logos

`logos/` is **empty**. The Eroski, AhorraMas and Condis logos were supplied as
images rendered in conversation rather than as files, so there were no bytes to
commit. The design specifies them as 600×180, white background, 10:3.

Nothing is broken by their absence. A logo that fails to load is recorded once
and its slot is hidden, so selecting a client theme swaps the whole palette and
simply shows no mark. Dropping the three PNGs in here, named as above, is all
that is needed — no code change.

## Trademark

The Eroski, AhorraMas and Condis marks are **registered trademarks of their
respective owners**. The design documentation is explicit that they are licensed
for demo use with each retailer's permission only. They are committed here for
the demo build; they are not covered by this repository's licence, and they must
not be reused for any other purpose or by any other party.
