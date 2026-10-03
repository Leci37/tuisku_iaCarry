# Static assets for the checkout screen

Served by Flask at `/static/assets/…`. Flask's default static folder is
`static/`, rooted at the directory holding `app.py` — that is, `serving/static`
— so these are reachable with no Python configuration.

The point of this folder is that **the checkout screen renders completely with
outbound internet blocked**. A shop floor with restricted egress must not show a
grid of broken images.

## Contents

| Folder | Files | Used for |
|---|---|---|
| `products/` | `<tag>_300.png` × 13 | Product thumbnails in the cart grid |
| `demo/` | `ziacarry_eval_img_1…4.png` | Demo frames posted to `/upload` |
| `logos/` | `<client>-logo.png` × 4 | Client mark in the top bar and by the total |
| `brand/` | `iacarry-mark.png` | iaCarry's own mark, top left, and the favicon |
| `fonts/` | `Ubuntu-{400,500,700}-latin.woff2`, `UFL.txt` | The face of the iaCarry wordmark, used for the whole screen |

`products/` and `demo/` were copied from the `readme_img/` folder of
`Leci37/stocks-prediction-Machine-learning-RealTime-TensorFlow`, which is where
the previous interface loaded them from over the public internet. They are the
same bytes, now version-controlled and served locally.

The demo frames are the 640×640 evaluation images. They are posted to `/upload`
unmodified — do not resize or re-encode them, or detection results will drift
away from the recorded evaluation.

## The client logos

`logos/` holds the Eroski, AhorraMas, Condis and Mercadona marks, each cut to
the mark on a transparent background (white inside the mark is kept) and at most
160 px tall. They have no common size on purpose: the page gives every logo the
same area (`fitLogo()` in the template), so a long wordmark and a stacked mark
read the same size, and a margin left in a file would shrink that logo. Section
I of `serving/verify/verify.py` checks that no file has one. The filenames are what the page
requests — `LOGO_IMG` in `iacarry_checkout.html` builds
`/static/assets/logos/<client>-logo.png` from the theme key, so a logo added for
a new client must be named for that key or it will not be found.
`tools/new_client_theme.py` produces the file with the right name and cut; see
"Adding a client" in `README_4_FRONTEND.md`.

The failure path still exists and is worth knowing: a logo that fails to load is
recorded once and its slot is hidden, so a missing file costs the mark but not
the theme — the palette still swaps and nothing else breaks. That is why a wrong
path shows up as "no logo" rather than as an error.

## Trademark

The Eroski, AhorraMas, Condis and Mercadona marks are **registered trademarks of their
respective owners**. The design documentation is explicit that they are licensed
for demo use with each retailer's permission only. They are committed here for
the demo build; they are not covered by this repository's licence, and they must
not be reused for any other purpose or by any other party.

## The iaCarry mark and the font

`brand/iacarry-mark.png` is the icon of the iaCarry logo, cut from
`presentation/images/old/logo_iacarry.png` and scaled to 160 px. It is our own.

`fonts/` holds Ubuntu (regular, medium, bold; latin subset, which covers all
seven languages), the typeface of the iaCarry wordmark. It is served from here
rather than from Google Fonts so the screen renders the same offline. The same
files ship with `zlecitool-core`. Licence: Ubuntu Font Licence 1.0, in
`fonts/UFL.txt`.
