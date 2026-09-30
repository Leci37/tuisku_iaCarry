# Presentation

The iaCarry funding pitch deck, kept in three forms so it can be read without
PowerPoint.

| File | What it is |
|---|---|
| `iaCarry_pitch_jose_elias.pptx` | The deck itself, unmodified (17 slides, January 2024). This is the one to present |
| `slides.md` | All of its text, one section per slide, readable on GitHub |
| `images/` | Ten images: the logo, the station for large and small purchases, and the current checkout screen per retailer. Older ones are in `images/old/` |
| `make_screens.py` | Regenerates the checkout screenshots |

`slides.md` ends with an appendix taken from an earlier deck
(`Pitch_deck_iaCarry_iaRecycle_05_1.pptx`, March 2024), which is not kept here.
That deck had what this one lacks: the demo screen and the seven "How does it
work?" steps. Its text and images are the only parts worth keeping.

Figures differ between the two decks. This deck says a 900 € licence and 45 €
maintenance per station per month, the earlier one 420 € and 165 €. Check which
is current before quoting either.

## images/

Ten images: the brand, the station, and the current checkout screen. Everything
that shows the old interface or is no longer used is in `images/old/`.

| File | Shows | From |
|---|---|---|
| `logo_iacarry_full.png` | The iaCarry logo with its tagline, "payment in just 6 seconds!" | own logo |
| `station_trolley.jpg` | The weighing station for **large purchases**: a trolley on the scale, overhead camera, screen, card terminal, exit gate | this deck, slide 2 |
| `station_basket.jpg` | The weighing station for **small purchases**: a basket on the scale, overhead camera, card terminal | this deck, slide 1 |
| `checkout_screen_2026.jpg` | The current checkout screen on demo photo 3, iaCarry theme | screenshot |
| `screen_1…6_*.jpg` | The current checkout screen per retailer, see below | screenshots |

### Checkout screen, one shot per retailer

Screenshots of the current screen, not from a deck. Regenerate them with
`python3 presentation/make_screens.py` (edit `SCENES` in it to change a shot).
The boxes come from the hand-labelled demo photos, not from the model.

| File | Retailer | Shows |
|---|---|---|
| `screen_1_mercadona_full.jpg` | Mercadona | Full cart, boxes on photo 1, 3-column view, Spanish |
| `screen_2_eroski_confidence.jpg` | Eroski | Full cart, "AI confidence" on, list view, English |
| `screen_3_condis_largetext.jpg` | Condis | Full cart, large text, grid view, anti-fraud panel open, French |
| `screen_4_ahorramas_paid.jpg` | AhorraMas | Full cart after paying: the "thank you" screen, simulated payment, Spanish |
| `screen_5_mercadona_empty_large.jpg` | Mercadona | Empty cart, large text, pay button disabled, Spanish |
| `screen_6_iacarry_empty.jpg` | iaCarry (own brand) | Empty cart, no retailer logo, English |

They show retailer logos, which are for the demo only (see the trademark note in
`serving/static/assets/README.md`).

### images/old/

Kept for the deck and for reference, no longer the current picture. `slides.md`
still shows them under the slides that use them.

| File | Shows | From |
|---|---|---|
| `demo_ui_full.jpg`, `demo_ui_detection.jpg`, `demo_ui_product_list.png` | The previous demo screen | earlier deck, slides 4 and 5 |
| `how_it_works_steps_1-4.jpg`, `how_it_works_steps_5-7.jpg` | Trolleys before and after recognition | earlier deck, slides 8 and 9 |
| `detection_boxes_clothing_furniture.jpg` | Box detection for clothing and furniture shops | this deck, slide 13 |
| `detection_iarecycle_waste.jpg` | iaRecycle detecting waste items | this deck, slide 14 |
| `chart_supermarket_size_spain_2021.png` | Supermarkets in Spain by size, 2021 | this deck, slide 9 |
| `station_camera_arm.png` | The station's overhead camera arm | this deck, slide 3 |
| `logo_iacarry.png`, `logo_tuisku.png`, `logo_iarecycle.png` | Icon-only logos | both decks |

Large PNGs from the decks were saved as JPEG (at most 1600 px on the long side)
to keep the folder light. The originals are still inside the `.pptx`.

**Left out on purpose:** stock photos, competitors' product shots, other
companies' logos (Tesco, Amazon Go, Carrefour, Tracxpoint, GitHub) and the
trolley dimension drawing. They stay inside the `.pptx` and are not extracted,
because they are not ours to reuse.
