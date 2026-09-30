# Station renders for Sora

Two images of the iaCarry weighing station for the presentation: the **basket**
station for small purchases and the **trolley** station for large ones.

| File | What it is |
|---|---|
| `draft_v1_basket.webp`, `draft_v1_trolley.webp` | First Sora results, reviewed below |
| `screen_basket_portrait.png`, `screen_trolley_portrait.png` | **What the vertical screen must show.** The real checkout interface, large text, laid out for a 1080x1920 signage screen, with exactly the products in each draft. Upload it with the prompt, or paste it onto the screen afterwards |
| `screen_basket.png`, `screen_trolley.png` | The same screen in its normal landscape layout |
| `frame_basket.png`, `frame_trolley.png` | The top-down "camera" photo shown inside each screen, composed from the catalogue images and piled with some overlap, as in a real cart |
| `make_station_screens.py` | Regenerates the frames and screens (`python3 presentation/sora/make_station_screens.py`) |
| `../images/logo_iacarry_full.png` | The logo to upload with each prompt |

Totals, from the catalogue prices: basket **10.98 €** (7 items), trolley
**28.09 €** (14 items). The 4.83 € and 27.29 € in the drafts were wrong.

## Review of the drafts

**Basket (`draft_v1_basket.webp`)**

- The screen is not the iaCarry interface: a product list with coloured dots and
  no names or prices. The total, 4.83 €, is wrong for what is in the basket.
- The screen leans out over the basket, between the camera and the products. A
  real camera there would film the back of the screen.
- The camera hangs at the top left, not centred over the basket.
- The logo appears only on the sign. It is not on the camera or the scale.
- Correct and worth keeping: the green LED edge on the scale, the green exit
  arrow, the card terminal, the store background, the products.

**Trolley (`draft_v1_trolley.webp`)**

- The screen is not the iaCarry interface, and its camera view shows a basket,
  not a trolley. The total, 27.29 €, is wrong: there are two Coca-Colas, so it
  is 28.09 €.
- The screen is portrait, which is kept on purpose (a vertical signage screen);
  `screen_trolley_portrait.png` is the interface laid out for it.
- The camera arm is short: the camera sits near the pillar, over the back of the
  trolley, not over its centre.
- No logo on the camera, the platform or the floor.
- Correct and worth keeping: the platform with its green LED strip, the exit
  gate with the green arrow, the card terminal, the busy store behind, the
  products.

## How to use the prompts

Upload three images with each prompt, in this order:

1. the draft (`draft_v1_*.webp`), the layout to keep
2. the vertical screen (`screen_*_portrait.png`), what the display must show
3. the logo (`../images/logo_iacarry_full.png`)

If the screen still comes out garbled, ask for a "blank white portrait screen"
instead and paste `screen_*_portrait.png` onto it afterwards. That is the most
reliable way to get a sharp interface.

What changes from the drafts, in both: a slightly more schematic look (real
materials, but a calmer background and every part clearly separated), the
camera and the screen made unmistakable, the screen vertical like a digital
advertising totem, and the logo on the camera, the scale and the floor. Only
the trolley station gets an "enter / do not enter" signal: a basket is placed
by hand and needs none.

## Trolley station

```
Edit the first attached image, a photo of the iaCarry self-checkout station for shopping trolleys in a Spanish supermarket. Keep its composition and camera angle: trolley on a low floor scale on the left, a totem in the middle, a card terminal, an exit gate on the right, an overhead camera above. Landscape 16:9.

Style: a clean, slightly schematic product visualisation. Materials, lighting and scale stay photorealistic, but simplify: calm, desaturated, softly blurred store background with few people; soft even light; every component clearly separated and easy to identify at a glance. No clutter.

1. Screen: replace the display with a tall vertical digital signage screen, like the 55-inch portrait advertising totems in shopping centres: slim black bezel, freestanding on its own base next to the scale, facing the customer. It shows exactly the second attached image, the iaCarry checkout interface: the top-down camera view of the trolley with coloured outlines and name labels, the product cards, the total "28.09€" and a large purple "Contactless pay" button. Reproduce it faithfully and sharply. No other text on the screen.

2. Camera: make it unmistakable. A compact white camera module at the end of a long anthracite arm from the top of the totem, hanging directly over the centre of the trolley and pointing straight down. A faint translucent light-blue cone runs from the lens down to the trolley to show what it sees. A small round iaCarry logo sticker (the purple-to-pink gradient blob with the white cart and leaf from the third attached image) on the side of the camera housing.

3. Scale: a low brushed-stainless floor platform with an anthracite frame and a short entry ramp, slightly larger than the trolley, with a green LED strip along its edge showing the weight has been read. The iaCarry logo from the third attached image printed flat on the platform surface beside the trolley wheels.

4. Enter / do not enter signal: at the start of the entry ramp, a slim post about 1.2 m high with a round two-light signal facing the approaching customer: a red "X" light on top (off) and a green arrow light below (lit, meaning the trolley may enter the scale). Icons only, no words.

5. Floor walkway: from the entry signal, over the platform, to the exit gate, a floor-marking lane about 1 m wide in light grey with thin purple edge lines and white chevron arrows pointing toward the exit. In the middle of the lane, a large flat, simplified iaCarry logo mark (only the gradient blob with the white cart) printed on the floor.

6. Payment: a black contactless card terminal on an angled bracket on the totem at hand height, screen lit with a contactless symbol.

7. Exit: a waist-high brushed-steel swing gate at the end of the lane, with a round green arrow light on its post.

8. Top of the totem: the iaCarry logo exactly as in the third attached image: the gradient blob, "iaCarry" and "payment in just 6 seconds!".

9. Keep the trolley and its products: Chips Ahoy!, Tosta Rica, Kellogg's Smacks, ColaCao, two Coca-Cola cans, Fanta, Estrella Galicia, Mahou 5 Estrellas, Mahou 0,0, El Águila, H&S, Colgate and Axe.

No text anywhere except the logo and the screen. No watermarks, no sci-fi glow, no cartoon look.
```

## Basket station

```
Edit the first attached image, a photo of the iaCarry compact self-checkout kiosk for hand baskets in a Spanish supermarket. Keep the kiosk, the green basket and its products, the stainless weighing plate with its green LED edge, the card terminal and the green light on the front. Portrait 3:4.

Style: a clean, slightly schematic product visualisation. Materials, lighting and scale stay photorealistic, but simplify: calm, desaturated, softly blurred store background; soft even light; every component clearly separated and easy to identify at a glance. No clutter.

1. Screen: replace the display with a vertical digital signage screen, like a 32-inch portrait advertising display: slim black bezel, mounted on the kiosk's pole behind the weighing plate at eye level, set back so it does not lean over the basket. It shows exactly the second attached image, the iaCarry checkout interface: the top-down camera view of the basket with coloured outlines and name labels, the product cards, the total "10.98€" and a large purple "Contactless pay" button. Reproduce it faithfully and sharply. No other text on the screen.

2. Camera: make it unmistakable. A compact white camera module on an arm that bends forward from the top of the pole, hanging directly over the centre of the basket and pointing straight down, with nothing between it and the basket. A faint translucent light-blue cone runs from the lens down to the basket to show what it sees. A small round iaCarry logo sticker (the purple-to-pink gradient blob with the white cart and leaf from the third attached image) on the side of the camera housing.

3. Scale: the brushed-stainless weighing plate on top of the waist-high anthracite kiosk, clearly a scale, with its green LED edge. The iaCarry logo from the third attached image printed flat on the plate's front edge, beside the basket.

4. Floor: in front of the kiosk, a round floor decal about 80 cm across: light grey with a thin purple ring and the simplified iaCarry logo mark (only the gradient blob with the white cart) in the middle, marking where to stand.

5. Payment: a black contactless card terminal on an angled bracket at the right side of the kiosk, screen lit with a contactless symbol.

6. Status light: keep the round green arrow light on the front of the kiosk, meaning payment is done and the customer may leave.

7. Top sign: the iaCarry logo exactly as in the third attached image, on a panel at the top of the pole: the gradient blob, "iaCarry" and "payment in just 6 seconds!".

8. Keep the basket's products: two Coca-Cola cans, Fanta Limón, Mahou 5 Estrellas, Colgate, H&S and Chips Ahoy!.

No text anywhere except the logo and the screen. No watermarks, no sci-fi glow, no cartoon look.
```

## Notes

- Retailer and product brands in these images are for the demo only (see
  `serving/static/assets/README.md`). For a version without brands, replace
  point 6 with "unbranded cans, boxes, a cocoa jar, toothpaste and a shampoo
  bottle", and regenerate the screens with other products.
- The screens are rendered from the real page. Changed for the render only: the
  camera source reads "Live camera", and the station label is in English (it
  is hardcoded in Spanish in the template). The vertical version also stacks the
  page in one column and hides what a customer at the station does not need:
  the retailer and language pickers, search, filters, the weight panels, and
  the fixed "98% accuracy" and "1.2s inference" figures.
