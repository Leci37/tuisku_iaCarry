# Station renders for Sora

Two images of the iaCarry weighing station for the presentation: the **basket**
station for small purchases and the **trolley** station for large ones.

| File | What it is |
|---|---|
| `draft_v1_basket.webp`, `draft_v1_trolley.webp` | First Sora results, reviewed below |
| `screen_basket.png`, `screen_trolley.png` | The real checkout screen, large text, showing exactly the products in each draft. Upload it with the prompt, or paste it onto the screen afterwards |
| `frame_basket.png`, `frame_trolley.png` | The schematic top-down "camera" photo shown inside each screen |
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
- The screen is portrait. The real interface is landscape.
- The camera arm is short: the camera sits near the pillar, over the back of the
  trolley, not over its centre.
- No logo on the camera, the platform or the floor.
- Correct and worth keeping: the platform with its green LED strip, the exit
  gate with the green arrow, the card terminal, the busy store behind, the
  products.

## How to use the prompts

Upload three images with each prompt, in this order:

1. the draft (`draft_v1_*.webp`), the layout to keep
2. the screen (`screen_*.png`), what the display must show
3. the logo (`../images/logo_iacarry_full.png`)

If the screen still comes out garbled, ask for a "blank white landscape screen"
instead and paste `screen_*.png` onto it afterwards. That is the most reliable
way to get a sharp interface.

## Trolley station

```
Edit the first attached image, a photo of the iaCarry self-checkout station for shopping trolleys. Keep its composition, camera angle, lighting, store background, trolley, products, weighing platform, card terminal and exit gate. Make it more realistic and fix the following.

1. Screen: make the display a landscape 24-inch touchscreen, mounted on the pillar at eye level and facing the customer. It shows exactly the second attached image, the iaCarry checkout interface: a top-down camera photo of the trolley's products with coloured outlines on the left, product cards on the right, the total "28.09€" and a large purple "Contactless pay" button. Reproduce it faithfully and sharply. Do not invent other text.

2. Camera: lengthen the arm so the white camera hangs directly over the centre of the trolley, pointing straight down. Put a small round iaCarry logo sticker on the side of the camera housing: the purple-to-pink gradient blob with the white shopping cart and leaf from the third attached image.

3. Platform: print the iaCarry logo from the third attached image on the stainless-steel top of the weighing platform, at the front edge, clearly readable, as a flat matte decal. Keep the green LED strip.

4. Floor walkway: from the platform to the exit gate, add a floor-marking lane about 1 m wide in light grey with thin purple edge lines and three white chevron arrows pointing toward the gate. In the middle of the lane, a large flat, simplified iaCarry logo mark (just the gradient blob with the white cart) printed on the floor, slightly worn, like real floor signage.

5. Pillar: keep the iaCarry logo panel at the top, exactly as in the third attached image: the gradient blob, the word "iaCarry" and "payment in just 6 seconds!".

6. Keep the trolley's products as they are: Chips Ahoy!, Tosta Rica, Kellogg's Smacks, ColaCao, two Coca-Cola cans, Fanta, Estrella Galicia, Mahou 5 Estrellas, Mahou 0,0, El Águila, H&S, Colgate and Axe.

Photorealistic, natural store lighting, 16:9. No extra text, no watermarks, no sci-fi glow.
```

## Basket station

```
Edit the first attached image, a photo of the iaCarry compact self-checkout kiosk for hand baskets. Keep its store background, lighting, the green basket and its products, the stainless weighing plate with the green LED edge, the card terminal and the green exit arrow on the front. Make it more realistic and fix the following.

1. Screen: move the display up and back so it no longer leans over the basket. It becomes a landscape 15-inch touchscreen on the pole behind the plate, at eye level, tilted slightly toward the customer. It shows exactly the second attached image, the iaCarry checkout interface: the top-down camera photo of the basket with coloured outlines on the left, the products on the right, the total "10.98€" and a large purple "Contactless pay" button. Reproduce it faithfully and sharply. Do not invent other text.

2. Camera: move the white camera so it hangs directly over the centre of the basket on a forward-bending arm, pointing straight down, with nothing between it and the basket. Put a small round iaCarry logo sticker on the side of the camera housing: the purple-to-pink gradient blob with the white shopping cart and leaf from the third attached image.

3. Scale: print the iaCarry logo from the third attached image on the stainless weighing plate, at the front edge beside the basket, as a flat matte decal, clearly readable.

4. Floor: in front of the kiosk, add a round floor decal about 80 cm across: a light grey circle with a thin purple ring and the simplified iaCarry logo mark (just the gradient blob with the white cart) in the middle, slightly worn, like real floor signage marking where to stand.

5. Top sign: keep the iaCarry logo panel above the screen, exactly as in the third attached image: the gradient blob, the word "iaCarry" and "payment in just 6 seconds!".

6. Keep the basket's products as they are: two Coca-Cola cans, Fanta Limón, Mahou 5 Estrellas, Colgate, H&S and Chips Ahoy!.

Photorealistic, natural store lighting, portrait 3:4. No extra text, no watermarks, no sci-fi glow.
```

## Notes

- Retailer and product brands in these images are for the demo only (see
  `serving/static/assets/README.md`). For a version without brands, replace
  point 6 with "unbranded cans, boxes, a cocoa jar, toothpaste and a shampoo
  bottle", and regenerate the screens with other products.
- The screens are rendered from the real page. Only two things are changed for
  the render: the camera source reads "Live camera", and the station label is in
  English (it is hardcoded in Spanish in the template).
