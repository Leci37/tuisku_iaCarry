# De un carro vacío a pagado

El recorrido de una compra en la caja, para el comprador y en el código, con
la línea de log de cada paso. Es el `serving/FLOW.md` de `tuisku_iaCarry`
llevado a la herramienta: la página es la misma, el servidor ya no.

**Tres cosas son costuras**, simuladas y dichas como tales: la cámara cenital
(no hay señal en directo), el sensor del carro (el interruptor «Vacío /
Lleno») y el pago con la puerta. Cada una es una función de
`static/js/station.js`. Todo lo demás es de verdad: la foto va al detector,
se cobra a la empresa, y cada número de la pantalla sale de la respuesta y del
catálogo de la empresa.

## En corto

```
 ┌──────────── la caja (navegador) ────────────┐   ┌────── iaCarry (web) ──────┐   ┌── detector ──┐
 │ carro (interruptor) → foto → POST           │──▶│ cobra → detector → apunta │──▶│ /v1/detect   │
 │                      /station/detect         │   │ la compra y la foto       │◀──│ (su servicio)│
 │ adaptar → cantidades, cajas, total → render │◀──│ JSON de /upload +         │   └──────────────┘
 │ pagar (simulado) → apunta «pagada»          │   │ request_id, checkout_id   │
 └─────────────────────────────────────────────┘   └───────────────────────────┘
```

Cada foto lleva un `X-Request-Id` que pone la página; sale en cada línea de
log de las tres capas y es la clave del cobro: una compra es un `grep`.

## Para el comprador

| # | Qué hace | Qué ve |
|---|---|---|
| 0 | Llega a la caja | Carro vacío, «Aún no se ha analizado ninguna imagen», `0,00 €`, pagar en gris, la tira gris «Coloca el carro para empezar» |
| 1 | Deja el carro | El paso 1 se pone en verde y el 2 se activa |
| 2 | Se analiza | La foto del carro con un círculo que gira encima, «Analizando el carro…»; pagar sigue desactivado |
| 3 | Sale bien | Las cajas de colores sobre cada producto, las fichas en el carro (`✓ × N`), el total, el peso estimado y la tira verde «Todo correcto, ya puedes pagar» |
| 3b | Falla, no encuentra nada o la empresa no tiene saldo | La tira roja («Fallo de detección — avisa a un asistente», o «No se ha reconocido ningún producto — avisa a un asistente») y no se puede pagar |
| 4 | Pulsa pagar | «Procesando pago…», el botón desactivado |
| 5 | Pagado | El tic verde, las gracias, el tique, «Pago simulado. No se ha cobrado nada.» y la nota «Modo demo: no hay pasarela de pago ni apertura de puerta conectadas.» |
| 6 | Pulsa «Nueva compra» | Todo vuelve al paso 0 |

El comprador nunca cambia una cantidad: lo que manda es la detección. Los
textos salen en su idioma (el selector de la caja), y con «Texto grande» más
grandes.

## En el código

### 0 — la página

| | |
|---|---|
| Ruta | `GET /station` → `station()` (`@device(home=True)`): sólo una caja emparejada; una persona va a «Probar» (`/demo`) |
| Plantilla | `templates/station.html` sobre `zt/bare.html` (sin barra ni pie) |
| Datos | `data-config`: la marca del supermercado (o las de la empresa de demostración), su catálogo (céntimos y gramos), las fotos de ejemplo si toca, el nombre de la caja, las URL de detectar y de pagar |
| Navegador | `station.js` lee `data-config`, espera el diccionario (`zt.i18nReady`), pinta y marca `#station[data-ready="1"]` |

```
navegador info  [iaCarry] lista — detección:/station/detect | pago:SIMULADO | sensor del carro:interruptor | cámara: ninguna
```

### 1 — el carro · costura

`setCartPresent(present, source)` es el único que cambia `state.cartFull`. Lo
llaman el interruptor del paso 1, «Nueva compra» y una detección que encuentra
productos (encontrarlos ya dice que el carro está). Con `false` se borran la
detección, la foto y el pago a la vez.

### 2 — la foto · costura

`submitFrame(blob, nombre)` es la costura de la cámara: lo que tenga una foto
la da aquí. Hoy, en «Probar» y en la empresa de demostración, las fotos de
ejemplo (`runDemo`) o un fichero; en una caja de verdad, quien integre la
cámara de la tienda llama a `window.iaCarry.submitFrame`.

```
submitFrame(blob, nombre, seq, rid)
   ├ ++detectSeq                    una respuesta tardía no pisa un carro más nuevo
   ├ setCamImage(blob)              la proporción del panel, la de la imagen
   ├ setDetect("busy","analysing")  el velo de ocupado, pagar desactivado
   └ fetch("/station/detect", POST, "file" + "checkout", X-Request-Id, 60 s)
```

### 3 — el servidor

`station_detect()` → `_detect("recognition", demo=False)` en `iacarry/routes.py`:

1. `read_upload()` del núcleo: PNG, JPG o WebP, 8 MB como mucho (400 con
   `errNoFile`, `errFileType` o `errFileTooLarge`).
2. Si ya hay una foto con ese `X-Request-Id` (un reintento), contesta la de
   antes: ni detector ni cobro.
3. `charge("recognition")`: sin saldo, 402 `errOrgCreditsNeeded` y la caja en
   rojo.
4. `end_idle_transaction()` y `detector.detect()`: un fallo es 503
   `errDetectorDown` (o `errDetectorBusy`), sin cobrar.
5. Si la empresa guarda fotos, la foto al almacén en
   `iacarry/frames/o<empresa>/` (el borrado diario del núcleo).
6. `service.record()`: la compra (`iacarry_checkout`: unidades, líneas, total
   en céntimos, peso esperado en gramos, con los productos activos del
   catálogo y probabilidad ≥ 0,45) y la foto (`iacarry_frame`: lo detectado,
   la versión del modelo, los tiempos).
7. `work.done()`: ahora sí, el cobro.

```
INFO  Foto analizada | org=supermercados-norte caja=3 rid=r1a2b3 unidades=12 total=2631 ms=212
WARN  Foto rechazada | org=… op=recognition motivo=errOrgCreditsNeeded rid=…
ERROR Detector | org=… rid=… <qué falló: sin respuesta, un 500, ocupado>
```

### 4 — la respuesta

El JSON de `/upload` de siempre, con `request_id`, `checkout_id` y
`frame_id`. `adaptDetections()` arregla las cajas (`width`/`height` son el
borde derecho e inferior), quita lo que está por debajo de 0,45 y cuenta
cada producto; `applyDetections()` lo deja en `state.live` y `render()` lo
pinta. El total sale de los precios del catálogo de la empresa, en céntimos.

### 5 — pagar · costura

`pay()` → `requestPayment()`: sin pasarela (`PAYMENT_BACKEND = false`), una
aprobación simulada y marcada como tal. Después `recordPaid()` manda
`POST /station/checkout/<id>/paid`, y la compra sale en «Compras» como
«Pagada (simulada)».

## Lo que mira cada prueba

`tests/test_station_browser.py` recorre esto en Chromium con el detector
falso: las cajas sobre los productos (A), cantidades, total y peso (B),
ocupado (C), los fallos en rojo (D), los idiomas (E), nada de fuera (F), cada
foto de ejemplo (H), el logo (I), el pago, y una caja emparejada que recibe la
foto por `window.iaCarry.submitFrame` y la cobra.
