# Lo que cuesta iaCarry

**Precios provisionales.** Los que hay en `tool.json` (`recognition: 1`,
`demo_recognition: 1`) están para que la herramienta cobre de verdad desde el
primer día, no porque estén decididos. Decidirlos es del dueño (plan,
§16, decisiones 3 y 4); esta página reúne lo que hace falta para hacerlo.

## Lo que se cobra hoy

| Operación | Cuándo | Créditos | Quién paga |
|---|---|---|---|
| `recognition` | cada foto que analiza una caja (`POST /station/detect`) | 1 | el saldo de la empresa |
| `demo_recognition` | cada foto en «Probar» (`POST /demo/detect`) | 1 | el saldo de la empresa |

- Se cobra **sólo si el detector contesta**: un fallo, un «ocupado» o una foto
  rechazada no cuestan nada (el `charge()` del núcleo).
- **Una foto, un cobro**: la caja pone un `X-Request-Id` a cada foto, y un
  reintento con el mismo (tras un corte de red) ni se analiza ni se cobra otra
  vez; contesta lo de la primera.
- Si la caja manda **varias fotos del mismo carro** (cuando haya cámara en
  directo, volverá a mirar si el carro cambia), cada una cuesta: el coste de
  una compra son sus fotos. Cada foto queda en `iacarry_frame` con su compra.
- El límite diario por cuenta es 5000 (`daily_limit`); por empresa, en
  «Clientes».

## Lo que dice la presentación

`tuisku_iaCarry/presentation/slides.md` (cifras «estimadas»):

| Concepto | Presentación |
|---|---|
| Licencia por caja | 900 € al mes (la presentación anterior decía 420 €) |
| Por reconocimiento | 0,01 € en compras de menos de 8 €; 0,04 € en las demás («5000 compras al mes por 200 €») |
| Mantenimiento por caja | 45 € al mes (antes, 165 €) |
| La caja (báscula, cámara, pantalla) | 18 000 €, aparte |

## Lo que falta decidir

| # | Qué | Para decidirlo |
|---|---|---|
| 1 | **Cuánto vale un crédito de empresa** (las recargas se apuntan con su importe: `flask zt org topup <empresa> 5000 --cents 50000`) | el precio por reconocimiento de la presentación y lo que cuesta cada foto en el detector (abajo) |
| 2 | **Créditos por foto** (`recognition`), o por compra | si se cobra por foto, un carro de varias fotos cuesta varias; si se quiere por compra, el cobro pasaría al pagar (otra operación) |
| 3 | Si «Probar» cuesta (`demo_recognition`) o se regala con los créditos de prueba | hoy cobra, como la caja |
| 4 | **La licencia mensual por caja** | al principio, facturada fuera y apuntada como recarga (`topup`); un cobro periódico del núcleo sólo si se vuelve rutina |
| 5 | Los créditos de prueba de un cliente nuevo | `flask zt org grant <empresa> 500 --note "prueba"` |

## Lo que cuesta cada foto

El detector mide su tiempo (`inference_ms`) y la herramienta el de ida y
vuelta (`round_trip_ms`); los dos quedan en `iacarry_frame`, con
`model_version`. Con el coste por hora de la máquina del detector (CPU o
GPU) y las fotos por hora que atiende, sale el coste por foto, que es el
suelo del precio de un `recognition`. El panel del núcleo (`/zt/admin`,
`flask zt stats`) da las operaciones cobradas por día y por empresa.
