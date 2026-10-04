# CLAUDE.md — herramienta zlecitool `iacarry`

Esta es una herramienta **zlecitool**: una app Flask pequeña que sólo contiene
lo suyo. Lo común lo pone el núcleo, `zlecitool-core`, instalado como
dependencia (la versión la fija `requirements.txt`).

Es **de empresa** (`"audience": "business"`): la usan las personas y las
máquinas (cajas, pantallas, programas) de una empresa que la tiene dada, y la
paga el saldo de la empresa. Lleva su propia marca (`"brand"` en `tool.json`,
`static/brand/`), sin publicidad ni aviso de cookies, y no tiene alta abierta.

**iaCarry** es el autocobro con IA para supermercados: la caja (`/station`, una
pantalla emparejada) manda cada foto del carro al detector, enseña lo
reconocido con los precios del catálogo de la empresa y cobra un
`recognition` por foto; la trastienda (panel, cajas, catálogo, marca,
compras, «Probar») es de las personas de la empresa. Lo cuenta el README.

Antes de escribir código, lee lo que el núcleo ofrece: su **CONTRATO.md**
(en desarrollo, `../zlecitool-core/CONTRATO.md`; si no está, en GitHub:
`Leci37/zlecitool-core`, en la etiqueta de `requirements.txt`).

## Reglas

1. **Lo común no se escribe aquí.** Cuentas, seguridad, créditos, estética,
   publicidad, idiomas, PDF, llamadas a IA, pruebas: vienen del núcleo. Si
   falta algo, **para y dilo**: se añade al núcleo, en otra sesión y en su
   repo, sale una versión nueva y aquí se sube. Nada de parches locales.
2. **Nada copiado** del núcleo ni de otra herramienta.
3. **Del núcleo sólo se importa lo que está en su CONTRATO.md.**
4. **Tablas con prefijo:** `__tablename__ = "iacarry_algo"`, siempre explícito.
   Las tablas `core_*` son del núcleo: no se leen ni se escriben directamente.
5. **Cada cosa en su sitio:** `app.py` sólo arranca; `iacarry/routes.py`
   recibe la petición y responde; `iacarry/service.py` hace el trabajo, sin
   Flask, para poder probarlo solo; `iacarry/models.py` son las tablas.
6. **Ficheros sólo dentro de** `current_tool().data_dir`.
7. **Pruebas en verde:** `pytest`. `testing.check_tool(app)` es el contrato
   del núcleo: no se quita ni se rodea.
8. **Subir el núcleo** es una decisión: leer su CHANGELOG, cambiar la
   etiqueta en `requirements.txt` y pasar las pruebas.
9. **Idioma:** documentación y comentarios en español; nombres en el código,
   en inglés.
10. **Toda ruta pide sesión** sin hacer nada; lo público se marca con `@public`.
     Nada de `login_required` a mano.
     Las filas llevan `org_id` → `core_org.id` (de qué empresa son) y
     `user_id` → `core_user.id` (quién lo hizo), y **cada consulta filtra por
     `current_org().id`**. Lo que sólo hace quien administra la empresa, con
     `org_admin_required`; la puerta (quién entra) es del núcleo.
11. **La sesión es de todas las herramientas:** sólo claves `iacarry.algo`, pequeñas.
12. **Un rechazo es una clave del diccionario** (`{"error": "errAlgo"}` o
     `notice("errAlgo")`), nunca una frase en el código. Los formularios POST
     llevan `csrf_token()`.
13. **Textos de la interfaz** en `i18n/ui.json`, en todos los idiomas de la
    herramienta a la vez (los 8 de base, o los de `"languages"` en la ficha), y
    en la plantilla con su clave: `<h2 data-i18n="clave">{{ zt_t('clave') }}</h2>`,
    para que el selector de idioma los cambie sin recargar. En JavaScript,
    `zt.t('clave')`; para hablar con el servidor, `zt.fetchJSON`, que enseña
    traducido el `{"error": "clave"}` que conteste.
14. **Las páginas extienden `zt/base.html`**: barra, idioma, cuenta, publicidad,
    pie legal y aviso de cookies vienen de ahí.
    En una de empresa, con su marca (la pone la carcasa desde la ficha), sin
    publicidad ni aviso de cookies.
    Una pantalla sin barra (un quiosco, una caja) extiende `zt/bare.html`.
15. **Ficheros al almacén del núcleo** (`current_storage()`), nunca a una carpeta
    propia: los logos en `iacarry/brand/o<empresa>/`, las fotos donde diga la
    regla 19. Si un día hace falta un PDF, el híbrido del núcleo: «Guardar
    como PDF» del navegador (`data-zt-print`) y «Descargar PDF»
    (`zlecitool_core.export.download_pdf`) sólo cuando haga falta el fichero.
16. **Lo que cuesta, en `tool.json`** (`"prices"`), nunca en el código: la
    ruta nombra la operación y cobra con `with charge("op") as work:` →
    `work.done()` sólo si salió bien; si no está `work.allowed`, devuelve
    `work.refusal()`. Cada operación lleva su texto `op_<operación>` en
    `i18n/ui.json`, y la pantalla enseña el precio con `zt_price('op')` antes
    de pulsar. Lo que cobra va por POST. El saldo no se lee ni se apunta a mano.
    Paga la empresa: el mismo `charge()`; en las pruebas,
    `testing.make_org(app, credits=…)` y `testing.signed_in_client(app, org=…)`.
17. **El detector no vive aquí.** TensorFlow nunca se importa en la web: el
    modelo es un servicio interno (`tuisku_iaCarry/serving/detector_service.py`)
    y se le llama sólo desde `iacarry/detector.py`, con `end_idle_transaction()`
    antes (no se espera con una conexión cogida) y sus errores como claves
    (`errDetectorDown`, `errDetectorBusy`). Se cobra con `charge()` y sólo si
    contestó (`work.done()` después de apuntar la foto). Ninguna prueba llama
    al de verdad: sin `IACARRY_DETECTOR_URL` está el falso, y
    `detector.fake_detector()` dice qué contesta. Si algún día hace falta una
    IA de texto, la del núcleo (`zlecitool_core.ai`), nunca `anthropic` u
    `openai` aquí.
18. **Las máquinas** (una caja, una pantalla de pared, el programa de un
    cliente) sólo entran en las vistas marcadas `@device` (un navegador
    emparejado con su código) o `@device(api=True)` (también un programa con su
    ficha de API, sin token CSRF); `current_device()` dice cuál. La pantalla
    adonde va un navegador al emparejarse, con `@device(home=True)` (aquí,
    `/station`). Nunca una cuenta de persona ni una contraseña guardada en una
    máquina.
19. **Lo que caduca** (las fotos que hace una caja en la tienda de un cliente)
    se declara en `"retention"` (aquí `[{"prefix": "frames/", "days": 7}]`) y
    lo de cada empresa va en su carpeta (`retention.org_key("frames/", nombre)`):
    el núcleo lo borra cada día. Nada de borrados a mano. Lo detectado se queda
    en `iacarry_frame` sin la imagen; una foto nunca va a los logs.
20. **La caja es un `state` y un `render()`.** `static/js/station.js` conserva la
    forma del original: `render()` es el único que escribe el DOM, `detectSeq`
    descarta respuestas viejas, cada foto lleva su `X-Request-Id` (la clave del
    cobro) y las costuras (cámara, sensor del carro, pago y puerta) son una
    función cada una, simuladas y dichas en la pantalla hasta que haya algo de
    verdad detrás. `window.iaCarry` es la superficie de integración y de
    pruebas: no se le quitan nombres. Los datos llegan en `data-config` desde
    el servidor; nada de constantes de una empresa en el JavaScript.
21. **El contrato de la página no cambia.** `/station/detect` y `/demo/detect`
    contestan como el `/upload` de siempre (`boundingBox` con `width`/`height`
    que son el borde derecho e inferior); el contrato honrado (`x1, y1, x2, y2`)
    es el del detector, y `Detection.for_page()` traduce.
22. **Precios en céntimos y pesos en gramos**, enteros, en la base de datos y
    en la página; sólo se convierten al enseñarlos (`euros`, `kilos`, `money()`).
23. **Las marcas de los supermercados** (`static/logos/`) sólo en la empresa de
    demostración y con permiso de cada uno; un cliente sube la suya. La
    portada pública nunca enseña ninguna.

## Ramas

Se trabaja en `develop`; `main` sólo para versiones publicadas.

## Comandos

```
pip install -r requirements-dev.txt     # dentro de tuisku_iaCarry: el núcleo está en ../../zlecitool-core
playwright install chromium
python app.py                           # con el detector falso
pytest                                  # las de navegador se saltan sin Chromium
flask --app app iacarry demo <empresa>  # la empresa de demostración
python scripts/landing_images.py        # las imágenes de la portada
```
