# Los logos de la empresa de demostración

Eroski, AhorraMas, Condis y Mercadona: los supermercados del desplegable de la
**empresa de demostración** de tuisku (`flask --app app iacarry demo <empresa>`),
cada uno recortado a su marca sobre transparente y de 160 px de alto como
mucho. Vienen de `tuisku_iaCarry/serving/static/assets/logos/`.

**Son marcas registradas de sus dueños.** Se usan sólo en la demostración y
con el permiso de cada supermercado; no los cubre la licencia de este código y
no se usan para nada más ni por nadie más. Un cliente de verdad nunca los ve:
sube su propio logo en «Marca». La portada pública no enseña ninguno.

El nombre de cada fichero es la clave del tema (`<clave>-logo.png`, la de
`DEMO_BRANDS` en `iacarry/service.py`). No llevan margen a propósito: la caja
les da a todos la misma área (`fitLogo()`), y un margen en el fichero
encogería ese logo.
