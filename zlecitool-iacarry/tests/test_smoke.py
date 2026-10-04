# -*- coding: utf-8 -*-
"""La prueba de humo: iaCarry arranca sobre el núcleo y cumple su contrato.

testing.check_tool(app) es el contrato del núcleo hecho prueba: entre otras
cosas, que cada página pide sesión, que la puerta de empresa funciona, que
paga la empresa, que las máquinas sólo entran donde dice @device, que cada
texto está en los ocho idiomas de la ficha y que el borrado por antigüedad
corre. Crece con cada versión del núcleo. No se quita ni se rodea.
"""

from pathlib import Path

from zlecitool_core import load_ficha, testing

ROOT = Path(__file__).resolve().parent.parent
FICHA = load_ficha(ROOT)


def test_the_tool_meets_the_core_contract(app):
    testing.check_tool(app)


def test_the_package_is_named_after_the_slug():
    assert FICHA.slug == "iacarry" and (ROOT / "iacarry" / "__init__.py").is_file()


def test_it_is_a_business_tool_with_its_own_brand_and_the_shoppers_languages():
    assert FICHA.is_business and FICHA.brand.name == "iaCarry" and FICHA.brand.pri == "#6c5dc7"
    assert FICHA.languages == ("es", "en", "eu", "ca", "gl", "pt", "fr", "de")
    assert dict(FICHA.retention) == {"frames/": 7}
    assert {op for op, _price in FICHA.prices} == {"recognition", "demo_recognition"}
