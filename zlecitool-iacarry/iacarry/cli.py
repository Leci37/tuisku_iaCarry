# -*- coding: utf-8 -*-
"""Los comandos de iaCarry (los del núcleo son ``flask zt …``).

    flask --app app iacarry demo <empresa> [--off]     la empresa de demostración de tuisku
    flask --app app iacarry frames <empresa> [--off]   guardar sus fotos (o no guardarlas nunca)

La empresa, por su identificador (``flask zt org list``). La demostración
enseña en sus cajas el selector de supermercados y las fotos de ejemplo:
sólo para la empresa de pruebas de tuisku, y con los logos usados con permiso.
"""

from __future__ import annotations

import click
from flask.cli import AppGroup

group = AppGroup("iacarry", help="iaCarry: la empresa de demostración y las fotos de cada empresa.")


def _org(key: str):
    from zlecitool_core.db import db
    from zlecitool_core.orgs import Org
    org = Org.query.filter_by(slug=key).first() or (db.session.get(Org, int(key)) if key.isdigit() else None)
    if org is None:
        raise click.ClickException(f"No hay ninguna empresa «{key}» (flask zt org list).")
    return org


@group.command("demo")
@click.argument("org_key", metavar="EMPRESA")
@click.option("--off", is_flag=True, help="Deja de ser la de demostración.")
def demo(org_key: str, off: bool) -> None:
    """Marca (o desmarca) la empresa de demostración de tuisku."""
    from zlecitool_core.db import db
    from .service import settings_of

    org = _org(org_key)
    settings_of(org.id).demo = not off
    db.session.commit()
    click.echo(f"{org.name}: {'ya no es' if off else 'es'} la empresa de demostración.")


@group.command("frames")
@click.argument("org_key", metavar="EMPRESA")
@click.option("--off", is_flag=True, help="No guardar nunca sus fotos (sólo lo detectado).")
def frames(org_key: str, off: bool) -> None:
    """Si se guardan las fotos de sus cajas (con los días de borrado de su contrato)."""
    from zlecitool_core.db import db
    from .service import settings_of

    org = _org(org_key)
    settings_of(org.id).keep_frames = not off
    db.session.commit()
    click.echo(f"{org.name}: sus fotos {'no se guardan' if off else 'se guardan (y se borran a sus días)'}.")
