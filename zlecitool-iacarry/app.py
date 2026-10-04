# -*- coding: utf-8 -*-
"""Arranque de la herramienta. Sólo arranque: lo propio vive en iacarry/.

    python app.py                                   desarrollo, en http://localhost:5000
    gunicorn -c gunicorn.conf.py "app:create_app()"  producción (Procfile)

El núcleo se enciende primero, con init_tool(app): sesión común, cuentas,
idiomas, seguridad. Después se registra lo propio. Con cada versión nueva del
núcleo, init_tool enciende más cosas sin que este fichero cambie.
"""

import os
import sys
from pathlib import Path

from flask import Flask

from zlecitool_core import init_tool
from zlecitool_core.config import load_env_file
from zlecitool_core.db import create_tables
from zlecitool_core.security import MissingSecretError


def create_app() -> Flask:
    app = Flask(__name__)
    init_tool(app)  # lee tool.json, aquí al lado

    from iacarry import models  # noqa: F401  (registra las tablas antes de crearlas)
    from iacarry.cli import group
    from iacarry.routes import bp

    app.register_blueprint(bp)
    app.cli.add_command(group)
    create_tables(app)
    return app


if __name__ == "__main__":
    load_env_file(Path(__file__).with_name(".env"))
    # El servidor de desarrollo sirve por http: con la bandera Secure, el
    # navegador no guardaría la cookie de sesión y no se podría entrar.
    os.environ.setdefault("ZLECITOOL_INSECURE_COOKIES", "1")
    try:
        application = create_app()
    except MissingSecretError as exc:
        sys.exit(f"{exc}\n\n(En desarrollo, ponlo en el .env: cp .env.example .env)")
    application.run(debug=True, port=int(os.environ.get("PORT", "5000")))
