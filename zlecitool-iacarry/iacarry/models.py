# -*- coding: utf-8 -*-
"""Las tablas de iaCarry.

Con el db del núcleo (uno para todas las herramientas) y SIEMPRE con
__tablename__ explícito y el prefijo iacarry_. iaCarry es una herramienta de
empresa: cada fila es de una empresa cliente (``org_id`` → ``core_org.id``) y
cada consulta filtra por la empresa de quien pide (``current_org().id`` en
routes.py; aquí llega como ``org_id``). Las tablas core_* son del núcleo: se
apunta a ellas, no se escriben.

Los precios van en céntimos y los pesos en gramos: enteros, sin redondeos.
"""

from __future__ import annotations

import json

from zlecitool_core.db import db, utcnow

#: Las categorías del catálogo: la clave de su texto en i18n/ui.json es stCat<Nombre>.
CATEGORIES = ("drinks", "snacks", "care", "food", "other")


class RetailerBrand(db.Model):
    """La marca del supermercado cliente en sus cajas («iaCarry │ para …»).

    La paleta sale de un solo color (``service.derive_palette``): ``pri`` rellena
    (botones, chips), ``ink`` es el mismo color oscurecido para el texto pequeño
    sobre los tintes, y el resto son los tintes de fondo. El logo va al almacén."""

    __tablename__ = "iacarry_retailer_brand"

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("core_org.id"), nullable=False, unique=True)
    display_name = db.Column(db.String(80), nullable=False)
    pri = db.Column(db.String(7), nullable=False)
    ink = db.Column(db.String(7), nullable=False)
    palette = db.Column(db.Text, nullable=False, default="{}")
    #: La clave del logo en el almacén (iacarry/brand/o<org>/logo-<n>.png) y su tamaño.
    logo_key = db.Column(db.String(200), nullable=True)
    logo_w = db.Column(db.Integer, nullable=True)
    logo_h = db.Column(db.Integer, nullable=True)
    updated_by = db.Column(db.Integer, db.ForeignKey("core_user.id"), nullable=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    @property
    def tints(self) -> dict:
        return json.loads(self.palette or "{}")


class Product(db.Model):
    """Un producto del catálogo de una empresa: una clase que conoce el modelo
    (``tag``) con el nombre, el precio y el peso de esa tienda."""

    __tablename__ = "iacarry_product"
    __table_args__ = (db.UniqueConstraint("org_id", "tag", name="uq_iacarry_product_org_tag"),)

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("core_org.id"), nullable=False, index=True)
    tag = db.Column(db.String(60), nullable=False)
    name = db.Column(db.String(120), nullable=False)
    short_name = db.Column(db.String(40), nullable=True)
    price_cents = db.Column(db.Integer, nullable=False, default=0)
    weight_g = db.Column(db.Integer, nullable=False, default=0)
    category = db.Column(db.String(20), nullable=False, default="other")
    color = db.Column(db.String(40), nullable=False, default="#6c5dc7")
    active = db.Column(db.Boolean, nullable=False, default=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)


class Station(db.Model):
    """Una caja: lo que iaCarry sabe de una pantalla emparejada de la empresa
    (``device_id`` → ``core_device.id``). Se crea sola la primera vez que la
    pantalla abre /station; en «Cajas» se le pone dónde está."""

    __tablename__ = "iacarry_station"

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("core_org.id"), nullable=False, index=True)
    device_id = db.Column(db.Integer, db.ForeignKey("core_device.id"), nullable=False, unique=True)
    location = db.Column(db.String(120), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    last_detection_at = db.Column(db.DateTime, nullable=True)


class Checkout(db.Model):
    """Un carro en una caja (o una prueba en «Probar», sin caja): lo que se
    reconoció en su última foto y si se pagó."""

    __tablename__ = "iacarry_checkout"

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("core_org.id"), nullable=False, index=True)
    station_id = db.Column(db.Integer, db.ForeignKey("iacarry_station.id"), nullable=True, index=True)
    #: Quién la hizo: la cuenta técnica de la caja, o la persona que probaba.
    user_id = db.Column(db.Integer, db.ForeignKey("core_user.id"), nullable=False)
    demo = db.Column(db.Boolean, nullable=False, default=False)
    started_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    #: detected | paid | abandoned
    status = db.Column(db.String(12), nullable=False, default="detected")
    units = db.Column(db.Integer, nullable=False, default=0)
    lines = db.Column(db.Integer, nullable=False, default=0)
    total_cents = db.Column(db.Integer, nullable=False, default=0)
    expected_weight_g = db.Column(db.Integer, nullable=False, default=0)
    #: {clase: unidades}, en JSON.
    items = db.Column(db.Text, nullable=False, default="{}")
    payment_ref = db.Column(db.String(60), nullable=True)
    payment_simulated = db.Column(db.Boolean, nullable=True)
    paid_at = db.Column(db.DateTime, nullable=True)

    @property
    def item_counts(self) -> dict:
        return json.loads(self.items or "{}")


class Frame(db.Model):
    """Una foto analizada. La imagen va al almacén, en la carpeta de su empresa
    con borrado por antigüedad (``frames/``); cuando el núcleo la borra, la fila
    se queda con lo detectado: ni imagen ni personas."""

    __tablename__ = "iacarry_frame"

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("core_org.id"), nullable=False, index=True)
    checkout_id = db.Column(db.Integer, db.ForeignKey("iacarry_checkout.id"), nullable=True, index=True)
    station_id = db.Column(db.Integer, db.ForeignKey("iacarry_station.id"), nullable=True)
    request_id = db.Column(db.String(64), nullable=True, index=True)
    storage_key = db.Column(db.String(200), nullable=True)
    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)
    #: Lo que contestó el detector (las cajas que pasan el umbral y las que no), en JSON.
    detections = db.Column(db.Text, nullable=False, default="[]")
    model_version = db.Column(db.String(40), nullable=True)
    inference_ms = db.Column(db.Integer, nullable=True)
    round_trip_ms = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)


class OrgSettings(db.Model):
    """Lo de iaCarry que cambia por empresa y no es de la marca ni del catálogo."""

    __tablename__ = "iacarry_org_settings"

    id = db.Column(db.Integer, primary_key=True)
    org_id = db.Column(db.Integer, db.ForeignKey("core_org.id"), nullable=False, unique=True)
    #: La empresa de demostración de tuisku: su caja enseña el selector de
    #: supermercados y las fotos de ejemplo. Ningún cliente lo ve.
    demo = db.Column(db.Boolean, nullable=False, default=False)
    #: Guardar las fotos (con los días de borrado de su contrato) o no guardarlas nunca.
    keep_frames = db.Column(db.Boolean, nullable=False, default=True)
