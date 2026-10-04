# -*- coding: utf-8 -*-
"""Las fotos de los carros se borran solas (el borrado por antigüedad del
núcleo, ``"retention"`` en la ficha): a los 7 días, o a los del contrato de
cada empresa. Lo detectado se queda."""

import io
import os
import time
import uuid

from zlecitool_core import retention
from zlecitool_core.artifacts import current_storage

from iacarry.models import Frame

from conftest import paired_station, photo


def _frame(app, owner):
    station = paired_station(app, owner)
    body = station.post("/station/detect", headers={"Sec-Fetch-Dest": "empty", "X-Request-Id": uuid.uuid4().hex},
                        data={"file": (io.BytesIO(photo(1)), "ziacarry_eval_img_1.png")},
                        content_type="multipart/form-data").get_json()
    with app.app_context():
        return Frame.query.filter_by(id=body["frame_id"]).one().storage_key, body


def _age(app, key, days):
    with app.app_context():
        path = current_storage()._path(key)
    stamp = time.time() - days * 86400
    os.utime(path, (stamp, stamp))


def test_photos_older_than_their_days_are_deleted_and_the_detection_stays(app, owner):
    key, body = _frame(app, owner)
    assert owner.get(f"/frames/{body['frame_id']}").status_code == 200
    _age(app, key, 8)
    with app.app_context():
        row = retention.sweep()
        assert row.ok and row.deleted == 1
        assert not current_storage().exists(key)
        assert Frame.query.one().detections != "[]", "lo detectado se queda"
    assert owner.get(f"/frames/{body['frame_id']}").status_code == 404
    assert 'data-i18n="iaPhotoGone"' in owner.get("/checkouts").get_data(as_text=True)


def test_a_fresh_photo_stays(app, owner):
    key, _body = _frame(app, owner)
    _age(app, key, 3)
    with app.app_context():
        retention.sweep()
        assert current_storage().exists(key)
