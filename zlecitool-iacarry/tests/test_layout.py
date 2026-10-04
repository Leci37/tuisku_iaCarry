# -*- coding: utf-8 -*-
"""Las convenciones del repo que nadie revisa a ojo hasta que fallan."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = "zlecitool-core"


def _lines(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def test_production_pins_the_core_to_a_version():
    pinned = [line for line in _lines("requirements.txt") if line.startswith(CORE)]
    assert len(pinned) == 1 and "@ git+https://github.com/Leci37/zlecitool-core@v" in pinned[0], (
        "requirements.txt tiene que fijar el núcleo a una etiqueta vX.Y.Z")


def test_development_uses_the_core_beside_this_repo():
    assert any(line.startswith("-e ../zlecitool-core") for line in _lines("requirements-dev.txt"))


def test_dev_requirements_cover_production():
    own = [line for line in _lines("requirements.txt") if not line.startswith(CORE)]
    missing = sorted(set(own) - set(_lines("requirements-dev.txt")))
    assert not missing, f"requirements-dev.txt no tiene {missing}: en desarrollo faltarían"


def test_production_runs_gunicorn_with_its_config():
    # Sin gunicorn.conf.py, gunicorn arranca UN proceso: una petición lenta
    # dejaría a todo el mundo esperando detrás.
    assert "-c gunicorn.conf.py" in (ROOT / "Procfile").read_text(encoding="utf-8")


def _gunicorn(monkeypatch, **env):
    import runpy
    for name in ("WEB_CONCURRENCY", "GUNICORN_THREADS", "GUNICORN_TIMEOUT", "GUNICORN_MAX_REQUESTS",
                 "GUNICORN_KEEPALIVE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return runpy.run_path(str(ROOT / "gunicorn.conf.py"))


def test_a_recycled_worker_does_not_kill_an_ai_job(monkeypatch):
    """Un proceso que se recicla (max_requests) espera a sus trabajos sin dar
    señales, y gunicorn lo mata a los ``timeout`` segundos: medido en
    loadtest/, con timeout=30 y trabajos de 40–50 s se perdieron 5 de 9. El
    timeout tiene que pasar de una llamada a la IA (ZLECITOOL_AI_TIMEOUT, 300)."""
    conf = _gunicorn(monkeypatch)
    assert conf["worker_class"] == "gthread" and conf["timeout"] > 300
    assert conf["max_requests"] >= 10000, "reciclar cada minuto con carga cuesta CPU y corta conexiones"
    conf = _gunicorn(monkeypatch, GUNICORN_TIMEOUT="600", GUNICORN_MAX_REQUESTS="500", GUNICORN_KEEPALIVE="75")
    assert (conf["timeout"], conf["max_requests"], conf["max_requests_jitter"], conf["keepalive"]) == (600, 500, 50, 75)
