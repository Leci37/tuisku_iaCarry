# -*- coding: utf-8 -*-
"""gunicorn en producción: cuántos procesos, cuántos hilos, cuánto esperar.

Sin esto, gunicorn arranca UN proceso, y una sola petición lenta (un PDF, una
llamada a la IA) deja a todo el mundo esperando detrás. Todo se puede cambiar
desde el entorno, sin tocar el código:

    WEB_CONCURRENCY    procesos      (por defecto, 2 × núcleos + 1)
    GUNICORN_THREADS   hilos por proceso (4): mientras uno espera a la base
                       de datos o a una API, los otros atienden
    GUNICORN_TIMEOUT   segundos sin dar señales antes de matar un proceso (360).
                       Un proceso que se recicla (max_requests) espera a sus
                       trabajos en segundo plano sin dar señales: uno más
                       largo que esto se pierde. Más que la llamada más larga
                       a la IA (ZLECITOOL_AI_TIMEOUT, 300). Con gthread no corta
                       una petición lenta: la señal la da otro hilo.
    GUNICORN_MAX_REQUESTS  peticiones antes de reciclar un proceso (10000)
    GUNICORN_KEEPALIVE segundos que se guarda abierta una conexión sin uso (2).
                       Detrás de un balanceador que reutiliza conexiones (un
                       ALB, 60 s), más que el suyo (75): si no, gunicorn cierra
                       la que el balanceador iba a usar, y eso es un 502.
    PORT               el puerto (5000)

Con 4 núcleos: 9 procesos × 4 hilos = 36 peticiones a la vez. Lo que de verdad
ocupa mucho tiempo no ocupa estos hilos: la IA va en trabajos en segundo plano
(ZLECITOOL_JOB_WORKERS por proceso) y el PDF del servidor, en su cola. Medido
con loadtest/ (loadtest/RESULTADOS.md y docs/PLAN.md del núcleo, §10).
"""

import multiprocessing
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2 + 1))
threads = int(os.environ.get("GUNICORN_THREADS", "4"))
worker_class = "gthread"
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "360"))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "2"))
# Reiniciar cada proceso de vez en cuando: una fuga de memoria pequeña no
# llega a tumbar el servidor. Con jitter, para que no se reinicien todos a la vez.
# No muy a menudo: medido (loadtest/), con 2000 a 300 pet/s cada proceso se
# reciclaba cada minuto (37 veces en 90 s): 0,5 s de CPU en arrancar la app,
# Chromium otra vez en el siguiente PDF, y sus conexiones abiertas, cortadas.
max_requests = int(os.environ.get("GUNICORN_MAX_REQUESTS", "10000"))
max_requests_jitter = max_requests // 10
# En un despliegue (SIGTERM), lo que se espera a las peticiones en curso.
graceful_timeout = int(os.environ.get("GUNICORN_GRACEFUL_TIMEOUT", "30"))
accesslog = "-"
