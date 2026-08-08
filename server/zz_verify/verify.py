"""Verification pass for the iaCarry checkout screen.

Drives the real template in a real browser against a stub /upload that replays a
recorded response. Every claim it makes is measured, not asserted.

    python3 server/zz_verify/verify.py

What this CANNOT establish: anything about the detector. The TensorFlow model,
its saved-model directory and the evaluation folder all live outside this
repository, so the responses here are replayed, not inferred. This verifies that
the front end reads the contract correctly and behaves correctly around it.

Requires: flask, pillow, playwright (plus a Chromium build).
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from playwright.sync_api import sync_playwright

import make_ground_truth

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(HERE)
FIXTURE = os.path.join(SERVER_DIR, "zz_sample_upload_response.json")
PORT = int(os.environ.get("VERIFY_PORT", "8099"))
BASE = "http://127.0.0.1:%d" % PORT
LANGS = ["es", "en", "eu", "ca", "pt", "fr", "de"]
THEMES = ["iacarry", "eroski", "ahorramas", "condis"]

CHROME = os.environ.get("CHROME_PATH") or next(
    (p for p in ["/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
                 "/opt/pw-browsers/chromium/chrome-linux/chrome"] if os.path.exists(p)), None)

results = []


def check(section, name, ok, detail=None):
    results.append((section, name, bool(ok)))
    print(("  PASS  " if ok else "  FAIL  ") + name
          + ("" if detail is None else "  -- " + str(detail)))


class Stub:
    """The stub Flask app, run as a subprocess so it can be killed mid-request."""

    def __init__(self, static, **env):
        self.static = static
        self.env = env

    def __enter__(self):
        e = dict(os.environ, STATIC=self.static, PORT=str(PORT),
                 **{k: str(v) for k, v in self.env.items()})
        self.p = subprocess.Popen([sys.executable, os.path.join(HERE, "stub_server.py"), SERVER_DIR],
                                  env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):                       # wait for the port to answer
            time.sleep(0.25)
            try:
                import urllib.request
                urllib.request.urlopen(BASE + "/", timeout=1).read()
                return self
            except Exception:
                if self.p.poll() is not None:
                    raise RuntimeError("stub server exited")
        raise RuntimeError("stub server did not start")

    def kill(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGKILL)
            self.p.wait()

    def __exit__(self, *a):
        self.kill()


def new_page(pw, width=1500, height=1150):
    b = pw.chromium.launch(executable_path=CHROME)
    pg = b.new_page(viewport={"width": width, "height": height})
    return b, pg


def detect(pg, demo="demo1", timeout=30000):
    pg.select_option("#demo", demo)
    pg.wait_for_function("window.iaCarry.state.detect.status==='ok'", timeout=timeout)
    pg.wait_for_timeout(250)


# Selectors that must never have content wider or taller than their own box.
CLIP_SELECTOR = (".fact, .fraud, .fraud .li, .sema, .prod, .metric, .badge-count, "
                 ".btn, .step, .pill, .qty, .price, #sSim, #sConfirmed")
CLIP_JS = """() => {
  const bad=[];
  document.querySelectorAll(%r).forEach(e=>{
    if(e.offsetParent===null) return;                 // hidden, nothing to clip
    if(e.scrollWidth>e.clientWidth+1 || e.scrollHeight>e.clientHeight+1)
      bad.push((e.id||e.className)+' '+e.scrollWidth+'/'+e.clientWidth
               +' '+e.scrollHeight+'/'+e.clientHeight);
  });
  if(document.documentElement.scrollWidth>window.innerWidth+1) bad.push('PAGE scrolls horizontally');
  return bad;}""" % CLIP_SELECTOR


def main():
    if not CHROME:
        print("No Chromium found; set CHROME_PATH."); return 2
    tmp = tempfile.mkdtemp(prefix="iacarry-verify-")
    # Two static roots on purpose:
    #   REAL_STATIC — exactly what is committed, so section F verifies the real assets.
    #   GT_STATIC   — the same tree with the demo frames replaced by ground-truth
    #                 images, so box positions can be measured against known truth.
    REAL_STATIC = os.path.join(SERVER_DIR, "static")
    GT_STATIC = os.path.join(tmp, "gt")
    shutil.copytree(REAL_STATIC, GT_STATIC)
    truth = make_ground_truth.main(os.path.join(GT_STATIC, "assets", "demo"), FIXTURE,
                                  os.path.join(SERVER_DIR, "iaCarry_Local_JS_1_Clouding.html"))
    fixture = json.load(open(FIXTURE))
    print()

    with sync_playwright() as pw:

        # -- A. geometry: boxes on products, square and non-square ------------
        print("[A] Detection boxes sit on the products")
        with Stub(GT_STATIC, MODE="ok", DELAY=0):
            b, pg = new_page(pw)
            pg.goto(BASE + "/", wait_until="networkidle")
            for demo, label, expect_aspect in [("demo1", "landscape 1280x720", "1280 / 720"),
                                               ("demo2", "square 800x800", "800 / 800"),
                                               ("demo3", "portrait 640x960", "640 / 960")]:
                detect(pg, demo)
                asp = pg.evaluate("window.iaCarry.state.camAspect")
                geo = pg.evaluate("""() => {
                  const c=document.querySelector('#cam').getBoundingClientRect();
                  return [...document.querySelectorAll('#cam .box')].map(x=>{const r=x.getBoundingClientRect();
                    return {l:(r.left-c.left)/c.width,t:(r.top-c.top)/c.height,
                            w:r.width/c.width,h:r.height/c.height};});}""")
                check("A", "%s: panel aspect from the image" % label, asp == expect_aspect, asp)
                check("A", "%s: box count == drawable detections" % label,
                      len(geo) == len(truth), "%d vs %d" % (len(geo), len(truth)))
                worst = max((abs(g[k] - t[k]) for g, t in zip(geo, truth) for k in "ltwh"), default=1)
                check("A", "%s: every box within 1%% of ground truth" % label, worst < 0.01,
                      "worst %.4f%%" % (worst * 100))
            b.close()

        # -- B. quantities, totals, weight ------------------------------------
        print("\n[B] Quantities, total and weight")
        with Stub(REAL_STATIC, MODE="ok", DELAY=0):
            b, pg = new_page(pw)
            pg.goto(BASE + "/", wait_until="networkidle")
            pg.select_option("#lang", "en")
            detect(pg)
            qty = pg.evaluate("window.iaCarry.state.live.qty")
            # Counted straight from the fixture, above threshold, known classes only.
            expect = {}
            for p in fixture["predictions"]:
                if p["probability"] < 0.45 or p["tagName"] == "pringles_200g":
                    continue
                expect[p["tagName"]] = expect.get(p["tagName"], 0) + 1
            check("B", "quantities == detection counts (repeated products)", qty == expect, qty)
            check("B", "sub-threshold detection dropped",
                  qty.get("cocacola_33cl") == 4, qty.get("cocacola_33cl"))
            check("B", "unknown class skipped, not thrown",
                  pg.evaluate("window.iaCarry.state.live.skipped").count("pringles_200g") == 1)
            check("B", "degenerate box counted but not drawn",
                  qty.get("colacao_383g") == 2
                  and len(pg.evaluate("window.iaCarry.state.live.boxes")) == 10)
            prices = {"cocacola_33cl": .8, "fanta_33cl": .73, "chipsahoy_300g": 3.12, "colacao_383g": 5.6}
            weights = {"cocacola_33cl": .34, "fanta_33cl": .34, "chipsahoy_300g": .3, "colacao_383g": .383}
            hand_total = sum(prices[k] * v for k, v in qty.items())
            hand_kg = sum(weights[k] * v for k, v in qty.items())
            check("B", "total == hand arithmetic",
                  pg.inner_text("#total").strip() == "%.2f€" % hand_total,
                  "%s vs %.2f€" % (pg.inner_text("#total").strip(), hand_total))
            check("B", "units and lines in the header badge",
                  pg.inner_text("#badgeCount").strip() == "%d items · %d ref." % (sum(qty.values()), len(qty)),
                  pg.inner_text("#badgeCount"))
            check("B", "expected weight == hand arithmetic",
                  pg.inner_text("#wVal").strip() == "%.2f kg" % hand_kg,
                  "%s vs %.2f kg" % (pg.inner_text("#wVal").strip(), hand_kg))
            check("B", "confidence is per-product, from its strongest detection",
                  pg.evaluate("window.iaCarry.state.live.conf")
                  == {"cocacola_33cl": 98, "fanta_33cl": 93, "chipsahoy_300g": 80, "colacao_383g": 91})
            b.close()

        # -- C. busy state ----------------------------------------------------
        print("\n[C] Busy state during a slow response")
        with Stub(REAL_STATIC, MODE="ok", DELAY=2):
            b, pg = new_page(pw)
            pg.goto(BASE + "/", wait_until="networkidle")
            pg.select_option("#demo", "demo1")
            pg.wait_for_function("window.iaCarry.state.detect.status==='busy'", timeout=8000)
            check("C", "busy veil visible", pg.is_visible("#cam .busy"))
            check("C", "pay button disabled in flight", pg.is_disabled("#pay"))
            check("C", "demo selector locked in flight", pg.is_disabled("#demo"))
            check("C", "traffic light reports analysing",
                  pg.eval_on_selector("#sema", "e=>getComputedStyle(e).backgroundColor")
                  != "rgb(233, 247, 239)")
            pg.wait_for_function("window.iaCarry.state.detect.status==='ok'", timeout=30000)
            check("C", "busy state clears on completion", not pg.is_visible("#cam .busy"))
            b.close()

        # -- D. red states ----------------------------------------------------
        print("\n[D] Failure states")
        RED = "rgb(253, 236, 236)"
        for mode, label in [("error500", "HTTP 500"), ("texterr", "plain-text error"),
                            ("empty", "model recognised nothing")]:
            with Stub(REAL_STATIC, MODE=mode, DELAY=0):
                b, pg = new_page(pw)
                pg.goto(BASE + "/", wait_until="networkidle")
                pg.select_option("#lang", "en")
                pg.select_option("#demo", "demo1")
                pg.wait_for_function("['error','ok'].includes(window.iaCarry.state.detect.status)", timeout=30000)
                pg.wait_for_timeout(300)
                check("D", "%s: traffic light red" % label,
                      pg.eval_on_selector("#sema", "e=>getComputedStyle(e).backgroundColor") == RED)
                check("D", "%s: payment blocked" % label, pg.is_disabled("#pay"))
                check("D", "%s: tells staff to call an assistant" % label,
                      "assistant" in pg.inner_text("#sema").lower(), pg.inner_text("#sema").strip())
                check("D", "%s: no stale cart under a red light" % label,
                      pg.inner_text("#total").strip() == "0.00€")
                pg.click("#pay", force=True)
                check("D", "%s: forced click cannot pay" % label, not pg.is_visible("#success"))
                b.close()
        # server killed mid-request
        stub = Stub(tmp, MODE="hang", DELAY=0).__enter__()
        b, pg = new_page(pw)
        pg.goto(BASE + "/", wait_until="networkidle")
        pg.select_option("#lang", "en")
        pg.select_option("#demo", "demo1")
        pg.wait_for_function("window.iaCarry.state.detect.status==='busy'", timeout=8000)
        time.sleep(1.0)
        stub.kill()
        pg.wait_for_function("window.iaCarry.state.detect.status==='error'", timeout=30000)
        pg.wait_for_timeout(300)
        check("D", "server killed mid-request: red",
              pg.eval_on_selector("#sema", "e=>getComputedStyle(e).backgroundColor") == RED)
        check("D", "server killed mid-request: payment blocked", pg.is_disabled("#pay"))
        check("D", "server killed mid-request: busy veil cleared", not pg.is_visible("#cam .busy"))
        b.close()

        # -- E. themes x languages --------------------------------------------
        print("\n[E] Four themes x seven languages, and large text")
        with Stub(REAL_STATIC, MODE="ok", DELAY=0):
            for width in (1500, 1280):
                b, pg = new_page(pw, width=width)
                pg.goto(BASE + "/", wait_until="networkidle")
                detect(pg)
                pg.click("#fraudBtn")                      # open the widest panel
                broken = []
                for big in (False, True):
                    if big:
                        pg.click("#big")
                    for theme in THEMES:
                        pg.select_option("#client", theme)
                        for lang in LANGS:
                            pg.select_option("#lang", lang)
                            pg.wait_for_timeout(45)
                            bad = pg.evaluate(CLIP_JS)
                            if bad:
                                broken.append("%s/%s/%s%s: %s"
                                              % (width, theme, lang, "+big" if big else "", bad[:2]))
                    # palette really did swap
                    pri = pg.evaluate("getComputedStyle(document.body).getPropertyValue('--pri').trim()")
                    check("E", "%dpx%s: theme palette applied" % (width, " +big" if big else ""),
                          pri == "#17398A", pri)
                check("E", "%dpx: no clipping across 4 themes x 7 langs x 2 text sizes" % width,
                      not broken, broken[:3])
                b.close()

            # three-column density explicitly, largest text, longest language
            b, pg = new_page(pw, width=1280)
            pg.goto(BASE + "/", wait_until="networkidle")
            detect(pg)
            pg.click("#big")
            worst = []
            for lang in LANGS:
                pg.select_option("#lang", lang)
                pg.wait_for_timeout(60)
                bad = pg.evaluate("""() => [...document.querySelectorAll('#pgrid .prod')]
                    .filter(p=>p.scrollWidth>p.clientWidth+1||p.scrollHeight>p.clientHeight+1)
                    .map(p=>p.querySelector('.pname').textContent)""")
                if bad:
                    worst.append((lang, bad))
            check("E", "large text does not clip in 3-column density", not worst, worst[:2])
            check("E", "density really is 3 columns",
                  pg.evaluate("getComputedStyle(document.querySelector('#pgrid')).gridTemplateColumns")
                  .count("px") == 3)
            b.close()

        # -- F. offline -------------------------------------------------------
        print("\n[F] Renders complete with outbound internet blocked")
        with Stub(REAL_STATIC, MODE="ok", DELAY=0):
            b = pw.chromium.launch(executable_path=CHROME)
            pg = b.new_page(viewport={"width": 1500, "height": 1400})
            outbound = []
            pg.route("**/*", lambda r, q: (r.continue_() if q.url.startswith(BASE)
                                           else (outbound.append(q.url), r.abort())))
            pg.goto(BASE + "/", wait_until="networkidle")
            detect(pg)
            pg.evaluate("document.querySelector('.list').scrollTop=99999")
            pg.wait_for_timeout(1200)
            imgs = pg.evaluate("""() => [...document.querySelectorAll('#pgrid img')]
                .map(i=>({src:i.getAttribute('src'), ok:i.complete&&i.naturalWidth>0}))""")
            check("F", "detection round-trip works offline",
                  pg.inner_text("#total").strip() != "0.00€", pg.inner_text("#total").strip())
            check("F", "every thumbnail rendered", imgs and all(i["ok"] for i in imgs),
                  [i["src"] for i in imgs if not i["ok"]])
            check("F", "thumbnails served locally", all(i["src"].startswith("/static/") for i in imgs))
            for theme in ["eroski", "ahorramas", "condis"]:
                pg.select_option("#client", theme)
                pg.wait_for_timeout(400)
                logos = pg.evaluate("""() => ['#logoBig','#logoSmall'].map(s=>{
                    const i=document.querySelector(s);
                    return {sel:s, src:i.getAttribute('src'),
                            ok:i.complete && i.naturalWidth>0};
                })""")
                check("F", "%s: logo slot shown" % theme,
                      pg.is_visible("#clientMark") and pg.is_visible("#logoSmall"))
                check("F", "%s: logo bytes actually decoded" % theme,
                      all(l["ok"] for l in logos), [l["src"] for l in logos if not l["ok"]])
                check("F", "%s: logo served locally" % theme,
                      all((l["src"] or "").startswith("/static/") for l in logos),
                      [l["src"] for l in logos])
            pg.select_option("#client", "iacarry")
            pg.wait_for_timeout(400)
            check("F", "own brand has no client mark",
                  not pg.is_visible("#clientMark") and not pg.is_visible("#logoSmall"))
            check("F", "no outbound request even attempted", not outbound, sorted(set(outbound))[:3])
            b.close()

        # -- G. backend seams --------------------------------------------------
        print("\n[G] Backend seams are stubs, and say so")
        with Stub(REAL_STATIC, MODE="ok", DELAY=0):
            b, pg = new_page(pw)
            pg.goto(BASE + "/", wait_until="networkidle")
            pg.select_option("#lang", "en")
            detect(pg)
            check("G", "payment backend flagged absent", pg.evaluate("window.iaCarry.PAYMENT_BACKEND") is False)
            check("G", "cart sensor flagged absent", pg.evaluate("window.iaCarry.CART_SENSOR_BACKEND") is False)
            check("G", "sensor toggle marked a stub in the DOM",
                  pg.get_attribute("#sensorToggle", "data-stub") == "cart-sensor")
            pg.click("#pay")
            pg.wait_for_function("window.iaCarry.state.pay.status==='paying'", timeout=5000)
            check("G", "overlay withheld until confirmation", not pg.is_visible("#success"))
            pg.wait_for_function("window.iaCarry.state.pay.status==='done'", timeout=15000)
            pg.wait_for_timeout(300)
            check("G", "approval flagged simulated", pg.evaluate("window.iaCarry.state.pay.simulated") is True)
            note = pg.inner_text("#sSim").lower()
            check("G", "notice names the missing gateway and door",
                  "payment" in note and "door" in note, pg.inner_text("#sSim"))
            check("G", "simulated path does not use the real confirmation wording",
                  "will open" not in pg.inner_text("#sConfirmed").lower(), pg.inner_text("#sConfirmed"))
            b.close()

        # real payment backend: one-line swap
        patched = os.path.join(tmp, "tmpl")
        os.makedirs(patched, exist_ok=True)
        src = open(os.path.join(SERVER_DIR, "iaCarry_Local_JS_1_Clouding.html"), encoding="utf-8").read()
        open(os.path.join(patched, "iaCarry_Local_JS_1_Clouding.html"), "w", encoding="utf-8").write(
            src.replace("const PAYMENT_BACKEND=false;", "const PAYMENT_BACKEND=true;"))
        shutil.copy(FIXTURE, patched)
        for pay_mode, label, expect_done in [("ok", "approved", True), ("decline", "declined", False),
                                             ("http500", "gateway 500", False)]:
            e = dict(os.environ, STATIC=REAL_STATIC, PORT=str(PORT), MODE="ok", DELAY="0", PAY_MODE=pay_mode)
            p = subprocess.Popen([sys.executable, os.path.join(HERE, "stub_server.py"), patched],
                                 env=e, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2.5)
            try:
                b, pg = new_page(pw)
                pg.goto(BASE + "/", wait_until="networkidle")
                pg.select_option("#lang", "en")
                detect(pg)
                pg.click("#pay")
                pg.wait_for_function("['done','error'].includes(window.iaCarry.state.pay.status)", timeout=20000)
                pg.wait_for_timeout(250)
                st = pg.evaluate("window.iaCarry.state.pay")
                check("G", "real backend %s: %s" % (label, "approves" if expect_done else "refuses"),
                      (st["status"] == "done") == expect_done, st)
                if expect_done:
                    check("G", "real backend: not flagged simulated", st["simulated"] is False)
                    check("G", "real backend: demo notice gone", not pg.is_visible("#sSim"))
                else:
                    check("G", "real backend %s: no success overlay" % label, not pg.is_visible("#success"))
                    check("G", "real backend %s: reason kept" % label, bool(st["detail"]), st["detail"])
                b.close()
            finally:
                p.send_signal(signal.SIGKILL); p.wait(); time.sleep(0.5)

    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 68)
    bad = [(s, n) for s, n, ok in results if not ok]
    by = {}
    for s, n, ok in results:
        by.setdefault(s, [0, 0])
        by[s][0] += 1
        by[s][1] += 0 if ok else 1
    for s in sorted(by):
        t, f = by[s]
        print("  section %s: %d checks, %d failed" % (s, t, f))
    print("  TOTAL: %d checks, %d failed" % (len(results), len(bad)))
    if bad:
        print("\n  FAILURES:")
        for s, n in bad:
            print("    [%s] %s" % (s, n))
    print("=" * 68)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
