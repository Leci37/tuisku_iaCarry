/* ───────────────────────────────────────────────────────────────────────────
   station.js — la pantalla de la caja (templates/station.html).

   Es el JavaScript de tuisku_iaCarry/serving/templates/iacarry_checkout.html,
   con la misma forma de trabajar (un `state`, `render()` como único escritor
   del DOM, `detectSeq` contra respuestas fuera de orden, un X-Request-Id por
   foto, el adaptador de las cajas, los estados de ocupado y de fallo, el pago
   sólo con confirmación) y tres cambios por estar sobre el núcleo:

   * los datos llegan del servidor en data-config (la marca del supermercado,
     su catálogo con precios en céntimos y pesos en gramos, las fotos de
     ejemplo, el nombre de la caja), no de constantes de la página;
   * los textos, del diccionario del núcleo (zt.t, claves st*); el idioma se
     cambia con zt.setLanguage() y se vuelve a pintar con "zt:language";
   * cada foto va a /station/detect (o /demo/detect en «Probar»), que la cobra
     a la empresa; el X-Request-Id de la foto es la clave del cobro: un
     reintento tras un corte de red se cobra una vez.
   ─────────────────────────────────────────────────────────────────────────── */
(function () {
  "use strict";
  const ROOT = document.getElementById("station");
  if (!ROOT) return;
  const CONFIG = JSON.parse(ROOT.getAttribute("data-config") || "{}");
  const $ = s => document.querySelector(s);

  /* ── los datos de la empresa ── */
  const THEMES = {};
  (CONFIG.themes || []).forEach(t => { THEMES[t.key] = t; });
  const FIRST_THEME = (CONFIG.themes && CONFIG.themes[0] && CONFIG.themes[0].key) || "iacarry";
  /* El catálogo: precio en céntimos y peso en gramos (enteros, sin redondeos). */
  const CATALOG = (CONFIG.catalogue || []).map(p => ({tag: p.tag, name: p.name, short: p.short, price: p.price,
                                                      weight: p.weight, color: p.color, cat: p.cat}));
  const CATEGORY_ORDER = ["drinks", "snacks", "care", "food", "other"];
  const CATS = ["all"].concat(CATEGORY_ORDER.filter(c => CATALOG.some(p => p.cat === c)));
  const catKey = c => "stCat" + c.charAt(0).toUpperCase() + c.slice(1);
  const PROD_IMG = CONFIG.productImg || "/static/products/";
  const DEMO_SOURCES = {camera: {live: true}};
  (CONFIG.photos || []).forEach(p => { DEMO_SOURCES[p.key] = {n: p.n, file: p.file, src: p.src}; });
  const LOGO_MISSING = new Set();

  const DMAP = {
    comfort: {cols: "repeat(auto-fill,minmax(300px,1fr))", gap: "12px", pad: "13px 14px", cgap: "10px", thumb: 48, img: 40, td: "row", tg: "11px", ta: "center", na: "left", bd: "row", ba: "center"},
    compact: {cols: "repeat(auto-fill,minmax(212px,1fr))", gap: "10px", pad: "10px 12px", cgap: "8px", thumb: 40, img: 32, td: "row", tg: "10px", ta: "center", na: "left", bd: "row", ba: "center"},
    cols3: {cols: "repeat(3,minmax(0,1fr))", gap: "9px", pad: "10px 8px", cgap: "7px", thumb: 44, img: 36, td: "column", tg: "7px", ta: "center", na: "center", bd: "column", ba: "center"}
  };
  const byTag = t => CATALOG.find(x => x.tag === t);
  const shortOf = t => { const c = byTag(t); return c ? (c.short || c.name.split(" ")[0]) : t; };
  const colorOf = t => { const c = byTag(t); return c ? c.color : "var(--pri)"; };

  /* ── los textos ──
     Del diccionario del núcleo. Lo que no es texto (la voz que lee el total,
     la coma de los decimales) va aquí, por idioma. */
  const SPEECH = {es: "es-ES", en: "en-GB", eu: "eu-ES", ca: "ca-ES", gl: "gl-ES", pt: "pt-PT", fr: "fr-FR", de: "de-DE"};
  const lang = () => (window.zt && zt.uiLang) ? zt.uiLang() : (document.documentElement.lang || "es");
  const dec = () => lang() === "en" ? "." : ",";
  const tr = (key, vars) => (window.zt && zt.t) ? zt.t(key, vars) : "";
  const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

  /* Los importes, en céntimos hasta pintarlos: 1234 → «12,34 €» («€12.34» en inglés). */
  function money(cents) {
    const e = Math.floor(cents / 100), c = String(Math.abs(cents) % 100).padStart(2, "0");
    return lang() === "en" ? "€" + e + "." + c : e + "," + c + " €";
  }
  const fmtKg = grams => (grams / 1000).toFixed(2).replace(".", dec()) + " kg";

  /* ── el adaptador de las cajas ──
     El contrato es el de siempre de /upload, que es lo que contesta
     /station/detect: { shape_img:[alto,ancho,c], inference_ms,
     predictions:[{probability, tagName, boundingBox}] }, y boundingBox trae
     mal los nombres desde TensorFlow: `width` es el BORDE DERECHO y `height`
     el de ABAJO (de 0 a 1). Hay que restar, no leer. */
  const DETECTION_MIN_SCORE = CONFIG.minScore || 0.45;
  const pct = v => (v * 100).toFixed(2) + "%";

  function boxFromPrediction(bb) {
    const x1 = +bb.left, y1 = +bb.top, x2 = +bb.width, y2 = +bb.height;
    if (![x1, y1, x2, y2].every(Number.isFinite)) return null;
    const cl = v => Math.max(0, Math.min(1, v));
    const l = cl(x1), t = cl(y1), w = cl(x2) - l, h = cl(y2) - t;
    if (w <= 0 || h <= 0) return null;           /* degenerada: cuenta la unidad, no se dibuja */
    return [pct(l), pct(t), pct(w), pct(h)];
  }

  function adaptDetections(json) {
    const qty = {}, conf = {}, boxes = [], skipped = [], seen = [];
    const preds = (json && Array.isArray(json.predictions)) ? json.predictions : [];
    preds.forEach(p => {
      if (!p || !Number.isFinite(+p.probability) || +p.probability < DETECTION_MIN_SCORE) return;
      const tag = p.tagName;
      /* Una clase que el catálogo no tiene no puede lanzar: una excepción aquí
         deja la caja en blanco, con un comprador delante. */
      if (!byTag(tag)) { skipped.push(tag); return; }
      const box = boxFromPrediction(p.boundingBox || {});
      if (box) boxes.push([...box, tag]);
      qty[tag] = (qty[tag] || 0) + 1;
      const cf = Math.round(+p.probability * 100);
      seen.push(cf);
      if (cf > (conf[tag] || 0)) conf[tag] = cf;
    });
    if (skipped.length) console.warn(LOG + " detecciones fuera, la clase no está en el catálogo:", [...new Set(skipped)]);
    const s = json && json.shape_img;
    const aspect = (Array.isArray(s) && +s[0] > 0 && +s[1] > 0) ? (+s[1] + " / " + (+s[0])) : "1 / 1";
    const meanConf = seen.length ? Math.round(seen.reduce((a, b) => a + b, 0) / seen.length) : null;
    return {qty, conf, boxes, aspect, skipped: [...new Set(skipped)], meanConf,
            inferenceMs: Number.isFinite(+json.inference_ms) ? +json.inference_ms : null};
  }

  function applyDetections(json, rid) {
    const adapted = adaptDetections(json);
    state.live = adapted;
    if (json && json.checkout_id) state.checkoutId = json.checkout_id;
    const units = unitsOf(adapted.qty);
    log(rid || (json && json.request_id), "detecciones", {
      delServidor: (json && Array.isArray(json.predictions)) ? json.predictions.length : 0,
      umbral: DETECTION_MIN_SCORE, unidades: units, lineas: Object.keys(adapted.qty).length,
      cajas: adapted.boxes.length, fueraDelCatalogo: adapted.skipped, totalCentimos: cartTotal(adapted.qty),
      pesoEsperadoG: expectedWeight(adapted.qty), compra: state.checkoutId});
    if (units > 0) setCartPresent(true, "detection");
    render();
    return adapted;
  }

  const confOf = (scene, tag) => (scene.conf && scene.conf[tag] != null) ? scene.conf[tag] : 95;
  const EMPTY_SCENE = {qty: {}, conf: {}, boxes: [], aspect: "1 / 1", skipped: [], meanConf: null, inferenceMs: null};
  const unitsOf = q => Object.values(q || {}).reduce((a, b) => a + b, 0);
  /* El peso esperado: la mitad del antifraude que sí se puede calcular. La otra
     mitad (la báscula) no existe, y la pantalla lo dice. */
  const expectedWeight = q => CATALOG.reduce((s, c) => s + (c.weight || 0) * (q && q[c.tag] || 0), 0);
  const cartTotal = q => CATALOG.reduce((s, c) => s + c.price * (q && q[c.tag] || 0), 0);

  /* ── el transporte ── */
  const UPLOAD_ENDPOINT = CONFIG.detectUrl || "/station/detect";
  const DETECTION_TIMEOUT_MS = 60000;
  let detectSeq = 0;
  const LOG = "[iaCarry]";
  const newRequestId = () => "r" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  function log(rid, msg, data) { const h = LOG + "[" + (rid || "-") + "] " + msg; if (data === undefined) console.info(h); else console.info(h, data); }
  function logWarn(rid, msg, data) { const h = LOG + "[" + (rid || "-") + "] " + msg; if (data === undefined) console.warn(h); else console.warn(h, data); }

  function setDetect(status, phase, detail) { state.detect = {status: status, phase: phase || "", detail: detail || ""}; render(); }
  function failDetection(detail, rid) {
    console.error(LOG + "[" + (rid || "-") + "] DETECCIÓN FALLIDA:", detail);
    state.live = null;                    /* nunca un carro viejo bajo una luz roja */
    setDetect("error", "", detail);
  }
  function resetDetection() { state.live = null; clearCamImage(); setDetect("idle", "", ""); }

  function setCamImage(blob) {
    clearCamImage();
    state.camUrl = URL.createObjectURL(blob);
    const img = $("#camImg");
    img.onload = () => {
      if (img.naturalWidth > 0 && img.naturalHeight > 0) { state.camAspect = img.naturalWidth + " / " + img.naturalHeight; render(); }
    };
    img.src = state.camUrl;
  }
  function clearCamImage() {
    if (state.camUrl) { URL.revokeObjectURL(state.camUrl); state.camUrl = null; }
    state.camAspect = null;
    $("#camImg").removeAttribute("src");
  }

  /* LA costura de la cámara: lo que tenga una foto (una de ejemplo, un fichero,
     la cámara cenital el día que la haya) la da aquí. */
  async function submitFrame(blob, filename, seq, rid) {
    if (seq === undefined) seq = ++detectSeq;
    rid = rid || newRequestId();
    const stale = () => seq !== detectSeq;
    log(rid, "foto al detector", {filename: filename, bytes: blob.size, endpoint: UPLOAD_ENDPOINT});
    setCamImage(blob);
    setDetect("busy", "analysing", "");
    const fd = new FormData();
    fd.append("file", blob, filename);
    if (state.checkoutId) fd.append("checkout", state.checkoutId);
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), DETECTION_TIMEOUT_MS);
    const t0 = performance.now();
    try {
      const res = await fetch(UPLOAD_ENDPOINT, {method: "POST", body: fd, signal: ac.signal, credentials: "same-origin",
                                                headers: {"X-Request-Id": rid}});
      const text = await res.text();
      const ms = +(performance.now() - t0).toFixed(0);
      if (stale()) { log(rid, "respuesta descartada: hay una foto más nueva", {ms: ms}); return null; }
      log(rid, "el detector contestó", {status: res.status, ms: ms});
      let json = null;
      try { json = JSON.parse(text); } catch (e) { /* abajo */ }
      /* Un rechazo del núcleo o del detector es una clave ({"error": "errOrgCreditsNeeded"}):
         la caja se pone en rojo («avisa a un asistente») y el detalle va al título. */
      if (!res.ok) throw new Error((json && json.error) || ("HTTP " + res.status + " " + text.slice(0, 120)));
      if (!json) throw new Error("respuesta vacía o rota");
      if (stale()) return null;
      state.detect = {status: "ok", phase: "", detail: ""};
      return applyDetections(json, rid);
    } catch (err) {
      if (stale()) return null;
      failDetection(err && err.name === "AbortError" ? "sin respuesta en " + (DETECTION_TIMEOUT_MS / 1000) + " s"
                                                     : String(err && err.message || err), rid);
      return null;
    } finally {
      clearTimeout(timer);
    }
  }

  async function runDemo(key) {
    const src = DEMO_SOURCES[key];
    if (!src || src.live) { log("-", "cámara en directo: aún no hay señal, panel limpio"); resetDetection(); return; }
    const seq = ++detectSeq, rid = newRequestId();
    state.live = null; state.pay = {status: "idle", simulated: false, ref: "", detail: ""};
    setDetect("busy", "loading", "");
    let blob;
    try {
      const r = await fetch(src.src, {credentials: "same-origin"});
      if (!r.ok) throw new Error("HTTP " + r.status + " para " + src.src);
      blob = await r.blob();
    } catch (err) {
      if (seq === detectSeq) failDetection("foto de ejemplo no disponible (" + String(err && err.message || err) + ")", rid);
      return;
    }
    if (seq !== detectSeq) { log(rid, "foto descartada: hay otra más nueva"); return; }
    await submitFrame(blob, src.file, seq, rid);
  }

  function payable() {
    return state.detect.status === "ok" && state.cartFull && state.pay.status === "idle" && unitsOf(state.live && state.live.qty) > 0;
  }

  /* ── costura 1 de 2: el pago y la puerta ──
     No hay pasarela ni apertura de puerta: requestPayment() devuelve una
     aprobación simulada, marcada como tal, y la pantalla dice que no se ha
     cobrado nada. Lo que sí se hace es apuntar la compra como pagada
     (simulada) en el servidor, para que se vea en «Compras». */
  const PAYMENT_ENDPOINT = "/payment";      /* SIN IMPLEMENTAR EN EL SERVIDOR */
  const PAYMENT_BACKEND = false;            /* la línea que lo hace de verdad */
  const PAYMENT_TIMEOUT_MS = 30000;

  async function requestPayment(order) {
    if (!PAYMENT_BACKEND) {
      await new Promise(r => setTimeout(r, 900));
      return {ok: true, simulated: true, ref: "DEMO-" + Date.now().toString(36).toUpperCase()};
    }
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), PAYMENT_TIMEOUT_MS);
    try {
      const res = await fetch(PAYMENT_ENDPOINT, {method: "POST", signal: ac.signal, headers: {"Content-Type": "application/json"}, body: JSON.stringify(order)});
      const text = await res.text();
      if (!res.ok) throw new Error("HTTP " + res.status + " " + text.slice(0, 120));
      const json = JSON.parse(text);
      if (!json || json.ok !== true) throw new Error((json && json.error) || "pago rechazado");
      return {ok: true, simulated: false, ref: json.ref || ""};
    } finally { clearTimeout(timer); }
  }

  async function recordPaid(result) {
    if (!state.checkoutId || !CONFIG.paidUrl) return;
    try {
      const res = await fetch(CONFIG.paidUrl.replace("__ID__", state.checkoutId), {method: "POST", credentials: "same-origin",
        headers: {"Content-Type": "application/json"}, body: JSON.stringify({simulated: !!result.simulated, ref: result.ref || ""})});
      if (!res.ok) logWarn("-", "no se pudo apuntar el pago en el servidor", res.status);
    } catch (e) { logWarn("-", "no se pudo apuntar el pago en el servidor", String(e)); }
  }

  async function pay() {
    if (!payable()) return;
    const rid = newRequestId();
    const q = state.live ? state.live.qty : {};
    const order = {totalCents: cartTotal(q), currency: "EUR", units: unitsOf(q), lines: Object.keys(q).length, items: q,
                   checkout: state.checkoutId};
    log(rid, "pago pedido", {modo: PAYMENT_BACKEND ? "REAL" : "SIMULADO (no hay pasarela)", pedido: order});
    state.pay = {status: "paying", simulated: false, ref: "", detail: ""};
    render();
    try {
      const r = await requestPayment(order);
      state.pay = {status: "done", simulated: !!r.simulated, ref: r.ref || "", detail: ""};
      log(rid, r.simulated ? "pago SIMULADO: no se ha cobrado nada" : "pago aprobado", {ref: r.ref});
      recordPaid(r);
    } catch (err) {
      console.error(LOG + "[" + rid + "] PAGO FALLIDO:", err);
      state.pay = {status: "error", simulated: false, ref: "",
                   detail: (err && err.name === "AbortError") ? "sin respuesta" : String(err && err.message || err)};
    }
    render();
  }

  /* ── costura 2 de 2: el sensor del carro ──
     El sensor físico no está conectado: el interruptor Vacío/Lleno lo
     sustituye. La señal de verdad sólo tendrá que llamar a setCartPresent(). */
  const CART_SENSOR_BACKEND = false;
  function setCartPresent(present, source) {
    const was = state.cartFull;
    state.cartFull = !!present;
    if (was !== state.cartFull) log("-", "carro", {presente: state.cartFull, por: source || "?", sensor: CART_SENSOR_BACKEND ? "real" : "sustituto"});
    if (!state.cartFull) {
      /* Sin carro no hay nada que describir: ni la foto, ni las cajas, ni el total. */
      state.pay = {status: "idle", simulated: false, ref: "", detail: ""};
      state.checkoutId = null;
      state.demo = "camera"; fillDemo();
      resetDetection();
    } else render();
  }

  /* `live`: lo adaptado de la última foto (null hasta que una sale bien).
     `detect`: idle | busy | ok | error. `pay`: idle | paying | done | error.
     `cartFull`: sólo lo escribe setCartPresent(). `checkoutId`: la compra de
     este carro en el servidor. */
  const state = {client: FIRST_THEME, demo: "camera", density: "cols3", cat: "all", query: "", big: false, showConf: false,
                 fraudOpen: false, cartFull: false, live: null, checkoutId: null,
                 detect: {status: "idle", phase: "", detail: ""}, camUrl: null, camAspect: null,
                 pay: {status: "idle", simulated: false, ref: "", detail: ""}};

  function fillDemo() {
    const sel = $("#demo");
    if (!sel) return;
    sel.innerHTML = Object.keys(DEMO_SOURCES).map(k => {
      const s = DEMO_SOURCES[k];
      return '<option value="' + k + '">' + esc(s.live ? (tr("stCamLive") || "—") : (tr("stDemoPhoto", {n: s.n}) || s.file)) + "</option>";
    }).join("");
    sel.value = state.demo;
  }

  function buildCats() {
    const host = $("#cats");
    CATS.forEach(key => {
      const b = document.createElement("button");
      b.type = "button"; b.dataset.cat = key;
      b.addEventListener("click", () => { state.cat = key; render(); });
      host.appendChild(b);
    });
  }

  /* Los logos de los supermercados tienen todas las formas: con la misma ÁREA
     (dentro de una caja máxima) se leen del mismo tamaño. */
  const LOGO_FIT = {logoBig: {area: 4400, w: 170, h: 40}, logoSmall: {area: 1900, w: 124, h: 30}};
  function fitLogo(img) {
    const f = LOGO_FIT[img.id], r = img.naturalWidth / img.naturalHeight;
    if (!f || !(r > 0)) return;
    let h = Math.sqrt(f.area / r), w = h * r;
    if (w > f.w) { w = f.w; h = w / r; }
    if (h > f.h) { h = f.h; w = h * r; }
    img.style.width = w.toFixed(1) + "px"; img.style.height = h.toFixed(1) + "px";
  }
  /* La flecha del selector es un SVG en una URL data:, que no lee variables CSS. */
  const chevFor = c => 'url("data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'14\' height=\'14\' fill=\'none\' stroke=\'' + c.replace("#", "%23") + '\' stroke-width=\'2.4\' stroke-linecap=\'round\'><path d=\'M4 6l5 5 5-5\'/></svg>")';

  function applyTheme() {
    const t = THEMES[state.client] || THEMES[FIRST_THEME] || {};
    const r = document.body.style;
    ["pri", "ps", "pt", "bg", "mt", "seg", "sa", "sb"].forEach(k => { if (t[k]) r.setProperty("--" + k, t[k]); });
    r.setProperty("--pri-ink", t.ink || t.pri);
    r.setProperty("--pri-fill", t.grad || t.pri);
    r.setProperty("--chev", chevFor(t.ink || t.pri || "#6c5dc7"));
    /* Sin un logo que cargue, ni hueco: nunca una imagen rota junto a «para». */
    const haveLogo = !!t.logo && !LOGO_MISSING.has(t.logo);
    $("#clientMark").style.display = haveLogo ? "inline-flex" : "none";
    $("#logoSmall").hidden = !haveLogo;
    if (haveLogo) {
      const big = $("#logoBig"), small = $("#logoSmall");
      const miss = () => { LOGO_MISSING.add(t.logo); render(); };
      big.onerror = miss; small.onerror = miss;
      big.onload = () => fitLogo(big); small.onload = () => fitLogo(small);
      if (big.getAttribute("src") !== t.logo) {
        big.style.width = small.style.width = "0px";
        big.src = t.logo; small.src = t.logo; big.alt = small.alt = t.name;
      }
    }
  }

  function setText(sel, text) { const el = $(sel); if (el && text) el.textContent = text; }

  function render() {
    applyTheme();
    const scene = state.live || EMPTY_SCENE, q = state.cartFull ? scene.qty : {};
    const det = state.detect.status;
    const inCart = CATALOG.filter(c => (q[c.tag] || 0) > 0);
    const totalCents = inCart.reduce((s, c) => s + c.price * q[c.tag], 0);
    const count = inCart.reduce((s, c) => s + q[c.tag], 0);
    $("#mCount").textContent = count; $("#fraudN").textContent = count;
    $("#mConf").textContent = (state.cartFull && scene.meanConf != null) ? scene.meanConf + " %" : "—";
    $("#mInf").textContent = (scene.inferenceMs > 0 && state.live) ? (scene.inferenceMs >= 1000 ? (scene.inferenceMs / 1000).toFixed(1).replace(".", dec()) + " s" : scene.inferenceMs + " ms") : "—";
    /* «—» y no «0,00 kg» sin detección: un peso cero ya sería una medida. */
    const grams = expectedWeight(q), hasKg = count > 0, kgTxt = hasKg ? fmtKg(grams) : "—";
    $("#wVal").textContent = kgTxt;
    setText("#fLi2", tr("stFraudLi2", {w: kgTxt}));
    setText("#verWTxt", hasKg ? tr("stVerW", {w: kgTxt}) : tr("stVerWNone"));
    $("#verifPill").style.display = hasKg ? "" : "none";
    setText("#badgeCount", tr("stBadge", {n: count, t: inCart.length}));
    $("#total").textContent = money(totalCents);
    const full = state.cartFull, paid = state.pay.status === "done";
    const done = b => { b.style.background = "#E9F7EF"; b.style.color = "#12864a"; b.style.boxShadow = "none"; b.textContent = "✓"; };
    const act = (b, t) => { b.style.background = "var(--pri-fill)"; b.style.color = "#fff"; b.style.boxShadow = "0 0 0 4px var(--ps)"; b.textContent = t; };
    const idle = (b, t) => { b.style.background = "#ECEEF2"; b.style.color = "var(--ter)"; b.style.boxShadow = "none"; b.textContent = t; };
    full ? done($("#s1")) : act($("#s1"), "1");
    paid ? done($("#s2")) : (full ? act($("#s2"), "2") : idle($("#s2"), "2"));
    paid ? done($("#s3")) : idle($("#s3"), "3");
    $("#s1b").style.color = full ? "#12864a" : "var(--ink)";
    $("#s2b").style.color = paid ? "#12864a" : (full ? "var(--ink)" : "var(--ter)");
    $("#s3b").style.color = paid ? "#12864a" : "var(--ter)";
    $("#btnEmpty").classList.toggle("on", !full); $("#btnFull").classList.toggle("on", full);
    const hint = tr("stSensorHint");
    if (hint) { $("#btnEmpty").title = hint; $("#btnFull").title = hint; }
    [["#dCom", "stDensityComfort"], ["#dCompact", "stDensityCompact"], ["#dCols", "stDensityCols"]].forEach(([s, k]) => {
      const v = tr(k); if (v) { $(s).title = v; $(s).setAttribute("aria-label", v); }
    });
    /* El semáforo lleva el estado de la detección, no sólo el del sensor: un
       carro vacío tras un fallo nunca se puede leer como «no debes nada». */
    const sema = $("#sema"), dot = sema.querySelector(".dot");
    const light = (bg, bd, fg, dc, glow, key) => {
      sema.style.background = bg; sema.style.border = "1px solid " + bd; sema.style.color = fg;
      dot.style.background = dc; dot.style.boxShadow = "0 0 0 4px " + glow;
      setText("#semaTxt", tr(key));
      sema.dataset.state = key;
    };
    const detUnits = unitsOf(state.live && state.live.qty);
    const RED = ["#FDECEC", "#F6C9C9", "#B3261E", "#E5484D", "rgba(229,72,77,.18)"];
    const TINT = ["var(--ps)", "var(--sb)", "var(--pri-ink)", "var(--pri)", "var(--sb)"];
    if (state.pay.status === "error") light(...RED, "stPayFail");
    else if (state.pay.status === "paying") light(...TINT, "stPaying");
    else if (det === "busy") light(...TINT, "stSemaBusy");
    else if (det === "error") light(...RED, "stSemaFail");
    else if (det === "ok" && detUnits === 0) light(...RED, "stSemaNone");
    else if (det === "ok" && full && count > 0) light("#E9F7EF", "#C6EBD5", "#12864a", "#23b364", "rgba(35,179,100,.18)", "stSemaReady");
    else light("#F5F6F8", "#E7E9EE", "#6B7080", "#B9BDC8", "rgba(185,189,200,.25)", "stSemaPlace");
    sema.title = state.pay.detail || state.detect.detail || "";
    const canPay = payable();
    setText("#payBtnTxt", tr(state.pay.status === "paying" ? "stPaying" : "stPayBtn"));
    $("#pay").disabled = !canPay;
    if ($("#demo")) $("#demo").disabled = (det === "busy");
    $("#success").hidden = !paid;
    if (paid) {
      /* Una aprobación simulada nunca usa la frase de la de verdad: promete un
         cobro y una puerta que no han pasado. */
      setText("#sConfirmed", state.pay.simulated ? tr("stSimConfirmed") : tr("stConfirmed", {t: money(totalCents)}));
      setText("#sTicket", tr("stTicket", {n: count}));
      $("#sSim").hidden = !state.pay.simulated;
      setText("#sSim", state.pay.simulated ? tr("stSimNote") : "");
    }
    const cam = $("#cam");
    cam.classList.toggle("has-img", !!state.camUrl);
    cam.classList.toggle("busy-on", det === "busy");
    cam.style.aspectRatio = state.camAspect || scene.aspect || "1 / 1";
    setText("#busyMsg", tr(state.detect.phase === "loading" ? "stCamLoading" : "stCamAnalysing"));
    setText("#camPh", tr("stCamIdle"));
    cam.querySelectorAll(".box").forEach(e => e.remove());
    (full ? scene.boxes : []).forEach(([l, tp, w, h, tag]) => {
      const c = colorOf(tag), d = document.createElement("div");
      d.className = "box"; d.dataset.tag = tag;
      d.style.cssText = "left:" + l + ";top:" + tp + ";width:" + w + ";height:" + h + ";border-color:" + c;
      const label = document.createElement("b");
      label.style.background = c; label.textContent = shortOf(tag);
      d.appendChild(label); cam.appendChild(d);
    });
    $("#seg").querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.d === state.density));
    $("#cats").querySelectorAll("button").forEach(b => {
      b.classList.toggle("on", b.dataset.cat === state.cat);
      const v = tr(catKey(b.dataset.cat)); if (v) b.textContent = v;
    });
    $("#conf").classList.toggle("on", state.showConf); $("#confCb").textContent = state.showConf ? "✓" : "";
    const D = DMAP[state.density], g = $("#pgrid");
    g.style.gridTemplateColumns = D.cols; g.style.gap = D.gap;
    const query = state.query.trim().toLowerCase();
    const rows = inCart.filter(c => state.cat === "all" || c.cat === state.cat).filter(c => !query || c.name.toLowerCase().includes(query));
    if (!rows.length) {
      g.innerHTML = '<div class="empty"><svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="9" cy="20" r="1"/><circle cx="18" cy="20" r="1"/><path d="M2 3h3l2.5 12h10L20 7H6"/></svg><div style="font-weight:700;font-size:15px">' + esc(tr(full ? "stEmptyMatch" : "stEmptyCart")) + "</div></div>";
      return;
    }
    g.innerHTML = rows.map(c => {
      const u = q[c.tag], cf = confOf(scene, c.tag), ok = cf >= 85;
      const conf = state.showConf ? '<span class="cbadge" style="color:' + (ok ? "#12864a" : "#9a6a00") + ";background:" + (ok ? "#E9F7EF" : "#FCF3E0") + '">' + esc(tr(ok ? "stRecognised" : "stDoubt")) + " · " + cf + "%</span>" : "";
      return '<div class="prod" data-tag="' + esc(c.tag) + '" style="padding:' + D.pad + ";gap:" + D.cgap + '"><span class="strip" style="background:' + esc(c.color) + '"></span>' +
        '<div class="pmeta" style="flex-direction:' + D.td + ";gap:" + D.tg + ";align-items:" + D.ta + '">' +
        '<div class="thumb" style="width:' + D.thumb + "px;height:" + D.thumb + 'px"><img src="' + PROD_IMG + esc(c.tag) + '_300.png" alt="" style="height:' + D.img + 'px"></div>' +
        '<div style="flex:1;min-width:0;width:100%;align-self:stretch;text-align:' + D.na + '"><div class="pname" style="font-size:' + (state.big ? 17 : (state.density === "cols3" ? 13 : 14)) + 'px">' + esc(c.name) + "</div>" +
        '<div class="pcost tabnum">' + money(c.price) + " " + esc(tr("stPerUnit")) + "</div>" + conf + "</div></div>" +
        '<div class="prow" style="flex-direction:' + D.bd + ";align-items:" + D.ba + ';gap:6px">' +
        '<span class="qty tabnum" style="font-size:' + (state.big ? 15 : 13) + 'px"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>× ' + u + "</span>" +
        '<div class="price tabnum" style="font-size:' + (state.big ? 19 : 15) + 'px">' + money(c.price * u) + "</div></div></div>";
    }).join("");
  }

  /* ── los eventos ── todo pasa por las dos costuras. */
  const clearFilters = () => { state.cat = "all"; state.query = ""; $("#search").value = ""; };
  $("#btnEmpty").addEventListener("click", () => { clearFilters(); setCartPresent(false, "interruptor (sustituye al sensor)"); });
  $("#btnFull").addEventListener("click", () => { state.pay = {status: "idle", simulated: false, ref: "", detail: ""}; setCartPresent(true, "interruptor (sustituye al sensor)"); });
  $("#pay").addEventListener("click", pay);
  $("#reset").addEventListener("click", () => { clearFilters(); setCartPresent(false, "compra nueva"); });
  $("#client").addEventListener("change", e => { state.client = e.target.value; clearFilters(); render(); });
  $("#lang").addEventListener("change", e => {
    const code = e.target.value;
    if (window.zt && zt.setLanguage) zt.setLanguage(code).catch(() => { e.target.value = lang(); });
  });
  document.addEventListener("zt:language", () => { $("#lang").value = lang(); fillDemo(); render(); });
  if ($("#demo")) $("#demo").addEventListener("change", e => { state.demo = e.target.value; runDemo(state.demo); });
  const upload = $("#uploadFrame");
  if (upload) upload.addEventListener("change", e => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    state.pay = {status: "idle", simulated: false, ref: "", detail: ""};
    submitFrame(file, file.name);
    e.target.value = "";
  });
  $("#seg").addEventListener("click", e => { const b = e.target.closest("button"); if (b) { state.density = b.dataset.d; render(); } });
  $("#search").addEventListener("input", e => { state.query = e.target.value; render(); });
  $("#conf").addEventListener("click", () => { state.showConf = !state.showConf; render(); });
  $("#fraudBtn").addEventListener("click", () => {
    state.fraudOpen = !state.fraudOpen;
    $("#fraudPanel").hidden = !state.fraudOpen;
    $("#fraudChev").style.transform = state.fraudOpen ? "rotate(180deg)" : "none";
  });
  $("#big").addEventListener("click", () => { state.big = !state.big; document.body.classList.toggle("big", state.big); $("#big").classList.toggle("on", state.big); render(); });
  $("#listen").addEventListener("click", () => {
    const sy = window.speechSynthesis; if (!sy) return;
    const cents = cartTotal(state.cartFull && state.live ? state.live.qty : {});
    const u = new SpeechSynthesisUtterance(tr("stSpeak", {a: Math.floor(cents / 100), b: cents % 100}));
    u.lang = SPEECH[lang()] || "es-ES"; u.rate = .95; sy.cancel(); sy.speak(u);
  });
  /* Una miniatura que no carga no deja un icono roto: se esconde (antes era un
     onerror= en línea, que la CSP del núcleo bloquea). */
  $("#pgrid").addEventListener("error", e => { if (e.target && e.target.tagName === "IMG") e.target.style.visibility = "hidden"; }, true);

  /* La superficie pública: submitFrame() es LA costura de la cámara; lo demás,
     para las pruebas y para lo que falta por conectar. */
  window.iaCarry = {submitFrame, runDemo, resetDetection, applyDetections, adaptDetections, payable, state, render,
                    DETECTION_MIN_SCORE, UPLOAD_ENDPOINT, DETECTION_TIMEOUT_MS, DEMO_SOURCES, CATALOG, money,
                    pay, requestPayment, setCartPresent, PAYMENT_ENDPOINT, PAYMENT_BACKEND, CART_SENSOR_BACKEND};

  buildCats();
  const ready = (window.zt && zt.i18nReady) ? zt.i18nReady : Promise.resolve();
  ready.then(() => {
    fillDemo();
    render();
    ROOT.dataset.ready = "1";
    console.info(LOG + " lista — detección:" + UPLOAD_ENDPOINT + " | pago:" + (PAYMENT_BACKEND ? "REAL" : "SIMULADO") +
                 " | sensor del carro:" + (CART_SENSOR_BACKEND ? "REAL" : "interruptor") + " | cámara: ninguna",
                 {productos: CATALOG.length, temas: Object.keys(THEMES).length, idioma: lang(), modo: CONFIG.mode});
  });
})();
