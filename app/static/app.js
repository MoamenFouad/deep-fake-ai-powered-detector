/* DeepShield frontend: vanilla JS, no build step. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
  };

  const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
  const VIDEO_TYPES = ["video/mp4", "video/quicktime", "video/webm", "video/x-msvideo", "video/x-matroska"];
  const MAX_IMAGE = 10 * 1024 * 1024;
  const MAX_VIDEO = 60 * 1024 * 1024;

  const state = { lang: store.get("ds-lang") || "en", file: null, kind: null, url: null, result: null, metrics: null };

  // ------------------------------------------------------------------ i18n
  const t = (key, vars = {}) => {
    const str = (I18N[state.lang] && I18N[state.lang][key]) ?? I18N.en[key] ?? key;
    return str.replace(/\{(\w+)\}/g, (_, k) => (vars[k] ?? ""));
  };
  const locale = () => (state.lang === "ar" ? "ar-EG" : "en-US");
  const pct = (p, digits = 1) => new Intl.NumberFormat(locale(), { style: "percent", maximumFractionDigits: digits }).format(p);
  const num = (n, digits = 1) => new Intl.NumberFormat(locale(), { maximumFractionDigits: digits }).format(n);
  const auc = (n) => new Intl.NumberFormat(locale(), { minimumFractionDigits: 3, maximumFractionDigits: 3 }).format(n);

  function applyLang() {
    const html = document.documentElement;
    html.lang = state.lang;
    html.dir = state.lang === "ar" ? "rtl" : "ltr";
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    document.querySelectorAll("[data-i18n-aria]").forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
    const toggle = $("#langToggle");
    toggle.textContent = state.lang === "ar" ? "English" : "العربية";
    toggle.lang = state.lang === "ar" ? "en" : "ar";
    toggle.setAttribute("aria-label", toggle.textContent);
    if (state.result) renderResult(state.result);
    if (state.metrics) renderMetrics(state.metrics);
    if (state.file && !state.result) setStatus(t("status_ready", { name: state.file.name }));
  }

  $("#langToggle").addEventListener("click", () => {
    state.lang = state.lang === "ar" ? "en" : "ar";
    store.set("ds-lang", state.lang);
    applyLang();
  });

  // ------------------------------------------------------------------ theme
  const savedTheme = store.get("ds-theme");
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  $("#themeToggle").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    store.set("ds-theme", root.dataset.theme);
  });

  // ------------------------------------------------------------------ file intake
  const drop = $("#drop");
  const input = $("#file");
  const status = $("#status");
  const analyzeBtn = $("#analyzeBtn");
  const resetBtn = $("#resetBtn");

  function setStatus(msg, isError = false) {
    status.textContent = msg;
    status.classList.toggle("error", isError);
  }

  function kindOf(file) {
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (IMAGE_TYPES.includes(file.type) || ["jpg", "jpeg", "png", "webp", "bmp"].includes(ext)) return "image";
    if (VIDEO_TYPES.includes(file.type) || ["mp4", "mov", "webm", "avi", "mkv"].includes(ext)) return "video";
    return null;
  }

  function choose(file) {
    if (!file) return;
    const kind = kindOf(file);
    if (!kind) return setStatus(t("err_type"), true);
    if (file.size > (kind === "image" ? MAX_IMAGE : MAX_VIDEO)) return setStatus(t("err_size"), true);

    clearResult();
    if (state.url) URL.revokeObjectURL(state.url);
    Object.assign(state, { file, kind, url: URL.createObjectURL(file) });

    const stage = $("#stage");
    stage.replaceChildren();
    const media = document.createElement(kind === "image" ? "img" : "video");
    media.src = state.url;
    if (kind === "video") { media.muted = true; media.controls = true; media.playsInline = true; }
    else media.alt = "";
    stage.append(media);

    $("#dropEmpty").hidden = true;
    $("#preview").hidden = false;
    drop.classList.add("has-file");
    drop.removeAttribute("role");
    drop.tabIndex = -1;
    analyzeBtn.disabled = false;
    resetBtn.hidden = false;
    setStatus(t("status_ready", { name: file.name }));
  }

  input.addEventListener("change", () => choose(input.files[0]));
  drop.addEventListener("click", () => {
    if (!drop.classList.contains("has-file")) input.click();  // the preview itself stays interactive
  });
  drop.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !drop.classList.contains("has-file")) { e.preventDefault(); input.click(); }
  });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", (e) => choose(e.dataTransfer.files[0]));

  document.querySelectorAll("[data-sample]").forEach((btn) => btn.addEventListener("click", async () => {
    try {
      const res = await fetch(btn.dataset.sample);
      if (!res.ok) throw new Error(res.statusText);
      const blob = await res.blob();
      choose(new File([blob], btn.dataset.sample.split("/").pop(), { type: blob.type || "image/jpeg" }));
    } catch { setStatus(t("err_network"), true); }
  }));

  resetBtn.addEventListener("click", () => {
    if (state.url) URL.revokeObjectURL(state.url);
    Object.assign(state, { file: null, kind: null, url: null });
    input.value = "";
    $("#stage").replaceChildren();
    $("#preview").hidden = true;
    $("#dropEmpty").hidden = false;
    drop.classList.remove("has-file", "busy");
    drop.setAttribute("role", "button");
    drop.tabIndex = 0;
    analyzeBtn.disabled = true;
    resetBtn.hidden = true;
    clearResult();
    setStatus("");
    drop.focus();
  });

  // ------------------------------------------------------------------ analysis
  async function analyze() {
    if (!state.file) return;
    analyzeBtn.disabled = true;
    drop.classList.add("busy");
    setStatus(t(state.kind === "video" ? "status_video" : "status_image"));
    const body = new FormData();
    body.append("file", state.file, state.file.name);
    try {
      const res = await fetch("/api/analyze", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (res.status === 503) {
        setStatus(t("status_loading"));
        setTimeout(analyze, 4000);
        return;
      }
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : res.statusText);
      state.result = data;
      renderResult(data);
      setStatus(t("status_done", { s: num(data.processing_ms / 1000) }));
    } catch (err) {
      setStatus(err instanceof TypeError ? t("err_network") : t("err_generic", { msg: err.message }), true);
    } finally {
      drop.classList.remove("busy");
      analyzeBtn.disabled = !state.file;
    }
  }
  analyzeBtn.addEventListener("click", analyze);

  function clearResult() {
    state.result = null;
    $("#resultBody").hidden = true;
    $("#resultBody").replaceChildren();
    $("#resultEmpty").hidden = false;
    document.querySelectorAll(".box").forEach((b) => b.remove());
  }

  // ------------------------------------------------------------------ rendering
  const toneVar = (v) => ({ fake: "var(--fake)", real: "var(--real)", uncertain: "var(--unsure)" }[v]);
  const el = (tag, attrs = {}, ...kids) => {
    const node = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => {
      if (k === "class") node.className = v;
      else if (k === "style") node.style.cssText = v;
      else node.setAttribute(k, v);
    });
    kids.flat().forEach((k) => k != null && node.append(k));
    return node;
  };

  function renderResult(r) {
    const body = $("#resultBody");
    body.replaceChildren();

    const sub = r.verdict === "uncertain"
      ? t("v_sub_unsure", { p: pct(r.fake_probability) })
      : t("v_sub", { p: pct(r.fake_probability), c: pct(r.confidence, 0) });
    body.append(el("div", { class: "verdict", "data-v": r.verdict },
      el("p", { class: "verdict-word" }, t(`v_${r.verdict}`)),
      el("p", { class: "verdict-sub" }, sub)));

    body.append(el("div", { class: "meter", role: "img", "aria-label": sub },
      el("div", { class: "meter-wrap" },
        el("div", { class: "meter-track" }),
        el("span", { class: "meter-pin", style: `--p:${r.fake_probability}` })),
      el("div", { class: "meter-scale" }, el("span", {}, t("meter_real")), el("span", {}, t("meter_fake")))));

    const engine = { "fusion-head": t("engine_fusion"), "ensemble-mean": t("engine_mean"), stub: t("engine_stub") }[r.engine] || r.engine;
    const facts = r.media_type === "video"
      ? [[t("f_frames"), num(r.frames_analyzed, 0)], [t("f_duration"), r.duration_seconds ? t("seconds", { s: num(r.duration_seconds) }) : "-"]]
      : [[t("f_faces"), num(r.notes.includes("no_face") ? 0 : r.faces.length, 0)]];
    facts.push([t("f_time"), t("seconds", { s: num(r.processing_ms / 1000) })], [t("f_engine"), engine]);
    body.append(el("dl", { class: "facts" }, facts.map(([k, v]) => el("div", {}, el("dt", {}, k), el("dd", {}, v)))));

    if (r.media_type === "video" && r.timeline.length) body.append(renderTimeline(r));

    const faces = el("div", { class: "faces" });
    r.faces.forEach((f, i) => faces.append(renderFace(f, i, r)));
    body.append(faces);

    if (r.notes.length) {
      body.append(el("ul", { class: "notes" }, r.notes.filter((n) => I18N.en[`note_${n}`]).map((n) => el("li", {}, t(`note_${n}`)))));
    }

    const dl = el("a", { class: "btn primary small", href: `/api/report/${r.id}`, download: "" }, t("download"));
    body.append(el("div", { class: "result-actions" }, dl));

    $("#resultEmpty").hidden = true;
    body.hidden = false;
    if (r.media_type === "image") drawBoxes(r);
  }

  function renderFace(f, i, r) {
    const node = $("#faceTpl").content.firstElementChild.cloneNode(true);
    const wipe = $(".wipe", node);
    $(".wipe-a", node).src = `data:image/png;base64,${f.crop_png}`;
    $(".wipe-b", node).src = `data:image/png;base64,${f.heatmap_png || f.crop_png}`;
    const range = $(".wipe-range", node);
    range.setAttribute("aria-label", t("wipe_label"));
    range.addEventListener("input", () => wipe.style.setProperty("--pos", `${range.value}%`));
    wipe.after(el("div", { class: "wipe-caption" }, el("span", {}, t("wipe_a")), el("span", {}, t("wipe_b"))));

    const title = r.media_type === "video" ? t("face_top") : (f.box ? t("face_n", { n: num(i + 1, 0) }) : t("face_whole"));
    $(".face-title", node).textContent = title;
    const score = $(".face-score", node);
    score.textContent = t("face_score", { p: pct(f.fake_probability) });
    score.style.color = toneVar(f.verdict);
    const dl = $(".backbones", node);
    Object.entries(f.model_scores).forEach(([name, p]) => {
      dl.append(el("div", {}, el("dt", {}, name), el("dd", {}, pct(p)),
        el("span", { class: "bar", "aria-hidden": "true" }, el("i", { style: `--p:${p}` }))));
    });
    return node;
  }

  function drawBoxes(r) {
    const img = $("#stage img");
    if (!img) return;
    const place = () => {
      document.querySelectorAll(".box").forEach((b) => b.remove());
      const sx = img.clientWidth / img.naturalWidth;
      const sy = img.clientHeight / img.naturalHeight;
      r.faces.forEach((f, i) => {
        if (!f.box) return;
        const b = el("div", { class: "box", style:
          `left:${f.box.x * sx}px;top:${f.box.y * sy}px;width:${f.box.w * sx}px;height:${f.box.h * sy}px;--box:${toneVar(f.verdict)}` },
        el("span", {}, `${num(i + 1, 0)}: ${pct(f.fake_probability, 0)}`));
        $("#stage").append(b);
      });
    };
    if (img.complete) place(); else img.addEventListener("load", place, { once: true });
    window.addEventListener("resize", place, { passive: true });
  }

  function renderTimeline(r) {
    const W = 560, H = 170, L = 34, R = 10, T = 12, B = 26;
    const pts = r.timeline;
    const tMax = Math.max(pts[pts.length - 1].t, 0.001);
    const rtl = state.lang === "ar";
    const x = (tv) => { const v = L + (tv / tMax) * (W - L - R); return rtl ? W - v + L - R : v; };
    const y = (p) => T + (1 - p) * (H - T - B);
    const ns = "http://www.w3.org/2000/svg";
    const s = (tag, attrs) => { const n = document.createElementNS(ns, tag); Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v)); return n; };

    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("timeline_title") });
    svg.append(s("rect", { class: "band-u", x: L, y: y(0.6), width: W - L - R, height: y(0.4) - y(0.6) }));
    [0, 0.5, 1].forEach((p) => {
      svg.append(s("line", { class: "axis", x1: L, x2: W - R, y1: y(p), y2: y(p) }));
      const lbl = s("text", { x: rtl ? W - R + 2 : L - 6, y: y(p) + 4, "text-anchor": rtl ? "start" : "end" });
      lbl.textContent = pct(p, 0);
      svg.append(lbl);
    });
    [0, tMax].forEach((tv, i) => {
      const lbl = s("text", { x: x(tv), y: H - 6, "text-anchor": i === 0 ? (rtl ? "end" : "start") : (rtl ? "start" : "end") });
      lbl.textContent = t("seconds", { s: num(tv, 0) });
      svg.append(lbl);
    });
    const valid = pts.filter((p) => p.fake_probability != null);
    if (valid.length > 1) {
      svg.append(s("polyline", { class: "line", points: valid.map((p) => `${x(p.t)},${y(p.fake_probability)}`).join(" ") }));
    }
    const tip = el("p", { class: "timeline-tip" }, t("timeline_tip"));
    pts.forEach((p) => {
      const has = p.fake_probability != null;
      const v = has ? p.fake_probability : 0;
      const c = s("circle", { class: "dot", cx: x(p.t), cy: has ? y(v) : y(0), r: has ? 4.5 : 3, tabindex: 0,
        fill: has ? (v >= 0.6 ? "var(--fake)" : v <= 0.4 ? "var(--real)" : "var(--unsure)") : "var(--rule)" });
      const label = has ? t("timeline_point", { t: num(p.t), p: pct(v) }) : t("timeline_noface", { t: num(p.t) });
      c.setAttribute("aria-label", label);
      const show = () => { tip.textContent = label; };
      c.addEventListener("mouseenter", show);
      c.addEventListener("focus", show);
      c.addEventListener("click", show);
      svg.append(c);
    });
    return el("div", { class: "timeline" }, el("p", { class: "h3" }, t("timeline_title")), svg, tip);
  }

  // ------------------------------------------------------------------ metrics table
  function renderMetrics(m) {
    const tbody = $("#metricsTable tbody");
    tbody.replaceChildren();
    if (!m || !m.test) {
      tbody.append(el("tr", {}, el("td", { colspan: 5, class: "muted" }, t("metrics_missing"))));
      return;
    }
    const fusion = m.test.fusion_head;
    const singles = Object.entries(m.test).filter(([k]) => k.startsWith("zero_shot:"));
    const bestSingle = (src) => Math.max(...singles.map(([, v]) => (v[src] && Number.isFinite(v[src].auc) ? v[src].auc : 0)));
    ["ffhq_stylegan", "dff", "celebdf", "dfd"].filter((s) => fusion[s]).forEach((src) => {
      tbody.append(el("tr", {},
        el("th", { scope: "row" }, t(`ds_${src}`)),
        el("td", {}, t(`kind_${src}`)),
        el("td", { class: "num" }, auc(fusion[src].auc)),
        el("td", { class: "num" }, pct(fusion[src].accuracy)),
        el("td", { class: "num" }, auc(bestSingle(src)))));
    });
    tbody.append(el("tr", { class: "total" },
      el("th", { scope: "row" }, t("all_datasets")), el("td", {}, ""),
      el("td", { class: "num" }, auc(fusion.overall.auc)),
      el("td", { class: "num" }, pct(fusion.overall.accuracy)),
      el("td", { class: "num" }, auc(bestSingle("overall")))));
  }

  fetch("/api/metrics").then((r) => (r.ok ? r.json() : null)).then((m) => { state.metrics = m || {}; renderMetrics(state.metrics); })
    .catch(() => { state.metrics = {}; renderMetrics(state.metrics); });

  applyLang();
})();
