/* DeepShield frontend: vanilla JS, no build step. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
  const VIDEO_TYPES = ["video/mp4", "video/quicktime", "video/webm", "video/x-msvideo", "video/x-matroska"];
  const MAX_IMAGE = 10 * 1024 * 1024;
  const MAX_VIDEO = 60 * 1024 * 1024;
  const HISTORY_SIZE = 8;

  const VERDICT = { fake: "Likely manipulated", real: "Likely authentic", uncertain: "Inconclusive" };
  const KICKER = { fake: "Manipulation detected", real: "No manipulation found", uncertain: "Too close to call" };
  const NOTES = {
    no_face: "No face was found, so the whole image was checked. Results are less reliable this way.",
    multiple_faces: "Several faces were found. The verdict follows the most suspicious face.",
    partial_faces: "A face was visible in only some frames. Frames without a face were skipped.",
  };
  const KINDS = {
    ffhq: "FFHQ photo", stylegan: "StyleGAN face", imdb_wiki: "Celebrity photo",
    sd_text2img: "Stable Diffusion image", sd_inpainting: "Diffusion edit", insightface: "InsightFace swap",
    celebdf_real: "Celeb-DF real", celebdf_swap: "Celeb-DF swap", dfd_real: "DFD real", dfd_swap: "DFD swap",
  };
  const ENGINE = { "fusion-head": "Fusion head", "ensemble-mean": "Two-model average", stub: "Test stub" };

  const state = { file: null, kind: null, url: null, result: null, busy: false, history: [], current: null };

  const pct = (p, d = 1) => new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: d }).format(p);
  const num = (n, d = 1) => new Intl.NumberFormat("en-US", { maximumFractionDigits: d }).format(n);
  const fixed = (n, d = 3) => n.toFixed(d);
  const tone = (v) => ({ fake: "var(--fake)", real: "var(--real)", uncertain: "var(--unsure)" }[v] || "var(--ink-3)");
  const verdictOf = (p) => (p >= 0.6 ? "fake" : p <= 0.4 ? "real" : "uncertain");

  const el = (tag, attrs = {}, ...kids) => {
    const node = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => {
      if (v == null || v === false) return;
      if (k === "class") node.className = v;
      else if (k === "style") node.style.cssText = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    });
    kids.flat().forEach((k) => k != null && k !== false && node.append(k));
    return node;
  };
  const tag = (n, small = false) => el("span", { class: small ? "tag sm" : "tag", "aria-hidden": "true" }, String(n));
  const icon = (d) => {
    const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s.setAttribute("viewBox", "0 0 24 24");
    s.setAttribute("aria-hidden", "true");
    s.innerHTML = d;
    return s;
  };
  const ICONS = {
    pdf: '<path d="M7 3h7l5 5v13H7Z" fill="none" stroke="currentColor" stroke-width="2"/><path d="M14 3v5h5M10 13h6M10 17h6" fill="none" stroke="currentColor" stroke-width="2"/>',
    copy: '<path d="M8 8h12v12H8Z" fill="none" stroke="currentColor" stroke-width="2"/><path d="M16 8V4H4v12h4" fill="none" stroke="currentColor" stroke-width="2"/>',
    again: '<path d="M4 12a8 8 0 1 0 2.4-5.7M4 4v4.5h4.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
  };

  // ------------------------------------------------------------------ toast
  let toastTimer;
  function toast(msg) {
    const node = $("#toast");
    node.textContent = msg;
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { node.hidden = true; }, 2200);
  }

  if (/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)) $("#pasteKey").textContent = "⌘ V";

  // ------------------------------------------------------------------ intake
  const drop = $("#drop");
  const input = $("#file");
  const statusEl = $("#status");
  const resetBtn = $("#resetBtn");

  function setStatus(msg, isError = false) {
    statusEl.textContent = msg;
    statusEl.classList.toggle("error", isError);
  }

  function kindOf(file) {
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (IMAGE_TYPES.includes(file.type) || ["jpg", "jpeg", "png", "webp", "bmp"].includes(ext)) return "image";
    if (VIDEO_TYPES.includes(file.type) || ["mp4", "mov", "webm", "avi", "mkv"].includes(ext)) return "video";
    return null;
  }

  function showMedia(kind, url) {
    const stage = $("#stage");
    stage.replaceChildren();
    const media = document.createElement(kind === "image" ? "img" : "video");
    media.src = url;
    if (kind === "video") Object.assign(media, { muted: true, controls: true, playsInline: true, preload: "metadata" });
    else media.alt = "";
    stage.append(media);
    $("#dropEmpty").hidden = true;
    $("#preview").hidden = false;
    drop.classList.add("has-file");
    drop.removeAttribute("role");
    drop.removeAttribute("aria-label");
    drop.tabIndex = -1;
    resetBtn.hidden = false;
  }

  function releaseCurrentUrl() {
    if (state.url && !state.history.some((h) => h.url === state.url)) URL.revokeObjectURL(state.url);
  }

  function choose(file) {
    if (!file || state.busy) return;
    const kind = kindOf(file);
    if (!kind) return setStatus("This file type is not supported. Use JPG, PNG, WEBP, MP4, MOV, WEBM or AVI.", true);
    if (file.size > (kind === "image" ? MAX_IMAGE : MAX_VIDEO)) {
      return setStatus("This file is too large. Photos can be up to 10 MB and videos up to 60 MB.", true);
    }
    clearResult();
    releaseCurrentUrl();
    Object.assign(state, { file, kind, url: URL.createObjectURL(file), current: null });
    showMedia(kind, state.url);
    analyze();
  }

  input.addEventListener("change", () => { choose(input.files[0]); input.value = ""; });
  drop.addEventListener("click", () => { if (!drop.classList.contains("has-file")) input.click(); });
  drop.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !drop.classList.contains("has-file")) { e.preventDefault(); input.click(); }
  });

  let dragDepth = 0;
  const hasFiles = (e) => Array.from(e.dataTransfer?.types || []).includes("Files");
  window.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth++;
    $("#dropOverlay").hidden = false;
    drop.classList.add("dragging");
  });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) { $("#dropOverlay").hidden = true; drop.classList.remove("dragging"); }
  });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth = 0;
    $("#dropOverlay").hidden = true;
    drop.classList.remove("dragging");
    choose(e.dataTransfer.files[0]);
    $("#analyze").scrollIntoView({ block: "start" });
  });

  window.addEventListener("paste", (e) => {
    const item = Array.from(e.clipboardData?.items || []).find((i) => i.kind === "file");
    const blob = item && item.getAsFile();
    if (!blob) return;
    e.preventDefault();
    const ext = (blob.type.split("/")[1] || "png").replace("jpeg", "jpg");
    choose(new File([blob], blob.name && blob.name !== "image.png" ? blob.name : `pasted.${ext}`, { type: blob.type }));
  });

  $$("[data-sample]").forEach((btn) => btn.addEventListener("click", async () => {
    try {
      const res = await fetch(btn.dataset.sample);
      if (!res.ok) throw new Error(res.statusText);
      const blob = await res.blob();
      choose(new File([blob], btn.dataset.sample.split("/").pop(), { type: blob.type || "image/jpeg" }));
    } catch { setStatus("The server could not be reached. Check your connection and try again.", true); }
  }));

  function resetIntake() {
    if (state.busy) return;
    releaseCurrentUrl();
    Object.assign(state, { file: null, kind: null, url: null, current: null });
    $("#stage").replaceChildren();
    $("#preview").hidden = true;
    $("#dropEmpty").hidden = false;
    drop.classList.remove("has-file", "busy");
    drop.setAttribute("role", "button");
    drop.setAttribute("aria-label", "Choose a photo or video to check");
    drop.tabIndex = 0;
    resetBtn.hidden = true;
    clearResult();
    setStatus("");
    renderHistory();
  }
  resetBtn.addEventListener("click", () => { resetIntake(); drop.focus(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && state.file && !state.busy && !e.target.closest("input, textarea")) resetIntake();
  });

  // ------------------------------------------------------------------ analysis
  async function analyze() {
    if (!state.file) return;
    state.busy = true;
    drop.classList.add("busy");
    $("#scanLabel").textContent = state.kind === "video" ? "Scanning 24 frames" : "Scanning";
    setStatus(state.kind === "video" ? "Checking the video. This can take up to a minute." : "Checking the photo. This takes a few seconds.");
    const body = new FormData();
    body.append("file", state.file, state.file.name);
    try {
      const res = await fetch("/api/analyze", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (res.status === 503) {
        setStatus("The models are still warming up. Trying again in a moment.");
        setTimeout(analyze, 4000);
        return;
      }
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : res.statusText);
      state.result = data;
      addHistory(data);
      renderResult(data);
      setStatus(`${state.file.name}, checked in ${num(data.processing_ms / 1000)} s`);
      if (matchMedia("(max-width: 1020px)").matches) $("#result").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setStatus(err instanceof TypeError
        ? "The server could not be reached. Check your connection and try again."
        : `The file could not be analyzed: ${err.message}`, true);
    } finally {
      state.busy = false;
      drop.classList.remove("busy");
    }
  }

  function clearResult() {
    state.result = null;
    $("#resultBody").hidden = true;
    $("#resultBody").replaceChildren();
    $("#resultEmpty").hidden = false;
    $$(".box").forEach((b) => b.remove());
  }

  // ------------------------------------------------------------------ history (this tab only)
  function addHistory(r) {
    const thumb = r.faces[0] && r.faces[0].crop_png ? `data:image/png;base64,${r.faces[0].crop_png}` : null;
    state.history.unshift({ id: r.id, name: state.file.name, kind: state.kind, url: state.url, file: state.file, result: r, thumb });
    while (state.history.length > HISTORY_SIZE) {
      const old = state.history.pop();
      if (old.url !== state.url) URL.revokeObjectURL(old.url);
    }
    state.current = r.id;
    renderHistory();
  }

  function openHistory(id) {
    const h = state.history.find((x) => x.id === id);
    if (!h || state.busy) return;
    releaseCurrentUrl();
    Object.assign(state, { file: h.file, kind: h.kind, url: h.url, result: h.result, current: h.id });
    showMedia(h.kind, h.url);
    renderResult(h.result);
    renderHistory();
    setStatus(`${h.name}, checked in ${num(h.result.processing_ms / 1000)} s`);
  }

  function renderHistory() {
    const list = $("#recentList");
    list.replaceChildren();
    $("#recent").hidden = state.history.length < 2;
    state.history.forEach((h) => {
      const r = h.result;
      const label = `${h.name}: ${VERDICT[r.verdict]}, ${pct(r.fake_probability, 0)}`;
      list.append(el("li", {}, el("button", {
        class: `recent-item${h.id === state.current ? " on" : ""}`, type: "button", title: label, "aria-label": label,
        onclick: () => openHistory(h.id),
      }, h.thumb ? el("img", { src: h.thumb, alt: "" }) : el("span", { class: "kind" }, h.kind === "video" ? "VIDEO" : "IMAGE"),
      el("i", { style: `background:${tone(r.verdict)}` }))));
    });
  }

  // ------------------------------------------------------------------ result
  function strength(p) {
    if (p >= 0.85) return "strong";
    if (p >= 0.6) return "some";
    return "very few";
  }

  function renderResult(r) {
    const body = $("#resultBody");
    body.replaceChildren();

    const sub = r.verdict === "uncertain"
      ? "The evidence sits between 40% and 60%, so DeepShield will not guess. Check the source of the file."
      : `${pct(r.confidence, 0)} confidence. The face shows ${strength(r.fake_probability)} signs of AI manipulation.`;
    body.append(el("section", { class: "verdict", "data-v": r.verdict, "aria-label": VERDICT[r.verdict] },
      el("div", { class: "verdict-row" },
        el("div", {},
          el("p", { class: "verdict-kicker" }, tag(1), KICKER[r.verdict]),
          el("h2", { class: "verdict-word" }, VERDICT[r.verdict])),
        el("p", { class: "verdict-num" }, el("b", {}, pct(r.fake_probability)), el("span", {}, "manipulation probability"))),
      el("div", { role: "img", "aria-label": `${pct(r.fake_probability)} manipulation probability on a scale from real to fake` },
        el("div", { class: "gauge" },
          el("span", { class: "gauge-track" }), el("span", { class: "gauge-band" }),
          el("span", { class: "gauge-pin", style: `--p:${r.fake_probability}` })),
        el("div", { class: "gauge-scale" }, el("span", {}, "Real"), el("span", {}, "Unsure"), el("span", {}, "Fake"))),
      el("p", { class: "verdict-sub" }, sub)));

    const facts = r.media_type === "video"
      ? [["Frames", num(r.frames_analyzed, 0)], ["Duration", r.duration_seconds ? `${num(r.duration_seconds)} s` : "-"]]
      : [["Faces", num(r.notes.includes("no_face") ? 0 : r.faces.length, 0)]];
    facts.push(["Time", `${num(r.processing_ms / 1000)} s`], ["Engine", ENGINE[r.engine] || r.engine]);
    body.append(el("dl", { class: "facts" }, facts.map(([k, v]) => el("div", {}, el("dt", {}, k), el("dd", {}, v)))));

    if (r.media_type === "video" && r.timeline.length) body.append(renderTimeline(r));

    body.append(el("div", { class: "faces" }, r.faces.map((f, i) => renderFace(f, i, r))));

    const notes = r.notes.filter((n) => NOTES[n]);
    if (notes.length) body.append(el("ul", { class: "notes" }, notes.map((n) => el("li", {}, NOTES[n]))));

    body.append(el("div", { class: "actions" },
      el("a", { class: "btn ink", href: `/api/report/${r.id}`, download: "" }, icon(ICONS.pdf), "Download PDF report"),
      el("button", { class: "btn line", type: "button", onclick: () => copySummary(r) }, icon(ICONS.copy), "Copy summary"),
      el("button", { class: "btn line", type: "button", onclick: () => { resetIntake(); input.click(); } }, icon(ICONS.again), "Check another file")));

    $("#resultEmpty").hidden = true;
    body.hidden = false;
    $$(".box").forEach((b) => b.remove());
    if (r.media_type === "image") drawBoxes(r);
  }

  async function copySummary(r) {
    const name = (state.history.find((h) => h.id === r.id) || {}).name || r.filename;
    const text = `DeepShield: ${VERDICT[r.verdict]} (${pct(r.fake_probability)} manipulation probability, `
      + `${pct(r.confidence, 0)} confidence). File: ${name}.`;
    try {
      await navigator.clipboard.writeText(text);
      toast("Summary copied");
    } catch { toast("Copy failed. Select the text and copy it yourself."); }
  }

  function renderFace(f, i, r) {
    const node = $("#faceTpl").content.firstElementChild.cloneNode(true);
    const wipe = $(".wipe", node);
    $(".wipe-a", node).src = `data:image/png;base64,${f.crop_png}`;
    $(".wipe-b", node).src = `data:image/png;base64,${f.heatmap_png || f.crop_png}`;
    const range = $(".wipe-range", node);
    range.addEventListener("input", () => wipe.style.setProperty("--pos", `${range.value}%`));

    $(".face-title", node).textContent = r.media_type === "video" ? "Most suspicious frame" : (f.box ? `Face ${i + 1}` : "Whole image");
    const score = $(".face-score", node);
    score.textContent = pct(f.fake_probability);
    score.style.color = tone(f.verdict);

    // Detector agreement
    const scores = Object.entries(f.model_scores);
    const dl = $(".backbones", node);
    scores.forEach(([name, p]) => dl.append(el("div", {},
      el("dt", { title: name }, name), el("dd", {}, pct(p)),
      el("span", { class: "bar", "aria-hidden": "true" }, el("i", { style: `--p:${p};--tone:${tone(verdictOf(p))}` })))));
    const sides = scores.map(([, p]) => p >= 0.5);
    const fused = f.fake_probability >= 0.5;
    let agree = "";
    if (scores.length >= 2) {
      if (!sides.every((s) => s === sides[0])) agree = "The two detectors disagree. The fusion layer weighs their evidence.";
      else if (sides[0] !== fused && f.verdict !== "uncertain") {
        agree = fused
          ? "On their raw scores both lean real, but the fusion layer reads their internal features as manipulated."
          : "On their raw scores both lean fake, but the fusion layer reads their internal features as authentic.";
      } else agree = sides[0] ? "Both detectors lean fake." : "Both detectors lean real.";
    }
    $(".agree", node).textContent = agree;

    // Closest known cases
    if (f.similar && f.similar.length) {
      const box = $(".exhibit.similar", node);
      box.hidden = false;
      const groups = [];
      f.similar.forEach((s) => {
        const g = groups.find((x) => x.kind === s.kind && x.label === s.label);
        if (g) g.sims.push(s.similarity); else groups.push({ kind: s.kind, label: s.label, sims: [s.similarity] });
      });
      const ul = $(".neighbors", box);
      groups.forEach((g) => ul.append(el("li", { title: `Similarity ${g.sims.map((v) => fixed(v, 2)).join(", ")}` },
        el("i", { style: `background:${g.label === "fake" ? "var(--fake)" : "var(--real)"}`, "aria-hidden": "true" }),
        el("span", {}, `${KINDS[g.kind] || g.kind} (${g.label})`),
        el("b", {}, `×${g.sims.length}`))));
      const k = f.similar.length;
      const fakes = f.similar.filter((s) => s.label === "fake");
      const side = fakes.length * 2 >= k ? fakes : f.similar.filter((s) => s.label === "real");
      const counts = {};
      side.forEach((s) => { counts[s.kind] = (counts[s.kind] || 0) + 1; });
      const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
      $(".sim-note", box).textContent = `${side.length} of the ${k} most similar known faces are ${side === fakes ? "fakes" : "real"}, `
        + `mostly ${KINDS[top] || top}.`;
    }
    return node;
  }

  function drawBoxes(r) {
    const img = $("#stage img");
    if (!img) return;
    const place = () => {
      $$(".box").forEach((b) => b.remove());
      if (state.result !== r) return;
      const sx = img.clientWidth / img.naturalWidth;
      const sy = img.clientHeight / img.naturalHeight;
      r.faces.forEach((f, i) => {
        if (!f.box) return;
        $("#stage").append(el("div", {
          class: `box v-${f.verdict}`,
          style: `left:${f.box.x * sx}px;top:${f.box.y * sy}px;width:${f.box.w * sx}px;height:${f.box.h * sy}px`,
        }, el("span", { class: "box-label" }, tag(i + 1), el("b", {}, pct(f.fake_probability)))));
      });
    };
    if (img.complete && img.naturalWidth) place(); else img.addEventListener("load", place, { once: true });
    window.addEventListener("resize", place, { passive: true });
  }

  function renderTimeline(r) {
    const W = 600, H = 190, L = 40, R = 12, T = 14, B = 28;
    const pts = r.timeline;
    const tMax = Math.max(pts[pts.length - 1].t, 0.001);
    const x = (tv) => L + (tv / tMax) * (W - L - R);
    const y = (p) => T + (1 - p) * (H - T - B);
    const ns = "http://www.w3.org/2000/svg";
    const s = (name, attrs) => { const n = document.createElementNS(ns, name); Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v)); return n; };

    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "group", "aria-label": "Manipulation probability for each sampled frame" });
    svg.append(s("rect", { class: "band", x: L, y: y(0.6), width: W - L - R, height: y(0.4) - y(0.6) }));
    [0, 0.5, 1].forEach((p) => {
      svg.append(s("line", { class: "grid", x1: L, x2: W - R, y1: y(p), y2: y(p) }));
      const lbl = s("text", { x: L - 8, y: y(p) + 4, "text-anchor": "end" });
      lbl.textContent = pct(p, 0);
      svg.append(lbl);
    });
    [0, tMax].forEach((tv, i) => {
      const lbl = s("text", { x: x(tv), y: H - 8, "text-anchor": i === 0 ? "start" : "end" });
      lbl.textContent = `${num(tv, 0)} s`;
      svg.append(lbl);
    });
    const valid = pts.filter((p) => p.fake_probability != null);
    if (valid.length > 1) {
      const line = valid.map((p) => `${x(p.t)},${y(p.fake_probability)}`).join(" ");
      svg.append(s("polygon", { class: "area", points: `${x(valid[0].t)},${y(0)} ${line} ${x(valid[valid.length - 1].t)},${y(0)}` }));
      svg.append(s("polyline", { class: "line", points: line }));
    }
    const cursor = s("line", { class: "cursor", x1: 0, x2: 0, y1: T, y2: H - B, visibility: "hidden" });
    svg.append(cursor);
    const tip = el("p", { class: "timeline-tip" }, "Click a point to jump the video to that moment.");
    const video = $("#stage video");
    pts.forEach((p) => {
      const has = p.fake_probability != null;
      const v = has ? p.fake_probability : 0;
      const label = has ? `${num(p.t)} s: ${pct(v)} manipulation probability` : `${num(p.t)} s: no face found`;
      const c = s("circle", { class: "dot", cx: x(p.t), cy: y(v), r: has ? 5 : 3.5, tabindex: 0, role: "button",
        fill: has ? tone(verdictOf(v)) : "var(--rule)", "aria-label": label });
      const show = () => {
        tip.textContent = label;
        cursor.setAttribute("x1", x(p.t));
        cursor.setAttribute("x2", x(p.t));
        cursor.setAttribute("visibility", "visible");
      };
      const seek = () => {
        show();
        svg.querySelectorAll(".dot.on").forEach((d) => d.classList.remove("on"));
        c.classList.add("on");
        if (video) { try { video.pause(); video.currentTime = p.t; } catch { /* not seekable yet */ } }
      };
      c.addEventListener("mouseenter", show);
      c.addEventListener("focus", show);
      c.addEventListener("click", seek);
      c.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); seek(); } });
      svg.append(c);
    });
    return el("section", { class: "timeline" },
      el("p", { class: "exhibit-title" }, tag(4, true), "Timeline", el("small", {}, `${num(pts.length, 0)} frames`)),
      svg, tip);
  }

  // ------------------------------------------------------------------ accuracy section
  const DATASETS = {
    ffhq_stylegan: ["140k Real and Fake Faces", "Generated faces (StyleGAN)"],
    dff: ["DeepFakeFace", "Diffusion images and swaps"],
    celebdf: ["Celeb-DF v2", "Celebrity face-swap videos"],
    dfd: ["Google DFD", "Actor face-swap videos"],
  };

  function renderMetrics(m) {
    const total = $("#accHero");
    const chart = $("#accBars");
    total.replaceChildren();
    chart.replaceChildren();
    if (!m || !m.test || !m.test.fusion_head) {
      total.hidden = true;
      chart.append(el("p", { class: "muted" }, "Results are not available yet."));
      return;
    }
    total.hidden = false;
    const fusion = m.test.fusion_head;
    const o = fusion.overall;
    total.append(
      el("div", {}, el("b", {}, fixed(o.auc)), el("p", {}, `test AUC on ${num(o.n, 0)} faces the model never saw`)),
      el("dl", {},
        el("div", {}, el("dt", {}, "Accuracy"), el("dd", {}, pct(o.accuracy))),
        el("div", {}, el("dt", {}, "Precision"), el("dd", {}, pct(o.precision))),
        el("div", {}, el("dt", {}, "Recall"), el("dd", {}, pct(o.recall))),
        el("div", {}, el("dt", {}, "F1"), el("dd", {}, fixed(o.f1, 2)))));

    const singles = Object.entries(m.test).filter(([k]) => k.startsWith("zero_shot:"));
    const bestSingle = (src) => Math.max(...singles.map(([, v]) => (v[src] && Number.isFinite(v[src].auc) ? v[src].auc : 0)));
    chart.append(el("div", { class: "acc-legend" },
      el("span", {}, el("i", { style: "background:var(--ink)" }), "DeepShield"),
      el("span", {}, el("i", { style: "background:#c3c4c7" }), "Best single public model")));
    const bar = (cls, v) => el("div", { class: `acc-bar ${cls}` },
      el("span", { class: "track" }, el("i", { style: `--v:${Math.max(0.5, v)}` })), el("b", {}, fixed(v)));
    Object.keys(DATASETS).filter((src) => fusion[src]).forEach((src) => {
      chart.append(el("div", { class: "acc-row" },
        el("p", { class: "acc-name" }, el("b", {}, DATASETS[src][0]), el("span", {}, DATASETS[src][1])),
        el("div", { class: "acc-bars" }, bar("ours", fusion[src].auc), bar("base", bestSingle(src)))));
    });
    chart.append(el("p", { class: "acc-note" }, "Bars start at 0.5, the coin-flip level."));
  }

  fetch("/api/metrics").then((res) => (res.ok ? res.json() : null))
    .then(renderMetrics)
    .catch(() => renderMetrics(null));
})();
