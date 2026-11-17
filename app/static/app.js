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

  const LABEL = { fake: "Likely manipulated", real: "Likely authentic", uncertain: "Inconclusive" };
  const NOTES = {
    multiple_faces: "Several faces were found. The verdict follows the most suspicious face.",
    rotated: "The photo was sideways, so it was turned upright to find the face.",
    partial_faces: "A face was visible in only some frames. Frames without a face were skipped.",
    unfamiliar_frame: "The most suspicious frame looks unlike anything DeepShield was trained on. Treat this result with extra care.",
  };
  const KINDS = {
    ffhq: "FFHQ photo", stylegan: "StyleGAN face", imdb_wiki: "Celebrity photo",
    sd_text2img: "Stable Diffusion image", sd_inpainting: "Diffusion edit", insightface: "InsightFace swap",
    celebdf_real: "Celeb-DF real", celebdf_swap: "Celeb-DF swap", dfd_real: "DFD real", dfd_swap: "DFD swap",
    flux_dev: "FLUX.1 face", flux_pro: "FLUX.1 pro face", sdxl: "SDXL face", t2i: "Text-to-image face",
    faceswap: "Face swap", celeba: "Celebrity photo", celebahq: "Celebrity photo", coco: "Everyday photo",
    flickr: "Everyday photo", pexels: "Stock photo", laion: "Web photo", reddit_real: "Photo shared online",
    reddit_ai: "AI image shared online", "flux.1-dev": "FLUX.1 image", "sd-3.5": "Stable Diffusion 3.5 image",
    "gpt-image-1": "GPT-image", "dalle-3": "DALL-E 3 image", "ideogram-3.0": "Ideogram image",
  };
  const kindName = (k) => KINDS[k] || `${k} image`;
  // The real stages of the pipeline, in order. Timings are typical, so the list
  // advances on its own and completes when the answer arrives.
  const STEPS = {
    image: [["Finding faces", 300], ["First detector (ViT)", 800], ["Second detector (SigLIP)", 1400],
      ["Combining the evidence", 1800], ["Building the evidence map", null]],
    video: [["Sampling up to 24 frames", 1500], ["Finding the face in each frame", 4000],
      ["Running both detectors on every frame", 11000], ["Combining the frames", 15000],
      ["Explaining the most suspicious frame", null]],
  };

  const state = { file: null, kind: null, url: null, result: null, busy: false, history: [], current: null, sample: null, face: 0 };

  const pct = (p, d = 1) => new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: d }).format(p);
  const num = (n, d = 1) => new Intl.NumberFormat("en-US", { maximumFractionDigits: d }).format(n);
  const fixed = (n, d = 3) => n.toFixed(d);
  const tone = (v) => ({ fake: "var(--fake)", real: "var(--real)", uncertain: "var(--unsure)" }[v] || "var(--mute)");
  const verdictOf = (p) => (p >= 0.6 ? "fake" : p <= 0.4 ? "real" : "uncertain");
  const size = (b) => (b >= 1048576 ? `${num(b / 1048576)} MB` : `${num(b / 1024, 0)} KB`);

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
  const icon = (d) => {
    const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    s.setAttribute("viewBox", "0 0 24 24");
    s.setAttribute("aria-hidden", "true");
    s.innerHTML = d;
    return s;
  };
  const ICONS = {
    pdf: '<path d="M12 4v11m0 0-4.5-4.5M12 15l4.5-4.5M5 19h14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>',
    copy: '<rect x="8" y="8" width="12" height="12" rx="3" fill="none" stroke="currentColor" stroke-width="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" fill="none" stroke="currentColor" stroke-width="2"/>',
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

  // ------------------------------------------------------------------ panels
  function showPanel(name) {
    $("#intro").hidden = name !== "intro";
    $("#checking").hidden = name !== "checking";
    $("#result").hidden = name !== "result";
    $("#app").classList.toggle("empty", name === "intro");
    $("#report").scrollTop = 0;
  }

  // ------------------------------------------------------------------ intake
  const drop = $("#drop");
  const input = $("#file");

  function kindOf(file) {
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    if (IMAGE_TYPES.includes(file.type) || ["jpg", "jpeg", "png", "webp", "bmp"].includes(ext)) return "image";
    if (VIDEO_TYPES.includes(file.type) || ["mp4", "mov", "webm", "avi", "mkv"].includes(ext)) return "video";
    return null;
  }

  function clearMedia() {
    releaseCurrentUrl();
    Object.assign(state, { file: null, kind: null, url: null, current: null, result: null });
    $("#stage").replaceChildren();
    $$(".box").forEach((b) => b.remove());
    $("#viewer").hidden = true;
    $("#dropEmpty").hidden = false;
    drop.classList.remove("has-file");
    drop.setAttribute("role", "button");
    drop.setAttribute("aria-label", "Choose a photo or video to check");
    drop.tabIndex = 0;
    $("#fileMeta").textContent = "";
    $("#tabMedia").textContent = "Photo";
    $$(".tab").forEach((t) => { t.disabled = t.dataset.tab !== "media"; });
    setTab("media");
  }

  function showError(title, msg, retry) {
    showPanel("checking");
    $(".progress").hidden = true;
    $("#checkTitle").textContent = title;
    $("#steps").replaceChildren();
    $("#progressBar").style.width = "0";
    const note = $("#checkNote");
    note.textContent = msg;
    note.classList.add("error");
    $$(".check-actions").forEach((n) => n.remove());
    $("#checking").append(el("div", { class: "actions check-actions" },
      retry ? el("button", { class: "pill dark", type: "button", onclick: retry }, "Try again") : null,
      el("button", { class: retry ? "pill" : "pill dark", type: "button", onclick: () => { resetIntake(); input.click(); } }, "Choose another file")));
  }

  function showMedia(kind, url, name, bytes) {
    const stage = $("#stage");
    stage.replaceChildren();
    $$(".box").forEach((b) => b.remove());
    const media = document.createElement(kind === "image" ? "img" : "video");
    media.src = url;
    if (kind === "video") Object.assign(media, { muted: true, controls: true, playsInline: true, preload: "metadata" });
    else media.alt = "The file being checked";
    stage.append(media);
    $("#dropEmpty").hidden = true;
    $("#viewer").hidden = false;
    drop.classList.add("has-file");
    drop.removeAttribute("role");
    drop.removeAttribute("aria-label");
    drop.tabIndex = -1;
    $("#tabMedia").textContent = kind === "video" ? "Video" : "Photo";
    setTab("media");
    $$(".tab").forEach((t) => { if (t.dataset.tab !== "media") t.disabled = true; });
    $("#fileMeta").textContent = `${name}, ${size(bytes)}`;
  }

  function releaseCurrentUrl() {
    if (state.url && !state.history.some((h) => h.url === state.url)) URL.revokeObjectURL(state.url);
  }

  function choose(file, sample = null) {
    if (!file || state.busy) return;
    const kind = kindOf(file);
    if (!kind) {
      clearMedia();
      return showError("This file type is not supported", "Use a JPG, PNG, WEBP or BMP photo, or an MP4, MOV, WEBM, AVI or MKV video.");
    }
    if (file.size > (kind === "image" ? MAX_IMAGE : MAX_VIDEO)) {
      clearMedia();
      return showError("This file is too large", "Photos can be up to 10 MB and videos up to 60 MB.");
    }
    releaseCurrentUrl();
    Object.assign(state, { file, kind, url: URL.createObjectURL(file), current: null, sample, result: null, face: 0 });
    showMedia(kind, state.url, file.name, file.size);
    analyze();
  }

  input.addEventListener("change", () => { choose(input.files[0]); input.value = ""; });
  $("#pickBtn").addEventListener("click", () => { if (!state.busy) input.click(); });
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
    $("#app").scrollIntoView({ block: "start" });
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
    if (state.busy) return;
    try {
      const res = await fetch(btn.dataset.sample);
      if (!res.ok) throw new Error(res.statusText);
      const blob = await res.blob();
      // The samples are unlabeled on purpose; the answer is shown with the result.
      choose(new File([blob], btn.dataset.sample.split("/").pop(), { type: blob.type || "image/jpeg" }),
        { truth: btn.dataset.truth, origin: btn.dataset.origin });
    } catch { showError("The server could not be reached", "Check your connection and try again."); }
  }));

  function resetIntake() {
    if (state.busy) return;
    clearMedia();
    showPanel("intro");
    renderHistory();
  }
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && state.file && !state.busy && !e.target.closest("input, textarea")) resetIntake();
  });

  // ------------------------------------------------------------------ view tabs
  function setTab(name) {
    $$(".tab").forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("on", on);
      t.setAttribute("aria-selected", String(on));
    });
    $$(".view").forEach((v) => v.classList.toggle("on", v.dataset.view === name));
    if (name === "media" && placeBoxes) placeBoxes();
  }
  $$(".tab").forEach((t) => t.addEventListener("click", () => { if (!t.disabled) setTab(t.dataset.tab); }));
  $("#wipeRange").addEventListener("input", (e) => $("#wipe").style.setProperty("--pos", `${e.target.value}%`));

  // ------------------------------------------------------------------ checking stages
  let stepTimers = [];
  function startSteps(kind) {
    stepTimers.forEach(clearTimeout);
    stepTimers = [];
    const steps = STEPS[kind];
    $("#checkTitle").textContent = kind === "video" ? "Checking your video" : "Checking your photo";
    const note = $("#checkNote");
    note.classList.remove("error");
    $(".progress").hidden = false;
    note.textContent = kind === "video" ? "Videos take 10 to 40 seconds." : "Photos take a few seconds.";
    $$(".check-actions").forEach((n) => n.remove());
    const list = $("#steps");
    list.replaceChildren(...steps.map(([label]) => el("li", {}, label)));
    const items = $$("li", list);
    const go = (i) => {
      items.forEach((li, j) => { li.className = j < i ? "done" : j === i ? "now" : ""; });
      $("#progressBar").style.width = `${Math.round(((i + 0.5) / steps.length) * 92)}%`;
    };
    go(0);
    steps.forEach(([, at], i) => { if (at != null) stepTimers.push(setTimeout(() => go(i + 1), at)); });
    showPanel("checking");
  }
  function finishSteps() {
    stepTimers.forEach(clearTimeout);
    $$("#steps li").forEach((li) => { li.className = "done"; });
    $("#progressBar").style.width = "100%";
  }

  // ------------------------------------------------------------------ analysis
  async function analyze() {
    if (!state.file) return;
    state.busy = true;
    startSteps(state.kind);
    const file = state.file;
    try {
      let res, data;
      for (let attempt = 0; ; attempt++) {
        const body = new FormData();
        body.append("file", file, file.name);
        res = await fetch("/api/analyze", { method: "POST", body });
        data = await res.json().catch(() => ({}));
        // 503 = the models are still loading after a restart. Keep waiting, up to about two minutes.
        if (res.status !== 503 || attempt >= 30) break;
        $("#checkNote").textContent = "The models are still warming up after a restart. This takes up to a minute.";
        await new Promise((resolve) => setTimeout(resolve, 4000));
      }
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : res.statusText);
      if (state.file !== file) return;
      finishSteps();
      data.sample = state.sample;
      state.result = data;
      addHistory(data);
      state.busy = false;
      renderResult(data);
    } catch (err) {
      stepTimers.forEach(clearTimeout);
      state.busy = false;
      if (err instanceof TypeError) showError("The server could not be reached", "Check your connection and try again.", analyze);
      else showError("This file could not be checked", err.message);
    } finally {
      state.busy = false;
    }
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
    Object.assign(state, { file: h.file, kind: h.kind, url: h.url, result: h.result, current: h.id, sample: h.result.sample });
    showMedia(h.kind, h.url, h.name, h.file.size);
    renderResult(h.result);
    renderHistory();
  }

  function renderHistory() {
    const list = $("#recentList");
    list.replaceChildren();
    $("#recent").hidden = state.history.length < 2;
    state.history.forEach((h) => {
      const r = h.result;
      const label = `${h.name}: ${LABEL[r.verdict]}, ${pct(r.fake_probability, 0)}`;
      list.append(el("li", {}, el("button", {
        class: `recent-item${h.id === state.current ? " on" : ""}`, type: "button", title: label, "aria-label": label,
        onclick: () => openHistory(h.id),
      }, h.thumb ? el("img", { src: h.thumb, alt: "" }) : el("span", { class: "kind" }, h.kind === "video" ? "VIDEO" : "IMAGE"),
      el("i", { style: `background:${tone(r.verdict)}` }))));
    });
  }

  // ------------------------------------------------------------------ result
  function headline(r) {
    const video = r.media_type === "video";
    const span = (cls, text) => el("span", { class: cls }, text);
    if (r.notes.includes("provenance_ai")) {
      return ["fake", ["This image says it was ", span("fake", r.provenance.kind === "edited" ? "edited with AI." : "made with AI.")]];
    }
    if (r.notes.includes("no_face")) return ["uncertain", [`We couldn't find a face in this ${video ? "video" : "image"}.`]];
    if (r.notes.includes("unfamiliar")) return ["uncertain", ["This photo is ", span("uncertain", "unlike anything we trained on.")]];
    if (r.verdict === "fake") {
      const faceLeads = Math.max(...r.faces.map((f) => f.fake_probability)) >= 0.5;
      return ["fake", video ? ["This video most likely contains a ", span("fake", "manipulated face.")]
        : faceLeads ? ["This face is most likely ", span("fake", "AI-generated or manipulated.")]
          : ["This picture is most likely ", span("fake", "AI-generated.")]];
    }
    if (r.verdict === "real") {
      return ["real", video ? ["The face in this video looks ", span("real", "authentic.")]
        : ["This face looks ", span("real", "authentic.")]];
    }
    return ["uncertain", ["This one is ", span("uncertain", "too close to call.")]];
  }

  function strength(p) {
    if (p >= 0.85) return "strong";
    if (p >= 0.6) return "some";
    return "very few";
  }

  const PROV_SOURCE = { c2pa: "Content Credentials", xmp: "metadata", exif: "EXIF data", png: "PNG text" };

  function subline(r) {
    if (r.notes.includes("provenance_ai")) {
      const pv = r.provenance;
      const who = pv.source === "c2pa" && pv.tool ? ` (signed by ${pv.tool})` : pv.tool ? ` (${pv.tool})` : "";
      const what = pv.kind === "edited" ? "it was edited with AI" : "it was generated by AI";
      return `Its own ${PROV_SOURCE[pv.source] || "metadata"}${who} record that ${what}. That label is more reliable than any detector.`;
    }
    if (r.notes.includes("no_face")) {
      return `DeepShield judges faces. The score above comes from the whole ${r.media_type === "video" ? "frame" : "image"} and is not a verdict.`;
    }
    if (r.notes.includes("unfamiliar")) {
      return "Unusual lighting, heavy editing or a newer AI generator can do this, so the score is not trusted. Check where the file came from.";
    }
    if (r.verdict === "uncertain") return "The evidence sits between 40% and 60%, so DeepShield will not guess. Check where the file came from.";
    const what = r.media_type === "video" ? `across ${num(r.frames_analyzed, 0)} frames` : `in ${num(r.processing_ms / 1000)} s`;
    const where = r.whole_image_probability != null ? "Together, the face and the whole picture show" : "The face shows";
    return `${pct(r.confidence, 0)} confidence, checked ${what}. ${where} ${strength(r.fake_probability)} signs of AI manipulation.`;
  }

  function renderResult(r) {
    const box = $("#result");
    box.replaceChildren();
    const [kind, words] = headline(r);
    const decisive = r.faces.reduce((best, f, i) => (f.fake_probability > r.faces[best].fake_probability ? i : best), 0);
    state.face = decisive;

    box.append(
      el("span", { class: `label ${kind}` }, r.notes.includes("provenance_ai") ? "Labeled as AI"
        : r.notes.includes("no_face") ? "No face found" : LABEL[kind]),
      el("h2", { class: "headline" }, words),
      el("div", { class: "meter", role: "img", "aria-label": `${pct(r.fake_probability)} manipulation probability` },
        el("span", { class: "meter-track" }, el("i", { style: `width:${Math.max(r.fake_probability * 100, 1.5)}%;background:${tone(r.verdict)}` })),
        el("b", {}, pct(r.fake_probability)),
        el("div", { class: "meter-scale", "aria-hidden": "true" }, el("span", {}, "Real"), el("span", {}, "Unsure"), el("span", {}, "Fake"))),
      el("p", { class: "sub" }, subline(r)));

    if (r.sample) {
      const right = r.verdict === r.sample.truth;
      const answer = r.sample.truth === "fake" ? "AI-generated" : "a real photo";
      box.append(el("p", { class: `truth ${right ? "right" : r.verdict === "uncertain" ? "" : "wrong"}` },
        el("b", {}, `Answer: this sample is ${answer}.`), ` ${r.sample.origin}. `,
        right ? "DeepShield got it right." : r.verdict === "uncertain" ? "DeepShield was not sure." : "DeepShield got this one wrong."));
    }

    if (r.media_type === "video" && r.timeline.length) box.append(renderTimeline(r));

    const hasFace = !r.notes.includes("no_face");
    if (hasFace && r.faces.length > 1) {
      box.append(el("div", { class: "sec" }, el("h3", {}, "Faces"),
        el("div", { class: "faces-pick", role: "group", "aria-label": "Choose a face" },
          r.faces.map((f, i) => el("button", {
            type: "button", class: i === decisive ? "on" : "", "data-face": i,
            onclick: () => selectFace(r, i),
          }, `Face ${i + 1}: ${pct(f.fake_probability, 0)}`)))));
    }
    if (r.media_type === "image") {
      const views = renderViews(r);
      if (views) box.append(views);
    }
    box.append(el("div", { class: "face-detail" }));
    if (r.media_type === "image") box.append(renderProvenance(r));

    const notes = r.notes.filter((n) => NOTES[n]);
    if (notes.length) box.append(el("ul", { class: "notes" }, notes.map((n) => el("li", {}, NOTES[n]))));

    box.append(el("div", { class: "actions" },
      el("a", { class: "pill dark", href: `/api/report/${r.id}`, download: "" }, icon(ICONS.pdf), "Download report"),
      el("button", { class: "pill", type: "button", onclick: () => copySummary(r) }, icon(ICONS.copy), "Copy summary"),
      el("button", { class: "pill", type: "button", onclick: () => { resetIntake(); input.click(); } }, "New check")));

    showPanel("result");
    selectFace(r, decisive);
    if (r.media_type === "image") drawBoxes(r);
    if (matchMedia("(max-width: 1100px)").matches) $("#report").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function selectFace(r, i) {
    state.face = i;
    const f = r.faces[i];
    $$(".faces-pick button").forEach((b) => b.classList.toggle("on", Number(b.dataset.face) === i));
    $$(".box").forEach((b) => b.classList.toggle("sel", Number(b.dataset.face) === i));
    const crop = f.crop_png ? `data:image/png;base64,${f.crop_png}` : null;
    const heat = f.heatmap_png ? `data:image/png;base64,${f.heatmap_png}` : null;
    const hasMap = Boolean(crop && heat && !r.notes.includes("no_face"));
    $$(".tab").forEach((t) => { if (t.dataset.tab !== "media") t.disabled = !hasMap; });
    if (hasMap) {
      $("#mapImg").src = heat;
      $("#wipeA").src = crop;
      $("#wipeB").src = heat;
    } else if (!$(".view.on[data-view='media']")) setTab("media");
    $(".face-detail").replaceChildren(...renderFaceDetail(f, r, hasMap));
  }

  function renderFaceDetail(f, r, hasMap) {
    const out = [];
    const who = r.media_type === "video" ? "the most suspicious frame" : r.faces.length > 1 ? `face ${state.face + 1}` : "this face";
    if (hasMap) {
      const open = () => {
        setTab("compare");
        if (matchMedia("(max-width: 1100px)").matches) $(".media-card").scrollIntoView({ behavior: "smooth", block: "start" });
      };
      out.push(el("div", { class: "sec" }, el("h3", {}, "Evidence map"),
        el("button", { class: "evidence", type: "button", onclick: open, "aria-label": `Compare ${who} with its evidence map` },
          el("img", { src: `data:image/png;base64,${f.crop_png}`, alt: "" }),
          el("img", { src: `data:image/png;base64,${f.heatmap_png}`, alt: "" }),
          el("span", {}, `Warm areas on ${who} pushed the score toward ${f.explains === "real" ? "authentic" : "manipulated"}. `,
            el("u", {}, "Compare side by side")))));
    }
    const scores = Object.entries(f.model_scores);
    if (scores.length && !r.notes.includes("no_face")) {
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
      out.push(el("div", { class: "sec" }, el("h3", {}, "Detector agreement"),
        el("div", { class: "detectors" }, scores.map(([name, p]) => el("div", { class: "det" },
          el("p", { title: name }, name), el("b", {}, pct(p)),
          el("span", { class: "bar", "aria-hidden": "true" }, el("i", { style: `width:${Math.max(p * 100, 2)}%;background:${tone(verdictOf(p))}` }))))),
        agree ? el("p", { class: "note" }, agree) : null));
    }
    if (f.similar && f.similar.length && !r.notes.includes("no_face")) {
      const groups = [];
      f.similar.forEach((s) => {
        const g = groups.find((x) => x.kind === s.kind && x.label === s.label);
        if (g) g.sims.push(s.similarity); else groups.push({ kind: s.kind, label: s.label, sims: [s.similarity] });
      });
      const k = f.similar.length;
      const fakes = f.similar.filter((s) => s.label === "fake");
      const side = fakes.length * 2 >= k ? fakes : f.similar.filter((s) => s.label === "real");
      const counts = {};
      side.forEach((s) => { counts[s.kind] = (counts[s.kind] || 0) + 1; });
      const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
      const note = f.unfamiliar
        ? `Even the closest known face is a weak match (similarity ${fixed(f.similar[0].similarity, 2)}), so this face is outside what the model has learned.`
        : `${side.length} of the ${k} most similar faces in the training data are ${side === fakes ? "fakes" : "real"}, mostly ${kindName(top)}.`;
      out.push(el("div", { class: "sec" }, el("h3", {}, "Closest known cases"),
        el("ul", { class: "cases" }, groups.map((g) => el("li", { title: `Similarity ${g.sims.map((v) => fixed(v, 2)).join(", ")}` },
          el("i", { style: `background:${g.label === "fake" ? "var(--fake)" : "var(--real)"}`, "aria-hidden": "true" }),
          el("span", {}, `${kindName(g.kind)}, ${g.label}`),
          el("b", {}, `${g.sims.length} of ${k}`)))),
        el("p", { class: "note" }, note)));
    }
    return out;
  }

  function renderViews(r) {
    if (r.whole_image_probability == null || r.notes.includes("no_face") || r.notes.includes("provenance_ai")) return null;
    const face = Math.max(...r.faces.map((f) => f.fake_probability));
    const whole = r.whole_image_probability;
    const rows = [["The face", face], ["The whole picture", whole]];
    const same = (face >= 0.5) === (whole >= 0.5);
    const note = same
      ? `Both views lean ${whole >= 0.5 ? "AI-made" : "real"}, so they back each other up.`
      : face >= 0.5
        ? "The face looks manipulated but the rest of the picture looks like a photo. That fits a face swap or an edited face."
        : "The face looks clean, but the scene around it looks generated. Newer generators often give themselves away in the background, light and texture.";
    return el("div", { class: "sec" }, el("h3", {}, "Two views"),
      el("div", { class: "detectors" }, rows.map(([name, p]) => el("div", { class: "det" },
        el("p", {}, name), el("b", {}, pct(p)),
        el("span", { class: "bar", "aria-hidden": "true" }, el("i", { style: `width:${Math.max(p * 100, 2)}%;background:${tone(verdictOf(p))}` }))))),
      el("p", { class: "note" }, `${note} The score at the top combines both.`));
  }

  function renderProvenance(r) {
    const pv = r.provenance || {};
    const lines = [];
    if (pv.ai) {
      lines.push(`Its ${PROV_SOURCE[pv.source] || "metadata"} say${pv.source === "c2pa" ? "" : "s"} it was ${pv.kind === "edited" ? "edited" : "made"} with AI${pv.tool ? ` (${pv.tool})` : ""}.`);
    } else if (pv.source === "c2pa") {
      lines.push(`It carries Content Credentials${pv.tool ? ` signed by ${pv.tool}` : ""} with no AI step recorded.`);
    } else {
      lines.push("No Content Credentials or AI label found. That is normal: screenshots, WhatsApp and most social apps remove them, so their absence proves nothing.");
    }
    if (pv.camera) lines.push(`The EXIF data names a camera: ${pv.camera}. Easy to edit, so it is a hint, not proof.`);
    return el("div", { class: "sec" }, el("h3", {}, "What the file says"),
      el("ul", { class: "prov" }, lines.map((t) => el("li", {}, t))),
      el("p", { class: "note" }, "ChatGPT, Gemini and Adobe sign their images with ",
        el("a", { href: "https://contentcredentials.org/verify", target: "_blank", rel: "noopener" }, "Content Credentials"),
        ". DeepShield reads them before looking at pixels."));
  }

  async function copySummary(r) {
    const name = (state.history.find((h) => h.id === r.id) || {}).name || r.filename;
    const text = `DeepShield: ${LABEL[r.verdict]} (${pct(r.fake_probability)} manipulation probability, `
      + `${pct(r.confidence, 0)} confidence). File: ${name}.`;
    try {
      await navigator.clipboard.writeText(text);
      toast("Summary copied");
    } catch { toast("Copy failed. Select the text and copy it yourself."); }
  }

  let placeBoxes = null;
  window.addEventListener("resize", () => { if (placeBoxes) placeBoxes(); }, { passive: true });

  function drawBoxes(r) {
    const img = $("#stage img");
    if (!img) return;
    const place = () => {
      $$(".box").forEach((b) => b.remove());
      if (state.result !== r || !img.clientWidth) return;
      // Boxes are in the pixels of the image the server analyzed (very large photos are scaled down first).
      const sx = img.clientWidth / (r.width || img.naturalWidth);
      const sy = img.clientHeight / (r.height || img.naturalHeight);
      r.faces.forEach((f, i) => {
        if (!f.box) return;
        $("#stage").append(el("div", {
          class: `box v-${f.verdict}${i === state.face ? " sel" : ""}`, "data-face": i,
          style: `left:${f.box.x * sx}px;top:${f.box.y * sy}px;width:${f.box.w * sx}px;height:${f.box.h * sy}px`,
        }, el("b", {}, r.faces.length > 1 ? `${i + 1}: ${pct(f.fake_probability)}` : pct(f.fake_probability))));
      });
    };
    placeBoxes = place;
    if (img.complete && img.naturalWidth) place(); else img.addEventListener("load", place, { once: true });
  }

  function renderTimeline(r) {
    const W = 600, H = 170, L = 40, R = 10, T = 12, B = 26;
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
      const lbl = s("text", { x: x(tv), y: H - 6, "text-anchor": i === 0 ? "start" : "end" });
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
    pts.forEach((p) => {
      const has = p.fake_probability != null;
      const v = has ? p.fake_probability : 0;
      const label = has ? `${num(p.t)} s: ${pct(v)} manipulation probability` : `${num(p.t)} s: no face found`;
      const c = s("circle", { class: "dot", cx: x(p.t), cy: y(v), r: has ? 5 : 3.5, tabindex: 0, role: "button",
        fill: has ? tone(verdictOf(v)) : "var(--line-2)", "aria-label": label });
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
        setTab("media");
        const video = $("#stage video");
        if (video) { try { video.pause(); video.currentTime = p.t; } catch { /* not seekable yet */ } }
      };
      c.addEventListener("mouseenter", show);
      c.addEventListener("focus", show);
      c.addEventListener("click", seek);
      c.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); seek(); } });
      svg.append(c);
    });
    return el("div", { class: "sec timeline" }, el("h3", {}, `Timeline, ${num(pts.length, 0)} frames`), svg, tip);
  }

  // ------------------------------------------------------------------ accuracy section
  // Held-out sources, grouped by what they test. [metrics key, name, what it is]
  const GROUPS = [
    ["AI-generated faces", "test_all", [
      ["ffhq_stylegan", "StyleGAN", "generated faces vs FFHQ photos"],
      ["dff", "Stable Diffusion, InsightFace", "generated and swapped vs celebrity photos"],
      ["flux_faces", "FLUX.1", "generated faces"],
      ["sdxl_faces", "Stable Diffusion XL", "generated faces"],
      ["sfhq_t2i", "FLUX.1 and SDXL portraits", "1024 px studio-style portraits"],
    ]],
    ["Face swaps", "test_all", [
      ["celebdf", "Celeb-DF v2", "celebrity face-swap videos"],
      ["dfd", "Google DFD", "actor face-swap videos"],
      ["faceswap", "CelebA swaps", "swapped celebrity photos"],
    ]],
    ["Real photos", "test_all", [
      ["pexels", "Stock photos", "professional lighting and colour"],
      ["coco", "Everyday photos (COCO)", "faces in ordinary scenes"],
      ["flickr", "Everyday photos (Flickr)", "faces in ordinary scenes"],
      ["celebahq", "Celebrity photos, high quality", "CelebA-HQ"],
      ["celeba", "Celebrity photos, low resolution", "CelebA"],
    ]],
    ["Never seen in training", "unseen", [
      ["midjourney", "Midjourney", "generated faces"],
      ["stylegan3", "StyleGAN3", "generated faces"],
      ["collabdiff", "Collaborative Diffusion", "generated faces"],
      ["wfir", "whichfaceisreal.com", "StyleGAN faces vs photos"],
      ["heygen", "HeyGen avatars", "talking-head video frames"],
      ["youtube", "Real video clips", "frames from real talking-head clips"],
      ["lfw", "Press photos (LFW)", "small, old news photos"],
    ]],
  ];

  function renderMetrics(m) {
    const total = $("#accHero");
    const chart = $("#accBars");
    total.replaceChildren();
    chart.replaceChildren();
    if (!m || !m.test_all || !m.test_all.fusion_head) {
      total.hidden = true;
      chart.append(el("p", { class: "muted" }, "Results are not available yet."));
      return;
    }
    total.hidden = false;
    const w = m.wild;
    // Version 3 replaced the photo head; its numbers on the version 2 test sets live in wild_metrics.json.
    const testSet = w ? w.crops_test.v3_face_head : m.test_all.fusion_head;
    const unseenSet = w ? w.crops_unseen.v3_face_head : m.unseen.fusion_head;
    const o = testSet.overall;
    const u = unseenSet.overall;
    if (w) {
      const now = w.wild_test.v3_combined.overall;
      const before = w.wild_test.v2_face_head.overall;
      const app = w.app_outcome_with_inconclusive_band;
      total.append(
        el("div", {}, el("b", {}, fixed(now.auc)), el("p", {}, `AUC on ${num(now.n, 0)} real-world images it never trained on (version 2: ${fixed(before.auc)})`)),
        el("div", {}, el("b", {}, fixed(o.auc)), el("p", {}, `AUC on ${num(o.n, 0)} held-out faces from research datasets`)),
        el("dl", {},
          el("div", {}, el("dt", {}, "Real Reddit photos called fake"), el("dd", {}, `${pct(app.v3_app.reddit_real.fake)} (was ${pct(app.v2_without_guard.reddit_real.fake)})`)),
          el("div", {}, el("dt", {}, "AI images on Reddit caught"), el("dd", {}, `${pct(app.v3_app.reddit_fake.fake, 0)} (was ${pct(app.v2_without_guard.reddit_fake.fake, 0)})`)),
          el("div", {}, el("dt", {}, "Accuracy, real-world images"), el("dd", {}, pct(now.accuracy))),
          el("div", {}, el("dt", {}, "AUC, never-seen sources"), el("dd", {}, fixed(u.auc)))));
    } else {
      total.append(
        el("div", {}, el("b", {}, fixed(o.auc)), el("p", {}, `AUC on ${num(o.n, 0)} held-out faces from the kinds of data it trained on`)),
        el("div", {}, el("b", {}, fixed(u.auc)), el("p", {}, `AUC on ${num(u.n, 0)} faces from sources it never trained on`)),
        el("dl", {},
          el("div", {}, el("dt", {}, "Accuracy"), el("dd", {}, pct(o.accuracy))),
          el("div", {}, el("dt", {}, "Unseen accuracy"), el("dd", {}, pct(u.accuracy))),
          el("div", {}, el("dt", {}, "Precision"), el("dd", {}, pct(o.precision))),
          el("div", {}, el("dt", {}, "Recall"), el("dd", {}, pct(o.recall)))));
    }

    chart.append(el("div", { class: "acc-legend" },
      el("span", {}, el("i", { style: "background:var(--ink)" }), "DeepShield: share judged correctly"),
      el("span", {}, el("i", { class: "tick" }), w ? "Version 2 (real-world images) or best single public detector" : "Best single public detector")));
    const line = (name, what, acc, base) => el("div", { class: `acc-line${acc < 0.7 ? " weak" : ""}` },
      el("p", { class: "acc-name" }, el("b", {}, name), el("span", {}, what)),
      el("span", { class: "track", role: "img", "aria-label": `${pct(acc, 0)} correct; comparison ${pct(base, 0)}` },
        el("i", { style: `--v:${acc}` }), el("u", { style: `--v:${base}` })),
      el("b", { class: "acc-val" }, pct(acc, 0)));
    if (w) {
      const now = w.wild_test.v3_combined;
      const before = w.wild_test.v2_face_head;
      const rows = [
        ["reddit", "Photos shared on Reddit", "real photos and AI images people posted"],
        ["openfake_test", "Newest generators", "GPT-image, Nano Banana, Midjourney 7, Flux.2, Seedream, Sora, Veo"],
        ["coco_wild", "Everyday photos (COCO)", "real photos of people, newly held out"],
      ].filter(([k]) => now[k]);
      chart.append(el("div", { class: "acc-group" }, el("h3", {}, "Real-world images, never trained on"),
        rows.map(([k, name, what]) => line(name, what, now[k].accuracy, before[k].accuracy))));
    }
    GROUPS.forEach(([title, split, rows]) => {
      const ours = w ? (split === "unseen" ? unseenSet : testSet) : m[split].fusion_head;
      const singles = Object.entries(m[split]).filter(([k]) => k.startsWith("zero_shot:")).map(([, v]) => v);
      const present = rows.filter(([src]) => ours[src]);
      if (!present.length) return;
      chart.append(el("div", { class: "acc-group" },
        el("h3", {}, title),
        present.map(([src, name, what]) => line(name, what, ours[src].accuracy,
          Math.max(...singles.map((v) => (v[src] ? v[src].accuracy : 0)))))));
    });
    chart.append(el("p", { class: "acc-note" },
      "A photo counts as correct when its probability falls on the right side of 50%. The weak spots are Flux.2 and other distilled generators, low-resolution photos, hard video face swaps and talking-head avatars; there the app often answers \"inconclusive\"."));
  }

  fetch("/api/metrics").then((res) => (res.ok ? res.json() : null))
    .then(renderMetrics)
    .catch(() => renderMetrics(null));
})();
