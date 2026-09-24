// Continuous Gain tab: Add captures -> Analyse & select -> Train -> Test & export.
// Thin UI over /api/cg/*; training itself goes through the EXISTING /api/local_training/* and /api/kaggle/* routes.
(() => {
  const panel = document.getElementById("cg-panel");
  const tab = document.getElementById("tab-cg");
  if (!panel || !tab) return;
  const body = document.getElementById("cg-body");

  const S = { id: null, data: null, stage: 1, job: null, pollTimer: null, seriesName: null, train: null, backend: "local", preset: "standard",
    cab: null, cabEnabled: false, cabDisplayName: "" };
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(d));
  // Rough elapsed-time estimates -- real measurements from this project's own end-to-end runs (parallel capture
  // rendering + a per-capture probe cache, see hybrid/continuous_gain/parallel.py), not a guarantee: "usually", not exact.
  // Measured time per step: a fixed part plus a per-capture part. The one
  // place to update when a step gets faster or slower.
  const STEP_TIMING = {
    analyse: { fixed: 20, perCapture: 2.6 },   // timing/dither audit + per-capture probes, measured on 19 captures
    generate: { fixed: 20, perCapture: 4 },    // training audio through each selected capture
    validate: { fixed: 15, perCapture: 3 },    // comparisons + audition sweep
  };
  const estimateSeconds = (step, count) => STEP_TIMING[step].fixed + count * STEP_TIMING[step].perCapture;
  const formatDuration = (seconds) => seconds < 90 ? `about ${Math.max(10, Math.round(seconds / 10) * 10)} seconds` : `about ${Math.round(seconds / 60)} minutes`;
  const say = (t, bad) => setStatus(t || "", Boolean(bad));       // the app-wide status line (app.js), not a second message area

  async function api(path, opts = {}) {
    const init = { method: opts.method || "GET", headers: {} };
    if (opts.json !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.json); }
    if (opts.form) init.body = opts.form;
    const r = await fetch(path, init);
    let data = null;
    try { data = await r.json(); } catch (_e) { /* non-JSON */ }
    if (!r.ok) throw new Error((data && data.error) || `${r.status} ${r.statusText}`);
    return data;
  }

  // ---------- panel open/close (the other tabs manage their own panels; we hide ours when any other tab is clicked)
  const otherPanels = ["nam-tools-panel", "sessions-panel", "ai-assistant-panel", "wizard-panel", "tone3000-panel", "settings-panel"];
  function setOpen(open) {
    panel.hidden = !open;
    tab.classList.toggle("active", open);
    tab.setAttribute("aria-pressed", open ? "true" : "false");
    if (open) {
      otherPanels.forEach((id) => { const el = document.getElementById(id); if (el) el.hidden = true; });
      document.querySelectorAll(".utility-tabs .mode-tab").forEach((b) => {
        if (b !== tab) { b.classList.remove("active"); b.setAttribute("aria-pressed", "false"); }
      });
      document.querySelectorAll(".workflow-nav, .layout").forEach((el) => { el.hidden = true; });
      const md = document.getElementById("mode-description"); if (md) md.hidden = true;
    }
  }
  tab.addEventListener("click", () => setOpen(true));
  const closeTab = () => { window.namTrainingHost.detach(); setOpen(false); };
  document.querySelectorAll(".utility-tabs .mode-tab").forEach((b) => {
    if (b !== tab) b.addEventListener("click", () => { if (!panel.hidden) closeTab(); }, true);
  });

  // ---------- projects (listed, loaded, exported and deleted through the Sessions tab; the tab only creates and opens them)
  let loadSeq = 0;
  async function load(id) {
    const changedProject = S.id !== id;
    const mine = ++loadSeq;
    const data = await api(`/api/cg/projects/${id}`);
    if (mine !== loadSeq) return;            // a newer load (e.g. the project the user just created) superseded this response
    S.id = id;                               // only once it loaded: a failed open must not retarget the project on screen
    S.data = data;
    if (changedProject) {
      const cab = data.bundle && data.bundle.cab;
      S.cab = cab && cab.selected ? { path: cab.ir_working_path, filename: cab.original_filename, sha256: cab.sha256 } : null;
      S.cabEnabled = Boolean(S.cab);
      S.cabDisplayName = cab?.display_name || "";
    }
    document.getElementById("cg-current-name").textContent = data.project.name;
    render();
  }
  // An inline field, not window.prompt(): packaged desktop WebViews (WKWebView/WebView2) commonly implement
  // window.confirm() but not window.prompt() at all, silently doing nothing on click -- see desktopConfirm's
  // comment in app.js for the same WKWebView-delegate distinction. This also matches how every other name entry
  // in the app (session name, model name) already works.
  const newProjectForm = document.getElementById("cg-new-project-form");
  const newProjectNameInput = document.getElementById("cg-new-project-name");
  const openNewProjectForm = () => { newProjectForm.hidden = false; newProjectNameInput.value = ""; newProjectNameInput.focus(); };
  const closeNewProjectForm = () => { newProjectForm.hidden = true; };
  document.getElementById("cg-new-project").addEventListener("click", openNewProjectForm);
  document.getElementById("cg-new-project-cancel").addEventListener("click", closeNewProjectForm);
  const createNewProject = async () => {
    const name = newProjectNameInput.value.trim();
    if (!name) { newProjectNameInput.focus(); return; }
    try {
      const d = await api("/api/cg/projects", { method: "POST", json: { name } });
      closeNewProjectForm();
      S.stage = 1;
      await load(d.project.id);
    } catch (e) { say(e.message, true); }
  };
  document.getElementById("cg-new-project-create").addEventListener("click", createNewProject);
  newProjectNameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); createNewProject(); }
    else if (e.key === "Escape") { e.preventDefault(); closeNewProjectForm(); }
  });
  document.getElementById("cg-open-sessions").addEventListener("click", () => document.getElementById("tab-sessions").click());
  // Sessions -> Load calls this: switch to this tab and open that project (the session id is the project id).
  window.namContinuousGain = {
    async open(projectId) {
      setOpen(true);
      try {
        S.stage = 1;
        await load(projectId);
        const st = S.data.project;
        S.stage = S.data.training && S.data.training.trained ? 4 : S.data.bundle ? 3 : (S.data.plan || S.data.analysis) ? 2 : 1;
        render();
        say(`Opened ${st.name}.`);
      } catch (e) { say(`Could not open this project: ${e.message}. Its working files are not on this computer (a session file carries the trained model, not the captures).`, true); }
    },
  };
  document.getElementById("cg-steps").addEventListener("click", (e) => {
    const b = e.target.closest("[data-cg-stage]"); if (!b) return;
    S.stage = Number(b.dataset.cgStage); render();
  });

  // ---------- jobs
  // A job belongs to the project that started it (pid): its follow-up actions
  // always target that project, even if the user has opened another one since.
  async function runJob(startPath, payload, after) {
    const pid = S.id;
    try {
      const j = await api(startPath, { method: "POST", json: payload || {} });
      S.job = { id: j.job_id, pid, message: "starting", log: [] };
      render();
      clearInterval(S.pollTimer);
      let busy = false;                      // a slow tick must not overlap the next one and finish the job twice
      const timer = S.pollTimer = setInterval(async () => {
        if (busy) return;
        busy = true;
        try {
          const st = await api(`/api/cg/jobs/${j.job_id}`);
          if (st.state !== "running") {
            clearInterval(timer);
            const failed = st.state === "error";
            if (S.job && S.job.id === j.job_id) S.job = null;
            if (S.id === pid) await load(pid);
            if (failed) say(st.error || "The job failed", true); else if (after) await after(st, pid);
          } else {
            S.job = { id: j.job_id, pid, message: st.message, log: st.log, elapsed: st.elapsed };
            const el = document.getElementById("cg-job-log"); if (el) { el.textContent = st.log.join("\n"); el.scrollTop = el.scrollHeight; } const m = document.getElementById("cg-job-msg"); if (m) m.textContent = `${st.message} (${Math.round(st.elapsed)} s)`;
          }
        } catch (e) { clearInterval(timer); say(e.message, true); }
        finally { busy = false; }
      }, 1500);
    } catch (e) { say(e.message, true); }
  }
  const jobBox = () => S.job && S.job.pid === S.id ? `<div class="training-activity-card" aria-live="polite"><div class="training-activity-title" id="cg-job-msg">${esc(S.job.message)}</div><details class="training-log-details" open><summary>Show detailed log</summary><pre class="log-tail" id="cg-job-log">${esc((S.job.log || []).join("\n"))}</pre></details></div>` : "";

  // ---------- charts (inline SVG, theme variables)
  function chart({ xs, series, xLabel, yLabel, height = 220, width = 460, xTicks, shade = [], points = [] }) {
    const m = { l: 46, r: 12, t: 12, b: 34 };
    const allY = series.flatMap((s) => s.ys.filter((v) => v !== null && Number.isFinite(v)));
    let y0 = Math.min(...allY), y1 = Math.max(...allY);
    if (!(y1 > y0)) { y0 -= 1; y1 += 1; }
    const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (width - m.l - m.r);
    const Y = (v) => height - m.b - ((v - y0) / (y1 - y0)) * (height - m.t - m.b);
    let out = `<svg class="cg-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(yLabel)} against ${esc(xLabel)}">`;
    shade.forEach((s) => { out += `<rect x="${X(s[0])}" y="${m.t}" width="${Math.max(0, X(s[1]) - X(s[0]))}" height="${height - m.t - m.b}" fill="${s[2]}" opacity="0.25"/>`; });
    for (let i = 0; i <= 4; i++) { const v = y0 + ((y1 - y0) * i) / 4; out += `<line x1="${m.l}" x2="${width - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--border)"/><text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end">${fmt(v, Math.abs(y1 - y0) < 5 ? 2 : 1)}</text>`; }
    (xTicks || xs).forEach((v) => { out += `<line x1="${X(v)}" x2="${X(v)}" y1="${height - m.b}" y2="${height - m.b + 4}" stroke="var(--text-muted)"/><text x="${X(v)}" y="${height - m.b + 16}" text-anchor="middle">${fmt(v, Number.isInteger(v) ? 0 : 1)}</text>`; });
    out += `<text x="${(width + m.l) / 2}" y="${height - 4}" text-anchor="middle">${esc(xLabel)}</text><text x="12" y="${height / 2}" transform="rotate(-90 12 ${height / 2})" text-anchor="middle">${esc(yLabel)}</text>`;
    series.forEach((s) => {
      const pts = xs.map((x, i) => [x, s.ys[i]]).filter((p) => p[1] !== null && Number.isFinite(p[1]));
      if (s.line !== false) out += `<polyline fill="none" stroke="${s.color || "var(--accent)"}" stroke-width="1.5" stroke-dasharray="${s.dash || ""}" points="${pts.map((p) => `${X(p[0])},${Y(p[1])}`).join(" ")}"/>`;
    });
    points.forEach((p) => { out += `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="${p.r || 5}" fill="${p.fill}" stroke="${p.stroke || "var(--text)"}" stroke-width="1"><title>${esc(p.title || "")}</title></circle>`; if (p.label) out += `<text x="${X(p.x)}" y="${Y(p.y) - 9}" text-anchor="middle" style="fill:var(--text)">${esc(p.label)}</text>`; });
    return out + "</svg>";
  }

  // ---------- shared markup helpers (the Builder's own classes: card + h2, cost-badge, coverage-table, btn)
  const badge = (kind, text) => `<span class="cost-badge cost-badge-${kind} cg-chip">${esc(text)}</span>`;
  const card = (title, inner, right = "") => `<section class="card"><h2>${title}${right}</h2>${inner}</section>`;
  const table = (head, rows) => `<div class="table-wrap"><table class="coverage-table cg-wrap">${head.length ? `<thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead>` : ""}<tbody>${rows}</tbody></table></div>`;
  const cols = (...parts) => parts.join("");          // one centred column of cards, in flow order (the Builder's layout)
  const HINTS = { 1: "Upload every fixed-gain capture of one amp and channel", 2: "See what each capture adds and how Input gain will map", 3: "Generate the training files and train", 4: "Check the result against your captures and export" };

  // ---------- Stage 1
  function stage1() {
    const d = S.data;
    if (!d) return `${card("Add captures", `<p class="info">Use <strong>New project</strong> to begin.</p>`)}`;
    const caps = Object.entries(d.project.captures).sort((a, b) => (a[1].position ?? 1e9) - (b[1].position ?? 1e9));
    const st = (d.analysis && d.analysis.audit) || {};
    const idx = (d.analysis && d.analysis.capture_files) || {};
    const fileStatus = {}; Object.entries(idx).forEach(([p, fn]) => { fileStatus[fn] = st[p]; });
    const rows = caps.map(([fn, c]) => {
      const a = fileStatus[fn];
      return `<tr><td>${esc(fn)}</td><td><input type="number" step="any" class="file-input" data-cg-pos="${esc(fn)}" value="${c.position ?? ""}" placeholder="?">${c.position_suggested ? `<div class="info">suggested from the file name — please confirm</div>` : ""}</td>
        <td>${a ? badge(a.status === "VALID" || a.status === "CORRECTED" ? "instant" : "bad", a.status) : `<span class="info">not analysed</span>`}</td>
        <td><button type="button" class="btn btn-secondary btn-small" data-cg-remove="${esc(fn)}">Remove</button></td></tr>`;
    }).join("");
    const issues = d.check.issues.map((i) => `<li>${esc(i.file ? i.file + ": " : "")}${esc(i.message)}</li>`).join("");
    const setup = card("Amplifier", `
        <label class="field-label" for="cg-name">Project name</label><input class="file-input" id="cg-name" value="${esc(d.project.name)}">
        <label class="field-label" for="cg-amp">Amplifier</label><input class="file-input" id="cg-amp" placeholder="e.g. Marshall JCM800 2203" value="${esc(d.project.amp)}">
        <label class="field-label" for="cg-channel">Channel / cabinet <span class="hint">(optional)</span></label><input class="file-input" id="cg-channel" value="${esc(d.project.channel)}">`)
      + card("Add captures", `<div class="cg-drop" id="cg-drop">Drag <strong>.nam</strong> files of this one amp/channel here, or <label class="btn btn-secondary btn-small">choose files<input type="file" id="cg-files" accept=".nam" multiple hidden></label></div>
        <p class="info">Any practical number of captures. Uploading all of them gives the fullest picture; it does not mean all are used for training.</p>`);
    const main = card("Fixed-gain captures",
      (caps.length ? table(["File", "Physical gain position", "Audit", ""], rows) : `<p class="info">No captures yet.</p>`)
      + `<p class="info">${esc(d.check.summary)}</p>${issues ? `<ul class="cg-issues">${issues}</ul>` : ""}
        <button type="button" class="btn btn-primary btn-block" id="cg-analyse" ${d.check.ready && !S.job ? "" : "disabled"}>Analyse captures</button>
        <p class="info">Runs each capture through the native renderer, in parallel (usually ${formatDuration(estimateSeconds("analyse", d.check.count))} for ${d.check.count} captures on this computer). Uncertain source material is flagged, never silently fixed.</p>${jobBox()}`,
      `<span class="cost-badge cost-badge-expensive">⚡ ${formatDuration(estimateSeconds("analyse", d.check.count))}</span>`);
    return cols(setup, main);
  }
  function bindStage1() {
    const d = S.data; if (!d) return;
    const meta = () => api(`/api/cg/projects/${S.id}`, { method: "PATCH", json: { name: val("cg-name"), amp: val("cg-amp"), channel: val("cg-channel") } }).then((d) => { S.data = d; document.getElementById("cg-current-name").textContent = d.project.name; });
    ["cg-name", "cg-amp", "cg-channel"].forEach((id) => { const el = document.getElementById(id); if (el) el.addEventListener("change", meta); });
    const upload = async (files) => {
      const fd = new FormData(); [...files].forEach((f) => fd.append("files", f));
      try { const r = await api(`/api/cg/projects/${S.id}/captures`, { method: "POST", form: fd }); S.data = r; say(r.problems && r.problems.length ? r.problems.join(" · ") : `Added ${r.added.length} capture(s).`, Boolean(r.problems && r.problems.length)); render(); }
      catch (e) { say(e.message, true); }
    };
    const drop = document.getElementById("cg-drop"), fi = document.getElementById("cg-files");
    if (drop) { ["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); })); ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); })); drop.addEventListener("drop", (e) => upload(e.dataTransfer.files)); }
    if (fi) fi.addEventListener("change", () => upload(fi.files));
    body.querySelectorAll("[data-cg-pos]").forEach((el) => el.addEventListener("change", async () => {
      try { S.data = await api(`/api/cg/projects/${S.id}/positions`, { method: "PATCH", json: { positions: { [el.dataset.cgPos]: el.value } } }); render(); } catch (e) { say(e.message, true); }
    }));
    body.querySelectorAll("[data-cg-remove]").forEach((el) => el.addEventListener("click", async () => {
      try { S.data = await api(`/api/cg/projects/${S.id}/captures/${encodeURIComponent(el.dataset.cgRemove)}`, { method: "DELETE" }); render(); } catch (e) { say(e.message, true); }
    }));
    const an = document.getElementById("cg-analyse");
    if (an) an.addEventListener("click", () => runJob(`/api/cg/projects/${S.id}/analyse`, {}, async (_st, pid) => {
      await api(`/api/cg/projects/${pid}/plan`, { method: "POST", json: { mode: "automatic", anchors: "fc" } });
      if (S.id === pid) { await load(pid); S.stage = 2; render(); }
    }));
  }
  const val = (id) => (document.getElementById(id) || {}).value || "";

  // ---------- Stage 2
  function stage2() {
    const d = S.data;
    if (!d || !d.analysis) return `${card("Analyse & select", `<p class="info">Add captures and run <strong>Analyse captures</strong> first.</p>${jobBox()}`)}`;
    const a = d.analysis, plan = d.plan;
    if (!plan) return `${card("Training subset", `<p class="info">No plan yet.</p><button type="button" class="btn btn-primary btn-block" id="cg-make-plan">Propose a training plan</button>`)}`;
    const sel = new Set(plan.selected);
    const cov = plan.coverage;
    const gains = a.profile.gains;
    const names = Object.keys(a.profile.series);
    if (!S.seriesName || !a.profile.series[S.seriesName]) S.seriesName = names[0];
    const ser = a.profile.series[S.seriesName];
    const rel = a.profile.series[S.seriesName].reliable;
    const roleOf = (g) => (cov.reasons[String(g)] || {}).role;
    const chartA = chart({
      xs: gains, xLabel: "Original physical gain position", yLabel: `${S.seriesName} (${ser.unit})`,
      series: [{ ys: ser.values, dash: "4 3", color: "var(--text-muted)" }],
      xTicks: gains.filter((g) => Number.isInteger(g)),
      points: gains.map((g, i) => {
        const role = roleOf(g);
        return { x: g, y: ser.values[i], r: role === "selected" ? 6 : 4, fill: role === "selected" ? "var(--accent)" : (role === "needs_review" ? "var(--danger)" : "var(--panel)"), stroke: rel[i] ? "var(--text)" : "var(--danger)", title: `Position ${g}: ${fmt(ser.values[i], 2)} ${ser.unit} (${role || "reference"})` };
      }),
    });
    const map = plan.mapping;
    const anchors = map.filter((m) => m.kind === "training_anchor").sort((x, y) => x.input_gain_db - y.input_gain_db);
    const shade = []; for (let i = 0; i + 1 < anchors.length; i++) shade.push([anchors[i].input_gain_db, anchors[i + 1].input_gain_db, i % 2 ? "var(--accent)" : "var(--amp-b)"]);
    const mapChart = chart({
      xs: [-20, 14], xLabel: "Player Input gain (dB)", yLabel: "Original physical position", xTicks: [-20, -15, -10, -5, 0, 5, 10, 14], shade,
      series: [{ ys: [Math.min(...map.map((m) => m.position)), Math.max(...map.map((m) => m.position))], line: false }],
      points: map.map((m) => ({ x: m.input_gain_db, y: m.position, r: m.kind === "training_anchor" ? 6 : 3.5, fill: m.kind === "training_anchor" ? "var(--accent)" : "var(--panel)", label: m.kind === "training_anchor" ? `G${m.position}` : "", title: `Position ${m.position} → ${fmt(m.input_gain_db)} dB (${m.kind === "training_anchor" ? "training anchor" : "interpolated"})` })),
    });
    const audit = a.audit;
    const rows = gains.filter((g) => cov.reasons[String(g)]).map((g) => {
      const r = cov.reasons[String(g)], au = audit[String(g)] || { status: "?" };
      const kind = r.role === "selected" ? "auto" : (r.role === "needs_review" ? "bad" : "instant");
      return `<tr><td>${plan.mode === "custom" ? `<input type="checkbox" data-cg-cust="${g}" ${sel.has(g) ? "checked" : ""}> ` : ""}G${g}</td><td>${badge(kind, r.role.replace("_", " "))}</td><td>${esc(au.status)}</td><td>${esc(r.reason)}</td></tr>`;
    }).join("");
    const phys = Object.entries(cov.phys || {}).filter(([, v]) => v.tol !== null).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${fmt(v.mean, 2)}</td><td>${fmt(v.max, 2)}</td><td>${v.tol}</td><td>${badge(v.max <= v.tol ? "instant" : "bad", v.max <= v.tol ? "within" : "outside")}</td></tr>`).join("");
    const modes = `<div class="mode-tabs cg-modes" role="group" aria-label="Selection mode">${[["automatic", "Automatic"], ["use_all", "Use all"], ["custom", "Custom"]].map(([v, l]) => `<button type="button" class="mode-tab ${plan.mode === v ? "active" : ""}" data-cg-mode="${v}" aria-pressed="${plan.mode === v}">${l}</button>`).join("")}</div>`;
    const setup = card("Training subset", `${modes}
        <p class="info">${Object.keys(d.project.captures).length} captures uploaded · <strong>${plan.selected.length} selected</strong> for training (${plan.selected.map((g) => "G" + g).join(", ")}) · ${a.eligible.length} eligible · coverage rule ${a.k_star ? `met at k* = ${a.k_star}` : "<strong>not met by any subset</strong>"}.</p>
        ${plan.warnings.map((w) => `<p class="info">${esc(w)}</p>`).join("")}
        <details id="cg-advanced" ${S.advOpen || plan.anchor_method === "fixed" ? "open" : ""}><summary>Advanced: anchor method</summary>
          <label class="field-label" for="cg-anchors">Anchor method</label><select id="cg-anchors" class="select-input"><option value="fc" ${plan.anchor_method === "fc" ? "selected" : ""}>FC response-distance anchors (default, production recipe)</option><option value="fixed" ${plan.anchor_method === "fixed" ? "selected" : ""}>Fixed 4 dB spacing (v3 alternative)</option></select></details>
`);
    const main = card("How the source amp changes", `<p class="info">Measured on the fit DIs; dashed line is the interpolation, markers are the captures (filled = selected, red ring = quarantined for this measure).</p>
        <select id="cg-series" class="select-input">${names.map((n) => `<option ${n === S.seriesName ? "selected" : ""}>${esc(n)}</option>`).join("")}</select>${chartA}`)
      + card("How the finished NAM will be controlled", `<p class="info">The Input-gain mapping used to build the training target. Shaded bands are untested regions between anchors, learned by interpolation. This is a control guide, not a physical-gain parameter.</p>${mapChart}`)
      + card("Why each capture", table(["Capture", "Role", "Audit", "Measured reason"], rows))
      + card("Measured coverage of the omitted captures", table(["Group", "Mean error", "Max error", "Working tolerance", ""], phys || `<tr><td colspan="5">Every eligible capture is selected, so nothing is omitted.</td></tr>`)
        + `<p class="info">Tolerances are working values, not perceptual measurements. Objective J = ${fmt(cov.J, 3)}.</p>
        <button type="button" class="btn btn-primary btn-block" id="cg-accept">Review complete — continue to Train</button>`);
    return cols(setup, main);
  }
  function bindStage2() {
    const d = S.data; if (!d) return;
    const mk = document.getElementById("cg-make-plan");
    // Plan changes are serialised and always start from the CURRENT plan (never one captured when the handlers were bound),
    // so a quick second change (e.g. Automatic right after switching the anchor method) cannot silently undo the first.
    const replan = (o) => {
      S.planQueue = (S.planQueue || Promise.resolve()).then(async () => {
        const cur = S.data.plan || {};
        const mode = o.mode ?? cur.mode ?? "automatic";
        const anchors = o.anchors ?? cur.anchor_method ?? "fc";
        const custom = mode === "custom" ? (o.custom ?? cur.selected) : null;
        try { S.data = await api(`/api/cg/projects/${S.id}/plan`, { method: "POST", json: { mode, custom, anchors } }); render(); } catch (e) { say(e.message, true); }
      });
    };
    if (mk) mk.addEventListener("click", () => replan({ mode: "automatic", anchors: "fc" }));
    if (!d.plan) return;
    body.querySelectorAll("[data-cg-mode]").forEach((el) => el.addEventListener("click", () => replan({ mode: el.dataset.cgMode })));
    body.querySelectorAll("[data-cg-cust]").forEach((el) => el.addEventListener("change", () => {
      const chosen = [...body.querySelectorAll("[data-cg-cust]")].filter((x) => x.checked).map((x) => Number(x.dataset.cgCust));
      replan({ mode: "custom", custom: chosen });
    }));
    const adv = document.getElementById("cg-advanced"); if (adv) adv.addEventListener("toggle", () => { S.advOpen = adv.open; });
    const an = document.getElementById("cg-anchors"); if (an) an.addEventListener("change", () => replan({ anchors: an.value }));
    const se = document.getElementById("cg-series"); if (se) se.addEventListener("change", () => { S.seriesName = se.value; render(); });
    const ac = document.getElementById("cg-accept"); if (ac) ac.addEventListener("click", async () => { await (S.planQueue || Promise.resolve()); S.stage = 3; render(); });
  }

  // ---------- Stage 3
  function stage3() {
    const d = S.data;
    if (!d || !d.plan) return `${card("Train", `<p class="info">Review the training plan in stage 2 first.</p>`)}`;
    const p = d.plan, b = d.bundle, t = d.training;
    const bundledCab = b && b.cab;
    const currentCabSha = S.cabEnabled && S.cab ? S.cab.sha256 : null;
    const bundledCabSha = bundledCab && bundledCab.selected ? bundledCab.sha256 : null;
    const stale = b && ((b.plan && b.plan.planned !== p.planned) || currentCabSha !== bundledCabSha ||
      (currentCabSha && (S.cabDisplayName || "") !== (bundledCab.display_name || "")));
    const range = [Math.min(...p.anchors_input_gain_db), Math.max(...p.anchors_input_gain_db)];
    const files = card("Training files", `
        <label class="field-label" for="cg-model-name">Model name</label><input class="file-input" id="cg-model-name" maxlength="100" value="${esc(d.project.name)}">
        <label class="field-label" for="cg-cab-file">Cabinet IR <span class="hint">(optional second NAM)</span></label>
        <input type="file" id="cg-cab-file" accept=".wav" class="file-input">
        <div id="cg-cab-info" class="info">${S.cab ? `${esc(S.cab.filename || "Cabinet IR")} selected. The trained and tested NAM remains cabless.` : "No cabinet selected."}</div>
        <label class="checkbox-row"><input type="checkbox" id="cg-cab-enabled" ${S.cabEnabled && S.cab ? "checked" : ""} ${S.cab ? "" : "disabled"}> Also create a second NAM with this exact cabinet embedded</label>
        <label class="field-label" for="cg-cab-name">Cabinet display name</label>
        <input class="file-input" id="cg-cab-name" maxlength="80" value="${esc(S.cabDisplayName)}" ${S.cab ? "" : "disabled"} placeholder="e.g. Modern Boutique 4x12">
        <p class="info">Training and validation use the head-only signal. Afterward NAM Mixer derives a separate head + cabinet NAM. The cabinet version uses NAM Sequential/Linear and may not load in A2-only players.</p>
        <button type="button" class="btn btn-primary btn-block" id="cg-generate" ${S.job ? "disabled" : ""}>${b ? "Recreate training files" : "Create training files"}</button>
        <p class="info">Renders the training audio through the ${p.selected.length} selected capture(s), in parallel -- usually ${formatDuration(estimateSeconds("generate", p.selected.length))}.</p>
        ${stale ? `<p class="info"><strong>The plan changed since these files were created — recreate them before training.</strong></p>` : ""}${jobBox()}
        ${b && t && t.core ? `<p class="info">Design <code>${esc(b.design_id)}</code><br>input <code>${esc(t.core.input_audio_sha256.slice(0, 12))}…</code> · target <code>${esc(t.core.target_audio_sha256.slice(0, 12))}…</code><br>output scale ${fmt(t.core.output_scale_c, 4)} — set the player's Output gain to ${fmt(t.core.peak_ceiling_gain_reduction_db, 1)} dB.</p>` : ""}`);
    const train = b && !stale ? card("Train", `<div id="cg-train-slot"></div>${t && t.trained ? `<p class="info">${badge("instant", "trained")} <code>${esc(t.output_nam_path.split("/").pop())}</code></p><button type="button" class="btn btn-primary btn-block" id="cg-goto4">Continue to Test &amp; export</button>` : ""}`) : "";
    const plan = card("Training plan", table([], `
          <tr><td>Amplifier</td><td>${esc(d.project.amp || "-")} ${esc(d.project.channel)}</td></tr>
          <tr><td>Captures</td><td>${Object.keys(d.project.captures).length} uploaded · ${p.selected.length} selected for training (${p.selected.map((g) => "G" + g).join(", ")})</td></tr>
          <tr><td>Input gain range</td><td>${fmt(range[0])} to ${fmt(range[1])} dB (${p.anchor_method === "fc" ? "FC response-distance anchors" : "fixed 4 dB spacing — Advanced"})</td></tr>
          <tr><td>Output compensation</td><td>${b ? "recorded in the training files (below)" : "one constant, computed when the training target is built"}</td></tr>
          <tr><td>Result</td><td>One standard <code>.nam</code>, a manifest and a player guide</td></tr>`))
      + card("Anchors", table(["Position", "Input gain", "Kind"], p.mapping.filter((m) => m.kind === "training_anchor").map((m) => `<tr><td>G${m.position}</td><td>${fmt(m.input_gain_db)} dB</td><td>training anchor</td></tr>`).join("")));
    const setup = plan, main = files + train;
    return cols(setup, main);
  }
  function bindStage3() {
    const d = S.data; if (!d || !d.plan) return;
    const cabFile = document.getElementById("cg-cab-file");
    if (cabFile) cabFile.addEventListener("change", async () => {
      const file = cabFile.files[0];
      if (!file) return;
      const form = new FormData(); form.append("file", file);
      try {
        const uploaded = await api("/api/cab/upload", { method: "POST", form });
        S.cab = uploaded; S.cabEnabled = true;
        if (!S.cabDisplayName) S.cabDisplayName = file.name.replace(/\.wav$/i, "");
        say(`Cabinet ${file.name} selected.`); render();
      } catch (e) { say(e.message, true); }
    });
    const cabEnabled = document.getElementById("cg-cab-enabled");
    if (cabEnabled) cabEnabled.addEventListener("change", () => { S.cabEnabled = cabEnabled.checked; render(); });
    const cabName = document.getElementById("cg-cab-name");
    if (cabName) cabName.addEventListener("change", () => { S.cabDisplayName = cabName.value.trim(); render(); });
    const g = document.getElementById("cg-generate");
    if (g) g.addEventListener("click", () => runJob(`/api/cg/projects/${S.id}/generate`, {
      model_name: val("cg-model-name"),
      cab_path: S.cabEnabled && S.cab ? S.cab.path : null,
      cab_display_name: S.cabDisplayName,
    }, async (_st, pid) => { if (S.id === pid) { say("Training files created."); render(); } }));
    const g4 = document.getElementById("cg-goto4"); if (g4) g4.addEventListener("click", () => { S.stage = 4; render(); });
    // The Kaggle / local training UI is the Builder's own section (app.js), hosted here for this design.
    const slot = document.getElementById("cg-train-slot");
    if (slot && d.bundle) {
      if (!S.trainHost) { S.trainHost = document.createElement("div"); S.trainHost.className = "cg-train-host"; }
      slot.replaceWith(S.trainHost);
      if (!window.namTrainingHost.attach(S.trainHost, d.bundle.design_id)) S.trainHost.innerHTML = `<p class="info">Another training is running (see the Builder). Wait for it to finish, then reopen this stage.</p>`;
    }
  }
  // The hosted section announces completion; reload so the project shows the trained model.
  document.addEventListener("nam:training-complete", (e) => {
    if (S.data && S.data.bundle && S.data.bundle.design_id === e.detail.designId && !(S.data.training && S.data.training.trained)) {
      load(S.id).catch((err) => say(`Training finished, but the project could not be reloaded: ${err.message}`, true));
    }
  });

  // ---------- Stage 4
  function stage4() {
    const d = S.data;
    if (!d || !d.training || !d.training.trained) return `${card("Test & export", `<p class="info">Train a model first (stage 3). Testing and export unlock once a trained .nam exists.</p>`)}`;
    const v = d.validation;
    const cabReady = d.training.embedded_artifact && d.training.embedded_artifact.state === "validated";
    const actions = card("Test & export", `
        <button type="button" class="btn btn-secondary btn-block" id="cg-validate" ${S.job ? "disabled" : ""}>${v ? "Re-run validation" : "Run validation"}</button>
        <p class="info">Usually ${formatDuration(estimateSeconds("validate", d.plan ? d.plan.selected.length : 4))} -- renders comparisons and the audition sweep, in parallel.</p>
        <a class="btn btn-primary btn-block btn-download-artifact" href="/api/cg/projects/${S.id}/nam/download?artifact=head" download>Download tested head-only NAM</a>
        ${cabReady ? `<a class="btn btn-secondary btn-block btn-download-artifact" href="/api/cg/projects/${S.id}/nam/download?artifact=cab" download>Download NAM with embedded cabinet</a>` : ""}
        <a class="btn btn-secondary btn-block btn-download-artifact" href="/api/cg/projects/${S.id}/export" download>Download provenance package</a>
        <p class="info">Validation is always performed on the head-only NAM against your original captures. The cabinet download is a separately validated exact derivative; some A2-only players may not support its Sequential architecture.</p>${jobBox()}`);
    if (!v) return cols(actions, card("Validation", `<p class="info">No validation has been run for this model.</p>`));
    const c = v.compatibility, sf = v.safety, pg = v.progression;
    const chip = (ok, yes, no) => badge(ok ? "instant" : "bad", ok ? yes : no);
    const setup = actions
      + card("Standard NAM compatibility", `<p>${chip(c.standard_nam, "standard .nam", "not verified")} ${chip(c.full.rendered_ok, "Full renders", "Full failed")} ${chip(c.lite && c.lite.rendered_ok, "Lite renders", "Lite failed")}</p><p class="info">architecture ${esc(c.architecture)} · sample rate ${esc(c.sample_rate)} · no extra runtime processing required</p>`)
      + card("Output safety", `<p class="info">Set the player's <strong>Output gain to ${fmt(sf.recommended_output_gain_db, 1)} dB</strong> and leave it. ${sf.scaled_peak_over_0dbfs_at_input_gain_db.length ? `<strong>Peaks above 0 dBFS at Input gain ${sf.scaled_peak_over_0dbfs_at_input_gain_db.join(", ")} dB</strong> with this DI — lower the output or the Input gain there.` : "No peaks above 0 dBFS across the Input-gain range on this DI."}</p>`);
    const au = v.audition, cov = v.coverage;
    const main = card("Gain progression vs your captures", table(["Position", "Input gain", "Role", "Level Δ dB", "HF Δ", "Crest Δ", "Dyn. range Δ", "EQ max Δ", "Level-matched ESR"],
          pg.positions.map((r) => `<tr><td>G${r.position}</td><td>${fmt(r.input_gain_db)} dB</td><td>${r.role === "training" ? badge("auto", "training") : `<span class="value-chip">reference</span>`}</td><td>${fmt(r.level_db, 2)}</td><td>${fmt(r.hf_db, 2)}</td><td>${fmt(r.crest_db, 2)}</td><td>${fmt(r.dyn_range_db, 2)}</td><td>${fmt(r.eq_max_db, 2)}</td><td>${fmt(r.lm_esr, 4)}</td></tr>`).join(""))
        + `<p class="info">Held-out DIs: ${esc(pg.held_out_dis.join(", "))}. Δ = model minus real capture at the plan's Input gain, one global output scale. “reference” rows were not training anchors, so they are independent evidence for the interpolated positions. ${pg.reversals.length ? `<strong>${pg.reversals.length} direction reversal(s)</strong> versus the real amp: ${pg.reversals.map((r) => `${r.measure} between G${r.between[0]}–G${r.between[1]}`).join("; ")}.` : "No direction reversals versus the real amp's progression (level, HF, crest)."}</p>`)
      + (cov ? card("Measured coverage of your uploaded captures", `<p class="info">${cov.selected.length} captures trained; ${cov.all_within_tolerance ? "every omitted capture is within the working tolerances of what the selection predicts" : "some omitted captures fall outside the working tolerances (see stage 2)"}.</p>`) : "")
      + card("Audition (optional)", `<p class="info">Sweep of the whole Input-gain range on one DI, ${au.sweep.seconds_per_step} s per step from ${au.sweep.input_gains_db[0]} to ${au.sweep.input_gains_db[au.sweep.input_gains_db.length - 1]} dB in 2 dB steps${au.sweep.attenuated_db ? ` (attenuated ${fmt(au.sweep.attenuated_db, 1)} dB as a whole for safe playback)` : ""}.</p>
        <audio controls preload="none" src="/api/cg/projects/${S.id}/audition/${au.sweep.file}"></audio>
        <label class="field-label" for="cg-cmp">Compare at position</label><select id="cg-cmp" class="select-input">${au.comparisons.map((c2, i) => `<option value="${i}">G${c2.position} (${fmt(c2.input_gain_db)} dB Input gain)</option>`).join("")}</select>
        <div class="cg-audio-pair" id="cg-cmp-pair"></div>
        <p class="info">The comparison plays the trained model and the real capture on the same DI at the same level. There is no captured sound at an interpolated point, so “reference” positions are the meaningful comparisons.</p>`);
    return cols(setup, main);
  }
  function bindStage4() {
    const d = S.data; if (!d) return;
    const v = document.getElementById("cg-validate");
    if (v) v.addEventListener("click", () => runJob(`/api/cg/projects/${S.id}/validate`, {}, async (_st, pid) => { if (S.id === pid) { say("Validation complete."); render(); } }));
    const sel = document.getElementById("cg-cmp");
    if (sel && d.validation) {
      const show = () => { const c = d.validation.audition.comparisons[Number(sel.value)]; document.getElementById("cg-cmp-pair").innerHTML = `<div><strong>Trained model</strong><audio controls preload="none" src="/api/cg/projects/${S.id}/audition/${c.model}"></audio></div><div><strong>Original capture G${c.position}</strong><audio controls preload="none" src="/api/cg/projects/${S.id}/audition/${c.original}"></audio></div>`; };
      sel.addEventListener("change", show); show();
    }
  }

  // ---------- render
  function render() {
    if (S.stage !== 3) window.namTrainingHost.detach();
    if (S.trainHost) S.trainHost.remove();          // keep the borrowed section alive while the body is rebuilt
    document.querySelectorAll(".cg-tab").forEach((b) => b.classList.toggle("active", Number(b.dataset.cgStage) === S.stage));
    const hint = document.getElementById("cg-hint"); if (hint) hint.textContent = HINTS[S.stage] || "";
    if (!S.data) { body.innerHTML = `${card("Continuous Gain", `<p class="info">No project selected. Use <strong>New project</strong> to start.</p>`)}`; return; }
    body.innerHTML = [stage1, stage2, stage3, stage4][S.stage - 1]();
    [bindStage1, bindStage2, bindStage3, bindStage4][S.stage - 1]();
  }
})();
