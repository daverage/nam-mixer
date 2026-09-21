// Continuous Gain tab: Add captures -> Analyse & select -> Train -> Test & export.
// Thin UI over /api/cg/*; training itself goes through the EXISTING /api/local_training/* and /api/kaggle/* routes.
(() => {
  const panel = document.getElementById("cg-panel");
  const tab = document.getElementById("tab-cg");
  if (!panel || !tab) return;
  const body = document.getElementById("cg-body");
  const msg = document.getElementById("cg-message");
  const picker = document.getElementById("cg-project-select");

  const S = { projects: [], id: null, data: null, stage: 1, job: null, pollTimer: null, seriesName: null, train: null, backend: "local", preset: "standard" };
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "-" : Number(v).toFixed(d));
  const say = (t, bad) => { msg.textContent = t || ""; msg.classList.toggle("is-error", Boolean(bad)); };

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
      refreshList();
    }
  }
  tab.addEventListener("click", () => setOpen(true));
  document.querySelectorAll(".utility-tabs .mode-tab").forEach((b) => {
    if (b !== tab) b.addEventListener("click", () => { if (!panel.hidden) setOpen(false); }, true);
  });

  // ---------- projects
  async function refreshList(selectId) {
    S.projects = await api("/api/cg/projects");
    picker.innerHTML = S.projects.length ? S.projects.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}${p.amp ? " · " + esc(p.amp) : ""}</option>`).join("") : `<option value="">No projects yet</option>`;
    const id = selectId || (S.projects.find((p) => p.id === S.id) ? S.id : (S.projects[0] && S.projects[0].id));
    if (id) { picker.value = id; await load(id); } else { S.id = null; S.data = null; render(); }
  }
  async function load(id) {
    S.id = id;
    S.data = await api(`/api/cg/projects/${id}`);
    render();
  }
  picker.addEventListener("change", () => load(picker.value));
  document.getElementById("cg-new-project").addEventListener("click", async () => {
    const name = window.prompt("Project name (e.g. the amp and channel)");
    if (!name) return;
    try { const d = await api("/api/cg/projects", { method: "POST", json: { name } }); await refreshList(d.project.id); S.stage = 1; render(); } catch (e) { say(e.message, true); }
  });
  document.getElementById("cg-delete-project").addEventListener("click", async () => {
    if (!S.id || !(await desktopConfirm("Delete this Continuous Gain project and its uploaded captures? Generated training files stay in the app's work folder."))) return;
    try { await api(`/api/cg/projects/${S.id}`, { method: "DELETE" }); S.id = null; await refreshList(); } catch (e) { say(e.message, true); }
  });
  document.getElementById("cg-steps").addEventListener("click", (e) => {
    const b = e.target.closest("[data-cg-stage]"); if (!b) return;
    S.stage = Number(b.dataset.cgStage); render();
  });

  // ---------- jobs
  async function runJob(startPath, payload, after) {
    try {
      const j = await api(startPath, { method: "POST", json: payload || {} });
      S.job = { id: j.job_id, message: "starting", log: [] };
      render();
      clearInterval(S.pollTimer);
      S.pollTimer = setInterval(async () => {
        try {
          const st = await api(`/api/cg/jobs/${j.job_id}`);
          S.job = { id: j.job_id, message: st.message, log: st.log, elapsed: st.elapsed };
          if (st.state !== "running") {
            clearInterval(S.pollTimer);
            const failed = st.state === "error";
            S.job = null;
            await load(S.id);
            if (failed) say(st.error || "The job failed", true); else if (after) await after(st);
          } else { const el = document.getElementById("cg-job-log"); if (el) { el.textContent = st.log.join("\n"); el.scrollTop = el.scrollHeight; } const m = document.getElementById("cg-job-msg"); if (m) m.textContent = `${st.message} (${Math.round(st.elapsed)} s)`; }
        } catch (e) { clearInterval(S.pollTimer); say(e.message, true); }
      }, 1500);
    } catch (e) { say(e.message, true); }
  }
  const jobBox = () => S.job ? `<div class="cg-row"><strong id="cg-job-msg">${esc(S.job.message)}</strong></div><pre class="cg-log" id="cg-job-log">${esc((S.job.log || []).join("\n"))}</pre>` : "";

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

  // ---------- Stage 1
  function stage1() {
    const d = S.data;
    if (!d) return `<p class="info">Create a project to begin.</p>`;
    const caps = Object.entries(d.project.captures).sort((a, b) => (a[1].position ?? 1e9) - (b[1].position ?? 1e9));
    const st = (d.analysis && d.analysis.audit) || {};
    const idx = (d.analysis && d.analysis.capture_files) || {};
    const fileStatus = {}; Object.entries(idx).forEach(([p, fn]) => { fileStatus[fn] = st[p]; });
    const rows = caps.map(([fn, c]) => {
      const a = fileStatus[fn];
      return `<tr><td>${esc(fn)}</td><td><input type="number" step="any" class="file-input" data-cg-pos="${esc(fn)}" value="${c.position ?? ""}" placeholder="?">${c.position_suggested ? ` <span class="cg-warn">suggested from the file name — please confirm</span>` : ""}</td>
        <td>${a ? `<span class="cg-chip ${a.status === "VALID" || a.status === "CORRECTED" ? "ok" : "bad"}">${esc(a.status)}</span>` : `<span class="cg-warn">not analysed</span>`}</td>
        <td><button type="button" class="btn btn-secondary btn-small" data-cg-remove="${esc(fn)}">Remove</button></td></tr>`;
    }).join("");
    const issues = d.check.issues.map((i) => `<li>${esc(i.file ? i.file + ": " : "")}${esc(i.message)}</li>`).join("");
    return `<section><h3>Amplifier</h3><div class="cg-row">
        <input class="file-input" id="cg-name" placeholder="Project name" value="${esc(d.project.name)}">
        <input class="file-input" id="cg-amp" placeholder="Amplifier" value="${esc(d.project.amp)}">
        <input class="file-input" id="cg-channel" placeholder="Channel / cabinet (optional)" value="${esc(d.project.channel)}"></div></section>
      <section><h3>Fixed-gain captures</h3>
        <div class="cg-drop" id="cg-drop">Drag <strong>.nam</strong> captures of this one amp/channel here, or <label class="btn btn-secondary btn-small">choose files<input type="file" id="cg-files" accept=".nam" multiple hidden></label>
          <div class="cg-warn">Any practical number of captures. Uploading all of them gives the fullest picture; it does not mean all are used for training.</div></div>
        ${caps.length ? `<table class="cg-table"><thead><tr><th>File</th><th>Physical gain position</th><th>Audit</th><th></th></tr></thead><tbody>${rows}</tbody></table>` : ""}
        <p class="info">${esc(d.check.summary)}</p>${issues ? `<ul class="cg-issues">${issues}</ul>` : ""}
        <div class="cg-row"><button type="button" class="btn btn-primary" id="cg-analyse" ${d.check.ready && !S.job ? "" : "disabled"}>Analyse captures</button>
          <span class="cg-warn">Runs each capture through the native renderer (about ${Math.max(1, Math.round(d.check.count * 0.15))} min). Uncertain source material is flagged, never silently fixed.</span></div>${jobBox()}</section>`;
  }
  function bindStage1() {
    const d = S.data; if (!d) return;
    const meta = () => api(`/api/cg/projects/${S.id}`, { method: "PATCH", json: { name: val("cg-name"), amp: val("cg-amp"), channel: val("cg-channel") } }).then(() => refreshListKeep());
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
    if (an) an.addEventListener("click", () => runJob(`/api/cg/projects/${S.id}/analyse`, {}, async () => { await api(`/api/cg/projects/${S.id}/plan`, { method: "POST", json: { mode: "automatic", anchors: "fc" } }); await load(S.id); S.stage = 2; render(); }));
  }
  const val = (id) => (document.getElementById(id) || {}).value || "";
  const refreshListKeep = async () => { S.projects = await api("/api/cg/projects"); const cur = picker.value; picker.innerHTML = S.projects.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}${p.amp ? " · " + esc(p.amp) : ""}</option>`).join(""); picker.value = cur; };

  // ---------- Stage 2
  function stage2() {
    const d = S.data;
    if (!d || !d.analysis) return `<p class="info">Add captures and run <strong>Analyse captures</strong> first.</p>${jobBox()}`;
    const a = d.analysis, plan = d.plan;
    if (!plan) return `<p class="info">No plan yet.</p><button type="button" class="btn btn-primary" id="cg-make-plan">Propose a training plan</button>`;
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
    const rows = gains.filter((g) => Number.isInteger(g) || !(cov.reasons[String(g)])).filter((g) => cov.reasons[String(g)]).map((g) => {
      const r = cov.reasons[String(g)], au = audit[String(g)] || { status: "?" };
      return `<tr><td>${plan.mode === "custom" ? `<input type="checkbox" data-cg-cust="${g}" ${sel.has(g) ? "checked" : ""}> ` : ""}G${g}</td><td><span class="cg-chip ${r.role}">${esc(r.role.replace("_", " "))}</span></td><td>${esc(au.status)}</td><td>${esc(r.reason)}</td></tr>`;
    }).join("");
    const phys = Object.entries(cov.phys || {}).filter(([, v]) => v.tol !== null).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${fmt(v.mean, 2)}</td><td>${fmt(v.max, 2)}</td><td>${v.tol}</td><td><span class="cg-chip ${v.max <= v.tol ? "ok" : "bad"}">${v.max <= v.tol ? "within" : "outside"}</span></td></tr>`).join("");
    const modes = [["automatic", "Automatic"], ["use_all", "Use all"], ["custom", "Custom"]].map(([v, l]) => `<label><input type="radio" name="cg-mode" value="${v}" ${plan.mode === v ? "checked" : ""}> ${l}</label>`).join(" ");
    return `<section><h3>Training subset</h3>
        <div class="cg-row">${modes}</div>
        <p class="info">${d.project.captures ? Object.keys(d.project.captures).length : 0} captures uploaded · <strong>${plan.selected.length} selected</strong> for training (${plan.selected.map((g) => "G" + g).join(", ")}) · ${a.eligible.length} eligible · coverage rule ${a.k_star ? `met at k* = ${a.k_star}` : "<strong>not met by any subset</strong>"}.</p>
        ${plan.warnings.map((w) => `<p class="cg-warn">${esc(w)}</p>`).join("")}
        <details><summary>Advanced</summary><div class="cg-row"><label>Anchor method <select id="cg-anchors" class="select-input"><option value="fc" ${plan.anchor_method === "fc" ? "selected" : ""}>FC response-distance anchors (default, production recipe)</option><option value="fixed" ${plan.anchor_method === "fixed" ? "selected" : ""}>Fixed 4 dB spacing (v3 alternative)</option></select></label></div></details></section>
      <section class="cg-two"><div><h3>How the source amp changes</h3><p class="cg-warn">Measured on the fit DIs; dashed line is the interpolation, markers are the captures (filled = selected, red ring = quarantined for this measure).</p>
          <select id="cg-series" class="select-input">${names.map((n) => `<option ${n === S.seriesName ? "selected" : ""}>${esc(n)}</option>`).join("")}</select>${chartA}</div>
        <div><h3>How the finished NAM will be controlled</h3><p class="cg-warn">The Input-gain mapping used to build the training target. Shaded bands are untested regions between anchors, learned by interpolation. This is a control guide, not a physical-gain parameter.</p>${mapChart}</div></section>
      <section><h3>Why each capture</h3><table class="cg-table"><thead><tr><th>Capture</th><th>Role</th><th>Audit</th><th>Measured reason</th></tr></thead><tbody>${rows}</tbody></table></section>
      <section><h3>Measured coverage of the omitted captures</h3><table class="cg-table"><thead><tr><th>Group</th><th>Mean error</th><th>Max error</th><th>Working tolerance</th><th></th></tr></thead><tbody>${phys || `<tr><td colspan="5">Every eligible capture is selected, so nothing is omitted.</td></tr>`}</tbody></table>
        <p class="cg-warn">Tolerances are working values, not perceptual measurements. Objective J = ${fmt(cov.J, 3)}.</p></section>
      <div class="cg-row"><button type="button" class="btn btn-primary" id="cg-accept">Review complete — continue to Train</button></div>`;
  }
  function bindStage2() {
    const d = S.data; if (!d) return;
    const mk = document.getElementById("cg-make-plan");
    const replan = async (mode, custom, anchors) => { try { S.data = await api(`/api/cg/projects/${S.id}/plan`, { method: "POST", json: { mode, custom, anchors } }); render(); } catch (e) { say(e.message, true); } };
    if (mk) mk.addEventListener("click", () => replan("automatic", null, "fc"));
    if (!d.plan) return;
    body.querySelectorAll("input[name=cg-mode]").forEach((el) => el.addEventListener("change", () => replan(el.value, el.value === "custom" ? d.plan.selected : null, d.plan.anchor_method)));
    body.querySelectorAll("[data-cg-cust]").forEach((el) => el.addEventListener("change", () => {
      const chosen = [...body.querySelectorAll("[data-cg-cust]")].filter((x) => x.checked).map((x) => Number(x.dataset.cgCust));
      replan("custom", chosen, d.plan.anchor_method);
    }));
    const an = document.getElementById("cg-anchors"); if (an) an.addEventListener("change", () => replan(d.plan.mode, d.plan.mode === "custom" ? d.plan.selected : null, an.value));
    const se = document.getElementById("cg-series"); if (se) se.addEventListener("change", () => { S.seriesName = se.value; render(); });
    const ac = document.getElementById("cg-accept"); if (ac) ac.addEventListener("click", () => { S.stage = 3; render(); });
  }

  // ---------- Stage 3
  function stage3() {
    const d = S.data;
    if (!d || !d.plan) return `<p class="info">Review the training plan in stage 2 first.</p>`;
    const p = d.plan, b = d.bundle, t = d.training;
    const stale = b && b.plan && b.plan.planned !== p.planned;
    const range = [Math.min(...p.anchors_input_gain_db), Math.max(...p.anchors_input_gain_db)];
    const presets = Object.entries(d.epoch_presets).map(([k, v]) => `<option value="${k}" ${k === S.preset ? "selected" : ""}>${k === "standard" ? "Standard (established recipe)" : k === "draft" ? "Draft (quick preview)" : "High definition"} · ${v} epochs</option>`).join("");
    const trainerBox = `<div id="cg-train-status" class="info"></div>`;
    return `<section><h3>Training plan</h3>
        <table class="cg-table"><tbody>
          <tr><th>Amplifier</th><td>${esc(d.project.amp || "-")} ${esc(d.project.channel)}</td></tr>
          <tr><th>Captures</th><td>${Object.keys(d.project.captures).length} uploaded · ${p.selected.length} selected for training (${p.selected.map((g) => "G" + g).join(", ")})</td></tr>
          <tr><th>Input gain range</th><td>${fmt(range[0])} to ${fmt(range[1])} dB (${p.anchor_method === "fc" ? "FC response-distance anchors" : "fixed 4 dB spacing — Advanced"})</td></tr>
          <tr><th>Output compensation</th><td>${b ? "set after generation (see below)" : "one constant, computed when the training target is built"}</td></tr>
          <tr><th>Result</th><td>One standard <code>.nam</code>, a manifest and a player guide</td></tr></tbody></table></section>
      <section><h3>Training settings</h3><div class="cg-row">
        <input class="file-input" id="cg-model-name" maxlength="100" placeholder="Model name" value="${esc((b && "") || d.project.name)}">
        <label>Epochs <select id="cg-preset" class="select-input">${presets}</select></label>
        <label>Backend <select id="cg-backend" class="select-input"><option value="local" ${S.backend === "local" ? "selected" : ""}>This computer</option><option value="kaggle" ${S.backend === "kaggle" ? "selected" : ""}>Kaggle GPU</option></select></label></div>
        <p class="cg-warn">More epochs do not guarantee a better model. The standard preset is the one the frozen recipe used (60 epochs).</p>
        <div class="cg-row"><button type="button" class="btn btn-primary" id="cg-generate" ${S.job ? "disabled" : ""}>${b ? "Regenerate training files" : "Generate training files"}</button>
          ${stale ? `<span class="cg-warn">The plan changed since these files were generated — regenerate before training.</span>` : ""}</div>${jobBox()}</section>
      ${b ? `<section><h3>Training files</h3><p class="info">Design <code>${esc(b.design_id)}</code>: ${t && t.core ? `input <code>${esc(t.core.input_audio_sha256.slice(0, 12))}…</code>, target <code>${esc(t.core.target_audio_sha256.slice(0, 12))}…</code>, output scale ${fmt(t.core.output_scale_c, 4)} (set the player's Output gain to ${fmt(t.core.peak_ceiling_gain_reduction_db, 1)} dB).` : ""}</p>
        <div class="cg-row"><button type="button" class="btn btn-primary" id="cg-train" ${stale || S.job ? "disabled" : ""}>Start training</button>${S.backend === "local" ? `<button type="button" class="btn btn-secondary" id="cg-setup">Set up local training</button><button type="button" class="btn btn-secondary" id="cg-cancel">Cancel</button>` : ""}</div>
        ${trainerBox}${t && t.trained ? `<p class="info"><span class="cg-chip ok">trained</span> ${esc(t.output_nam_path)} <button type="button" class="btn btn-primary btn-small" id="cg-goto4">Continue to Test &amp; export</button></p>` : ""}</section>` : ""}`;
  }
  async function pollTrainer() {
    clearInterval(S.trainTimer);
    const el = () => document.getElementById("cg-train-status");
    const tick = async () => {
      if (!el() || S.stage !== 3) { clearInterval(S.trainTimer); return; }
      try {
        if (S.backend === "local") {
          const s = await api("/api/local_training/status");
          const pr = s.progress ? ` · epoch ${s.progress.epoch}/${s.progress.total_epochs}` : "";
          el().textContent = `Local training: ${s.state}${pr}${s.state === "not_configured" ? " — click “Set up local training” first" : ""}`;
          if (s.state === "complete") { clearInterval(S.trainTimer); await load(S.id); }
        } else if (S.train && S.train.job_id) {
          const s = await api(`/api/kaggle/jobs/${S.train.job_id}?design_id=${encodeURIComponent(S.data.bundle.design_id)}`);
          el().textContent = `Kaggle: ${s.state}${s.error ? " — " + s.error : ""}`;
          if (s.state === "complete") { clearInterval(S.trainTimer); await load(S.id); }
        }
      } catch (e) { el().textContent = e.message; }
    };
    S.trainTimer = setInterval(tick, 4000); tick();
  }
  function bindStage3() {
    const d = S.data; if (!d || !d.plan) return;
    const pre = document.getElementById("cg-preset"); if (pre) pre.addEventListener("change", () => { S.preset = pre.value; });
    const be = document.getElementById("cg-backend"); if (be) be.addEventListener("change", () => { S.backend = be.value; render(); });
    const g = document.getElementById("cg-generate");
    if (g) g.addEventListener("click", () => runJob(`/api/cg/projects/${S.id}/generate`, { model_name: val("cg-model-name") }, async () => { say("Training files generated."); render(); }));
    const tr = document.getElementById("cg-train");
    if (tr) tr.addEventListener("click", async () => {
      try {
        if (S.backend === "local") { await api("/api/local_training/start", { method: "POST", json: { design_id: d.bundle.design_id, epoch_preset: S.preset } }); }
        else { S.train = await api("/api/kaggle/train", { method: "POST", json: { design_id: d.bundle.design_id, epoch_preset: S.preset } }); }
        say("Training started."); pollTrainer();
      } catch (e) { say(e.message, true); }
    });
    const su = document.getElementById("cg-setup"); if (su) su.addEventListener("click", async () => { try { await api("/api/local_training/setup", { method: "POST", json: {} }); say("Local training setup started."); pollTrainer(); } catch (e) { say(e.message, true); } });
    const ca = document.getElementById("cg-cancel"); if (ca) ca.addEventListener("click", async () => { try { await api("/api/local_training/cancel", { method: "POST", json: {} }); } catch (e) { say(e.message, true); } });
    const g4 = document.getElementById("cg-goto4"); if (g4) g4.addEventListener("click", () => { S.stage = 4; render(); });
    if (d.bundle) pollTrainer();
  }

  // ---------- Stage 4
  function stage4() {
    const d = S.data;
    if (!d || !d.training || !d.training.trained) return `<p class="info">Train a model first (stage 3). Testing and export unlock once a trained .nam exists.</p>`;
    const v = d.validation;
    const exportBtn = `<a class="btn btn-primary" href="/api/cg/projects/${S.id}/export" download>Download export package (.nam + guide)</a>`;
    let html = `<section><div class="cg-row"><button type="button" class="btn btn-secondary" id="cg-validate" ${S.job ? "disabled" : ""}>${v ? "Re-run validation" : "Run validation"}</button>${exportBtn}</div>
      <p class="cg-warn">Export is never blocked by validation or by listening. Validation reports measurements against your original captures on held-out DIs; it is not a perceptual score.</p>${jobBox()}</section>`;
    if (!v) return html + `<p class="info">No validation has been run for this model.</p>`;
    const c = v.compatibility, sf = v.safety, pg = v.progression;
    const chip = (ok, yes, no) => `<span class="cg-chip ${ok ? "ok" : "bad"}">${ok ? yes : no}</span>`;
    html += `<section><h3>Standard NAM compatibility</h3><p>${chip(c.standard_nam, "standard .nam", "not verified")} ${chip(c.full.rendered_ok, "Full renders", "Full failed")} ${chip(c.lite && c.lite.rendered_ok, "Lite renders", "Lite failed")} <span class="cg-warn">architecture ${esc(c.architecture)} · sample rate ${esc(c.sample_rate)} · no extra runtime processing required</span></p></section>
      <section><h3>Output safety</h3><p class="info">Set the player's <strong>Output gain to ${fmt(sf.recommended_output_gain_db, 1)} dB</strong> and leave it. ${sf.scaled_peak_over_0dbfs_at_input_gain_db.length ? `<strong>Peaks above 0 dBFS at Input gain ${sf.scaled_peak_over_0dbfs_at_input_gain_db.join(", ")} dB</strong> with this DI — lower the output or the Input gain there.` : "No peaks above 0 dBFS across the Input-gain range on this DI."}</p></section>
      <section><h3>Gain progression vs your captures (held-out DIs: ${esc(pg.held_out_dis.join(", "))})</h3>
        <table class="cg-table"><thead><tr><th>Position</th><th>Input gain</th><th>Role</th><th>Level Δ dB</th><th>HF Δ</th><th>Crest Δ</th><th>Dyn. range Δ</th><th>EQ max Δ</th><th>Level-matched ESR</th></tr></thead><tbody>
        ${pg.positions.map((r) => `<tr><td>G${r.position}</td><td>${fmt(r.input_gain_db)} dB</td><td><span class="cg-chip ${r.role === "training" ? "selected" : ""}">${r.role}</span></td><td>${fmt(r.level_db, 2)}</td><td>${fmt(r.hf_db, 2)}</td><td>${fmt(r.crest_db, 2)}</td><td>${fmt(r.dyn_range_db, 2)}</td><td>${fmt(r.eq_max_db, 2)}</td><td>${fmt(r.lm_esr, 4)}</td></tr>`).join("")}</tbody></table>
        <p class="cg-warn">Δ = model minus real capture at the plan's Input gain, one global output scale. “reference” rows were not training anchors, so they are independent evidence for the interpolated positions. ${pg.reversals.length ? `<strong>${pg.reversals.length} direction reversal(s)</strong> versus the real amp: ${pg.reversals.map((r) => `${r.measure} between G${r.between[0]}–G${r.between[1]}`).join("; ")}.` : "No direction reversals versus the real amp's progression (level, HF, crest)."}</p></section>`;
    const cov = v.coverage;
    if (cov) html += `<section><h3>Measured coverage of your uploaded captures</h3><p class="info">${cov.selected.length} captures trained; ${cov.all_within_tolerance ? "every omitted capture is within the working tolerances of what the selection predicts" : "some omitted captures fall outside the working tolerances (see stage 2)"}.</p></section>`;
    const au = v.audition;
    html += `<section><h3>Audition (optional)</h3>
        <p class="info">Sweep of the whole Input-gain range on one DI, ${au.sweep.seconds_per_step} s per step from ${au.sweep.input_gains_db[0]} to ${au.sweep.input_gains_db[au.sweep.input_gains_db.length - 1]} dB in 2 dB steps${au.sweep.attenuated_db ? ` (attenuated ${fmt(au.sweep.attenuated_db, 1)} dB as a whole for safe playback)` : ""}.</p>
        <audio controls preload="none" src="/api/cg/projects/${S.id}/audition/${au.sweep.file}"></audio>
        <div class="cg-row"><label>Compare at position <select id="cg-cmp" class="select-input">${au.comparisons.map((c, i) => `<option value="${i}">G${c.position} (${fmt(c.input_gain_db)} dB Input gain)</option>`).join("")}</select></label></div>
        <div class="cg-audio-pair" id="cg-cmp-pair"></div>
        <p class="cg-warn">The comparison plays the trained model and the real capture on the same DI at the same level. There is no captured sound at an interpolated point, so “reference” positions are the meaningful comparisons.</p></section>`;
    return html;
  }
  function bindStage4() {
    const d = S.data; if (!d) return;
    const v = document.getElementById("cg-validate");
    if (v) v.addEventListener("click", () => runJob(`/api/cg/projects/${S.id}/validate`, {}, async () => { say("Validation complete."); render(); }));
    const sel = document.getElementById("cg-cmp");
    if (sel && d.validation) {
      const show = () => { const c = d.validation.audition.comparisons[Number(sel.value)]; document.getElementById("cg-cmp-pair").innerHTML = `<div><strong>Trained model</strong><audio controls preload="none" src="/api/cg/projects/${S.id}/audition/${c.model}"></audio></div><div><strong>Original capture G${c.position}</strong><audio controls preload="none" src="/api/cg/projects/${S.id}/audition/${c.original}"></audio></div>`; };
      sel.addEventListener("change", show); show();
    }
  }

  // ---------- render
  function render() {
    document.querySelectorAll(".cg-step").forEach((b) => b.classList.toggle("active", Number(b.dataset.cgStage) === S.stage));
    if (!S.data) { body.innerHTML = `<p class="info">No project selected. Use <strong>New project</strong> to start.</p>`; return; }
    body.innerHTML = [stage1, stage2, stage3, stage4][S.stage - 1]();
    [bindStage1, bindStage2, bindStage3, bindStage4][S.stage - 1]();
  }
})();
