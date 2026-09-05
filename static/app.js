// Minimal UI wiring. No build step, no framework -- plain DOM + fetch.

const statusEl = document.getElementById("status");

function setStatus(msg, isError) {
  statusEl.textContent = msg;
  statusEl.style.color = isError ? "#c0362c" : "";
}

function fmtSigned(x) {
  const v = parseFloat(x);
  return (v >= 0 ? "+" : "") + v.toFixed(1);
}

// Resolved server-side paths for the uploaded .nam files, keyed by "a"/"b" --
// filled in once each upload completes, read by the Render Amps handler.
const ampServerPaths = { a: null, b: null };

async function uploadNam(slot, fileInputId, infoElId) {
  const fileInput = document.getElementById(fileInputId);
  const infoEl = document.getElementById(infoElId);
  const file = fileInput.files[0];
  ampServerPaths[slot] = null;
  if (!file) {
    infoEl.textContent = "";
    return;
  }
  infoEl.textContent = `Uploading ${file.name}...`;
  const formData = new FormData();
  formData.append("file", file);
  try {
    const resp = await fetch("/api/nam/upload", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      infoEl.textContent = "Error: " + data.error;
      setStatus("Failed to load " + file.name, true);
      return;
    }
    ampServerPaths[slot] = data.path;
    infoEl.textContent =
      `${file.name} -- ${data.architecture}, ${data.sample_rate} Hz, ${data.calibration_status}`;
    setStatus("Loaded " + file.name);
  } catch (err) {
    setStatus("Request failed: " + err, true);
  }
}

document.getElementById("amp-a-file").addEventListener("change", () =>
  uploadNam("a", "amp-a-file", "amp-a-info")
);
document.getElementById("amp-b-file").addEventListener("change", () =>
  uploadNam("b", "amp-b-file", "amp-b-info")
);

// ---- Input profile controls (instrument / profile / custom gain / calibration) ----
// Selecting a different profile changes the actual signal fed to both NAMs,
// so it's an EXPENSIVE-path setting like the amp files/DI -- it requires
// clicking Render Amps again, unlike crossover/transition/trim below.

const profilesData = JSON.parse(document.getElementById("input-profiles-data").textContent);
const instrumentSelect = document.getElementById("instrument-select");
const profileSelect = document.getElementById("input-profile-select");
const profileDescription = document.getElementById("input-profile-description");
const customGainRow = document.getElementById("custom-gain-row");
const customGainSlider = document.getElementById("custom-gain-slider");
const customGainValue = document.getElementById("custom-gain-value");
const calibrationModeSelect = document.getElementById("calibration-mode-select");
const referenceDbuInput = document.getElementById("reference-dbu-input");
const renderWarnings = document.getElementById("render-warnings");
const suggestedCrossoverNote = document.getElementById("suggested-crossover-note");

function currentProfile() {
  const profiles = profilesData[instrumentSelect.value] || [];
  return profiles.find((p) => p.id === profileSelect.value);
}

function updateProfileDescription() {
  const profile = currentProfile();
  if (!profile) {
    profileDescription.textContent = "";
    return;
  }
  profileDescription.textContent = profile.description;
  customGainRow.hidden = !profile.requires_custom_gain;
  customGainSlider.hidden = !profile.requires_custom_gain;
}

function populateProfileSelect() {
  const profiles = profilesData[instrumentSelect.value] || [];
  profileSelect.innerHTML = "";
  profiles.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.requires_custom_gain ? `${p.label} -- Custom` : `${p.label} (${fmtSigned(p.gain_db)} dB)`;
    profileSelect.appendChild(opt);
  });
  updateProfileDescription();
}

function markProfileStale() {
  if (havePair) {
    previewButtons.forEach((btn) => (btn.disabled = true));
    renderStatus.textContent = "Input profile changed -- click Render Amps to update.";
  }
}

instrumentSelect.addEventListener("change", () => {
  populateProfileSelect();
  markProfileStale();
  updateCoverage();
});
profileSelect.addEventListener("change", () => {
  updateProfileDescription();
  markProfileStale();
  updateCoverage();
});
customGainSlider.addEventListener("input", () => {
  customGainValue.textContent = `${fmtSigned(customGainSlider.value)} dB`;
  markProfileStale();
  updateCoverage();
});
calibrationModeSelect.addEventListener("change", markProfileStale);
referenceDbuInput.addEventListener("change", markProfileStale);

// DI filenames beginning with "bass_" are a trivial, documented instrument
// hint (see hybrid/input_profiles.py) -- used only as a default, never as a
// claim about what pickup actually produced the recording.
const diSelector = document.getElementById("di-selector");

function applyInstrumentHintFromDi() {
  const desired = diSelector.value.startsWith("bass_") ? "bass" : "guitar";
  if (instrumentSelect.value !== desired) {
    instrumentSelect.value = desired;
    populateProfileSelect();
  }
}

diSelector.addEventListener("change", () => {
  applyInstrumentHintFromDi();
  markProfileStale();
});

applyInstrumentHintFromDi();
populateProfileSelect();

const crossoverSlider = document.getElementById("crossover-slider");
const crossoverValue = document.getElementById("crossover-value");
const DEFAULT_CROSSOVER_DBFS = crossoverSlider.value;
crossoverSlider.addEventListener("input", () => {
  crossoverValue.textContent = `${parseFloat(crossoverSlider.value).toFixed(1)} dBFS`;
  scheduleUpdate();
});

const transitionSlider = document.getElementById("transition-slider");
const transitionValue = document.getElementById("transition-value");
transitionSlider.addEventListener("input", () => {
  transitionValue.textContent = `${transitionSlider.value} dB`;
  scheduleUpdate();
});

document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    transitionSlider.value = btn.dataset.value;
    transitionValue.textContent = `${btn.dataset.value} dB`;
    scheduleUpdate();
  });
});

document.getElementById("auto-level-match").addEventListener("change", scheduleUpdate);

const ampBTrimSlider = document.getElementById("amp-b-trim");
const ampBTrimValue = document.getElementById("amp-b-trim-value");
ampBTrimSlider.addEventListener("input", () => {
  ampBTrimValue.textContent = `${fmtSigned(ampBTrimSlider.value)} dB`;
  scheduleUpdate();
});

async function notImplementedAction(url) {
  try {
    const resp = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const data = await resp.json();
    setStatus(data.error || "Not implemented yet.", true);
  } catch (err) {
    setStatus("Request failed: " + err, true);
  }
}

const previewButtons = [
  document.getElementById("btn-preview-a"),
  document.getElementById("btn-preview-hybrid"),
  document.getElementById("btn-preview-b"),
];
const player = document.getElementById("player");
const trimReadout = document.getElementById("trim-readout");
const renderStatus = document.getElementById("render-status");
const journeyCanvas = document.getElementById("journey-canvas");
const journeyTooltip = document.getElementById("journey-tooltip");
const journeyEmpty = document.getElementById("journey-empty");
const coverageTable = document.getElementById("coverage-table");
const coverageTbody = document.getElementById("coverage-tbody");
const coverageEmpty = document.getElementById("coverage-empty");
const coverageWarning = document.getElementById("coverage-warning");

let havePair = false;
let updateTimer = null;
let lastJourneyData = null;
let lastSourcePlayed = null;

function hybridParamsBody() {
  return {
    crossover_dbfs: parseFloat(crossoverSlider.value),
    transition_width_db: parseFloat(transitionSlider.value),
    auto_level: document.getElementById("auto-level-match").checked,
    manual_b_trim_db: parseFloat(ampBTrimSlider.value) || 0.0,
  };
}

function scheduleUpdate() {
  if (!havePair) return;
  clearTimeout(updateTimer);
  updateTimer = setTimeout(() => {
    updateTrimReadout();
    updateJourney();
    updateCoverage();
  }, 150);
}

async function updateTrimReadout() {
  try {
    const resp = await fetch("/api/blend_info", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(hybridParamsBody()),
    });
    const data = await resp.json();
    if (!resp.ok) {
      trimReadout.textContent = "Trim error: " + data.error;
      return;
    }
    trimReadout.textContent =
      `Auto match ${fmtSigned(data.auto_trim_db)} dB  ·  ` +
      `manual tweak ${fmtSigned(data.manual_trim_db)} dB  ·  ` +
      `effective trim ${fmtSigned(data.effective_b_trim_db)} dB`;
  } catch (err) {
    trimReadout.textContent = "Trim update failed: " + err;
  }
}

async function updateJourney() {
  try {
    const resp = await fetch("/api/blend_curve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(hybridParamsBody()),
    });
    const data = await resp.json();
    if (!resp.ok) return;
    lastJourneyData = data;
    journeyEmpty.hidden = true;
    drawJourney();
  } catch (err) {
    // Visualization is a debug aid, not critical path -- fail quietly.
  }
}

async function updateCoverage() {
  if (!havePair) return;
  try {
    const resp = await fetch("/api/profile_coverage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        crossover_dbfs: parseFloat(crossoverSlider.value),
        transition_width_db: parseFloat(transitionSlider.value),
        instrument_type: instrumentSelect.value,
        custom_input_gain_db: parseFloat(customGainSlider.value) || 0.0,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) return;

    coverageTbody.innerHTML = "";
    data.coverage.forEach((row) => {
      const tr = document.createElement("tr");
      const cell = (text) => {
        const td = document.createElement("td");
        td.textContent = text;
        return td;
      };
      tr.appendChild(cell(row.label));
      tr.appendChild(cell(`${Math.round(row.amp_a_fraction * 100)}%`));
      tr.appendChild(cell(`${Math.round(row.transition_fraction * 100)}%`));
      tr.appendChild(cell(`${Math.round(row.amp_b_fraction * 100)}%`));
      coverageTbody.appendChild(tr);
    });
    coverageTable.hidden = false;
    coverageEmpty.hidden = true;

    coverageWarning.hidden = !data.reachability_warning;
    coverageWarning.textContent = data.reachability_warning || "";
  } catch (err) {
    // Diagnostic panel only -- fail quietly.
  }
}

// ---- Journey chart: envelope + crossover band/threshold + A<->B mix strip,
// with axis labels, a legend-matched color scheme, a hover readout, and a
// playhead synced to whatever is actually playing in the <audio> element.

const CHART = {
  marginLeft: 40,
  marginBottom: 20,
  marginTop: 6,
  mixHeight: 34,
  gap: 8,
  dbMin: -60,
  dbMax: 0,
  colorA: [51, 102, 204],
  colorB: [230, 126, 34],
};

function resizeCanvasForDPR() {
  const rect = journeyCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(rect.width * dpr));
  const h = Math.max(1, Math.round(rect.height * dpr));
  if (journeyCanvas.width !== w || journeyCanvas.height !== h) {
    journeyCanvas.width = w;
    journeyCanvas.height = h;
  }
}

function chartGeometry() {
  const dpr = window.devicePixelRatio || 1;
  const W = journeyCanvas.width;
  const H = journeyCanvas.height;
  const ml = CHART.marginLeft * dpr;
  const mb = CHART.marginBottom * dpr;
  const mt = CHART.marginTop * dpr;
  const mixH = CHART.mixHeight * dpr;
  const gap = CHART.gap * dpr;
  const envTop = mt;
  const envBottom = H - mb - mixH - gap;
  const mixTop = envBottom + gap;
  const mixBottom = H - mb;
  const plotLeft = ml;
  const plotRight = W;
  return { dpr, W, H, envTop, envBottom, mixTop, mixBottom, plotLeft, plotRight };
}

function drawJourney() {
  resizeCanvasForDPR();
  const data = lastJourneyData;
  const ctx = journeyCanvas.getContext("2d");
  const g = chartGeometry();
  ctx.clearRect(0, 0, g.W, g.H);
  if (!data || data.times.length < 2) return;

  const n = data.times.length;
  const duration = data.times[n - 1];
  const plotWidth = g.plotRight - g.plotLeft;
  const xAt = (i) => g.plotLeft + (i / (n - 1)) * plotWidth;
  const xAtTime = (t) => g.plotLeft + (duration > 0 ? t / duration : 0) * plotWidth;
  const envYAt = (db) => {
    const clamped = Math.max(CHART.dbMin, Math.min(CHART.dbMax, db));
    const frac = (clamped - CHART.dbMin) / (CHART.dbMax - CHART.dbMin);
    return g.envBottom - frac * (g.envBottom - g.envTop);
  };

  const dpr = g.dpr;

  // -- dB axis gridlines + labels --
  ctx.strokeStyle = "rgba(128,128,128,0.18)";
  ctx.fillStyle = "rgba(128,128,128,0.8)";
  ctx.font = `${11 * dpr}px sans-serif`;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let db = CHART.dbMax; db >= CHART.dbMin; db -= 20) {
    const y = envYAt(db);
    ctx.beginPath();
    ctx.moveTo(g.plotLeft, y);
    ctx.lineTo(g.plotRight, y);
    ctx.stroke();
    ctx.fillText(`${db}`, g.plotLeft - 6 * dpr, y);
  }

  // -- crossover transition band + threshold line --
  const lo = data.crossover_dbfs - data.transition_width_db / 2.0;
  const hi = data.crossover_dbfs + data.transition_width_db / 2.0;
  ctx.fillStyle = "rgba(122, 92, 255, 0.15)";
  ctx.fillRect(g.plotLeft, envYAt(hi), plotWidth, envYAt(lo) - envYAt(hi));
  ctx.strokeStyle = "rgba(122, 92, 255, 0.7)";
  ctx.setLineDash([5 * dpr, 4 * dpr]);
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath();
  ctx.moveTo(g.plotLeft, envYAt(data.crossover_dbfs));
  ctx.lineTo(g.plotRight, envYAt(data.crossover_dbfs));
  ctx.stroke();
  ctx.setLineDash([]);

  // -- envelope trace --
  ctx.strokeStyle = getComputedStyle(document.body).color;
  ctx.lineWidth = 1.5 * dpr;
  ctx.beginPath();
  data.envelope_db.forEach((db, i) => {
    const x = xAt(i), y = envYAt(db);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // -- mix strip: color interpolated between Amp A and Amp B by blend weight --
  for (let i = 0; i < n - 1; i++) {
    const t = data.blend_weight[i];
    const r = Math.round(CHART.colorA[0] + (CHART.colorB[0] - CHART.colorA[0]) * t);
    const gr = Math.round(CHART.colorA[1] + (CHART.colorB[1] - CHART.colorA[1]) * t);
    const b = Math.round(CHART.colorA[2] + (CHART.colorB[2] - CHART.colorA[2]) * t);
    ctx.fillStyle = `rgb(${r},${gr},${b})`;
    ctx.fillRect(xAt(i), g.mixTop, xAt(i + 1) - xAt(i) + 1, g.mixBottom - g.mixTop);
  }
  ctx.fillStyle = "#fff";
  ctx.font = `${11 * dpr}px sans-serif`;
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  ctx.fillText("A", g.plotLeft + 5 * dpr, (g.mixTop + g.mixBottom) / 2);
  ctx.textAlign = "right";
  ctx.fillText("B", g.plotRight - 5 * dpr, (g.mixTop + g.mixBottom) / 2);

  // -- time axis labels --
  ctx.fillStyle = "rgba(128,128,128,0.8)";
  ctx.font = `${11 * dpr}px sans-serif`;
  ctx.textBaseline = "top";
  const ticks = [0, 0.25, 0.5, 0.75, 1.0];
  ticks.forEach((f) => {
    const t = f * duration;
    const x = xAtTime(t);
    ctx.textAlign = f === 0 ? "left" : f === 1 ? "right" : "center";
    ctx.fillText(`${t.toFixed(1)}s`, x, g.mixBottom + 4 * dpr);
  });

  // -- playhead, synced to the <audio> element's actual playback position --
  if (!player.paused && !player.ended && player.duration && lastSourcePlayed) {
    const frac = player.currentTime / player.duration;
    const x = g.plotLeft + frac * plotWidth;
    ctx.strokeStyle = "#c0362c";
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    ctx.moveTo(x, g.envTop);
    ctx.lineTo(x, g.mixBottom);
    ctx.stroke();
  }
}

// Hover tooltip: nearest sample's time/envelope/mix.
journeyCanvas.addEventListener("mousemove", (evt) => {
  if (!lastJourneyData) return;
  const rect = journeyCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const g = chartGeometry();
  const xCss = evt.clientX - rect.left;
  const xCanvas = xCss * dpr;
  const plotWidth = g.plotRight - g.plotLeft;
  if (xCanvas < g.plotLeft || xCanvas > g.plotRight) {
    journeyTooltip.hidden = true;
    return;
  }
  const n = lastJourneyData.times.length;
  const frac = (xCanvas - g.plotLeft) / plotWidth;
  const idx = Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
  const t = lastJourneyData.times[idx];
  const db = lastJourneyData.envelope_db[idx];
  const mix = lastJourneyData.blend_weight[idx];
  journeyTooltip.hidden = false;
  journeyTooltip.style.left = `${xCss}px`;
  journeyTooltip.textContent =
    `t=${t.toFixed(2)}s  env=${db.toFixed(1)} dBFS  mix=${Math.round(mix * 100)}% B`;
});
journeyCanvas.addEventListener("mouseleave", () => {
  journeyTooltip.hidden = true;
});

player.addEventListener("timeupdate", () => { if (lastSourcePlayed) drawJourney(); });
player.addEventListener("play", () => drawJourney());
player.addEventListener("pause", () => drawJourney());
player.addEventListener("ended", () => drawJourney());
window.addEventListener("resize", () => drawJourney());

document.getElementById("btn-render-pair").addEventListener("click", async () => {
  const amp_a_path = ampServerPaths.a;
  const amp_b_path = ampServerPaths.b;
  const di_file = document.getElementById("di-selector").value;
  if (!amp_a_path || !amp_b_path || !di_file) {
    setStatus("Choose both Amp A and Amp B .nam files and pick a DI clip first.", true);
    return;
  }

  const profile = currentProfile();
  const instrument_type = instrumentSelect.value;
  const input_profile_id = profileSelect.value;
  const custom_input_gain_db = profile && profile.requires_custom_gain ? parseFloat(customGainSlider.value) : null;
  const calibration_mode = calibrationModeSelect.value;
  const reference_input_level_dbu = parseFloat(referenceDbuInput.value) || 12.0;

  renderStatus.textContent = "Rendering (running NAM inference twice)...";
  renderWarnings.hidden = true;
  previewButtons.forEach((btn) => (btn.disabled = true));
  try {
    const resp = await fetch("/api/render_pair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        amp_a_path, amp_b_path, di_file,
        instrument_type, input_profile_id, custom_input_gain_db,
        calibration_mode, reference_input_level_dbu,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      renderStatus.textContent = "Error: " + data.error;
      setStatus("Render failed.", true);
      return;
    }
    renderStatus.textContent =
      `Rendered ${data.duration_s.toFixed(1)}s @ ${data.sample_rate} Hz -- ` +
      `input peak ${data.input_peak_dbfs.toFixed(1)} dBFS.`;

    if (data.warnings && data.warnings.length) {
      renderWarnings.hidden = false;
      renderWarnings.textContent = data.warnings.join(" ");
    }

    suggestedCrossoverNote.innerHTML = "";
    if (data.suggested_crossover_dbfs !== null && data.suggested_crossover_dbfs !== undefined) {
      const suggested = data.suggested_crossover_dbfs;
      crossoverSlider.value = suggested.toFixed(1);
      crossoverValue.textContent = `${suggested.toFixed(1)} dBFS`;
      suggestedCrossoverNote.textContent =
        `Crossover set to ${suggested.toFixed(1)} dBFS, suggested from this DI's active-signal level. `;
      const resetBtn = document.createElement("button");
      resetBtn.type = "button";
      resetBtn.className = "link-btn";
      resetBtn.textContent = `Reset to ${parseFloat(DEFAULT_CROSSOVER_DBFS).toFixed(1)}`;
      resetBtn.addEventListener("click", () => {
        crossoverSlider.value = DEFAULT_CROSSOVER_DBFS;
        crossoverValue.textContent = `${parseFloat(DEFAULT_CROSSOVER_DBFS).toFixed(1)} dBFS`;
        scheduleUpdate();
      });
      suggestedCrossoverNote.appendChild(resetBtn);
    }

    previewButtons.forEach((btn) => (btn.disabled = false));
    havePair = true;
    updateTrimReadout();
    updateJourney();
    updateCoverage();
    setStatus("Amp pair rendered and cached -- sliders now only recompute the blend.");
  } catch (err) {
    renderStatus.textContent = "Request failed: " + err;
    setStatus("Render failed.", true);
  }
});

async function preview(source) {
  const body = source === "hybrid" ? { source, ...hybridParamsBody() } : { source };
  try {
    const resp = await fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const data = await resp.json();
      setStatus(data.error || "Preview failed.", true);
      return;
    }
    if (source === "hybrid") {
      const auto = resp.headers.get("X-Auto-Trim-Db");
      const manual = resp.headers.get("X-Manual-Trim-Db");
      const effective = resp.headers.get("X-Effective-Trim-Db");
      trimReadout.textContent =
        `Auto match ${fmtSigned(auto)} dB  ·  manual tweak ${fmtSigned(manual)} dB  ·  effective trim ${fmtSigned(effective)} dB`;
    }
    const blob = await resp.blob();
    player.src = URL.createObjectURL(blob);
    lastSourcePlayed = source;
    player.play();
    setStatus(`Playing ${source.toUpperCase()}.`);
  } catch (err) {
    setStatus("Request failed: " + err, true);
  }
}

document.getElementById("btn-preview-a").addEventListener("click", () => preview("a"));
document.getElementById("btn-preview-hybrid").addEventListener("click", () => preview("hybrid"));
document.getElementById("btn-preview-b").addEventListener("click", () => preview("b"));
const trainingInputFile = document.getElementById("training-input-file");
const trainingInputStatus = document.getElementById("training-input-status");
const generateStatus = document.getElementById("generate-status");
const generateResult = document.getElementById("generate-result");
let trainingInputReady = false;

async function refreshTrainingInputStatus() {
  try {
    const resp = await fetch("/api/training_input/status");
    const data = await resp.json();
    trainingInputReady = !!data.ready;
    trainingInputStatus.textContent = data.ready
      ? `Ready: ${data.path} (${data.frame_count} frames @ ${data.sample_rate} Hz)`
      : `Missing: ${data.error || "no official training input uploaded yet"}`;
  } catch (err) {
    trainingInputStatus.textContent = "Could not check training input status: " + err;
  }
}
refreshTrainingInputStatus();

trainingInputFile.addEventListener("change", async () => {
  const file = trainingInputFile.files[0];
  if (!file) return;
  trainingInputStatus.textContent = `Uploading ${file.name}...`;
  const formData = new FormData();
  formData.append("file", file);
  try {
    const resp = await fetch("/api/training_input/upload", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      trainingInputStatus.textContent = "Error: " + data.error;
      trainingInputReady = false;
      return;
    }
    trainingInputReady = true;
    trainingInputStatus.textContent = `Ready: ${data.path} (${data.frame_count} frames @ ${data.sample_rate} Hz)`;
  } catch (err) {
    trainingInputStatus.textContent = "Upload failed: " + err;
  }
});

document.getElementById("btn-generate").addEventListener("click", async () => {
  if (!havePair) {
    setStatus("Render and audition an amp pair first.", true);
    return;
  }
  if (!trainingInputReady) {
    setStatus("Upload the official NAM training input first.", true);
    return;
  }
  generateStatus.textContent = "Generating training bundle (running NAM inference on the official input)...";
  generateResult.hidden = true;
  try {
    const resp = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(hybridParamsBody()),
    });
    const data = await resp.json();
    if (!resp.ok) {
      generateStatus.textContent = "Error: " + (data.error || "generation failed");
      setStatus("Training bundle generation failed.", true);
      return;
    }
    generateStatus.textContent = `Bundle generated: ${data.design_id}`;
    generateResult.hidden = false;
    generateResult.innerHTML = `
      <div><strong>Bundle:</strong> <code>${data.bundle_dir}</code></div>
      <div><strong>Target:</strong> <code>${data.target_path}</code> (peak ${data.safety_report.final_peak_dbfs.toFixed(1)} dBFS, safety reduction ${data.safety_report.gain_reduction_db.toFixed(2)} dB)</div>
      <div><strong>Calibration:</strong> ${data.calibration_summary.effective_mode} (requested ${data.calibration_summary.requested_mode})</div>
      <div><strong>Train it with:</strong> <code>${data.training_command}</code></div>
      ${data.warnings && data.warnings.length ? `<div class="warning-box">${data.warnings.join(" ")}</div>` : ""}
    `;
    setStatus("Training bundle ready.");

    lastDesignId = data.design_id;
    lastTrainingCommand = data.training_command;
    document.getElementById("a2-training-section").hidden = false;
    document.getElementById("local-training-command").textContent = lastTrainingCommand;
    refreshKaggleStatus();
  } catch (err) {
    generateStatus.textContent = "Request failed: " + err;
    setStatus("Training bundle generation failed.", true);
  }
});

// --- Kaggle GPU training backend -----------------------------------------
let lastDesignId = null;
let lastTrainingCommand = "";
let kaggleAuthenticated = false;
let kaggleJobPollTimer = null;

const kaggleStatusEl = document.getElementById("kaggle-status");
const kaggleConnectBtn = document.getElementById("btn-kaggle-connect");
const trainA2Btn = document.getElementById("btn-train-a2");
const kaggleProgressEl = document.getElementById("kaggle-progress");
const kaggleResultEl = document.getElementById("kaggle-result");
const kaggleBackendRadio = document.getElementById("a2-backend-kaggle");
const localBackendRadio = document.getElementById("a2-backend-local");
const kagglePanel = document.getElementById("kaggle-panel");
const localPanel = document.getElementById("local-panel");

function updateBackendPanels() {
  const useKaggle = kaggleBackendRadio.checked;
  kagglePanel.hidden = !useKaggle;
  localPanel.hidden = useKaggle;
}
kaggleBackendRadio.addEventListener("change", updateBackendPanels);
localBackendRadio.addEventListener("change", updateBackendPanels);

async function refreshKaggleStatus() {
  try {
    const resp = await fetch("/api/kaggle/status" + (lastDesignId ? `?design_id=${encodeURIComponent(lastDesignId)}` : ""));
    const data = await resp.json();
    if (!data.cli_installed) {
      kaggleStatusEl.textContent = "Kaggle CLI not installed. Run: pip install kaggle";
      kaggleConnectBtn.hidden = true;
      kaggleAuthenticated = false;
      trainA2Btn.disabled = true;
      return;
    }
    if (!data.authenticated) {
      kaggleStatusEl.textContent = `Kaggle CLI ${data.cli_version || ""} installed, not connected.`;
      kaggleConnectBtn.hidden = false;
      kaggleAuthenticated = false;
      trainA2Btn.disabled = true;
      return;
    }
    kaggleAuthenticated = true;
    kaggleConnectBtn.hidden = true;
    trainA2Btn.disabled = false;
    const quota = data.quota_available ? (data.quota_raw || "available") : "unavailable";
    kaggleStatusEl.textContent = `Connected ✓  CLI ${data.cli_version || "?"}  GPU: NVIDIA T4  Quota: ${quota}`;

    if (data.job && data.job.state && !["complete", "failed"].includes(data.job.state)) {
      pollKaggleJob(data.job.design_id, data.job.job_id);
    }
  } catch (err) {
    kaggleStatusEl.textContent = "Could not check Kaggle status: " + err;
  }
}

kaggleConnectBtn.addEventListener("click", async () => {
  kaggleStatusEl.textContent = "Starting Kaggle authentication...";
  try {
    const resp = await fetch("/api/kaggle/auth/start", { method: "POST" });
    const data = await resp.json();
    kaggleStatusEl.textContent = data.started
      ? "A Kaggle login flow was started. Complete it in your browser, then click Connect Kaggle again to refresh."
      : `Run this yourself, then refresh: ${data.command}`;
  } catch (err) {
    kaggleStatusEl.textContent = "Could not start Kaggle auth: " + err;
  }
});

trainA2Btn.addEventListener("click", async () => {
  if (!lastDesignId) {
    setStatus("Generate a training bundle first.", true);
    return;
  }
  trainA2Btn.disabled = true;
  kaggleProgressEl.hidden = false;
  kaggleProgressEl.textContent = "Uploading training pair...";
  kaggleResultEl.hidden = true;
  try {
    const resp = await fetch("/api/kaggle/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ design_id: lastDesignId }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      kaggleProgressEl.textContent = "Error: " + (data.error || "training request failed");
      trainA2Btn.disabled = false;
      return;
    }
    pollKaggleJob(lastDesignId, data.job_id);
  } catch (err) {
    kaggleProgressEl.textContent = "Request failed: " + err;
    trainA2Btn.disabled = false;
  }
});

function pollKaggleJob(designId, jobId) {
  if (kaggleJobPollTimer) clearInterval(kaggleJobPollTimer);
  kaggleProgressEl.hidden = false;

  const poll = async () => {
    try {
      const resp = await fetch(`/api/kaggle/jobs/${encodeURIComponent(jobId)}?design_id=${encodeURIComponent(designId)}`);
      const data = await resp.json();
      if (!resp.ok) {
        kaggleProgressEl.textContent = "Error checking job: " + (data.error || "unknown error");
        return;
      }
      const progressText = data.progress ? `epoch ${data.progress.epoch}/${data.progress.total_epochs}` : "";
      kaggleProgressEl.textContent = `${data.state}${progressText ? " — " + progressText : ""}`;

      if (data.state === "complete") {
        clearInterval(kaggleJobPollTimer);
        trainA2Btn.disabled = false;
        kaggleResultEl.hidden = false;
        kaggleResultEl.innerHTML = `
          <div><strong>Model:</strong> <code>${data.output_nam_path || "(unknown)"}</code></div>
          <div><strong>SHA-256:</strong> <code>${data.output_nam_sha256 || ""}</code></div>
        `;
        setStatus("Kaggle A2 training complete.");
      } else if (data.state === "failed") {
        clearInterval(kaggleJobPollTimer);
        trainA2Btn.disabled = false;
        kaggleProgressEl.textContent = "Failed: " + (data.error || "unknown error");
        setStatus("Kaggle A2 training failed.", true);
      }
    } catch (err) {
      kaggleProgressEl.textContent = "Polling error: " + err;
    }
  };
  poll();
  kaggleJobPollTimer = setInterval(poll, 10000);
}

updateBackendPanels();
