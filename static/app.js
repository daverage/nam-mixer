// Minimal UI wiring. No build step, no framework -- plain DOM + fetch.

const statusEl = document.getElementById("status");

function setStatus(msg, isError) {
  statusEl.textContent = msg;
  statusEl.style.color = isError ? "#b00" : "#666";
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
      `${file.name} -- architecture=${data.architecture} sample_rate=${data.sample_rate} ` +
      `[${data.calibration_status}]`;
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

const crossoverSlider = document.getElementById("crossover-slider");
const crossoverValue = document.getElementById("crossover-value");
crossoverSlider.addEventListener("input", () => {
  crossoverValue.textContent = `${crossoverSlider.value} dBFS`;
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
document.getElementById("amp-b-trim").addEventListener("input", scheduleUpdate);

const dryGainSlider = document.getElementById("dry-gain-slider");
const dryGainValue = document.getElementById("dry-gain-value");
dryGainSlider.addEventListener("input", () => {
  dryGainValue.textContent = `${dryGainSlider.value} dB`;
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

let havePair = false;
let updateTimer = null;

function hybridParamsBody() {
  return {
    crossover_dbfs: parseFloat(crossoverSlider.value),
    transition_width_db: parseFloat(transitionSlider.value),
    auto_level: document.getElementById("auto-level-match").checked,
    manual_b_trim_db: parseFloat(document.getElementById("amp-b-trim").value) || 0.0,
    dry_gain_db: parseFloat(dryGainSlider.value) || 0.0,
  };
}

function scheduleUpdate() {
  if (!havePair) return;
  clearTimeout(updateTimer);
  updateTimer = setTimeout(() => {
    updateTrimReadout();
    updateJourney();
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
      `Auto match ${data.auto_trim_db.toFixed(1)} dB, ` +
      `manual tweak ${data.manual_trim_db.toFixed(1)} dB, ` +
      `effective trim ${data.effective_b_trim_db.toFixed(1)} dB`;
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
    drawJourney(data);
  } catch (err) {
    // Visualization is a debug aid, not critical path -- fail quietly.
  }
}

function drawJourney(data) {
  const ctx = journeyCanvas.getContext("2d");
  const W = journeyCanvas.width;
  const H = journeyCanvas.height;
  ctx.clearRect(0, 0, W, H);

  const n = data.times.length;
  if (n < 2) return;

  const envTop = 0;
  const envHeight = H * 0.6;
  const mixTop = envHeight + 10;
  const mixHeight = H - mixTop;

  const dbMin = -60, dbMax = 0;
  const xAt = (i) => (i / (n - 1)) * W;
  const envYAt = (db) => envTop + envHeight * (1 - (Math.max(dbMin, Math.min(dbMax, db)) - dbMin) / (dbMax - dbMin));

  // Shade the crossover transition band on the envelope panel.
  const lo = data.crossover_dbfs - data.transition_width_db / 2.0;
  const hi = data.crossover_dbfs + data.transition_width_db / 2.0;
  ctx.fillStyle = "rgba(150, 100, 200, 0.15)";
  ctx.fillRect(0, envYAt(hi), W, envYAt(lo) - envYAt(hi));
  ctx.strokeStyle = "rgba(150, 100, 200, 0.6)";
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(0, envYAt(data.crossover_dbfs));
  ctx.lineTo(W, envYAt(data.crossover_dbfs));
  ctx.stroke();
  ctx.setLineDash([]);

  // Envelope trace.
  ctx.strokeStyle = "#444";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  data.envelope_db.forEach((db, i) => {
    const x = xAt(i), y = envYAt(db);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Mix panel: filled area, color interpolated between Amp A (blue) and
  // Amp B (orange) by the blend weight at each point.
  const colorA = [51, 102, 204];
  const colorB = [230, 126, 34];
  for (let i = 0; i < n - 1; i++) {
    const t = data.blend_weight[i];
    const r = Math.round(colorA[0] + (colorB[0] - colorA[0]) * t);
    const g = Math.round(colorA[1] + (colorB[1] - colorA[1]) * t);
    const b = Math.round(colorA[2] + (colorB[2] - colorA[2]) * t);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(xAt(i), mixTop, xAt(i + 1) - xAt(i) + 1, mixHeight);
  }

  // Mix panel labels.
  ctx.fillStyle = "#fff";
  ctx.font = "11px sans-serif";
  ctx.fillText("A", 4, mixTop + mixHeight / 2 + 4);
  ctx.textAlign = "right";
  ctx.fillText("B", W - 4, mixTop + mixHeight / 2 + 4);
  ctx.textAlign = "left";
}

document.getElementById("btn-render-pair").addEventListener("click", async () => {
  const amp_a_path = ampServerPaths.a;
  const amp_b_path = ampServerPaths.b;
  const di_file = document.getElementById("di-selector").value;
  if (!amp_a_path || !amp_b_path || !di_file) {
    setStatus("Choose both Amp A and Amp B .nam files and pick a DI clip first.", true);
    return;
  }
  renderStatus.textContent = "Rendering (running NAM inference twice)...";
  previewButtons.forEach((btn) => (btn.disabled = true));
  try {
    const resp = await fetch("/api/render_pair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ amp_a_path, amp_b_path, di_file }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      renderStatus.textContent = "Error: " + data.error;
      setStatus("Render failed.", true);
      return;
    }
    renderStatus.textContent = `Rendered ${data.duration_s.toFixed(1)}s @ ${data.sample_rate} Hz.`;
    previewButtons.forEach((btn) => (btn.disabled = false));
    havePair = true;
    updateTrimReadout();
    updateJourney();
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
      trimReadout.textContent = `Auto match ${auto} dB, manual tweak ${manual} dB, effective trim ${effective} dB`;
    }
    const blob = await resp.blob();
    player.src = URL.createObjectURL(blob);
    player.play();
    setStatus(`Playing ${source.toUpperCase()}.`);
  } catch (err) {
    setStatus("Request failed: " + err, true);
  }
}

document.getElementById("btn-preview-a").addEventListener("click", () => preview("a"));
document.getElementById("btn-preview-hybrid").addEventListener("click", () => preview("hybrid"));
document.getElementById("btn-preview-b").addEventListener("click", () => preview("b"));
document.getElementById("btn-generate").addEventListener("click", () => notImplementedAction("/api/generate"));
