// Minimal UI wiring. No build step, no framework -- plain DOM + fetch.

const statusEl = document.getElementById("status");

function setStatus(msg, isError) {
  statusEl.textContent = msg;
  statusEl.style.color = isError ? "#b00" : "#666";
}

async function loadNam(pathInputId, infoElId) {
  const path = document.getElementById(pathInputId).value.trim();
  const infoEl = document.getElementById(infoElId);
  if (!path) {
    setStatus("Enter a .nam file path first.", true);
    return;
  }
  try {
    const resp = await fetch("/api/nam/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      infoEl.textContent = "Error: " + data.error;
      setStatus("Failed to load " + path, true);
      return;
    }
    infoEl.textContent =
      `architecture=${data.architecture} sample_rate=${data.sample_rate} ` +
      `[${data.calibration_status}]`;
    setStatus("Loaded " + path);
  } catch (err) {
    setStatus("Request failed: " + err, true);
  }
}

document.getElementById("btn-load-a").addEventListener("click", () =>
  loadNam("amp-a-path", "amp-a-info")
);
document.getElementById("btn-load-b").addEventListener("click", () =>
  loadNam("amp-b-path", "amp-b-info")
);

const crossoverSlider = document.getElementById("crossover-slider");
const crossoverValue = document.getElementById("crossover-value");
crossoverSlider.addEventListener("input", () => {
  crossoverValue.textContent = `${crossoverSlider.value} dBFS`;
});

const transitionSlider = document.getElementById("transition-slider");
const transitionValue = document.getElementById("transition-value");
transitionSlider.addEventListener("input", () => {
  transitionValue.textContent = `${transitionSlider.value} dB`;
});

document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    transitionSlider.value = btn.dataset.value;
    transitionValue.textContent = `${btn.dataset.value} dB`;
  });
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

document.getElementById("btn-render-pair").addEventListener("click", async () => {
  const amp_a_path = document.getElementById("amp-a-path").value.trim();
  const amp_b_path = document.getElementById("amp-b-path").value.trim();
  const di_file = document.getElementById("di-selector").value;
  if (!amp_a_path || !amp_b_path || !di_file) {
    setStatus("Enter both amp paths and pick a DI clip first.", true);
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
    setStatus("Amp pair rendered and cached -- sliders now only recompute the blend.");
  } catch (err) {
    renderStatus.textContent = "Request failed: " + err;
    setStatus("Render failed.", true);
  }
});

async function preview(source) {
  const body = { source };
  if (source === "hybrid") {
    body.crossover_dbfs = parseFloat(crossoverSlider.value);
    body.transition_width_db = parseFloat(transitionSlider.value);
    body.auto_level = document.getElementById("auto-level-match").checked;
    body.manual_b_trim_db = parseFloat(document.getElementById("amp-b-trim").value) || 0.0;
  }
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
    } else {
      trimReadout.textContent = "";
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
