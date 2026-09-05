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

document.getElementById("btn-preview-a").addEventListener("click", () => notImplementedAction("/api/preview"));
document.getElementById("btn-preview-hybrid").addEventListener("click", () => notImplementedAction("/api/preview"));
document.getElementById("btn-preview-b").addEventListener("click", () => notImplementedAction("/api/preview"));
document.getElementById("btn-generate").addEventListener("click", () => notImplementedAction("/api/generate"));
