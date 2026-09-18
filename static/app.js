// Minimal UI wiring. No build step, no framework -- plain DOM + fetch.

// The desktop shell (desktop/src-tauri) loads this exact page in a native
// window. WKWebView/WebView2 do not honor the HTML `download` attribute --
// a click on `<a download>` just silently does nothing there. When running
// inside Tauri, intercept every such click and hand it to a native command
// that shows a real save dialog instead. Ordinary browser use is untouched.
const isTauriDesktop = typeof window.__TAURI__ !== "undefined";
// Labels for buttons/links that actually go through the native save-dialog
// path above (NOT the couple of purely server-side "download this into the
// app's own folder" actions, e.g. fetching nam_render itself) -- "Download"
// implies a browser's silent auto-save-to-Downloads, which is not what
// happens here; calling it "Save" matches the dialog the user actually
// sees, so a save that succeeded doesn't read as the button doing nothing.
function desktopSaveLabel(text) {
  return isTauriDesktop ? text.replace(/^Download\b/, "Save") : text;
}
// The ONE place every desktop save goes through -- the delegated click
// listener below (for raw innerHTML `<a download>` anchors) and
// triggerFileDownload/saveBlobAsFile (for anything built through those
// helpers) all call this, so every save in the app notifies the user the
// same way at each stage: starting, saved-to-path, cancelled, or failed.
// Uses the existing global status line (setStatus), not a blocking
// window.alert, so it behaves like every other in-app status message.
async function desktopSave(invokeName, args, filename) {
  setStatus(`Saving ${filename}…`);
  try {
    const path = await window.__TAURI__.core.invoke(invokeName, args);
    setStatus(`Saved ${filename} to ${path}`);
    return true;
  } catch (err) {
    if (String(err) === "save cancelled") {
      setStatus("Save cancelled.");
    } else {
      console.error("Desktop save failed:", err);
      setStatus(`Could not save ${filename}: ${err}`, true);
    }
    return false;
  }
}

// Plain window.confirm() -- WKWebView (Tauri's engine on macOS) supports
// it natively via its own UI delegate, so there's no need for (and real
// risk in) routing this through a Rust dialog command instead: an earlier
// attempt at that used tauri-plugin-dialog's blocking_show() from an async
// command, which hung indefinitely with no visible dialog and no error,
// breaking every delete button on desktop. Kept as its own named function
// (rather than inlining window.confirm at each call site) so every
// confirmation in the app is easy to find and change together later.
async function desktopConfirm(message, title = "Confirm") {
  return window.confirm(message);
}

if (isTauriDesktop) {
  // target="_blank" links (the footer's GitHub/license links) don't open
  // anything in WKWebView/WebView2 -- there's no real "new window" handler
  // -- so route them to the OS's default browser via open_external_url
  // instead of letting the click fall through and silently do nothing.
  document.addEventListener("click", async (event) => {
    const externalLink = event.target.closest('a[target="_blank"]');
    if (externalLink) {
      event.preventDefault();
      const url = externalLink.getAttribute("href");
      try {
        await window.__TAURI__.core.invoke("open_external_url", { url });
      } catch (err) {
        console.error("Could not open external link:", err);
      }
      return;
    }
  });
  document.addEventListener("click", async (event) => {
    const link = event.target.closest("a[download]");
    if (!link) return;
    event.preventDefault();
    const href = link.getAttribute("href");
    const filename = link.getAttribute("download") || "download";
    if (href.startsWith("blob:")) {
      const resp = await fetch(href);
      const buf = await resp.arrayBuffer();
      const dataBase64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
      await desktopSave("save_bytes", { filename, dataBase64 }, filename);
    } else {
      const absoluteUrl = new URL(href, window.location.origin).toString();
      await desktopSave("save_file_from_url", { url: absoluteUrl, filename }, filename);
    }
  });
}

const statusEl = document.getElementById("status");
const systemUsageEl = document.getElementById("system-usage");
const activityIndicator = document.getElementById("activity-indicator");
const activityMessage = document.getElementById("activity-message");
let activitySequence = 0;
const activities = new Map();

// One shared, unmistakable activity indicator for operations that can take
// long enough to look like a frozen page. Section-local messages remain useful
// detail, while this stays visible at the bottom of the window regardless of
// which panel is open.
function beginActivity(message) {
  const id = ++activitySequence;
  let currentMessage = message;
  const started = Date.now();
  const render = () => {
    activityMessage.textContent = `${currentMessage} · ${formatElapsed((Date.now() - started) / 1000)}`;
  };
  activities.set(id, render);
  render();
  activityIndicator.hidden = false;
  const timer = setInterval(render, 1000);
  const finish = (finishedMessage = "") => {
    clearInterval(timer);
    activities.delete(id);
    if (activities.size) {
      [...activities.values()].at(-1)();
    } else {
      activityIndicator.hidden = true;
    }
    if (finishedMessage) setStatus(finishedMessage);
  };
  finish.update = (nextMessage) => {
    if (!activities.has(id)) return;
    currentMessage = nextMessage;
    render();
  };
  return finish;
}
async function refreshSystemUsage() {
  if (!systemUsageEl) return;
  try {
    const response = await fetch("/api/system/usage");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "unavailable");
    const parts = [`CPU ${Number(data.cpu_percent).toFixed(0)}%`, `RAM ${data.memory_used_gb}/${data.memory_total_gb} GB`];
    if (data.gpu && data.gpu.utilization_percent !== null && data.gpu.utilization_percent !== undefined) {
      // Apple Silicon reports real, live usage too (via ioreg's
      // IOAccelerator PerformanceStatistics -- see app.py's
      // _apple_gpu_usage), just against unified memory rather than a
      // separate VRAM pool, so label it accordingly instead of "VRAM".
      const memoryLabel = data.gpu.accelerator === "mps" ? "Memory" : "VRAM";
      parts.push(`GPU ${Number(data.gpu.utilization_percent).toFixed(0)}% · ${memoryLabel} ${(data.gpu.memory_used_mb / 1024).toFixed(1)}/${(data.gpu.memory_total_mb / 1024).toFixed(1)} GB`);
    } else if (data.gpu && data.gpu.name) {
      // Fallback only: ioreg's output shape didn't match, so just confirm
      // the Metal/MPS device local training will actually use.
      parts.push(`GPU ${data.gpu.name} (Metal)`);
    }
    systemUsageEl.textContent = parts.join("  ·  ");
  } catch {
    systemUsageEl.textContent = "System usage unavailable";
  }
}
refreshSystemUsage();
setInterval(refreshSystemUsage, 3000);

// ---- First-launch welcome guide -----------------------------------------
// Shown once (tracked via localStorage, so it survives a refresh but not a
// private window/cleared site data -- acceptable here since it's purely a
// convenience prompt, never state anything else depends on) then available
// again any time from Settings > Getting started.
const WELCOME_SEEN_KEY = "nam-mixer-welcome-seen";
const welcomeOverlay = document.getElementById("welcome-overlay");
function showWelcome() { welcomeOverlay.hidden = false; }
function hideWelcome() {
  welcomeOverlay.hidden = true;
  try { localStorage.setItem(WELCOME_SEEN_KEY, "1"); } catch { /* private window etc. -- just re-show next time */ }
}
document.getElementById("btn-welcome-dismiss").addEventListener("click", hideWelcome);
document.getElementById("btn-show-welcome").addEventListener("click", showWelcome);
let welcomeAlreadySeen = false;
try { welcomeAlreadySeen = localStorage.getItem(WELCOME_SEEN_KEY) === "1"; } catch { /* default to showing it */ }
if (!welcomeAlreadySeen) showWelcome();

// ---- Design mode tabs (Dynamic Hybrid / Parallel Blend / Character Blend) ----
// Tabs are DESIGN MODES, not separate applications -- Amp A/B, the preview
// DI, input profile/calibration, render, test gain, Listen controls, the
// Cabinet IR stage, official training input, A2 quality, and training all
// stay SHARED between tabs (see docs/blend-mode.md). Only the crossover/
// transition/level-match controls, the journey/coverage diagnostics, and
// the Create A2 wording differ per mode. Switching tabs never re-renders.
let currentMode = "hybrid";
const modeTabs = document.querySelectorAll(".mode-tab[data-mode]");
const modePanels = document.querySelectorAll("[data-mode-panel]");
const btnPreviewMix = document.getElementById("btn-preview-mix");
const autoLevelMatchLabel = document.getElementById("auto-level-match-label");
const createA2Title = document.getElementById("create-a2-title");
const createA2Description = document.getElementById("create-a2-description");
const workflowTabs = document.querySelectorAll(".workflow-tab");
const workflowHint = document.getElementById("workflow-hint");
const modeDescription = document.getElementById("mode-description");
const staleCallout = document.getElementById("render-stale-callout");
const staleBannerMessage = document.getElementById("render-stale-message");
const rebuildPreviewButton = document.getElementById("btn-rebuild-preview");
let workflowStage = "configure";
let statusClearTimer = null;

// Each stage already explains itself via its own section headers, so the
// nav no longer repeats a per-stage description here -- that text only
// existed to be overwritten by the "prepare your amps first" warning below,
// and its varying length could force the tab buttons to wrap. The hint
// element is kept only for that warning; it stays empty (and collapsed via
// CSS) the rest of the time.
function setWorkflowStage(stage) {
  workflowStage = stage;
  document.body.dataset.workflowStage = stage;
  workflowTabs.forEach((tab) => {
    const active = tab.dataset.workflowStage === stage;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-current", active ? "step" : "false");
  });
  workflowHint.textContent = "";
}

workflowTabs.forEach((tab) => tab.addEventListener("click", () => {
  const stage = tab.dataset.workflowStage;
  if ((stage === "compare" || stage === "shape" || stage === "finish" || stage === "create") && !havePair) {
    workflowHint.textContent = "Prepare your amps first (step 1) before moving on.";
    return;
  }
  setWorkflowStage(stage);
}));
setWorkflowStage(workflowStage);

document.getElementById("btn-continue-shape").addEventListener("click", () => setWorkflowStage("shape"));

const HYBRID_LEVEL_MATCH_LABEL = "Keep Amp B as loud as Amp A at the changeover";
const BLEND_LEVEL_MATCH_LABEL = "Keep Amp B as loud as Amp A while you play";
const HYBRID_A2_DESCRIPTION =
  "Uses the current changeover and level settings to make a training target from the official NAM input. " +
  "Your pickup choice shapes preview only; it is not baked into the training input.";
const BLEND_A2_DESCRIPTION =
  "Uses the current fixed mix and level settings to make a training target from the official NAM input. " +
  "Your pickup choice shapes preview only; it is not baked into the training input.";
const CHARACTER_A2_DESCRIPTION =
  "Uses the current tone, feel, and drive choices to make a training target from the official NAM input. " +
  "Character Blend usually benefits from High definition training.";

function applyModeVisibility() {
  modePanels.forEach((el) => {
    el.hidden = el.dataset.modePanel !== currentMode;
  });
  btnPreviewMix.textContent = currentMode === "blend" ? "Blend" : currentMode === "character" ? "Character" : "Hybrid";
  autoLevelMatchLabel.textContent = currentMode === "blend" ? BLEND_LEVEL_MATCH_LABEL : HYBRID_LEVEL_MATCH_LABEL;
  createA2Title.textContent = currentMode === "blend" ? "Make a Blend A2" : currentMode === "character" ? "Make a Character A2" : "Make a Hybrid A2";
  createA2Description.textContent = currentMode === "blend" ? BLEND_A2_DESCRIPTION : currentMode === "character" ? CHARACTER_A2_DESCRIPTION : HYBRID_A2_DESCRIPTION;
  const auditionMode = document.getElementById("audition-mode");
  auditionMode.textContent = currentMode === "blend" ? "Always mixed" : currentMode === "character" ? "Combine tone and feel" : "Changes as you play harder";
  modeDescription.textContent = currentMode === "blend"
    ? "Both amps are present all the time at one fixed ratio. Use this for a permanent mixed rig."
    : currentMode === "character"
      ? "Choose broad tone, playing feel, and drive character from either amp. This creates one combined character."
      : "Amp A handles quieter playing and Amp B takes over as the input becomes louder. Use this for a clean-to-driven response.";
}

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    // Utility tabs temporarily hide the builder layout. Returning through a
    // design tab restores it before applying the selected mode.
    setToolsOpen(false);
    setSessionsOpen(false);
    currentMode = tab.dataset.mode;
    invalidateLiveAudition("Design mode changed — live blend stopped.");
    modeTabs.forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-pressed", t === tab ? "true" : "false");
    });
    applyModeVisibility();
    if (workflowStage === "create" || workflowStage === "finish") setWorkflowStage("shape");
    // Switching modes never re-renders NAM inference -- just recompute the
    // (already-rendered) mix/journey/coverage panels for the new mode.
    scheduleUpdate();
  });
});
applyModeVisibility();

function setStatus(msg, isError) {
  clearTimeout(statusClearTimer);
  statusEl.textContent = msg;
  statusEl.classList.toggle("is-error", Boolean(isError));
  if (msg && !isError) {
    statusClearTimer = setTimeout(() => {
      if (statusEl.textContent === msg) statusEl.textContent = "";
    }, 6000);
  }
}

function isCalibrationOnlyWarning(warnings) {
  return warnings.length > 0 && warnings.every((warning) => warning.startsWith("Input calibration unavailable"));
}

function renderWarningText(warnings) {
  if (isCalibrationOnlyWarning(warnings)) {
    return "Calibration data is unavailable in one or both captures. The preview uses each capture's recorded digital level.";
  }
  return warnings.join(" ");
}

function fmtSigned(x) {
  const v = parseFloat(x);
  return (v >= 0 ? "+" : "") + v.toFixed(1);
}

function formatElapsed(seconds) {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}m ${String(whole % 60).padStart(2, "0")}s`;
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
  const stopActivity = beginActivity(`Uploading ${file.name}…`);
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
  } finally {
    stopActivity();
  }
}

document.getElementById("amp-a-file").addEventListener("change", () =>
  uploadNam("a", "amp-a-file", "amp-a-info")
);
document.getElementById("amp-b-file").addEventListener("change", () =>
  uploadNam("b", "amp-b-file", "amp-b-info")
);

// ---- Shared Cabinet IR stage (both design modes) ----
// See docs/blend-mode.md "SHARED CABINET IR STAGE"/"CAB UI". The cab sits
// AFTER the amp combination and is applied identically to Amp A/Result/Amp B
// previews (fair comparisons) -- see hybrid/cab_ir.py and app.py's
// _parse_cab_params/_resolve_cab_design.
let cabServerPath = null;
const cabFileInput = document.getElementById("cab-file");
const cabInfoEl = document.getElementById("cab-info");
const cabPreviewEnabled = document.getElementById("cab-preview-enabled");
const cabExportMode = document.getElementById("cab-export-mode");
const cabStatusEl = document.getElementById("cab-status");

function updateCabStatus() {
  if (!cabServerPath) {
    cabStatusEl.textContent = "Cab: off";
  } else if (cabExportMode.value === "embedded") {
    cabStatusEl.textContent = "Cab: Sequential Embedded (Experimental) -- exact IR in a separate Linear/FIR stage";
  } else if (cabExportMode.value === "learned") {
    cabStatusEl.textContent = "Cab: Baked In -- trained into the A2 model";
  } else if (cabPreviewEnabled.checked) {
    cabStatusEl.textContent = "Cab: preview only -- exported A2 remains amp/head only";
  } else {
    cabStatusEl.textContent = "Cab: off";
  }
}

cabFileInput.addEventListener("change", async () => {
  resetGeneratedModel("The cabinet changed. Create new training files before starting another training run.");
  const file = cabFileInput.files[0];
  cabServerPath = null;
  cabPreviewEnabled.checked = false;
  cabExportMode.value = "none";
  cabPreviewEnabled.disabled = true;
  cabExportMode.disabled = true;
  if (!file) {
    cabInfoEl.textContent = "";
    updateCabStatus();
    return;
  }
  const stopActivity = beginActivity(`Uploading cabinet ${file.name}…`);
  const formData = new FormData();
  formData.append("file", file);
  try {
    const resp = await fetch("/api/cab/upload", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      cabInfoEl.textContent = "Error: " + data.error;
      updateCabStatus();
      return;
    }
    cabServerPath = data.path;
    cabPreviewEnabled.disabled = false;
    cabExportMode.disabled = false;
    const durationS = data.duration_s !== undefined ? data.duration_s.toFixed(2) : "?";
    const preparedMs = data.prepared_duration_ms !== undefined ? data.prepared_duration_ms.toFixed(1) : null;
    const energy999Ms = data.energy_999_ms !== undefined ? data.energy_999_ms.toFixed(1) : null;
    let info =
      `${file.name} -- ${data.original_sample_rate} Hz, ${data.original_channels}ch, ${durationS}s` +
      (data.leading_samples_trimmed ? ` (${data.leading_samples_trimmed} leading samples trimmed)` : "");
    if (preparedMs !== null) info += ` -- prepared length ${preparedMs} ms`;
    if (energy999Ms !== null) info += `, 99.9% energy by ${energy999Ms} ms`;
    cabInfoEl.textContent = info;
  } catch (err) {
    cabInfoEl.textContent = "Upload failed: " + err;
  } finally {
    stopActivity();
  }
  updateCabStatus();
});

cabPreviewEnabled.addEventListener("change", () => {
  updateCabStatus();
  // Does NOT require re-rendering the amps (cab runs after amp combination),
  // but it DOES change what the final exported model would contain, so any
  // already-generated training files are now stale for this setting -- same
  // invalidation boundary as the cab-file-upload handler above, see
  // resetGeneratedModel's docstring.
  resetGeneratedModel("The cabinet setting changed. Create new training files before starting another training run.");
  invalidateLiveAudition("Cabinet setting changed — start live blend again to load the matching stems.");
  if (lastPreviewSource) scheduleAuditionRefresh(lastPreviewSource);
});
cabExportMode.addEventListener("change", () => {
  if (cabExportMode.value === "embedded" && !sequentialEmbeddedWarningAcknowledged) {
    const proceed = confirm(
      "Experimental compatibility\n\n" +
      "This export uses NAM's Sequential architecture to place the trained amp model before " +
      "an embedded Linear/FIR cabinet stage. Although this is a valid NAM model structure, " +
      "some NAM players only accept A2 architectures and may reject this file. Use Baked In " +
      "for broader compatibility."
    );
    if (!proceed) {
      cabExportMode.value = "learned";
      updateCabStatus();
      return;
    }
    sequentialEmbeddedWarningAcknowledged = true;
  }
  // "If Bake cab into A2 is enabled, automatically ensure Use cab in preview
  // is also enabled" -- docs/blend-mode.md "CAB UI".
  updateCabStatus();
  resetGeneratedModel("The cabinet setting changed. Create new training files before starting another training run.");
  invalidateLiveAudition("Cabinet setting changed — start live blend again to load the matching stems.");
  if (lastPreviewSource) scheduleAuditionRefresh(lastPreviewSource);
});
updateCabStatus();

// ---- Input profile controls (instrument / profile / custom gain / calibration) ----
// Selecting a different profile changes the actual signal fed to both NAMs,
// so it's an EXPENSIVE-path setting like the amp files/DI -- it requires
// clicking Render Amps again, unlike crossover/transition/trim below.

const profilesData = JSON.parse(document.getElementById("input-profiles-data").textContent);
const instrumentSelect = document.getElementById("instrument-select");
let instrumentExplicitlySelected = false;
const profileSelect = document.getElementById("input-profile-select");
const profileDescription = document.getElementById("input-profile-description");
const customGainRow = document.getElementById("custom-gain-row");
const customGainSlider = document.getElementById("custom-gain-slider");
const customGainValue = document.getElementById("custom-gain-value");
const calibrationModeSelect = document.getElementById("calibration-mode-select");
const referenceDbuInput = document.getElementById("reference-dbu-input");
const testGainSlider = document.getElementById("test-gain-slider");
const testGainValue = document.getElementById("test-gain-value");
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

// Anything that changes what the two amps actually receive (amp files, DI
// clip, instrument/profile/custom-gain, calibration mode, reference level)
// invalidates the cached RenderedPair -- see hybrid/pipeline.py's
// render_pair() docstring for the authoritative list. This makes that
// staleness impossible to miss: preview buttons disable, the Render Amps
// button gets a pulsing highlight, and the status line names WHAT changed
// (not a generic "input profile changed" for every case).
function markProfileStale(reason, { preserveAudition = false } = {}) {
  const wasCurrentPreview = havePair;
  if (!preserveAudition) pendingAuditionResume = null;
  resetGeneratedModel("The source or input settings changed. Create new training files when you are happy with the new sound.");
  // This is the single invalidation boundary for every setting that changes
  // NAM inference.  A warning alone must never leave old audio usable.
  renderGeneration += 1;
  activeRenderId = null;
  havePair = false;
  clearAudition();
  invalidateLiveAudition("Amp pair changed — start live blend again after rendering.");
  previewButtons.forEach((btn) => (btn.disabled = true));
  liveBlendButton.disabled = true;
  document.getElementById("btn-character-low-level-check").disabled = true;
  wizardAnalyseButton.disabled = true;
  clearTimeout(updateTimer);
  clearTimeout(auditionRefreshTimer);
  syncTrainingControls();
  renderPairBtn.classList.add("btn-render-stale");
  renderStatus.textContent = `${reason || "A setting that affects amp rendering changed"} -- click Render Amps to update.`;
  if (wasCurrentPreview) {
    staleBannerMessage.textContent = `${reason || "A source setting"} changed. The current preview can no longer be used.`;
    staleCallout.hidden = false;
    renderPairBtn.textContent = "Rebuild amp preview";
  }
}

document.getElementById("amp-a-file").addEventListener("change", () => markProfileStale("Amp A changed"));
document.getElementById("amp-b-file").addEventListener("change", () => markProfileStale("Amp B changed"));

instrumentSelect.addEventListener("change", () => {
  instrumentExplicitlySelected = true;
  populateProfileSelect();
  markProfileStale("Instrument changed");
  updateCoverage();
});
profileSelect.addEventListener("change", () => {
  instrumentExplicitlySelected = true;
  updateProfileDescription();
  markProfileStale("Input profile changed");
  updateCoverage();
});
customGainSlider.addEventListener("input", () => {
  customGainValue.textContent = `${fmtSigned(customGainSlider.value)} dB`;
  markProfileStale("Custom input gain changed");
  updateCoverage();
});
calibrationModeSelect.addEventListener("change", () => markProfileStale("Calibration mode changed"));
referenceDbuInput.addEventListener("change", () => markProfileStale("Reference level changed"));

// Independent per-amp pre-render input trim -- see hybrid/pipeline.py's
// RenderedPair docstring. A render-stage control like the profile/
// calibration settings above (it changes what each amp actually receives),
// not a blend-stage one, so it invalidates the cached RenderedPair too.
const ampAInputGainSlider = document.getElementById("amp-a-input-gain-slider");
const ampAInputGainValue = document.getElementById("amp-a-input-gain-value");
const ampBInputGainSlider = document.getElementById("amp-b-input-gain-slider");
const ampBInputGainValue = document.getElementById("amp-b-input-gain-value");
ampAInputGainSlider.addEventListener("input", () => {
  ampAInputGainValue.textContent = `${fmtSigned(ampAInputGainSlider.value)} dB`;
  markProfileStale("Amp A input trim changed");
});
ampBInputGainSlider.addEventListener("input", () => {
  ampBInputGainValue.textContent = `${fmtSigned(ampBInputGainSlider.value)} dB`;
  markProfileStale("Amp B input trim changed");
});

// Real audio gain (unlike the deprecated preview-only dry_gain_db) -- see
// hybrid/pipeline.py's render_pair() docstring. Does NOT affect the
// coverage table (that's computed from the un-gained source envelope so it
// can compare hypothetical profiles independently of this stress-test knob).
//
// Unlike the other expensive/render-triggering controls above, this one
// auto-renders on its own (debounced) instead of requiring a manual "Render
// Amps" click -- it lives in the Listen card specifically so pushing the
// input level and hearing the result feels like one continuous action, even
// though each step is still a real (briefly non-instant) NAM re-render.
const testGainStatus = document.getElementById("test-gain-status");
const TEST_GAIN_DEBOUNCE_MS = 600;
let testGainRenderTimer = null;
let pendingAuditionResume = null;

testGainSlider.addEventListener("input", () => {
  testGainValue.textContent = `${fmtSigned(testGainSlider.value)} dB`;
  const shouldRender = havePair || testGainRenderTimer !== null;
  if (lastPreviewSource) pendingAuditionResume = {
    source: lastPreviewSource, position: player.currentTime || 0, playing: !player.paused,
  };
  if (testGainRenderTimer) clearTimeout(testGainRenderTimer);
  markProfileStale("Test gain changed", { preserveAudition: true });
  if (!shouldRender) return;
  const requestGeneration = renderGeneration;
  testGainStatus.textContent = "Will re-render shortly...";
  testGainRenderTimer = setTimeout(async () => {
    testGainRenderTimer = null;
    if (requestGeneration !== renderGeneration) return;
    previewButtons.forEach((btn) => (btn.disabled = true));
    setRenderBusy(true);
    player.classList.add("player-busy");
    const stopActivity = beginActivity("Re-rendering both amps for the new input level…");
    try {
      const data = await doRenderPair();
      if (requestGeneration !== renderGeneration) return;
      applyRenderResult(data, { applySuggestedCrossover: false });
      testGainStatus.textContent = `Updated -- input peak ${data.input_peak_dbfs.toFixed(1)} dBFS.`;
      if (pendingAuditionResume) {
        const resume = pendingAuditionResume;
        pendingAuditionResume = null;
        await preview(resume.source, { resumeState: resume, quiet: true });
      }
    } catch (err) {
      testGainStatus.textContent = "Error: " + err.message;
      setStatus("Test-gain re-render failed.", true);
    } finally {
      stopActivity();
      setRenderBusy(false);
      player.classList.remove("player-busy");
    }
  }, TEST_GAIN_DEBOUNCE_MS);
});

// DI filenames beginning with "bass_" are a trivial, documented instrument
// hint (see hybrid/input_profiles.py) -- used only as a default, never as a
// claim about what pickup actually produced the recording.
const diSelector = document.getElementById("di-selector");

function applyInstrumentHintFromDi() {
  if (instrumentExplicitlySelected) return;
  const desired = diSelector.value.startsWith("bass_") ? "bass" : "guitar";
  if (instrumentSelect.value !== desired) {
    instrumentSelect.value = desired;
    populateProfileSelect();
  }
}

diSelector.addEventListener("change", () => {
  applyInstrumentHintFromDi();
  markProfileStale("DI clip changed");
  invalidateModelComparison("The musical test performance changed. Build the comparison again.");
});

applyInstrumentHintFromDi();
populateProfileSelect();

const crossoverSlider = document.getElementById("crossover-slider");
const crossoverValue = document.getElementById("crossover-value");
const DEFAULT_CROSSOVER_DBFS = crossoverSlider.value;
// What "Reset" after a suggested-crossover auto-set should actually go back
  // to -- the app's built-in default UNLESS a saved/imported session
// most recently supplied a different crossover as your real baseline, in
// which case that's what "Reset" should mean (see applySessionSettings).
let crossoverBaseline = { value: DEFAULT_CROSSOVER_DBFS, label: "default" };

// --- "Guitar volume feel" 0-10 knob -----------------------------------
// A cosmetic, per-DI-calibrated alternative to the raw dBFS crossover
// slider (see static/app.js's crossoverSlider above, which remains the
// single source of truth -- this knob only ever reads/writes that value).
// crossoverKnobCalibration.{minDb,maxDb} anchor "0"/"10" to this render's
// OWN measured instrument-level-into-NAM envelope (blend_envelope_percentiles
// from /api/render_pair, i.e. pair.envelope_db -- the profiled envelope
// that actually drives build_hybrid()'s crossfade, per the user's request
// to calibrate against real "instrument level into NAM" signal rather than
// an assumed universal volume-pot law -- see docs/README known caveats on
// why a literal per-guitar pot-taper simulation is deliberately not done).
// Mapping the knob LINEARLY onto that dBFS range is already the "audio
// taper" behaviour: dB is itself a log scale of amplitude, so a linear
// sweep in dB reads as the perceptually-log sweep a real volume pot aims
// for -- no extra exponential curve is needed on top.
const crossoverKnobSlider = document.getElementById("crossover-knob-slider");
const crossoverKnobValue = document.getElementById("crossover-knob-value");
const crossoverKnobNote = document.getElementById("crossover-knob-note");
let crossoverKnobCalibration = { minDb: -40.0, maxDb: -6.0, calibrated: false };

function dbFromKnob(knobValue) {
  const { minDb, maxDb } = crossoverKnobCalibration;
  return minDb + (parseFloat(knobValue) / 10.0) * (maxDb - minDb);
}

function knobFromDb(db) {
  const { minDb, maxDb } = crossoverKnobCalibration;
  if (maxDb <= minDb) return 0;
  return Math.min(10, Math.max(0, ((parseFloat(db) - minDb) / (maxDb - minDb)) * 10.0));
}

// Reflects the raw dBFS crossover value (however it changed -- manual drag,
// a preset load, the suggested-crossover auto-set, or its reset link) back
// onto the knob's position, without re-triggering the raw slider's own
// input handling.
function syncCrossoverKnobFromDb() {
  const knobPos = knobFromDb(crossoverSlider.value);
  crossoverKnobSlider.value = knobPos.toFixed(1);
  crossoverKnobValue.textContent = `${knobPos.toFixed(1)} / 10`;
  updateTransitionAroundSwitchNote();
}

const transitionAroundSwitchNote = document.getElementById("transition-around-switch-note");
function updateTransitionAroundSwitchNote() {
  const width = parseFloat(transitionSlider.value);
  transitionAroundSwitchNote.textContent =
    `The change happens across a ${width.toFixed(1)} dB range. Below it you hear Amp A; above it you hear Amp B.`;
}

function updateCrossoverKnobCalibration(blendEnvelopePercentiles) {
  const p10 = blendEnvelopePercentiles && blendEnvelopePercentiles.p10;
  const p90 = blendEnvelopePercentiles && blendEnvelopePercentiles.p90;
  if (p10 == null || p90 == null || p90 <= p10) {
    crossoverKnobNote.textContent =
      "0 is the quiet end of this performance; 10 is the loud end. Prepare the amps to calibrate this control.";
    return;
  }
  crossoverKnobCalibration = { minDb: p10, maxDb: p90, calibrated: true };
  crossoverKnobNote.textContent =
    `Calibrated from this performance: 0 is about ${p10.toFixed(1)} dBFS (quiet), ` +
    `10 is about ${p90.toFixed(1)} dBFS (loud). This is a useful playing guide, not a simulation of a particular guitar's volume pot.`;
  syncCrossoverKnobFromDb();
}

crossoverKnobSlider.addEventListener("input", () => {
  const db = dbFromKnob(crossoverKnobSlider.value);
  crossoverSlider.value = db.toFixed(2);
  crossoverValue.textContent = `${db.toFixed(1)} dBFS`;
  crossoverKnobValue.textContent = `${parseFloat(crossoverKnobSlider.value).toFixed(1)} / 10`;
  scheduleUpdate();
  scheduleAuditionRefresh();
});

crossoverSlider.addEventListener("input", () => {
  crossoverValue.textContent = `${parseFloat(crossoverSlider.value).toFixed(1)} dBFS`;
  syncCrossoverKnobFromDb();
  scheduleUpdate();
  scheduleAuditionRefresh();
});

const transitionSlider = document.getElementById("transition-slider");
const transitionValue = document.getElementById("transition-value");
transitionSlider.addEventListener("input", () => {
  transitionValue.textContent = `${transitionSlider.value} dB`;
  updateTransitionAroundSwitchNote();
  scheduleUpdate();
  scheduleAuditionRefresh();
});

syncCrossoverKnobFromDb();

const presetButtons = document.querySelectorAll(".preset-btn");
function syncPresetButtonStates() {
  presetButtons.forEach((btn) => {
    const active = parseFloat(btn.dataset.value) === parseFloat(transitionSlider.value);
    btn.setAttribute("aria-pressed", active ? "true" : "false");
  });
}
presetButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    transitionSlider.value = btn.dataset.value;
    transitionValue.textContent = `${btn.dataset.value} dB`;
    syncPresetButtonStates();
    updateTransitionAroundSwitchNote();
    scheduleUpdate();
    scheduleAuditionRefresh();
  });
});
transitionSlider.addEventListener("input", syncPresetButtonStates);
syncPresetButtonStates();

const mixSlider = document.getElementById("mix-slider");
const mixValue = document.getElementById("mix-value");
function updateMixValueLabel() {
  const b = parseInt(mixSlider.value, 10);
  mixValue.textContent = `Amp A ${100 - b}% / Amp B ${b}%`;
}
mixSlider.addEventListener("input", () => {
  updateMixValueLabel();
  if (liveAudition.active) liveAudition.setMix(parseInt(mixSlider.value, 10) / 100.0);
  else scheduleAuditionRefresh();
  scheduleUpdate();
});
updateMixValueLabel();

// ---- Tone Wizard ---------------------------------------------------------
// This is intentionally a thin, reversible guide over the existing controls.
// It never invents a guitar model or bypasses the render/level-match path.
const wizardBody = document.getElementById("wizard-body");
const wizardInstrument = document.getElementById("wizard-instrument");
const wizardProfile = document.getElementById("wizard-profile");
const wizardProfileDescription = document.getElementById("wizard-profile-description");
const wizardSwitch = document.getElementById("wizard-switch");
const wizardSwitchValue = document.getElementById("wizard-switch-value");
const wizardMoreB = document.getElementById("wizard-more-b");
const wizardMoreBValue = document.getElementById("wizard-more-b-value");
const wizardDynamicQuestion = document.getElementById("wizard-dynamic-question");
const wizardFullQuestion = document.getElementById("wizard-full-question");
const wizardDynamicRoleNote = document.getElementById("wizard-dynamic-role-note");
const wizardCharacterQuestion = document.getElementById("wizard-character-question");
const wizardToneSource = document.getElementById("wizard-tone-source");
const wizardDriveSource = document.getElementById("wizard-drive-source");
const wizardResult = document.getElementById("wizard-result");
const wizardAnalyseButton = document.getElementById("btn-wizard-analyse");
const recipePromptInput = document.getElementById("recipe-prompt-input");
const recipePromptApplyButton = document.getElementById("btn-recipe-prompt-apply");
const recipeUseLocalAi = document.getElementById("recipe-use-local-ai");
const recipeAiProviderLabel = document.getElementById("recipe-ai-provider-label");
const recipeAiStatus = document.getElementById("recipe-ai-status");
const recipeUseWebResearch = document.getElementById("recipe-use-web-research");
const recipeUseTone3000 = document.getElementById("recipe-use-tone3000");
const recipeTone3000RigScope = document.getElementById("recipe-tone3000-rig-scope");
const recipeTone3000Author = document.getElementById("recipe-tone3000-author");
const recipePromptContainer = recipePromptInput.closest(".recipe-prompt");
// A live Flask process may serve a cached template while static assets have
// refreshed. Create these optional conversation controls defensively so that
// a mixed-version page still boots and the recipe button remains usable.
const recipeResult = document.getElementById("recipe-result") || (() => {
  const element = document.createElement("div");
  element.id = "recipe-result";
  element.className = "recipe-result";
  element.hidden = true;
  element.setAttribute("aria-live", "polite");
  recipePromptContainer.append(element);
  return element;
})();
const recipeConversationResetButton = document.getElementById("btn-recipe-conversation-reset") || (() => {
  const element = document.createElement("button");
  element.type = "button";
  element.id = "btn-recipe-conversation-reset";
  element.className = "btn btn-secondary btn-small recipe-conversation-reset";
  element.textContent = "Start new conversation";
  element.hidden = true;
  recipePromptContainer.append(element);
  return element;
})();
const recipeSaveMarkdownButton = document.getElementById("btn-recipe-save-markdown") || (() => {
  const element = document.createElement("button");
  element.type = "button";
  element.id = "btn-recipe-save-markdown";
  element.className = "btn btn-secondary btn-small";
  element.textContent = "Save conversation as Markdown";
  element.hidden = true;
  recipePromptContainer.append(element);
  return element;
})();
let localRecipeAiAvailable = false;
let recipeConversationHistory = [];
let recipeSourcePlan = null;
let selectedTone3000Capture = null;
const aiTone3000Context = document.getElementById("ai-tone3000-context");

async function loadLocalRecipeAiStatus() {
  try {
    const response = await fetch("/api/local_llm/status");
    const data = await response.json();
    const configured = Boolean(response.ok && data.enabled);
    const providerLabel = data.provider === "cloudflare"
      ? "Use Cloudflare Workers AI"
      : data.provider === "custom" ? "Use custom AI" : "Use local AI";
    if (recipeAiProviderLabel) recipeAiProviderLabel.textContent = providerLabel;
    localRecipeAiAvailable = configured && data.reachable !== false;
    recipeUseLocalAi.disabled = !localRecipeAiAvailable;
    recipeUseLocalAi.checked = localRecipeAiAvailable;
    if (!configured) {
      recipeAiStatus.textContent = "(not configured — choose a provider and save it in Settings > AI Assistant)";
    } else if (data.reachable === false) {
      recipeAiStatus.textContent = `(configured for ${data.base_url}, but nothing responded — start your local LLM host, or check Settings)`;
    } else {
      recipeAiStatus.textContent = `(ready: ${data.model})`;
    }
  } catch (_error) {
    localRecipeAiAvailable = false;
    if (recipeAiProviderLabel) recipeAiProviderLabel.textContent = "Use AI";
    recipeAiStatus.textContent = "(could not check status — see Settings > AI Assistant to set it up)";
  }
}
loadLocalRecipeAiStatus();

function selectedWizardBehaviour() {
  return document.querySelector('input[name="wizard-behaviour"]:checked').value;
}
function populateWizardProfiles({ preserveCurrent = true } = {}) {
  const profiles = profilesData[wizardInstrument.value] || [];
  const desired = preserveCurrent && instrumentSelect.value === wizardInstrument.value
    ? profileSelect.value : profiles[0]?.id;
  wizardProfile.innerHTML = "";
  profiles.forEach((profile) => {
    const option = document.createElement("option");
    option.value = profile.id;
    option.textContent = profile.label;
    wizardProfile.appendChild(option);
  });
  wizardProfile.value = desired || "";
  updateWizardProfileDescription();
}
function updateWizardProfileDescription() {
  const profile = (profilesData[wizardInstrument.value] || []).find((item) => item.id === wizardProfile.value);
  wizardProfileDescription.textContent = profile
    ? `${profile.description}${profile.requires_custom_gain ? " Choose its relative level in Advanced controls after applying." : ""}`
    : "";
}
function updateWizardLabels() {
  wizardSwitchValue.textContent = `${parseFloat(wizardSwitch.value).toFixed(1)} / 10`;
  wizardMoreBValue.textContent = `${wizardMoreB.value}% Amp B`;
  const isFixed = selectedWizardBehaviour() === "fixed";
  const isCharacter = selectedWizardBehaviour() === "character";
  wizardDynamicQuestion.hidden = isFixed || isCharacter;
  wizardFullQuestion.hidden = !isFixed;
  wizardDynamicRoleNote.hidden = isFixed || isCharacter;
  wizardCharacterQuestion.hidden = !isCharacter;
}
function setModeFromWizard(mode) {
  currentMode = mode;
  invalidateLiveAudition("Design mode changed — live blend stopped.");
  modeTabs.forEach((tab) => {
    const active = tab.dataset.mode === mode;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-pressed", active ? "true" : "false");
  });
  applyModeVisibility();
}

wizardSwitch.addEventListener("input", updateWizardLabels);
wizardMoreB.addEventListener("input", updateWizardLabels);
wizardInstrument.addEventListener("change", () => populateWizardProfiles({ preserveCurrent: false }));
wizardProfile.addEventListener("change", updateWizardProfileDescription);
document.querySelectorAll('input[name="wizard-behaviour"]').forEach((input) => input.addEventListener("change", updateWizardLabels));
populateWizardProfiles();
updateWizardLabels();

function applyWizardSettings() {
  instrumentExplicitlySelected = true;
  const profileId = wizardProfile.value;
  if (instrumentSelect.value !== wizardInstrument.value) {
    instrumentSelect.value = wizardInstrument.value;
    populateProfileSelect();
  }
  profileSelect.value = profileId;
  updateProfileDescription();
  markProfileStale("Tone Wizard instrument or pickup profile changed");

  const behaviour = selectedWizardBehaviour();
  if (behaviour === "fixed") {
    setModeFromWizard("blend");
    mixSlider.value = wizardMoreB.value;
    updateMixValueLabel();
  } else if (behaviour === "character") {
    setModeFromWizard("character");
    const toneMix = wizardToneSource.value === "b" ? 100 : 0;
    const driveMix = wizardDriveSource.value === "b" ? 100 : 0;
    [["tone", toneMix], ["feel", driveMix], ["drive", driveMix]].forEach(([name, value]) => {
      document.getElementById(`${name}-slider`).value = value;
      document.getElementById(`${name}-value`).textContent = `${value}% B`;
    });
    document.getElementById("drive-morph-enabled").checked = false;
  } else {
    setModeFromWizard("hybrid");
    crossoverKnobSlider.value = wizardSwitch.value;
    const crossoverDb = dbFromKnob(wizardSwitch.value);
    crossoverSlider.value = crossoverDb.toFixed(2);
    crossoverBaseline = { value: crossoverSlider.value, label: "wizard" };
    crossoverValue.textContent = `${crossoverDb.toFixed(1)} dBFS`;
    syncCrossoverKnobFromDb();
    transitionSlider.value = behaviour === "smooth" ? "12" : "6";
    transitionValue.textContent = `${transitionSlider.value} dB`;
    syncPresetButtonStates();
    updateTransitionAroundSwitchNote();
  }
  scheduleUpdate();
  scheduleAuditionRefresh();
  wizardResult.hidden = false;
  const recipe = behaviour === "character"
    ? `Character recipe applied: tone from Amp ${wizardToneSource.value.toUpperCase()}, feel and drive from Amp ${wizardDriveSource.value.toUpperCase()}.`
    : "Starting point applied.";
  wizardResult.textContent = havePair
    ? `${recipe} Re-render the amps now so the selected instrument profile drives both NAMs.`
    : `${recipe} Upload both NAMs and render the amps to calibrate the guitar-volume switch point.`;
  setWorkflowStage("configure");
}
document.getElementById("btn-wizard-apply").addEventListener("click", applyWizardSettings);

// A deliberately local, explainable first pass at natural-language recipes.
// It identifies the common "keep one amp's tone/feel, but let gain progress
// from it to the other" request. The resulting controls stay visible and
// editable, rather than hiding a black-box interpretation behind an AI call.
function recipeTextIncludes(text, phrases) {
  return phrases.some((phrase) => text.includes(phrase));
}

// Looks for a percentage figure mentioned near one of the given keywords,
// e.g. "70% vox" or "eq at 70 percent" -> 70. Returns null if none found.
function percentNear(text, keywords) {
  for (const keyword of keywords) {
    const before = text.match(new RegExp(`(\\d{1,3})\\s*(?:percent|%)[a-z0-9\\s]{0,15}\\b${keyword}`));
    if (before) return Math.min(100, Number(before[1]));
    const after = text.match(new RegExp(`\\b${keyword}[a-z0-9\\s]{0,40}?(\\d{1,3})\\s*(?:percent|%)`));
    if (after) return Math.min(100, Number(after[1]));
  }
  return null;
}

function recipeFromPrompt(prompt) {
  const text = prompt.toLowerCase().replace(/%/g, " percent ").replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ").trim();
  const namedAmpFamilies = ["fender", "vox", "marshall", "mesa", "boogie", "orange", "peavey", "soldano", "engl", "hiwatt", "bogner", "princeton", "deluxe", "tweed", "jcm", "ac30"];
  const namedAmpCount = namedAmpFamilies.filter((name) => text.includes(name)).length;
  const hasAmpTransition = namedAmpCount >= 2 && recipeTextIncludes(text, ["from", "to", "into", "toward", "towards", "transition", "goes", "change"]);
  const wantsGainJourney = recipeTextIncludes(text, ["gain", "drive", "crunch", "overdrive", "dirty", "play harder", "dig in", "starts", "ends"]);
  const wantsCharacter = recipeTextIncludes(text, ["eq", "tone", "feel", "response", "touch"]);
  const wantsParallel = recipeTextIncludes(text, ["parallel", "constant", "always mixed", "always blend", "both all the time", "permanent mix"]);

  if (wantsParallel) {
    const explicitAmpBPercent = text.match(/\b(\d{1,3})\s*(?:percent\s*)?(?:amp\s*)?b\b/);
    const mixB = explicitAmpBPercent
      ? Math.min(100, Number(explicitAmpBPercent[1]))
      : recipeTextIncludes(text, ["mostly amp b", "more amp b"])
        ? 70
        : recipeTextIncludes(text, ["mostly amp a", "more amp a"])
          ? 30
          : 50;
    return {
      mode: "blend", mixB,
      explanation: `Parallel Blend: both amps stay present at ${100 - mixB}% Amp A / ${mixB}% Amp B, independent of playing level.`,
    };
  }

  if (wantsCharacter && (hasAmpTransition || wantsGainJourney)) {
    // A stated tone/feel split (e.g. "70% vox") names the FIRST amp mentioned,
    // which this heuristic treats as Amp A -- so it converts to a %-toward-B figure.
    const tonePercentA = percentNear(text, ["eq", "feel", "tone", "response", "touch"]);
    const toneB = tonePercentA != null ? Math.max(0, 100 - tonePercentA) : 0;
    const fiftyFifty = /\b50\s*\/?\s*50\b/.test(text);
    const drivePercentB = percentNear(text, ["gain", "drive", "volume", "level"]);
    const driveHigh = fiftyFifty ? 50 : (drivePercentB != null ? drivePercentB : 100);
    const driveMid = Math.round(driveHigh / 2);
    return {
      mode: "character", tone: toneB, feel: toneB, drive: 0,
      driveLow: 0, driveMid, driveHigh,
      explanation: `Character Blend: tone and feel stay ${100 - toneB}% Amp A / ${toneB}% Amp B; drive morphs from Amp A at low playing level to ${driveMid}% Amp B at medium level and ${driveHigh}% Amp B at high level.`,
    };
  }
  if (wantsGainJourney) {
    return {
      mode: "hybrid", switchKnob: 7, width: 6,
      explanation: "Dynamic Hybrid: Amp A stays dominant for quieter playing and Amp B takes over as you play harder.",
    };
  }
  return null;
}

function setCharacterSlider(name, value) {
  document.getElementById(`${name}-slider`).value = value;
  document.getElementById(`${name}-value`).textContent = `${value}% B`;
}

function escapeMarkdownHtml(value) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function renderMarkdownInline(value) {
  let html = escapeMarkdownHtml(value);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
}

function renderSafeMarkdown(value) {
  const lines = String(value).replaceAll("\r\n", "\n").split("\n");
  const blocks = [];
  let list = null;
  const closeList = () => {
    if (list) { blocks.push(`<${list.tag}>${list.items.join("")}</${list.tag}>`); list = null; }
  };
  for (const line of lines) {
    const unordered = line.match(/^\s*[-*]\s+(.+)/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)/);
    const heading = line.match(/^(#{1,3})\s+(.+)/);
    if (unordered || ordered) {
      const tag = unordered ? "ul" : "ol";
      if (!list || list.tag !== tag) { closeList(); list = { tag, items: [] }; }
      list.items.push(`<li>${renderMarkdownInline((unordered || ordered)[1])}</li>`);
    } else {
      closeList();
      if (!line.trim()) continue;
      if (heading) blocks.push(`<h${heading[1].length}>${renderMarkdownInline(heading[2])}</h${heading[1].length}>`);
      else blocks.push(`<p>${renderMarkdownInline(line)}</p>`);
    }
  }
  closeList();
  return blocks.join("") || "<p></p>";
}

function addRecipeConversationMessage(role, text) {
  recipeResult.hidden = false;
  recipeConversationResetButton.hidden = false;
  recipeSaveMarkdownButton.hidden = false;
  const message = document.createElement("article");
  message.className = `recipe-message recipe-message-${role}`;
  message.dataset.markdown = text;
  message.innerHTML = renderSafeMarkdown(text);
  recipeResult.prepend(message);
  recipeResult.scrollTop = 0;
}

function appendTone3000DiscussButtonsToLastMessage(results) {
  const message = recipeResult.firstElementChild;
  if (!message) return;
  const container = document.createElement("div");
  container.className = "ai-tone3000-candidate-list";
  const seen = new Set();
  results.forEach((result) => {
    if (seen.has(result.id)) return;
    seen.add(result.id);
    const row = document.createElement("div");
    row.className = "ai-tone3000-candidate-row";
    const label = document.createElement("span");
    label.className = "ai-tone3000-candidate-label";
    label.textContent = `${result.title} — ${result.creator}${Number.isFinite(result.match_score) ? ` · ${result.match_score}% metadata fit` : ""}`;
    row.append(label, createTone3000DiscussButton(result));
    container.append(row);
  });
  if (container.children.length) message.append(container);
}

function applyRecipe(recipe, prefix = "", { showMessage = true, noRecipeMessage = "I couldn't identify a blend direction yet. Try naming what should stay from Amp A and what should take over from Amp B—for example, ‘keep Amp A's EQ and feel; let its gain become Amp B crunch as I play harder.’" } = {}) {
  if (!recipe) {
    if (showMessage) addRecipeConversationMessage("assistant", noRecipeMessage);
    return;
  }
  setModeFromWizard(recipe.mode);
  if (recipe.mode === "character") {
    setCharacterSlider("tone", recipe.tone);
    setCharacterSlider("feel", recipe.feel);
    setCharacterSlider("drive", recipe.drive);
    setCharacterSlider("drive-low", recipe.driveLow);
    setCharacterSlider("drive-mid", recipe.driveMid);
    setCharacterSlider("drive-high", recipe.driveHigh);
    document.getElementById("drive-morph-enabled").checked = true;
  } else if (recipe.mode === "hybrid") {
    crossoverKnobSlider.value = recipe.switchKnob;
    const crossoverDb = dbFromKnob(recipe.switchKnob);
    crossoverSlider.value = crossoverDb.toFixed(2);
    crossoverBaseline = { value: crossoverSlider.value, label: "recipe" };
    crossoverValue.textContent = `${crossoverDb.toFixed(1)} dBFS`;
    transitionSlider.value = recipe.width;
    transitionValue.textContent = `${recipe.width} dB`;
    syncCrossoverKnobFromDb();
    syncPresetButtonStates();
    updateTransitionAroundSwitchNote();
  } else {
    mixSlider.value = recipe.mixB;
    updateMixValueLabel();
  }
  scheduleUpdate();
  scheduleAuditionRefresh();
  if (showMessage) addRecipeConversationMessage("assistant", `${prefix}${recipe.explanation}\n\nThe settings above are a starting point: adjust them, then render and audition the result.`);
}

async function applyRecipeFromPrompt() {
  const prompt = recipePromptInput.value.trim();
  if (!prompt) {
    addRecipeConversationMessage("assistant", "Tell me what you want to change, or describe the sound you are after.");
    return;
  }
  let recipe = recipeFromPrompt(prompt);
  let prefix = "";
  let promptWasAdded = false;
  if (localRecipeAiAvailable && recipeUseLocalAi.checked && prompt.trim()) {
    addRecipeConversationMessage("user", prompt);
    promptWasAdded = true;
    recipePromptApplyButton.disabled = true;
    const stopActivity = beginActivity("AI is working — preparing your recipe and research…");
    try {
      // An explicit amp-family change in the current brief is a deliberate
      // source-plan revision, not a follow-up tweak to the old pair.
      if (recipeSourcePlan) {
        const ampTerms = [...new Set((prompt.toLowerCase().match(/\b(fender|vox|marshall|mesa|boogie|orange|peavey|soldano|engl|hiwatt|bogner|princeton|deluxe|tweed|jcm|ac30)\b/g) || []))];
        const existingPlan = `${recipeSourcePlan.ampA} ${recipeSourcePlan.ampB}`.toLowerCase();
        if (ampTerms.some((term) => !existingPlan.includes(term))) recipeSourcePlan = null;
      }
      const research = {
        web: recipeUseWebResearch.checked,
        tone3000: recipeUseTone3000.checked,
        rig_scope: recipeTone3000RigScope.value,
        author: recipeTone3000Author.value.trim(),
      };
      const tone3000Context = selectedTone3000Capture
        ? {
          id: selectedTone3000Capture.id,
          title: selectedTone3000Capture.title,
          creator: selectedTone3000Capture.creator,
          description: selectedTone3000Capture.description || "No description",
          models: selectedTone3000Capture.models.map((model) => model.name),
          ...(selectedTone3000Capture.sourceRole ? { source_role: selectedTone3000Capture.sourceRole } : {}),
        }
        : null;
      const response = await fetch("/api/local_llm/recipe", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt, tone3000_context: tone3000Context, source_plan: recipeSourcePlan, history: recipeConversationHistory, research }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "AI provider is unavailable");
      if (data.source_plan) recipeSourcePlan = data.source_plan;
      if (selectedTone3000Capture && data.selected_source_role) {
        selectedTone3000Capture.sourceRole = data.selected_source_role;
        showSelectedTone3000Capture(selectedTone3000Capture);
      }
      const assistantContent = data.recipe
        ? `${data.reply}\n\n${data.recipe.explanation}`
        : data.reply;
      const researchWarning = (data.research_warnings || []).length
        ? `\n\nResearch note: ${data.research_warnings.join(" ")}`
        : "";
      const tone3000Candidates = (data.tone3000_results || []).length
        ? `\n\nTONE3000 capture candidates:\n${data.tone3000_results.map((match) => (
          `- ${match.title} — ${match.creator}${match.query ? ` (searched: ${match.query})` : ""}`
        )).join("\n")}`
        : "";
      recipeConversationHistory.push({ role: "user", content: prompt }, { role: "assistant", content: assistantContent });
      recipeConversationHistory = recipeConversationHistory.slice(-8);
      addRecipeConversationMessage("assistant", assistantContent + tone3000Candidates + researchWarning);
      if ((data.tone3000_results || []).length) appendTone3000DiscussButtonsToLastMessage(data.tone3000_results);
      if (data.recipe) applyRecipe(data.recipe, "", { showMessage: false });
      recipePromptInput.value = "";
      return;
    } catch (error) {
      const reason = error && error.message ? error.message : "an unknown error";
      prefix = `The AI provider's response could not be used (${reason}), so the built-in suggestion was used. `;
    } finally {
      stopActivity();
      recipePromptApplyButton.disabled = false;
    }
  }
  if (!promptWasAdded) addRecipeConversationMessage("user", prompt);
  applyRecipe(recipe, prefix, {
    noRecipeMessage: `${prefix}I couldn't turn that into settings with the built-in suggestion tool. Try naming what should stay from Amp A and what should take over from Amp B, or enable an AI provider for open-ended questions like file/pack choices.`,
  });
  recipePromptInput.value = "";
}

recipePromptApplyButton.addEventListener("click", () => { applyRecipeFromPrompt(); });
recipePromptInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); applyRecipeFromPrompt(); }
});
recipeConversationResetButton.addEventListener("click", () => {
  recipeConversationHistory = [];
  recipeSourcePlan = null;
  recipeResult.replaceChildren();
  recipeResult.hidden = true;
  recipeConversationResetButton.hidden = true;
  recipeSaveMarkdownButton.hidden = true;
  recipePromptInput.focus();
});
async function saveBlobAsFile(filename, blob) {
  if (isTauriDesktop) {
    // Read the blob directly rather than going through an <a download>
    // click -- the anchor-based path revokes its object URL synchronously
    // right after the click, before an async Tauri save could ever read it.
    const buf = await blob.arrayBuffer();
    const dataBase64 = btoa(String.fromCharCode(...new Uint8Array(buf)));
    await desktopSave("save_bytes", { filename, dataBase64 }, filename);
    return filename;
  }
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  return filename;
}

async function triggerFileDownload(url, filename) {
  if (isTauriDesktop) {
    const absoluteUrl = new URL(url, window.location.origin).toString();
    await desktopSave("save_file_from_url", { url: absoluteUrl, filename }, filename);
    return filename;
  }
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  return filename;
}

recipeSaveMarkdownButton.addEventListener("click", () => {
  const messages = [...recipeResult.querySelectorAll(".recipe-message")].reverse();
  const markdown = [
    "# NAM Mixer AI Assistant conversation",
    "",
    ...messages.flatMap((message) => [
      `## ${message.classList.contains("recipe-message-user") ? "You" : "AI Assistant"}`,
      "",
      message.dataset.markdown.trim(),
      "",
    ]),
  ].join("\n");
  saveBlobAsFile("nam-mixer-ai-conversation.md", new Blob([markdown], { type: "text/markdown;charset=utf-8" }));
});

wizardAnalyseButton.addEventListener("click", async () => {
  wizardResult.hidden = false;
  wizardResult.textContent = "Listening to the rendered pair…";
  wizardAnalyseButton.disabled = true;
  const stopActivity = beginActivity("Analysing the rendered amps…");
  try {
    const resp = await fetch("/api/wizard/insight", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ render_id: activeRenderId }) });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Could not analyse the rendered amps.");
    wizardResult.textContent = `${data.level_text} ${data.tone_text} ${data.feel_text}`;
  } catch (err) {
    wizardResult.textContent = `Analysis unavailable: ${err.message}`;
  } finally {
    stopActivity();
    wizardAnalyseButton.disabled = !havePair;
  }
});

const characterSliders = ["tone", "feel", "drive", "drive-low", "drive-mid", "drive-high"];
characterSliders.forEach((name) => {
  const slider = document.getElementById(`${name}-slider`);
  const label = document.getElementById(`${name}-value`);
  const update = () => { label.textContent = `${slider.value}% B`; scheduleUpdate(); scheduleAuditionRefresh(); };
  slider.addEventListener("input", update);
});
document.getElementById("drive-morph-enabled").addEventListener("change", () => { scheduleUpdate(); scheduleAuditionRefresh(); });

document.getElementById("auto-level-match").addEventListener("change", () => {
  invalidateLiveAudition("Level-match setting changed — start live blend again to load the matching stems.");
  scheduleUpdate();
  scheduleAuditionRefresh();
});

const ampBTrimSlider = document.getElementById("amp-b-trim");
const ampBTrimValue = document.getElementById("amp-b-trim-value");
ampBTrimSlider.addEventListener("input", () => {
  ampBTrimValue.textContent = `${fmtSigned(ampBTrimSlider.value)} dB`;
  invalidateLiveAudition("Level trim changed — start live blend again to load the matching stems.");
  scheduleUpdate();
  scheduleAuditionRefresh();
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
  document.getElementById("btn-preview-mix"),
  document.getElementById("btn-preview-b"),
];
const player = document.getElementById("player");
const liveBlendButton = document.getElementById("btn-live-blend");
const liveBlendStatus = document.getElementById("live-blend-status");
const autoAuditionToggle = document.getElementById("auto-audition");
const trimReadout = document.getElementById("trim-readout");
const renderStatus = document.getElementById("render-status");
const rendererRetryButton = document.getElementById("btn-renderer-retry");
const rendererHelp = document.getElementById("renderer-help");
const rendererPath = document.getElementById("renderer-path");
const journeyCanvas = document.getElementById("journey-canvas");
const journeyTooltip = document.getElementById("journey-tooltip");
// The chart is drawn while its <details> may still be collapsed (0x0 canvas
// at that point, since a closed <details> reports zero size) -- redraw with
// the already-fetched data once the user actually opens it.
document.getElementById("hybrid-only-diagnostics").addEventListener("toggle", (evt) => {
  if (evt.target.open) drawJourney();
});
const journeyEmpty = document.getElementById("journey-empty");
const coverageTable = document.getElementById("coverage-table");
const coverageTbody = document.getElementById("coverage-tbody");
const coverageEmpty = document.getElementById("coverage-empty");
const coverageWarning = document.getElementById("coverage-warning");

let havePair = false;
let activeRenderId = null;
let renderGeneration = 0;
let updateTimer = null;
let lastJourneyData = null;
let lastSourcePlayed = null;
let previewRequestId = 0;
let auditionRefreshTimer = null;

async function refreshRendererReadiness() {
  rendererRetryButton.disabled = true;
  renderStatus.textContent = "Checking the native renderer…";
  try {
    const resp = await fetch("/api/renderer/readiness");
    const data = await resp.json();
    if (data.verified) {
      renderStatus.textContent = "Renderer ready.";
      renderStatus.dataset.rendererState = "ready";
      rendererRetryButton.hidden = true;
      rendererHelp.hidden = true;
      rendererPath.textContent = data.path ? `Verified executable: ${data.path}` : "";
    } else {
      const state = data.found ? "unusable" : "missing";
      renderStatus.dataset.rendererState = state;
      renderStatus.textContent = data.found
        ? `Renderer found but could not be verified: ${data.error}`
        : `Renderer is not installed or could not be found: ${data.error}`;
      rendererPath.textContent = data.path ? `Found executable: ${data.path}` : "No executable path was found.";
      rendererRetryButton.hidden = false;
      rendererHelp.hidden = false;
    }
    return data;
  } catch (err) {
    renderStatus.textContent = `Could not verify the renderer: ${err}`;
    renderStatus.dataset.rendererState = "check-failed";
    rendererRetryButton.hidden = false;
    rendererHelp.hidden = false;
    return { found: false, verified: false, error: String(err) };
  } finally {
    rendererRetryButton.disabled = false;
  }
}
rendererRetryButton.addEventListener("click", refreshRendererReadiness);
refreshRendererReadiness();

function clearAudition() {
  // Do not leave an old result playing after an upstream setting has changed.
  previewRequestId += 1;
  player.pause();
  const previousUrl = player.src;
  player.onloadedmetadata = null;
  player.removeAttribute("src");
  player.load();
  if (previousUrl.startsWith("blob:")) URL.revokeObjectURL(previousUrl);
  lastPreviewSource = null;
  lastSourcePlayed = null;
}

function scheduleAuditionRefresh(source = lastPreviewSource) {
  // Refresh can follow an explicit play action, but controls must never cause
  // sound to start by themselves.
  if (!source || !havePair || !autoAuditionToggle.checked || liveAudition.active || player.paused) return;
  clearTimeout(auditionRefreshTimer);
  auditionRefreshTimer = setTimeout(() => preview(source, { preservePosition: true, quiet: true }), 140);
}

// NAM rendering remains server-side.  Once the pair is rendered, this holds
// two decoded stems in Web Audio and changes their *linear* blend gain at
// audio rate.  That is the exact Fixed Blend equation, not a preview shortcut.
const liveAudition = {
  active: false,
  requestId: 0,
  context: null,
  sourceA: null,
  sourceB: null,
  gainA: null,
  gainB: null,
  compressor: null,
  outputGain: null,
  stems: null,
  mixB: 0.5,
  stop() {
    this.requestId += 1;
    [this.sourceA, this.sourceB].forEach((source) => {
      if (source) { try { source.stop(); } catch (_) { /* already stopped */ } }
    });
    this.active = false;
    this.sourceA = this.sourceB = this.gainA = this.gainB = null;
    this.outputGain = this.stems = null;
    liveBlendButton.disabled = !havePair;
    updateLiveAuditionButton();
  },
  setMix(mixB, immediate = false) {
    if (!this.active) return;
    this.mixB = mixB;
    this.updateOutputGain(immediate);
    const now = this.context.currentTime;
    if (immediate) {
      this.gainA.gain.cancelScheduledValues(now);
      this.gainB.gain.cancelScheduledValues(now);
      this.gainA.gain.setValueAtTime(1 - mixB, now);
      this.gainB.gain.setValueAtTime(mixB, now);
      return;
    }
    // A short ramp prevents zipper/click artefacts while dragging.
    this.gainA.gain.setTargetAtTime(1 - mixB, now, 0.012);
    this.gainB.gain.setTargetAtTime(mixB, now, 0.012);
  },
  updateOutputGain(immediate = false) {
    if (!this.active || !this.outputGain || !this.stems) return;
    const a = this.stems.getChannelData(0);
    const b = this.stems.getChannelData(1);
    let peak = 0;
    for (let i = 0; i < a.length; i++) {
      peak = Math.max(peak, Math.abs(a[i] * (1 - this.mixB) + b[i] * this.mixB));
    }
    const beforeDb = peak > 0 ? 20 * Math.log10(peak) : -Infinity;
    const params = outputGainParamsBody();
    // Match compute_auto_output_gain_db: boost only, targeting -3 dBFS.
    const gainDb = params.output_gain_mode === "auto"
      ? (peak > 0 ? Math.max(0, -3 - beforeDb) : 0)
      : params.manual_output_gain_db;
    const gain = 10 ** (gainDb / 20);
    const now = this.context.currentTime;
    this.outputGain.gain.cancelScheduledValues(now);
    if (immediate) this.outputGain.gain.setValueAtTime(gain, now);
    else this.outputGain.gain.setTargetAtTime(gain, now, 0.012);
    const values = {
      "X-Output-Gain-Mode": params.output_gain_mode,
      "X-Output-Gain-Db": String(gainDb),
      "X-Peak-Before-Output-Gain-Dbfs": String(beforeDb),
      "X-Peak-After-Output-Gain-Dbfs": String(beforeDb + gainDb),
      "X-Output-Gain-Will-Clip-Preview": String(beforeDb + gainDb > -1),
    };
    updateOutputGainReadout({ get: (name) => values[name] ?? null });
  },
};

function updateLiveAuditionButton() {
  if (liveAudition.active) liveBlendButton.textContent = "Stop live mix adjustment";
  else liveBlendButton.textContent = currentMode === "blend" ? "Adjust the mix while listening" : "Live mix: Always-on mix only";
}

function splitStereoBuffer(context, decoded, channel) {
  const buffer = context.createBuffer(1, decoded.length, decoded.sampleRate);
  buffer.copyToChannel(decoded.getChannelData(channel), 0);
  return buffer;
}

function invalidateLiveAudition(message) {
  liveAudition.stop();
  liveBlendStatus.textContent = message;
}

async function startLiveBlend() {
  if (!havePair) return;
  if (currentMode !== "blend") {
    liveBlendStatus.textContent = "Live mix adjustment is available in Always-on mix. This result updates after you adjust a control.";
    return;
  }
  if (liveAudition.active) {
    liveAudition.stop();
    liveBlendStatus.textContent = "Live blend stopped.";
    return;
  }
  liveBlendButton.disabled = true;
  const requestId = ++liveAudition.requestId;
  liveBlendStatus.textContent = "Loading the prepared amps...";
  try {
    const resp = await fetch("/api/live_blend_stems", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // Keep stems unscaled; browser gain follows current controls even while
      // this request is loading, and auto gain follows the live mix ratio.
      body: JSON.stringify({ render_id: activeRenderId, ...blendParamsBody(), ...cabParamsBody(), output_gain_mode: "manual", manual_output_gain_db: 0 }),
    });
    if (!resp.ok) throw new Error((await resp.json()).error || "Could not load live stems.");
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) throw new Error("This browser does not support Web Audio.");
    const context = liveAudition.context || new Context();
    liveAudition.context = context;
    await context.resume();
    const decoded = await context.decodeAudioData(await (await resp.blob()).arrayBuffer());
    if (requestId !== liveAudition.requestId || !havePair || currentMode !== "blend") return;
    if (decoded.numberOfChannels < 2) throw new Error("Live stem response was not stereo.");
    player.pause();
    const sourceA = context.createBufferSource();
    const sourceB = context.createBufferSource();
    sourceA.buffer = splitStereoBuffer(context, decoded, 0);
    sourceB.buffer = splitStereoBuffer(context, decoded, 1);
    sourceA.loop = sourceB.loop = true;
    const gainA = context.createGain();
    const gainB = context.createGain();
    const outputGain = context.createGain();
    const compressor = context.createDynamicsCompressor();
    compressor.threshold.value = -3;
    compressor.knee.value = 4;
    compressor.ratio.value = 12;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.12;
    sourceA.connect(gainA).connect(outputGain);
    sourceB.connect(gainB).connect(outputGain);
    outputGain.connect(compressor);
    compressor.connect(context.destination);
    liveAudition.sourceA = sourceA;
    liveAudition.sourceB = sourceB;
    liveAudition.gainA = gainA;
    liveAudition.gainB = gainB;
    liveAudition.compressor = compressor;
    liveAudition.outputGain = outputGain;
    liveAudition.stems = decoded;
    liveAudition.active = true;
    liveAudition.setMix(parseInt(mixSlider.value, 10) / 100.0, true);
    sourceA.start();
    sourceB.start();
    updateLiveAuditionButton();
    liveBlendStatus.textContent = "Adjust the mix while listening. Changes are immediate.";
  } catch (err) {
    if (requestId !== liveAudition.requestId) return;
    liveAudition.stop();
    liveBlendStatus.textContent = "Live blend unavailable: " + err.message;
  } finally {
    if (requestId === liveAudition.requestId) liveBlendButton.disabled = !havePair;
  }
}

function hybridParamsBody() {
  return {
    crossover_dbfs: parseFloat(crossoverSlider.value),
    transition_width_db: parseFloat(transitionSlider.value),
    auto_level: document.getElementById("auto-level-match").checked,
    manual_b_trim_db: parseFloat(ampBTrimSlider.value) || 0.0,
  };
}

function blendParamsBody() {
  return {
    mix_b: parseInt(mixSlider.value, 10) / 100.0,
    auto_level: document.getElementById("auto-level-match").checked,
    manual_b_trim_db: parseFloat(ampBTrimSlider.value) || 0.0,
  };
}

function characterParamsBody() {
  const pct = (name) => parseInt(document.getElementById(`${name}-slider`).value, 10) / 100.0;
  const morph = document.getElementById("drive-morph-enabled").checked;
  return { tone_mix_b: pct("tone"), feel_mix_b: pct("feel"), drive_mix_b: pct("drive"),
    drive_low_mix_b: morph ? pct("drive-low") : null, drive_mid_mix_b: morph ? pct("drive-mid") : null,
    drive_high_mix_b: morph ? pct("drive-high") : null };
}

// Rendered display for a LowLevelResponseCheck dict (see
// hybrid.character_blend.LowLevelResponseCheck / docs/blend-mode-fixes.md
// Phase 6) -- shared by the on-demand check button and the Generate result.
function renderLowLevelResponseHtml(check) {
  if (!check) return "";
  const rows = check.levels_db
    .map((lv, i) => `<tr><td>${fmtSigned(lv)} dB</td><td>${check.output_rms_dbfs[i].toFixed(1)} dBFS</td></tr>`)
    .join("");
  const verdict = check.ok
    ? `<div class="ok-line">&#10003; continuous low-level response, no dead zone</div>`
    : `<div class="warning-box">&#10007; low-level collapse detected (max step error ${check.max_step_error_db.toFixed(1)} dB). Do not train this design -- see docs/blend-mode-fixes.md.</div>`;
  return `
    <div><strong>LOW-LEVEL RESPONSE</strong></div>
    <table class="coverage-table"><tbody>${rows}</tbody></table>
    ${verdict}
  `;
}

// Mode-aware params for whichever mode tab is active -- shared by
// /api/mix_info, /api/preview (source=hybrid/blend), and /api/generate.
function currentModeParamsBody() {
  if (currentMode === "blend") return { mode: "blend", ...blendParamsBody() };
  if (currentMode === "character") return { mode: "character", ...characterParamsBody() };
  return { mode: "hybrid", ...hybridParamsBody() };
}

function cabParamsBody() {
  const enabled = cabPreviewEnabled.checked && !!cabServerPath;
  return {
    cab_path: enabled ? cabServerPath : null,
    cab_preview_enabled: enabled,
  };
}

// --- Output gain (shared, post-combination, all modes) -----------------
// Mirrors cabParamsBody()'s "shared control" pattern -- see
// hybrid/design.py's output_gain_mode/manual_output_gain_db and
// hybrid/safety.py's compute_auto_output_gain_db/apply_output_gain.
const outputGainAutoCheckbox = document.getElementById("output-gain-auto");
const outputGainManualSlider = document.getElementById("output-gain-manual-slider");
const outputGainManualValue = document.getElementById("output-gain-manual-value");
const outputGainReadout = document.getElementById("output-gain-readout");
const outputGainWarning = document.getElementById("output-gain-warning");

function outputGainParamsBody() {
  return {
    output_gain_mode: outputGainAutoCheckbox.checked ? "auto" : "manual",
    manual_output_gain_db: parseFloat(outputGainManualSlider.value) || 0.0,
  };
}

outputGainAutoCheckbox.addEventListener("change", () => {
  outputGainManualSlider.disabled = outputGainAutoCheckbox.checked;
  liveAudition.updateOutputGain();
  scheduleUpdate();
  scheduleAuditionRefresh();
});
outputGainManualSlider.addEventListener("input", () => {
  outputGainManualValue.textContent = `${fmtSigned(outputGainManualSlider.value)} dB`;
  liveAudition.updateOutputGain();
  scheduleUpdate();
  scheduleAuditionRefresh();
});

// Reads the X-Output-Gain-* headers a /api/preview response carries for
// source=hybrid/blend/character (see app.py's /api/preview) and updates the
// readout + clip warning. Called from preview() below.
function updateOutputGainReadout(headers) {
  const mode = headers.get("X-Output-Gain-Mode");
  if (mode === null) return; // not a combined-result preview (e.g. raw A/B)
  const gainDb = headers.get("X-Output-Gain-Db");
  const peakBefore = headers.get("X-Peak-Before-Output-Gain-Dbfs");
  const peakAfter = headers.get("X-Peak-After-Output-Gain-Dbfs");
  const willClip = headers.get("X-Output-Gain-Will-Clip-Preview") === "true";
  outputGainReadout.textContent =
    `${mode === "auto" ? "Automatic" : "Manual"} output adjustment: ${fmtSigned(gainDb)} dB`;
  if (willClip) {
    outputGainWarning.hidden = false;
    outputGainWarning.textContent =
      mode === "manual"
        ? "This manual output setting is too high and will distort the preview. Reduce it so what you hear matches the model you will create."
        : "Automatic output level is unexpectedly too high for preview. Please report this.";
  } else {
    outputGainWarning.hidden = true;
  }
}

function scheduleUpdate() {
  resetGeneratedModel("The sound changed. Create new training files before starting another training run.");
  if (!havePair) return;
  clearTimeout(updateTimer);
  updateTimer = setTimeout(() => {
    updateTrimReadout();
    if (currentMode === "hybrid") {
      updateJourney();
      updateCoverage();
    }
  }, 150);
}

async function updateTrimReadout() {
  try {
    const resp = await fetch("/api/mix_info", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ render_id: activeRenderId, ...currentModeParamsBody() }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      trimReadout.textContent = "Trim error: " + data.error;
      return;
    }
    if (data.mode === "character") {
      document.getElementById("character-readout").textContent =
        `Drive donor trajectory: ${Math.round(data.drive_weight_b_min * 100)}–${Math.round(data.drive_weight_b_max * 100)}% Amp B.`;
    } else trimReadout.textContent = `Auto match ${fmtSigned(data.auto_trim_db)} dB  ·  effective trim ${fmtSigned(data.effective_b_trim_db)} dB`;
  } catch (err) {
    trimReadout.textContent = "Trim update failed: " + err;
  }
}

async function updateJourney() {
  try {
    const resp = await fetch("/api/blend_curve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ render_id: activeRenderId, ...hybridParamsBody() }),
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
        render_id: activeRenderId,
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
      if (row.profile_id === profileSelect.value) {
        tr.classList.add("current-profile");
      }
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

  const n0 = data.times.length;
  const meanMixPct = Math.round(
    (data.blend_weight.reduce((a, b) => a + b, 0) / n0) * 100
  );
  journeyCanvas.setAttribute(
    "aria-label",
    `Crossfade journey over ${data.times[n0 - 1].toFixed(1)} seconds. ` +
      `Crossover at ${data.crossover_dbfs.toFixed(1)} dBFS with a ${data.transition_width_db} dB transition band. ` +
      `Averages ${meanMixPct}% toward Amp B across the clip.`
  );

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

const renderPairBtn = document.getElementById("btn-render-pair");
rebuildPreviewButton.addEventListener("click", () => {
  rebuildPreviewButton.disabled = true;
  setWorkflowStage("configure");
  renderPairBtn.click();
});
const renderDependentControls = [
  "amp-a-file", "amp-b-file", "di-selector", "instrument-select", "input-profile-select",
  "custom-gain-slider", "calibration-mode-select", "reference-dbu-input",
  "amp-a-input-gain-slider", "amp-b-input-gain-slider", "test-gain-slider", "btn-wizard-apply",
].map((id) => document.getElementById(id));

function setRenderBusy(busy) {
  renderDependentControls.forEach((control) => { if (control) control.disabled = busy; });
}

// The actual expensive call (real NAM inference) -- shared by the manual
// "Render Amps" button and the auto-triggered test-gain re-render below.
// Throws with a user-facing message on any failure.
async function doRenderPair() {
  const amp_a_path = ampServerPaths.a;
  const amp_b_path = ampServerPaths.b;
  const di_file = document.getElementById("di-selector").value;
  if (!amp_a_path || !amp_b_path || !di_file) {
    throw new Error("Choose both Amp A and Amp B .nam files and pick a DI clip first.");
  }
  const renderer = await refreshRendererReadiness();
  if (!renderer.verified) throw new Error(renderer.error || "Native renderer is not ready. Build native/nam_render, then retry.");

  const profile = currentProfile();
  const instrument_type = instrumentSelect.value;
  const input_profile_id = profileSelect.value;
  const custom_input_gain_db = profile && profile.requires_custom_gain ? parseFloat(customGainSlider.value) : null;
  const calibration_mode = calibrationModeSelect.value;
  const parsedReferenceDbu = parseFloat(referenceDbuInput.value);
  const reference_input_level_dbu = Number.isFinite(parsedReferenceDbu) ? parsedReferenceDbu : 12.0;
  const test_gain_db = parseFloat(testGainSlider.value) || 0.0;
  const amp_a_input_gain_db = parseFloat(ampAInputGainSlider.value) || 0.0;
  const amp_b_input_gain_db = parseFloat(ampBInputGainSlider.value) || 0.0;

  const resp = await fetch("/api/render_pair", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      amp_a_path, amp_b_path, di_file,
      instrument_type, input_profile_id, custom_input_gain_db,
      calibration_mode, reference_input_level_dbu, test_gain_db,
      amp_a_input_gain_db, amp_b_input_gain_db,
    }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.error || "Render failed.");
  }
  return data;
}

// Applies a successful render's side effects. `applySuggestedCrossover` is
// false for the automatic test-gain re-render -- jumping the crossover
// slider every time you nudge the test-gain control would fight the whole
// point of stress-testing a design you already settled on.
function applyRenderResult(data, { applySuggestedCrossover }) {
  if (liveAudition.active) {
    liveAudition.stop();
    liveBlendStatus.textContent = "Amp pair changed — start live blend again to load the new stems.";
  }
  renderPairBtn.classList.remove("btn-render-stale");
  updateCrossoverKnobCalibration(data.blend_envelope_percentiles);
  if (data.warnings && data.warnings.length) {
    renderWarnings.hidden = false;
    renderWarnings.className = isCalibrationOnlyWarning(data.warnings) ? "info-box" : "warning-box";
    renderWarnings.textContent = renderWarningText(data.warnings);
  } else {
    renderWarnings.hidden = true;
  }

  if (applySuggestedCrossover) {
    suggestedCrossoverNote.innerHTML = "";
    if (data.suggested_crossover_dbfs !== null && data.suggested_crossover_dbfs !== undefined) {
      const suggested = data.suggested_crossover_dbfs;
      crossoverSlider.value = suggested.toFixed(1);
      crossoverValue.textContent = `${suggested.toFixed(1)} dBFS`;
      syncCrossoverKnobFromDb();
      suggestedCrossoverNote.textContent =
        `Crossover set to ${suggested.toFixed(1)} dBFS, suggested from this DI's active-signal level. `;
      const resetBtn = document.createElement("button");
      resetBtn.type = "button";
      resetBtn.className = "link-btn";
      const baselineLabel = crossoverBaseline.label === "default" ? "default" : `preset "${crossoverBaseline.label}"`;
      resetBtn.textContent = `Reset to ${baselineLabel} (${parseFloat(crossoverBaseline.value).toFixed(1)})`;
      resetBtn.addEventListener("click", () => {
        crossoverSlider.value = crossoverBaseline.value;
        crossoverValue.textContent = `${parseFloat(crossoverBaseline.value).toFixed(1)} dBFS`;
        syncCrossoverKnobFromDb();
        scheduleUpdate();
      });
      suggestedCrossoverNote.appendChild(resetBtn);
    }
  }

  activeRenderId = data.render_id;
  previewButtons.forEach((btn) => (btn.disabled = false));
  liveBlendButton.disabled = false;
  document.getElementById("btn-character-low-level-check").disabled = false;
  havePair = true;
  staleCallout.hidden = true;
  renderPairBtn.textContent = "Prepare amps for comparison";
  syncTrainingControls();
  wizardAnalyseButton.disabled = false;
  setWorkflowStage("compare");
  updateTrimReadout();
  updateJourney();
  updateCoverage();
}

renderPairBtn.addEventListener("click", async () => {
  // Only auto-apply the suggested crossover on the VERY FIRST render of a
  // session, AND only when nothing has already given the crossover slider
  // an explicit value -- a re-render (new DI, profile, calibration change,
  // or just clicking Render Amps again) must never silently discard a
  // crossover the user already set, exactly like the test-gain auto-render
  // below already avoids doing (see its comment). Also covers importing a
  // session THEN clicking Render Amps for the first time this
  // session -- crossoverBaseline.label is no longer "default" once a preset
  // has been loaded (see applySessionSettings), so that imported value
  // survives its first render too, not just subsequent ones.
  const isFirstRenderThisSession = !havePair && crossoverBaseline.label === "default";
  markProfileStale("Rendering amps");
  const requestGeneration = renderGeneration;
  renderPairBtn.disabled = true;
  setRenderBusy(true);
  const stopActivity = beginActivity("Rendering both amps — this can take a moment…");
  renderWarnings.hidden = true;
  previewButtons.forEach((btn) => (btn.disabled = true));
  document.getElementById("btn-character-low-level-check").disabled = true;
  try {
    const data = await doRenderPair();
    if (requestGeneration !== renderGeneration) return;
    applyRenderResult(data, { applySuggestedCrossover: isFirstRenderThisSession });
    renderStatus.textContent =
      `Rendered ${data.duration_s.toFixed(1)}s @ ${data.sample_rate} Hz -- ` +
      `input peak ${data.input_peak_dbfs.toFixed(1)} dBFS.`;
    setStatus("Amp pair rendered and cached -- sliders now only recompute the blend.");
  } catch (err) {
    renderStatus.textContent = "Error: " + err.message;
    setStatus("Render failed.", true);
  } finally {
    stopActivity();
    setRenderBusy(false);
    renderPairBtn.disabled = false;
    rebuildPreviewButton.disabled = false;
  }
});

let lastPreviewSource = null;

async function preview(requestedSource, { preservePosition = false, quiet = false, resumeState = null } = {}) {
  if (!havePair || !activeRenderId) return;
  // "mix" means "whichever design mode's combined result is active" --
  // resolves to the active design source without a separate approximation.
  const source = requestedSource === "mix" ? currentMode : requestedSource;
  lastPreviewSource = requestedSource;
  const requestId = ++previewRequestId;
  const wasPlaying = resumeState ? resumeState.playing : !player.paused;
  const resumeAt = resumeState ? resumeState.position : preservePosition && Number.isFinite(player.currentTime) ? player.currentTime : 0;
  const modeParams = ["hybrid", "blend", "character"].includes(source) ? currentModeParamsBody() : {};
  const outputGainParams = ["hybrid", "blend", "character"].includes(source) ? outputGainParamsBody() : {};
  const body = { source, render_id: activeRenderId, ...modeParams, ...cabParamsBody(), ...outputGainParams };
  try {
    const resp = await fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const data = await resp.json();
      if (requestId !== previewRequestId) return;
      if (resp.status === 409) markProfileStale("The rendered pair has been replaced");
      setStatus(data.error || "Preview failed.", true);
      return;
    }
    // Ignore an older cached-blend request that returned after a newer slider
    // value. This prevents audible jumps backwards during a fast drag.
    if (requestId !== previewRequestId) return;
    if (source === "hybrid" || source === "blend") {
      const auto = resp.headers.get("X-Auto-Trim-Db");
      const effective = resp.headers.get("X-Effective-Trim-Db");
      trimReadout.textContent =
        `Auto match ${fmtSigned(auto)} dB  ·  effective trim ${fmtSigned(effective)} dB`;
    }
    updateOutputGainReadout(resp.headers);
    const blob = await resp.blob();
    if (requestId !== previewRequestId) return;
    const previousUrl = player.src.startsWith("blob:") ? player.src : null;
    player.src = URL.createObjectURL(blob);
    if (previousUrl) URL.revokeObjectURL(previousUrl);
    lastSourcePlayed = requestedSource;
    player.onloadedmetadata = () => {
      if (requestId !== previewRequestId) return;
      if (resumeAt > 0 && player.duration) player.currentTime = Math.min(resumeAt, Math.max(0, player.duration - 0.02));
      if (wasPlaying) player.play();
    };
    if (!quiet) setStatus(`Playing ${source.toUpperCase()}.`);
  } catch (err) {
    if (requestId !== previewRequestId) return;
    setStatus("Request failed: " + err, true);
  }
}

document.getElementById("btn-preview-a").addEventListener("click", () => preview("a"));
document.getElementById("btn-preview-mix").addEventListener("click", () => preview("mix"));
document.getElementById("btn-preview-b").addEventListener("click", () => preview("b"));
liveBlendButton.addEventListener("click", startLiveBlend);
const trainingInputFile = document.getElementById("training-input-file");
const trainingInputStatus = document.getElementById("training-input-status");
const trainingInputDefaultNote = document.getElementById("training-input-default-note");
const generateStatus = document.getElementById("generate-status");
const generateResult = document.getElementById("generate-result");
let trainingInputReady = false;

async function refreshTrainingInputStatus() {
  try {
    const resp = await fetch("/api/training_input/status");
    const data = await resp.json();
    trainingInputReady = !!data.ready;
    // A "ready" file is always byte-identical to the official NAM v3.0.0
    // input -- validate_training_input (hybrid/training_target.py) rejects
    // anything else outright, including a different custom sweep. So
    // "ready" and "the bundled default (or an identical copy of it) is
    // loaded" are the same fact; make that obvious instead of leaving the
    // bundled file silently in place with no indication it's the default.
    if (trainingInputDefaultNote) trainingInputDefaultNote.hidden = !data.ready;
    trainingInputStatus.textContent = data.ready
      ? `Ready: valid training input loaded (${data.sample_rate / 1000} kHz)`
      : "Add the official NAM training input to continue.";
  } catch (err) {
    trainingInputStatus.textContent = "Could not check training input status: " + err;
    trainingInputReady = false;
    if (trainingInputDefaultNote) trainingInputDefaultNote.hidden = true;
  } finally {
    syncTrainingControls();
  }
}
refreshTrainingInputStatus();

trainingInputFile.addEventListener("change", async () => {
  const file = trainingInputFile.files[0];
  if (!file) return;
  trainingInputReady = false;
  syncTrainingControls();
  const stopActivity = beginActivity(`Uploading training input ${file.name}…`);
  const formData = new FormData();
  formData.append("file", file);
  try {
    const resp = await fetch("/api/training_input/upload", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      trainingInputStatus.textContent = "Error: " + data.error;
      trainingInputReady = false;
      if (trainingInputDefaultNote) trainingInputDefaultNote.hidden = true;
      return;
    }
    trainingInputReady = true;
    if (trainingInputDefaultNote) trainingInputDefaultNote.hidden = false;
    trainingInputStatus.textContent = `Ready: valid training input loaded (${data.sample_rate / 1000} kHz)`;
  } catch (err) {
    trainingInputStatus.textContent = "Upload failed: " + err;
  } finally {
    stopActivity();
    syncTrainingControls();
  }
});

const characterLowLevelBtn = document.getElementById("btn-character-low-level-check");
const characterLowLevelStatus = document.getElementById("character-low-level-status");
const characterLowLevelResult = document.getElementById("character-low-level-result");
characterLowLevelBtn.addEventListener("click", async () => {
  if (!havePair) {
    setStatus("Render and audition an amp pair first.", true);
    return;
  }
  characterLowLevelBtn.disabled = true;
  const requestGeneration = renderGeneration;
  characterLowLevelResult.hidden = true;
  const stopActivity = beginActivity("Checking the quiet-playing response…");
  try {
    const resp = await fetch("/api/character/low_level_check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ render_id: activeRenderId, ...characterParamsBody() }),
    });
    const data = await resp.json();
    if (requestGeneration !== renderGeneration) return;
    if (!resp.ok) {
      characterLowLevelStatus.textContent = "Error: " + (data.error || "low-level check failed");
      return;
    }
    characterLowLevelStatus.textContent = "Quiet-playing check complete.";
    characterLowLevelResult.hidden = false;
    characterLowLevelResult.innerHTML = renderLowLevelResponseHtml(data.low_level_response);
  } catch (err) {
    if (requestGeneration !== renderGeneration) return;
    characterLowLevelStatus.textContent = "Request failed: " + err;
  } finally {
    stopActivity();
    characterLowLevelBtn.disabled = !havePair;
  }
});

const generateBtn = document.getElementById("btn-generate");
const modelNameInput = document.getElementById("model-name");
const cabDisplayNameInput = document.getElementById("cab-display-name");
let generationPending = false;
modelNameInput.addEventListener("change", () => {
  const name = modelNameInput.value.trim();
  if (!name || !activeSessionId) return;
  activeSessionName = name;
  persistActiveSession().catch((err) => {
    console.warn("Could not save the updated model name:", err);
    sessionSettingsStatus.textContent = "Could not save the model name: " + err.message;
  });
});
// Shared by the explicit "Create training files" button AND by the Train
// buttons (local/Kaggle), which call this first whenever there's no bundle
// yet -- that's the whole point of the merge: a user can go straight from
// rendering to clicking Train without a separate manual Generate step.
// Returns true only once `lastDesignId` is actually set on success.
async function runGenerate() {
  if (generationPending) return false;
  if (!havePair) {
    setStatus("Render and audition an amp pair first.", true);
    return false;
  }
  if (!trainingInputReady) {
    setStatus("Upload the official NAM training input first.", true);
    return false;
  }
  generationPending = true;
  syncTrainingControls();
  const requestGeneration = renderGeneration;
  const stopActivity = beginActivity("Creating training files — processing the official input…");
  generateResult.hidden = true;
  try {
    const resp = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        render_id: activeRenderId,
        ...currentModeParamsBody(),
        // A browser can retain a cached page template while fetching a newer
        // app.js after an update. Treat the new optional field defensively so
        // that mismatch cannot block generation; the API derives a real name
        // from the selected amps until the page is refreshed.
        model_name: modelNameInput?.value?.trim() || "",
        cab_path: cabServerPath || null,
        cab_preview_enabled: cabPreviewEnabled.checked,
        cab_export_mode: cabExportMode.value,
        cab_display_name: cabDisplayNameInput?.value?.trim() || "",
        ...outputGainParamsBody(),
      }),
    });
    const data = await resp.json();
    if (requestGeneration !== renderGeneration) return false;
    if (!resp.ok) {
      generateStatus.textContent = "Error: " + (data.error || "generation failed");
      setStatus("Training bundle generation failed.", true);
      return false;
    }
    generateStatus.textContent = `Training files are ready for ${data.model_name}.`;
    generateResult.hidden = false;
    const newWarningsText = data.warnings ? data.warnings.join(" ") : "";
    const displayWarningsText = data.warnings ? renderWarningText(data.warnings) : "";
    const alreadyShownAbove =
      newWarningsText && !renderWarnings.hidden && renderWarnings.textContent === displayWarningsText;
    const warningsAreCalibrationOnly = isCalibrationOnlyWarning(data.warnings || []);
    const warningsHtml = newWarningsText
      ? `<div class="${warningsAreCalibrationOnly ? "info-box" : "warning-box"}">${
          alreadyShownAbove
            ? "The same calibration guidance shown in Choose amps applies to these training files."
            : renderWarningText(data.warnings)
        }</div>`
      : "";
    const cabLine = data.cab_summary
      ? `<div><strong>Cabinet:</strong> ${data.cab_summary.export_mode === "embedded" ? "exact cabinet in experimental Sequential NAM" : data.cab_summary.baked ? "learned approximation in this A2" : "not included in this model"}</div>`
      : "";
    const lowLevelHtml = data.low_level_response ? renderLowLevelResponseHtml(data.low_level_response) : "";
    generateResult.innerHTML = `
      <div><strong>Ready to train:</strong> ${data.model_name}</div>
      <div>The final model will be saved as <code>${data.download_filename}</code>.</div>
      ${cabLine}
      <details><summary>Technical details</summary><div><strong>Training files:</strong> <code>${data.bundle_dir}</code></div><div><strong>Output level:</strong> ${data.safety_report.final_peak_dbfs.toFixed(1)} dBFS</div><div><strong>Calibration:</strong> ${data.calibration_summary.effective_mode}</div></details>
      ${warningsHtml}
      ${lowLevelHtml}
    `;
    setStatus("Training bundle ready.");

    lastDesignId = data.design_id;
    completedNamArtifact = null;
    completedValidationReport = null;
    invalidateModelComparison("");
    syncComparisonPanel();
    // Keep the SAME session identity across a regenerate (e.g. after
    // loading a session, tweaking a setting, and clicking "Create training
    // files" again) -- this used to always mint a fresh random id here
    // regardless of whether one was already active, silently leaving the
    // previously loaded/saved session behind as an orphaned duplicate
    // instead of updating it.
    activeSessionId = activeSessionId || sessionId();
    activeSessionName = data.model_name;
    activeSessionGenerated = true;
    try {
      await writeSession(await currentSession(data.model_name, true));
    } catch (err) {
      // Generation is still valid if its convenience session cannot be saved.
      console.warn("Could not save generated session:", err);
    }
    document.getElementById("a2-training-section").hidden = false;
    if (data.default_epoch_preset === "high_def") document.getElementById("a2-preset-high_def").checked = true;
    refreshLocalTraining();
    refreshKaggleStatus();
    return true;
  } catch (err) {
    if (requestGeneration !== renderGeneration) return false;
    generateStatus.textContent = "Request failed: " + err;
    setStatus("Training bundle generation failed.", true);
    return false;
  } finally {
    stopActivity();
    generationPending = false;
    syncTrainingControls();
  }
}

generateBtn.addEventListener("click", () => { runGenerate(); });

// --- Kaggle GPU training backend -----------------------------------------
let lastDesignId = null;
// Kept with a saved session only after training completes. The file itself
// remains app-managed on the server; this is a safe, route-based download
// reference rather than a raw filesystem path.
let completedNamArtifact = null;
let activeSessionId = null;
let activeSessionName = null;
let activeSessionGenerated = false;
let kaggleAuthenticated = false;
let kaggleJobPollTimer = null;

const kaggleStatusEl = document.getElementById("kaggle-status");
const kaggleInstallBtn = document.getElementById("btn-kaggle-install");
const kaggleConnectBtn = document.getElementById("btn-kaggle-connect");
const trainA2Btn = document.getElementById("btn-train-a2");
const kaggleProgressBox = document.getElementById("kaggle-progress-box");
const kaggleProgressState = document.getElementById("kaggle-progress-state");
const kaggleProgressMeta = document.getElementById("kaggle-progress-meta");
const kaggleProgressBarTrack = document.getElementById("kaggle-progress-bar-track");
const kaggleProgressBarFill = document.getElementById("kaggle-progress-bar-fill");
const kaggleCurrentLine = document.getElementById("kaggle-current-line");
const kaggleLogTail = document.getElementById("kaggle-log-tail");
const kaggleResultEl = document.getElementById("kaggle-result");
const kaggleRefreshBtn = document.getElementById("btn-kaggle-refresh");
const kaggleBackendRadio = document.getElementById("a2-backend-kaggle");
const localBackendRadio = document.getElementById("a2-backend-local");
const kagglePanel = document.getElementById("kaggle-panel");
const localPanel = document.getElementById("local-panel");
let kaggleAuthPollTimer = null;
let kaggleJobSubmittedAt = null;
let localTrainingActive = false;
let kaggleTrainingActive = false;
let kaggleActivityStop = null;

function trainingIsActive() {
  return localTrainingActive || kaggleTrainingActive;
}

function resetGeneratedModel(reason) {
  // Running jobs and completed artifacts own a frozen bundle. Keep their
  // identity while editing the preview, including choosing a comparison DI.
  if (!lastDesignId || trainingIsActive() || completedNamArtifact) return;
  lastDesignId = null;
  completedNamArtifact = null;
  completedValidationReport = null;
  invalidateModelComparison("");
  activeSessionId = null;
  activeSessionName = null;
  activeSessionGenerated = false;
  // Do NOT force the section hidden here: syncTrainingControls keeps it
  // visible as long as generating is still possible, so Train can just
  // regenerate a fresh bundle on click instead of the user needing to
  // notice it disappeared and re-click "Create training files" first.
  kaggleResultEl.hidden = true;
  localResultEl.hidden = true;
  if (reason) setStatus(reason);
}

function syncTrainingControls() {
  const locked = trainingIsActive();
  const canGenerate = havePair && activeRenderId && trainingInputReady;
  generateBtn.disabled = locked || generationPending || !canGenerate;
  // Train buttons generate their own bundle on click if one isn't already
  // there (see localTrainBtn/trainA2Btn) -- so the quality/backend/train
  // controls only need the same preconditions as generating, not an
  // already-completed generate step. This is what lets a user go straight
  // from rendering to clicking Train without a separate "Create training
  // files" click in between.
  document.getElementById("a2-training-section").hidden = !(canGenerate || lastDesignId);
  trainingInputFile.disabled = locked;
  modelNameInput.disabled = locked;
  document.querySelectorAll('input[name="a2-epoch-preset"], input[name="a2-backend"]').forEach((input) => { input.disabled = locked; });
}

// --- Training quality (epoch preset: draft=20 / standard=60 / high_def=120) -
function selectedEpochPreset() {
  const checked = document.querySelector('input[name="a2-epoch-preset"]:checked');
  return checked ? checked.value : "standard";
}

const localTrainingStatus = document.getElementById("local-training-status");
const localTrainingMeta = document.getElementById("local-training-meta");
const localTrainingProgressTrack = document.getElementById("local-training-progress-track");
const localTrainingProgressFill = document.getElementById("local-training-progress-fill");
const localTrainingCurrentLine = document.getElementById("local-training-current-line");
const localTrainingLog = document.getElementById("local-training-log");
const localSetupBtn = document.getElementById("btn-local-setup");
const localSetupSizeHint = document.getElementById("local-setup-size-hint");
const localTrainBtn = document.getElementById("btn-local-train");
const localTrainBlockedReasonEl = document.getElementById("local-train-blocked-reason");
const localCancelBtn = document.getElementById("btn-local-cancel");
const localResultEl = document.getElementById("local-result");
let localTrainingPoll = null;
let localTrainingActivityStop = null;
// The design a completed "training" state actually belongs to -- captured
// at the moment Train locally is clicked, since LocalTrainingManager is a
// single global slot with no design_id of its own to poll back.
let localTrainingDesignId = null;
let completedValidationReport = null;

const comparisonPanel = document.getElementById("model-comparison-panel");
const comparisonBuildBtn = document.getElementById("btn-build-comparison");
const comparisonStopBtn = document.getElementById("btn-stop-comparison");
const comparisonSwitches = document.getElementById("comparison-switches");
const comparisonStatus = document.getElementById("comparison-status");
const comparisonMetrics = document.getElementById("comparison-metrics");
const comparisonMetricsBody = document.getElementById("comparison-metrics-body");
const comparisonUnavailableNote = document.getElementById("comparison-unavailable-note");
let comparisonRequestGeneration = 0;
let comparisonPlayback = null;
let comparisonData = null;

function stopModelComparison(invalidateRequest = false) {
  if (invalidateRequest) {
    comparisonRequestGeneration += 1;
    comparisonBuildBtn.disabled = completedNamArtifact?.embeddedArtifact?.state === "validated";
  }
  if (comparisonPlayback) {
    try { comparisonPlayback.source.stop(); } catch (_) { /* already stopped */ }
    comparisonPlayback.context.close().catch(() => {});
    comparisonPlayback = null;
  }
}

function invalidateModelComparison(message = "") {
  stopModelComparison(true);
  comparisonData = null;
  if (comparisonSwitches) comparisonSwitches.hidden = true;
  if (comparisonMetrics) comparisonMetrics.hidden = true;
  if (comparisonStatus && message) comparisonStatus.textContent = message;
}

function syncComparisonPanel() {
  comparisonPanel.hidden = !(completedNamArtifact && lastDesignId);
  // The comparison only ever renders/evaluates the reusable head model
  // (see buildModelComparison's body.model_path = completedNamArtifact.toolPath,
  // which is always the head's output_nam_path, never a packaged Sequential
  // embedded-cab file). For a Sequential Embedded design that means the
  // comparison would silently compare the amp-only head against the
  // teacher while the actual deliverable is amp+cab -- misleading rather
  // than useful, so disable it with an explanation until it can evaluate
  // the real embedded package too.
  const embeddedValidated = completedNamArtifact?.embeddedArtifact?.state === "validated";
  if (comparisonBuildBtn) comparisonBuildBtn.disabled = embeddedValidated;
  if (comparisonUnavailableNote) {
    comparisonUnavailableNote.hidden = !embeddedValidated;
    if (embeddedValidated) {
      comparisonUnavailableNote.textContent = "Unavailable for this design: this comparison only evaluates the reusable head model, not the amp+cab Sequential package you'd actually export. Use the downloaded model in a NAM player to audition it instead.";
    }
  }
}

function selectComparisonSource(id) {
  if (!comparisonPlayback || !comparisonData) return;
  const selected = comparisonData.variants.find((item) => item.id === id && Number.isInteger(item.channel));
  if (!selected) return;
  const now = comparisonPlayback.context.currentTime;
  comparisonPlayback.gains.forEach((gain, index) => gain.gain.setValueAtTime(index === selected.channel ? 1 : 0, now));
  document.querySelectorAll("[data-comparison-source]").forEach((button) => {
    button.disabled = !comparisonData.variants.some((item) => item.id === button.dataset.comparisonSource && Number.isInteger(item.channel));
    button.setAttribute("aria-pressed", button.dataset.comparisonSource === id ? "true" : "false");
  });
  comparisonStatus.textContent = `Playing ${id === "teacher" ? "the frozen teacher" : id === "full" ? "Full model" : "Lite model"} at actual output level.`;
}

function comparisonMetricsHtml(data) {
  return data.variants.filter((item) => item.id !== "teacher").map((item) => {
    if (!item.metrics) return `<div><strong>${escapeHtml(item.id)}:</strong> unavailable — ${escapeHtml(item.error || "not exported")}</div>`;
    return `<div><strong>${escapeHtml(item.id)}:</strong> raw ESR ${Number(item.metrics.raw_esr).toFixed(4)} · gain-normalized ESR ${Number(item.metrics.gain_normalized_esr).toFixed(4)} · RMS difference ${Number(item.metrics.rms_difference).toFixed(5)} · peak difference ${Number(item.metrics.peak_difference).toFixed(5)}</div>`;
  }).join("");
}

async function buildModelComparison() {
  if (!completedNamArtifact || !lastDesignId) return;
  stopModelComparison(true);
  const generation = comparisonRequestGeneration;
  comparisonBuildBtn.disabled = true;
  comparisonStatus.textContent = "Rendering the frozen teacher, Full, and Lite on the selected musical performance…";
  comparisonSwitches.hidden = true;
  comparisonMetrics.hidden = true;
  const inputGain = Number(document.querySelector('input[name="comparison-level"]:checked')?.value || 0);
  const body = { design_id: lastDesignId, di_file: diSelector.value, input_gain_db: inputGain };
  if (completedNamArtifact.toolPath) body.model_path = completedNamArtifact.toolPath;
  try {
    const response = await fetch("/api/comparison", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await response.json();
    if (generation !== comparisonRequestGeneration) return;
    if (!response.ok) throw new Error(data.error || "comparison could not be built");
    const audioResponse = await fetch(data.audio_url, { cache: "no-store" });
    if (!audioResponse.ok) throw new Error("comparison audio expired before playback");
    const bytes = await audioResponse.arrayBuffer();
    if (generation !== comparisonRequestGeneration) return;
    const Context = window.AudioContext || window.webkitAudioContext;
    const context = new Context();
    const decoded = await context.decodeAudioData(bytes);
    if (generation !== comparisonRequestGeneration) { await context.close(); return; }
    const source = context.createBufferSource();
    const splitter = context.createChannelSplitter(decoded.numberOfChannels);
    const gains = Array.from({ length: decoded.numberOfChannels }, () => context.createGain());
    source.buffer = decoded;
    source.loop = true;
    source.connect(splitter);
    gains.forEach((gain, index) => { gain.gain.value = index === 0 ? 1 : 0; splitter.connect(gain, index); gain.connect(context.destination); });
    source.start(context.currentTime + 0.03);
    comparisonPlayback = { context, source, gains };
    comparisonData = data;
    comparisonMetricsBody.innerHTML = comparisonMetricsHtml(data);
    comparisonMetrics.hidden = false;
    comparisonSwitches.hidden = false;
    selectComparisonSource("teacher");
  } catch (err) {
    if (generation === comparisonRequestGeneration) comparisonStatus.textContent = "Comparison unavailable: " + err.message;
  } finally {
    if (generation === comparisonRequestGeneration) comparisonBuildBtn.disabled = false;
  }
}

comparisonBuildBtn.addEventListener("click", buildModelComparison);
comparisonStopBtn.addEventListener("click", () => { stopModelComparison(true); comparisonStatus.textContent = "Comparison stopped."; });
document.querySelectorAll("[data-comparison-source]").forEach((button) => button.addEventListener("click", () => selectComparisonSource(button.dataset.comparisonSource)));
document.querySelectorAll('input[name="comparison-level"]').forEach((input) => input.addEventListener("change", () => invalidateModelComparison("Playing level changed. Build the comparison again.")));

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = String(value ?? "");
  return node.innerHTML;
}

function validationSummaryHtml(report) {
  if (!report) return `<div class="warning-box">Validation report unavailable for this export.</div>`;
  const label = report.state === "passed" ? "Technical validation passed" : report.state === "needs_attention" ? "Validation needs attention" : "Validation unavailable";
  const checks = (report.checks || []).map((check) => `${escapeHtml(check.id).replaceAll("_", " ")}: ${escapeHtml(check.state)}`).join(" · ");
  const detailRows = (report.checks || []).map((check) => {
    const metrics = check.metrics && Object.keys(check.metrics).length
      ? `<pre class="log-tail">${escapeHtml(JSON.stringify(check.metrics, null, 2))}</pre>` : "";
    return `<div><strong>${escapeHtml(check.id).replaceAll("_", " ")} — ${escapeHtml(check.state)}</strong><br><span>${escapeHtml(check.reason || "")}</span>${metrics}</div>`;
  }).join("");
  const cabinet = report.cabinet?.note ? `<div><strong>Cabinet:</strong> ${escapeHtml(report.cabinet.note)}</div>` : "";
  return `<div class="${report.state === "passed" ? "info" : "warning-box"}"><strong>${label}.</strong> ${escapeHtml(report.summary)}<br><small>${checks}</small><details><summary>Validation metrics and reasons</summary>${detailRows}${cabinet}</details></div>`;
}

function renderLocalDownloadResult(designId, validationReport = null, downloadFilename = "model.nam", embeddedArtifact = null) {
  const downloadUrl = `/api/local_training/download?design_id=${encodeURIComponent(designId)}`;
  const namFilename = downloadFilename || "model.nam";
  const embeddedValidated = embeddedArtifact?.state === "validated";
  const embeddedFilename = namFilename.replace(/\.nam$/, "-embedded-experimental-full.nam");
  const finalDownloadUrl = embeddedValidated ? `${downloadUrl}&artifact=embedded` : downloadUrl;
  const finalFilename = embeddedValidated ? embeddedFilename : namFilename;
  completedNamArtifact = { type: "local", designId, downloadUrl: finalDownloadUrl, filename: finalFilename, embeddedArtifact };
  completedValidationReport = validationReport;
  syncComparisonPanel();
  persistActiveSession().catch((err) => console.warn("Could not update completed session:", err));
  localResultEl.hidden = false;
  // Embedded selection makes the validated Sequential package the final
  // deliverable. Keep the conventional head available only as an explicit
  // secondary download so the primary action cannot silently omit the cab.
  const headHtml = embeddedValidated
    ? `<a href="${downloadUrl}" download="${escapeHtml(namFilename)}" class="btn btn-secondary btn-block btn-download-artifact">${desktopSaveLabel("Download reusable head-only A2")}</a>`
    : "";
  const finalLabel = embeddedValidated ? "Download final Sequential (amp + embedded cab)" : `Download ${namFilename}`;
  localResultEl.innerHTML = `<a href="${finalDownloadUrl}" download="${escapeHtml(finalFilename)}" class="btn btn-primary btn-block btn-download-artifact">${desktopSaveLabel(finalLabel)}</a><div class="hint">Saved as <code>${escapeHtml(finalFilename)}</code>${embeddedValidated ? " (validated Sequential amp + embedded cab)" : ""}</div>${headHtml}${validationSummaryHtml(validationReport)}`;
}

async function refreshLocalTraining() {
  try {
    const resp = await fetch("/api/local_training/status");
    const data = await resp.json();
    const stateLabel = data.state === "ready"
      ? "Local training environment is ready."
      : data.state === "not_configured"
        ? "Set up the dedicated local training environment once."
        : data.state === "cancelled"
          ? "Local process stopped. You can set up or train again when ready."
        : `Local training: ${data.state.replace("_", " ")}.`;
    localTrainingStatus.textContent = stateLabel;
    // The Train button can be blocked for a few independent reasons; always
    // say which one, rather than leaving a disabled button unexplained.
    // Note: a missing training bundle is NOT one of them -- Train generates
    // one itself on click (see localTrainBtn's click handler) -- so this
    // only needs to check what Train genuinely cannot do anything about.
    let blockedReason = "";
    if (!data.ready) {
      blockedReason = data.state === "setting_up"
        ? "Waiting for the local environment setup to finish…"
        : data.state === "failed"
          ? "Local environment setup failed -- see the log below, then click Set up local training again."
          : "Click “Set up local training” first -- the dedicated Python environment is not ready.";
    } else if (["setting_up", "training", "cancelling"].includes(data.state)) {
      blockedReason = "A local setup/training process is currently running.";
    } else if (!lastDesignId && !(havePair && activeRenderId && trainingInputReady)) {
      blockedReason = "Render and audition an amp pair, and upload the official NAM training input, before training.";
    }
    if (localTrainBlockedReasonEl) {
      localTrainBlockedReasonEl.textContent = blockedReason;
      localTrainBlockedReasonEl.hidden = !blockedReason;
    }
    localTrainingLog.textContent = data.log_tail || "(no local training output yet)";
    if (data.elapsed_s !== null && data.elapsed_s !== undefined) {
      const minutes = Math.floor(data.elapsed_s / 60);
      const seconds = data.elapsed_s % 60;
      const outcome = data.exit_code === null || data.exit_code === undefined ? "running" : `exit ${data.exit_code}`;
      const progress = data.progress ? `Epoch ${data.progress.epoch}/${data.progress.total_epochs}` : "Preparing the first epoch…";
      localTrainingMeta.textContent = `Elapsed ${minutes}m ${seconds}s · ${progress} · ${outcome}`;
      if (data.progress && data.progress.total_epochs > 0) {
        localTrainingProgressTrack.hidden = false;
        localTrainingProgressFill.style.width = `${Math.min(100, Math.max(0, (data.progress.epoch / data.progress.total_epochs) * 100))}%`;
      } else {
        localTrainingProgressTrack.hidden = true;
        localTrainingProgressFill.style.width = "0%";
      }
    } else {
      localTrainingMeta.textContent = "";
      localTrainingProgressTrack.hidden = true;
      localTrainingProgressFill.style.width = "0%";
    }
    if (data.latest_line && ["setting_up", "training", "cancelling"].includes(data.state)) {
      localTrainingCurrentLine.hidden = false;
      localTrainingCurrentLine.textContent = data.latest_line;
    } else {
      localTrainingCurrentLine.hidden = true;
      localTrainingCurrentLine.textContent = "";
    }
    localTrainingActive = ["setting_up", "training", "cancelling"].includes(data.state);
    if (localTrainingActive) {
      const phase = data.state === "setting_up" ? "Setting up local training" : data.state === "cancelling" ? "Stopping local training" : "Training model";
      const detail = data.progress ? ` — epoch ${data.progress.epoch}/${data.progress.total_epochs}` : " — preparing the trainer";
      if (!localTrainingActivityStop) localTrainingActivityStop = beginActivity(`${phase}${detail}…`);
      else localTrainingActivityStop.update(`${phase}${detail}…`);
    } else if (localTrainingActivityStop) {
      localTrainingActivityStop();
      localTrainingActivityStop = null;
    }
    // Once the dedicated environment is ready, there's nothing left to set
    // up -- showing "Set up" next to a working environment invites
    // re-running pip install for no reason and looks like the previous
    // setup didn't take.
    localSetupBtn.hidden = Boolean(data.ready);
    localSetupSizeHint.hidden = Boolean(data.ready);
    localSetupBtn.disabled = localTrainingActive;
    localTrainBtn.disabled = !data.ready || localTrainingActive || !(lastDesignId || (havePair && activeRenderId && trainingInputReady));
    localCancelBtn.hidden = !localTrainingActive;
    localCancelBtn.disabled = data.state === "cancelling";
    syncTrainingControls();
    if (localTrainingActive) {
      if (!localTrainingPoll) localTrainingPoll = setInterval(refreshLocalTraining, 1000);
    } else if (localTrainingPoll) {
      clearInterval(localTrainingPoll); localTrainingPoll = null;
    }
    if (data.state === "complete" && data.exit_code === 0 && localTrainingDesignId) {
      renderLocalDownloadResult(localTrainingDesignId, data.validation_report, data.download_filename, data.embedded_artifact);
    } else if (data.state !== "complete") {
      localResultEl.hidden = true;
    }
  } catch (err) {
    localTrainingStatus.textContent = "Could not check local training: " + err;
    localSetupBtn.disabled = false;
    localTrainBtn.disabled = false;
  }
}

localSetupBtn.addEventListener("click", async () => {
  localSetupBtn.disabled = true;
  localTrainingStatus.textContent = "Creating the dedicated environment and installing training packages…";
  if (!localTrainingActivityStop) localTrainingActivityStop = beginActivity("Setting up local training — installing the dedicated environment…");
  let setupError = "";
  try {
    const resp = await fetch("/api/local_training/setup", { method: "POST" });
    const data = await resp.json();
    if (!resp.ok) setupError = data.error || "Local setup could not start.";
  } catch (err) {
    setupError = "Local setup could not start: " + err.message;
    localSetupBtn.disabled = false;
  } finally {
    await refreshLocalTraining();
    // refreshLocalTraining reports the normal "not configured" state. Keep
    // the startup failure visible instead, especially when a frozen build
    // needs the user to install a real Python interpreter.
    if (setupError) localTrainingStatus.textContent = setupError;
  }
});

localTrainBtn.addEventListener("click", async () => {
  if (!lastDesignId) {
    localTrainBtn.disabled = true;
    const ok = await runGenerate();
    if (!ok || !lastDesignId) { localTrainBtn.disabled = !lastDesignId; return; }
  }
  localTrainBtn.disabled = true;
  localResultEl.hidden = true;
  localTrainingDesignId = lastDesignId;
  if (!localTrainingActivityStop) localTrainingActivityStop = beginActivity("Training model — preparing the first epoch…");
  try {
    const resp = await fetch("/api/local_training/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ design_id: lastDesignId, epoch_preset: selectedEpochPreset() }),
    });
    const data = await resp.json();
    if (!resp.ok) localTrainingStatus.textContent = data.error || "Local training could not start.";
  } catch (err) {
    localTrainingStatus.textContent = "Local training could not start: " + err.message;
    localTrainBtn.disabled = false;
  } finally {
    await refreshLocalTraining();
  }
});

localCancelBtn.addEventListener("click", async () => {
  localCancelBtn.disabled = true;
  localTrainingStatus.textContent = "Stopping the local process…";
  if (localTrainingActivityStop) localTrainingActivityStop.update("Stopping local training…");
  try {
    const resp = await fetch("/api/local_training/cancel", { method: "POST" });
    const data = await resp.json();
    if (!resp.ok) localTrainingStatus.textContent = data.error || "Could not stop the local process.";
  } catch (err) {
    localTrainingStatus.textContent = "Could not stop the local process: " + err;
  }
  await refreshLocalTraining();
});

// A completed job's .nam used to be shown only as a bare server-side path
// (e.g. work/a2/<design>/kaggle/<job>/output/a2_output/export/hybrid_a2.nam)
// -- unusable for a user who isn't on the machine running Flask. A direct
// download link (backed by GET /api/kaggle/jobs/<id>/download) is the actual
// easy path to the file; the full path is kept below it for reference.
function renderKaggleDownloadResult(designId, jobId, data) {
  kaggleResultEl.hidden = false;
  const downloadUrl = `/api/kaggle/jobs/${encodeURIComponent(jobId)}/download?design_id=${encodeURIComponent(designId)}`;
  // Use the EXACT filename the server will actually save the download as
  // (`data.download_filename`, computed identically in app.py's
  // _job_dict_for_client) -- NOT the internal `hybrid_a2.nam` export
  // basename baked inside the Kaggle kernel (data.output_nam_path's own
  // basename), which previously made the button's label lie about what
  // file the browser would actually save.
  const namFilename = data.download_filename || "model.nam";
  const embeddedValidated = data.embedded_artifact?.state === "validated";
  const embeddedFilename = namFilename.replace(/\.nam$/, "-embedded-experimental-full.nam");
  const finalDownloadUrl = embeddedValidated ? `${downloadUrl}&artifact=embedded` : downloadUrl;
  const finalFilename = embeddedValidated ? embeddedFilename : namFilename;
  completedNamArtifact = { type: "kaggle", designId, jobId, downloadUrl: finalDownloadUrl, filename: finalFilename, toolPath: data.output_nam_path || null, embeddedArtifact: data.embedded_artifact || null };
  completedValidationReport = data.local_validation?.validation_report || null;
  syncComparisonPanel();
  persistActiveSession().catch((err) => console.warn("Could not update completed session:", err));
  // Embedded selection makes the validated Sequential package the final
  // deliverable. Keep the conventional head available only as an explicit
  // secondary download so the primary action cannot silently omit the cab.
  const headHtml = embeddedValidated
    ? `<a href="${downloadUrl}" download="${namFilename}" class="btn btn-secondary btn-block">${desktopSaveLabel("Download reusable head-only A2")}</a>`
    : "";
  const finalLabel = embeddedValidated ? "Download final Sequential (amp + embedded cab)" : `Download ${namFilename}`;
  kaggleResultEl.innerHTML = `
    <a href="${finalDownloadUrl}" download="${finalFilename}" class="btn btn-primary btn-block">${desktopSaveLabel(finalLabel)}</a>
    <div class="hint" title="${data.output_nam_path || ""}">Final artifact: <code>${finalFilename}</code>${embeddedValidated ? " (validated Sequential amp + embedded cab)" : ` Full path: <code>${data.output_nam_path || "(unknown)"}</code>`}</div>
    <div><strong>SHA-256:</strong> <code>${data.output_nam_sha256 || ""}</code></div>
    ${headHtml}
    ${validationSummaryHtml(completedValidationReport)}
  `;
}

// State label, elapsed-since-submit, and a progress bar/log tail when
// available -- a bare repeating "running" string with no other signal made
// it look stuck even while training was progressing normally.
const KAGGLE_ACTIVE_STATES = new Set([
  "preparing", "uploading", "uploading_dataset", "verifying_dataset",
  "creating_kernel", "verifying_kernel", "queued", "running",
  "downloading", "validating", "cancelling"
]);

const KAGGLE_STATE_LABELS = {
  preparing: "Preparing cloud training…",
  uploading: "Uploading training bundle…",
  uploading_dataset: "Uploading training bundle…",
  verifying_dataset: "Checking the cloud training bundle…",
  creating_kernel: "Starting the cloud training job…",
  verifying_kernel: "Checking the cloud training job…",
  queued: "Waiting for a cloud GPU…",
  running: "Training model on the cloud GPU…",
  downloading: "Downloading the trained model…",
  validating: "Validating the trained model…",
  complete: "Cloud training complete.",
  failed: "Cloud training failed."
};

function kaggleStateLabel(state) {
  return KAGGLE_STATE_LABELS[state] || (state ? `Cloud training: ${String(state).replaceAll("_", " ")}…` : "Checking cloud training…");
}

function renderKaggleProgress(stateLabel, data) {
  kaggleProgressBox.hidden = false;
  kaggleProgressState.textContent = stateLabel;

  const metaParts = [];
  if (kaggleJobSubmittedAt) {
    const elapsedS = Math.round((Date.now() - kaggleJobSubmittedAt) / 1000);
    const mins = Math.floor(elapsedS / 60);
    const secs = elapsedS % 60;
    metaParts.push(`elapsed ${mins}m ${secs}s`);
  }
  metaParts.push(`last checked ${new Date().toLocaleTimeString()}`);

  if (data && data.progress) {
    const { epoch, total_epochs } = data.progress;
    metaParts.unshift(`Epoch ${epoch}/${total_epochs}`);
    kaggleProgressBarTrack.hidden = false;
    kaggleProgressBarFill.style.width = `${total_epochs > 0 ? Math.min(100, (epoch / total_epochs) * 100) : 0}%`;
  } else {
    kaggleProgressBarTrack.hidden = true;
    kaggleProgressBarFill.style.width = "0%";
  }
  kaggleProgressMeta.textContent = metaParts.join(" — ");

  if (data && typeof data.log_tail === "string") {
    kaggleLogTail.textContent = data.log_tail.trim() || "(no log output yet)";
    const latestLine = data.log_tail.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).at(-1);
    if (latestLine) {
      kaggleCurrentLine.hidden = false;
      kaggleCurrentLine.textContent = latestLine;
    }
  } else if (!data) {
    kaggleCurrentLine.hidden = true;
    kaggleCurrentLine.textContent = "";
  }

  const state = data?.state;
  if (state && KAGGLE_ACTIVE_STATES.has(state)) {
    const detail = data.progress ? ` — epoch ${data.progress.epoch}/${data.progress.total_epochs}` : "";
    if (!kaggleActivityStop) kaggleActivityStop = beginActivity(`${stateLabel}${detail}`);
    else kaggleActivityStop.update(`${stateLabel}${detail}`);
  } else if (state && kaggleActivityStop) {
    kaggleActivityStop();
    kaggleActivityStop = null;
  }
}

function updateBackendPanels() {
  const useKaggle = kaggleBackendRadio.checked;
  kagglePanel.hidden = !useKaggle;
  localPanel.hidden = useKaggle;
}
kaggleBackendRadio.addEventListener("change", updateBackendPanels);
localBackendRadio.addEventListener("change", updateBackendPanels);
localBackendRadio.addEventListener("change", refreshLocalTraining);

function formatGpuQuota(raw) {
  // `kaggle quota`'s raw CLI table output, best-effort extraction of just
  // the GPU row -- never fatal if the format doesn't match (older/newer
  // CLI versions), just falls back to something readable.
  if (!raw) return "available";
  const match = raw.match(/GPU\s+([\d.]+)h\s+([\d.]+)h\s+([\d.]+)h/);
  if (!match) return raw.split("\n")[0].trim() || "available";
  return `${match[2]}h remaining of ${match[3]}h`;
}

async function refreshKaggleStatus() {
  try {
    const resp = await fetch("/api/kaggle/status" + (lastDesignId ? `?design_id=${encodeURIComponent(lastDesignId)}` : ""));
    const data = await resp.json();
    if (!data.cli_installed) {
      kaggleStatusEl.textContent = isTauriDesktop
        ? "Kaggle CLI not found. Install the standalone Kaggle CLI, ensure its executable is on PATH, then refresh this check."
        : "Kaggle CLI not installed. Run: pip install kaggle";
      kaggleConnectBtn.hidden = true;
      kaggleInstallBtn.hidden = isTauriDesktop;
      kaggleInstallBtn.disabled = data.install_state?.state === "running";
      if (data.install_state?.state === "running") kaggleStatusEl.textContent = "Installing Kaggle CLI…";
      if (data.install_state?.state === "failed") kaggleStatusEl.textContent += ` Install failed: ${data.install_state.error}`;
      kaggleAuthenticated = false;
      trainA2Btn.disabled = true;
      return;
    }
    kaggleInstallBtn.hidden = true;
    if (!data.authenticated) {
      kaggleStatusEl.textContent = `Kaggle CLI ${data.cli_version || ""} installed${data.cli_path ? ` at ${data.cli_path}` : ""}, not connected.`;
      kaggleConnectBtn.hidden = false;
      kaggleAuthenticated = false;
      trainA2Btn.disabled = true;
      return;
    }
    kaggleAuthenticated = true;
    kaggleConnectBtn.hidden = true;
    const quota = data.quota_available ? formatGpuQuota(data.quota_raw) : "unavailable";
    kaggleStatusEl.textContent = `Connected ✓  CLI ${data.cli_version || "?"}  GPU: NVIDIA T4  Quota: ${quota}`;

    const activeJob = data.job && data.job.state && KAGGLE_ACTIVE_STATES.has(data.job.state);
    kaggleTrainingActive = !!activeJob;
    syncTrainingControls();
    trainA2Btn.disabled = !!activeJob;
    if (activeJob) {
      pollKaggleJob(data.job.design_id, data.job.job_id);
    } else if (data.job && data.job.state === "complete") {
      // Re-show the download link for a job that already finished before
      // this page load (e.g. after a refresh) -- otherwise the only way to
      // get the .nam back would be to re-run training.
      renderKaggleDownloadResult(data.job.design_id, data.job.job_id, data.job);
    }
    return true;
  } catch (err) {
    kaggleStatusEl.textContent = "Could not check Kaggle status: " + err;
    return false;
  }
}
refreshKaggleStatus();

kaggleRefreshBtn.addEventListener("click", () => refreshKaggleStatus());

kaggleInstallBtn.addEventListener("click", async () => {
  kaggleInstallBtn.disabled = true;
  const stopActivity = beginActivity("Installing the Kaggle CLI…");
  kaggleStatusEl.textContent = "Installing Kaggle CLI into this app environment…";
  try {
    const resp = await fetch("/api/kaggle/install", { method: "POST" });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "installation could not start");
    const timer = setInterval(async () => {
      const statusResp = await fetch("/api/kaggle/status");
      const status = await statusResp.json();
      await refreshKaggleStatus();
      if (status.cli_installed || status.install_state?.state === "failed") {
        clearInterval(timer);
        kaggleInstallBtn.disabled = false;
        stopActivity();
      }
    }, 2000);
  } catch (err) {
    kaggleStatusEl.textContent = "Could not install Kaggle CLI: " + err;
    kaggleInstallBtn.disabled = false;
    stopActivity();
  }
});

kaggleConnectBtn.addEventListener("click", async () => {
  kaggleConnectBtn.disabled = true;
  const stopActivity = beginActivity("Connecting to Kaggle…");
  kaggleStatusEl.textContent = "Starting Kaggle authentication...";
  try {
    const resp = await fetch("/api/kaggle/auth/start", { method: "POST" });
    const data = await resp.json();
    if (!resp.ok) {
      kaggleStatusEl.textContent = "Error: " + (data.error || "could not start Kaggle auth");
      kaggleConnectBtn.disabled = false;
      stopActivity();
      return;
    }
    kaggleStatusEl.textContent = data.started
      ? "A Kaggle login flow was started. Complete it in your browser -- this updates automatically once connected."
      : `Run this yourself -- this updates automatically once connected: ${data.command}`;
  } catch (err) {
    kaggleStatusEl.textContent = "Could not start Kaggle auth: " + err;
    kaggleConnectBtn.disabled = false;
    stopActivity();
    return;
  }
  pollKaggleAuth(stopActivity);
});

// After starting `kaggle auth login` we can't know when the user finishes the
// browser OAuth flow, so poll status for a while rather than requiring a
// manual page refresh -- this is exactly the flow the user found confusing
// (Connect Kaggle appearing to do nothing until a full reload).
function pollKaggleAuth(stopActivity) {
  if (kaggleAuthPollTimer) clearInterval(kaggleAuthPollTimer);
  let attempts = 0;
  const maxAttempts = 100; // ~5 minutes at 3s
  kaggleAuthPollTimer = setInterval(async () => {
    attempts += 1;
    await refreshKaggleStatus();
    if (kaggleAuthenticated || attempts >= maxAttempts) {
      clearInterval(kaggleAuthPollTimer);
      kaggleAuthPollTimer = null;
      kaggleConnectBtn.disabled = false;
      stopActivity();
      if (!kaggleAuthenticated) {
        kaggleStatusEl.textContent += " Still not connected -- click Connect Kaggle again once you've finished logging in, or Refresh.";
      }
    }
  }, 3000);
}

trainA2Btn.addEventListener("click", async () => {
  if (!lastDesignId) {
    trainA2Btn.disabled = true;
    const ok = await runGenerate();
    if (!ok || !lastDesignId) { trainA2Btn.disabled = !lastDesignId; return; }
  }
  trainA2Btn.disabled = true;
  kaggleTrainingActive = true;
  syncTrainingControls();
  kaggleJobSubmittedAt = Date.now();
  renderKaggleProgress("Uploading training bundle…", { state: "uploading" });
  kaggleResultEl.hidden = true;
  try {
    const resp = await fetch("/api/kaggle/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ design_id: lastDesignId, epoch_preset: selectedEpochPreset() }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      kaggleTrainingActive = false;
      syncTrainingControls();
      renderKaggleProgress("Cloud training could not start: " + (data.error || "training request failed"), { state: "failed" });
      trainA2Btn.disabled = false;
      return;
    }
    pollKaggleJob(lastDesignId, data.job_id);
  } catch (err) {
    kaggleTrainingActive = false;
    syncTrainingControls();
    renderKaggleProgress("Cloud training request failed: " + err, { state: "failed" });
    trainA2Btn.disabled = false;
  }
});

const KAGGLE_JOB_POLL_MS = 10000;
let kaggleLastJobData = null;
let kaggleTickTimer = null;

function pollKaggleJob(designId, jobId) {
  if (kaggleJobPollTimer) clearInterval(kaggleJobPollTimer);
  if (kaggleTickTimer) clearInterval(kaggleTickTimer);
  trainA2Btn.disabled = true;
  if (!kaggleJobSubmittedAt) kaggleJobSubmittedAt = Date.now();
  renderKaggleProgress("Checking cloud training…", { state: "preparing" });

  // A 1s local ticker keeps "elapsed" visibly moving between the slower
  // network polls below -- reassurance that the page itself hasn't frozen,
  // independent of whether Kaggle actually has anything new to report.
  kaggleTickTimer = setInterval(() => {
    if (kaggleLastJobData) renderKaggleProgress(kaggleStateLabel(kaggleLastJobData.state), kaggleLastJobData);
  }, 1000);

  const poll = async () => {
    try {
      const resp = await fetch(`/api/kaggle/jobs/${encodeURIComponent(jobId)}?design_id=${encodeURIComponent(designId)}`);
      const data = await resp.json();
      if (!resp.ok) {
        renderKaggleProgress("Could not check cloud training: " + (data.error || "unknown error"), { state: "failed" });
        return;
      }
      kaggleLastJobData = data;
      kaggleTrainingActive = !["complete", "failed"].includes(data.state);
      syncTrainingControls();
      renderKaggleProgress(kaggleStateLabel(data.state), data);

      if (data.state === "complete") {
        clearInterval(kaggleJobPollTimer);
        clearInterval(kaggleTickTimer);
        trainA2Btn.disabled = false;
        kaggleTrainingActive = false;
        syncTrainingControls();
        renderKaggleDownloadResult(designId, jobId, data);
        setStatus("Kaggle A2 training complete.");
      } else if (data.state === "failed") {
        clearInterval(kaggleJobPollTimer);
        clearInterval(kaggleTickTimer);
        trainA2Btn.disabled = false;
        kaggleTrainingActive = false;
        syncTrainingControls();
        renderKaggleProgress("Cloud training failed: " + (data.error || "unknown error"), data);
        if (data.output_available) renderKaggleDownloadResult(designId, jobId, data);
        setStatus("Kaggle A2 training failed.", true);
      }
    } catch (err) {
      renderKaggleProgress("Could not refresh cloud training: " + err, kaggleLastJobData);
    }
  };
  poll();
  kaggleJobPollTimer = setInterval(poll, KAGGLE_JOB_POLL_MS);
}

// ---- Sessions -------------------------------------------------------------
// File-backed library of named control snapshots. Amp/cab files are NOT
// re-uploaded -- a session stores the app-managed paths already resolved by
// /api/nam/upload and /api/cab/upload. Loading never renders automatically,
// so a stale or missing file is reported by the normal Render Amps flow.

// applySessionSettings appends " (restored)" to these labels purely for
// display. collectSessionSettings used to read that same mutated DOM text
// straight back as the "clean" label -- so load-then-save (without
// re-uploading the file) baked " (restored)" into the stored session, and
// each further load/save cycle appended ANOTHER one on top of whatever was
// already there. Stripping it on both read and write means an old session
// that already has several stacked up self-heals the next time it's saved.
function stripRestoredSuffix(label) {
  return label.replace(/(\s*\(restored\))+$/, "");
}

function collectSessionSettings() {
  return {
    mode: currentMode,
    ampA: { path: ampServerPaths.a, label: stripRestoredSuffix(document.getElementById("amp-a-info").textContent) },
    ampB: { path: ampServerPaths.b, label: stripRestoredSuffix(document.getElementById("amp-b-info").textContent) },
    diFile: diSelector.value,
    instrument: instrumentSelect.value,
    inputProfileId: profileSelect.value,
    customGainDb: customGainSlider.value,
    calibrationMode: calibrationModeSelect.value,
    referenceDbu: referenceDbuInput.value,
    testGainDb: testGainSlider.value,
    ampAInputGainDb: ampAInputGainSlider.value,
    ampBInputGainDb: ampBInputGainSlider.value,
    crossover: crossoverSlider.value,
    transition: transitionSlider.value,
    mix: mixSlider.value,
    character: Object.fromEntries(
      characterSliders.map((name) => [name, document.getElementById(`${name}-slider`).value])
    ),
    driveMorphEnabled: document.getElementById("drive-morph-enabled").checked,
    autoLevelMatch: document.getElementById("auto-level-match").checked,
    ampBTrim: ampBTrimSlider.value,
    cab: {
      path: cabServerPath,
      label: stripRestoredSuffix(cabInfoEl.textContent),
      previewEnabled: cabPreviewEnabled.checked,
      exportMode: cabExportMode.value,
    },
    outputGainAuto: outputGainAutoCheckbox.checked,
    outputGainManualDb: outputGainManualSlider.value,
    modelName: document.getElementById("model-name").value,
  };
}

function applySessionSettings(s) {
  instrumentExplicitlySelected = true;
  ampServerPaths.a = s.ampA.path;
  ampServerPaths.b = s.ampB.path;
  document.getElementById("amp-a-info").textContent = s.ampA.path ? `${stripRestoredSuffix(s.ampA.label)} (restored)` : "";
  document.getElementById("amp-b-info").textContent = s.ampB.path ? `${stripRestoredSuffix(s.ampB.label)} (restored)` : "";

  diSelector.value = s.diFile;
  instrumentSelect.value = s.instrument;
  populateProfileSelect();
  profileSelect.value = s.inputProfileId;
  updateProfileDescription();
  customGainSlider.value = s.customGainDb;
  customGainValue.textContent = `${fmtSigned(customGainSlider.value)} dB`;
  calibrationModeSelect.value = s.calibrationMode;
  referenceDbuInput.value = s.referenceDbu;
  testGainSlider.value = s.testGainDb;
  testGainValue.textContent = `${fmtSigned(testGainSlider.value)} dB`;
  ampAInputGainSlider.value = s.ampAInputGainDb;
  ampAInputGainValue.textContent = `${fmtSigned(ampAInputGainSlider.value)} dB`;
  ampBInputGainSlider.value = s.ampBInputGainDb;
  ampBInputGainValue.textContent = `${fmtSigned(ampBInputGainSlider.value)} dB`;

  crossoverSlider.value = s.crossover;
  crossoverValue.textContent = `${parseFloat(crossoverSlider.value).toFixed(1)} dBFS`;
  syncCrossoverKnobFromDb();
  // "Reset to ..." after a suggested-crossover auto-set (see
  // applyRenderResult) should go back to what THIS preset set the crossover
  // to, not silently fall back to the app's hardcoded built-in default.
  crossoverBaseline = { value: s.crossover, label: s.modelName || "loaded settings" };
  transitionSlider.value = s.transition;
  transitionValue.textContent = `${transitionSlider.value} dB`;
  updateTransitionAroundSwitchNote();
  syncPresetButtonStates();
  mixSlider.value = s.mix;
  updateMixValueLabel();

  characterSliders.forEach((name) => {
    const slider = document.getElementById(`${name}-slider`);
    slider.value = s.character[name];
    document.getElementById(`${name}-value`).textContent = `${slider.value}% B`;
  });
  document.getElementById("drive-morph-enabled").checked = s.driveMorphEnabled;

  document.getElementById("auto-level-match").checked = s.autoLevelMatch;
  ampBTrimSlider.value = s.ampBTrim;
  ampBTrimValue.textContent = `${fmtSigned(ampBTrimSlider.value)} dB`;

  cabServerPath = s.cab.path;
  cabInfoEl.textContent = s.cab.path ? `${stripRestoredSuffix(s.cab.label)} (restored)` : "";
  cabPreviewEnabled.disabled = !s.cab.path;
  cabExportMode.disabled = !s.cab.path;
  cabPreviewEnabled.checked = s.cab.previewEnabled;
  cabExportMode.value = s.cab.exportMode || (s.cab.baked ? "learned" : "none");
  applyExperimentalArchitecturesVisibility();
  updateCabStatus();

  outputGainAutoCheckbox.checked = s.outputGainAuto;
  outputGainManualSlider.disabled = outputGainAutoCheckbox.checked;
  outputGainManualSlider.value = s.outputGainManualDb;
  outputGainManualValue.textContent = `${fmtSigned(outputGainManualSlider.value)} dB`;

  document.getElementById("model-name").value = s.modelName || "";

  const tab = document.getElementById(`tab-${s.mode}`);
  if (tab) {
    currentMode = s.mode;
    modeTabs.forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-pressed", t === tab ? "true" : "false");
    });
    applyModeVisibility();
  }

  // A restored pair still needs a real re-render before anything is
  // trustworthy (the server file may be gone, or NAM inference simply
  // hasn't run this session).  Use the normal invalidation path so playback,
  // cached audition audio, and the server capability all become unavailable.
  markProfileStale("Settings restored");
  updateCoverage();
}

const sessionSettingsStatus = document.getElementById("session-settings-status");
const sessionsPanel = document.getElementById("sessions-panel");
const sessionList = document.getElementById("session-list");
const sessionManagerStatus = document.getElementById("session-manager-status");
const sessionNameInput = document.getElementById("session-name");

async function readSessions() {
  const response = await fetch("/api/sessions");
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "could not list sessions");
  return data;
}

async function writeSession(session) {
  const response = await fetch("/api/sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(session) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "could not save session");
  return data;
}

function sessionId() {
  return globalThis.crypto?.randomUUID?.() || `session-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function bytesToBase64(bytes) {
  let text = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    text += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(text);
}

async function portableArtifact() {
  if (!completedNamArtifact) return null;
  if (completedNamArtifact.nam_base64) return completedNamArtifact;
  if (!completedNamArtifact.downloadUrl) throw new Error("the completed NAM is no longer available to include in this session");
  const response = await fetch(completedNamArtifact.downloadUrl);
  if (!response.ok) throw new Error("could not include the completed NAM in this session");
  const bytes = new Uint8Array(await response.arrayBuffer());
  return { filename: completedNamArtifact.filename || "model.nam", nam_base64: bytesToBase64(bytes) };
}

async function currentSession(name, generated = activeSessionGenerated) {
  return {
    type: "nam-mixer-session", version: 1,
    id: activeSessionId || sessionId(), name, savedAt: new Date().toISOString(),
    settings: collectSessionSettings(), designId: lastDesignId,
    artifact: await portableArtifact(), validationReport: completedValidationReport, ...(generated ? { generated: true } : {}),
  };
}

async function persistActiveSession() {
  if (!activeSessionId) return;
  const name = document.getElementById("model-name").value.trim() || activeSessionName || "Generated session";
  activeSessionName = name;
  await writeSession(await currentSession(name));
}

function sessionSummary(session) {
  const settings = session.settings || {};
  const amps = [settings.ampA?.label, settings.ampB?.label].filter(Boolean).join(" / ") || "No amps selected";
  const mode = { hybrid: "Dynamic Hybrid", blend: "Parallel Blend", character: "Character Blend" }[settings.mode] || "Unknown mode";
  return { amps, mode, di: settings.diFile || "No test performance", artifact: session.artifact, validation: session.validationReport || null };
}

function downloadSessionNam(artifact) {
  triggerFileDownload(artifact.downloadUrl, artifact.filename || "model.nam");
}

function downloadSessionJson(session) {
  const filename = `${(session.name || "nam-mixer-session").replace(/[^a-z0-9_-]+/gi, "-")}.nam-mixer-session.json`;
  triggerFileDownload(`/api/sessions/${encodeURIComponent(session.id)}/download`, filename);
}

function importedSession(raw, filename) {
  if (!raw || typeof raw !== "object") throw new Error("file does not contain a session");
  const session = { ...raw };
  if (session.type !== "nam-mixer-session" || session.version !== 1) throw new Error("unsupported session file");
  if (!session.settings || typeof session.settings !== "object") throw new Error("session has no settings");
  // Imported sessions are a new server-side record; keep their content but
  // assign a fresh id so importing cannot overwrite an existing session.
  session.id = sessionId();
  session.name = String(session.name || filename.replace(/\.json$/i, "") || "Imported session").slice(0, 80);
  session.savedAt = typeof session.savedAt === "string" ? session.savedAt : new Date().toISOString();
  session.designId = session.designId || null;
  session.artifact = session.artifact || null;
  session.generated = false;
  return session;
}

// Deleting a session whose training is still active used to just fail
// with a raw server error ("cannot delete a session while its Kaggle
// training job is active") and leave the user stuck. This retries with
// `cancel_active_jobs=1` -- but only after a SEPARATE, explicit warning
// naming exactly what that cascades into (stopping the Kaggle
// kernel/dataset or the local training process) -- rather than silently
// cancelling active training as a side effect of the first delete
// confirmation, which said nothing about training jobs at all.
async function deleteSessionCascading(sessionId, name) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  if (response.ok) return;
  const data = await response.json().catch(() => ({}));
  if (data.active_kaggle_job || data.active_local_job) {
    const jobKind = data.active_kaggle_job ? "Kaggle cloud training job" : "local training process";
    const proceed = await desktopConfirm(
      `“${name}” has an active ${jobKind}. Deleting this session will also cancel that ${jobKind} now. Continue?`,
      "Cancel active training?",
    );
    if (!proceed) {
      const err = new Error(`Delete cancelled -- "${name}" and its ${jobKind} were left as they were.`);
      err.userCancelled = true;
      throw err;
    }
    const retry = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}?cancel_active_jobs=1`, { method: "DELETE" });
    if (!retry.ok) throw new Error((await retry.json().catch(() => ({}))).error || "could not delete session");
    return;
  }
  throw new Error(data.error || "could not delete session");
}

async function renderSessions() {
  const sessions = await readSessions();
  sessionList.replaceChildren();
  if (!sessions.length) {
    const empty = document.createElement("p");
    empty.className = "info";
    empty.textContent = "No saved sessions yet.";
    sessionList.append(empty);
    return;
  }
  for (const session of sessions) {
    const summary = sessionSummary(session);
    const card = document.createElement("article"); card.className = "session-card";
    const heading = document.createElement("div"); heading.className = "session-card-heading";
    const name = document.createElement("strong"); name.textContent = session.name || "Untitled session";
    const saved = document.createElement("time");
    const date = new Date(session.savedAt);
    saved.dateTime = Number.isNaN(date.valueOf()) ? "" : date.toISOString();
    saved.textContent = Number.isNaN(date.valueOf()) ? "Unknown save time" : date.toLocaleString();
    heading.append(name, saved);
    const actions = document.createElement("div"); actions.className = "session-card-actions";
    const details = document.createElement("div"); details.className = "session-details"; details.hidden = true;
    const settings = session.settings || {};
    const shape = settings.mode === "blend"
      ? `Mix: ${settings.mix ?? "—"}% Amp B`
      : settings.mode === "character"
        ? `Tone: ${settings.character?.tone ?? "—"}% Amp B; Feel: ${settings.character?.feel ?? "—"}% Amp B; Drive: ${settings.character?.drive ?? "—"}% Amp B`
        : `Changeover: ${settings.crossover ?? "—"} dBFS; Transition: ${settings.transition ?? "—"} dB`;
    const validationText = summary.validation
      ? ` · Validation: ${summary.validation.state || "unavailable"} — ${summary.validation.summary || "no summary"}`
      : summary.artifact ? " · Validation report unavailable" : "";
    details.textContent = `Mode: ${summary.mode} · Amps: ${summary.amps} · Test performance: ${summary.di} · Input profile: ${settings.inputProfileId || "—"} · ${shape} · Level match: ${settings.autoLevelMatch ? "on" : "off"} · Cabinet: ${settings.cab?.path ? "selected" : "off"}${summary.artifact ? ` · NAM: ${summary.artifact.filename || "available"}` : " · No completed NAM recorded"}${validationText}`;
    const detailButton = document.createElement("button"); detailButton.type = "button"; detailButton.className = "btn btn-secondary btn-small"; detailButton.textContent = "Details";
    detailButton.addEventListener("click", () => { details.hidden = !details.hidden; detailButton.textContent = details.hidden ? "Details" : "Hide details"; });
    const loadButton = document.createElement("button"); loadButton.type = "button"; loadButton.className = "btn btn-primary btn-small"; loadButton.textContent = "Load";
    loadButton.addEventListener("click", () => {
      try {
        applySessionSettings(session.settings);
        lastDesignId = session.designId || null;
        completedNamArtifact = session.artifact || null;
        completedValidationReport = session.validationReport || null;
        invalidateModelComparison("");
        syncComparisonPanel();
        if (completedNamArtifact && lastDesignId) document.getElementById("a2-training-section").hidden = false;
        activeSessionId = session.id;
        activeSessionName = session.name;
        activeSessionGenerated = session.generated === true;
        sessionSettingsStatus.textContent = `Loaded ${session.name || "session"}`;
        setSessionsOpen(false);
        setWorkflowStage("configure");
        setStatus(`Loaded ${session.name || "session"}.`);
      } catch (err) { sessionManagerStatus.textContent = "Could not load this session: " + err; }
    });
    const deleteButton = document.createElement("button"); deleteButton.type = "button"; deleteButton.className = "btn btn-secondary btn-small"; deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", async () => {
      const name = session.name || "Untitled session";
      const deleteMessage = session.generated
        ? `Delete “${name}” and its entire training bundle? This removes all files under work/a2/${session.designId}.`
        : `Delete “${name}”? This removes its saved session file.`;
      if (!(await desktopConfirm(deleteMessage, "Delete session"))) return;
      try {
        await deleteSessionCascading(session.id, name);
        sessionManagerStatus.textContent = "Session deleted.";
        await renderSessions();
      } catch (err) {
        sessionManagerStatus.textContent = err.userCancelled ? err.message : "Delete failed: " + err.message;
      }
    });
    const exportButton = document.createElement("button"); exportButton.type = "button"; exportButton.className = "btn btn-secondary btn-small"; exportButton.textContent = "Export JSON";
    exportButton.addEventListener("click", () => downloadSessionJson(session));
    actions.append(detailButton, loadButton, exportButton);
    if (summary.artifact?.downloadUrl) {
      const downloadButton = document.createElement("button"); downloadButton.type = "button"; downloadButton.className = "btn btn-secondary btn-small"; downloadButton.textContent = desktopSaveLabel("Download NAM");
      downloadButton.addEventListener("click", () => downloadSessionNam(summary.artifact));
      actions.append(downloadButton);
    }
    if (summary.artifact?.toolPath) {
      const toolsButton = document.createElement("button"); toolsButton.type = "button"; toolsButton.className = "btn btn-secondary btn-small"; toolsButton.textContent = "Open in NAM Tools";
      toolsButton.addEventListener("click", async () => {
        setSessionsOpen(false); setToolsOpen(true);
        await setToolNam({ path: summary.artifact.toolPath }, summary.artifact.filename || "Session NAM");
      });
      actions.append(toolsButton);
    }
    actions.append(deleteButton);
    card.append(heading, actions, details); sessionList.append(card);
  }
}

document.getElementById("tab-sessions").addEventListener("click", async () => {
  sessionManagerStatus.textContent = "";
  setSessionsOpen(true);
  sessionNameInput.focus();
  try { await renderSessions(); } catch (err) { sessionManagerStatus.textContent = "Could not load sessions: " + err; }
});
document.getElementById("btn-close-sessions").addEventListener("click", () => setSessionsOpen(false));

document.getElementById("session-save-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = sessionNameInput.value.trim();
  if (!name) return;
  try {
    // A session tied to an ALREADY-TRAINED bundle has its model name frozen
    // server-side (renaming it would invalidate that build's validation
    // report -- see app.py's immutable-name-after-training guard). "Save
    // as <name>" here must not try to rename that locked record; treat it
    // as bookmarking the current settings under a new, independent plain
    // session instead, so it can never collide with that guard.
    const reuseActiveId = activeSessionId && !activeSessionGenerated;
    const session = await currentSession(name, false);
    if (!reuseActiveId) session.id = sessionId();
    activeSessionId = session.id;
    activeSessionName = name;
    activeSessionGenerated = false;
    await writeSession(session);
    sessionNameInput.value = "";
    sessionManagerStatus.textContent = `Saved “${name}”.`;
    sessionSettingsStatus.textContent = `Saved ${name}`;
    await renderSessions();
  } catch (err) { sessionManagerStatus.textContent = "Save failed: " + err.message; }
});

document.getElementById("btn-export-current-session").addEventListener("click", async () => {
  const name = sessionNameInput.value.trim() || `Session ${new Date().toLocaleString()}`;
  try {
    if (!activeSessionId) activeSessionId = sessionId();
    activeSessionName = name;
    const session = await writeSession(await currentSession(name));
    sessionNameInput.value = "";
    sessionManagerStatus.textContent = `Saved and exported “${name}”.`;
    sessionSettingsStatus.textContent = `Saved ${name}`;
    downloadSessionJson(session);
    await renderSessions();
  } catch (err) { sessionManagerStatus.textContent = "Export failed: " + err; }
});

const importSessionFileInput = document.getElementById("import-session-file");
document.getElementById("btn-import-session").addEventListener("click", () => importSessionFileInput.click());
importSessionFileInput.addEventListener("change", async () => {
  const file = importSessionFileInput.files[0];
  importSessionFileInput.value = "";
  if (!file) return;
  try {
    const session = importedSession(JSON.parse(await file.text()), file.name);
    await writeSession(session);
    sessionManagerStatus.textContent = `Imported ${session.name}.`;
    sessionSettingsStatus.textContent = `Imported ${session.name}`;
    await renderSessions();
  } catch (err) {
    sessionManagerStatus.textContent = "Import failed: " + err;
  }
});

updateBackendPanels();

// ---- NAM Tools: all file writes happen through the server's strict
// approved-path validator; this UI only selects an app-managed source NAM. ----
const builderTab = document.getElementById("tab-builder");
const toolsTab = document.getElementById("tab-tools");
const toolsPanel = document.getElementById("nam-tools-panel");
const sessionsTab = document.getElementById("tab-sessions");
const tone3000Tab = document.getElementById("tab-tone3000");
const tone3000Panel = document.getElementById("tone3000-panel");
const aiAssistantTab = document.getElementById("tab-ai-assistant");
const aiAssistantPanel = document.getElementById("ai-assistant-panel");
const wizardTab = document.getElementById("tab-wizard");
const wizardPanel = document.getElementById("wizard-panel");
const settingsTab = document.getElementById("tab-settings");
const settingsPanel = document.getElementById("settings-panel");
const toolEditors = document.getElementById("tool-editors");
const toolInfo = document.getElementById("tool-nam-info");
const toolResult = document.getElementById("tool-result");
let toolNamPath = null;
let originalToolMetadata = {};
let toolLoudnessDb = null;
const toolVolumeSlider = document.getElementById("tool-volume-slider");
const toolVolumeValue = document.getElementById("tool-volume-value");
const toolVolumeBaseline = document.getElementById("tool-volume-baseline");
const toolCalibrationStatus = document.getElementById("tool-calibration-status");

function setBuilderTabActive(active) {
  builderTab.classList.toggle("active", active);
  builderTab.setAttribute("aria-pressed", active ? "true" : "false");
}

function setToolsOpen(open) {
  toolsPanel.hidden = !open;
  if (open) {
    sessionsPanel.hidden = true;
    sessionsTab.classList.remove("active");
    sessionsTab.setAttribute("aria-pressed", "false");
    aiAssistantPanel.hidden = true;
    aiAssistantTab.classList.remove("active");
    aiAssistantTab.setAttribute("aria-pressed", "false");
    wizardPanel.hidden = true;
    wizardTab.classList.remove("active");
    wizardTab.setAttribute("aria-pressed", "false");
    tone3000Panel.hidden = true;
    tone3000Tab.classList.remove("active");
    tone3000Tab.setAttribute("aria-pressed", "false");
    settingsPanel.hidden = true;
    settingsTab.classList.remove("active");
    settingsTab.setAttribute("aria-pressed", "false");
  }
  document.querySelectorAll(".workflow-nav, .layout").forEach((el) => { el.hidden = open; });
  document.getElementById("mode-description").hidden = open;
  toolsTab.classList.toggle("active", open);
  toolsTab.setAttribute("aria-pressed", open ? "true" : "false");
  setBuilderTabActive(!open);
}
function setSessionsOpen(open) {
  sessionsPanel.hidden = !open;
  if (open) {
    toolsPanel.hidden = true;
    toolsTab.classList.remove("active");
    toolsTab.setAttribute("aria-pressed", "false");
    aiAssistantPanel.hidden = true;
    aiAssistantTab.classList.remove("active");
    aiAssistantTab.setAttribute("aria-pressed", "false");
    wizardPanel.hidden = true;
    wizardTab.classList.remove("active");
    wizardTab.setAttribute("aria-pressed", "false");
    tone3000Panel.hidden = true;
    tone3000Tab.classList.remove("active");
    tone3000Tab.setAttribute("aria-pressed", "false");
    settingsPanel.hidden = true;
    settingsTab.classList.remove("active");
    settingsTab.setAttribute("aria-pressed", "false");
  }
  document.querySelectorAll(".workflow-nav, .layout").forEach((el) => { el.hidden = open; });
  document.getElementById("mode-description").hidden = open;
  sessionsTab.classList.toggle("active", open);
  sessionsTab.setAttribute("aria-pressed", open ? "true" : "false");
  setBuilderTabActive(!open);
}
function setAiAssistantOpen(open) {
  aiAssistantPanel.hidden = !open;
  if (open) {
    toolsPanel.hidden = true;
    sessionsPanel.hidden = true;
    wizardPanel.hidden = true;
    tone3000Panel.hidden = true;
    settingsPanel.hidden = true;
    toolsTab.classList.remove("active"); toolsTab.setAttribute("aria-pressed", "false");
    sessionsTab.classList.remove("active"); sessionsTab.setAttribute("aria-pressed", "false");
    wizardTab.classList.remove("active"); wizardTab.setAttribute("aria-pressed", "false");
    tone3000Tab.classList.remove("active"); tone3000Tab.setAttribute("aria-pressed", "false");
    settingsTab.classList.remove("active"); settingsTab.setAttribute("aria-pressed", "false");
  }
  document.querySelectorAll(".workflow-nav, .layout").forEach((el) => { el.hidden = open; });
  document.getElementById("mode-description").hidden = open;
  aiAssistantTab.classList.toggle("active", open);
  aiAssistantTab.setAttribute("aria-pressed", open ? "true" : "false");
  setBuilderTabActive(!open);
}
function setWizardOpen(open) {
  wizardPanel.hidden = !open;
  if (open) {
    wizardInstrument.value = instrumentSelect.value;
    populateWizardProfiles();
    toolsPanel.hidden = true;
    sessionsPanel.hidden = true;
    aiAssistantPanel.hidden = true;
    tone3000Panel.hidden = true;
    settingsPanel.hidden = true;
    toolsTab.classList.remove("active"); toolsTab.setAttribute("aria-pressed", "false");
    sessionsTab.classList.remove("active"); sessionsTab.setAttribute("aria-pressed", "false");
    aiAssistantTab.classList.remove("active"); aiAssistantTab.setAttribute("aria-pressed", "false");
    tone3000Tab.classList.remove("active"); tone3000Tab.setAttribute("aria-pressed", "false");
    settingsTab.classList.remove("active"); settingsTab.setAttribute("aria-pressed", "false");
  }
  document.querySelectorAll(".workflow-nav, .layout").forEach((el) => { el.hidden = open; });
  document.getElementById("mode-description").hidden = open;
  wizardTab.classList.toggle("active", open);
  wizardTab.setAttribute("aria-pressed", open ? "true" : "false");
  setBuilderTabActive(!open);
}
function setTone3000Open(open) {
  tone3000Panel.hidden = !open;
  if (open) {
    toolsPanel.hidden = true; sessionsPanel.hidden = true; aiAssistantPanel.hidden = true; wizardPanel.hidden = true; settingsPanel.hidden = true;
    toolsTab.classList.remove("active"); toolsTab.setAttribute("aria-pressed", "false");
    sessionsTab.classList.remove("active"); sessionsTab.setAttribute("aria-pressed", "false");
    aiAssistantTab.classList.remove("active"); aiAssistantTab.setAttribute("aria-pressed", "false");
    wizardTab.classList.remove("active"); wizardTab.setAttribute("aria-pressed", "false");
    settingsTab.classList.remove("active"); settingsTab.setAttribute("aria-pressed", "false");
  }
  document.querySelectorAll(".workflow-nav, .layout").forEach((el) => { el.hidden = open; });
  document.getElementById("mode-description").hidden = open;
  tone3000Tab.classList.toggle("active", open);
  tone3000Tab.setAttribute("aria-pressed", open ? "true" : "false");
  setBuilderTabActive(!open);
}
function setSettingsOpen(open) {
  settingsPanel.hidden = !open;
  if (open) {
    toolsPanel.hidden = true; sessionsPanel.hidden = true; aiAssistantPanel.hidden = true; wizardPanel.hidden = true; tone3000Panel.hidden = true;
    toolsTab.classList.remove("active"); toolsTab.setAttribute("aria-pressed", "false");
    sessionsTab.classList.remove("active"); sessionsTab.setAttribute("aria-pressed", "false");
    aiAssistantTab.classList.remove("active"); aiAssistantTab.setAttribute("aria-pressed", "false");
    wizardTab.classList.remove("active"); wizardTab.setAttribute("aria-pressed", "false");
    tone3000Tab.classList.remove("active"); tone3000Tab.setAttribute("aria-pressed", "false");
    loadSettings();
  }
  document.querySelectorAll(".workflow-nav, .layout").forEach((el) => { el.hidden = open; });
  document.getElementById("mode-description").hidden = open;
  settingsTab.classList.toggle("active", open);
  settingsTab.setAttribute("aria-pressed", open ? "true" : "false");
  setBuilderTabActive(!open);
}
function setBuilderOpen() {
  setToolsOpen(false);
  setSessionsOpen(false);
  setAiAssistantOpen(false);
  setWizardOpen(false);
  setTone3000Open(false);
  setSettingsOpen(false);
}
function showToolResult(data) {
  toolResult.hidden = false;
  toolResult.replaceChildren();
  const changed = document.createElement("div");
  changed.textContent = `Validated changes: ${data.changed_paths.join(", ")}`;
  const link = document.createElement("button"); link.type = "button"; link.className = "btn btn-primary"; link.textContent = desktopSaveLabel(`Download ${data.filename}`);
  link.addEventListener("click", () => triggerFileDownload(data.download_url, data.filename));
  toolResult.append(changed, link);
  if (data.validation_report_invalidated) {
    const validation = document.createElement("div");
    validation.className = "warning-box";
    validation.textContent = "This edited NAM has different bytes from its source. Any earlier validation report does not apply; validate the edited export separately.";
    toolResult.append(validation);
  }
  if (data.warning) { const warning = document.createElement("div"); warning.className = "warning-box"; warning.textContent = data.warning; toolResult.append(warning); }
}
function updateToolVolumeReadout() { toolVolumeValue.textContent = `${Number(toolVolumeSlider.value).toFixed(1)} dB`; }
async function setToolNam(data, label) {
  const response = await fetch("/api/nam/tools/inspect", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: data.path }) });
  const inspection = await response.json();
  if (!response.ok) { toolInfo.textContent = `Error: ${inspection.error}`; return; }
  toolNamPath = inspection.path;
  originalToolMetadata = inspection.metadata || {};
  toolLoudnessDb = Number.isFinite(inspection.loudness_db) ? inspection.loudness_db : null;
  const fieldMap = {
    name: "tool-meta-name", modeled_by: "tool-meta-modeled-by",
    gear_make: "tool-meta-gear-make", gear_model: "tool-meta-gear-model", tone_type: "tool-meta-tone-type",
  };
  Object.entries(fieldMap).forEach(([key, id]) => { document.getElementById(id).value = originalToolMetadata[key] ?? ""; });

  const readOnly = inspection.read_only_metadata || {};
  const exportInfo = document.getElementById("tool-export-info");
  exportInfo.replaceChildren();
  const rows = [
    ["Architecture", inspection.architecture || "Unknown"],
    ["Gear type", readOnly.gear_type || "Not specified"],
    ["Recognised output scales", String(inspection.head_scales.length)],
  ];
  for (const [label, value] of rows) {
    const dt = document.createElement("dt"); dt.textContent = label;
    const dd = document.createElement("dd"); dd.textContent = value;
    exportInfo.append(dt, dd);
  }

  const calibration = inspection.calibration || {};
  toolCalibrationStatus.textContent = calibration.status === "Calibrated NAM"
    ? `Calibration: input ${calibration.input_level_dbu.toFixed(1)} dBu · output ${calibration.output_level_dbu.toFixed(1)} dBu (read-only)`
    : "Calibration metadata unavailable. Do not invent these values; a generated hybrid records input calibration only when both source NAMs are calibrated.";
  if (inspection.volume_unsupported_reason) {
    // e.g. an embedded-cab export's "Sequential" architecture -- see
    // hybrid/sequential_nam.py. Metadata editing below still works fine;
    // only the volume slider (which needs a recognised head_scale) is
    // unavailable for this file.
    toolVolumeSlider.disabled = true;
    toolVolumeBaseline.textContent = "Volume adjustment isn't supported for this NAM's architecture "
      + `(${inspection.architecture || "unknown"}). The metadata editor below still works normally.`;
  } else if (toolLoudnessDb !== null) {
    toolVolumeSlider.value = Math.max(Number(toolVolumeSlider.min), Math.min(Number(toolVolumeSlider.max), toolLoudnessDb));
    toolVolumeSlider.disabled = false;
    toolVolumeBaseline.textContent = `Current measured loudness: ${toolLoudnessDb.toFixed(1)} dB. Drag to choose the final level.`;
  } else {
    toolVolumeSlider.disabled = true;
    toolVolumeBaseline.textContent = "This NAM has no measured loudness metadata, so an absolute output slider cannot be set safely.";
  }
  updateToolVolumeReadout();
  toolEditors.hidden = false;
  toolInfo.textContent = `${label} — ${inspection.architecture || "NAM"}; ${inspection.head_scales.length} recognised output scale${inspection.head_scales.length === 1 ? "" : "s"}.`;
  toolResult.hidden = true;
}
toolsTab.addEventListener("click", () => setToolsOpen(true));
document.getElementById("btn-close-tools").addEventListener("click", () => setToolsOpen(false));
builderTab.addEventListener("click", () => setBuilderOpen());
tone3000Tab.addEventListener("click", () => setTone3000Open(true));
document.getElementById("btn-close-tone3000").addEventListener("click", () => setTone3000Open(false));

// ---- Settings: exposes the same env vars the app has always read (see
// hybrid/settings.py), with a place to change them without a shell. ----
const settingsGroups = document.getElementById("settings-groups");
const settingsStatus = document.getElementById("settings-status");
let settingsFields = [];

async function loadSettings() {
  settingsStatus.textContent = "Loading…";
  try {
    const response = await fetch("/api/settings");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "failed to load settings");
    settingsFields = data.settings || [];
    renderSettings();
    settingsStatus.textContent = "";
  } catch (err) {
    settingsStatus.textContent = "Load failed: " + err;
  }
  loadSetupChecklist();
}

// ---- Getting-started checklist: aggregates the nam_render / local A2
// training / Kaggle / local LLM status endpoints each subsystem already
// exposes into one glance-able list, so a fresh checkout doesn't require
// discovering each setup button separately. See /api/setup/status.
const setupChecklist = document.getElementById("setup-checklist");

async function loadSetupChecklist() {
  if (!setupChecklist) return;
  setupChecklist.replaceChildren();
  const loading = document.createElement("p");
  loading.className = "info";
  loading.textContent = "Checking setup status…";
  setupChecklist.append(loading);
  try {
    const response = await fetch("/api/setup/status");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "failed to load setup status");
    renderSetupChecklist(data.items || []);
  } catch (err) {
    setupChecklist.replaceChildren();
    const failed = document.createElement("p");
    failed.className = "info";
    failed.textContent = "Could not check setup status: " + err;
    setupChecklist.append(failed);
  }
}

function renderSetupChecklist(items) {
  setupChecklist.replaceChildren();
  for (const item of items.filter((i) => i.applicable)) {
    const row = document.createElement("div");
    row.className = "setup-checklist-item" + (item.ready ? " is-ready" : "");
    const icon = document.createElement("span");
    icon.className = "setup-checklist-icon";
    icon.textContent = item.ready ? "✓" : "○";
    const body = document.createElement("div");
    body.className = "setup-checklist-body";
    const label = document.createElement("span");
    label.className = "setup-checklist-label";
    label.textContent = item.label;
    const detail = document.createElement("span");
    detail.className = "setup-checklist-detail";
    detail.textContent = item.detail || "";
    body.append(label, detail);
    row.append(icon, body);
    setupChecklist.append(row);
  }
}

function renderSettings() {
  settingsGroups.replaceChildren();
  const groups = new Map();
  for (const field of settingsFields) {
    if (!groups.has(field.group)) groups.set(field.group, []);
    groups.get(field.group).push(field);
  }
  for (const [groupName, fields] of groups) {
    const isAdvanced = groupName === "Advanced";
    const section = document.createElement(isAdvanced ? "details" : "div");
    section.className = isAdvanced ? "settings-group settings-group-advanced" : "settings-group";
    if (isAdvanced) {
      const summary = document.createElement("summary");
      summary.textContent = groupName;
      section.append(summary);
    } else {
      const heading = document.createElement("h3");
      heading.textContent = groupName;
      section.append(heading);
    }
    const provider = (settingsFields.find((field) => field.name === "NAM_MIXER_AI_PROVIDER") || {}).value || "local";
    for (const field of fields) {
      const row = document.createElement("label");
      row.className = field.kind === "checkbox" ? "settings-row settings-row-checkbox" : "settings-row";
      if (field.kind === "number") row.classList.add("settings-row-narrow");
      if (field.providers?.length) {
        row.hidden = !field.providers.includes(provider);
        row.dataset.providerField = "true";
      }
      const labelText = document.createElement("span");
      labelText.className = "settings-row-label";
      labelText.textContent = field.label + (field.restart_required ? " (restart required)" : "");
      const input = field.kind === "select" ? document.createElement("select") : document.createElement("input");
      if (field.kind !== "checkbox") input.className = "select-input";
      if (field.kind === "checkbox") input.type = "checkbox";
      else if (field.kind !== "select") input.type = field.kind === "number" ? "number" : field.kind === "secret" ? "password" : "text";
      input.dataset.settingName = field.name;
      let suggestionsList = null;
      if (field.suggestions?.length) {
        const listId = `setting-suggestions-${field.name}`;
        const datalist = document.createElement("datalist");
        datalist.id = listId;
        for (const suggestion of field.suggestions) {
          const option = document.createElement("option");
          option.value = suggestion;
          datalist.append(option);
        }
        input.setAttribute("list", listId);
        suggestionsList = datalist;
      }
      if (field.kind === "select") {
        for (const optionData of field.options || []) {
          const option = document.createElement("option");
          option.value = optionData.value;
          option.textContent = optionData.label;
          input.append(option);
        }
      }
      const desc = document.createElement("span");
      desc.className = "info";
      if (field.kind === "secret") {
        // The real value never comes back from the server (see
        // hybrid/settings.py's get_settings); leaving this blank on save
        // means "unchanged", not "clear it".
        input.placeholder = field.has_value ? "Currently set — leave blank to keep unchanged" : (field.placeholder || "");
        input.value = "";
        desc.textContent = field.description + (field.has_value ? " (a key is currently saved)" : "");
      } else if (field.kind === "checkbox") {
        input.checked = !!field.value;
        desc.textContent = field.description;
      } else {
        input.placeholder = field.placeholder || "";
        input.value = field.value || "";
        if (field.name === "NAM_MIXER_AI_MODEL" && provider === "cloudflare" && input.value === "@cf/openai/gpt-oss-20b") {
          input.value = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
        }
        desc.textContent = field.description;
      }
      if (field.kind === "checkbox") row.append(input, labelText, desc);
      else row.append(labelText, input, desc);
      section.append(row);
      if (suggestionsList) section.append(suggestionsList);
      if (field.name === "NAM_MIXER_AI_PROVIDER") {
        const setupRow = document.createElement("div");
        setupRow.className = "settings-row settings-download-row";
        setupRow.dataset.cloudflareSetup = "true";
        setupRow.hidden = provider !== "cloudflare";
        const setupLink = document.createElement("a");
        setupLink.href = "https://developers.cloudflare.com/workers-ai/get-started/rest-api/";
        setupLink.target = "_blank";
        setupLink.rel = "noopener noreferrer";
        setupLink.className = "btn btn-secondary btn-small";
        setupLink.textContent = "Open Cloudflare setup";
        const setupNote = document.createElement("span");
        setupNote.className = "info";
        setupNote.textContent = "Create the Workers AI token and copy your Account ID, then return here.";
        setupRow.append(setupLink, setupNote);
        section.append(setupRow);
      }
      if (field.kind === "secret" && field.name === "NAM_MIXER_AI_API_KEY") {
        const clear = document.createElement("button");
        clear.type = "button";
        clear.className = "btn btn-secondary btn-small";
        clear.textContent = "Clear API token";
        clear.hidden = !field.has_value || row.hidden;
        clear.addEventListener("click", async () => {
          if (!confirm("Clear the stored API token?")) return;
          const response = await fetch("/api/settings", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({values: {}, clear_secrets: [field.name]})});
          if (!response.ok) { settingsStatus.textContent = "Could not clear API token."; return; }
          await loadSettings();
        });
        section.append(clear);
      }
      if (field.name === "NAM_RENDER_EXE" && !isTauriDesktop) {
        // The desktop build always ships nam_render already bundled inside
        // the app itself (see packaging/backend/nam_mixer_backend.spec) --
        // this "fetch it from GitHub releases" flow only applies to a
        // browser/web checkout that hasn't built/downloaded one yet.
        section.append(renderNamRenderDownloadRow());
      }
      if (field.name === "NAM_MIXER_AI_MODEL") {
        section.append(renderLocalLlmStatusRow());
      }
      if (field.name === "TONE3000_API_KEY") {
        section.append(renderTone3000ApiKeyLinkRow());
      }
    }
    settingsGroups.append(section);
  }
  applyExperimentalArchitecturesVisibility();
}

// ---- Experimental NAM architectures gate (Sequential Embedded) ----
// Off by default; see hybrid/settings.py's
// NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES and app.py's server-side
// enforcement in _resolve_cab_design -- this UI-side hide is a convenience,
// not the real gate.
const cabExportOptionEmbedded = document.getElementById("cab-export-option-embedded");
const cabExportExperimentalHint = document.getElementById("cab-export-experimental-hint");
let sequentialEmbeddedWarningAcknowledged = false;

function experimentalArchitecturesEnabled() {
  const field = settingsFields.find((f) => f.name === "NAM_MIXER_ENABLE_EXPERIMENTAL_ARCHITECTURES");
  return !!field?.value;
}

function applyExperimentalArchitecturesVisibility() {
  const enabled = experimentalArchitecturesEnabled();
  if (cabExportOptionEmbedded) cabExportOptionEmbedded.hidden = !enabled;
  if (cabExportExperimentalHint) cabExportExperimentalHint.hidden = enabled;
  if (!enabled && cabExportMode.value === "embedded") {
    cabExportMode.value = "learned";
    updateCabStatus();
  }
}

function renderTone3000ApiKeyLinkRow() {
  const row = document.createElement("div");
  row.className = "settings-row settings-download-row";
  const link = document.createElement("a");
  link.href = "https://www.tone3000.com";
  link.target = "_blank";
  link.rel = "noopener";
  link.className = "btn btn-secondary btn-small";
  link.textContent = "Get a TONE3000 API key";
  row.append(link);
  return row;
}

function renderNamRenderDownloadRow() {
  const row = document.createElement("div");
  row.className = "settings-row settings-download-row";
  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn btn-secondary btn-small";
  button.textContent = "Download nam_render automatically";
  const status = document.createElement("span");
  status.className = "info";
  status.textContent = "Fetches the prebuilt renderer for this OS from GitHub releases -- no separate install or manual path needed.";
  button.addEventListener("click", async () => {
    button.disabled = true;
    status.textContent = "Downloading…";
    try {
      const response = await fetch("/api/renderer/download", { method: "POST" });
      const data = await response.json();
      if (!data.ok) throw new Error(data.error || "download failed");
      status.textContent = `Installed at ${data.path}. Reloading settings…`;
      await loadSettings();
      refreshRendererReadiness();
    } catch (err) {
      status.textContent = "Download failed: " + err;
    } finally {
      button.disabled = false;
    }
  });
  row.append(button, status);
  return row;
}

function renderLocalLlmStatusRow() {
  const row = document.createElement("div");
  row.className = "settings-row settings-download-row";
  row.dataset.aiStatusRow = "true";
  const status = document.createElement("span");
  status.className = "info";
  status.textContent = "Checking AI provider configuration…";

  const testButton = document.createElement("button");
  testButton.type = "button";
  testButton.className = "btn btn-secondary btn-small";
  testButton.textContent = "Test connection";

  const pullButton = document.createElement("button");
  pullButton.type = "button";
  pullButton.className = "btn btn-secondary btn-small";
  pullButton.textContent = "Pull gemma4:e4b via Ollama";
  pullButton.hidden = true;
  let pullActivityStop = null;

  async function refreshStatus() {
    try {
      const response = await fetch("/api/local_llm/status");
      const data = await response.json();
      const selectedProvider = settingsGroups.querySelector('[data-setting-name="NAM_MIXER_AI_PROVIDER"]')?.value || "local";
      if (!data.enabled && selectedProvider === "cloudflare") {
        status.textContent = data.error === "not configured"
          ? "Cloudflare is selected. Enter your API token, then save Settings."
          : `Cloudflare is not ready: ${data.error || "complete the Account ID, model, and API token fields above."}`;
        pullButton.hidden = true;
      } else if (!data.enabled && selectedProvider === "custom") {
        status.textContent = data.error === "not configured"
          ? "Custom AI is selected. Enter a model and HTTPS base URL, then save Settings."
          : `Custom AI is not ready: ${data.error || "check the provider fields above."}`;
        pullButton.hidden = true;
      } else if (!data.enabled) {
        status.textContent = "Not configured -- set a model name above to enable the AI Assistant tab. "
          + "Any OpenAI-compatible local host works (Ollama, LM Studio, etc.); we recommend Ollama + gemma4:e4b "
          + "if you don't already have one running.";
        pullButton.hidden = false;
      } else if (data.provider !== "local") {
        status.textContent = `Configured for ${data.provider === "cloudflare" ? "Cloudflare Workers AI" : "a custom AI provider"} (model: ${data.model}). Testing sends a small real request that may count against quota or billing.`;
        pullButton.hidden = true;
        if (data.provider === "cloudflare") {
          const usageLink = document.createElement("a");
          usageLink.href = "https://dash.cloudflare.com/";
          usageLink.target = "_blank";
          usageLink.rel = "noopener noreferrer";
          usageLink.textContent = " Open Cloudflare usage dashboard";
          usageLink.className = "settings-inline-link";
          status.append(usageLink);
        }
      } else if (data.reachable) {
        status.textContent = `Local provider reachable (model: ${data.model}).`;
        pullButton.hidden = true;
      } else {
        status.textContent = `Configured for ${data.base_url}, but nothing responded there. `
          + "Make sure your local LLM host (Ollama, LM Studio, etc.) is running.";
        pullButton.hidden = false;
      }
    } catch (err) {
      status.textContent = "Could not check local AI assistant status: " + err;
    }
  }

  async function pollPullStatus() {
    const response = await fetch("/api/local_llm/pull_status");
    const data = await response.json();
    if (data.status === "running") {
      status.textContent = `Pulling ${data.model}… ${(data.log_tail || "").split("\n").slice(-1)[0] || ""}`;
      if (!pullActivityStop) pullActivityStop = beginActivity(`Pulling ${data.model} via Ollama…`);
      else pullActivityStop.update(`Pulling ${data.model} via Ollama…`);
      setTimeout(pollPullStatus, 1500);
    } else if (data.status === "done") {
      status.textContent = `Pulled ${data.model}. Set "Local AI assistant: model name" above to ${data.model} and save.`;
      pullButton.disabled = false;
      if (pullActivityStop) { pullActivityStop(); pullActivityStop = null; }
    } else if (data.status === "error") {
      status.textContent = "Pull failed: " + data.error;
      pullButton.disabled = false;
      if (pullActivityStop) { pullActivityStop(); pullActivityStop = null; }
    }
  }

  pullButton.addEventListener("click", async () => {
    pullButton.disabled = true;
    pullActivityStop = beginActivity("Starting the local AI model download…");
    status.textContent = "Starting download…";
    try {
      const response = await fetch("/api/local_llm/pull", { method: "POST" });
      const data = await response.json();
      if (!data.ok) throw new Error(data.error || "pull failed");
      pollPullStatus();
    } catch (err) {
      status.textContent = "Could not start pull: " + err;
      pullButton.disabled = false;
      if (pullActivityStop) { pullActivityStop(); pullActivityStop = null; }
    }
  });

  testButton.addEventListener("click", async () => {
    testButton.disabled = true;
    const stopActivity = beginActivity("Testing the AI provider connection…");
    status.textContent = "Testing connection (this may use provider quota)…";
    try {
      const response = await fetch("/api/local_llm/test", {method: "POST"});
      const data = await response.json();
      status.textContent = data.ok ? "Connected." : (data.error || "Connection test failed.");
      if (!data.ok && data.diagnostics) {
        const detail = document.createElement("span");
        detail.className = "info";
        const diagnostics = data.diagnostics;
        const shape = diagnostics.shape || `content=${diagnostics.content_type || "unknown"}, length=${diagnostics.content_length ?? "unknown"}`;
        detail.textContent = ` Diagnostic: ${shape}${diagnostics.response_keys ? `; response keys: ${diagnostics.response_keys.join(", ")}` : ""}`;
        status.append(detail);
      }
    } catch (err) { status.textContent = "Connection test failed: " + err; }
    finally { stopActivity(); testButton.disabled = false; }
  });

  refreshStatus();
  row.refreshAiStatus = refreshStatus;
  row.append(pullButton, testButton, status);
  return row;
}

document.getElementById("btn-save-settings").addEventListener("click", async () => {
  const values = {};
  settingsGroups.querySelectorAll("[data-setting-name]").forEach((input) => {
    values[input.dataset.settingName] = input.type === "checkbox" ? input.checked : input.value;
  });
  settingsStatus.textContent = "Saving…";
  try {
    const response = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "save failed");
    settingsStatus.textContent = `Saved (${data.saved.length} setting${data.saved.length === 1 ? "" : "s"}).`;
    await loadSettings();
    // Re-check anything the just-saved values could have changed the
    // availability of, so the AI Assistant/TONE3000 tabs reflect reality
    // immediately rather than only after a page reload.
    await Promise.all([loadLocalRecipeAiStatus(), refreshTone3000Status()]);
  } catch (err) {
    settingsStatus.textContent = "Save failed: " + err;
  }
});
settingsGroups.addEventListener("change", async (event) => {
  if (event.target.dataset.settingName !== "NAM_MIXER_AI_PROVIDER") return;
  const provider = event.target.value;
  settingsGroups.querySelectorAll("[data-cloudflare-setup]").forEach((row) => { row.hidden = provider !== "cloudflare"; });
  if (provider === "cloudflare") {
    const modelInput = settingsGroups.querySelector('[data-setting-name="NAM_MIXER_AI_MODEL"]');
    if (modelInput && (!modelInput.value || modelInput.value === "@cf/openai/gpt-oss-20b")) {
      modelInput.value = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
    }
  }
  settingsGroups.querySelector("[data-ai-status-row]")?.refreshAiStatus?.();
  settingsGroups.querySelectorAll("[data-provider-field]").forEach((row) => {
    const fieldName = row.querySelector("[data-setting-name]")?.dataset.settingName;
    const field = settingsFields.find((candidate) => candidate.name === fieldName);
    row.hidden = Boolean(field?.providers?.length && !field.providers.includes(provider));
  });
  // Persist the provider immediately. This is intentionally a narrow save so
  // changing provider cannot be lost because another settings control is
  // blank/hidden; the full Save button still persists the remaining fields.
  const providerValues = { NAM_MIXER_AI_PROVIDER: provider };
  const modelInput = settingsGroups.querySelector('[data-setting-name="NAM_MIXER_AI_MODEL"]');
  if (modelInput?.value) providerValues.NAM_MIXER_AI_MODEL = modelInput.value;
  settingsStatus.textContent = "Saving provider…";
  try {
    const response = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values: providerValues }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "provider save failed");
    await loadSettings();
    settingsStatus.textContent = "Provider saved.";
  } catch (err) {
    settingsStatus.textContent = "Provider save failed: " + err;
  }
});
document.getElementById("btn-reload-settings").addEventListener("click", () => loadSettings());
settingsTab.addEventListener("click", () => setSettingsOpen(true));
document.getElementById("btn-close-settings").addEventListener("click", () => setSettingsOpen(false));
const tone3000Query = document.getElementById("tone3000-query");
const tone3000RigScope = document.getElementById("tone3000-rig-scope");
const tone3000Author = document.getElementById("tone3000-author");
const tone3000Status = document.getElementById("tone3000-status");
const tone3000Results = document.getElementById("tone3000-results");

let tone3000ApiKeyConfigured = false;
async function refreshTone3000Status() {
  try {
    const response = await fetch("/api/settings");
    const data = await response.json();
    const field = (data.settings || []).find((f) => f.name === "TONE3000_API_KEY");
    tone3000ApiKeyConfigured = Boolean(field && field.has_value);
    tone3000Status.textContent = tone3000ApiKeyConfigured
      ? ""
      : "Not set up — add a TONE3000 API key in Settings to search captures.";
  } catch (_error) {
    tone3000ApiKeyConfigured = false;
    tone3000Status.textContent = "Could not check TONE3000 setup — see Settings.";
  }
}
refreshTone3000Status();

function scorePlanTermAgainst(value, text) {
  const terms = new Set((value.toLowerCase().match(/[a-z0-9]+/g) || []).filter((term) => term.length >= 3));
  let score = 0;
  terms.forEach((term) => { if (text.includes(term)) score += term.length; });
  return score;
}

// Mirrors the server's _plan_score fallback in app.py -- if the durable
// source plan (set once the AI has proposed Amp A/Amp B families) already
// distinguishes this candidate, we know its role before ever asking the AI
// again, so the "discuss this pack" flow does not need to re-ask "should it
// be Amp A or Amp B" for a pack that was already searched for one of them.
function inferSourceRoleForResult(result) {
  if (!recipeSourcePlan || !recipeSourcePlan.ampA || !recipeSourcePlan.ampB) return null;
  const text = `${result.title || ""} ${result.creator || ""} ${result.description || ""} ${result.query || ""}`.toLowerCase();
  const aScore = scorePlanTermAgainst(recipeSourcePlan.ampA, text);
  const bScore = scorePlanTermAgainst(recipeSourcePlan.ampB, text);
  if (aScore === bScore) return null;
  return aScore > bScore ? "a" : "b";
}

function createTone3000DiscussButton(result, onError) {
  const discuss = document.createElement("button");
  discuss.type = "button"; discuss.className = "btn btn-secondary btn-small"; discuss.textContent = "See files / ask AI which to use";
  discuss.addEventListener("click", async () => {
    discuss.disabled = true;
    discuss.textContent = "Loading pack…";
    try {
      const packResponse = await fetch(`/api/tone3000/tones/${encodeURIComponent(result.id)}/models`);
      const pack = await packResponse.json();
      if (!packResponse.ok) throw new Error(pack.error || "Could not load this TONE3000 pack");
      const inferredRole = inferSourceRoleForResult(result);
      showSelectedTone3000Capture({ ...result, models: pack.models, sourceRole: inferredRole || undefined });
      setAiAssistantOpen(true);
      recipePromptInput.value = inferredRole
        ? `Which specific file in this pack should I use for Amp ${inferredRole.toUpperCase()}, and why?`
        : pack.models.length > 1
          ? "This TONE3000 pack has several NAM files. Which specific one should I use for my tone, and should it be Amp A or Amp B?"
          : "Should this TONE3000 file be Amp A or Amp B for my tone?";
      recipePromptInput.focus();
    } catch (error) {
      if (onError) onError(error.message);
    } finally {
      discuss.disabled = false;
      discuss.textContent = "See files / ask AI which to use";
    }
  });
  return discuss;
}

function showSelectedTone3000Capture(capture) {
  selectedTone3000Capture = capture;
  aiTone3000Context.hidden = false;
  aiTone3000Context.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = "Discussing: " + capture.title;
  const note = document.createElement("p");
  note.textContent = "The AI can recommend a specific NAM file below. Download any file directly when you are ready.";
  if (capture.sourceRole) {
    const role = document.createElement("p");
    role.className = "info";
    role.textContent = `Assigned role: Amp ${capture.sourceRole.toUpperCase()} (${capture.sourceRole === "a" ? "clean/foundation" : "driven"} source).`;
    aiTone3000Context.append(heading, note, role);
  }
  const fileDetails = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = `${capture.models.length} NAM file${capture.models.length === 1 ? "" : "s"} ready to download`;
  const models = document.createElement("div");
  models.className = "ai-tone3000-models";
  capture.models.forEach((model) => {
    const download = document.createElement("button");
    download.type = "button";
    download.className = "btn btn-secondary btn-small";
    download.textContent = desktopSaveLabel("Download " + model.name);
    const url = `/api/tone3000/tones/${encodeURIComponent(capture.id)}/models/${encodeURIComponent(model.id)}/download`;
    download.addEventListener("click", () => triggerFileDownload(url, model.name));
    models.append(download);
  });
  fileDetails.append(summary, models);
  const clear = document.createElement("button");
  clear.type = "button"; clear.className = "btn btn-secondary btn-small"; clear.textContent = "Stop discussing this pack";
  clear.addEventListener("click", () => { selectedTone3000Capture = null; aiTone3000Context.hidden = true; });
  if (!capture.sourceRole) aiTone3000Context.append(heading, note);
  aiTone3000Context.append(fileDetails, clear);
}

document.getElementById("btn-tone3000-search").addEventListener("click", async () => {
  const query = tone3000Query.value.trim();
  if (!query) { tone3000Status.textContent = "Enter an amp or tone to search for."; return; }
  tone3000Status.textContent = "Searching TONE3000…";
  const stopActivity = beginActivity("Searching TONE3000 captures…");
  tone3000Results.hidden = true;
  tone3000Results.replaceChildren();
  try {
    const response = await fetch("/api/tone3000/search", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, rig_scope: tone3000RigScope.value, author: tone3000Author.value.trim() }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "TONE3000 search failed");
    data.results.forEach((result) => {
      const card = document.createElement("article");
      card.className = "tone3000-result";
      const heading = document.createElement("h3");
      heading.textContent = result.title;
      const meta = document.createElement("p");
      meta.textContent = "By " + result.creator + (Number.isFinite(result.match_score) ? ` · ${result.match_score}% metadata fit` : "");
      const description = document.createElement("p");
      description.textContent = result.description || "No description supplied.";
      if (result.match_reason) description.title = result.match_reason;
      const discuss = createTone3000DiscussButton(result, (message) => { tone3000Status.textContent = message; });
      card.append(heading, meta, description, discuss);
      tone3000Results.append(card);
    });
    tone3000Results.hidden = false;
    tone3000Status.textContent = data.results.length + " capture" + (data.results.length === 1 ? "" : "s") + " found.";
  } catch (error) {
    tone3000Status.textContent = error.message;
  } finally {
    stopActivity();
  }
});
aiAssistantTab.addEventListener("click", () => {
  setAiAssistantOpen(true);
  recipePromptInput.focus();
});
document.getElementById("btn-close-ai-assistant").addEventListener("click", () => setAiAssistantOpen(false));
wizardTab.addEventListener("click", () => setWizardOpen(true));
document.getElementById("btn-close-wizard").addEventListener("click", () => setWizardOpen(false));
document.getElementById("btn-tool-upload").addEventListener("click", async () => {
  const file = document.getElementById("tool-nam-file").files[0];
  if (!file) { toolInfo.textContent = "Choose a .nam file first."; return; }
  const form = new FormData(); form.append("file", file);
  const response = await fetch("/api/nam/upload", { method: "POST", body: form });
  const data = await response.json();
  if (!response.ok) { toolInfo.textContent = `Error: ${data.error}`; return; }
  await setToolNam(data, file.name);
});
document.getElementById("btn-tool-generated").addEventListener("click", async () => {
  const response = await fetch("/api/nam/tools/generated"); const data = await response.json();
  if (!response.ok || !data.path) { toolInfo.textContent = data.error || "No locally generated NAM is available yet."; return; }
  const inspect = await fetch("/api/nam/inspect", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: data.path }) });
  const summary = await inspect.json();
  if (!inspect.ok) { toolInfo.textContent = `Error: ${summary.error}`; return; }
  await setToolNam({ ...summary, path: data.path }, data.filename);
});
toolVolumeSlider.addEventListener("input", updateToolVolumeReadout);
document.getElementById("btn-tool-volume").addEventListener("click", async () => {
  if (!toolNamPath || toolLoudnessDb === null) { toolInfo.textContent = "This NAM needs measured loudness metadata for the absolute output slider."; return; }
  const db_change = Number(toolVolumeSlider.value) - toolLoudnessDb;
  const response = await fetch("/api/nam/tools/volume", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: toolNamPath, db_change }) });
  const data = await response.json();
  if (!response.ok) { toolInfo.textContent = `Error: ${data.error}`; return; }
  showToolResult(data);
});
document.getElementById("btn-tool-metadata").addEventListener("click", async () => {
  if (!toolNamPath) return;
  const metadata = {};
  const fieldMap = {
    name: "tool-meta-name", modeled_by: "tool-meta-modeled-by",
    gear_make: "tool-meta-gear-make", gear_model: "tool-meta-gear-model", tone_type: "tool-meta-tone-type",
  };
  Object.entries(fieldMap).forEach(([key, id]) => {
    const raw = document.getElementById(id).value.trim();
    const value = raw;
    const previous = originalToolMetadata[key] ?? "";
    if (value !== previous) metadata[key] = value === "" ? null : value;
  });
  if (!Object.keys(metadata).length) { toolInfo.textContent = "Enter at least one metadata field."; return; }
  const response = await fetch("/api/nam/tools/metadata", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: toolNamPath, metadata }) });
  const data = await response.json();
  if (!response.ok) { toolInfo.textContent = `Error: ${data.error}`; return; }
  showToolResult(data);
});

// Load settings eagerly (not just when the Settings tab opens) so the
// experimental-architectures gate on cab-export-mode reflects a
// previously-saved preference immediately, without requiring a detour
// through the Settings tab first.
loadSettings();
