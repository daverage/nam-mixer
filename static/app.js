// Minimal UI wiring. No build step, no framework -- plain DOM + fetch.

const statusEl = document.getElementById("status");

// ---- Design mode tabs (Dynamic Hybrid / Parallel Blend / Character Blend) ----
// Tabs are DESIGN MODES, not separate applications -- Amp A/B, the preview
// DI, input profile/calibration, render, test gain, Listen controls, the
// Cabinet IR stage, official training input, A2 quality, and training all
// stay SHARED between tabs (see docs/blend-mode.md). Only the crossover/
// transition/level-match controls, the journey/coverage diagnostics, and
// the Create A2 wording differ per mode. Switching tabs never re-renders.
let currentMode = "hybrid";
const modeTabs = document.querySelectorAll(".mode-tab");
const modePanels = document.querySelectorAll("[data-mode-panel]");
const btnPreviewMix = document.getElementById("btn-preview-mix");
const autoLevelMatchLabel = document.getElementById("auto-level-match-label");
const createA2Title = document.getElementById("create-a2-title");
const createA2Description = document.getElementById("create-a2-description");

const HYBRID_LEVEL_MATCH_LABEL = "Auto level match Amp B to Amp A near the crossover";
const BLEND_LEVEL_MATCH_LABEL = "Auto level match Amp B to Amp A over active playing";
const HYBRID_A2_DESCRIPTION =
  "Freezes the CURRENT crossover/transition/trim into an immutable design, " +
  "then blends the official NAM training excitation through it (unmodified " +
  "by the input profile above -- that's a design/preview-only control, see docs/phase3.md).";
const BLEND_A2_DESCRIPTION =
  "Freezes the CURRENT fixed mix/trim into an immutable design, then combines " +
  "the official NAM training excitation through both amps at that same ratio " +
  "(unmodified by the input profile above -- that's a design/preview-only control).";
const CHARACTER_A2_DESCRIPTION =
  "Freezes the measured amp-character analysis plus Tone, Feel, and Drive controls, " +
  "then builds the same deterministic one-donor teacher for the official input. " +
  "Character Blend recommends High def (120 epochs).";

function applyModeVisibility() {
  modePanels.forEach((el) => {
    el.hidden = el.dataset.modePanel !== currentMode;
  });
  btnPreviewMix.textContent = currentMode === "blend" ? "Blend" : currentMode === "character" ? "Character" : "Hybrid";
  autoLevelMatchLabel.textContent = currentMode === "blend" ? BLEND_LEVEL_MATCH_LABEL : HYBRID_LEVEL_MATCH_LABEL;
  createA2Title.textContent = currentMode === "blend" ? "Create Blend A2" : currentMode === "character" ? "Create Character A2" : "Create Hybrid A2";
  createA2Description.textContent = currentMode === "blend" ? BLEND_A2_DESCRIPTION : currentMode === "character" ? CHARACTER_A2_DESCRIPTION : HYBRID_A2_DESCRIPTION;
  const auditionMode = document.getElementById("audition-mode");
  auditionMode.textContent = currentMode === "blend" ? "Parallel Blend" : currentMode === "character" ? "Character Blend" : "Dynamic Hybrid";
}

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    currentMode = tab.dataset.mode;
    modeTabs.forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab ? "true" : "false");
    });
    applyModeVisibility();
    // Switching modes never re-renders NAM inference -- just recompute the
    // (already-rendered) mix/journey/coverage panels for the new mode.
    scheduleUpdate();
  });
});
applyModeVisibility();

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

// ---- Shared Cabinet IR stage (both design modes) ----
// See docs/blend-mode.md "SHARED CABINET IR STAGE"/"CAB UI". The cab sits
// AFTER the amp combination and is applied identically to Amp A/Result/Amp B
// previews (fair comparisons) -- see hybrid/cab_ir.py and app.py's
// _parse_cab_params/_resolve_cab_design.
let cabServerPath = null;
const cabFileInput = document.getElementById("cab-file");
const cabInfoEl = document.getElementById("cab-info");
const cabPreviewEnabled = document.getElementById("cab-preview-enabled");
const cabBaked = document.getElementById("cab-baked");
const cabStatusEl = document.getElementById("cab-status");

function updateCabStatus() {
  if (!cabServerPath) {
    cabStatusEl.textContent = "Cab: off";
  } else if (cabBaked.checked) {
    cabStatusEl.textContent = "Cab: baked -- exported A2 will include this cabinet";
  } else if (cabPreviewEnabled.checked) {
    cabStatusEl.textContent = "Cab: preview only -- exported A2 remains amp/head only";
  } else {
    cabStatusEl.textContent = "Cab: off";
  }
}

cabFileInput.addEventListener("change", async () => {
  const file = cabFileInput.files[0];
  cabServerPath = null;
  cabPreviewEnabled.checked = false;
  cabBaked.checked = false;
  cabPreviewEnabled.disabled = true;
  cabBaked.disabled = true;
  if (!file) {
    cabInfoEl.textContent = "";
    updateCabStatus();
    return;
  }
  cabInfoEl.textContent = `Uploading ${file.name}...`;
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
    cabBaked.disabled = false;
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
  }
  updateCabStatus();
});

cabPreviewEnabled.addEventListener("change", () => {
  if (!cabPreviewEnabled.checked) cabBaked.checked = false; // bake requires preview
  updateCabStatus();
  invalidateLiveAudition("Cabinet setting changed — start live blend again to load the matching stems.");
  if (lastPreviewSource) scheduleAuditionRefresh(lastPreviewSource);
});
cabBaked.addEventListener("change", () => {
  // "If Bake cab into A2 is enabled, automatically ensure Use cab in preview
  // is also enabled" -- docs/blend-mode.md "CAB UI".
  if (cabBaked.checked) cabPreviewEnabled.checked = true;
  updateCabStatus();
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
function markProfileStale(reason) {
  if (havePair) {
    previewButtons.forEach((btn) => (btn.disabled = true));
    renderPairBtn.classList.add("btn-render-stale");
    renderStatus.textContent = `${reason || "A setting that affects amp rendering changed"} -- click Render Amps to update.`;
  }
}

document.getElementById("amp-a-file").addEventListener("change", () => markProfileStale("Amp A changed"));
document.getElementById("amp-b-file").addEventListener("change", () => markProfileStale("Amp B changed"));

instrumentSelect.addEventListener("change", () => {
  populateProfileSelect();
  markProfileStale("Instrument changed");
  updateCoverage();
});
profileSelect.addEventListener("change", () => {
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

testGainSlider.addEventListener("input", () => {
  testGainValue.textContent = `${fmtSigned(testGainSlider.value)} dB`;
  if (!havePair) return; // nothing rendered yet -- Render Amps sets the baseline first
  if (testGainRenderTimer) clearTimeout(testGainRenderTimer);
  testGainStatus.textContent = "Will re-render shortly...";
  testGainRenderTimer = setTimeout(async () => {
    previewButtons.forEach((btn) => (btn.disabled = true));
    player.classList.add("player-busy");
    testGainStatus.textContent = "Pushing amp input (running NAM inference twice)...";
    try {
      const data = await doRenderPair();
      applyRenderResult(data, { applySuggestedCrossover: false });
      testGainStatus.textContent = `Updated -- input peak ${data.input_peak_dbfs.toFixed(1)} dBFS.`;
      if (lastPreviewSource) {
        await preview(lastPreviewSource); // refresh whatever's currently loaded/playing
      }
    } catch (err) {
      testGainStatus.textContent = "Error: " + err.message;
      setStatus("Test-gain re-render failed.", true);
    } finally {
      player.classList.remove("player-busy");
    }
  }, TEST_GAIN_DEBOUNCE_MS);
});

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
  markProfileStale("DI clip changed");
});

applyInstrumentHintFromDi();
populateProfileSelect();

const crossoverSlider = document.getElementById("crossover-slider");
const crossoverValue = document.getElementById("crossover-value");
const DEFAULT_CROSSOVER_DBFS = crossoverSlider.value;
crossoverSlider.addEventListener("input", () => {
  crossoverValue.textContent = `${parseFloat(crossoverSlider.value).toFixed(1)} dBFS`;
  scheduleUpdate();
  scheduleAuditionRefresh();
});

const transitionSlider = document.getElementById("transition-slider");
const transitionValue = document.getElementById("transition-value");
transitionSlider.addEventListener("input", () => {
  transitionValue.textContent = `${transitionSlider.value} dB`;
  scheduleUpdate();
  scheduleAuditionRefresh();
});

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
let previewRequestId = 0;
let auditionRefreshTimer = null;

function scheduleAuditionRefresh(source = "mix") {
  // Refresh can follow an explicit play action, but controls must never cause
  // sound to start by themselves.
  if (!havePair || !autoAuditionToggle.checked || liveAudition.active || player.paused) return;
  clearTimeout(auditionRefreshTimer);
  auditionRefreshTimer = setTimeout(() => preview(source, { preservePosition: true, quiet: true }), 140);
}

// NAM rendering remains server-side.  Once the pair is rendered, this holds
// two decoded stems in Web Audio and changes their *linear* blend gain at
// audio rate.  That is the exact Fixed Blend equation, not a preview shortcut.
const liveAudition = {
  active: false,
  context: null,
  sourceA: null,
  sourceB: null,
  gainA: null,
  gainB: null,
  compressor: null,
  stop() {
    [this.sourceA, this.sourceB].forEach((source) => {
      if (source) { try { source.stop(); } catch (_) { /* already stopped */ } }
    });
    this.active = false;
    this.sourceA = this.sourceB = this.gainA = this.gainB = null;
    updateLiveAuditionButton();
  },
  setMix(mixB, immediate = false) {
    if (!this.active) return;
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
};

function updateLiveAuditionButton() {
  if (liveAudition.active) liveBlendButton.textContent = "Stop instant live mix";
  else liveBlendButton.textContent = currentMode === "blend" ? "Start instant live mix" : "Instant mix: Parallel Blend";
}

function splitStereoBuffer(context, decoded, channel) {
  const buffer = context.createBuffer(1, decoded.length, decoded.sampleRate);
  buffer.copyToChannel(decoded.getChannelData(channel), 0);
  return buffer;
}

function invalidateLiveAudition(message) {
  if (!liveAudition.active) return;
  liveAudition.stop();
  liveBlendStatus.textContent = message;
}

async function startLiveBlend() {
  if (!havePair) return;
  if (currentMode !== "blend") {
    liveBlendStatus.textContent = "Instant audio-rate mixing is available in Parallel Blend. This mode still refreshes the exact Result while you drag.";
    return;
  }
  if (liveAudition.active) {
    liveAudition.stop();
    liveBlendStatus.textContent = "Live blend stopped.";
    return;
  }
  liveBlendButton.disabled = true;
  liveBlendStatus.textContent = "Loading cached amp stems...";
  try {
    const resp = await fetch("/api/live_blend_stems", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...blendParamsBody(), ...cabParamsBody() }),
    });
    if (!resp.ok) throw new Error((await resp.json()).error || "Could not load live stems.");
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) throw new Error("This browser does not support Web Audio.");
    const context = liveAudition.context || new Context();
    liveAudition.context = context;
    await context.resume();
    const decoded = await context.decodeAudioData(await (await resp.blob()).arrayBuffer());
    if (decoded.numberOfChannels < 2) throw new Error("Live stem response was not stereo.");
    player.pause();
    const sourceA = context.createBufferSource();
    const sourceB = context.createBufferSource();
    sourceA.buffer = splitStereoBuffer(context, decoded, 0);
    sourceB.buffer = splitStereoBuffer(context, decoded, 1);
    sourceA.loop = sourceB.loop = true;
    const gainA = context.createGain();
    const gainB = context.createGain();
    const compressor = context.createDynamicsCompressor();
    compressor.threshold.value = -3;
    compressor.knee.value = 4;
    compressor.ratio.value = 12;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.12;
    sourceA.connect(gainA).connect(compressor);
    sourceB.connect(gainB).connect(compressor);
    compressor.connect(context.destination);
    liveAudition.sourceA = sourceA;
    liveAudition.sourceB = sourceB;
    liveAudition.gainA = gainA;
    liveAudition.gainB = gainB;
    liveAudition.compressor = compressor;
    liveAudition.active = true;
    liveAudition.setMix(parseInt(mixSlider.value, 10) / 100.0, true);
    sourceA.start();
    sourceB.start();
    updateLiveAuditionButton();
    const trim = resp.headers.get("X-Effective-Trim-Db");
    liveBlendStatus.textContent = `Live A/B blend running — drag Mix for immediate changes (B trim ${fmtSigned(trim)} dB).`;
  } catch (err) {
    liveAudition.stop();
    liveBlendStatus.textContent = "Live blend unavailable: " + err.message;
  } finally {
    liveBlendButton.disabled = !havePair;
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

function scheduleUpdate() {
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
      body: JSON.stringify(currentModeParamsBody()),
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

  const profile = currentProfile();
  const instrument_type = instrumentSelect.value;
  const input_profile_id = profileSelect.value;
  const custom_input_gain_db = profile && profile.requires_custom_gain ? parseFloat(customGainSlider.value) : null;
  const calibration_mode = calibrationModeSelect.value;
  const reference_input_level_dbu = parseFloat(referenceDbuInput.value) || 12.0;
  const test_gain_db = parseFloat(testGainSlider.value) || 0.0;

  const resp = await fetch("/api/render_pair", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      amp_a_path, amp_b_path, di_file,
      instrument_type, input_profile_id, custom_input_gain_db,
      calibration_mode, reference_input_level_dbu, test_gain_db,
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
  if (data.warnings && data.warnings.length) {
    renderWarnings.hidden = false;
    renderWarnings.textContent = data.warnings.join(" ");
  } else {
    renderWarnings.hidden = true;
  }

  if (applySuggestedCrossover) {
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
  }

  previewButtons.forEach((btn) => (btn.disabled = false));
  liveBlendButton.disabled = false;
  document.getElementById("btn-character-low-level-check").disabled = false;
  havePair = true;
  updateTrimReadout();
  updateJourney();
  updateCoverage();
}

renderPairBtn.addEventListener("click", async () => {
  renderPairBtn.disabled = true;
  renderStatus.textContent = "Rendering (running NAM inference twice)...";
  renderWarnings.hidden = true;
  previewButtons.forEach((btn) => (btn.disabled = true));
  document.getElementById("btn-character-low-level-check").disabled = true;
  try {
    const data = await doRenderPair();
    applyRenderResult(data, { applySuggestedCrossover: true });
    renderStatus.textContent =
      `Rendered ${data.duration_s.toFixed(1)}s @ ${data.sample_rate} Hz -- ` +
      `input peak ${data.input_peak_dbfs.toFixed(1)} dBFS.`;
    setStatus("Amp pair rendered and cached -- sliders now only recompute the blend.");
  } catch (err) {
    renderStatus.textContent = "Error: " + err.message;
    setStatus("Render failed.", true);
  } finally {
    renderPairBtn.disabled = false;
  }
});

let lastPreviewSource = null;

async function preview(requestedSource, { preservePosition = false, quiet = false } = {}) {
  // "mix" means "whichever design mode's combined result is active" --
  // resolves to the active design source without a separate approximation.
  const source = requestedSource === "mix" ? currentMode : requestedSource;
  lastPreviewSource = source;
  const requestId = ++previewRequestId;
  const wasPlaying = !player.paused;
  const resumeAt = preservePosition && Number.isFinite(player.currentTime) ? player.currentTime : 0;
  const modeParams = ["hybrid", "blend", "character"].includes(source) ? currentModeParamsBody() : {};
  const body = { source, ...modeParams, ...cabParamsBody() };
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
    // Ignore an older cached-blend request that returned after a newer slider
    // value. This prevents audible jumps backwards during a fast drag.
    if (requestId !== previewRequestId) return;
    if (source === "hybrid" || source === "blend") {
      const auto = resp.headers.get("X-Auto-Trim-Db");
      const effective = resp.headers.get("X-Effective-Trim-Db");
      trimReadout.textContent =
        `Auto match ${fmtSigned(auto)} dB  ·  effective trim ${fmtSigned(effective)} dB`;
    }
    const blob = await resp.blob();
    const previousUrl = player.src.startsWith("blob:") ? player.src : null;
    player.src = URL.createObjectURL(blob);
    lastSourcePlayed = requestedSource;
    player.onloadedmetadata = () => {
      if (requestId !== previewRequestId) return;
      if (resumeAt > 0 && player.duration) player.currentTime = Math.min(resumeAt, Math.max(0, player.duration - 0.02));
      if (wasPlaying) player.play();
      if (previousUrl) URL.revokeObjectURL(previousUrl);
    };
    if (!quiet) setStatus(`Playing ${source.toUpperCase()}.`);
  } catch (err) {
    setStatus("Request failed: " + err, true);
  }
}

document.getElementById("btn-preview-a").addEventListener("click", () => preview("a"));
document.getElementById("btn-preview-mix").addEventListener("click", () => preview("mix"));
document.getElementById("btn-preview-b").addEventListener("click", () => preview("b"));
liveBlendButton.addEventListener("click", startLiveBlend);
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

const characterLowLevelBtn = document.getElementById("btn-character-low-level-check");
const characterLowLevelStatus = document.getElementById("character-low-level-status");
const characterLowLevelResult = document.getElementById("character-low-level-result");
characterLowLevelBtn.addEventListener("click", async () => {
  if (!havePair) {
    setStatus("Render and audition an amp pair first.", true);
    return;
  }
  characterLowLevelBtn.disabled = true;
  characterLowLevelResult.hidden = true;
  characterLowLevelStatus.textContent = "Sweeping 0 to -36 dB (running NAM inference several times on a short excerpt)...";
  try {
    const resp = await fetch("/api/character/low_level_check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(characterParamsBody()),
    });
    const data = await resp.json();
    if (!resp.ok) {
      characterLowLevelStatus.textContent = "Error: " + (data.error || "low-level check failed");
      return;
    }
    characterLowLevelStatus.textContent = "Low-level response sweep complete.";
    characterLowLevelResult.hidden = false;
    characterLowLevelResult.innerHTML = renderLowLevelResponseHtml(data.low_level_response);
  } catch (err) {
    characterLowLevelStatus.textContent = "Request failed: " + err;
  } finally {
    characterLowLevelBtn.disabled = false;
  }
});

const generateBtn = document.getElementById("btn-generate");
const modelNameInput = document.getElementById("model-name");
generateBtn.addEventListener("click", async () => {
  if (!havePair) {
    setStatus("Render and audition an amp pair first.", true);
    return;
  }
  if (!trainingInputReady) {
    setStatus("Upload the official NAM training input first.", true);
    return;
  }
  generateBtn.disabled = true;
  generateStatus.textContent = "Generating training bundle (running NAM inference on the official input)...";
  generateResult.hidden = true;
  try {
    const resp = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...currentModeParamsBody(),
        // A browser can retain a cached page template while fetching a newer
        // app.js after an update. Treat the new optional field defensively so
        // that mismatch cannot block generation; the API derives a real name
        // from the selected amps until the page is refreshed.
        model_name: modelNameInput?.value?.trim() || "",
        cab_path: cabServerPath || null,
        cab_preview_enabled: cabPreviewEnabled.checked,
        cab_baked: cabBaked.checked,
      }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      generateStatus.textContent = "Error: " + (data.error || "generation failed");
      setStatus("Training bundle generation failed.", true);
      return;
    }
    generateStatus.textContent = `Bundle generated: ${data.model_name}`;
    generateResult.hidden = false;
    const newWarningsText = data.warnings ? data.warnings.join(" ") : "";
    const alreadyShownAbove =
      newWarningsText && !renderWarnings.hidden && renderWarnings.textContent === newWarningsText;
    const warningsHtml = newWarningsText
      ? `<div class="warning-box">${
          alreadyShownAbove
            ? "Same calibration caveat shown above in Amps &amp; Input applies to this bundle."
            : newWarningsText
        }</div>`
      : "";
    const cabLine = data.cab_summary
      ? `<div><strong>Cab:</strong> ${data.cab_summary.baked ? "baked into this A2" : "not baked (preview only)"} -- ${data.cab_summary.original_filename}</div>`
      : "";
    const lowLevelHtml = data.low_level_response ? renderLowLevelResponseHtml(data.low_level_response) : "";
    generateResult.innerHTML = `
      <div><strong>Bundle:</strong> <code>${data.bundle_dir}</code></div>
      <div><strong>Final model:</strong> <code>${data.download_filename}</code></div>
      <div><strong>Target:</strong> <code>${data.target_path}</code> (peak ${data.safety_report.final_peak_dbfs.toFixed(1)} dBFS, safety reduction ${data.safety_report.gain_reduction_db.toFixed(2)} dB)</div>
      <div><strong>Calibration:</strong> ${data.calibration_summary.effective_mode} (requested ${data.calibration_summary.requested_mode})</div>
      ${cabLine}
      <div><strong>Train it with:</strong> <code>${data.training_command}</code></div>
      ${warningsHtml}
      ${lowLevelHtml}
    `;
    setStatus("Training bundle ready.");

    lastDesignId = data.design_id;
    document.getElementById("a2-training-section").hidden = false;
    if (data.default_epoch_preset === "high_def") document.getElementById("a2-preset-high_def").checked = true;
    refreshLocalTraining();
    refreshKaggleStatus();
  } catch (err) {
    generateStatus.textContent = "Request failed: " + err;
    setStatus("Training bundle generation failed.", true);
  } finally {
    generateBtn.disabled = false;
  }
});

// --- Kaggle GPU training backend -----------------------------------------
let lastDesignId = null;
let kaggleAuthenticated = false;
let kaggleJobPollTimer = null;

const kaggleStatusEl = document.getElementById("kaggle-status");
const kaggleConnectBtn = document.getElementById("btn-kaggle-connect");
const trainA2Btn = document.getElementById("btn-train-a2");
const kaggleProgressBox = document.getElementById("kaggle-progress-box");
const kaggleProgressState = document.getElementById("kaggle-progress-state");
const kaggleProgressMeta = document.getElementById("kaggle-progress-meta");
const kaggleProgressBarTrack = document.getElementById("kaggle-progress-bar-track");
const kaggleProgressBarFill = document.getElementById("kaggle-progress-bar-fill");
const kaggleLogTail = document.getElementById("kaggle-log-tail");
const kaggleResultEl = document.getElementById("kaggle-result");
const kaggleRefreshBtn = document.getElementById("btn-kaggle-refresh");
const kaggleBackendRadio = document.getElementById("a2-backend-kaggle");
const localBackendRadio = document.getElementById("a2-backend-local");
const kagglePanel = document.getElementById("kaggle-panel");
const localPanel = document.getElementById("local-panel");
let kaggleAuthPollTimer = null;
let kaggleJobSubmittedAt = null;

// --- Training quality (epoch preset: draft=20 / standard=60 / high_def=120) -
function selectedEpochPreset() {
  const checked = document.querySelector('input[name="a2-epoch-preset"]:checked');
  return checked ? checked.value : "standard";
}

const localTrainingStatus = document.getElementById("local-training-status");
const localTrainingMeta = document.getElementById("local-training-meta");
const localTrainingLog = document.getElementById("local-training-log");
const localSetupBtn = document.getElementById("btn-local-setup");
const localTrainBtn = document.getElementById("btn-local-train");
let localTrainingPoll = null;

async function refreshLocalTraining() {
  try {
    const resp = await fetch("/api/local_training/status");
    const data = await resp.json();
    localTrainingStatus.textContent = data.state === "ready"
      ? "Local training environment is ready."
      : data.state === "not_configured"
        ? "Set up the dedicated local training environment once."
        : `Local training: ${data.state.replace("_", " ")}.`;
    localTrainingLog.textContent = data.log_tail || "(no local training output yet)";
    if (data.elapsed_s !== null && data.elapsed_s !== undefined) {
      const minutes = Math.floor(data.elapsed_s / 60);
      const seconds = data.elapsed_s % 60;
      const outcome = data.exit_code === null || data.exit_code === undefined ? "running" : `exit ${data.exit_code}`;
      const progress = data.progress ? `Epoch ${data.progress.epoch}/${data.progress.total_epochs}` : "Waiting for first epoch…";
      localTrainingMeta.textContent = `Elapsed ${minutes}m ${seconds}s · ${progress} · ${outcome}`;
    } else {
      localTrainingMeta.textContent = "";
    }
    localSetupBtn.disabled = data.state === "setting_up" || data.state === "training";
    localTrainBtn.disabled = !data.ready || !lastDesignId || data.state === "setting_up" || data.state === "training";
    if (data.state === "setting_up" || data.state === "training") {
      if (!localTrainingPoll) localTrainingPoll = setInterval(refreshLocalTraining, 1000);
    } else if (localTrainingPoll) {
      clearInterval(localTrainingPoll); localTrainingPoll = null;
    }
  } catch (err) { localTrainingStatus.textContent = "Could not check local training: " + err; }
}

localSetupBtn.addEventListener("click", async () => {
  localSetupBtn.disabled = true;
  localTrainingStatus.textContent = "Creating the dedicated environment and installing training packages…";
  const resp = await fetch("/api/local_training/setup", { method: "POST" });
  const data = await resp.json();
  if (!resp.ok) localTrainingStatus.textContent = data.error || "Local setup could not start.";
  await refreshLocalTraining();
});

localTrainBtn.addEventListener("click", async () => {
  if (!lastDesignId) return;
  localTrainBtn.disabled = true;
  const resp = await fetch("/api/local_training/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ design_id: lastDesignId, epoch_preset: selectedEpochPreset() }),
  });
  const data = await resp.json();
  if (!resp.ok) localTrainingStatus.textContent = data.error || "Local training could not start.";
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
  kaggleResultEl.innerHTML = `
    <a href="${downloadUrl}" download class="btn btn-primary btn-block">Download ${namFilename}</a>
    <div class="hint" title="${data.output_nam_path || ""}">Full path: <code>${data.output_nam_path || "(unknown)"}</code></div>
    <div><strong>SHA-256:</strong> <code>${data.output_nam_sha256 || ""}</code></div>
  `;
}

// State label, elapsed-since-submit, and a progress bar/log tail when
// available -- a bare repeating "running" string with no other signal made
// it look stuck even while training was progressing normally.
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
    metaParts.unshift(`epoch ${epoch}/${total_epochs}`);
    kaggleProgressBarTrack.hidden = false;
    kaggleProgressBarFill.style.width = `${Math.min(100, (epoch / total_epochs) * 100)}%`;
  } else {
    kaggleProgressBarTrack.hidden = true;
  }
  kaggleProgressMeta.textContent = metaParts.join(" — ");

  if (data && typeof data.log_tail === "string") {
    kaggleLogTail.textContent = data.log_tail.trim() || "(no log output yet)";
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
    const quota = data.quota_available ? formatGpuQuota(data.quota_raw) : "unavailable";
    kaggleStatusEl.textContent = `Connected ✓  CLI ${data.cli_version || "?"}  GPU: NVIDIA T4  Quota: ${quota}`;

    const activeJob = data.job && data.job.state && !["complete", "failed"].includes(data.job.state);
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

kaggleConnectBtn.addEventListener("click", async () => {
  kaggleConnectBtn.disabled = true;
  kaggleStatusEl.textContent = "Starting Kaggle authentication...";
  try {
    const resp = await fetch("/api/kaggle/auth/start", { method: "POST" });
    const data = await resp.json();
    if (!resp.ok) {
      kaggleStatusEl.textContent = "Error: " + (data.error || "could not start Kaggle auth");
      kaggleConnectBtn.disabled = false;
      return;
    }
    kaggleStatusEl.textContent = data.started
      ? "A Kaggle login flow was started. Complete it in your browser -- this updates automatically once connected."
      : `Run this yourself -- this updates automatically once connected: ${data.command}`;
  } catch (err) {
    kaggleStatusEl.textContent = "Could not start Kaggle auth: " + err;
    kaggleConnectBtn.disabled = false;
    return;
  }
  pollKaggleAuth();
});

// After starting `kaggle auth login` we can't know when the user finishes the
// browser OAuth flow, so poll status for a while rather than requiring a
// manual page refresh -- this is exactly the flow the user found confusing
// (Connect Kaggle appearing to do nothing until a full reload).
function pollKaggleAuth() {
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
      if (!kaggleAuthenticated) {
        kaggleStatusEl.textContent += " Still not connected -- click Connect Kaggle again once you've finished logging in, or Refresh.";
      }
    }
  }, 3000);
}

trainA2Btn.addEventListener("click", async () => {
  if (!lastDesignId) {
    setStatus("Generate a training bundle first.", true);
    return;
  }
  trainA2Btn.disabled = true;
  kaggleJobSubmittedAt = Date.now();
  renderKaggleProgress("Uploading training pair...", null);
  kaggleResultEl.hidden = true;
  try {
    const resp = await fetch("/api/kaggle/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ design_id: lastDesignId, epoch_preset: selectedEpochPreset() }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      renderKaggleProgress("Error: " + (data.error || "training request failed"), null);
      trainA2Btn.disabled = false;
      return;
    }
    pollKaggleJob(lastDesignId, data.job_id);
  } catch (err) {
    renderKaggleProgress("Request failed: " + err, null);
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
  renderKaggleProgress("Checking job...", null);

  // A 1s local ticker keeps "elapsed" visibly moving between the slower
  // network polls below -- reassurance that the page itself hasn't frozen,
  // independent of whether Kaggle actually has anything new to report.
  kaggleTickTimer = setInterval(() => {
    if (kaggleLastJobData) renderKaggleProgress(kaggleLastJobData.state, kaggleLastJobData);
  }, 1000);

  const poll = async () => {
    try {
      const resp = await fetch(`/api/kaggle/jobs/${encodeURIComponent(jobId)}?design_id=${encodeURIComponent(designId)}`);
      const data = await resp.json();
      if (!resp.ok) {
        renderKaggleProgress("Error checking job: " + (data.error || "unknown error"), null);
        return;
      }
      kaggleLastJobData = data;
      renderKaggleProgress(data.state, data);

      if (data.state === "complete") {
        clearInterval(kaggleJobPollTimer);
        clearInterval(kaggleTickTimer);
        trainA2Btn.disabled = false;
        renderKaggleDownloadResult(designId, jobId, data);
        setStatus("Kaggle A2 training complete.");
      } else if (data.state === "failed") {
        clearInterval(kaggleJobPollTimer);
        clearInterval(kaggleTickTimer);
        trainA2Btn.disabled = false;
        renderKaggleProgress("Failed: " + (data.error || "unknown error"), data);
        setStatus("Kaggle A2 training failed.", true);
      }
    } catch (err) {
      renderKaggleProgress("Polling error: " + err, kaggleLastJobData);
    }
  };
  poll();
  kaggleJobPollTimer = setInterval(poll, KAGGLE_JOB_POLL_MS);
}

updateBackendPanels();
