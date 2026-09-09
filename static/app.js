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
const modeTabs = document.querySelectorAll(".mode-tab[data-mode]");
const modePanels = document.querySelectorAll("[data-mode-panel]");
const btnPreviewMix = document.getElementById("btn-preview-mix");
const autoLevelMatchLabel = document.getElementById("auto-level-match-label");
const createA2Title = document.getElementById("create-a2-title");
const createA2Description = document.getElementById("create-a2-description");
const workflowTabs = document.querySelectorAll(".workflow-tab");
const workflowHint = document.getElementById("workflow-hint");
const modeDescription = document.getElementById("mode-description");
let workflowStage = "configure";

const WORKFLOW_HINTS = {
  configure: "Add two amps and choose a test performance to begin.",
  shape: "Choose whether the amps change with your playing, stay mixed, or combine their character.",
  listen: "Compare Amp A, the result, and Amp B. Add a cabinet or adjust output level only if needed.",
  create: "Turn the sound you chose into one NAM model, then choose where to train it.",
};

function setWorkflowStage(stage) {
  workflowStage = stage;
  document.body.dataset.workflowStage = stage;
  workflowTabs.forEach((tab) => {
    const active = tab.dataset.workflowStage === stage;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-current", active ? "step" : "false");
  });
  workflowHint.textContent = WORKFLOW_HINTS[stage];
}

workflowTabs.forEach((tab) => tab.addEventListener("click", () => setWorkflowStage(tab.dataset.workflowStage)));
setWorkflowStage(workflowStage);

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
  auditionMode.textContent = currentMode === "blend" ? "Parallel Blend" : currentMode === "character" ? "Character Blend" : "Dynamic Hybrid";
  modeDescription.textContent = currentMode === "blend"
    ? "Both amps are present all the time at one fixed ratio. Use this for a permanent mixed rig."
    : currentMode === "character"
      ? "Choose broad tone, playing feel, and drive character from either amp. This creates one combined character."
      : "Amp A handles quieter playing and Amp B takes over as the input becomes louder. Use this for a clean-to-driven response.";
}

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    // Tools temporarily hides the builder layout. Returning through any
    // normal design tab must restore it before applying the selected mode.
    setToolsOpen(false);
    currentMode = tab.dataset.mode;
    modeTabs.forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-pressed", t === tab ? "true" : "false");
    });
    applyModeVisibility();
    if (workflowStage !== "configure") setWorkflowStage("shape");
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

function formatElapsed(seconds) {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}m ${String(whole % 60).padStart(2, "0")}s`;
}

// Long operations should never look frozen. The caller owns the final status
// message; stopping the timer deliberately leaves that message intact.
function showElapsed(statusElement, message) {
  const started = Date.now();
  const update = () => { statusElement.textContent = `${message} · ${formatElapsed((Date.now() - started) / 1000)}`; };
  update();
  const timer = setInterval(update, 1000);
  return () => clearInterval(timer);
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
  const stopElapsed = showElapsed(infoEl, `Uploading ${file.name}`);
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
    stopElapsed();
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
  resetGeneratedModel("The cabinet changed. Create new training files before starting another training run.");
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
  const stopElapsed = showElapsed(cabInfoEl, `Uploading ${file.name}`);
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
  } finally {
    stopElapsed();
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
  resetGeneratedModel("The source or input settings changed. Create new training files when you are happy with the new sound.");
  if (havePair) {
    clearAudition();
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

testGainSlider.addEventListener("input", () => {
  testGainValue.textContent = `${fmtSigned(testGainSlider.value)} dB`;
  if (!havePair) return; // nothing rendered yet -- Render Amps sets the baseline first
  if (testGainRenderTimer) clearTimeout(testGainRenderTimer);
  testGainStatus.textContent = "Will re-render shortly...";
  testGainRenderTimer = setTimeout(async () => {
    previewButtons.forEach((btn) => (btn.disabled = true));
    setRenderBusy(true);
    player.classList.add("player-busy");
    const stopElapsed = showElapsed(testGainStatus, "Preparing both amps for the new input level");
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
      stopElapsed();
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
const wizardToggle = document.getElementById("btn-wizard-toggle");
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
  modeTabs.forEach((tab) => {
    const active = tab.dataset.mode === mode;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-pressed", active ? "true" : "false");
  });
  applyModeVisibility();
}

wizardToggle.addEventListener("click", () => {
  const isOpen = wizardBody.hidden;
  wizardBody.hidden = !isOpen;
  wizardToggle.setAttribute("aria-expanded", String(isOpen));
  wizardToggle.textContent = isOpen ? "Close wizard" : "Set up a sound";
  if (isOpen) {
    wizardInstrument.value = instrumentSelect.value;
    populateWizardProfiles();
  }
});
wizardSwitch.addEventListener("input", updateWizardLabels);
wizardMoreB.addEventListener("input", updateWizardLabels);
wizardInstrument.addEventListener("change", () => populateWizardProfiles({ preserveCurrent: false }));
wizardProfile.addEventListener("change", updateWizardProfileDescription);
document.querySelectorAll('input[name="wizard-behaviour"]').forEach((input) => input.addEventListener("change", updateWizardLabels));
populateWizardProfiles();
updateWizardLabels();

document.getElementById("btn-wizard-apply").addEventListener("click", () => {
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
});

wizardAnalyseButton.addEventListener("click", async () => {
  wizardResult.hidden = false;
  wizardResult.textContent = "Listening to the rendered pair…";
  wizardAnalyseButton.disabled = true;
  try {
    const resp = await fetch("/api/wizard/insight", { method: "POST" });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Could not analyse the rendered amps.");
    wizardResult.textContent = `${data.level_text} ${data.tone_text} ${data.feel_text}`;
  } catch (err) {
    wizardResult.textContent = `Analysis unavailable: ${err.message}`;
  } finally {
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

function clearAudition() {
  // Do not leave an old result playing after an upstream setting has changed.
  previewRequestId += 1;
  player.pause();
  player.removeAttribute("src");
  player.load();
  lastPreviewSource = null;
  lastSourcePlayed = null;
}

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
  if (liveAudition.active) liveBlendButton.textContent = "Stop live mix adjustment";
  else liveBlendButton.textContent = currentMode === "blend" ? "Adjust the mix while listening" : "Live mix: Always-on mix only";
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
    liveBlendStatus.textContent = "Live mix adjustment is available in Always-on mix. This result updates after you adjust a control.";
    return;
  }
  if (liveAudition.active) {
    liveAudition.stop();
    liveBlendStatus.textContent = "Live blend stopped.";
    return;
  }
  liveBlendButton.disabled = true;
  liveBlendStatus.textContent = "Loading the prepared amps...";
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
    liveBlendStatus.textContent = "Adjust the mix while listening. Changes are immediate.";
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
  scheduleUpdate();
  scheduleAuditionRefresh();
});
outputGainManualSlider.addEventListener("input", () => {
  outputGainManualValue.textContent = `${fmtSigned(outputGainManualSlider.value)} dB`;
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

  const profile = currentProfile();
  const instrument_type = instrumentSelect.value;
  const input_profile_id = profileSelect.value;
  const custom_input_gain_db = profile && profile.requires_custom_gain ? parseFloat(customGainSlider.value) : null;
  const calibration_mode = calibrationModeSelect.value;
  const reference_input_level_dbu = parseFloat(referenceDbuInput.value) || 12.0;
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

  previewButtons.forEach((btn) => (btn.disabled = false));
  liveBlendButton.disabled = false;
  document.getElementById("btn-character-low-level-check").disabled = false;
  havePair = true;
  wizardAnalyseButton.disabled = false;
  setWorkflowStage("listen");
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
  renderPairBtn.disabled = true;
  setRenderBusy(true);
  const stopElapsed = showElapsed(renderStatus, "Preparing both amps for comparison");
  renderWarnings.hidden = true;
  previewButtons.forEach((btn) => (btn.disabled = true));
  document.getElementById("btn-character-low-level-check").disabled = true;
  try {
    const data = await doRenderPair();
    applyRenderResult(data, { applySuggestedCrossover: isFirstRenderThisSession });
    renderStatus.textContent =
      `Rendered ${data.duration_s.toFixed(1)}s @ ${data.sample_rate} Hz -- ` +
      `input peak ${data.input_peak_dbfs.toFixed(1)} dBFS.`;
    setStatus("Amp pair rendered and cached -- sliders now only recompute the blend.");
  } catch (err) {
    renderStatus.textContent = "Error: " + err.message;
    setStatus("Render failed.", true);
  } finally {
    stopElapsed();
    setRenderBusy(false);
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
  const outputGainParams = ["hybrid", "blend", "character"].includes(source) ? outputGainParamsBody() : {};
  const body = { source, ...modeParams, ...cabParamsBody(), ...outputGainParams };
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
    updateOutputGainReadout(resp.headers);
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
      ? `Ready: valid training input loaded (${data.sample_rate / 1000} kHz)`
      : "Add the official NAM training input to continue.";
  } catch (err) {
    trainingInputStatus.textContent = "Could not check training input status: " + err;
  }
}
refreshTrainingInputStatus();

trainingInputFile.addEventListener("change", async () => {
  const file = trainingInputFile.files[0];
  if (!file) return;
  const stopElapsed = showElapsed(trainingInputStatus, `Uploading ${file.name}`);
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
    trainingInputStatus.textContent = `Ready: valid training input loaded (${data.sample_rate / 1000} kHz)`;
  } catch (err) {
    trainingInputStatus.textContent = "Upload failed: " + err;
  } finally {
    stopElapsed();
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
  const stopElapsed = showElapsed(characterLowLevelStatus, "Checking how the sound responds to quiet playing");
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
    characterLowLevelStatus.textContent = "Quiet-playing check complete.";
    characterLowLevelResult.hidden = false;
    characterLowLevelResult.innerHTML = renderLowLevelResponseHtml(data.low_level_response);
  } catch (err) {
    characterLowLevelStatus.textContent = "Request failed: " + err;
  } finally {
    stopElapsed();
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
  const stopElapsed = showElapsed(generateStatus, "Creating training files from the official NAM input");
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
        ...outputGainParamsBody(),
      }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      generateStatus.textContent = "Error: " + (data.error || "generation failed");
      setStatus("Training bundle generation failed.", true);
      return;
    }
    generateStatus.textContent = `Training files are ready for ${data.model_name}.`;
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
      ? `<div><strong>Cabinet:</strong> ${data.cab_summary.baked ? "included in this model" : "used for preview only"}</div>`
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
    activeSessionId = sessionId();
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
  } catch (err) {
    generateStatus.textContent = "Request failed: " + err;
    setStatus("Training bundle generation failed.", true);
  } finally {
    stopElapsed();
    generateBtn.disabled = false;
  }
});

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
let localTrainingActive = false;
let kaggleTrainingActive = false;

function trainingIsActive() {
  return localTrainingActive || kaggleTrainingActive;
}

function resetGeneratedModel(reason) {
  // A live training job owns an immutable bundle. Keep its controls and
  // download state intact even if the user starts exploring a new sound.
  if (!lastDesignId || trainingIsActive()) return;
  lastDesignId = null;
  completedNamArtifact = null;
  activeSessionId = null;
  activeSessionName = null;
  activeSessionGenerated = false;
  document.getElementById("a2-training-section").hidden = true;
  kaggleResultEl.hidden = true;
  localResultEl.hidden = true;
  if (reason) setStatus(reason);
}

function syncTrainingControls() {
  const locked = trainingIsActive();
  generateBtn.disabled = locked;
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
const localTrainingLog = document.getElementById("local-training-log");
const localSetupBtn = document.getElementById("btn-local-setup");
const localTrainBtn = document.getElementById("btn-local-train");
const localCancelBtn = document.getElementById("btn-local-cancel");
const localResultEl = document.getElementById("local-result");
let localTrainingPoll = null;
// The design a completed "training" state actually belongs to -- captured
// at the moment Train locally is clicked, since LocalTrainingManager is a
// single global slot with no design_id of its own to poll back.
let localTrainingDesignId = null;

function renderLocalDownloadResult(designId) {
  const downloadUrl = `/api/local_training/download?design_id=${encodeURIComponent(designId)}`;
  completedNamArtifact = { type: "local", designId, downloadUrl, filename: "trained-model.nam" };
  persistActiveSession().catch((err) => console.warn("Could not update completed session:", err));
  localResultEl.hidden = false;
  localResultEl.innerHTML = `<a href="${downloadUrl}" download class="btn btn-primary btn-block">Download trained .nam</a>`;
}

async function refreshLocalTraining() {
  try {
    const resp = await fetch("/api/local_training/status");
    const data = await resp.json();
    localTrainingStatus.textContent = data.state === "ready"
      ? "Local training environment is ready."
      : data.state === "not_configured"
        ? "Set up the dedicated local training environment once."
        : data.state === "cancelled"
          ? "Local process stopped. You can set up or train again when ready."
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
    localTrainingActive = ["setting_up", "training", "cancelling"].includes(data.state);
    localSetupBtn.disabled = localTrainingActive;
    localTrainBtn.disabled = !data.ready || !lastDesignId || localTrainingActive;
    localCancelBtn.hidden = !localTrainingActive;
    localCancelBtn.disabled = data.state === "cancelling";
    syncTrainingControls();
    if (localTrainingActive) {
      if (!localTrainingPoll) localTrainingPoll = setInterval(refreshLocalTraining, 1000);
    } else if (localTrainingPoll) {
      clearInterval(localTrainingPoll); localTrainingPoll = null;
    }
    if (data.state === "complete" && data.exit_code === 0 && localTrainingDesignId) {
      renderLocalDownloadResult(localTrainingDesignId);
    } else if (data.state !== "complete") {
      localResultEl.hidden = true;
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
  localResultEl.hidden = true;
  localTrainingDesignId = lastDesignId;
  const resp = await fetch("/api/local_training/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ design_id: lastDesignId, epoch_preset: selectedEpochPreset() }),
  });
  const data = await resp.json();
  if (!resp.ok) localTrainingStatus.textContent = data.error || "Local training could not start.";
  await refreshLocalTraining();
});

localCancelBtn.addEventListener("click", async () => {
  localCancelBtn.disabled = true;
  localTrainingStatus.textContent = "Stopping the local process…";
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
  completedNamArtifact = { type: "kaggle", designId, jobId, downloadUrl, filename: namFilename };
  persistActiveSession().catch((err) => console.warn("Could not update completed session:", err));
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
  kaggleTrainingActive = true;
  syncTrainingControls();
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
      kaggleTrainingActive = false;
      syncTrainingControls();
      renderKaggleProgress("Error: " + (data.error || "training request failed"), null);
      trainA2Btn.disabled = false;
      return;
    }
    pollKaggleJob(lastDesignId, data.job_id);
  } catch (err) {
    kaggleTrainingActive = false;
    syncTrainingControls();
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
      kaggleTrainingActive = !["complete", "failed"].includes(data.state);
      syncTrainingControls();
      renderKaggleProgress(data.state, data);

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

// ---- Sessions -------------------------------------------------------------
// File-backed library of named control snapshots. Amp/cab files are NOT
// re-uploaded -- a session stores the app-managed paths already resolved by
// /api/nam/upload and /api/cab/upload. Loading never renders automatically,
// so a stale or missing file is reported by the normal Render Amps flow.

function collectSessionSettings() {
  return {
    mode: currentMode,
    ampA: { path: ampServerPaths.a, label: document.getElementById("amp-a-info").textContent },
    ampB: { path: ampServerPaths.b, label: document.getElementById("amp-b-info").textContent },
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
      label: cabInfoEl.textContent,
      previewEnabled: cabPreviewEnabled.checked,
      baked: cabBaked.checked,
    },
    outputGainAuto: outputGainAutoCheckbox.checked,
    outputGainManualDb: outputGainManualSlider.value,
    modelName: document.getElementById("model-name").value,
  };
}

function applySessionSettings(s) {
  ampServerPaths.a = s.ampA.path;
  ampServerPaths.b = s.ampB.path;
  document.getElementById("amp-a-info").textContent = s.ampA.path ? `${s.ampA.label} (restored)` : "";
  document.getElementById("amp-b-info").textContent = s.ampB.path ? `${s.ampB.label} (restored)` : "";

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
  cabInfoEl.textContent = s.cab.path ? `${s.cab.label} (restored)` : "";
  cabPreviewEnabled.disabled = !s.cab.path;
  cabBaked.disabled = !s.cab.path;
  cabPreviewEnabled.checked = s.cab.previewEnabled;
  cabBaked.checked = s.cab.baked;
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
  // hasn't run this session) -- flag it exactly like any other stale change
  // rather than pretending the (unrendered) result is already valid.
  if (ampServerPaths.a && ampServerPaths.b && diSelector.value) {
    renderPairBtn.classList.add("btn-render-stale");
    renderStatus.textContent = "Settings restored -- click Render Amps to rebuild this pair.";
  }
  updateCoverage();
}

const sessionSettingsStatus = document.getElementById("session-settings-status");
const sessionManager = document.getElementById("session-manager");
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
    artifact: await portableArtifact(), ...(generated ? { generated: true } : {}),
  };
}

async function persistActiveSession() {
  if (!activeSessionId) return;
  const name = activeSessionName || document.getElementById("model-name").value.trim() || "Generated session";
  await writeSession(await currentSession(name));
}

function sessionSummary(session) {
  const settings = session.settings || {};
  const amps = [settings.ampA?.label, settings.ampB?.label].filter(Boolean).join(" / ") || "No amps selected";
  const mode = { hybrid: "Dynamic Hybrid", blend: "Parallel Blend", character: "Character Blend" }[settings.mode] || "Unknown mode";
  return { amps, mode, di: settings.diFile || "No test performance", artifact: session.artifact };
}

function downloadSessionNam(artifact) {
  const link = document.createElement("a");
  link.href = artifact.downloadUrl;
  link.download = artifact.filename || "model.nam";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function downloadSessionJson(session) {
  const link = document.createElement("a");
  link.href = `/api/sessions/${encodeURIComponent(session.id)}/download`;
  link.download = `${(session.name || "nam-mixer-session").replace(/[^a-z0-9_-]+/gi, "-")}.nam-mixer-session.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
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
    details.textContent = `Mode: ${summary.mode} · Amps: ${summary.amps} · Test performance: ${summary.di} · Input profile: ${settings.inputProfileId || "—"} · ${shape} · Level match: ${settings.autoLevelMatch ? "on" : "off"} · Cabinet: ${settings.cab?.path ? "selected" : "off"}${summary.artifact ? ` · NAM: ${summary.artifact.filename || "available"}` : " · No completed NAM recorded"}`;
    const detailButton = document.createElement("button"); detailButton.type = "button"; detailButton.className = "btn btn-secondary btn-small"; detailButton.textContent = "Details";
    detailButton.addEventListener("click", () => { details.hidden = !details.hidden; detailButton.textContent = details.hidden ? "Details" : "Hide details"; });
    const loadButton = document.createElement("button"); loadButton.type = "button"; loadButton.className = "btn btn-primary btn-small"; loadButton.textContent = "Load";
    loadButton.addEventListener("click", () => {
      try {
        applySessionSettings(session.settings);
        lastDesignId = session.designId || null;
        completedNamArtifact = session.artifact || null;
        activeSessionId = session.id;
        activeSessionName = session.name;
        activeSessionGenerated = session.generated === true;
        sessionManager.close();
        sessionSettingsStatus.textContent = `Loaded ${session.name || "session"}`;
        setStatus(`Loaded ${session.name || "session"}.`);
      } catch (err) { sessionManagerStatus.textContent = "Could not load this session: " + err; }
    });
    const deleteButton = document.createElement("button"); deleteButton.type = "button"; deleteButton.className = "btn btn-secondary btn-small"; deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", async () => {
      const deleteMessage = session.generated
        ? `Delete “${session.name || "Untitled session"}” and its entire training bundle? This removes all files under work/a2/${session.designId}.`
        : `Delete “${session.name || "Untitled session"}”? This removes its saved session file.`;
      if (!confirm(deleteMessage)) return;
      try {
        const response = await fetch(`/api/sessions/${encodeURIComponent(session.id)}`, { method: "DELETE" });
        if (!response.ok) throw new Error((await response.json()).error || "could not delete session");
        sessionManagerStatus.textContent = "Session deleted.";
        await renderSessions();
      } catch (err) { sessionManagerStatus.textContent = "Delete failed: " + err; }
    });
    const exportButton = document.createElement("button"); exportButton.type = "button"; exportButton.className = "btn btn-secondary btn-small"; exportButton.textContent = "Export JSON";
    exportButton.addEventListener("click", () => downloadSessionJson(session));
    actions.append(detailButton, loadButton, exportButton);
    if (summary.artifact?.downloadUrl) {
      const downloadButton = document.createElement("button"); downloadButton.type = "button"; downloadButton.className = "btn btn-secondary btn-small"; downloadButton.textContent = "Download NAM";
      downloadButton.addEventListener("click", () => downloadSessionNam(summary.artifact));
      actions.append(downloadButton);
    }
    if (summary.artifact?.toolPath) {
      const toolsButton = document.createElement("button"); toolsButton.type = "button"; toolsButton.className = "btn btn-secondary btn-small"; toolsButton.textContent = "Open in NAM Tools";
      toolsButton.addEventListener("click", async () => {
        sessionManager.close(); setToolsOpen(true);
        await setToolNam({ path: summary.artifact.toolPath }, summary.artifact.filename || "Session NAM");
      });
      actions.append(toolsButton);
    }
    actions.append(deleteButton);
    card.append(heading, actions, details); sessionList.append(card);
  }
}

document.getElementById("btn-manage-sessions").addEventListener("click", async () => {
  sessionManagerStatus.textContent = "";
  if (!sessionManager.open) sessionManager.showModal();
  sessionNameInput.focus();
  try { await renderSessions(); } catch (err) { sessionManagerStatus.textContent = "Could not load sessions: " + err; }
});

document.getElementById("session-save-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = sessionNameInput.value.trim();
  if (!name) return;
  try {
    const session = await currentSession(name);
    activeSessionId = session.id;
    activeSessionName = name;
    await writeSession(session);
    sessionNameInput.value = "";
    sessionManagerStatus.textContent = `Saved “${name}”.`;
    sessionSettingsStatus.textContent = `Saved ${name}`;
    await renderSessions();
  } catch (err) { sessionManagerStatus.textContent = "Save failed: " + err; }
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
const toolsTab = document.getElementById("tab-tools");
const toolsPanel = document.getElementById("nam-tools-panel");
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

function setToolsOpen(open) {
  toolsPanel.hidden = !open;
  document.querySelectorAll(".workflow-nav, .tone-wizard, .layout").forEach((el) => { el.hidden = open; });
  document.getElementById("mode-description").hidden = open;
  toolsTab.classList.toggle("active", open);
  toolsTab.setAttribute("aria-pressed", open ? "true" : "false");
}
function showToolResult(data) {
  toolResult.hidden = false;
  toolResult.replaceChildren();
  const changed = document.createElement("div");
  changed.textContent = `Validated changes: ${data.changed_paths.join(", ")}`;
  const link = document.createElement("a"); link.href = data.download_url; link.download = data.filename; link.className = "btn btn-primary"; link.textContent = `Download ${data.filename}`;
  toolResult.append(changed, link);
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
    name: "tool-meta-name", modeled_by: "tool-meta-modeled-by", gear_type: "tool-meta-gear-type",
    gear_make: "tool-meta-gear-make", gear_model: "tool-meta-gear-model", tone_type: "tool-meta-tone-type",
  };
  Object.entries(fieldMap).forEach(([key, id]) => { document.getElementById(id).value = originalToolMetadata[key] ?? ""; });
  const calibration = inspection.calibration || {};
  toolCalibrationStatus.textContent = calibration.status === "Calibrated NAM"
    ? `Calibration: input ${calibration.input_level_dbu.toFixed(1)} dBu · output ${calibration.output_level_dbu.toFixed(1)} dBu (read-only)`
    : "Calibration metadata unavailable. Do not invent these values; a generated hybrid records input calibration only when both source NAMs are calibrated.";
  if (toolLoudnessDb !== null) {
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
    name: "tool-meta-name", modeled_by: "tool-meta-modeled-by", gear_type: "tool-meta-gear-type",
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
