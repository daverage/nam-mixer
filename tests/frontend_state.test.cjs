const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
function section(start, end) {
  assert.ok(source.includes(start) && source.includes(end));
  return source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start)));
}
function deferred() {
  let resolve;
  const promise = new Promise(r => { resolve = r; });
  return { promise, resolve };
}

test('cancelled comparison permits a retry and old completion cannot unlock the new request', async () => {
  const requests = [];
  const sandbox = {
    completedNamArtifact: {}, lastDesignId: 'saved-design',
    comparisonRequestGeneration: 0, comparisonPlayback: null, comparisonData: null,
    comparisonBuildBtn: {}, comparisonSwitches: {}, comparisonMetrics: {}, comparisonStatus: {},
    diSelector: { value: 'music.wav' },
    document: { querySelector: () => ({ value: '-24' }) },
    fetch: () => { const request = deferred(); requests.push(request); return request.promise; },
  };
  vm.createContext(sandbox);
  vm.runInContext(section('function stopModelComparison(', 'function syncComparisonPanel('), sandbox);
  vm.runInContext(section('async function buildModelComparison(', 'comparisonBuildBtn.addEventListener('), sandbox);
  const old = sandbox.buildModelComparison();
  sandbox.invalidateModelComparison('Level changed');
  assert.equal(sandbox.comparisonBuildBtn.disabled, false);
  const retry = sandbox.buildModelComparison();
  requests[0].resolve({ ok: false, json: async () => ({ error: 'old failure' }) });
  await old;
  assert.equal(sandbox.comparisonBuildBtn.disabled, true);
  assert.ok(!sandbox.comparisonStatus.textContent.includes('old failure'));
  requests[1].resolve({ ok: false, json: async () => ({ error: 'current failure' }) });
  await retry;
  assert.equal(sandbox.comparisonBuildBtn.disabled, false);
  assert.ok(sandbox.comparisonStatus.textContent.includes('current failure'));
});

test('editing preview settings preserves a completed artifact and its frozen design', () => {
  const artifact = { filename: 'completed.nam' };
  const sandbox = { lastDesignId: 'saved-design', completedNamArtifact: artifact, trainingIsActive: () => false };
  vm.createContext(sandbox);
  vm.runInContext(section('function resetGeneratedModel(', 'function syncTrainingControls('), sandbox);
  sandbox.resetGeneratedModel('DI changed');
  assert.equal(sandbox.completedNamArtifact, artifact);
  assert.equal(sandbox.lastDesignId, 'saved-design');
});

test('wizard applies instrument, pickup, blend choice and protects them from DI hints', () => {
  const sandbox = {
    instrumentExplicitlySelected: false,
    wizardInstrument: { value: 'bass' }, wizardProfile: { value: 'active_bass' },
    instrumentSelect: { value: 'guitar' }, profileSelect: { value: 'single_coil' },
    populateProfileSelect() { this.profileSelect.value = 'passive_bass'; },
    updateProfileDescription() {}, markProfileStale() {},
    selectedWizardBehaviour: () => 'fixed', setModeFromWizard(mode) { this.currentMode = mode; },
    mixSlider: {}, wizardMoreB: { value: '70' }, updateMixValueLabel() {},
    scheduleUpdate() {}, scheduleAuditionRefresh() {}, wizardResult: {}, havePair: false,
    setWorkflowStage() {}, diSelector: { value: 'guitar_clean.wav' },
  };
  // Functions used as globals are deliberately independent of a JS receiver.
  sandbox.populateProfileSelect = () => { sandbox.profileSelect.value = 'passive_bass'; };
  sandbox.setModeFromWizard = mode => { sandbox.currentMode = mode; };
  vm.createContext(sandbox);
  vm.runInContext(section('function applyWizardSettings()', 'document.getElementById("btn-wizard-apply").addEventListener'), sandbox);
  vm.runInContext(section('function applyInstrumentHintFromDi()', 'diSelector.addEventListener'), sandbox);
  sandbox.applyWizardSettings();
  assert.equal(sandbox.instrumentSelect.value, 'bass');
  assert.equal(sandbox.profileSelect.value, 'active_bass');
  assert.equal(sandbox.currentMode, 'blend');
  assert.equal(sandbox.mixSlider.value, '70');
  sandbox.applyInstrumentHintFromDi();
  assert.equal(sandbox.instrumentSelect.value, 'bass');
  assert.equal(sandbox.profileSelect.value, 'active_bass');
});
function setup() {
  const revoked = [];
  const sandbox = {
    havePair: true, activeRenderId: 'render-a', currentMode: 'blend',
    previewRequestId: 0, lastPreviewSource: null, lastSourcePlayed: null,
    player: {
      src: 'blob:old', paused: true, pause() {}, load() {},
      removeAttribute() { this.src = ''; },
    },
    URL: { revokeObjectURL: url => revoked.push(url), createObjectURL: () => 'blob:new' },
    setStatus() {}, updateOutputGainReadout() {}, fmtSigned: String,
    currentModeParamsBody: () => ({}), outputGainParamsBody: () => ({}),
    cabParamsBody: () => ({}), blendParamsBody: () => ({}),
    trimReadout: {}, liveBlendButton: {}, liveBlendStatus: {}, window: {},
  };
  vm.createContext(sandbox);
  vm.runInContext(section('function clearAudition()', 'function scheduleAuditionRefresh'), sandbox);
  vm.runInContext(section('const liveAudition =', 'function hybridParamsBody()'), sandbox);
  vm.runInContext(section('async function preview(', 'document.getElementById("btn-preview-a")'), sandbox);
  return { sandbox, revoked };
}

test('invalidation during preview body download cannot reinstall old audio', async () => {
  const { sandbox, revoked } = setup();
  const body = deferred();
  const downloading = deferred();
  sandbox.fetch = async () => ({
    ok: true, headers: { get() { return '0'; } },
    blob() { downloading.resolve(); return body.promise; },
  });
  const pending = sandbox.preview('a');
  await downloading.promise;
  sandbox.clearAudition();
  body.resolve({});
  await pending;
  assert.equal(sandbox.player.src, '');
  assert.equal(sandbox.lastSourcePlayed, null);
  assert.deepEqual(revoked, ['blob:old']);
});

test('preview without a current render makes no request', async () => {
  const { sandbox } = setup();
  sandbox.havePair = false;
  sandbox.fetch = () => { throw new Error('must not fetch'); };
  await sandbox.preview('mix');
  assert.equal(sandbox.previewRequestId, 0);
});

test('automatic refresh keeps the selected A/B source', () => {
  let scheduled;
  const sandbox = {
    lastPreviewSource: 'b', havePair: true, autoAuditionToggle: { checked: true },
    liveAudition: { active: false }, player: { paused: false }, auditionRefreshTimer: null,
    clearTimeout() {}, setTimeout(callback) { scheduled = callback; },
    preview(source, options) { sandbox.request = { source, options }; },
  };
  vm.createContext(sandbox);
  vm.runInContext(section('function scheduleAuditionRefresh(', '// NAM rendering remains server-side.'), sandbox);
  sandbox.scheduleAuditionRefresh(); scheduled();
  assert.equal(sandbox.request.source, 'b');
  assert.equal(sandbox.request.options.preservePosition, true);
});

test('a fresh render resumes selected audio at the saved position', async () => {
  const { sandbox } = setup();
  let plays = 0;
  sandbox.player.duration = 20;
  sandbox.player.play = () => { plays += 1; };
  sandbox.fetch = async () => ({ ok: true, headers: { get: () => '0' }, blob: async () => ({}) });
  await sandbox.preview('b', { resumeState: { position: 7, playing: true }, quiet: true });
  sandbox.player.onloadedmetadata();
  assert.equal(sandbox.lastPreviewSource, 'b');
  assert.equal(sandbox.player.currentTime, 7);
  assert.equal(plays, 1);
});

test('invalidation cancels live audio while decoding, even before it is active', async () => {
  const { sandbox } = setup();
  const decoding = deferred();
  const decoded = deferred();
  sandbox.window.AudioContext = class {
    async resume() {}
    decodeAudioData() { decoding.resolve(); return decoded.promise; }
    createBufferSource() { throw new Error('stale audio must not start'); }
  };
  sandbox.fetch = async () => ({
    ok: true, blob: async () => ({ arrayBuffer: async () => new ArrayBuffer(0) }),
  });
  const pending = sandbox.startLiveBlend();
  await decoding.promise;
  sandbox.invalidateLiveAudition('Settings changed');
  decoded.resolve({ numberOfChannels: 2 });
  await pending;
  assert.equal(sandbox.liveBlendStatus.textContent, 'Settings changed');
  assert.equal(vm.runInContext('liveAudition.active', sandbox), false);
});

function installModeTabs(sandbox) {
  const tabs = ['blend', 'character'].map(mode => ({
    dataset: { mode }, classList: { toggle() {} }, setAttribute() {},
    addEventListener(_event, handler) { this.click = handler; },
  }));
  Object.assign(sandbox, {
    modeTabs: tabs, setToolsOpen() {}, setSessionsOpen() {}, applyModeVisibility() {},
    workflowStage: 'listen', setWorkflowStage() {}, scheduleUpdate() {},
  });
  vm.runInContext(section('modeTabs.forEach((tab) => {', 'applyModeVisibility();\n\nfunction setStatus'), sandbox);
  return tabs;
}

test('mode switch during decoding cancels audio even when switching back before completion', async () => {
  const { sandbox } = setup();
  const tabs = installModeTabs(sandbox);
  const decoding = deferred(), decoded = deferred();
  let starts = 0;
  sandbox.window.AudioContext = class {
    async resume() {}
    decodeAudioData() { decoding.resolve(); return decoded.promise; }
    createBufferSource() { starts++; throw new Error('cancelled audio must not start'); }
  };
  sandbox.fetch = async () => ({ ok: true, blob: async () => ({ arrayBuffer: async () => new ArrayBuffer(0) }) });
  const pending = sandbox.startLiveBlend();
  await decoding.promise;
  tabs[1].click();
  tabs[0].click();
  decoded.resolve({ numberOfChannels: 2 });
  await pending;
  assert.equal(starts, 0);
  assert.equal(vm.runInContext('liveAudition.active', sandbox), false);
  assert.equal(sandbox.liveBlendButton.disabled, false);
});

test('live output controls reach the audio gain and automatic gain follows the mix', async () => {
  const { sandbox } = setup();
  const nodes = [];
  const audioParam = () => ({ value: 0, cancelScheduledValues() {},
    setValueAtTime(value) { this.value = value; }, setTargetAtTime(value) { this.value = value; } });
  const node = kind => { const n = { kind, connections: [], connect(to) { this.connections.push(to); return to; } }; nodes.push(n); return n; };
  const decoded = { numberOfChannels: 2, length: 2, sampleRate: 48000,
    getChannelData(channel) { return channel === 0 ? new Float32Array([0.125, -0.125]) : new Float32Array([0.5, -0.5]); } };
  const context = {
    currentTime: 0, destination: {}, async resume() {}, async decodeAudioData() { return decoded; },
    createBuffer() { return { copyToChannel() {} }; },
    createBufferSource() { return Object.assign(node('source'), { start() {}, stop() {} }); },
    createGain() { return Object.assign(node('gain'), { gain: audioParam() }); },
    createDynamicsCompressor() { return Object.assign(node('compressor'), { threshold: {}, knee: {}, ratio: {}, attack: {}, release: {} }); },
  };
  const controls = { checked: false, addEventListener(_name, fn) { this.change = fn; } };
  const slider = { value: '0', addEventListener(_name, fn) { this.input = fn; } };
  Object.assign(sandbox, {
    mixSlider: { value: '0' }, outputGainAutoCheckbox: controls, outputGainManualSlider: slider,
    outputGainManualValue: {}, scheduleUpdate() {}, scheduleAuditionRefresh() {},
    outputGainParamsBody: () => ({ output_gain_mode: controls.checked ? 'auto' : 'manual', manual_output_gain_db: Number(slider.value) }),
  });
  sandbox.window.AudioContext = function () { return context; };
  let requestBody;
  sandbox.fetch = async (_url, options) => {
    requestBody = JSON.parse(options.body);
    return { ok: true, blob: async () => ({ arrayBuffer: async () => new ArrayBuffer(0) }) };
  };
  vm.runInContext(section('outputGainAutoCheckbox.addEventListener(', '// Reads the X-Output-Gain-*'), sandbox);
  await sandbox.startLiveBlend();
  const live = vm.runInContext('liveAudition', sandbox);
  assert.equal(live.active, true);
  assert.equal(requestBody.manual_output_gain_db, 0);
  assert.ok(live.gainA.connections.includes(live.outputGain));
  assert.ok(live.gainB.connections.includes(live.outputGain));
  assert.ok(live.outputGain.connections.includes(live.compressor));
  slider.value = '-12'; slider.input();
  assert.ok(Math.abs(live.outputGain.gain.value - 10 ** (-12 / 20)) < 1e-8);
  controls.checked = true; controls.change();
  assert.ok(Math.abs(live.outputGain.gain.value - 10 ** (-3 / 20) / 0.125) < 1e-8);
  live.setMix(1);
  assert.ok(Math.abs(live.outputGain.gain.value - 10 ** (-3 / 20) / 0.5) < 1e-8);
  installModeTabs(sandbox)[1].click();
  assert.equal(live.active, false);
  assert.equal(live.stems, null);
});

test('renderer readiness distinguishes missing state and recovers on retry', async () => {
  const responses = [
    { found: false, verified: false, error: 'not installed' },
    { found: true, verified: true, path: '/tmp/nam_render' },
  ];
  const retry = { disabled: false, hidden: true, addEventListener(_name, callback) { this.callback = callback; } };
  const sandbox = {
    rendererRetryButton: retry,
    rendererHelp: { hidden: true },
    rendererPath: { textContent: '' },
    renderStatus: { textContent: '', dataset: {} },
    fetch: async () => ({ json: async () => responses.shift() }),
  };
  vm.createContext(sandbox);
  vm.runInContext(section('async function refreshRendererReadiness()', 'refreshRendererReadiness();'), sandbox);

  await sandbox.refreshRendererReadiness();
  assert.equal(sandbox.renderStatus.dataset.rendererState, 'missing');
  assert.equal(retry.hidden, false);
  assert.equal(sandbox.rendererHelp.hidden, false);

  await sandbox.refreshRendererReadiness();
  assert.equal(sandbox.renderStatus.dataset.rendererState, 'ready');
  assert.equal(sandbox.renderStatus.textContent, 'Renderer ready.');
  assert.equal(retry.hidden, true);
});
