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

test('TONE3000 settings validation distinguishes secret and publishable keys', () => {
  const sandbox = {};
  vm.createContext(sandbox);
  vm.runInContext(section('function settingValidationMessage(', 'function renderSettings()'), sandbox);
  const field = {
    label: 'TONE3000 API key', required_prefix: 't3k_cs_',
    validation_message: 'Use the TONE3000 Secret Key beginning t3k_cs_.',
  };
  assert.equal(sandbox.settingValidationMessage(field, 't3k_cs_actual-secret'), '');
  assert.match(sandbox.settingValidationMessage(field, 't3k_pub_publishable'), /t3k_cs_/);
  assert.match(sandbox.settingValidationMessage(field, 't3k_cs_'), /t3k_cs_/);
  assert.equal(sandbox.settingValidationMessage(field, ''), '');
});

test('welcome cookie is versioned and independent of the backend port', () => {
  const sandbox = { WELCOME_COOKIE_NAME: 'nam-mixer-welcome-version' };
  vm.createContext(sandbox);
  vm.runInContext(section('function welcomeCookieHasVersion(', 'document.getElementById("btn-welcome-dismiss")'), sandbox);
  assert.equal(sandbox.welcomeCookieHasVersion('other=x; nam-mixer-welcome-version=1', '1'), true);
  assert.equal(sandbox.welcomeCookieHasVersion('nam-mixer-welcome-version=1', '2'), false);
});

test('conversation export includes optional structured AI and research debug', () => {
  const sandbox = {};
  vm.createContext(sandbox);
  vm.runInContext(section('function buildRecipeConversationMarkdown(', 'recipeSaveMarkdownButton.addEventListener'), sandbox);
  const messages = [
    { role: 'user', text: 'Find a clean tone.' },
    { role: 'assistant', text: 'Try this source.' },
  ];
  const debugTrace = [{
    ai_calls: [{ messages: [{ role: 'user', content: 'exact bounded prompt' }] }],
    research: { tone3000_queries: ['Vox AC30'] },
  }];

  const ordinary = sandbox.buildRecipeConversationMarkdown(messages);
  const debug = sandbox.buildRecipeConversationMarkdown(messages, { includeDebug: true, debugTrace });

  assert.match(ordinary, /## You\n\nFind a clean tone\./);
  assert.doesNotMatch(ordinary, /AI and research debug/);
  assert.match(debug, /# AI and research debug/);
  assert.match(debug, /exact bounded prompt/);
  assert.match(debug, /Vox AC30/);
  assert.match(debug, /Authorization headers are excluded/);
});

test('choosing a Tools NAM opens it immediately without a second confirmation', async () => {
  const calls = [];
  class FakeFormData {
    append(name, value) { this.entry = [name, value]; }
  }
  const sandbox = {
    trainingIsActive: () => false,
    toolInspectorRequestId: 0,
    toolInfo: {},
    toolSourceCard: { classList: { toggle() {} } },
    toolChooseFileButton: {},
    toolGeneratedButton: {},
    FormData: FakeFormData,
    fetch: async (url, options) => {
      calls.push({ url, options });
      return { ok: true, json: async () => ({ path: '/uploads/amp.nam' }) };
    },
    setToolNam: async (data, label) => { sandbox.opened = { data, label }; },
  };
  vm.createContext(sandbox);
  vm.runInContext(section('function setToolSourceBusy(', 'toolChooseFileButton.addEventListener'), sandbox);

  const file = { name: 'amp.nam' };
  await sandbox.openToolNamFile(file);

  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/nam/upload');
  assert.deepEqual(calls[0].options.body.entry, ['file', file]);
  assert.deepEqual(sandbox.opened, { data: { path: '/uploads/amp.nam' }, label: 'amp.nam' });
});

test('selecting a different NAM clears stale Inspector output before upload completes', () => {
  const prompt = { textContent: '', classList: { remove() {} } };
  const result = { hidden: false, classList: { remove() {} }, children: [1], replaceChildren(...children) { this.children = children; } };
  const sandbox = {
    toolInspectorRequestId: 4,
    document: { getElementById(id) { return id === 'tool-inspector-prompt' ? prompt : result; } },
  };
  vm.createContext(sandbox);
  vm.runInContext(section('function clearNamInspectorResult(', 'async function openToolNamFile('), sandbox);
  sandbox.clearNamInspectorResult('Opening next.nam…');
  assert.equal(sandbox.toolInspectorRequestId, 5);
  assert.equal(prompt.textContent, 'Opening next.nam…');
  assert.equal(result.hidden, true);
  assert.deepEqual(result.children, []);
});

test('Tools results appear inline with a calm collapsed validation note', () => {
  function element(tagName = 'div') {
    return {
      tagName, children: [], className: '', textContent: '',
      classList: { add() {}, remove() {} },
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      addEventListener() {},
      scrollIntoView() {},
    };
  }
  const resultEl = element();
  resultEl.hidden = true;
  const sandbox = {
    document: { createElement: tagName => element(tagName) },
    desktopSaveLabel: value => value,
    triggerFileDownload() {},
  };
  vm.createContext(sandbox);
  vm.runInContext(section('function showToolResult(', 'function updateToolVolumeReadout()'), sandbox);

  sandbox.showToolResult({
    filename: 'edited.nam', download_url: '/download/edited.nam',
    changed_paths: ['metadata.name'], validation_report_invalidated: true,
  }, resultEl);

  assert.equal(resultEl.hidden, false);
  assert.equal(resultEl.children[0].textContent, 'New NAM ready');
  assert.equal(resultEl.children[2].textContent, 'Download edited.nam');
  assert.equal(resultEl.children[3].tagName, 'details');
  assert.equal(resultEl.children[3].children[0].textContent, 'About validation reports');
  assert.doesNotMatch(source, /This edited NAM has different bytes from its source/);
});

test('NAM Inspector is a read-only overview with plain facts and expandable technical detail', () => {
  const html = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
  assert.match(html, /<h3 id="tool-overview-title">About this NAM<\/h3>/);
  assert.match(html, /tool-inspector-result/);
  assert.match(html, /tool-inspector-prompt/);
  assert.doesNotMatch(html, /tool-export-info/);   // no duplicate read-only block in the Metadata card
  // The overview comes before the editing tools.
  assert.ok(html.indexOf('tool-inspector-result') < html.indexOf('tool-volume-slider'));

  function element(tagName = 'div') {
    return {
      tagName, children: [], className: '', textContent: '', hidden: true,
      classList: { add() {}, remove() {} },
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
    };
  }
  const result = element();
  const sandbox = { document: { createElement: element } };
  vm.createContext(sandbox);
  vm.runInContext(section('const INSPECTOR_TONE_TYPES', 'toolsTab.addEventListener'), sandbox);
  sandbox.showNamInspectorResult({
    identity: { filename: 'amp.nam', name: 'Amp', tone_type: 'hi_gain' },
    architecture: { name: 'SlimmableContainer', sample_rate: 48000, input_channels: 1, output_channels: 1, a2_packed: true, full_lite_supported: true },
    calibration: { status: 'complete', input_level_dbu: 12, output_level_dbu: -3 },
    cabinet: { label: 'Amp only / no embedded cabinet detected' }, temporal: {},
    validation: { status: 'passed', summary: 'NAMCore render passed', branches: { full: { detail: 'Full passed' }, lite: { detail: 'Lite passed' } } },
  }, result);
  assert.equal(result.hidden, false);
  const [head, facts, details] = result.children;
  assert.equal(head.children[0].textContent, 'Amp');
  assert.equal(head.children[1].textContent, '✓ Plays in NAMCore');
  const fact = (label) => facts.children.find((f) => f.children[0].textContent === label).children[1].textContent;
  assert.equal(fact('Architecture'), 'A2 (SlimmableContainer), Full + Lite');
  assert.equal(fact('Sample rate'), '48 kHz');
  assert.equal(fact('Calibration'), 'Input 12.0 dBu · Output -3.0 dBu');
  assert.match(fact('Cabinet'), /Amp only/);
  assert.equal(fact('Tone type'), 'High gain');
  const other = element();
  sandbox.showNamInspectorResult({ identity: { gear_make: 'Peavey', gear_model: 'Peavey 6505', tone_type: 'metal' }, validation: { status: 'passed' } }, other);
  const otherFact = (label) => other.children[1].children.find((f) => f.children[0].textContent === label).children[1].textContent;
  assert.equal(otherFact('Gear'), 'Peavey 6505');
  assert.equal(otherFact('Tone type'), 'Metal');
  assert.equal(details.tagName, 'details');
  assert.equal(details.children[0].textContent, 'Technical details');
  assert.match(details.children.at(-1).textContent, /Lite passed/);

  const failed = element();
  sandbox.showNamInspectorResult({ architecture: { name: 'LSTM' }, validation: { status: 'failed', summary: 'Render failed: bad' } }, failed);
  assert.equal(failed.children[0].children[1].textContent, '✕ Render failed: bad');
});

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
    showWizardResult() {}, setWorkflowStage() {}, diSelector: { value: 'guitar_clean.wav' },
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

test('plain-English Vox-to-Marshall request becomes a Vox tone/feel and dynamic Marshall drive recipe', () => {
  const sandbox = {};
  vm.createContext(sandbox);
  vm.runInContext(section('function recipeTextIncludes(', 'function setCharacterSlider('), sandbox);
  const recipe = sandbox.recipeFromPrompt('Keep the Vox EQ and feel, then move from Vox gain to Marshall crunch as I play harder.');
  assert.equal(recipe.mode, 'character');
  assert.equal(recipe.tone, 0);
  assert.equal(recipe.feel, 0);
  assert.deepEqual([recipe.driveLow, recipe.driveMid, recipe.driveHigh], [0, 50, 100]);
});

test('plain-English request with explicit tone/gain percentages carries them into the recipe', () => {
  const sandbox = {};
  vm.createContext(sandbox);
  vm.runInContext(section('function recipeTextIncludes(', 'function setCharacterSlider('), sandbox);
  const recipe = sandbox.recipeFromPrompt(
    'I want to have a vox ac30 and blend it with a marshall. The eq and feel should be 70% vox, '
    + 'the gain structure should start off with vox and at around 80 percent volume be 50/50 vox and marshall.'
  );
  assert.equal(recipe.mode, 'character');
  assert.equal(recipe.tone, 30);
  assert.equal(recipe.feel, 30);
  assert.deepEqual([recipe.driveLow, recipe.driveMid, recipe.driveHigh], [0, 25, 50]);
});

test('plain-English parallel request creates a constant parallel blend', () => {
  const sandbox = {};
  vm.createContext(sandbox);
  vm.runInContext(section('function recipeTextIncludes(', 'function setCharacterSlider('), sandbox);
  const recipe = sandbox.recipeFromPrompt('Make a parallel blend, mostly Amp B, with both amps on all the time.');
  assert.equal(recipe.mode, 'blend');
  assert.equal(recipe.mixB, 70);
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

test('validation reports made before the Full/Lite fix are flagged', () => {
  const sandbox = { escapeHtml: (value) => String(value ?? '') };  // the real one needs the DOM
  vm.createContext(sandbox);
  vm.runInContext(section('function validationSummaryHtml(', 'function renderLocalDownloadResult('), sandbox);
  const report = { state: 'passed', summary: 'ok', checks: [] };
  assert.match(sandbox.validationSummaryHtml({ ...report, schema_version: 2 }), /wrong way round/);
  assert.doesNotMatch(sandbox.validationSummaryHtml({ ...report, schema_version: 3 }), /wrong way round/);
});

test('a Continuous Gain job finishing after the user opened another project acts on its own project', async () => {
  const cg = fs.readFileSync(path.join(__dirname, '../static/cg.js'), 'utf8');
  const start = cg.indexOf('  async function runJob(');
  const end = cg.indexOf('  const jobBox', start);
  assert.ok(start > 0 && end > start);
  const loads = [];
  let tick = null;
  const sandbox = {
    S: { id: 'A', job: null, pollTimer: null },
    api: async (url) => (url === '/start' ? { job_id: 'j1' } : { state: 'done', log: [], message: '', elapsed: 1 }),
    render: () => {}, say: () => {},
    load: async (id) => { loads.push(id); },
    setInterval: (fn) => { tick = fn; return 1; }, clearInterval: () => {},
    document: { getElementById: () => null },
  };
  vm.createContext(sandbox);
  vm.runInContext(cg.slice(start, end) + '\nthis.runJob = runJob;', sandbox);
  let afterPid = null;
  await sandbox.runJob('/start', {}, async (_st, pid) => { afterPid = pid; });
  sandbox.S.id = 'B';                          // the user opens project B while A's job runs
  await tick();
  assert.equal(afterPid, 'A');                 // the follow-up targets A, not B
  assert.deepEqual(loads, []);                 // B's screen is not reloaded with A's result
});

test('a stale coverage response cannot overwrite a newer one', async () => {
  const el = () => ({ hidden: false, textContent: '', innerHTML: '', value: '0', classList: { add() {} }, appendChild() {} });
  const pending = [];
  const sandbox = {
    havePair: true, activeRenderId: 'r1',
    crossoverSlider: el(), transitionSlider: el(), customGainSlider: el(), profileSelect: el(),
    coverageTbody: el(), coverageTable: el(), coverageEmpty: el(), coverageWarning: el(),
    instrumentSelect: el(),
    document: { createElement: () => el() },
    fetch: () => new Promise((resolve) => pending.push(resolve)),
  };
  vm.createContext(sandbox);
  vm.runInContext(section('let coverageRequestSeq = 0;', '// ---- Journey chart'), sandbox);
  const older = sandbox.updateCoverage();
  const newer = sandbox.updateCoverage();
  const response = (activeSignal) => ({ ok: true, json: async () => ({ coverage: [], active_signal: activeSignal, reachability_warning: null }) });
  pending[1](response(true));            // the newer request answers first...
  await newer;
  pending[0](response(false));           // ...then the stale one (e.g. an earlier, silent DI)
  await older;
  assert.equal(sandbox.coverageEmpty.hidden, true);   // the stale "no active playing" did not take over
});

test('Continuous Gain step estimates come from one timing table and keep their measured values', () => {
  const cg = fs.readFileSync(path.join(__dirname, '../static/cg.js'), 'utf8');
  const start = cg.indexOf('  const STEP_TIMING = {');
  const end = cg.indexOf('  const formatDuration', start);
  const sandbox = {};
  vm.createContext(sandbox);
  vm.runInContext(cg.slice(start, end) + '\nthis.estimateSeconds = estimateSeconds;', sandbox);
  assert.equal(sandbox.estimateSeconds('analyse', 19), 20 + 19 * 2.6);   // the previous inline formulas
  assert.equal(sandbox.estimateSeconds('generate', 5), 20 + 5 * 4);
  assert.equal(sandbox.estimateSeconds('validate', 4), 15 + 12);
  assert.doesNotMatch(cg, /formatDuration\(\d+ \+/);                    // no inline formula left
});

test('timing readout wording and Original/Corrected request body', () => {
  const sandbox = {};
  vm.createContext(sandbox);
  vm.runInContext(section('let timingChoice', 'function timingEvidenceLines(') + '\nthis.setChoice = (c) => { timingChoice = c; };', sandbox);
  const summary = (status, correction) => sandbox.timingSummary({
    alignment_diagnostic: { status }, timing_correction: correction || { available: false, offset_samples: null },
  });
  assert.equal(summary('fixed_offset', { available: true, offset_samples: 7 }),
    'Timing: Fixed offset detected - Amp B lags Amp A by 7 samples');
  assert.equal(summary('fixed_offset', { available: true, offset_samples: -3 }),
    'Timing: Fixed offset detected - Amp B leads Amp A by 3 samples');
  assert.equal(summary('aligned'), 'Timing: No stable fixed offset detected');
  assert.equal(summary('ambiguous'), 'Timing: No trustworthy fixed timing correction identified');
  // A per-DI fixed offset that cross-DI verification did not confirm is not offered.
  assert.equal(summary('fixed_offset'), 'Timing: No trustworthy fixed timing correction identified');
  assert.equal(summary('insufficient_signal'), 'Timing: Insufficient signal for reliable analysis');
  assert.match(sandbox.timingChoiceHint(7), /^Optional\. .*Corrected moves Amp B earlier by 7 samples/);
  assert.match(sandbox.timingChoiceHint(-3), /Corrected moves Amp B later by 3 samples/);

  sandbox.setChoice({ available: true, offsetSamples: 7, corrected: false });
  assert.equal(JSON.stringify(sandbox.timingParamsBody()), '{}');
  sandbox.setChoice({ available: true, offsetSamples: 7, corrected: true });
  assert.equal(JSON.stringify(sandbox.timingParamsBody()), '{"alignment_enabled":true,"alignment_offset_samples":7}');
  sandbox.setChoice({ available: false, offsetSamples: null, corrected: true });
  assert.equal(JSON.stringify(sandbox.timingParamsBody()), '{}');
});

function timingSandbox() {
  const note = { textContent: '', hidden: true };
  const readout = {
    hidden: true, dataset: {},
    querySelector: (sel) => ({
      '.timing-readout-note': note,
      '.timing-readout-summary': {},
      '.timing-choice': {},
      '.timing-choice-hint': {},
      '.timing-readout-body': { textContent: '', appendChild() {} },
    })[sel],
  };
  const sandbox = {
    ampServerPaths: { a: '/m/a.nam', b: '/m/b.nam' },
    diSelector: { value: 'moderate_brit.wav' },
    document: { querySelectorAll: (sel) => (sel === '.timing-readout' ? [readout] : []), createElement: () => ({}) },
    note,
  };
  vm.createContext(sandbox);
  vm.runInContext(section('let timingChoice', 'function markProfileStale(') +
    '\nthis.choice = () => timingChoice;', sandbox);
  return sandbox;
}

function timingRender(offset) {
  return {
    alignment_diagnostic: { status: offset ? 'fixed_offset' : 'ambiguous', reason: '', windows: [],
      windows_reliable: 0, windows_analysed: 0, agreement_fraction: 0, max_lag_samples: 256 },
    alignment_verification: null,
    timing_correction: offset ? { available: true, offset_samples: offset } : { available: false, offset_samples: null },
  };
}

const saveTiming = (sb) => JSON.parse(JSON.stringify(sb.timingSessionSettings()));

test('session timing: Original saved -> reload -> Original', () => {
  const sb = timingSandbox();
  sb.renderTimingReadout(timingRender(7));                 // verified, user leaves Original
  const saved = saveTiming(sb);
  assert.deepEqual(saved, { choice: 'original', offsetSamples: null, method: 'fixed-frozen-offset' });
  const reloaded = timingSandbox();
  reloaded.restoreTimingIntent(saved);
  reloaded.renderTimingReadout(timingRender(7));
  assert.equal(reloaded.choice().corrected, false);
  assert.equal(reloaded.note.hidden, true);
});

test('session timing: Corrected +7 restored only when the new render verifies exactly +7', () => {
  const sb = timingSandbox();
  sb.renderTimingReadout(timingRender(7));
  sb.choice().corrected = true;
  const saved = saveTiming(sb);
  assert.deepEqual(saved, { choice: 'corrected', offsetSamples: 7, method: 'fixed-frozen-offset' });

  const reloaded = timingSandbox();
  reloaded.restoreTimingIntent(saved);
  assert.equal(reloaded.choice().corrected, false);        // loading alone applies nothing
  assert.deepEqual(saveTiming(reloaded), saved);           // saving again before a render keeps the intent
  reloaded.renderTimingReadout(null);                      // the Render click's own invalidation does not consume it
  reloaded.renderTimingReadout(timingRender(7));
  assert.equal(reloaded.choice().corrected, true);
  assert.equal(JSON.stringify(reloaded.timingParamsBody()), '{"alignment_enabled":true,"alignment_offset_samples":7}');
  assert.match(reloaded.note.textContent, /restored/);
  reloaded.renderTimingReadout(timingRender(7));           // consumed once: a later render starts at Original
  assert.equal(reloaded.choice().corrected, false);
});

test('session timing: Corrected +7 that no longer verifies falls back to Original with an explanation', () => {
  const message = 'Saved timing correction was +7 samples, but this render no longer verifies that fixed offset. ' +
    'Original timing has been restored.';
  for (const render of [timingRender(8), timingRender(null)]) {
    const sb = timingSandbox();
    sb.restoreTimingIntent({ choice: 'corrected', offsetSamples: 7, method: 'fixed-frozen-offset' });
    sb.renderTimingReadout(render);
    assert.equal(sb.choice().corrected, false);
    assert.equal(JSON.stringify(sb.timingParamsBody()), '{}');
    assert.equal(sb.note.textContent, message);
    assert.equal(sb.note.hidden, false);
  }
  // An unknown method is never trusted, even with a matching integer.
  const sb = timingSandbox();
  sb.restoreTimingIntent({ choice: 'corrected', offsetSamples: 7, method: 'something-else' });
  sb.renderTimingReadout(timingRender(7));
  assert.equal(sb.choice().corrected, false);
});

test('session timing: old sessions without a timing field behave exactly as before', () => {
  const sb = timingSandbox();
  sb.restoreTimingIntent(undefined);
  sb.renderTimingReadout(timingRender(7));
  assert.equal(sb.choice().corrected, false);
  assert.equal(sb.note.hidden, true);
  assert.equal(sb.note.textContent, '');
});

test('session timing: a source change invalidates a saved or restored correction', () => {
  const saved = { choice: 'corrected', offsetSamples: 7, method: 'fixed-frozen-offset' };
  const sb = timingSandbox();
  sb.restoreTimingIntent(saved);
  sb.ampServerPaths.b = '/m/other-b.nam';                  // a different Amp B, which happens to verify +7 too
  assert.equal(saveTiming(sb).choice, 'original');
  sb.renderTimingReadout(timingRender(7));
  assert.equal(sb.choice().corrected, false);
  assert.match(sb.note.textContent, /different Amp A\/Amp B\/DI/);

  const di = timingSandbox();
  di.restoreTimingIntent(saved);
  di.diSelector.value = 'clean_mayer.wav';
  di.renderTimingReadout(timingRender(7));
  assert.equal(di.choice().corrected, false);

  const restored = timingSandbox();                        // restored Corrected, then the source changes
  restored.restoreTimingIntent(saved);
  restored.renderTimingReadout(timingRender(7));
  assert.equal(restored.choice().corrected, true);
  restored.renderTimingReadout(null);                      // markProfileStale's reset
  assert.equal(restored.choice().corrected, false);
  assert.equal(JSON.stringify(restored.timingParamsBody()), '{}');
});

test('loading a session never renders: it only parks the timing intent', () => {
  const body = section('function applySessionSettings(', 'const sessionSettingsStatus');
  assert.doesNotMatch(body, /doRenderPair|render_pair|fetch\(/);
  assert.match(body, /markProfileStale\("Settings restored"\);\s*restoreTimingIntent\(s\.timing\);/);
  const restore = section('function restoreTimingIntent(', 'function resolveTimingRestore(');
  assert.doesNotMatch(restore, /fetch\(|scheduleUpdate|timingChoice/);
});

test('timing buttons are not transition presets', () => {
  // A plain ".preset-btn" selector made Original/Corrected reset the
  // transition width and cleared their own pressed state.
  assert.match(source, /const presetButtons = document\.querySelectorAll\("\.preset-btn\[data-value\]"\);/);
  const html = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
  for (const button of html.match(/<button[^>]*data-timing=[^>]*>/g)) assert.doesNotMatch(button, /data-value/);
});

test('startup update notice offers this platform installer, the release page, or git pull', () => {
  const sandbox = {};
  vm.createContext(sandbox);
  vm.runInContext(section('function updateNoticeContent(', 'async function checkForUpdateAtStartup('), sandbox);
  const base = { latest_version: 'v0.6.0', current_version: 'v0.5.0', release_url: 'https://r', asset_url: 'https://a.dmg' };
  const packaged = sandbox.updateNoticeContent({ ...base, is_packaged: true });
  assert.equal(packaged.href, 'https://a.dmg');
  assert.equal(packaged.action, 'Download v0.6.0');
  assert.match(packaged.title, /v0\.6\.0 is available \(you have v0\.5\.0\)/);
  const noInstaller = sandbox.updateNoticeContent({ ...base, asset_url: null, is_packaged: true });
  assert.equal(noInstaller.href, 'https://r');
  assert.equal(noInstaller.action, 'View release');
  const source = sandbox.updateNoticeContent({ ...base, is_packaged: false });
  assert.equal(source.href, 'https://r');
  assert.match(source.detail, /git pull/);
  assert.match(source.detail, /release page/);
});

test('a tone type outside the standard list is kept, not silently cleared by a metadata edit', () => {
  assert.match(source, /option\[data-from-file\]/);
  assert.match(source, /\(from the file\)/);
  const setup = source.indexOf('toneSelect.append(option)');
  const fill = source.indexOf('Object.entries(fieldMap).forEach(([key, id]) => { document.getElementById(id).value = originalToolMetadata[key]');
  assert.ok(setup > 0 && setup < fill);   // the option exists before the select value is set
});

test('AI Assistant states which AI it uses and whether it is ready; there is no on/off checkbox', () => {
  const html = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
  assert.doesNotMatch(html, /recipe-use-local-ai/);
  assert.match(html, /id="recipe-ai-status"/);
  const sandbox = {};
  vm.createContext(sandbox);
  vm.runInContext(section('const AI_PROVIDER_NAMES', 'function showRecipeAiStatus('), sandbox);
  const ready = sandbox.recipeAiStatusView({ enabled: true, provider: 'local', model: 'gemma4:e4b', reachable: true });
  assert.equal(ready.ready, true);
  assert.equal(ready.text, 'Using Local AI: gemma4:e4b. Ready.');
  assert.equal(ready.settings, false);
  const cloud = sandbox.recipeAiStatusView({ enabled: true, provider: 'cloudflare', model: '@cf/x', reachable: true });
  assert.match(cloud.text, /Cloudflare Workers AI: @cf\/x/);
  const down = sandbox.recipeAiStatusView({ enabled: true, provider: 'local', model: 'gemma4:e4b', reachable: false });
  assert.equal(down.ready, false);
  assert.equal(down.state, 'unavailable');
  assert.match(down.text, /isn't responding.*Ollama.*built-in rules/);
  const off = sandbox.recipeAiStatusView({ enabled: false, provider: 'local' });
  assert.equal(off.state, 'off');
  assert.match(off.text, /No AI set up.*built-in rules/);
  assert.equal(off.settings, true);
  assert.equal(sandbox.recipeAiStatusView(null, false).ready, false);
  // The AI is used whenever it is ready -- no separate opt-in.
  assert.match(source, /if \(localRecipeAiAvailable && prompt\.trim\(\)\) \{/);
});

test('AI status code tolerates page markup without the status elements', () => {
  assert.match(source, /recipeAiSettingsButton\?\.addEventListener\("click"/);
  const sandbox = { localRecipeAiAvailable: false, recipeAiStatus: null, recipeAiStatusText: null, recipeAiSettingsButton: null };
  vm.createContext(sandbox);
  vm.runInContext(section('function showRecipeAiStatus(', 'async function loadLocalRecipeAiStatus('), sandbox);
  sandbox.showRecipeAiStatus({ ready: true, state: 'ready', text: 'x', settings: false });   // must not throw
  assert.equal(vm.runInContext('localRecipeAiAvailable', sandbox), true);
});

test('a change made while paused is fetched when play is pressed, never played stale', () => {
  const sandbox = {
    havePair: true, lastPreviewSource: 'mix', autoAuditionToggle: { checked: true },
    liveAudition: { active: false }, player: { paused: true }, auditionRefreshTimer: null,
    setTimeout: () => 1, clearTimeout: () => {},
  };
  vm.createContext(sandbox);
  vm.runInContext(section('let auditionNeedsRefresh', '// NAM rendering remains server-side'), sandbox);
  sandbox.scheduleAuditionRefresh();                       // e.g. a cabinet added while paused
  assert.equal(vm.runInContext('auditionNeedsRefresh', sandbox), true);
  // The play handler fetches the current audio instead of replaying the old blob.
  const play = section('player.addEventListener("play", () => {', 'player.addEventListener("pause"');
  assert.match(play, /auditionNeedsRefresh && lastPreviewSource && havePair && !liveAudition\.active/);
  assert.match(play, /preview\(lastPreviewSource, \{ preservePosition: true, quiet: true, playWhenReady: true \}\)/);
  // Any fresh preview clears the flag.
  assert.match(section('async function preview(', 'const requestId = ++previewRequestId;'), /auditionNeedsRefresh = false;/);
  // Controls still never start sound by themselves: with audio playing, the normal refresh runs instead.
  vm.runInContext('auditionNeedsRefresh = false; player.paused = false;', sandbox);
  sandbox.scheduleAuditionRefresh();
  assert.equal(vm.runInContext('auditionNeedsRefresh', sandbox), false);
});

test('mode panels: plain level readout, advanced detail labelled, no controls that do nothing', () => {
  const sandbox = { fmtSigned: (v) => `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(1)}` };
  vm.createContext(sandbox);
  vm.runInContext(section('function trimReadoutText(', 'function applyModeVisibility('), sandbox);
  assert.equal(sandbox.trimReadoutText(-9.6, -9.6), 'Amp B is turned down 9.6 dB to match Amp A.');
  assert.equal(sandbox.trimReadoutText(-9.6, -8.6), 'Amp B is turned down 8.6 dB (automatic -9.6 dB, plus your +1.0 dB).');
  assert.equal(sandbox.trimReadoutText(0, 0), "Amp B's level is unchanged to match Amp A.");

  const visibility = section('function applyModeVisibility(', 'createA2Title.textContent');
  assert.match(visibility, /getElementById\("level-match-card"\)\.hidden = currentMode === "character"/);
  assert.match(visibility, /getElementById\("btn-live-blend"\)\.hidden = currentMode !== "blend"/);
  // applyModeVisibility runs at startup: it must not touch consts declared after that call.
  const firstCall = source.indexOf('\napplyModeVisibility();');
  for (const name of [...visibility.matchAll(/\b([a-zA-Z]\w+)\.(?:hidden|textContent)\s*=/g)].map((m) => m[1])) {
    const decl = source.indexOf(`\nconst ${name} =`);   // top-level declarations only
    assert.ok(decl === -1 || decl < firstCall, `${name} is declared after the startup applyModeVisibility() call`);
  }

  const html = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
  for (const summary of ['Advanced: timing between the amps', "Advanced: fine-tune Amp B's level",
    'Advanced: change the drive as you play harder', 'Advanced: training input', 'Advanced: technical evidence']) {
    assert.ok(html.includes(`<summary>${summary}</summary>`), summary);
  }
  assert.doesNotMatch(html, /byte-identical/);
  assert.doesNotMatch(source, /Drive donor trajectory|LOW-LEVEL RESPONSE|High definition training|Live mix: Always-on mix only/);
});

test('quiet-playing result leads with a plain verdict and keeps numbers under Advanced', () => {
  const sandbox = { fmtSigned: (v) => `${v}` };
  vm.createContext(sandbox);
  vm.runInContext(section('function renderLowLevelResponseHtml(', '// Mode-aware params'), sandbox);
  const check = { ok: true, levels_db: [-30, -20], output_rms_dbfs: [-40, -30], max_step_error_db: 0.4 };
  const ok = sandbox.renderLowLevelResponseHtml(check);
  assert.ok(ok.indexOf('Quiet playing works') < ok.indexOf('Advanced: measurements'));
  const bad = sandbox.renderLowLevelResponseHtml({ ...check, ok: false, max_step_error_db: 12 });
  assert.match(bad, /drops out when you play softly/);
});

test('invalidating an idle live mix says nothing', () => {
  const sandbox = { liveAudition: { active: false, requestId: 3, loadingRequest: null, stop() { this.requestId += 1; } }, liveBlendStatus: { textContent: 'before' } };
  vm.createContext(sandbox);
  vm.runInContext(section('function invalidateLiveAudition(', 'async function startLiveBlend('), sandbox);
  sandbox.invalidateLiveAudition('Design mode changed - live blend stopped.');
  assert.equal(sandbox.liveBlendStatus.textContent, 'before');
});

test('default choices: clean_mayer DI, guitar, and the vintage/PAF reference pickup', () => {
  const html = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
  assert.match(html, /\{% if f == default_di_file %\} selected\{% endif %\}/);
  const sandbox = {};
  vm.createContext(sandbox);
  vm.runInContext(section('const DEFAULT_PROFILE_BY_INSTRUMENT', 'function populateProfileSelect('), sandbox);
  const guitar = [{ id: 'vintage_single' }, { id: 'standard_single' }, { id: 'vintage_humbucker' }];
  assert.equal(sandbox.defaultProfileFor('guitar', guitar), 'vintage_humbucker');   // not simply the first entry
  assert.equal(sandbox.defaultProfileFor('bass', [{ id: 'standard_jp' }, { id: 'modern_passive' }]), 'standard_jp');
  assert.equal(sandbox.defaultProfileFor('guitar', [{ id: 'p90' }]), 'p90');         // falls back if missing
  assert.match(section('function populateProfileSelect(', '// Anything that changes'), /profileSelect\.value = defaultProfileFor\(/);
  assert.match(section('function populateWizardProfiles(', 'function updateWizardProfileDescription('), /defaultProfileFor\(wizardInstrument\.value, profiles\)/);
});

test('Finish & polish names the Compare the sound card and links to its cabinet picker', () => {
  const html = fs.readFileSync(path.join(__dirname, '../templates/index.html'), 'utf8');
  assert.doesNotMatch(html + source, /listening card/i);
  assert.match(html, /id="btn-cab-go-choose"[^>]*>Choose a cabinet in Compare the sound</);
  const link = section('document.getElementById("btn-cab-go-choose")?.addEventListener', 'cabRemoveButton.addEventListener');
  assert.match(link, /data-workflow-stage="shape"/);
  assert.match(link, /scrollIntoView/);
});

test('autosave keeps ONE record, writes only real changes, and clears once work is saved', async () => {
  const calls = [];
  let settings = { ampA: { path: null }, ampB: { path: null }, mix: '50' };
  const sandbox = {
    collectSessionSettings: () => JSON.parse(JSON.stringify(settings)),
    fetch: async (url, opts = {}) => { calls.push({ url, method: opts.method, body: opts.body && JSON.parse(opts.body) }); return { ok: true }; },
    setTimeout: () => 0, clearTimeout: () => {}, console,
  };
  vm.createContext(sandbox);
  vm.runInContext(section('const AUTOSAVE_ID', 'function restoreAutosave(') + '\nthis.autosave = autosave; this.AUTOSAVE_ID = AUTOSAVE_ID;', sandbox);
  const auto = sandbox.autosave;
  auto.baseline = auto.snapshot(); auto.ready = true;         // a fresh page

  await auto.flush();
  assert.equal(calls.length, 0);                               // unchanged fresh page: nothing written

  settings = { ...settings, ampA: { path: '/a.nam' } };
  await auto.flush(); await auto.flush();
  settings = { ...settings, mix: '70' };
  await auto.flush();
  const posts = calls.filter((c) => c.method === 'POST');
  assert.equal(posts.length, 3);
  assert.ok(posts.every((c) => c.body.id === sandbox.AUTOSAVE_ID && c.body.autosave === true));   // always the same record
  assert.ok(posts.every((c) => c.body.artifact === null));                                          // settings only
  assert.equal(new Set(posts.map((c) => c.body.id)).size, 1);

  auto.markSaved();                                            // e.g. Save / Create training files / Load
  await Promise.resolve();
  assert.ok(calls.some((c) => c.method === 'DELETE' && c.url.endsWith(`/api/sessions/${sandbox.AUTOSAVE_ID}`)));
  const before = calls.length;
  await auto.flush();
  assert.equal(calls.length, before);                          // saved state again: nothing new written
  assert.equal(auto.hasUnsavedWork(), false);
  settings = { ...settings, mix: '80' };
  assert.equal(auto.hasUnsavedWork(), true);
});

test('loading a Builder session prepares the amps when both are set', () => {
  const load = section('function loadBuilderSession(', 'async function persistActiveSession(');
  assert.match(load, /applySessionSettings\(session\.settings\);/);
  assert.match(load, /if \(ampServerPaths\.a && ampServerPaths\.b\) \{[\s\S]*renderPairBtn\.click\(\);/);
  // Load and Restore both go through it; loading over unsaved work asks first.
  assert.match(source, /if \(session\.autosave\) \{ restoreAutosave\(session\); return; \}/);
  assert.match(source, /autosave\.hasUnsavedWork\(\) && !window\.confirm\(/);
});
