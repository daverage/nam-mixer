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
