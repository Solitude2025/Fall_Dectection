// Dependency-free regression tests for the upload queue and drop handlers.
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const source = readFileSync(join(__dirname, '../backend/src/main/resources/static/app.js'), 'utf8');

function setup() {
  const elements = new Map();
  const uploads = [];
  let nextJob = 0;
  const jobs = new Map();
  function element() {
    const classes = new Set();
    return {
      textContent: '', value: '', dataset: {}, children: [], listeners: {}, disabled: false,
      classList: { add: (s) => classes.add(s), remove: (s) => classes.delete(s), contains: (s) => classes.has(s), toggle(s, value) { if (value) classes.add(s); else classes.delete(s); } },
      setAttribute() {}, getAttribute() { return null; }, removeAttribute() {},
      append(...items) { this.children.push(...items); }, replaceChildren() { this.children = []; },
      addEventListener(name, fn) { this.listeners[name] = fn; }, pause() {}, click() {},
    };
  }
  class XHR {
    constructor() { this.upload = {}; }
    open() {}
    send(data) {
      const job = { job_id: `job${++nextJob}`, filename: data.file.name, status: 'queued', progress: 0 };
      uploads.push(data.file.name); jobs.set(job.job_id, job);
      this.status = 202; this.responseText = JSON.stringify(job); this.onload();
    }
  }
  const context = vm.createContext({
    console, AbortController, XMLHttpRequest: XHR,
    FormData: class { append(key, value) { this[key] = value; } },
    document: { getElementById(id) { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); }, createElement: element, querySelector: element, querySelectorAll: () => [] },
    window: { setTimeout: () => 0, clearTimeout() {}, setInterval() {}, addEventListener() {} },
    fetch: async (url) => {
      let value;
      if (url === '/api/camera/status') value = { running: false };
      else if (url === '/api/videos') value = [...jobs.values()];
      else if (url.startsWith('/api/videos/')) value = jobs.get(url.split('/').pop());
      else if (url.includes('stats')) value = {};
      else value = [];
      return { ok: true, json: async () => value };
    },
  });
  vm.runInContext(source, context);
  return { context, elements, uploads, jobs, run: (code) => vm.runInContext(code, context) };
}
const file = (name, size = 100) => ({ name, size, lastModified: 1 });
function add(app, files) { app.context.files = files; app.run('addVideoFiles(files)'); }
async function start(app) { await app.run('uploadVideo({ preventDefault() {} })'); }
function finish(app, id, status = 'completed') {
  Object.assign(app.jobs.get(id), { status, progress: 100, has_fall: false, events: [], result_url: `/api/videos/${id}/result` });
}

test('multi-file drop switches to batch, validates files and deduplicates', () => {
  const app = setup();
  add(app, [file('a.mp4'), file('b.MOV'), file('bad.txt'), file('big.mp4', 101 * 1024 * 1024), file('empty.mp4', 0)]);
  add(app, [file('a.mp4')]);
  assert.equal(app.run('state.uploadKind'), 'batch');
  assert.equal(app.run('state.videoQueue.length'), 2);
  assert.equal(app.elements.get('videoFile').multiple, true);
});

test('drop handler imports files and clears drag styling', () => {
  const app = setup();
  const zone = app.elements.get('videoDropZone');
  let prevented = false;
  zone.classList.add('drag-over');
  zone.listeners.drop({ preventDefault() { prevented = true; }, dataTransfer: { files: [file('drop.mp4')] } });
  assert.equal(prevented, true);
  assert.equal(zone.classList.contains('drag-over'), false);
  assert.equal(app.run('state.videoQueue[0].filename'), 'drop.mp4');
});

test('batch runs serially, continues after analysis failure, even outside upload mode', async () => {
  const app = setup(); add(app, [file('a.mp4'), file('b.mp4')]);
  await start(app);
  assert.deepEqual(app.uploads, ['a.mp4']);
  await app.run('pollVideoJobs()');
  assert.deepEqual(app.uploads, ['a.mp4']);
  finish(app, 'job1', 'failed');
  await app.run('pollVideoJobs()');
  assert.deepEqual(app.uploads, ['a.mp4', 'b.mp4']);
  assert.equal(app.run('state.activeVideoItem.job.job_id'), 'job2');
  finish(app, 'job2'); await app.run('pollVideoJobs()');
  assert.equal(app.run('state.queueRunning'), false);
  assert.equal(app.run('state.activeVideoItem'), null);
  assert.equal(app.run('state.videoQueue[1].status'), 'completed');
});

test('viewing history does not steal the active queue job', async () => {
  const app = setup(); add(app, [file('a.mp4'), file('b.mp4')]); await start(app);
  app.jobs.set('history', { job_id: 'history', filename: 'old.mp4', status: 'completed', events: [], result_url: '/old.mp4' });
  await app.run("openVideoJob('history')");
  await app.run('pollVideoJobs()');
  assert.equal(app.elements.get('videoTitle').textContent, 'old.mp4');
  assert.equal(app.run('state.activeVideoItem.job.job_id'), 'job1');
  finish(app, 'job1'); await app.run('pollVideoJobs()');
  assert.deepEqual(app.uploads, ['a.mp4', 'b.mp4']);
});

test('pause finishes current video and resumes remaining files on demand', async () => {
  const app = setup(); add(app, [file('a.mp4'), file('b.mp4')]); await start(app);
  app.elements.get('pauseVideoQueue').listeners.click();
  finish(app, 'job1'); await app.run('pollVideoJobs()');
  assert.deepEqual(app.uploads, ['a.mp4']);
  assert.equal(app.elements.get('analyzeVideo').disabled, false);
  await start(app); assert.deepEqual(app.uploads, ['a.mp4', 'b.mp4']);
});

test('camera activity prevents uploading, and single mode does not overwrite an active job', async () => {
  const app = setup(); add(app, [file('a.mp4')]);
  app.run('state.camera = { running: true }'); await start(app);
  assert.equal(app.uploads.length, 0);
  app.run('state.camera.running = false'); await start(app);
  add(app, [file('replacement.mp4')]);
  assert.equal(app.run('state.videoQueue.length'), 1);
  assert.equal(app.run('state.activeVideoItem.filename'), 'a.mp4');
});

test('ambiguous upload failure pauses remaining work without automatic retry', async () => {
  const app = setup(); add(app, [file('a.mp4'), file('b.mp4')]);
  app.run("sendVideoFile = async () => { const e = new Error('connection lost'); e.pauseQueue = true; throw e; }");
  await start(app);
  assert.equal(app.run('state.queueRunning'), false);
  assert.equal(app.run('state.activeVideoItem'), null);
  assert.equal(app.run('state.videoQueue[1].status'), 'waiting');
  assert.equal(app.elements.get('analyzeVideo').disabled, false);
});

test('poll failure retains the active job and never uploads the next file', async () => {
  const app = setup(); add(app, [file('a.mp4'), file('b.mp4')]); await start(app);
  app.run("request = async () => { throw new Error('offline'); }");
  await app.run('pollVideoJobs()');
  assert.equal(app.run('state.activeVideoItem.job.job_id'), 'job1');
  assert.deepEqual(app.uploads, ['a.mp4']);
});
