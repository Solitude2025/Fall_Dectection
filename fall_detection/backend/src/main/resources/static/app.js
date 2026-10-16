const state = {
  camera: null,
  events: [],
  lastAlertEvent: null,
  audioContext: null,
  streamRetryAt: 0,
  mode: 'device',
  videoJobId: null,
  uploading: false,
  refreshing: false,
  uploadKind: 'single',
  videoQueue: [],
  queueRunning: false,
  activeVideoItem: null,
  videoPolling: false,
  queueSerial: 0,
};

const $ = (id) => document.getElementById(id);

function showToast(message) {
  const node = $('toast');
  node.textContent = message;
  node.classList.add('show');
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => node.classList.remove('show'), 2600);
}

async function request(url, options = {}, timeoutMs = 8000) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
      signal: controller.signal,
    });
  } catch (error) {
    if (error.name === 'AbortError') throw new Error(`请求超时（${timeoutMs / 1000} 秒）`);
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try { const body = JSON.parse(text); message = body.error || body.detail || body.message || text; } catch (_) {}
    throw new Error(message || `请求失败 (${response.status})`);
  }
  return response.json();
}

function formatDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', { hour12: false });
}

function beep() {
  // Browsers require a user gesture before audio can start. The start button
  // initialises this context; if polling happens before that, visual alert
  // still works and the next user gesture enables sound.
  try {
    if (!state.audioContext) state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (state.audioContext.state === 'suspended') state.audioContext.resume();
    const oscillator = state.audioContext.createOscillator();
    const gain = state.audioContext.createGain();
    oscillator.frequency.value = 740;
    oscillator.type = 'square';
    gain.gain.setValueAtTime(0.08, state.audioContext.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, state.audioContext.currentTime + 0.35);
    oscillator.connect(gain).connect(state.audioContext.destination);
    oscillator.start();
    oscillator.stop(state.audioContext.currentTime + 0.35);
  } catch (_) { /* visual alert remains available */ }
}

function connectStream(url = '/api/camera/stream') {
  if ($('stream').getAttribute('src')) return;
  document.querySelector('.video-placeholder').textContent = '正在连接实时画面…';
  // An MJPEG response never finishes, so browsers do not consistently fire
  // the normal image `load` event. Show the stream container as soon as the
  // request is opened; the error handler below will replace it if the stream
  // really fails.
  $('videoFrame').classList.add('has-stream');
  $('stream').src = `${url}?t=${Date.now()}`;
}

function disconnectStream(message = '启动摄像头后显示实时画面') {
  $('stream').removeAttribute('src');
  $('videoFrame').classList.remove('has-stream');
  document.querySelector('.video-placeholder').textContent = message;
}

function renderCamera(camera) {
  state.camera = camera;
  const online = Boolean(camera && camera.running);
  $('cameraStatus').textContent = online ? '摄像头运行中' : '摄像头未启动';
  $('cameraStatus').className = `status-pill ${online ? 'online' : 'offline'}`;
  $('latency').textContent = online && camera.latency_ms != null
    ? `输出 ${camera.output_fps ?? '—'} FPS · 姿态 ${camera.inference_fps ?? '—'} FPS`
    : '等待推理';
  $('latency').title = online
    ? `单帧推理 ${camera.latency_ms ?? '—'} ms · 画面延迟 ${camera.frame_age_ms ?? '—'} ms`
    : '';
  $('frameInfo').textContent = online
    ? `${camera.camera_width ?? '—'}×${camera.camera_height ?? '—'} · 采集 ${camera.captured_fps ?? '—'} FPS · 帧号 ${camera.frame_idx ?? '—'}`
    : '帧号：—';
  if (camera.last_error) $('latency').textContent = camera.last_error;
  if (online && state.mode === 'device' && Date.now() >= state.streamRetryAt) connectStream();
  if (!online) disconnectStream(camera.last_error || camera.error || '启动摄像头后显示实时画面');
}

function renderAlarm(camera, events) {
  const active = events.find((event) => event.status === 'active');
  const alarming = Boolean(camera && camera.alarm) || Boolean(active);
  $('alarmPanel').classList.toggle('alert', alarming);
  $('alarmOverlay').classList.toggle('hidden', !alarming);
  $('alarmTitle').textContent = alarming ? '需要确认' : '当前安全';
  $('alarmDescription').textContent = alarming ? '检测到疑似跌倒，请查看实时画面并确认现场。' : '当前没有待处理的跌倒报警。';
  $('activeEvent').classList.toggle('hidden', !active);
  if (active) {
    $('activeEventId').textContent = active.event_id;
    $('activeConfidence').textContent = `${Math.round((active.confidence || 0) * 100)}%`;
    $('activeStartTime').textContent = formatDate(active.start_time);
    $('activeExpiresAt').textContent = formatDate(active.expires_at);
    if (state.lastAlertEvent !== active.event_id) {
      state.lastAlertEvent = active.event_id;
      if (state.mode === 'device') beep();
    }
  }
}

function renderEvents(events) {
  state.events = events || [];
  const active = state.events.filter((event) => event.status === 'active').length;
  const cleared = state.events.filter((event) => event.status === 'cleared').length;
  $('activeEvents').textContent = active;
  $('clearedEvents').textContent = cleared;
  $('latestEvent').textContent = state.events[0] ? formatDate(state.events[0].start_time) : '暂无';
  const rows = state.events.slice(0, 20).map((event) => `
    <tr>
      <td>${formatDate(event.start_time)}</td>
      <td>${event.camera_id || 'laptop_camera'}</td>
      <td>${Math.round((event.confidence || 0) * 100)}%</td>
      <td><span class="state ${event.status === 'active' ? 'active' : 'cleared'}">${event.status === 'active' ? '待处理' : '已解除'}</span></td>
      <td>${event.status === 'active' ? `<button class="link-button clear-row" data-event-id="${event.event_id}">解除</button>` : '—'}</td>
    </tr>`).join('');
  $('eventRows').innerHTML = rows || '<tr><td colspan="5" class="empty">还没有报警事件</td></tr>';
  document.querySelectorAll('.clear-row').forEach((button) => button.addEventListener('click', () => clearEvent(button.dataset.eventId)));
}

function renderStats(stats) {
  $('totalEvents').textContent = stats.total ?? 0;
  const trend = stats.trend || [];
  const values = trend.map((item) => Number(item.count || 0));
  const max = Math.max(1, ...values);
  $('trendChart').innerHTML = trend.length ? trend.map((item) => `
    <div class="bar-column"><span class="bar-value">${item.count}</span><div class="bar" style="height:${Math.max(3, (item.count / max) * 145)}px"></div><span class="bar-label">${item.date.slice(5)}</span></div>`).join('') : '<p class="muted">最近 7 天还没有事件数据</p>';
}

async function refresh() {
  // The page polls once per second. Do not start another refresh while a
  // previous database request is still waiting or the requests pile up.
  if (state.refreshing) return;
  state.refreshing = true;
  // Camera status is the liveness signal for this page. Event history and
  // statistics use the database and may be temporarily slow when MySQL is
  // starting; they must not hide a healthy live camera behind “后端未连接”.
  let camera;
  try {
    camera = await request('/api/camera/status', {}, 2500);
    renderCamera(camera);
  } catch (error) {
    $('cameraStatus').textContent = '后端未连接';
    $('cameraStatus').className = 'status-pill offline';
    state.refreshing = false;
    return;
  }

  const [eventsResult, statsResult] = await Promise.allSettled([
    request('/api/events?limit=50', {}, 2500),
    request('/api/events/stats?days=7', {}, 2500),
  ]);
  let events = state.events;
  if (eventsResult.status === 'fulfilled') {
    events = eventsResult.value || [];
    renderEvents(events);
  }
  if (statsResult.status === 'fulfilled') renderStats(statsResult.value);
  renderAlarm(camera, events);
  if (eventsResult.status === 'rejected' || statsResult.status === 'rejected') {
    $('latency').title = '实时摄像头已连接，历史事件或统计暂时不可用';
  }
  state.refreshing = false;
}

async function startCamera() {
  try {
    if (state.activeVideoItem || state.queueRunning) throw new Error('视频检测队列正在运行，请先暂停并等待当前视频完成');
    if (!state.audioContext) state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const result = await request('/api/camera/start', { method: 'POST' });
    renderCamera(result);
    if (result.last_error || result.running === false) {
      throw new Error(result.last_error || result.error || '摄像头未能启动');
    }
    state.streamRetryAt = 0;
    connectStream();
    showToast('摄像头已启动');
  } catch (error) { showToast(`启动失败：${error.message}`); }
  refresh();
}

async function stopCamera() {
  try { await request('/api/camera/stop', { method: 'POST' }); disconnectStream(); showToast('摄像头已停止'); } catch (error) { showToast(`停止失败：${error.message}`); }
  refresh();
}

async function clearEvent(eventId) {
  try { await request(`/api/events/${encodeURIComponent(eventId)}/clear?reason=manual`, { method: 'POST' }); showToast('报警已人工解除'); refresh(); } catch (error) { showToast(`解除失败：${error.message}`); }
}

$('stream').addEventListener('load', () => {
  $('videoFrame').classList.add('has-stream');
});
$('stream').addEventListener('error', () => {
  if (!$('stream').getAttribute('src')) return;
  state.streamRetryAt = Date.now() + 3000;
  disconnectStream('实时画面连接失败，正在重试…');
});
$('startCamera').addEventListener('click', startCamera);
$('stopCamera').addEventListener('click', stopCamera);
$('refresh').addEventListener('click', refresh);
$('clearAlarm').addEventListener('click', () => { const active = state.events.find((event) => event.status === 'active'); if (active) clearEvent(active.event_id); });
$('clearVideoHistory').addEventListener('click', clearVideoHistory);

function switchMode(mode) {
  state.mode = mode;
  const device = mode === 'device';
  $('deviceMode').classList.toggle('hidden', !device);
  $('deviceActions').classList.toggle('hidden', !device);
  $('uploadMode').classList.toggle('hidden', device);
  $('deviceTab').classList.toggle('selected', device);
  $('uploadTab').classList.toggle('selected', !device);
  $('deviceTab').setAttribute('aria-pressed', String(device));
  $('uploadTab').setAttribute('aria-pressed', String(!device));
  if (!device) { disconnectStream(); loadVideoHistory(); }
  else refresh();
}

function mediaTime(value) {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

const jobLabels = { uploading: '正在上传', queued: '等待分析', processing: '正在分析', encoding: '正在生成回放', completed: '分析完成', failed: '分析失败' };

function renderVideoJob(job) {
  $('videoTitle').textContent = job.filename;
  $('videoDuration').textContent = job.duration_seconds == null ? '—' : mediaTime(job.duration_seconds);
  $('videoCount').textContent = job.event_count ?? '—';
  const done = job.status === 'completed';
  const failed = job.status === 'failed';
  const warnings = job.warnings || [];
  $('videoResult').textContent = done ? (job.has_fall ? '检测到疑似跌倒' : warnings.length ? '未触发报警 · 需复核' : '未触发报警') : jobLabels[job.status] || job.status;
  $('videoResult').className = `status-pill ${done && job.has_fall ? 'state active' : 'offline'}`;
  $('videoWarnings').textContent = warnings.join(' ');
  $('videoWarnings').classList.toggle('hidden', !done || !warnings.length);
  if (!state.activeVideoItem) {
    $('videoProgress').classList.toggle('hidden', done || failed);
    $('videoProgress').value = job.progress || 0;
    $('uploadMessage').textContent = failed ? job.error : done ? (job.decision_policy === 'observed_transition' ? '分析完成，可回放检测画面。' : '这是旧版本分析结果。使用新版判断请重新上传视频。') : `${jobLabels[job.status] || '处理中'} ${job.progress || 0}%`;
  }
  $('videoPlayback').classList.toggle('hidden', !done);
  $('replayPlaceholder').classList.toggle('hidden', done);
  $('replayStatus').textContent = failed ? job.error : '正在分析，完成后会在这里显示回放。';
  if (!done) $('videoPlayback').pause();
  if (done && $('videoPlayback').dataset.jobId !== job.job_id) {
    $('videoPlayback').src = job.result_url;
    $('videoPlayback').dataset.jobId = job.job_id;
  }
  $('videoEventList').replaceChildren();
  if (done) (job.events || []).forEach((event, index) => {
    const button = document.createElement('button');
    button.className = 'video-event';
    button.textContent = `事件 ${index + 1} · ${mediaTime(event.video_time_seconds)} · 模型分数 ${Math.round(event.confidence * 100)}% · 点击回放`;
    button.addEventListener('click', () => {
      $('videoPlayback').currentTime = Math.max(0, event.video_time_seconds - 2);
      $('videoPlayback').play().catch(() => {});
    });
    $('videoEventList').append(button);
  });
}

async function loadVideoHistory() {
  try {
    const jobs = await request('/api/videos');
    $('videoHistory').replaceChildren();
    if (!jobs.length) { $('videoHistory').textContent = '暂无上传记录'; return; }
    jobs.forEach((job) => {
      const row = document.createElement('div');
      row.className = 'history-item';
      const info = document.createElement('button');
      info.className = 'history-item-info';
      const title = document.createElement('strong'); title.textContent = job.filename;
      const description = document.createElement('span'); description.textContent = `${jobLabels[job.status] || job.status} · ${job.event_count || 0} 个事件 · ${formatDate(job.created_at)}`;
      info.append(title, description);
      info.addEventListener('click', async () => {
        try { await openVideoJob(job.job_id); }
        catch (error) { showToast(error.message); }
      });
      const remove = document.createElement('button');
      remove.className = 'button button-small button-danger history-delete';
      remove.type = 'button';
      remove.textContent = '删除';
      remove.addEventListener('click', async () => {
        if (!window.confirm(`确定删除“${job.filename}”这条分析历史吗？`)) return;
        try {
          await request(`/api/videos/${encodeURIComponent(job.job_id)}`, { method: 'DELETE' });
          if (state.videoJobId === job.job_id) {
            state.videoJobId = null;
            $('videoPlayback').removeAttribute('src');
            $('videoPlayback').load();
          }
          showToast('已删除这条分析历史');
          await loadVideoHistory();
        } catch (error) { showToast(`删除失败：${error.message}`); }
      });
      row.append(info, remove);
      $('videoHistory').append(row);
    });
  } catch (error) { $('videoHistory').textContent = error.message; }
}

async function clearVideoHistory() {
  try {
    const jobs = await request('/api/videos');
    if (!jobs.length) { showToast('当前没有可清除的历史'); return; }
    if (!window.confirm('确定清除全部已完成或失败的视频分析历史吗？正在分析的任务会保留。')) return;
    const result = await request('/api/videos', { method: 'DELETE' });
    state.videoJobId = null;
    $('videoPlayback').removeAttribute('src');
    $('videoPlayback').load();
    showToast(result.skipped_active ? `已清除 ${result.deleted} 条，正在分析的任务已保留` : `已清除 ${result.deleted} 条历史`);
    await loadVideoHistory();
  } catch (error) { showToast(`清除失败：${error.message}`); }
}

function setUploadKind(kind) {
  if (kind === 'single' && (state.activeVideoItem || state.queueRunning || state.videoQueue.some((item) => item.status === 'waiting'))) {
    showToast('请先完成或移除待检测任务，再切换到单视频分析');
    return;
  }
  state.uploadKind = kind;
  const batch = kind === 'batch';
  $('videoFile').multiple = batch;
  $('fileLabelText').textContent = batch ? '选择多个视频（可继续添加）' : '选择视频';
  $('batchHelp').classList.toggle('hidden', !batch);
  $('dropHint').textContent = batch ? '将视频拖到此处可继续添加到队列，支持多个文件。选好后点击“开始批量检测”。' : '将视频拖到此处，选好后点击“上传并分析”；拖入多个文件会自动切换为批量检测。';
  for (const [id, selected] of [['singleVideoMode', !batch], ['batchVideoMode', batch]]) {
    $(id).classList.toggle('selected', selected);
    $(id).setAttribute('aria-pressed', String(selected));
  }
  renderVideoQueue();
}

function addVideoFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  // Never silently discard extra files dropped in single-video mode.
  if (files.length > 1) setUploadKind('batch');
  if (state.uploadKind === 'single' && (state.activeVideoItem || state.queueRunning)) {
    showToast('当前视频正在检测，请等待完成，或切换到批量导入检测');
    return;
  }
  if (state.uploadKind === 'single') state.videoQueue = [];
  let added = 0;
  const errors = [];
  files.forEach((file) => {
    if (!/\.(mp4|avi|mov|mkv|webm)$/i.test(file.name)) { errors.push(`${file.name}：不支持的视频格式`); return; }
    if (!file.size) { errors.push(`${file.name}：文件为空`); return; }
    if (file.size > 100 * 1024 * 1024) { errors.push(`${file.name}：不能超过 100 MB`); return; }
    const duplicate = state.videoQueue.some((item) => item.file && item.file.name === file.name && item.file.size === file.size && item.file.lastModified === file.lastModified);
    if (duplicate) return;
    state.videoQueue.push({ id: ++state.queueSerial, filename: file.name, file, status: 'waiting', progress: 0, job: null, error: null });
    added++;
  });
  $('videoFile').value = '';
  renderVideoQueue();
  if (errors.length) showToast(errors.join('；'));
  if (added) $('uploadMessage').textContent = `已添加 ${added} 个视频。${state.queueRunning ? '将按顺序继续检测。' : '点击开始按钮进行检测。'}`;
  if (state.queueRunning && !state.activeVideoItem) processVideoQueue();
}

function renderVideoQueue() {
  const items = state.videoQueue;
  const waiting = items.filter((item) => item.status === 'waiting').length;
  const completed = items.filter((item) => item.status === 'completed').length;
  const failed = items.filter((item) => item.status === 'failed').length;
  $('batchQueuePanel').classList.toggle('hidden', !items.length);
  $('batchSummary').textContent = `共 ${items.length} 个 · 完成 ${completed} · 失败 ${failed} · 待检测 ${waiting}`;
  $('analyzeVideo').disabled = !waiting || state.queueRunning || Boolean(state.activeVideoItem);
  $('analyzeVideo').textContent = state.queueRunning ? '队列检测中…' : state.uploadKind === 'batch' ? '开始批量检测' : '上传并分析';
  $('pauseVideoQueue').classList.toggle('hidden', !state.queueRunning);
  $('batchQueue').replaceChildren();
  items.forEach((item) => {
    const row = document.createElement('li'); row.className = 'batch-item'; row.dataset.status = item.status;
    const info = document.createElement('div'); info.className = 'batch-item-info';
    const name = document.createElement('strong'); name.textContent = item.filename;
    const detail = document.createElement('span');
    detail.textContent = item.status === 'waiting' ? '等待检测' : item.status === 'failed' ? `失败：${item.error || item.job?.error || '分析失败'}` : item.status === 'completed' ? `${item.job.has_fall ? '检测到疑似跌倒' : '未触发报警'} · ${item.job.event_count || 0} 个事件${item.job.warnings?.length ? ' · 需人工复核' : ''}` : `${jobLabels[item.status] || item.status} ${item.progress || 0}%`;
    info.append(name, detail);
    const actions = document.createElement('div'); actions.className = 'batch-item-actions';
    if (item.job && ['completed', 'failed'].includes(item.status)) {
      const view = document.createElement('button'); view.type = 'button'; view.className = 'link-button'; view.textContent = '查看结果';
      view.addEventListener('click', () => openVideoJob(item.job.job_id).catch((error) => showToast(error.message)));
      actions.append(view);
    }
    if (item.status === 'waiting') {
      const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'link-button'; remove.textContent = '移除';
      remove.setAttribute('aria-label', `移除 ${item.filename}`);
      remove.addEventListener('click', () => { state.videoQueue = state.videoQueue.filter((entry) => entry !== item); renderVideoQueue(); });
      actions.append(remove);
    }
    row.append(info, actions); $('batchQueue').append(row);
  });
}

async function openVideoJob(jobId) {
  const job = await request(`/api/videos/${encodeURIComponent(jobId)}`);
  state.videoJobId = jobId;
  renderVideoJob(job);
}

function sendVideoFile(item) {
  return new Promise((resolve, reject) => {
    const data = new FormData(); data.append('file', item.file);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/videos');
    xhr.timeout = 190000;
    xhr.upload.onprogress = (progress) => {
      if (!progress.lengthComputable) return;
      item.progress = Math.round(progress.loaded / progress.total * 100);
      $('videoProgress').value = item.progress;
      $('uploadMessage').textContent = `正在上传 ${item.filename}：${item.progress}%`;
      renderVideoQueue();
    };
    xhr.onload = () => {
      let body; try { body = JSON.parse(xhr.responseText); } catch (_) { body = {}; }
      if (xhr.status >= 200 && xhr.status < 300 && body.job_id) { resolve(body); return; }
      const error = new Error(body.error || body.detail || body.message || '上传失败，请检查视频和服务状态');
      error.pauseQueue = xhr.status === 409 || xhr.status >= 500 || (xhr.status >= 200 && xhr.status < 300);
      reject(error);
    };
    const interrupted = () => {
      const error = new Error('上传连接中断或超时，请先在最近视频分析中确认是否已接收；队列已暂停');
      error.pauseQueue = true; reject(error);
    };
    xhr.onerror = interrupted; xhr.ontimeout = interrupted; xhr.onabort = interrupted;
    xhr.send(data);
  });
}

async function uploadVideo(event) {
  event.preventDefault();
  if (state.queueRunning || state.activeVideoItem) return;
  if (state.camera?.running) { showToast('请先切换到设备模式，停止摄像头后再上传'); return; }
  state.queueRunning = true;
  renderVideoQueue();
  await processVideoQueue();
}

async function processVideoQueue() {
  if (!state.queueRunning || state.activeVideoItem) return;
  const item = state.videoQueue.find((entry) => entry.status === 'waiting');
  if (!item) {
    state.queueRunning = false; renderVideoQueue();
    $('videoProgress').classList.add('hidden');
    $('uploadMessage').textContent = '本次队列已处理完毕，可逐条查看结果。';
    return;
  }
  state.activeVideoItem = item;
  state.uploading = true;
  item.status = 'uploading';
  $('videoProgress').classList.remove('hidden'); $('videoProgress').value = 0;
  $('uploadMessage').textContent = `正在上传 ${item.filename}…`;
  renderVideoQueue();
  try {
    const job = await sendVideoFile(item);
    item.job = job; item.status = job.status; item.progress = job.progress || 0;
    item.file = null; // release the local file once the server has accepted it
    state.videoJobId = job.job_id;
    renderVideoJob(job);
    loadVideoHistory();
  } catch (error) {
    item.status = 'failed'; item.error = error.message; item.file = null;
    state.activeVideoItem = null;
    if (error.pauseQueue) state.queueRunning = false;
    $('uploadMessage').textContent = `${error.message}${error.pauseQueue ? '；检查服务后点击开始，继续剩余任务。' : ''}`;
    showToast(error.message);
  } finally {
    state.uploading = false; renderVideoQueue();
  }
  if (!state.activeVideoItem) {
    if (state.queueRunning) await processVideoQueue();
    else $('videoProgress').classList.add('hidden');
  }
}

async function pollVideoJobs() {
  // Poll the active queue item independently of whichever historical replay
  // is being viewed, and keep the queue moving even in device mode.
  if (state.videoPolling || state.uploading) return;
  const item = state.activeVideoItem;
  const jobId = item?.job?.job_id || state.videoJobId;
  if (!jobId) return;
  state.videoPolling = true;
  try {
    const job = await request(`/api/videos/${encodeURIComponent(jobId)}`);
    if (item) {
      item.job = job; item.status = job.status; item.progress = job.progress || 0;
      $('videoProgress').value = item.progress;
      $('uploadMessage').textContent = `${item.filename}：${jobLabels[job.status] || job.status} ${item.progress}%`;
      renderVideoQueue();
    }
    if (state.videoJobId === jobId) renderVideoJob(job);
    if (['completed', 'failed'].includes(job.status)) {
      if (item) {
        state.activeVideoItem = null;
        if (state.queueRunning) await processVideoQueue();
        else {
          $('videoProgress').classList.add('hidden'); renderVideoQueue();
          $('uploadMessage').textContent = '当前视频已结束，队列已暂停。点击开始可继续剩余任务。';
        }
      }
      // Do not clear a newly selected/started job during an awaited upload.
      if (state.videoJobId === jobId) state.videoJobId = null;
      loadVideoHistory();
    }
  } catch (error) {
    $('uploadMessage').textContent = `状态查询失败：${error.message}。正在重试，不会重复上传。`;
  } finally {
    state.videoPolling = false;
  }
}

$('singleVideoMode').addEventListener('click', () => setUploadKind('single'));
$('batchVideoMode').addEventListener('click', () => setUploadKind('batch'));
$('videoFile').addEventListener('change', (event) => addVideoFiles(event.target.files));
$('chooseDroppedVideo').addEventListener('click', () => $('videoFile').click());
$('pauseVideoQueue').addEventListener('click', () => {
  state.queueRunning = false; renderVideoQueue();
  $('uploadMessage').textContent = '当前视频将继续完成，后续任务已暂停。';
});

let videoDragDepth = 0;
const isFileDrag = (event) => Array.from(event.dataTransfer?.types || []).includes('Files');
$('videoDropZone').addEventListener('dragenter', (event) => {
  if (!isFileDrag(event)) return;
  event.preventDefault(); videoDragDepth++; $('videoDropZone').classList.add('drag-over');
});
$('videoDropZone').addEventListener('dragover', (event) => {
  if (!isFileDrag(event)) return;
  event.preventDefault(); event.dataTransfer.dropEffect = 'copy';
});
$('videoDropZone').addEventListener('dragleave', (event) => {
  event.preventDefault(); videoDragDepth = Math.max(0, videoDragDepth - 1);
  if (!videoDragDepth) $('videoDropZone').classList.remove('drag-over');
});
$('videoDropZone').addEventListener('drop', (event) => {
  event.preventDefault(); videoDragDepth = 0; $('videoDropZone').classList.remove('drag-over');
  addVideoFiles(event.dataTransfer.files);
});
// Prevent a misplaced file drop from navigating away and losing the queue.
window.addEventListener('dragover', (event) => { if (isFileDrag(event)) event.preventDefault(); });
window.addEventListener('drop', (event) => { if (isFileDrag(event)) event.preventDefault(); });
window.addEventListener('beforeunload', (event) => {
  if (state.activeVideoItem || state.videoQueue.some((item) => item.status === 'waiting')) {
    event.preventDefault(); event.returnValue = '';
  }
});

$('deviceTab').addEventListener('click', () => switchMode('device'));
$('uploadTab').addEventListener('click', () => switchMode('upload'));
$('uploadForm').addEventListener('submit', uploadVideo);
renderVideoQueue();
window.setInterval(pollVideoJobs, 1000);
refresh();
window.setInterval(refresh, 1000);
