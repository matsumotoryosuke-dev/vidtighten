import { S } from './state.js';
import { $, fmt, fmtDur, fmtElapsed, plur, toast, api, apiFetch, escH, EVT_FILE_LOADED, EVT_RESULT_APPLIED, EVT_STATE_RESET } from './utils.js';
import { cancelPreviewDebounce } from './settings.js';
import { missingCoreDeps, coreDepBlockerMessage, degradedMessages, formatDegradedToast } from './preflight.js';

let _onRender         = () => {};
let _onRestoreSession = () => {};

// ── VBR MP3 → M4A seamless swap ─────────────────────────────────────────────
// When the backend transcodes a VBR MP3 to M4A, we swap the player source so
// subsequent playback uses the sample-accurate M4A timestamps.  The swap
// preserves currentTime and paused state so the user doesn't notice the switch.
let _m4aPollTimer  = null;
let _m4aTargetPath = null;   // path currently being polled

function _stopM4aPoll() {
  if (_m4aPollTimer) { clearTimeout(_m4aPollTimer); _m4aPollTimer = null; }
  _m4aTargetPath = null;
}

function _swapToM4a(path) {
  const player = $('player');
  if (!player || S.filePath !== path) return;
  const ct     = player.currentTime;
  const paused = player.paused;
  // Re-fetch the same URL — backend now serves M4A; append _v=m4a as cache-bust.
  player.src = `/api/stream?path=${encodeURIComponent(path)}&_v=m4a`;
  player.addEventListener('canplay', () => {
    player.currentTime = ct;
    if (!paused) player.play().catch(() => {});
  }, { once: true });
}

async function _pollM4aStatus(path, attempt) {
  _m4aPollTimer = null;
  if (S.filePath !== path) return;   // user loaded a different file
  try {
    const r = await (await fetch(`/api/media/transcode_status?path=${encodeURIComponent(path)}`)).json();
    if (r.status === 'ready') {
      _swapToM4a(path);
      return;
    }
    if (r.status === 'failed' || r.status === 'not_applicable') return;
    // pending or not_started — keep polling (up to 2 minutes)
    if (attempt < 24) {
      _m4aTargetPath = path;
      _m4aPollTimer  = setTimeout(() => _pollM4aStatus(path, attempt + 1), 5000);
    }
  } catch (_e) { /* network error — stop polling */ }
}

function _startM4aPoll(path) {
  _stopM4aPoll();
  if (!path.toLowerCase().endsWith('.mp3')) return;
  _m4aTargetPath = path;
  // First check after 3 s; for cached files from previous runs this fires "ready" immediately
  _m4aPollTimer  = setTimeout(() => _pollM4aStatus(path, 0), 3000);
}

// ── High-res video → low-res preview proxy seamless swap ────────────────────
// Mirrors the M4A swap above: the backend background-transcodes a small,
// fast-decode proxy for high-resolution footage (e.g. 4K 10-bit 4:2:2 HEVC,
// which frequently has no hardware decode path and plays extremely laggy).
// Only the <video> preview source is ever swapped — audio analysis and export
// always use the original file.
let _proxyPollTimer  = null;
let _proxyTargetPath = null;
let _proxyWaitStarted = null;   // Date.now() when the pending-banner was shown, or null

function _stopProxyPoll() {
  if (_proxyPollTimer) { clearTimeout(_proxyPollTimer); _proxyPollTimer = null; }
  _proxyTargetPath  = null;
  _proxyWaitStarted = null;
}

function _clearProxyPendingUI() {
  $('video-panel')?.classList.remove('proxy-pending');
}

function _swapToProxy(path) {
  const player = $('player');
  if (!player || S.filePath !== path) return;
  const wasPending = $('video-panel')?.classList.contains('proxy-pending');
  const ct     = player.currentTime;
  const paused = player.paused;
  player.src = `/api/stream?path=${encodeURIComponent(path)}&_v=proxy`;
  player.addEventListener('canplay', () => {
    player.currentTime = ct;
    if (!paused) player.play().catch(() => {});
  }, { once: true });
  _clearProxyPendingUI();
  // Only announce a "switch" when a previous (original) source was actually
  // playing — for the pending-banner path this is the FIRST successful load,
  // not a swap, so a "switched to proxy" toast would be confusing.
  if (!wasPending) toast('Switched to a low-res preview proxy for smoother playback', 3000);
}

function _updateProxyElapsedLabel() {
  const el = $('proxy-pending-elapsed');
  if (!el || !_proxyWaitStarted) return;
  const secs = Math.floor((Date.now() - _proxyWaitStarted) / 1000);
  el.textContent = secs >= 60
    ? `${Math.floor(secs / 60)}m ${secs % 60}s elapsed — `
    : `${secs}s elapsed — `;
}

async function _pollProxyStatus(path, attempt) {
  _proxyPollTimer = null;
  if (S.filePath !== path) return;   // user loaded a different file
  _updateProxyElapsedLabel();
  try {
    const r = await (await fetch(`/api/media/proxy_status?path=${encodeURIComponent(path)}`)).json();
    if (r.status === 'ready') {
      _swapToProxy(path);
      return;
    }
    if (r.status === 'failed') {
      // Proxy generation failed — nothing will ever become playable via the
      // normal path. Surface this clearly rather than waiting forever.
      _clearProxyPendingUI();
      toast('Could not generate a preview for this file — video playback is unavailable, but analysis and export are unaffected.', 8000);
      return;
    }
    if (r.status === 'not_applicable') { _clearProxyPendingUI(); return; }
    // pending or not_started — keep polling (up to ~10 minutes; proxy transcode
    // of a long 4K file can take a while since source decode is itself slow)
    if (attempt < 120) {
      _proxyTargetPath = path;
      _proxyPollTimer  = setTimeout(() => _pollProxyStatus(path, attempt + 1), 5000);
    } else {
      _clearProxyPendingUI();
      toast('Preview generation is taking unusually long — video playback may be unavailable this session.', 8000);
    }
  } catch (_e) { /* network error — stop polling */ }
}

function _startProxyPoll(path) {
  _stopProxyPoll();
  _proxyTargetPath  = path;
  _proxyWaitStarted = Date.now();
  _updateProxyElapsedLabel();
  // First check after 3 s; for cached files from previous runs this fires "ready" immediately
  _proxyPollTimer  = setTimeout(() => _pollProxyStatus(path, 0), 3000);
}

function _showStatHint(html) {
  const hint = $('stat-hint');
  if (hint) { hint.innerHTML = html; hint.style.display = ''; }
}

export function initAnalysis({ onRender, onRestoreSession }) {
  _onRender         = onRender;
  _onRestoreSession = onRestoreSession;
}

// First-run dependency preflight — DOM wiring. Pure decision logic
// (missingCoreDeps/coreDepBlockerMessage/degradedMessages/formatDegradedToast)
// lives in preflight.js and is unit tested there; this function is verified
// manually per the project's convention for DOM-dependent code (see
// tests/js/preflight.test.js and tx-search.test.js for the same split).
export async function checkCaps() {
  let d;
  try {
    d = await (await fetch('/api/capabilities')).json();
  } catch {
    return;   // backend unreachable — nothing to report
  }

  S.whisperAvailable = d.whisper_available;
  const badge = $('whisper-badge');
  if (S.whisperAvailable) {
    badge.textContent = '✓ ready';
    badge.className = 'whisper-badge ok';
    $('f-enable').disabled = false;
  } else {
    $('f-enable').disabled = true;
  }

  // CRITICAL: ffmpeg/ffprobe missing → the app can't function. Shown
  // immediately, independent of the (network-bound) Ollama check below.
  const missing = missingCoreDeps(d);
  if (missing.length) {
    const overlay = $('dep-blocker-overlay');
    const msgEl   = $('dep-blocker-msg');
    if (overlay && msgEl) {
      msgEl.textContent = coreDepBlockerMessage(missing);
      overlay.classList.add('open');
    }
  }

  // DEGRADED: whisper/whisperx/japanese are already known synchronously
  // from /api/capabilities above. Ollama is checked separately — it needs a
  // live network call to the local daemon (see web.py:api_capabilities'
  // docstring) — via the existing /api/llm/models endpoint, so a slow/absent
  // Ollama can only delay this non-blocking toast, never the hard blocker.
  let ollamaAvailable;
  try {
    const llm = await (await fetch('/api/llm/models')).json();
    ollamaAvailable = llm.available;
  } catch { /* unknown — omitted below rather than guessed at */ }

  const toastMsg = formatDegradedToast(degradedMessages(d, ollamaAvailable));
  if (toastMsg) toast(toastMsg, 8000);
}

// Convert a text/uri-list entry (file:///…) to a local path string, or null.
function _uriToLocalPath(uriList) {
  const first = uriList.trim().split(/\r?\n/)[0].trim();
  if (!first.startsWith('file://')) return null;
  try { return decodeURIComponent(new URL(first).pathname); } catch { return null; }
}

// handleFileDrop — path resolution layers for pywebview / WKWebView / browser:
//   1. f.path          — Electron-style webviews set this on File objects
//   2. text/uri-list   — WKWebView may expose file:// URIs for Finder drags
//   3. pywebview bridge— get_dropped_paths() reads NSPasteboard via Python API
//   4. pywebview filepicker — auto-open native dialog (WKWebView from HTTP origin
//      blocks local file URIs in dataTransfer; js_api may not be registered)
//   5. Upload fallback — for real browsers (size-guarded)
export async function handleFileDrop(f, dataTransfer) {
  if (!f) return;

  // Layer 1: Electron / some CEF-based webviews
  if (f.path) { loadFilePath(f.path); return; }

  // Layer 2: WKWebView exposes text/uri-list for Finder drags
  if (dataTransfer) {
    const path = _uriToLocalPath(dataTransfer.getData('text/uri-list') || '');
    if (path) { loadFilePath(path); return; }
  }

  // Layer 3: pywebview native bridge — get_dropped_paths() reads NSPasteboard.
  if (window.pywebview?.api?.get_dropped_paths) {
    try {
      const paths = await window.pywebview.api.get_dropped_paths();
      if (paths?.length) { loadFilePath(paths[0]); return; }
    } catch {}
  }

  // Layer 4: pywebview present but get_dropped_paths unavailable (old build).
  // Don't open a surprise dialog — just tell the user to use Open File.
  if (window.pywebview) {
    toast('Could not read dropped file path. Use the Open File button instead.', 4000);
    return;
  }

  // Layer 5: browser upload fallback (size-guarded)
  const GB = 1024 ** 3;
  if (f.size > 4 * GB) {
    toast('File too large to upload. Use the Open File button so VidTighten reads it by path without copying it.'); return;
  }
  const fd = new FormData(); fd.append('file', f);
  const r = await (await fetch('/api/upload', {method:'POST',body:fd})).json();
  if (r.path) loadFilePath(r.path);
  else toast('Upload failed: ' + (r.error || 'unknown'));
}

export async function loadFilePath(path) {
  _stopM4aPoll();     // cancel any pending poll for the previous file
  _stopProxyPoll();
  const r = await api('/api/probe', {path});
  if (r.error) { toast('Error: ' + r.error); return; }
  S.filePath = path;
  S.media    = r.media;

  const name = path.split('/').pop();
  $('file-name').textContent     = name;
  $('file-duration').textContent = fmt(r.media.duration);
  $('dropzone').classList.add('hidden');
  // Show audio-only banner when there is no video track
  const videoPanel = $('video-panel');
  if (videoPanel) videoPanel.classList.toggle('audio-only', !r.media.has_video);
  videoPanel?.classList.remove('proxy-pending');   // reset from any previous file
  const player = $('player');
  player.style.display = 'block';

  // For VBR MP3: if the M4A sidecar is already cached, use it directly so
  // player.currentTime is sample-accurate from the very first request.
  // If not yet ready, set src to the original and start polling — the swap
  // happens transparently when the sidecar finishes transcoding (~30 s).
  // (A VBR MP3 without its M4A sidecar is still fully PLAYABLE — just with
  // imprecise seeking — so serving it as an interim source is safe.)
  let streamUrl = `/api/stream?path=${encodeURIComponent(path)}`;
  let skipInitialSrc = false;
  if (path.toLowerCase().endsWith('.mp3')) {
    const m4aSt = await fetch(`/api/media/transcode_status?path=${encodeURIComponent(path)}`)
      .then(r => r.json()).catch(() => ({ status: 'unknown' }));
    if (m4aSt.status === 'ready') {
      streamUrl = `/api/stream?path=${encodeURIComponent(path)}&_v=m4a`;
    } else {
      _startM4aPoll(path);   // polls until ready, then swaps src seamlessly
    }
  } else if (r.media.has_video) {
    // High-res video: if a preview proxy is already cached, use it immediately
    // so playback is smooth from the first frame. If not yet ready, DO NOT
    // fall back to the original as an interim source — some codecs/profiles
    // (10-bit 4:2:2 HEVC and similar) have no decode path in WKWebView at all,
    // so "interim playback" can black-screen with a hard error instead of
    // degrading gracefully. Show the pending banner and wait for the proxy.
    const proxySt = await fetch(`/api/media/proxy_status?path=${encodeURIComponent(path)}`)
      .then(r => r.json()).catch(() => ({ status: 'unknown' }));
    if (proxySt.status === 'ready') {
      streamUrl = `/api/stream?path=${encodeURIComponent(path)}&_v=proxy`;
    } else if (proxySt.status !== 'not_applicable') {
      skipInitialSrc = true;
      videoPanel?.classList.add('proxy-pending');
      _startProxyPoll(path);   // polls until ready, then sets src seamlessly
    }
  }
  if (!skipInitialSrc) player.src = streamUrl;
  $('btn-export').disabled        = false;
  $('wf-playhead').style.display = '';

  document.dispatchEvent(new CustomEvent(EVT_FILE_LOADED));

  // Try restoring session; if none, auto-analyze
  let sr;
  try { sr = await api('/api/session/load', {path}); }
  catch (e) { toast('Session load error: ' + e.message, 8000); sr = { session: null }; }

  try {
    if (sr.session) {
      _onRestoreSession(sr.session);
      $('btn-analyze').disabled  = false;
      toast('Session restored');
      // Warm the audio cache in the background
      api('/api/analyze/redetect_silence', {
        path, threshold_db: parseFloat($('s-threshold').value),
        min_duration: parseFloat($('s-min-dur').value),
        hangover_ms: parseInt($('s-hangover').value),
      }).catch(() => {});
    } else {
      resetState();
      await startAnalysis();
    }
  } catch (e) {
    toast('Load error: ' + e.message, 10000);
    console.error('loadFilePath post-session error:', e);
  }
}

export function resetState() {
  cancelPreviewDebounce();
  S.silPreview = null;
  $('player')?.pause();
  S.waveformData      = [];
  S.waveformThreshold = 0;
  S.waveformMaxAmp    = 0;
  S.removalCandidates = [];
  S.telopEntries      = [];
  S.words             = [];
  S.removalsEnabled   = new Set();
  S.telopsEnabled     = new Set();
  S.history = []; S.historyIndex = -1;
  // Clear any stale error/hint from a previous file before the new analysis result arrives.
  const hint = $('stat-hint');
  if (hint) { hint.textContent = ''; hint.style.display = 'none'; }
  _onRender();
  // Notify subscribers (e.g. telop overlay) to clear any caches keyed on the old entries.
  document.dispatchEvent(new CustomEvent(EVT_STATE_RESET));
}

export async function startAnalysis() {
  if (!S.filePath) { toast('No file loaded', 3000); return; }
  try {
  stopPolling();
  const params = {
    path:             S.filePath,
    threshold_db:     parseFloat($('s-threshold').value),
    min_duration:     parseFloat($('s-min-dur').value),
    hangover_ms:      parseInt($('s-hangover').value),
    padding_ms:       parseInt($('s-padding').value),
    use_whisper:      $('f-enable').checked && S.whisperAvailable,
    fillers_japanese: $('f-japanese').checked,
    fillers_english:  $('f-english').checked,
    fillers_custom:   $('f-custom').value.split(',').map(s=>s.trim()).filter(Boolean),
    whisper_model:    $('f-model').value,
  };
  setAnalyzing(true);
  const r = await api('/api/analyze/start', params);
  if (r.error) { toast('Error: '+r.error); setAnalyzing(false); return; }
  S.taskId = r.task_id;
  S.polling = setInterval(pollStatus, 600);
  } catch (e) {
    toast('Analysis start error: ' + e.message, 10000);
    console.error('startAnalysis error:', e);
    setAnalyzing(false);
  }
}

async function pollStatus() {
  if (!S.taskId) return;
  try {
    const d = await (await fetch(`/api/analyze/status/${S.taskId}`)).json();
    $('progress-fill').style.width = (d.progress||0) + '%';

    // Build a label: stage + elapsed time + ETA for long Whisper runs
    let label = (d.stage||'').replace(/_/g,' ');
    const elapsed = d.created_at ? (Date.now() / 1000 - d.created_at) : 0;
    if (elapsed >= 5) {
      label += ` · ${fmtElapsed(elapsed)}`;
      // ETA during Whisper transcription (at least 5% progress, not yet near end)
      const pct = d.progress || 0;
      if (pct >= 5 && pct < 95 && elapsed >= 10 &&
          (d.stage || '').startsWith('transcrib')) {
        const etaSecs = (elapsed / pct) * (100 - pct);
        if (etaSecs > 30) label += ` · ~${fmtElapsed(etaSecs)} left`;
      }
    }
    $('progress-label').textContent = label;

    if (d.status === 'done') { stopPolling(); setAnalyzing(false); applyResult(d.result); }
    else if (d.status === 'cancelled') { stopPolling(); setAnalyzing(false); toast('Analysis cancelled.', 3000); }
    else if (d.status === 'error') {
      stopPolling(); setAnalyzing(false);
      _showStatHint(`<strong style="color:var(--red)">⚠ Analysis failed</strong><br>${escH(d.error || 'Unknown error')}`);
      toast('Analysis failed — see panel for details', 6000);
    }
  } catch {}
}

function stopPolling() { if (S.polling) { clearInterval(S.polling); S.polling=null; } }

function setAnalyzing(on) {
  $('progress-inline').style.display = on ? 'flex' : 'none';
  if (on) $('progress-fill').style.width = '0%';
  $('btn-analyze').disabled  = on;
  if (on) $('player')?.pause();
}

export async function cancelAnalysis() {
  if (!S.taskId) return;
  stopPolling();
  try {
    await api('/api/analyze/cancel', { task_id: S.taskId });
  } catch {}
  setAnalyzing(false);
  toast('Analysis cancelled.', 3000);
}

function applyResult(result) {
  S.media             = result.media;
  S.waveformData      = result.waveform || [];
  S.waveformThreshold = result.waveform_threshold ?? 0;
  S.waveformMaxAmp    = result.waveform_max_amp    ?? 0;
  S.removalCandidates = result.removal_candidates;
  S.telopEntries      = result.telop_entries;
  S.words             = result.words || [];
  S.removalsEnabled   = new Set(S.removalCandidates.map(c=>c.id));
  S.telopsEnabled     = new Set(S.telopEntries.map(t=>t.id));
  S.history=[]; S.historyIndex=-1;
  document.dispatchEvent(new CustomEvent(EVT_RESULT_APPLIED, { detail: { result } }));
  _onRender();
  updateStats(result.stats);
  $('btn-accept-all').disabled = false;
  $('btn-keep-all').disabled   = false;
  $('btn-copy-transcript').disabled = S.telopEntries.length === 0;
  $('btn-analyze').disabled  = false;
  if (result.stats.whisper_warning) {
    _showStatHint(
      `<strong style="color:var(--yellow)">⚠ Filler detection skipped</strong> — ` +
      `${escH(result.stats.whisper_warning)}`
    );
    toast(`Analysis done — ${result.stats.total_removals} silence cuts (filler detection skipped)`, 7000);
  } else if (result.stats.total_removals === 0) {
    const thresh = parseFloat($('s-threshold').value);
    const suggestThresh = Math.min(-25, thresh + 10);
    _showStatHint(
      `No silence regions found at <strong>${thresh} dB</strong>.<br>` +
      `Try raising the threshold to <strong>${suggestThresh} dB</strong> in Settings, ` +
      `or lower Min Duration.`
    );
    toast('No silences found — adjust threshold in Settings', 5000);
  } else if (result.stats.kept_percent < 10) {
    const thresh = parseFloat($('s-threshold').value);
    const suggestThresh = Math.max(-70, thresh - 10);
    _showStatHint(
      `<strong style="color:var(--yellow)">⚠ ${result.stats.kept_percent}% of video kept</strong> — ` +
      `most of the file is flagged for removal.<br>` +
      `Try lowering the threshold to <strong>${suggestThresh} dB</strong> in Settings.`
    );
    toast(`Analysis done — ${result.stats.total_removals} cuts (${result.stats.kept_percent}% kept)`, 6000);
  } else {
    const hint = $('stat-hint');
    if (hint) hint.style.display = 'none';
    toast(`Analysis done — ${result.stats.total_removals} cuts detected`);
  }
}

function updateStats(stats) {
  $('fs-orig').textContent  = fmtDur(stats.original_duration);
  $('fs-kept').textContent  = fmtDur(stats.kept_duration);
  $('fs-cuts').textContent  = stats.total_removals;
  if (stats.telop_count > 0) {
    $('fs-telop').textContent = stats.telop_count;
    $('fs-telop-wrap').style.display = '';
  }
}
