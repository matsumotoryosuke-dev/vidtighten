import { S } from './modules/state.js';
import { $, fmt, fmtDur, plur, toast, api, apiFetch, dlBlob,
         EVT_FILE_LOADED, EVT_RESULT_APPLIED,
         EVT_STATE_RESET } from './modules/utils.js';
import { timeToX, applyZoom, updateZoomLabel }                 from './modules/zoom.js';
import { initHistory, snap, pushHistory, undo, redo, updateUndoRedo } from './modules/history.js';
import {
  initWaveform, resizeCanvas, renderWaveform, renderOverlays,
  toggleRemoval, seekTo, getFollowPlayhead,
  initWaveformEvents, _updatePlayhead,
} from './modules/waveform.js';
import {
  renderTranscript, syncTranscriptHighlight,
  initTranscriptEvents, setTranscriptDeps, focusRegionInWaveform,
  getFocusedId, invalidateWordCaches,
  loadTranscriptDisplayPrefs, initTranscriptContextMenu,
} from './modules/transcript.js';
import {
  renderSilenceList, initTransportEvents, handleNormalPlaybackTimeUpdate,
} from './modules/transport.js';
import {
  initAnalysis, checkCaps, handleFileDrop, loadFilePath,
  resetState, startAnalysis, cancelAnalysis,
} from './modules/analysis.js';
import {
  SETTINGS_DEFAULTS, bindSlider, _readSettingsFromUI, _applySettingsToUI,
  loadSavedSettings, saveSettingsToStorage, _updateSilPreview,
} from './modules/settings.js';
import {
  buildSegsJS, updateLiveStats, refreshCacheInfo,
  scheduleAutosave, openDrawer, closeDrawer,
} from './modules/export-module.js';
import {
  initExportDialog, openExportDialog, closeExportDialog, isExportDialogOpen,
} from './modules/export-dialog.js';
import { initDebug } from './modules/debug.js';
import {
  initTelopOverlay, tickTelopOverlay,
  setTelopOverlayEnabled, setTelopOverlayAvailable, invalidateTelopOverlay,
  onTelopSelectChange,
} from './modules/telop-overlay.js';
import { initWordEdit, getSelectedWordIds, deleteWords, clearWordSelection, advanceWordEditCounter } from './modules/word-edit.js';
import { initTxSearch } from './modules/tx-search.js';
import { initLlmCorrect } from './modules/llm-correct.js';

// ── renderAll coordinator ──────────────────────────────────────────
function renderAll() {
  renderWaveform();
  renderOverlays();
  renderTranscript();
  updateLiveStats();
}

// ── restoreSession (here to avoid circular analysis <-> export) ───
function restoreSession(s) {
  S.removalCandidates = s.removalCandidates || [];
  S.telopEntries      = s.telopEntries      || [];
  S.words             = s.words             || [];
  S.waveformData      = s.waveformData      || [];
  S.waveformThreshold = s.waveformThreshold ?? 0;
  S.waveformMaxAmp    = s.waveformMaxAmp    ?? 0;
  S.removalsEnabled   = new Set(s.removalsEnabled||[]);
  S.telopsEnabled     = new Set(s.telopsEnabled||[]);
  // Reset word-edit state: clears the lazy word-lookup cache (_wordMap) so
  // that switching files within the same browser session doesn't carry over
  // the previous file's word objects (both files share IDs like "w0","w1").
  // initWordEdit() also resets _nextId and drag state.
  initWordEdit();
  // Advance the word-edit ID counter past any existing 'wd{N}' IDs in the
  // restored session so deleteWords() never produces a conflicting ID.
  const maxWd = S.removalCandidates
    .filter(c => c.type === 'word' && /^wd(\d+)$/.test(c.id))
    .map(c => parseInt(c.id.slice(2), 10))
    .reduce((m, n) => Math.max(m, n), -1);
  if (maxWd >= 0) advanceWordEditCounter(maxWd + 1);
  S.history = [snap()]; S.historyIndex = 0;
  if (S.removalCandidates.length) {
    $('btn-export').disabled = false;
  }
  renderAll();
  updateUndoRedo();
  _updateSilPreview();
  invalidateTelopOverlay();   // warm overlay cache with restored telop entries
}

// ── Module init ────────────────────────────────────────────────────
initHistory({
  onHistory: () => { clearWordSelection(); renderAll(); invalidateTelopOverlay(); updateUndoRedo(); scheduleAutosave(); },
});

initWaveform({
  onRender:  () => { renderAll(); },
  onHistory: () => { pushHistory(); renderAll(); scheduleAutosave(); },
});

initAnalysis({
  onRender:          () => renderAll(),
  onRestoreSession:  restoreSession,
});

// Wire transcript <-> transport dep-injection to break circular
setTranscriptDeps({
  onRenderSilenceList: (container) => renderSilenceList(container),
});

initTranscriptEvents({
  onToggle: id => { toggleRemoval(id); pushHistory(); renderAll(); invalidateTelopOverlay(); scheduleAutosave(); },
  onSeek:   t  => seekTo(t),
});

loadTranscriptDisplayPrefs();
initTranscriptContextMenu();
initTxSearch({
  onChange: () => { renderTranscript(); invalidateTelopOverlay(); scheduleAutosave(); },
});
initLlmCorrect({
  onApplied: () => { renderTranscript(); invalidateTelopOverlay(); scheduleAutosave(); },
});

initTransportEvents({
  onToggle:              id => { toggleRemoval(id); pushHistory(); renderAll(); invalidateTelopOverlay(); scheduleAutosave(); },
  player:                $('player'),
  focusRegionInWaveform: focusRegionInWaveform,
  getFocusedId:          getFocusedId,
});

// ── RAF loop ───────────────────────────────────────────────────────
let _rafId  = null;
let _lastCt = -1;
let _wfRect = null;   // cached BoundingClientRect for waveform-wrap (Fix 2)
const player   = $('player');
const wfWrap   = $('waveform-wrap');
const playhead = $('wf-playhead');

function tick() {
  const ct = player.currentTime;

  // Fix 1: Skip all DOM writes when paused and time hasn't changed.
  // Still reschedule so that a paused-seek triggers a fresh frame.
  if (player.paused && ct === _lastCt) {
    _rafId = requestAnimationFrame(tick);
    return;
  }
  _lastCt = ct;
  syncTranscriptHighlight();

  if (!player.paused && !player.ended) {
    const dur = player.duration;
    if (dur) {
      $('seek-bar').value = Math.round(ct * 100);
      $('vtime').textContent = `${fmt(ct)} / ${fmt(dur)}`;
      // Follow playhead: keep playhead at ~30% from left when zoomed
      if (getFollowPlayhead() && S.zoom.level > 1) {
        const ANCHOR = 0.3;
        const newOffset = ct / dur - ANCHOR / S.zoom.level;
        S.zoom.offset = Math.max(0, Math.min(1 - 1 / S.zoom.level, newOffset));
        renderWaveform();
        renderOverlays();
      }
      // Waveform playhead — use cached rect (Fix 2)
      if (!_wfRect) _wfRect = wfWrap.getBoundingClientRect();
      const W = _wfRect.width;
      const px = timeToX(ct, W);
      playhead.style.transform = `translateX(${px}px)`;
      playhead.style.opacity = (px >= 0 && px <= W) ? '1' : '0';
      tickTelopOverlay(ct);
    }
  }

  _rafId = requestAnimationFrame(tick);
}

function startRAF() {
  if (_rafId) cancelAnimationFrame(_rafId);
  _rafId = requestAnimationFrame(tick);
}

// ── Export stem helper ─────────────────────────────────────────────
const _stem = () => S.filePath ? S.filePath.split('/').pop().replace(/\.[^.]+$/, '') : 'export';

// ── Event listeners ────────────────────────────────────────────────

// File loading
$('btn-open').addEventListener('click', async () => {
  try {
    const r = await api('/api/filepicker', {});
    if (r.path) { loadFilePath(r.path); return; }
  } catch {}
  $('file-input').click();
});

$('file-input').addEventListener('change', async e => {
  const f = e.target.files[0];
  if (!f) return;
  if (f.path) { loadFilePath(f.path); return; }
  const GB = 1024 ** 3;
  if (f.size > 4 * GB) {
    toast('File too large to upload. Use the Open File button so VidTighten reads it by path without copying it.'); return;
  }
  const fd = new FormData();
  fd.append('file', f);
  const r = await (await fetch('/api/upload', {method:'POST',body:fd})).json();
  if (r.path) loadFilePath(r.path);
  else toast('Upload failed: ' + (r.error || 'unknown'));
});

const dz = $('dropzone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', async e => {
  e.preventDefault(); dz.classList.remove('dragover');
  handleFileDrop(e.dataTransfer.files[0], e.dataTransfer);
});

// Document-level drop so dragging onto any part of the app works
document.addEventListener('dragover', e => { e.preventDefault(); });
document.addEventListener('drop', async e => {
  e.preventDefault();
  if (!dz.classList.contains('hidden')) return;
  handleFileDrop(e.dataTransfer.files[0], e.dataTransfer);
});

document.addEventListener(EVT_FILE_LOADED, () => {
  startRAF();
  // Show telop toggle for all files (video and audio-only)
  setTelopOverlayAvailable(true);
  // Clear stale overlay text from the previous file (entries were wiped in resetState).
  invalidateTelopOverlay();
});
document.addEventListener(EVT_RESULT_APPLIED, () => {
  pushHistory();
  scheduleAutosave();
  _updateSilPreview();
  invalidateTelopOverlay();
});
// After resetState() wipes S.telopEntries, clear the overlay cache so no
// stale text remains from the previous file.
document.addEventListener(EVT_STATE_RESET, () => { invalidateTelopOverlay(); invalidateWordCaches(); initWordEdit(); clearWordSelection(); });

// Player metadata
player.addEventListener('loadedmetadata', () => {
  $('seek-bar').max = Math.round(player.duration*100);
  if (S.media && player.duration && isFinite(player.duration)) {
    S.media.duration = player.duration;
    renderAll();
  }
});

// Analysis controls
$('btn-analyze').addEventListener('click', startAnalysis);
$('btn-cancel-analysis').addEventListener('click', cancelAnalysis);

$('btn-export').addEventListener('click', e => {
  e.stopPropagation();
  isExportDialogOpen() ? closeExportDialog() : openExportDialog();
});


// Undo/Redo
$('btn-undo').addEventListener('click', undo);
$('btn-redo').addEventListener('click', redo);

// ── Unified keyboard dispatcher ──────────────────────────────────
document.addEventListener('keydown', e => {
  // ESC exits CSS fullscreen before any other handler
  if (e.key === 'Escape' && $('video-panel').classList.contains('is-fullscreen')) {
    toggleFullscreen();
    return;
  }

  const tag = document.activeElement?.tagName;
  const inInput = tag === 'INPUT' || tag === 'TEXTAREA' ||
                  document.activeElement?.isContentEditable;

  // Undo / redo — always active
  if ((e.metaKey||e.ctrlKey) && e.key==='z') {
    e.shiftKey ? redo() : undo();
    e.preventDefault();
    return;
  }

  // ⌘E / Ctrl+E — open export dialog
  if ((e.metaKey||e.ctrlKey) && e.key==='e') {
    e.preventDefault();
    isExportDialogOpen() ? closeExportDialog() : openExportDialog();
    return;
  }

  // Escape — close settings drawer + shortcuts overlay + dep blocker + clear word selection
  if (e.key === 'Escape') {
    closeDrawer();
    closeExportDialog();
    $('shortcuts-overlay').classList.remove('open');
    $('dep-blocker-overlay').classList.remove('open');
    clearWordSelection();
    return;
  }

  if (inInput) return;

  // Delete / Backspace — remove selected word(s) from transcript
  if ((e.key === 'Delete' || e.key === 'Backspace') && !e.metaKey && !e.ctrlKey) {
    const wordIds = getSelectedWordIds();
    if (wordIds.length) {
      e.preventDefault();
      deleteWords(wordIds);
      clearWordSelection();
      pushHistory();
      renderAll();
      invalidateTelopOverlay();
      scheduleAutosave();
      return;
    }
  }

  // ? — keyboard shortcut help
  if (e.key === '?') {
    e.preventDefault();
    $('shortcuts-overlay').classList.toggle('open');
    return;
  }

  // Space — play / pause
  if (e.key === ' ' && player.src) {
    e.preventDefault();
    if (player.paused) player.play(); else player.pause();
    return;
  }

  // Arrow keys — seek ±5 s
  if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && player.src) {
    e.preventDefault();
    const delta = e.key === 'ArrowLeft' ? -5 : 5;
    player.currentTime = Math.max(0, Math.min(player.duration || 0,
                                              player.currentTime + delta));
    return;
  }


  // S — toggle Settings drawer
  if (e.key === 's' || e.key === 'S') {
    e.preventDefault();
    $('drawer').classList.contains('open') ? closeDrawer() : openDrawer();
    return;
  }

  // F — toggle fullscreen (only when video loaded)
  if ((e.key === 'f' || e.key === 'F') && player.src) {
    e.preventDefault();
    toggleFullscreen();
    return;
  }

  // T — toggle telop overlay (only when button is visible, i.e. video is loaded)
  if ((e.key === 't' || e.key === 'T') && player.src) {
    const btn = $('btn-telop-overlay');
    if (btn && btn.style.display !== 'none') {
      e.preventDefault();
      btn.click();
    }
    return;
  }

  // M — toggle mute
  if ((e.key === 'm' || e.key === 'M') && player.src) {
    e.preventDefault();
    $('btn-mute').click();
    return;
  }
});

// Video player controls
$('btn-playpause').addEventListener('click', () => {
  if (player.paused) player.play(); else player.pause();
});
player.addEventListener('play',  () => { $('btn-playpause').textContent = '⏸'; });
player.addEventListener('pause', () => { $('btn-playpause').textContent = '▶'; });
$('seek-bar').addEventListener('input', () => {
  player.currentTime = $('seek-bar').value / 100;
});

// Volume — restore last-used level (defaults to 100 for a fresh install).
const savedVolume = parseInt(localStorage.getItem('vt_volume') ?? '100', 10);
player.volume = Math.max(0, Math.min(100, savedVolume)) / 100;
$('volume-slider').value = Math.round(player.volume * 100);
function _updateMuteIcon() {
  $('btn-mute').textContent = (player.muted || player.volume === 0) ? '🔇' : '🔊';
}
_updateMuteIcon();
$('btn-mute').addEventListener('click', () => {
  player.muted = !player.muted;
  _updateMuteIcon();
});
$('volume-slider').addEventListener('input', () => {
  player.muted  = false;
  player.volume = $('volume-slider').value / 100;
  localStorage.setItem('vt_volume', $('volume-slider').value);
  _updateMuteIcon();
});
player.addEventListener('volumechange', _updateMuteIcon);

// Transcript timeupdate for normal-playback removal skip (highlight sync moved to RAF tick)
player.addEventListener('timeupdate', () => {
  handleNormalPlaybackTimeUpdate(player.currentTime);
});
// Fix 3: Remove duplicate timeupdate→tickTelopOverlay (RAF covers playback).
// Fix 1: Reset _lastCt on seeked so a paused-seek forces a fresh RAF tick.
player.addEventListener('seeked', () => { _lastCt = -1; syncTranscriptHighlight(); });

// Settings drawer
$('btn-settings').addEventListener('click', openDrawer);
$('btn-close-drawer').addEventListener('click', closeDrawer);
$('drawer-back').addEventListener('click', closeDrawer);

// Dependency preflight blocker (ffmpeg/ffprobe missing) — see checkCaps() in
// modules/analysis.js, which decides whether to open this.
$('dep-blocker-dismiss').addEventListener('click', () => {
  $('dep-blocker-overlay').classList.remove('open');
});

$('btn-settings-apply').addEventListener('click', () => {
  const s = _readSettingsFromUI();
  saveSettingsToStorage(s);
  // Font-size and res affect max_em; invalidate in case the user changed them
  // without blurring (so the 'change' event on those inputs may not have fired).
  invalidateTelopOverlay();
  toast('Settings saved');
  closeDrawer();
});

$('btn-settings-revert').addEventListener('click', () => {
  _applySettingsToUI(loadSavedSettings());
  // Programmatic .value changes don't fire 'change' events — force a rebuild
  // so the cache reflects the reverted font-size / resolution.
  invalidateTelopOverlay();
  toast('Settings reverted');
});

// Slider bindings
bindSlider('s-threshold', 's-threshold-val', v => `${v} dB`);
bindSlider('s-min-dur',   's-min-dur-val',   v => `${v} s`);
bindSlider('s-hangover',  's-hangover-val',  v => `${v} ms`);
bindSlider('s-padding',   's-padding-val',   v => `${v} ms`);

// Threshold line moves in real time + preview badge
$('s-threshold').addEventListener('input', renderWaveform);
$('s-threshold').addEventListener('input', _updateSilPreview);
$('s-min-dur').addEventListener('input', _updateSilPreview);
$('s-hangover').addEventListener('input', _updateSilPreview);

// Telop overlay toggle
$('btn-telop-overlay').addEventListener('click', () => {
  setTelopOverlayEnabled(!$('btn-telop-overlay').classList.contains('active'));
});

// ── Telop quick-edit control panel ───────────────────────────────────────────

// Map: tc-control-id → real-settings-id
const TC_SYNC = [
  ['tc-font',         't-font'],
  ['tc-font-size',    't-font-size'],
  ['tc-font-color',   't-font-color'],
  ['tc-pos-y',        't-pos-y'],
  ['tc-line-spacing', 't-line-spacing'],
];

function _syncFromSettings() {
  for (const [tcId, tId] of TC_SYNC) {
    const src = $(tId), dst = $(tcId);
    if (src && dst) dst.value = src.value;
  }
}

function _openTelopCtrl() {
  _syncFromSettings();
  $('stats-content').style.display = 'none';
  $('telop-ctrl').style.display    = 'flex';
}

function _closeTelopCtrl() {
  $('telop-ctrl').style.display    = 'none';
  $('stats-content').style.display = 'flex';
}

onTelopSelectChange(selected => {
  if (selected) _openTelopCtrl();
  else _closeTelopCtrl();
});

// tc → real settings: update the canonical input and fire 'input'+'change'
for (const [tcId, tId] of TC_SYNC) {
  const tcEl = $(tcId), tEl = $(tId);
  if (!tcEl || !tEl) continue;
  tcEl.addEventListener('input', () => {
    tEl.value = tcEl.value;
    tEl.dispatchEvent(new Event('input',  { bubbles: true }));
    tEl.dispatchEvent(new Event('change', { bubbles: true }));
  });
}

// Close button — dispatch ESC to trigger the overlay's keydown handler
$('tc-close')?.addEventListener('click', () => {
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
});

// Fullscreen toggle — CSS-based so it works in pywebview WKWebView
function toggleFullscreen() {
  const panel = $('video-panel');
  const entering = !panel.classList.contains('is-fullscreen');
  panel.classList.toggle('is-fullscreen', entering);
  $('btn-fullscreen').classList.toggle('active', entering);
  $('btn-fullscreen').title = entering ? 'Exit fullscreen (F)' : 'Toggle fullscreen (F)';
  invalidateTelopOverlay();
}
$('btn-fullscreen').addEventListener('click', toggleFullscreen);

// Invalidate wrap cache when res or font-size change (max_em depends on both)
$('t-font-size').addEventListener('change', invalidateTelopOverlay);
$('t-res').addEventListener('change', invalidateTelopOverlay);

// Re-render overlay when style-only settings change (no cache rebuild needed).
// t-font-color uses 'input' so the color picker updates live while dragging;
// the rest fire 'change' (number inputs commit on blur/Enter, select on pick).
// Guard against NaN currentTime (no media loaded) — NaN comparisons are always
// false so the entry lookup clears the overlay text unnecessarily.
function _refreshTelopStyle() {
  if (isFinite(player.currentTime)) tickTelopOverlay(player.currentTime);
}
$('t-font-color').addEventListener('input',    _refreshTelopStyle);
$('t-font').addEventListener('change',         _refreshTelopStyle);
// 'input' (not 'change'): _intVal falls back gracefully for invalid intermediates.
$('t-pos-y').addEventListener('input',         _refreshTelopStyle);
$('t-line-spacing').addEventListener('input',  _refreshTelopStyle);

// ── H-resizer: waveform-panel height at bottom of workspace ───────────────
{
  const waveformPanel = $('waveform-panel');
  const resizer       = $('h-resizer');
  const MIN_WF        = 60;   // minimum waveform height
  const MAX_WF        = 320;  // maximum waveform height

  const savedWfH = parseInt(localStorage.getItem('vt_wf_h') || '0');
  if (savedWfH > 0) waveformPanel.style.height = savedWfH + 'px';

  resizer.addEventListener('mousedown', e => {
    e.preventDefault();
    const startY = e.clientY;
    const startH = waveformPanel.offsetHeight;

    resizer.classList.add('dragging');
    document.body.style.cursor     = 'ns-resize';
    document.body.style.userSelect = 'none';

    const onMove = ev => {
      // dragging UP = taller waveform (invert delta)
      const newH = Math.max(MIN_WF, Math.min(startH - (ev.clientY - startY), MAX_WF));
      waveformPanel.style.height = newH + 'px';
    };
    // Fix 7: shared cleanup for mouseup AND window mouseleave (mouse released outside window)
    const cleanup = () => {
      resizer.classList.remove('dragging');
      document.body.style.cursor     = '';
      document.body.style.userSelect = '';
      localStorage.setItem('vt_wf_h', waveformPanel.offsetHeight);
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup',   cleanup);
      window.removeEventListener('mouseleave',  cleanup);
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   cleanup);
    window.addEventListener('mouseleave',  cleanup);
  });
}

// ── V-resizer: left-col width ─────────────────────────────────────────────
{
  const leftCol  = $('left-col');
  const vResizer = $('v-resizer');
  const MIN_LEFT = 280;   // minimum left-col width
  const MIN_TX   = 240;   // minimum transcript width

  const savedW = parseInt(localStorage.getItem('vt_left_w') || '0');
  if (savedW > 0) leftCol.style.width = savedW + 'px';

  vResizer.addEventListener('mousedown', e => {
    e.preventDefault();
    const startX = e.clientX;
    const startW = leftCol.offsetWidth;
    const maxW   = $('main-area').offsetWidth - MIN_TX - 4;

    vResizer.classList.add('dragging');
    document.body.style.cursor     = 'col-resize';
    document.body.style.userSelect = 'none';

    const onMove = ev => {
      const newW = Math.max(MIN_LEFT, Math.min(startW + ev.clientX - startX, maxW));
      leftCol.style.width = newW + 'px';
    };
    // Fix 7: shared cleanup for mouseup AND window mouseleave (mouse released outside window)
    const cleanup = () => {
      vResizer.classList.remove('dragging');
      document.body.style.cursor     = '';
      document.body.style.userSelect = '';
      localStorage.setItem('vt_left_w', leftCol.offsetWidth);
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup',   cleanup);
      window.removeEventListener('mouseleave',  cleanup);
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   cleanup);
    window.addEventListener('mouseleave',  cleanup);
  });
}

// Cache clear
$('btn-clear-cache').addEventListener('click', async () => {
  const clearSessions = $('cache-clear-sessions').checked;
  const label = clearSessions ? 'uploads, exports, and sessions' : 'uploads and exports';
  if (!confirm(`Clear ${label}? This cannot be undone.`)) return;
  try {
    const r = await apiFetch('/api/cache/clear', { sessions: clearSessions });
    const d = await r.json();
    const total = d.removed_uploads + d.removed_exports + d.removed_sessions;
    toast(`Cleared ${plur(total, 'file')}`);
    await refreshCacheInfo();
  } catch (e) { toast('Clear failed: ' + e.message); }
});

// Recent sessions dropdown
(function _initRecentSessions() {
  const btn      = $('btn-recent');
  const dropdown = $('recent-dropdown');
  const list     = $('recent-list');
  const empty    = $('recent-empty');
  let _fetching  = false;

  function _relTime(ts) {
    const secs = (Date.now() / 1000) - ts;
    if (secs < 120)    return 'just now';
    if (secs < 3600)   return `${Math.round(secs / 60)}m ago`;
    if (secs < 86400)  return `${Math.round(secs / 3600)}h ago`;
    return `${Math.round(secs / 86400)}d ago`;
  }

  async function _open() {
    const r = btn.getBoundingClientRect();
    dropdown.style.left = r.left + 'px';
    dropdown.classList.add('open');

    if (_fetching) return;
    _fetching = true;
    try {
      const data = await (await fetch('/api/session/list')).json();
      const sessions = data.sessions || [];
      if (sessions.length === 0) {
        list.innerHTML = '';
        list.appendChild(empty);
        return;
      }
      list.innerHTML = '';
      for (const s of sessions) {
        const item = document.createElement('div');
        item.className = 'recent-item' + (s.file_exists ? '' : ' missing');
        item.title = s.file_exists ? s.file_path : `File not found: ${s.file_path}`;

        const meta = [];
        if (s.telop_count)   meta.push(plur(s.telop_count,   'caption'));
        if (s.removal_count) meta.push(plur(s.removal_count, 'cut'));

        item.innerHTML = `
          <span class="recent-icon">${s.file_exists ? '🎬' : '⚠️'}</span>
          <div class="recent-info">
            <div class="recent-name">${s.file_name}</div>
            <div class="recent-meta">${_relTime(s.saved_at)}${meta.length ? ' · ' + meta.join(', ') : ''}</div>
          </div>
          <button class="recent-remove-btn" title="Remove from list" aria-label="Remove session">✕</button>`;

        const removeBtn = item.querySelector('.recent-remove-btn');
        removeBtn.addEventListener('click', async e => {
          e.stopPropagation();
          try { await api('/api/session/delete', {path: s.file_path}); } catch {}
          item.remove();
          if (!list.children.length) list.appendChild(empty);
        });

        if (s.file_exists) {
          item.addEventListener('click', e => {
            if (e.target.closest('.recent-remove-btn')) return;
            _close();
            loadFilePath(s.file_path);
          });
        }
        list.appendChild(item);
      }
    } catch (err) {
      list.innerHTML = '';
      empty.textContent = 'Could not load sessions';
      list.appendChild(empty);
    } finally {
      _fetching = false;
    }
  }

  function _close() { dropdown.classList.remove('open'); }

  btn.addEventListener('click', e => {
    e.stopPropagation();
    dropdown.classList.contains('open') ? _close() : _open();
  });
  document.addEventListener('click', e => {
    if (!dropdown.contains(e.target)) _close();
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') _close(); });
})();

// Shortcuts overlay — close on backdrop click
$('shortcuts-overlay').addEventListener('click', e => {
  if (e.target === $('shortcuts-overlay')) $('shortcuts-overlay').classList.remove('open');
});

// Expose loadFilePath for pywebview menu bar (evaluate_js can't reach ES module scope)
window._vtLoadFile = loadFilePath;

// Re-analyze (same as clicking the toolbar button)
window._vtReanalyze = () => {
  const btn = $('btn-analyze');
  if (btn && !btn.disabled) btn.click();
};

// Detect Silence Only — re-run silence detection without Whisper
window._vtRedetectSilence = async () => {
  if (!S.filePath) return;
  try {
    const r = await api('/api/analyze/redetect_silence', {
      path:         S.filePath,
      threshold_db: parseFloat($('s-threshold').value),
      min_duration: parseFloat($('s-min-dur').value),
      hangover_ms:  parseInt($('s-hangover').value),
    });
    if (r.error) { toast('Re-detect failed: ' + r.error, 4000); return; }
    const newSils = r.candidates;
    const kept    = S.removalCandidates.filter(c => c.type !== 'silence');
    const prevDisabled = S.removalCandidates
      .filter(c => c.type === 'silence' && !S.removalsEnabled.has(c.id));
    const MATCH_SLACK  = 0.5;
    const userDisabledNew = new Set(
      newSils
        .filter(nr => prevDisabled.some(od =>
          Math.abs(nr.start - od.start) <= MATCH_SLACK &&
          Math.abs(nr.end   - od.end)   <= MATCH_SLACK))
        .map(nr => nr.id)
    );
    S.removalCandidates = [...newSils, ...kept].sort((a, b) => a.start - b.start);
    const keptEnabled = new Set([...S.removalsEnabled].filter(id => kept.some(x => x.id === id)));
    S.removalsEnabled  = new Set([
      ...newSils.filter(c => !userDisabledNew.has(c.id)).map(c => c.id),
      ...keptEnabled,
    ]);
    S.silPreview = null;
    pushHistory(); renderAll(); scheduleAutosave();
    const durStr = newSils.length > 0 ? ` · ${fmtDur(r.total_duration)}` : '';
    const note   = userDisabledNew.size ? ` (${userDisabledNew.size} user-kept preserved)` : '';
    toast(`Silences re-detected — ${plur(newSils.length, 'region')}${durStr}${note}`);
  } catch (err) {
    toast('Re-detect error: ' + err.message, 4000);
  }
};

// Open export dialog, optionally on a specific tab ('roughcut' | 'telop')
window._vtExport = (tab) => {
  const btn = $('btn-export');
  if (!btn || btn.disabled) return;
  btn.click();
  if (tab) {
    setTimeout(() => {
      const tabBtn = document.querySelector(`.export-tab[data-tab="${tab}"]`);
      if (tabBtn) tabBtn.click();
    }, 50);
  }
};

// ── Global error surfacing (helps debug in pywebview where console isn't visible) ──
// Fix 4: Debounce ResizeObserver with rAF coalescing to avoid synchronous
// layout reads on every pixel of resize drag.
// Fix 2: Invalidate cached _wfRect so the next RAF tick re-reads the rect.
if (typeof ResizeObserver !== 'undefined') {
  let _resizeRaf = null;
  const ro = new ResizeObserver(() => {
    if (_resizeRaf) return;
    _resizeRaf = requestAnimationFrame(() => {
      _resizeRaf = null;
      _wfRect = null;   // Fix 2: invalidate cached rect
      _refreshTelopStyle();
    });
  });
  ro.observe(player);
}

window.addEventListener('unhandledrejection', e => {
  const msg = e.reason?.message ?? String(e.reason) ?? '';
  // Fix 8: Suppress benign browser/network errors that are not actionable for the user.
  if (/AbortError|The play\(\) request|NetworkError|Load was interrupted/i.test(msg)) return;
  toast('JS Error: ' + (msg || 'Unknown async error'), 10000);
  console.error('Unhandled rejection:', e.reason);
});
window.onerror = (msg, src, line, col, err) => {
  toast(`JS Error: ${msg} (${src?.split('/').pop()}:${line})`, 10000);
  return false;
};

// Fix 2: Invalidate cached waveform rect on window resize so the next RAF tick re-reads it.
window.addEventListener('resize', () => { _wfRect = null; });

// ── Init ───────────────────────────────────────────────────────────
initExportDialog();
initTelopOverlay();
_applySettingsToUI(loadSavedSettings());
startRAF();
checkCaps();
resizeCanvas();
initWaveformEvents();
initDebug();
