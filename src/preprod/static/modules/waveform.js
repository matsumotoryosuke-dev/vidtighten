import { S } from './state.js';
import { $, fmt, fmtDur, toast } from './utils.js';
import { timeToX, xToTime, clampZoomOffset, applyZoom, updateZoomLabel } from './zoom.js';

// Private module vars
let _dragging  = null;  // {id, side:'left'|'right'|'body', startX, origStart, origEnd}
let _drawing   = null;  // {startFrac, currentFrac}
let _panning   = null;  // {startX, startOffset}
let _followPlayhead = true;
let _manualIdx = 0;
let _onRender  = () => {};
let _onHistory = () => {};

// Fix 5: Cache threshold element and its parsed value to avoid 60 DOM queries/sec.
let _thresholdEl = null;
let _thrDb       = -40;

// DOM refs — resolved lazily on first use
let _wfCanvas = null;
let _wfCtx    = null;
let _wfWrap   = null;
let _roLayer  = null;
let _drawOvl  = null;

function _initDOM() {
  if (_wfCanvas) return;
  _wfCanvas = $('wf-canvas');
  _wfCtx    = _wfCanvas.getContext('2d');
  _wfWrap   = $('waveform-wrap');
  _roLayer  = $('removal-overlays');
  _drawOvl  = $('draw-overlay');
}

export function initWaveform({ onRender, onHistory }) {
  _onRender  = onRender;
  _onHistory = onHistory;
}

export function resizeCanvas() {
  _initDOM();
  const dpr = window.devicePixelRatio || 1;
  const rect = _wfWrap.getBoundingClientRect();
  _wfCanvas.width  = rect.width  * dpr;
  _wfCanvas.height = rect.height * dpr;
  _wfCanvas.style.width  = rect.width  + 'px';
  _wfCanvas.style.height = rect.height + 'px';
  _wfCtx.scale(dpr, dpr);
  renderWaveform();
}

export function renderWaveform() {
  _initDOM();
  const rect = _wfWrap.getBoundingClientRect();
  const W = rect.width, H = rect.height;
  if (!W || !H) return;
  _wfCtx.clearRect(0, 0, W, H);

  const data = S.waveformData;
  const dur  = S.media?.duration || 0;
  if (!data.length || !dur) return;

  // When a preview is live, substitute preview silences for committed ones so the
  // waveform instantly reflects what the current slider values would give.
  const silRegions  = S.silPreview !== null
    ? S.silPreview
    : S.removalCandidates.filter(c => c.type === 'silence' && S.removalsEnabled.has(c.id));
  const nonSilRms   = S.removalCandidates.filter(c => c.type !== 'silence' && S.removalsEnabled.has(c.id));
  const enabledRms  = [...silRegions, ...nonSilRms];
  const { level, offset } = S.zoom;

  // Visible data window
  const visStart = offset * data.length;
  const visCount = data.length / level;
  const barW     = W / visCount;
  const halfH    = H / 2;

  // Fix 5: Use cached threshold value instead of querying the DOM every frame.
  // _thresholdEl is set on first call; _thrDb is kept current via an 'input' listener
  // registered in initWaveformEvents.
  if (!_thresholdEl) {
    _thresholdEl = document.getElementById('s-threshold');
    if (_thresholdEl) _thrDb = parseFloat(_thresholdEl.value) || -40;
  }
  const sliderDb = _thrDb;
  const thr = S.waveformMaxAmp > 0
    ? Math.pow(10, sliderDb / 20) / S.waveformMaxAmp
    : (S.waveformThreshold || 0);

  // Sort removal regions by start once (O(M log M), M typically < 100) so each
  // bar lookup is O(log M) binary search instead of O(M) linear scan.
  const rmSorted = [...enabledRms].sort((a, b) => a.start - b.start);
  const rmStarts = rmSorted.map(r => r.start);
  /** O(log M) lookup: rightmost region with start ≤ t that also contains t. */
  function _findRm(t) {
    let lo = 0, hi = rmSorted.length - 1, best = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (rmStarts[mid] <= t) { best = mid; lo = mid + 1; }
      else hi = mid - 1;
    }
    return (best >= 0 && t <= rmSorted[best].end) ? rmSorted[best] : null;
  }

  // Compute tight visible index range so off-screen bars are never iterated.
  // Without this the loop is O(data.length) every RAF frame during follow-playhead+zoom;
  // with it the loop is O(data.length / zoom_level) — a 10x+ speedup at high zoom.
  const iStart = Math.max(0, Math.floor(visStart) - 1);
  const iEnd   = Math.min(data.length - 1, Math.ceil(visStart + visCount) + 1);

  for (let i = iStart; i <= iEnd; i++) {
    const x = (i - visStart) * barW;
    if (x + barW < 0 || x > W) continue;   // fine-grained guard for boundary bars

    const t   = (i + 0.5) / data.length * dur;
    const amp = data[i];
    const h   = Math.max(1.5, amp * halfH * 0.88);
    const belowThr = thr > 0 && amp < thr;

    const rm = _findRm(t);
    let color;
    if (rm) {
      // Inside a removal region: dim bars that are below threshold
      // 'word' = user-deleted word (amber, same visual weight as auto-detected filler)
      const base = (rm.type === 'filler' || rm.type === 'word') ? '#ffcd2a' : rm.type === 'manual' ? '#ab9f8b' : '#eb5e28';
      color = belowThr ? base + '55' : base;
    } else {
      color = `rgba(204,197,185,${0.25 + amp * 0.55})`;
    }
    _wfCtx.fillStyle = color;
    _wfCtx.fillRect(x, halfH - h, Math.max(barW - 0.5, 0.5), h * 2);
  }

  // Draw threshold indicator lines (faint dashed, symmetric around centre)
  if (thr > 0 && thr < 1) {
    const thY = halfH - thr * halfH * 0.88;
    _wfCtx.save();
    _wfCtx.strokeStyle = 'rgba(255,255,255,0.13)';
    _wfCtx.lineWidth = 1;
    _wfCtx.setLineDash([3, 6]);
    _wfCtx.beginPath();
    _wfCtx.moveTo(0, thY);      _wfCtx.lineTo(W, thY);
    _wfCtx.moveTo(0, H - thY); _wfCtx.lineTo(W, H - thY);
    _wfCtx.stroke();
    _wfCtx.restore();
  }
}

export function renderOverlays() {
  _initDOM();
  _roLayer.innerHTML = '';
  const dur = S.media?.duration || 0;
  if (!dur) return;

  const { level, offset } = S.zoom;

  for (const c of S.removalCandidates) {
    const enabled = S.removalsEnabled.has(c.id);
    const left  = (c.start/dur - offset) * level * 100;
    const width = Math.max((c.end - c.start)/dur * level * 100, 0.1);

    if (left + width < 0 || left > 100) continue;  // off-screen, skip

    const div = document.createElement('div');
    div.className = `ro type-${c.type}${enabled ? '' : ' disabled'}`;
    div.style.left  = left + '%';
    div.style.width = width + '%';
    div.dataset.id  = c.id;

    const lh = document.createElement('div'); lh.className = 'ro-handle'; lh.dataset.side = 'left';
    const fill = document.createElement('div'); fill.className = 'ro-fill';
    const rh = document.createElement('div'); rh.className = 'ro-handle'; rh.dataset.side = 'right';

    const wordCount = (c.type === 'word' && c.wordIds) ? c.wordIds.length : 0;
    // Show word label in tooltip so users can identify the word in the waveform
    const wordLabel = (c.type === 'word' && c.label) ? `"${c.label}" · ` : '';
    fill.title = wordCount > 0
      ? `${wordLabel}${wordCount}w · ${fmt(c.start)}–${fmt(c.end)} · ${fmtDur(c.end-c.start)}`
      : `${c.type} · ${fmt(c.start)}–${fmt(c.end)} · ${fmtDur(c.end-c.start)}`;

    // Duration label — CSS overflow:hidden on .ro-fill clips it when region is too narrow
    const lbl = document.createElement('span');
    lbl.className = 'ro-dur';
    lbl.textContent = wordCount > 0
      ? `${wordCount}w · ${fmtDur(c.end - c.start)}`
      : fmtDur(c.end - c.start);
    fill.appendChild(lbl);

    div.appendChild(lh); div.appendChild(fill); div.appendChild(rh);
    if (c.type === 'manual') {
      const del = document.createElement('button');
      del.className = 'ro-del';
      del.textContent = '✕';
      del.setAttribute('aria-label', 'Delete region');
      div.appendChild(del);
    }
    _roLayer.appendChild(div);
  }
}

export function toggleRemoval(id) {
  if (S.removalsEnabled.has(id)) S.removalsEnabled.delete(id);
  else S.removalsEnabled.add(id);
  _onHistory();
}

export function deleteRegion(id) {
  S.removalCandidates = S.removalCandidates.filter(c => c.id !== id);
  S.removalsEnabled.delete(id);
  _onHistory();
}

// Handle drag
function startHandleDrag(e, id, side) {
  _initDOM();
  e.stopPropagation(); e.preventDefault();
  const c = S.removalCandidates.find(r => r.id === id);
  if (!c) return;
  _dragging = { id, side, startX: e.clientX, origStart: c.start, origEnd: c.end };
  document.addEventListener('mousemove', onHandleDrag);
  document.addEventListener('mouseup', endHandleDrag);
}

function onHandleDrag(e) {
  _initDOM();
  if (!_dragging) return;
  const rect = _wfWrap.getBoundingClientRect();
  const dur  = S.media?.duration || 1;
  const dx   = e.clientX - _dragging.startX;
  const dt   = dx / rect.width / S.zoom.level * dur;  // zoom-corrected
  const c    = S.removalCandidates.find(r => r.id === _dragging.id);
  if (!c) return;
  if (_dragging.side === 'left') {
    c.start = Math.max(0, Math.min(_dragging.origEnd - 0.05, _dragging.origStart + dt));
  } else {
    c.end   = Math.max(_dragging.origStart + 0.05, Math.min(dur, _dragging.origEnd + dt));
  }
  c.start = Math.round(c.start * 1000) / 1000;
  c.end   = Math.round(c.end   * 1000) / 1000;
  _onRender();
}

function endHandleDrag() {
  if (!_dragging) return;
  _dragging = null;
  document.removeEventListener('mousemove', onHandleDrag);
  document.removeEventListener('mouseup', endHandleDrag);
  _onHistory();
}

export function seekTo(t) {
  const player = document.getElementById('player');
  if (player && player.src) player.currentTime = t;
}

function onPan(e) {
  _initDOM();
  if (!_panning) return;
  const rect = _wfWrap.getBoundingClientRect();
  const dx   = e.clientX - _panning.startX;
  const dfrac = dx / rect.width / S.zoom.level;
  S.zoom.offset = _panning.startOffset - dfrac;
  clampZoomOffset();
  renderWaveform();
  renderOverlays();
  _updatePlayhead();
}

function endPan() {
  _initDOM();
  _panning = null;
  _wfWrap.classList.remove('panning');
  document.removeEventListener('mousemove', onPan);
  document.removeEventListener('mouseup',   endPan);
}

function onDraw(e) {
  _initDOM();
  if (!_drawing) return;
  const rect = _wfWrap.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  _drawing.currentFrac = frac;
  const left  = Math.min(_drawing.startFrac, frac);
  const width = Math.abs(frac - _drawing.startFrac);
  _drawOvl.style.left  = (left  * 100) + '%';
  _drawOvl.style.width = (width * 100) + '%';
}

function endDraw(e) {
  _initDOM();
  if (!_drawing) return;
  const rect = _wfWrap.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  const a    = xToTime(Math.min(_drawing.startFrac, frac) * rect.width, rect.width);
  const b    = xToTime(Math.max(_drawing.startFrac, frac) * rect.width, rect.width);
  _drawing = null;
  _drawOvl.style.display = 'none';
  document.removeEventListener('mousemove', onDraw);
  document.removeEventListener('mouseup',   endDraw);
  if (b - a < 0.05) { seekTo(xToTime(frac * rect.width, rect.width)); return; }
  const id = `m${_manualIdx++}`;
  S.removalCandidates.push({id, start:Math.round(a*1000)/1000, end:Math.round(b*1000)/1000, type:'manual', label:'manual'});
  S.removalCandidates.sort((x,y) => x.start - y.start);
  S.removalsEnabled.add(id);
  _onHistory();
  toast('Manual region added');
}

export function getFollowPlayhead() { return _followPlayhead; }
function setFollowPlayhead(val) { _followPlayhead = val; }

export function _updatePlayhead() {
  _initDOM();
  const player  = document.getElementById('player');
  const playhead = document.getElementById('wf-playhead');
  if (!player || !playhead) return;
  const W = _wfWrap.getBoundingClientRect().width;
  const px = timeToX(player.currentTime, W);
  playhead.style.transform = `translateX(${px}px)`;
  playhead.style.opacity = (px >= 0 && px <= W) ? '1' : '0';
}

export function initWaveformEvents() {
  _initDOM();

  window.addEventListener('resize', resizeCanvas);

  // Fix 5: Keep _thrDb in sync with the slider so renderWaveform never touches the DOM.
  const thrEl = document.getElementById('s-threshold');
  if (thrEl) {
    _thresholdEl = thrEl;
    _thrDb = parseFloat(thrEl.value) || -40;
    thrEl.addEventListener('input', () => { _thrDb = parseFloat(thrEl.value) || -40; });
  }

  _wfWrap.addEventListener('mousedown', e => {
    if (e.target !== _wfCanvas && e.target !== _wfWrap) return;
    const dur = S.media?.duration;
    if (!dur) return;
    const rect = _wfWrap.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;  // 0-1 within canvas

    if (S.zoom.level > 1) {
      // PAN mode when zoomed
      _panning = { startX: e.clientX, startOffset: S.zoom.offset };
      _wfWrap.classList.add('panning');
      document.addEventListener('mousemove', onPan);
      document.addEventListener('mouseup',   endPan);
      return;
    }

    // DRAW / SEEK mode at 1x
    _drawing = { startFrac: frac, currentFrac: frac };
    _drawOvl.style.display = 'block';
    _drawOvl.style.left    = (frac * 100) + '%';
    _drawOvl.style.width   = '0%';
    document.addEventListener('mousemove', onDraw);
    document.addEventListener('mouseup',   endDraw);
  });

  _wfWrap.addEventListener('wheel', e => {
    e.preventDefault();
    if (!S.media?.duration) return;
    const rect = _wfWrap.getBoundingClientRect();
    const pivotFrac = (e.clientX - rect.left) / rect.width;
    const factor    = e.deltaY < 0 ? 1.5 : 1/1.5;
    applyZoom(S.zoom.level * factor, pivotFrac);
    renderWaveform(); renderOverlays(); _updatePlayhead();
  }, { passive: false });

  // Waveform hover time cursor — shows timecode at mouse position
  _wfWrap.addEventListener('mousemove', e => {
    if (!S.media?.duration) return;
    const tip  = document.getElementById('wf-time-tip');
    const rect = _wfWrap.getBoundingClientRect();
    const x    = e.clientX - rect.left;
    tip.textContent = fmt(Math.max(0, xToTime(x, rect.width)));
    tip.style.display = 'block';
    // Keep tooltip inside the container
    tip.style.left = Math.min(x + 8, rect.width - 56) + 'px';
  });
  _wfWrap.addEventListener('mouseleave', () => { document.getElementById('wf-time-tip').style.display = 'none'; });

  // Zoom buttons
  document.getElementById('btn-zoom-in') .addEventListener('click', () => { applyZoom(S.zoom.level * 2,   0.5); renderWaveform(); renderOverlays(); _updatePlayhead(); });
  document.getElementById('btn-zoom-out').addEventListener('click', () => { applyZoom(S.zoom.level / 2,   0.5); renderWaveform(); renderOverlays(); _updatePlayhead(); });
  document.getElementById('btn-zoom-fit').addEventListener('click', () => { S.zoom.level=1; S.zoom.offset=0; updateZoomLabel(); renderWaveform(); renderOverlays(); _updatePlayhead(); });

  // Follow playhead toggle
  const followBtn = document.getElementById('btn-follow-playhead');
  followBtn.classList.add('active'); // reflect default-on state
  followBtn.addEventListener('click', () => {
    _followPlayhead = !_followPlayhead;
    followBtn.classList.toggle('active', _followPlayhead);
  });

  // Removal overlay event delegation
  const roLayer = document.getElementById('removal-overlays');
  roLayer.addEventListener('click', e => {
    const del = e.target.closest('.ro-del');
    if (del) { e.stopPropagation(); deleteRegion(del.closest('.ro').dataset.id); return; }
    const fill = e.target.closest('.ro-fill');
    if (!fill) return;
    e.stopPropagation();
    toggleRemoval(fill.closest('.ro').dataset.id);
  });
  roLayer.addEventListener('mousedown', e => {
    const handle = e.target.closest('.ro-handle');
    if (!handle) return;
    startHandleDrag(e, handle.closest('.ro').dataset.id, handle.dataset.side);
  });
}
