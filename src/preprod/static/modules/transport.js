import { S } from './state.js';
import { $, fmt, fmtDur } from './utils.js';

// ── Normal-playback skip timer ────────────────────────────────────────────────
// timeupdate fires every ~250ms; deletions shorter than that can slip through.
// We schedule a pre-fire setTimeout so even sub-250ms regions are jumped over.
let _normalSkipTimer = null;

function _clearNormalSkipTimer() {
  if (_normalSkipTimer) { clearTimeout(_normalSkipTimer); _normalSkipTimer = null; }
}

// Stored callbacks
let _onToggle = () => {};
let _player   = null;

// focusRegionInWaveform is in transcript.js — injected via init to avoid circular
let _focusRegionInWaveform = () => {};
let _getFocusedId          = () => null;

export function initTransportEvents({ onToggle, player, focusRegionInWaveform, getFocusedId }) {
  _onToggle              = onToggle;
  _player                = player;
  _focusRegionInWaveform = focusRegionInWaveform;
  _getFocusedId          = getFocusedId;
}

/**
 * Skip enabled removal regions during playback.
 *
 * There is exactly ONE playback control in the app (the floating transport bar
 * in the video preview), so this is the only skip mechanism needed — no
 * separate "Play Edited" mode.  Called on every `timeupdate` event from app.js
 * (~250ms cadence).  Two layers:
 *   1. Immediate jump when the playhead is already inside a removal region.
 *   2. Pre-fire setTimeout scheduled 30ms before the next upcoming region, so
 *      deletions shorter than 250ms (a single timeupdate interval) are not missed.
 *
 * Guards `next.id` membership in S.removalsEnabled inside the timer callback to
 * prevent stale jumps after a file switch clears the candidate list.
 */
export function handleNormalPlaybackTimeUpdate(currentTime) {
  if (!_player || _player.paused) { _clearNormalSkipTimer(); return; }

  const removals = S.removalCandidates
    .filter(c => S.removalsEnabled.has(c.id))
    .sort((a, b) => a.start - b.start);

  // Layer 1: already inside a removal → jump immediately
  for (const r of removals) {
    if (currentTime >= r.start && currentTime < r.end) {
      _clearNormalSkipTimer();
      _player.currentTime = r.end;
      return;
    }
  }

  // Layer 2: schedule a pre-fire jump for the next upcoming removal so
  // very short deletions aren't missed between timeupdate ticks.
  _clearNormalSkipTimer();
  const next = removals.find(r => r.start > currentTime);
  if (next) {
    const delay = Math.max(0, (next.start - currentTime) * 1000 - 30);
    if (delay < 2000) {   // only arm when the region is within 2 s
      _normalSkipTimer = setTimeout(() => {
        _normalSkipTimer = null;
        if (!_player || _player.paused) return;
        // Guard: removal may have been toggled off or file switched
        if (!S.removalsEnabled.has(next.id)) return;
        const ct = _player.currentTime;
        if (ct >= next.start - 0.3 && ct < next.end) {
          _player.currentTime = next.end;
        }
      }, delay);
    }
  }
}

export function renderSilenceList(container) {
  const focusedId = _getFocusedId();
  const silences = S.removalCandidates.filter(c => c.type === 'silence');
  const manuals  = S.removalCandidates.filter(c => c.type === 'manual');
  const all      = [...silences, ...manuals].sort((a, b) => a.start - b.start);

  if (!all.length) {
    container.innerHTML = '<div style="padding:24px;color:var(--text3);font-size:16px">No silence detected.</div>';
    return;
  }

  const frag = document.createDocumentFragment();

  for (const c of all) {
    const on  = S.removalsEnabled.has(c.id);
    const dur = c.end - c.start;

    const row = document.createElement('div');
    row.className = `region-card ${on ? 'rc-on' : 'rc-off'}${c.id === focusedId ? ' rc-focused' : ''}`;
    row.dataset.id = c.id;
    row.title = 'Click to focus in timeline';

    // Clicking the row focuses in waveform
    row.addEventListener('click', e => {
      if (e.target.closest('.rc-badge')) return;
      _focusRegionInWaveform(c);
    });

    // CUT / KEEP badge
    const badge = document.createElement('div');
    badge.className = 'rc-badge';
    badge.textContent = on ? 'CUT' : 'KEEP';
    badge.title = on ? 'Click to keep' : 'Click to cut';
    badge.addEventListener('click', e => { e.stopPropagation(); _onToggle(c.id); });

    // Timestamp
    const timeEl = document.createElement('div');
    timeEl.className = 'rc-time-col';
    timeEl.textContent = fmt(c.start);

    // Type label
    const typeEl = document.createElement('div');
    typeEl.className = 'rc-type';
    typeEl.textContent = c.type === 'manual' ? 'manual' : 'silence';

    // Duration
    const durEl = document.createElement('div');
    durEl.className = 'rc-dur';
    durEl.textContent = fmtDur(dur);

    row.appendChild(badge);
    row.appendChild(timeEl);
    row.appendChild(typeEl);
    row.appendChild(durEl);
    frag.appendChild(row);
  }

  container.innerHTML = '';
  container.appendChild(frag);
}
