import { S } from './state.js';
import { $, fmt, fmtDur, plur, toast, api, apiFetch, dlBlob, _fmtBytes } from './utils.js';

let _saveTick = null;

// ── Telop text rebuild ────────────────────────────────────────────────────────

/**
 * Rebuild a telop entry's display text excluding deleted word IDs.
 *
 * When words are deleted via word-level editing, the audio is cut but the
 * original telop/subtitle text still contains those words.  This function
 * reconstructs the text from the remaining (non-deleted) words so exported
 * subtitles and telop FCPXML match the cut audio.
 *
 * If no words in the entry are deleted the original entry object is returned
 * unchanged (referential equality, safe to cache-check with ===).
 *
 * Space rule: insert a space between adjacent tokens only when BOTH are
 * non-CJK — mirrors Python's _join_word_texts and transcript.js spacing.
 *
 * @param {{id:string, start:number, end:number, text:string}} entry
 * @param {{id:string, seg_id:string|null, text:string}[]}     allWords
 * @param {Set<string>} deletedWordIdSet — word IDs currently being removed
 * @returns {{id:string, start:number, end:number, text:string}}
 */
export function rebuildTelopText(entry, allWords, deletedWordIdSet) {
  if (!deletedWordIdSet.size) return entry;

  const entryWords = allWords.filter(w => w.seg_id === entry.id);
  if (!entryWords.length) return entry;        // no word data — leave unchanged

  if (!entryWords.some(w => deletedWordIdSet.has(w.id))) return entry;

  const remaining = entryWords.filter(w => !deletedWordIdSet.has(w.id));
  if (!remaining.length) return { ...entry, text: '' };

  // Same range as _hasCJK in transcript.js: U+3000–U+9FFF, U+F900–U+FAFF
  const _isCJK = s => /[　-鿿豈-﫿]/.test(s);
  const parts   = remaining.map(w => w.text).filter(Boolean);
  if (!parts.length) return { ...entry, text: '' };

  const buf = [parts[0]];
  for (let i = 1; i < parts.length; i++) {
    if (!_isCJK(parts[i - 1]) && !_isCJK(parts[i])) buf.push(' ');
    buf.push(parts[i]);
  }
  return { ...entry, text: buf.join('') };
}

/**
 * Build keep-segments from removal regions with per-type padding.
 *
 * Each element of `removals` may be [start, end] or [start, end, type].
 * Word-type regions ('word') get zero padding — their boundaries are
 * already snapped to audio energy by refine_word_boundary on the backend.
 * All other types receive the full paddingMs on both sides.
 *
 * Mirror of Python's _build_segments_typed() in web.py.
 */
export function buildSegsJS(removals, dur, paddingMs=0) {
  if (!removals.length) return [[0, dur]];
  const ps = paddingMs / 1000;

  // Apply per-type padding before merging.
  const prepadded = [];
  for (const r of removals) {
    const [s, e, type] = r;
    if (type === 'word') {
      // Word deletions: zero padding — exact audio-snapped boundaries.
      if (e > s) prepadded.push([s, e]);
    } else {
      const ns = s + ps, ne = e - ps;
      if (ne > ns) prepadded.push([ns, ne]);
    }
  }
  if (!prepadded.length) return [[0, dur]];

  const sorted = prepadded.sort((a,b)=>a[0]-b[0]);
  const merged = [sorted[0].slice()];
  for (let i=1; i<sorted.length; i++) {
    const [s,e] = sorted[i];
    if (s <= merged[merged.length-1][1]) merged[merged.length-1][1]=Math.max(merged[merged.length-1][1],e);
    else merged.push([s,e]);
  }
  const keeps = [];
  if (merged[0][0] > 0.001) keeps.push([0, merged[0][0]]);
  for (let i=0; i<merged.length-1; i++) {
    const gs=merged[i][1], ge=merged[i+1][0];
    if (ge-gs>0.001) keeps.push([gs, ge]);
  }
  if (merged[merged.length-1][1] < dur-0.001) keeps.push([merged[merged.length-1][1], dur]);
  return keeps.filter(([s,e])=>e-s>=0.05);
}

export function updateLiveStats() {
  const dur = S.media?.duration;
  if (!dur) return;
  const padMs = parseInt($('s-padding').value);
  const enabledC = S.removalCandidates.filter(c => S.removalsEnabled.has(c.id));
  // Pass [start, end, type] so buildSegsJS can apply zero padding for word deletions.
  const enabledR = enabledC.map(c => [c.start, c.end, c.type]);
  const keeps = buildSegsJS(enabledR, dur, padMs);
  const kept    = keeps.reduce((s,[a,b])=>s+(b-a), 0);
  const nCuts   = enabledC.length;
  $('fs-orig').textContent  = fmtDur(dur);
  $('fs-kept').textContent  = fmtDur(kept);
  $('fs-cuts').textContent  = nCuts;
  const statHint = $('stat-hint');
  if (statHint) statHint.style.display = 'none';
}

export async function refreshCacheInfo() {
  const lbl = $('cache-size-label');
  if (!lbl) return;
  try {
    const r = await fetch('/api/cache/info');
    if (!r.ok) return;
    const d = await r.json();
    lbl.textContent =
      `Uploads: ${_fmtBytes(d.upload_bytes)}  ·  ` +
      `Exports: ${_fmtBytes(d.export_bytes)}  ·  ` +
      `Sessions: ${_fmtBytes(d.session_bytes)} (${d.session_count})  ·  ` +
      `Total: ${_fmtBytes(d.total_bytes)}`;
  } catch { lbl.textContent = 'Unable to load cache info'; }
}

export function scheduleAutosave() {
  clearTimeout(_saveTick);
  const statusEl = $('fs-save-status');
  if (statusEl) statusEl.textContent = '· Saving…';
  _saveTick = setTimeout(async () => {
    if (!S.filePath) { if (statusEl) statusEl.textContent = ''; return; }
    await api('/api/session/save', { path: S.filePath, state: {
      removalCandidates:  S.removalCandidates,
      telopEntries:       S.telopEntries,
      words:              S.words,
      waveformData:       S.waveformData,
      waveformThreshold:  S.waveformThreshold,
      waveformMaxAmp:     S.waveformMaxAmp,
      removalsEnabled:    [...S.removalsEnabled],
      telopsEnabled:      [...S.telopsEnabled],
    }});
    if (statusEl) {
      statusEl.textContent = '· Saved';
      setTimeout(() => { statusEl.textContent = ''; }, 2500);
    }
  }, 1500);
}

export function openDrawer()  {
  $('drawer').classList.add('open');
  $('drawer-back').classList.add('open');
  refreshCacheInfo();
}

export function closeDrawer() {
  $('drawer').classList.remove('open');
  $('drawer-back').classList.remove('open');
}
