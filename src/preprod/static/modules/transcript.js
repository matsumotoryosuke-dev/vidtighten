import { S, collectDeletedWordIds } from './state.js';
import { $, fmt, fmtDur, plur, escH, setToggleA11y } from './utils.js';
import { clampZoomOffset, updateZoomLabel } from './zoom.js';

let _focusedId     = null;
let _lastActiveSeg = null;

let _lastActiveWord  = null;
let _renderedWords   = [];              // S.words filtered to those with a valid seg_id
let _wordStartsCache = [];              // _renderedWords[i].start — binary-search index
let _wordElMap       = new Map();       // wordId  → DOM span (O(1) RAF highlight)
let _wordsBySegId    = new Map();       // seg_id  → word[]  (O(1) per-segment render)

// Segment element cache: populated during renderFullTranscript for O(log n) + O(1)
// segment highlight in syncTranscriptHighlight (replaces querySelectorAll scan).
let _segElCache    = /** @type {Element[]} */  ([]);   // index-aligned with _segDataCache
let _segDataCache  = /** @type {{start:number,end:number}[]} */ ([]);  // sorted by start

let _onToggle = () => {};
let _onSeek   = () => {};

let _onRenderSilenceList  = (container) => {};

export function setTranscriptDeps({ onRenderSilenceList }) {
  _onRenderSilenceList = onRenderSilenceList;
}

// ── Transcript display prefs (show/hide deleted silence & filler words) ──────
// Deleted content is hidden by default for a cleaner read. This toggle is
// purely cosmetic — it never touches S.removalsEnabled, so what actually gets
// cut on export/playback is completely unaffected by what the panel displays.

/** Read persisted show/hide prefs into S. Safe to call even without localStorage. */
export function loadTranscriptDisplayPrefs() {
  try {
    S.showDeletedSilence = localStorage.getItem('vt_show_deleted_silence') === '1';
    S.showDeletedFillers = localStorage.getItem('vt_show_deleted_fillers') === '1';
  } catch { /* localStorage unavailable — keep the false defaults from state.js */ }
}

function _syncContextMenuChecks() {
  $('tx-ctx-silence')?.classList.toggle('checked', S.showDeletedSilence);
  $('tx-ctx-fillers')?.classList.toggle('checked', S.showDeletedFillers);
}

function _closeContextMenu() {
  $('tx-context-menu')?.classList.remove('open');
}

function _openContextMenu(x, y) {
  const menu = $('tx-context-menu');
  if (!menu) return;
  _syncContextMenuChecks();
  menu.style.left = x + 'px';
  menu.style.top  = y + 'px';
  menu.classList.add('open');
  // Clamp to viewport using the menu's real (now-laid-out) size.
  const r  = menu.getBoundingClientRect();
  const vw = window.innerWidth, vh = window.innerHeight;
  if (r.right  > vw) menu.style.left = Math.max(8, vw - r.width  - 8) + 'px';
  if (r.bottom > vh) menu.style.top  = Math.max(8, vh - r.height - 8) + 'px';
}

/** Wire the right-click context menu on the transcript panel. Call once at init. */
export function initTranscriptContextMenu() {
  const scroll = $('tx-scroll');
  const menu   = $('tx-context-menu');
  const optBtn = $('btn-tx-options');
  if (!scroll || !menu) return;

  scroll.addEventListener('contextmenu', e => {
    e.preventDefault();
    _openContextMenu(e.clientX, e.clientY);
  });
  // Discoverable click-trigger (⋮ button in the transcript toolbar) — opens
  // the same menu anchored below the button, for users who never right-click.
  optBtn?.addEventListener('click', () => {
    const r = optBtn.getBoundingClientRect();
    _openContextMenu(r.left, r.bottom + 4);
  });

  document.addEventListener('click', e => {
    if (menu.classList.contains('open') && !menu.contains(e.target) && e.target !== optBtn) {
      _closeContextMenu();
    }
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') _closeContextMenu();
  });

  $('tx-ctx-silence')?.addEventListener('click', () => {
    S.showDeletedSilence = !S.showDeletedSilence;
    try { localStorage.setItem('vt_show_deleted_silence', S.showDeletedSilence ? '1' : '0'); } catch {}
    _closeContextMenu();
    renderTranscript();
  });
  $('tx-ctx-fillers')?.addEventListener('click', () => {
    S.showDeletedFillers = !S.showDeletedFillers;
    try { localStorage.setItem('vt_show_deleted_fillers', S.showDeletedFillers ? '1' : '0'); } catch {}
    _closeContextMenu();
    renderTranscript();
  });
}

function _buildWordCaches() {
  // Only include words that are assigned to a segment (seg_id != null).
  // Words with seg_id:null are never rendered in the transcript, so keeping them
  // in _wordStartsCache would cause the RAF highlight to de-highlight the previous
  // word without lighting up a new one when playback passes through such a word.
  _renderedWords   = S.words.filter(w => w.seg_id != null);
  _wordStartsCache = _renderedWords.map(w => w.start);
  _wordElMap.clear();
  _wordsBySegId.clear();
  _lastActiveWord = null;
  // Group words by seg_id once — render loop uses this instead of .filter().
  // Skip null-seg words: they have no containing segment to render into.
  for (const w of _renderedWords) {
    const bucket = _wordsBySegId.get(w.seg_id);
    if (bucket) bucket.push(w);
    else _wordsBySegId.set(w.seg_id, [w]);
  }
}

export function renderTranscript() {
  const container = $('tx-content');
  const empty     = $('tx-empty');
  const dur       = S.media?.duration || 0;
  const hasTranscript = S.telopEntries.length > 0;

  if (!S.removalCandidates.length && !hasTranscript) {
    container.innerHTML = '';
    empty.style.display = 'flex';
    return;
  }
  empty.style.display = 'none';

  if (hasTranscript) {
    // Full transcript mode (Whisper available)
    renderFullTranscript(container, dur);
  } else {
    // Silence-only list mode (no Whisper transcript)
    _onRenderSilenceList(container);
  }
}

/** True if s contains any Hiragana, Katakana, or CJK Unified character (U+3040+). */
const _hasCJK = s => /[\u3000-\u9fff\uf900-\ufaff]/.test(s);

/**
 * Render word-level <span class="tx-word"> elements inside a .tx-seg-inline.
 * Words that time-overlap a filler candidate get toggle behaviour;
 * all other words seek the player to their start time on click.
 * Populates _wordElMap for O(1) RAF-tick highlight updates.
 */
function _renderWordsIntoSeg(segEl, segWords, segFillers, cutWordIds) {
  for (let i = 0; i < segWords.length; i++) {
    const w = segWords[i];
    const wordEl = document.createElement('span');
    wordEl.className     = 'tx-word';
    wordEl.dataset.id    = w.id;
    wordEl.dataset.start = w.start;
    wordEl.dataset.end   = w.end;
    wordEl.textContent   = w.text;

    if (w.confidence !== null && w.confidence !== undefined && w.confidence < 0.5) {
      wordEl.classList.add('tx-word-uncertain');
    }

    // Strike-through: word is inside an enabled 'word' removal candidate
    if (cutWordIds.has(w.id)) {
      wordEl.classList.add('tx-word-cut');
    }

    // Time-based filler overlap check
    const matchFiller = segFillers.find(
      f => w.start >= f.start - 0.15 && w.end <= f.end + 0.15
    );
    // Deleted (on=true) fillers hidden by S.showDeletedFillers render as plain
    // words (seek-on-click, no filler styling) — cosmetic only, the underlying
    // removal candidate and its enabled state are completely untouched.
    const matchFillerOn = matchFiller && S.removalsEnabled.has(matchFiller.id);
    if (matchFiller && (!matchFillerOn || S.showDeletedFillers)) {
      const on = matchFillerOn;
      wordEl.classList.add('tx-filler');
      if (!on) wordEl.classList.add('off');
      else     wordEl.classList.add('tx-filler-cut');   // deleted + shown → strikethrough
      wordEl.dataset.id = matchFiller.id;   // override: toggle handler reads data-id
      wordEl.title = `Filler: "${w.text}"  ·  click to toggle`;
      setToggleA11y(wordEl, on,
        `Filler "${w.text}", ${on ? 'marked for removal' : 'kept'}`);
    } else {
      // Show confidence alongside timestamp for uncertain words so users can
      // quickly evaluate whether a low-confidence word needs re-checking.
      const conf = (w.confidence !== null && w.confidence !== undefined) ? w.confidence : null;
      wordEl.title = (conf !== null && conf < 0.5)
        ? `${fmt(w.start)}  ·  confidence ${Math.round(conf * 100)}%`
        : fmt(w.start);
    }

    _wordElMap.set(w.id, wordEl);
    segEl.appendChild(wordEl);
    // Inter-word space: insert only when BOTH current and next token are non-CJK.
    // Matches the _join_word_texts rule in segments.py:
    //   "GitHub" + "プロジェクト" → "GitHubプロジェクト" (no space at Latin→CJK boundary)
    //   "Claude" + "Code"        → "Claude Code"        (space between Latin words)
    //   "です"   + "ね"          → "ですね"             (no space between CJK tokens)
    // Blanked words (tx-search replace merged their text into a neighbor) are
    // skipped — an empty string isn't CJK either, so without this guard a
    // blanked word between two non-CJK words would still get a phantom space.
    const nextW = segWords[i + 1];
    if (nextW && w.text && nextW.text && !_hasCJK(w.text) && !_hasCJK(nextW.text)) {
      segEl.appendChild(document.createTextNode(' '));
    }
  }
}

export function renderFullTranscript(container, dur) {
  _buildWordCaches();
  _segElCache   = [];
  _segDataCache = [];
  const segs    = S.telopEntries;
  const silences = S.removalCandidates.filter(c => c.type === 'silence');
  const fillers  = S.removalCandidates.filter(c => c.type === 'filler');

  // Word IDs that are inside enabled 'word' removal candidates → strike-through
  const cutWordIds = collectDeletedWordIds();

  // Determine which fillers live textually inside a segment
  const inlineFillerIds = new Set();
  for (const seg of segs) {
    for (const f of fillers) {
      if (f.label && f.label !== '〜' &&
          f.start >= seg.start - 0.15 && f.end <= seg.end + 0.15 &&
          seg.text.includes(f.label)) {
        inlineFillerIds.add(f.id);
      }
    }
  }

  // Build sorted event list
  const events = [];
  for (const seg of segs)  events.push({kind:'seg',    item:seg,    t:seg.start});
  for (const sil of silences) events.push({kind:'sil', item:sil,    t:sil.start});
  for (const f of fillers) {
    if (!inlineFillerIds.has(f.id)) events.push({kind:'filler', item:f, t:f.start});
  }
  events.sort((a, b) => a.t - b.t);

  // Group events into paragraphs.
  // A silence >= PARA_BREAK_S becomes a paragraph separator.
  const PARA_BREAK_S = 2.0;
  const paragraphs = [];
  let curPara = {items: [], sepBefore: null};

  for (const ev of events) {
    if (ev.kind === 'sil') {
      const d = ev.item.end - ev.item.start;
      if (d >= PARA_BREAK_S) {
        paragraphs.push(curPara);
        curPara = {items: [], sepBefore: ev.item};
      } else {
        curPara.items.push({kind:'sil-inline', item:ev.item});
      }
    } else {
      curPara.items.push({kind: ev.kind, item: ev.item});
    }
  }
  paragraphs.push(curPara);

  const frag = document.createDocumentFragment();

  for (const para of paragraphs) {
    // Long-silence separator before this paragraph — deleted (on=true) silence
    // hidden by S.showDeletedSilence renders nothing here (cosmetic only; the
    // paragraph break itself still happens so text flow stays readable).
    if (para.sepBefore) {
      const sil = para.sepBefore;
      const on  = S.removalsEnabled.has(sil.id);
      if (!on || S.showDeletedSilence) {
        const sep = document.createElement('div');
        sep.className  = `tx-sil-sep ${on ? 'on' : 'off'}`;
        sep.dataset.id = sil.id;
        sep.textContent = `// ${fmtDur(sil.end - sil.start)} //`;
        sep.title = `${fmt(sil.start)} – ${fmt(sil.end)}  ·  click to toggle`;
        setToggleA11y(sep, on, `Long silence ${fmtDur(sil.end - sil.start)} at ${fmt(sil.start)}, ${on ? 'marked for removal' : 'kept'}`);
        frag.appendChild(sep);
      }
    }

    if (para.items.length === 0) continue;

    // Paragraph block
    const paraDiv = document.createElement('div');
    paraDiv.className = 'tx-paragraph';

    // Timestamp anchor
    const firstSeg  = para.items.find(it => it.kind === 'seg');
    const anyItem   = firstSeg || para.items[0];
    if (anyItem) {
      const timeEl = document.createElement('div');
      timeEl.className = 'tx-para-time';
      timeEl.textContent = fmt(anyItem.item.start);
      paraDiv.appendChild(timeEl);
    }

    const contentEl = document.createElement('div');
    contentEl.className = 'tx-para-content';

    for (const pitem of para.items) {

      if (pitem.kind === 'seg') {
        // Sentence span
        const seg  = pitem.item;
        const span = document.createElement('span');
        span.className    = 'tx-seg-inline';
        span.dataset.start = seg.start;
        span.dataset.end   = seg.end;
        span.dataset.telopId = seg.id;
        span.setAttribute('aria-label', `${fmt(seg.start)}: ${seg.text}`);

        // Inline filler highlights
        const segFillers = fillers.filter(f =>
          inlineFillerIds.has(f.id) &&
          f.start >= seg.start - 0.1 && f.end <= seg.end + 0.1
        );

        // Word-level rendering when Whisper word timestamps are available
        const segWords = _wordsBySegId.get(seg.id) ?? [];
        if (segWords.length > 0) {
          // Individual .tx-word spans handle seek; making the container a button
          // suppresses text/word selection in WKWebView — only set for word-less segs.
          _renderWordsIntoSeg(span, segWords, segFillers, cutWordIds);
        } else if (segFillers.length === 0) {
          span.tabIndex = 0;
          span.setAttribute('role', 'button');
          span.textContent = seg.text;
        } else {
          span.tabIndex = 0;
          span.setAttribute('role', 'button');
          // Fallback: substring-match filler chip rendering (no word timestamps)
          let remaining = seg.text;
          for (const f of segFillers) {
            const idx = remaining.indexOf(f.label);
            if (idx >= 0) {
              if (idx > 0) span.appendChild(document.createTextNode(remaining.slice(0, idx)));
              const on = S.removalsEnabled.has(f.id);
              // Deleted + hidden → plain text, no filler styling (cosmetic only).
              if (on && !S.showDeletedFillers) {
                span.appendChild(document.createTextNode(f.label));
              } else {
                const fw = document.createElement('span');
                fw.className  = `tx-filler${on ? ' tx-filler-cut' : ' off'}`;
                fw.textContent = f.label;
                fw.dataset.id  = f.id;
                fw.title = `Filler: "${f.label}"  ·  click to toggle`;
                setToggleA11y(fw, on, `Filler "${f.label}", ${on ? 'marked for removal' : 'kept'}`);
                span.appendChild(fw);
              }
              remaining = remaining.slice(idx + f.label.length);
            }
          }
          if (remaining) span.appendChild(document.createTextNode(remaining));
        }

        // Register in segment cache for O(log n) RAF highlight (avoids querySelectorAll)
        _segElCache.push(span);
        _segDataCache.push({ start: seg.start, end: seg.end });

        contentEl.appendChild(span);
        contentEl.appendChild(document.createTextNode(' '));

      } else if (pitem.kind === 'sil-inline') {
        // Inline silence marker: // 0.8s // — deleted+hidden renders nothing.
        const sil = pitem.item;
        const on  = S.removalsEnabled.has(sil.id);
        if (!on || S.showDeletedSilence) {
          const marker = document.createElement('span');
          marker.className  = `tx-sil-inline ${on ? 'on' : 'off'}`;
          marker.dataset.id = sil.id;
          marker.textContent = `// ${fmtDur(sil.end - sil.start)} //`;
          marker.title = `${fmt(sil.start)} – ${fmt(sil.end)}  ·  click to toggle`;
          setToggleA11y(marker, on, `Silence ${fmtDur(sil.end - sil.start)} at ${fmt(sil.start)}, ${on ? 'marked for removal' : 'kept'}`);
          contentEl.appendChild(marker);
          contentEl.appendChild(document.createTextNode(' '));
        }

      } else if (pitem.kind === 'filler') {
        // Standalone filler chip — deleted+hidden renders nothing (cosmetic
        // only; a standalone filler has no surrounding sentence text to fall
        // back to, unlike inline fillers which degrade to plain text).
        const f  = pitem.item;
        const on = S.removalsEnabled.has(f.id);
        if (!on || S.showDeletedFillers) {
          const chip = document.createElement('span');
          chip.className  = `tx-filler${on ? ' tx-filler-cut' : ' off'}`;
          chip.dataset.id = f.id;
          chip.textContent = f.label === '〜' ? `<filler ~${fmtDur(f.end - f.start)}>` : f.label;
          chip.title = `Filler: "${f.label}"  ${fmt(f.start)}–${fmt(f.end)}  ·  click to toggle`;
          setToggleA11y(chip, on, `Filler word "${f.label}" at ${fmt(f.start)}, ${on ? 'marked for removal' : 'kept'}`);
          contentEl.appendChild(chip);
          contentEl.appendChild(document.createTextNode(' '));
        }
      }
    }

    paraDiv.appendChild(contentEl);
    frag.appendChild(paraDiv);
  }

  container.innerHTML = '';
  container.appendChild(frag);
}

export function focusRegionInWaveform(region) {
  const dur = S.media?.duration || 1;
  const PAD = Math.max(1.5, (region.end - region.start) * 1.2);
  const viewStart = Math.max(0,   region.start - PAD);
  const viewEnd   = Math.min(dur, region.end   + PAD);
  const viewSpan  = viewEnd - viewStart;

  // Zoom so this region + context fills the waveform
  S.zoom.level  = Math.min(32, dur / viewSpan);
  S.zoom.offset = viewStart / dur;
  clampZoomOffset();
  updateZoomLabel();

  // Seek player to region start
  const player = document.getElementById('player');
  if (player && player.src) player.currentTime = region.start;

  // Highlight the focused card in silence-list view
  _focusedId = region.id;
  document.querySelectorAll('.region-card').forEach(el => {
    el.classList.toggle('rc-focused', el.dataset.id === region.id);
  });

  // Highlight the focused sentence in full transcript view
  const rt = region.start;
  document.querySelectorAll('.tx-seg-inline').forEach(el => {
    const s = parseFloat(el.dataset.start ?? -1);
    const e = parseFloat(el.dataset.end   ?? -1);
    const match = rt >= s && rt < e;
    el.classList.toggle('tx-seg-focused', match);
    if (match) el.closest('.tx-paragraph')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });

  // Also highlight inline silence / filler that matches this region
  let silenceScrollDone = false;
  document.querySelectorAll('.tx-filler, .tx-sil-inline, .tx-sil-sep').forEach(el => {
    const match = el.dataset.id === region.id;
    el.classList.toggle('tx-item-focused', match);
    if (match && !silenceScrollDone) {
      silenceScrollDone = true;
      const scrollTarget = el.classList.contains('tx-sil-sep')
        ? el
        : (el.closest('.tx-paragraph') || el);
      scrollTarget.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  });

  _onUpdateTransportUI();
}

export function getFocusedId() { return _focusedId; }
export function setFocusedId(id) { _focusedId = id; }

export function syncTranscriptHighlight() {
  const player = document.getElementById('player');
  if (!player) return;
  const ct = player.currentTime;

  // ── Segment highlight — O(log n) binary search + O(1) Map lookup ─────────
  // Fallback to querySelectorAll if cache not yet populated (e.g. silence-only mode).
  let activeEl = null;
  if (_segDataCache.length > 0) {
    // Binary search: find last segment whose start ≤ ct
    let lo = 0, hi = _segDataCache.length - 1, bestSeg = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (_segDataCache[mid].start <= ct) { bestSeg = mid; lo = mid + 1; }
      else hi = mid - 1;
    }
    const newActive = (bestSeg >= 0 && ct < _segDataCache[bestSeg].end)
      ? _segElCache[bestSeg] : null;

    if (newActive !== _lastActiveSeg) {
      if (_lastActiveSeg) _lastActiveSeg.classList.remove('tx-seg-active');
      _lastActiveSeg = newActive;
      if (newActive) {
        newActive.classList.add('tx-seg-active');
        (newActive.closest('.tx-paragraph') || newActive)
          .scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
    activeEl = newActive;
  } else {
    // Fallback: silence-only mode or cache not yet built
    document.querySelectorAll('.tx-seg-inline').forEach(el => {
      const start = parseFloat(el.dataset.start ?? -1);
      const end   = parseFloat(el.dataset.end   ?? -1);
      const active = ct >= start && ct < end;
      el.classList.toggle('tx-seg-active', active);
      if (active) activeEl = el;
    });
    if (activeEl && activeEl !== _lastActiveSeg) {
      _lastActiveSeg = activeEl;
      (activeEl.closest('.tx-paragraph') || activeEl)
        .scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    if (!activeEl) _lastActiveSeg = null;
  }

  // ── Word-level highlight (binary search — O(log n) + 2 class toggles per tick) ──
  if (_wordStartsCache.length === 0) return;

  let lo = 0, hi = _wordStartsCache.length - 1, bestIdx = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (_wordStartsCache[mid] <= ct) { bestIdx = mid; lo = mid + 1; }
    else hi = mid - 1;
  }

  const activeWord = (bestIdx >= 0 && ct < _renderedWords[bestIdx].end)
    ? _renderedWords[bestIdx] : null;

  if (activeWord?.id !== _lastActiveWord?.id) {
    if (_lastActiveWord) {
      const oldEl = _wordElMap.get(_lastActiveWord.id);
      if (oldEl) oldEl.classList.remove('tx-word-active');
    }
    _lastActiveWord = activeWord;
    if (activeWord) {
      const newEl = _wordElMap.get(activeWord.id);
      if (newEl) newEl.classList.add('tx-word-active');
    }
  }
}

export function initTranscriptEvents({ onToggle, onSeek }) {
  _onToggle = onToggle;
  _onSeek   = onSeek;

  const c = $('tx-content');
  c.addEventListener('click', e => {
    // Filler / silence toggle (highest priority)
    const tog = e.target.closest('.tx-filler, .tx-sil-inline, .tx-sil-sep');
    if (tog) { _onToggle(tog.dataset.id); return; }
    // Word-level seek (higher priority than sentence-level)
    const word = e.target.closest('.tx-word');
    if (word && !word.classList.contains('tx-filler')) {
      _onSeek(parseFloat(word.dataset.start));
      return;
    }
    // Segment-level click: find the word span nearest to the click X position.
    // This handles clicks on inter-word spaces (text nodes make e.target = container).
    const seg = e.target.closest('.tx-seg-inline');
    if (seg) {
      const words = seg.querySelectorAll('.tx-word');
      if (words.length > 0) {
        const cx = e.clientX;
        let best = null, bestDist = Infinity;
        words.forEach(w => {
          const r = w.getBoundingClientRect();
          const dist = Math.abs(cx - (r.left + r.right) / 2);
          if (dist < bestDist) { bestDist = dist; best = w; }
        });
        if (best) { _onSeek(parseFloat(best.dataset.start)); return; }
      }
      _onSeek(parseFloat(seg.dataset.start));
    }
  });
  c.addEventListener('keydown', e => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const tog = e.target.closest('.tx-filler, .tx-sil-inline, .tx-sil-sep');
    if (tog) { e.preventDefault(); _onToggle(tog.dataset.id); return; }
    const word = e.target.closest('.tx-word');
    if (word && !word.classList.contains('tx-filler')) {
      e.preventDefault(); _onSeek(parseFloat(word.dataset.start)); return;
    }
    const seg = e.target.closest('.tx-seg-inline');
    if (seg) {
      e.preventDefault();
      const firstWord = seg.querySelector('.tx-word');
      _onSeek(firstWord ? parseFloat(firstWord.dataset.start) : parseFloat(seg.dataset.start));
    }
  });
}

export function invalidateWordCaches() {
  _renderedWords   = [];
  _wordStartsCache = [];
  _wordElMap.clear();
  _wordsBySegId.clear();
  _lastActiveWord  = null;
  _segElCache      = [];
  _segDataCache    = [];
  _lastActiveSeg   = null;
}
