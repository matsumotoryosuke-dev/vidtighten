/**
 * Transcript find & replace — operates on BOTH S.telopEntries[i].text (what
 * gets exported/shown for word-less segments) AND S.words[].text (what
 * word-level rendering actually displays), so a replace stays correct on
 * screen and in the export.
 *
 * Word-level Japanese tokens from Whisper are often much finer than a search
 * term (e.g. "AIエージェント" splits into separate "AI"/"エージェント"
 * tokens) — a naive per-word substring replace would silently miss those and
 * leave the on-screen text stale even though entry.text updated. Instead
 * _applyWordReplace joins an entry's words with the same spacing rule the
 * renderer uses, finds the match in that joined string, and redistributes:
 * the first word touched gets (its own unmatched prefix + replacement + the
 * last touched word's unmatched suffix), every word strictly between gets
 * blanked to ''. Word count/ids/timestamps are untouched — only .text moves —
 * so seek-to-word and audio-cut timing stay intact.
 *
 * Deliberately NOT wired into the removalCandidates undo/redo stack (snap()
 * in history.js doesn't capture telopEntries/words text, and adding an
 * 8000+-word deep clone to every history push would be real overhead for
 * every toggle in the app, not just text edits) — Replace All shows the
 * match count before it runs as the safety net instead.
 */
import { S } from './state.js';
import { $, toast } from './utils.js';

let _query        = '';
let _matches       = [];   // telopEntries currently containing _query
let _currentIndex  = -1;
let _onChange      = () => {};

// Same range as _hasCJK in transcript.js / _isCJK in export-module.js.
const _hasCJK = s => /[　-鿿豈-﫿]/.test(s);

export function _escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function _cssEscape(s) {
  return window.CSS?.escape ? CSS.escape(s) : s.replace(/[^a-zA-Z0-9_-]/g, '\\$&');
}

export function _containsQuery(text, query) {
  return !!text && text.toLowerCase().includes(query.toLowerCase());
}

/** Case-insensitive global replace. Returns {text, count}. */
export function _replaceOccurrences(text, query, replacement) {
  if (!text) return { text, count: 0 };
  const re = new RegExp(_escapeRegExp(query), 'gi');
  let count = 0;
  const out = text.replace(re, () => { count++; return replacement; });
  return { text: out, count };
}

/** Join word texts with the renderer's spacing rule (space only between two
 *  non-CJK tokens; a blanked word from an earlier replace contributes neither
 *  text nor a space, matching _renderWordsIntoSeg's guard), returning the
 *  joined string and each word's start offset. */
export function _joinWithOffsets(words) {
  let text = '';
  const offsets = [];
  for (let i = 0; i < words.length; i++) {
    offsets.push(text.length);
    text += words[i].text || '';
    const next = words[i + 1];
    if (next && words[i].text && next.text && !_hasCJK(words[i].text) && !_hasCJK(next.text)) text += ' ';
  }
  return { text, offsets };
}

/** Replace every occurrence of query across entryWords (words belonging to one
 *  telop entry, in order), redistributing across word boundaries when a match
 *  spans more than one token. Mutates the word objects' .text in place. */
export function _applyWordReplace(entryWords, query, replacement) {
  if (!entryWords.length) return;

  const { text: joined, offsets } = _joinWithOffsets(entryWords);
  const re = new RegExp(_escapeRegExp(query), 'gi');
  const matches = [...joined.matchAll(re)];
  if (!matches.length) return;

  // Right-to-left so mutating a later match doesn't invalidate earlier offsets.
  for (let m = matches.length - 1; m >= 0; m--) {
    const matchStart = matches[m].index;
    const matchEnd    = matchStart + matches[m][0].length;

    let firstIdx = -1, lastIdx = -1;
    for (let i = 0; i < entryWords.length; i++) {
      const wStart = offsets[i];
      const wEnd   = wStart + (entryWords[i].text || '').length;
      if (wEnd > matchStart && wStart < matchEnd) {
        if (firstIdx === -1) firstIdx = i;
        lastIdx = i;
      }
    }
    if (firstIdx === -1) continue;   // match fell entirely in inter-word spacing

    const firstW = entryWords[firstIdx];
    const lastW  = entryWords[lastIdx];
    const prefix = (firstW.text || '').slice(0, Math.max(0, matchStart - offsets[firstIdx]));
    const suffix = (lastW.text  || '').slice(Math.max(0, matchEnd - offsets[lastIdx]));

    firstW.text = prefix + replacement + suffix;
    for (let i = firstIdx + 1; i <= lastIdx; i++) entryWords[i].text = '';
  }
}

/** Replace every occurrence in entry.text and its word tokens (from `allWords`).
 *  Returns the number of occurrences replaced (per entry.text). */
export function _applyReplace(entry, allWords, query, replacement) {
  const { text, count } = _replaceOccurrences(entry.text, query, replacement);
  if (count === 0) return 0;
  entry.text = text;
  _applyWordReplace(allWords.filter(w => w.seg_id === entry.id), query, replacement);
  return count;
}

function _segEl(entryId) {
  return document.querySelector(`.tx-seg-inline[data-telop-id="${_cssEscape(entryId)}"]`);
}

export function initTxSearch({ onChange }) {
  _onChange = onChange || (() => {});

  const btn       = $('btn-tx-search');
  const bar       = $('tx-search-bar');
  const findEl    = $('tx-search-find');
  const replEl    = $('tx-search-replace');
  const countEl   = $('tx-search-count');
  const prevBtn   = $('tx-search-prev');
  const nextBtn   = $('tx-search-next');
  const closeBtn  = $('tx-search-close');
  const repOneBtn = $('tx-search-replace-one');
  const repAllBtn = $('tx-search-replace-all');
  if (!btn || !bar) return;   // page variant without the search bar — skip silently

  function _clearHighlights() {
    document.querySelectorAll('.tx-search-match, .tx-search-current')
      .forEach(el => el.classList.remove('tx-search-match', 'tx-search-current'));
  }

  function _updateHighlights() {
    _clearHighlights();
    _matches.forEach((entry, i) => {
      const el = _segEl(entry.id);
      if (!el) return;
      el.classList.add('tx-search-match');
      if (i === _currentIndex) el.classList.add('tx-search-current');
    });
  }

  function _scrollToCurrent() {
    _segEl(_matches[_currentIndex]?.id)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  function _updateCount() {
    countEl.textContent = !_query ? '' : (_matches.length ? `${_currentIndex + 1}/${_matches.length}` : '0/0');
  }

  function _updateReplaceButtons() {
    repOneBtn.disabled = !_matches.length;
    repAllBtn.disabled = !_matches.length;
  }

  function _recompute(query) {
    _query = query;
    _matches = _query ? S.telopEntries.filter(e => _containsQuery(e.text, _query)) : [];
    _currentIndex = _matches.length ? 0 : -1;
    _updateCount();
    _updateHighlights();
    _updateReplaceButtons();
    if (_currentIndex >= 0) _scrollToCurrent();
  }

  function _next() {
    if (!_matches.length) return;
    _currentIndex = (_currentIndex + 1) % _matches.length;
    _updateCount(); _updateHighlights(); _scrollToCurrent();
  }

  function _prev() {
    if (!_matches.length) return;
    _currentIndex = (_currentIndex - 1 + _matches.length) % _matches.length;
    _updateCount(); _updateHighlights(); _scrollToCurrent();
  }

  function _replaceOne() {
    const entry = _matches[_currentIndex];
    if (!entry) return;
    _applyReplace(entry, S.words, _query, replEl.value);
    _onChange();        // re-render with updated text first...
    _recompute(_query);  // ...then re-scan + re-highlight the fresh DOM
  }

  function _replaceAll() {
    if (!_matches.length) return;
    const targets = [..._matches];
    let count = 0;
    for (const entry of targets) count += _applyReplace(entry, S.words, _query, replEl.value);
    _onChange();
    toast(`Replaced ${plur_(count, 'occurrence')} across ${plur_(targets.length, 'segment')}`);
    _recompute(_query);
  }

  function plur_(n, word) { return n === 1 ? `1 ${word}` : `${n} ${word}s`; }

  function open() {
    bar.classList.add('open');
    btn.classList.add('active');
    findEl.focus();
    findEl.select();
    _recompute(findEl.value);
  }

  function close() {
    bar.classList.remove('open');
    btn.classList.remove('active');
    _query = ''; _matches = []; _currentIndex = -1;
    _clearHighlights();
  }

  btn.addEventListener('click', () => (bar.classList.contains('open') ? close() : open()));

  document.addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'f') {
      e.preventDefault();
      open();
    }
  });

  findEl.addEventListener('input', () => _recompute(findEl.value));
  findEl.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); e.shiftKey ? _prev() : _next(); }
    else if (e.key === 'Escape') { e.preventDefault(); close(); }
  });
  replEl.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); _replaceOne(); }
    else if (e.key === 'Escape') { e.preventDefault(); close(); }
  });

  prevBtn.addEventListener('click', _prev);
  nextBtn.addEventListener('click', _next);
  closeBtn.addEventListener('click', close);
  repOneBtn.addEventListener('click', _replaceOne);
  repAllBtn.addEventListener('click', _replaceAll);
}
