/**
 * word-edit.js — word-level delete-to-cut for VidTighten's transcript editor.
 *
 * Public API:
 *   getSelectedWordIds()   → string[]   IDs of currently selected .tx-word spans
 *   clearWordSelection()   → void       clear highlight and internal selection state
 *   deleteWords(wordIds)   → string     removal candidate ID (or null if nothing to do)
 *   initWordEdit()         → void       reset internal counter on state reset
 *
 * Selection is custom mouse-based (mousedown → mousemove → mouseup on .tx-word spans)
 * rather than relying on window.getSelection(), which is unreliable in WKWebView.
 */

import { S } from './state.js';

let _nextId  = 0;
let _wordMap = null;   // lazy Map<wordId, word> — built on first deleteWords call

// ── Custom selection state ───────────────────────────────────────────────────
let _customSelectedIds = new Set();
let _dragAnchorId      = null;
let _isDragging        = false;

// Drag-session cache: populated on mousedown, cleared on mouseup.
// Avoids O(N) querySelectorAll on every mousemove event during long drags.
let _dragEls  = /** @type {Element[]}  */ ([]);
let _dragIds  = /** @type {string[]}   */ ([]);

// ── DOM helpers ──────────────────────────────────────────────────────────────

/** Toggle .tx-word-selected class to match _customSelectedIds (uses cached list during drag). */
function _applyHighlight() {
  const els = _dragEls.length ? _dragEls
    : Array.from(document.querySelectorAll('.tx-word:not(.tx-filler)'));
  for (const el of els) {
    el.classList.toggle('tx-word-selected', _customSelectedIds.has(el.dataset.id));
  }
}

/**
 * Extend the selection from the anchor word to hoverId (inclusive),
 * updating _customSelectedIds.  Works regardless of drag direction.
 */
function _extendDragTo(hoverId) {
  const aIdx = _dragIds.indexOf(_dragAnchorId);
  const hIdx = _dragIds.indexOf(hoverId);
  if (aIdx === -1 || hIdx === -1) return;
  const lo = Math.min(aIdx, hIdx);
  const hi = Math.max(aIdx, hIdx);
  _customSelectedIds = new Set(_dragIds.slice(lo, hi + 1));
}

// ── Mouse event handlers (registered once at module load) ────────────────────

function _onWordMouseDown(e) {
  const el = e.target.closest?.('.tx-word');
  if (!el) {
    // Click outside any word: clear selection and abort any in-progress drag.
    // Not guarded by inControl — the drag state must always be reset here to
    // prevent a ghost drag rebuilding the selection on the next mousemove.
    const inControl = e.target.closest('button, input, select, textarea, [role=button]');
    if (!inControl) clearWordSelection();
    _isDragging   = false;
    _dragAnchorId = null;
    return;
  }
  // Filler words are toggled via click handler — do not pull them into word selection.
  // Letting the event proceed normally means the click event fires → filler toggles.
  if (el.classList.contains('tx-filler')) {
    clearWordSelection();
    _isDragging   = false;
    _dragAnchorId = null;
    return;
  }
  // Snapshot all deletable word elements once per drag (O(N) DOM scan, amortised).
  _dragEls  = Array.from(document.querySelectorAll('.tx-word:not(.tx-filler)'));
  _dragIds  = _dragEls.map(el => el.dataset.id);

  _dragAnchorId      = el.dataset.id;
  _isDragging        = true;
  _customSelectedIds = new Set([_dragAnchorId]);
  _applyHighlight();
  // Prevent browser from starting its own text selection (breaks WKWebView)
  e.preventDefault();
}

function _onWordMouseMove(e) {
  if (!_isDragging || !_dragAnchorId) return;
  const el = e.target.closest?.('.tx-word');
  if (!el || el.classList.contains('tx-filler')) return;
  _extendDragTo(el.dataset.id);
  _applyHighlight();
}

function _onWordMouseUp() {
  _isDragging = false;
  _dragEls    = [];
  _dragIds    = [];
}

// Register once at module load — safe because these are no-ops until _dragAnchorId is set.
// Guard for Node.js / vitest environments where document is not defined.
if (typeof document !== 'undefined') {
  document.addEventListener('mousedown', _onWordMouseDown);
  document.addEventListener('mousemove', _onWordMouseMove);
  document.addEventListener('mouseup',   _onWordMouseUp);
}

// ── Public API ───────────────────────────────────────────────────────────────

/** Reset the ID counter and word lookup cache on state reset / new file load. */
export function initWordEdit() {
  _nextId  = 0;
  _wordMap = null;
  clearWordSelection();
  _dragAnchorId = null;
  _isDragging   = false;
}

/**
 * Advance the word-edit ID counter to at least `minValue`.
 * Called after session restore to prevent ID collisions with existing
 * 'word'-type removal candidates that were persisted from a prior session.
 * (A page reload resets _nextId to 0, but the restored candidates can
 * already have IDs like "wd0", "wd1" — the counter must leap past them.)
 */
export function advanceWordEditCounter(minValue) {
  if (minValue > _nextId) _nextId = minValue;
}

/**
 * Return the word IDs of all currently selected .tx-word spans.
 * Selection is managed by the custom mouse handlers above.
 */
export function getSelectedWordIds() {
  return [..._customSelectedIds];
}

/** Clear the custom word selection and remove all highlight classes. */
export function clearWordSelection() {
  _customSelectedIds = new Set();
  _dragAnchorId      = null;
  // Remove highlight class from any remaining marked elements (no-op in Node.js)
  if (typeof document !== 'undefined') {
    document.querySelectorAll('.tx-word-selected')
      .forEach(el => el.classList.remove('tx-word-selected'));
  }
}

/**
 * Build a readable label from an array of word text strings.
 * CJK text (kanji, hiragana, katakana) → concatenate without spaces.
 * Latin/mixed → space-join; truncate to "first 3 words…" for >4 words.
 *
 * Exported so it can be unit-tested without a DOM environment.
 *
 * @param {string[]} texts — word text strings in display order
 * @returns {string}
 */
export function buildWordLabel(texts) {
  const joined = texts.join('');
  if (/[⺀-鿿぀-ヿ豈-﫿]/.test(joined)) return joined;
  if (texts.length > 4) return texts.slice(0, 3).join(' ') + '…';
  return texts.join(' ');
}

/**
 * Create a single "word" removal candidate covering the range of the given
 * word IDs, add it to S.removalCandidates, and enable it immediately.
 *
 * The `wordIds` back-pointer lets the transcript renderer mark the affected
 * spans with .tx-word-cut without mutating S.words.
 *
 * @param {string[]} wordIds — IDs of words to delete (from getSelectedWordIds)
 * @returns {string|null}    — the new removal candidate ID, or null if nothing done
 */
export function deleteWords(wordIds) {
  if (!wordIds || !wordIds.length) return null;

  // Build lookup Map lazily (S.words is immutable post-analysis)
  if (!_wordMap) _wordMap = new Map(S.words.map(w => [w.id, w]));

  // Resolve word objects — ignore IDs not found (defensive)
  const words = wordIds.map(id => _wordMap.get(id)).filter(Boolean);

  if (!words.length) return null;

  // Sort by time so start/end are correct regardless of selection direction
  words.sort((a, b) => a.start - b.start);

  const candidateId = `wd${_nextId++}`;
  const candidate = {
    id:      candidateId,
    start:   words[0].start,
    end:     words[words.length - 1].end,
    type:    'word',
    label:   buildWordLabel(words.map(w => w.text)),
    wordIds: words.map(w => w.id),
  };

  S.removalCandidates.push(candidate);
  S.removalsEnabled.add(candidateId);

  return candidateId;
}
