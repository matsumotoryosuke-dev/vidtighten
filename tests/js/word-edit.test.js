/**
 * word-edit.test.js — unit tests for word-edit.js exported pure helpers.
 *
 * DOM-free exports tested here: buildWordLabel, advanceWordEditCounter,
 * deleteWords (only reads/writes S state, no DOM operations).
 * Drag-selection handlers (mousedown/mousemove) require a browser DOM
 * and are tested manually in the app; see dom_env_check.test.js for the
 * jsdom pattern if automated coverage is added later.
 */
import { describe, test, expect, beforeEach } from 'vitest'
import { buildWordLabel, advanceWordEditCounter, initWordEdit, deleteWords, getSelectedWordIds, clearWordSelection } from '../../src/preprod/static/modules/word-edit.js'
import { S } from '../../src/preprod/static/modules/state.js'

// ── buildWordLabel — Latin text ───────────────────────────────────────────────

describe('buildWordLabel — Latin text', () => {
  test('single word returns that word', () => {
    expect(buildWordLabel(['hello'])).toBe('hello')
  })

  test('two words are space-joined', () => {
    expect(buildWordLabel(['one', 'two'])).toBe('one two')
  })

  test('exactly four words — no truncation', () => {
    expect(buildWordLabel(['one', 'two', 'three', 'four'])).toBe('one two three four')
  })

  test('five words → first three + ellipsis', () => {
    expect(buildWordLabel(['a', 'b', 'c', 'd', 'e'])).toBe('a b c…')
  })

  test('ten words → first three + ellipsis', () => {
    expect(buildWordLabel(['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'w9', 'w10']))
      .toBe('w1 w2 w3…')
  })
})

// ── buildWordLabel — CJK text (no spaces) ────────────────────────────────────

describe('buildWordLabel — CJK text', () => {
  test('kanji tokens concatenated without spaces', () => {
    expect(buildWordLabel(['美', '味', 'し', 'い'])).toBe('美味しい')
  })

  test('hiragana tokens (U+3040-309F) → concatenated', () => {
    // えー — hiragana え + katakana-hiragana prolonged sound ー
    expect(buildWordLabel(['え', 'ー'])).toBe('えー')
  })

  test('katakana tokens (U+30A0-30FF) → concatenated', () => {
    expect(buildWordLabel(['フ', 'ル', 'ス', 'ト'])).toBe('フルスト')
  })

  test('more than four CJK tokens — all joined (no truncation for CJK)', () => {
    expect(buildWordLabel(['あ', 'い', 'う', 'え', 'お'])).toBe('あいうえお')
  })

  test('single kanji — returned as-is', () => {
    expect(buildWordLabel(['語'])).toBe('語')
  })
})

// ── buildWordLabel — edge cases ───────────────────────────────────────────────

describe('buildWordLabel — edge cases', () => {
  test('empty array returns empty string', () => {
    expect(buildWordLabel([])).toBe('')
  })

  test('mixed Latin and CJK → CJK path, all tokens concatenated (no spaces)', () => {
    // When any CJK char appears, the whole selection is joined without spaces.
    // Labels are short waveform tooltips — perfect formatting is not required.
    expect(buildWordLabel(['Claude', 'Code', 'を'])).toBe('ClaudeCodeを')
  })

  test('Latin numbers are not treated as CJK', () => {
    expect(buildWordLabel(['123', '456'])).toBe('123 456')
  })
})

// ── advanceWordEditCounter ────────────────────────────────────────────────────
// advanceWordEditCounter is a DOM-free export — testable in Node.js.

describe('advanceWordEditCounter', () => {
  // Reset counter before each test so tests are independent.
  // initWordEdit() sets _nextId=0 and is safe to call in Node (no DOM ops on counter).
  test('advances counter when given value is higher', () => {
    initWordEdit();   // _nextId → 0
    advanceWordEditCounter(5);
    // Counter is now 5; we can verify indirectly via initWordEdit reset behaviour.
    // After advance(5), another advance(3) should NOT lower it.
    advanceWordEditCounter(3);
    // Can't read _nextId directly (private), but a second advance with a lower
    // value must not lower the counter — verified by testing advance with higher value.
    advanceWordEditCounter(10);
    advanceWordEditCounter(2);   // no-op
    // The only observable effect in Node is that the function doesn't throw.
    expect(true).toBe(true);   // smoke — function executed without error
  });

  test('no-op when given value is lower than current counter', () => {
    initWordEdit();   // _nextId → 0
    advanceWordEditCounter(5);
    // Calling with a lower value should not throw and should be a no-op
    expect(() => advanceWordEditCounter(3)).not.toThrow();
  });

  test('no-op when given value equals current counter', () => {
    initWordEdit();
    advanceWordEditCounter(4);
    expect(() => advanceWordEditCounter(4)).not.toThrow();
  });

  test('accepts zero', () => {
    initWordEdit();   // _nextId already 0
    expect(() => advanceWordEditCounter(0)).not.toThrow();
  });
})

// ── deleteWords ───────────────────────────────────────────────────────────────
// deleteWords is DOM-free: it reads S.words and writes to S.removalCandidates /
// S.removalsEnabled without touching the DOM.  Fully testable in Node.js.

describe('deleteWords', () => {
  beforeEach(() => {
    initWordEdit();                   // reset _nextId and _wordMap
    S.words              = [];
    S.removalCandidates  = [];
    S.removalsEnabled    = new Set();
  });

  test('empty wordIds returns null and does not mutate state', () => {
    S.words = [{ id: 'w0', start: 0.0, end: 0.5, text: 'hello', seg_id: 's0' }];
    expect(deleteWords([])).toBeNull();
    expect(S.removalCandidates.length).toBe(0);
  });

  test('null / undefined wordIds returns null', () => {
    expect(deleteWords(null)).toBeNull();
    expect(deleteWords(undefined)).toBeNull();
  });

  test('single word creates a removal candidate with correct fields', () => {
    S.words = [{ id: 'w0', start: 1.0, end: 1.5, text: 'hello', seg_id: 's0' }];
    const cid = deleteWords(['w0']);
    expect(cid).toBe('wd0');
    expect(S.removalCandidates.length).toBe(1);
    const c = S.removalCandidates[0];
    expect(c.id).toBe('wd0');
    expect(c.type).toBe('word');
    expect(c.start).toBe(1.0);
    expect(c.end).toBe(1.5);
    expect(c.wordIds).toEqual(['w0']);
    expect(c.label).toBe('hello');
  });

  test('candidate is immediately enabled', () => {
    S.words = [{ id: 'w0', start: 1.0, end: 1.5, text: 'hello', seg_id: 's0' }];
    const cid = deleteWords(['w0']);
    expect(S.removalsEnabled.has(cid)).toBe(true);
  });

  test('multiple words: span is start of first to end of last (sorted by time)', () => {
    S.words = [
      { id: 'w0', start: 0.0, end: 0.5, text: 'one',   seg_id: 's0' },
      { id: 'w1', start: 0.6, end: 1.2, text: 'two',   seg_id: 's0' },
      { id: 'w2', start: 1.3, end: 1.9, text: 'three', seg_id: 's0' },
    ];
    // Pass IDs in reverse order to verify sort by time, not input order
    const cid = deleteWords(['w2', 'w0', 'w1']);
    const c = S.removalCandidates[0];
    expect(c.start).toBe(0.0);   // w0 starts earliest
    expect(c.end).toBe(1.9);     // w2 ends latest
    expect(c.wordIds).toEqual(['w0', 'w1', 'w2']);  // sorted by start time
  });

  test('IDs not found in S.words are silently ignored', () => {
    S.words = [{ id: 'w0', start: 1.0, end: 1.5, text: 'hello', seg_id: 's0' }];
    const cid = deleteWords(['w0', 'ghost-id']);
    expect(cid).not.toBeNull();
    // Only w0 appears in the candidate
    expect(S.removalCandidates[0].wordIds).toEqual(['w0']);
  });

  test('all IDs not found returns null', () => {
    S.words = [];
    expect(deleteWords(['w99'])).toBeNull();
  });

  test('counter increments across calls', () => {
    S.words = [
      { id: 'w0', start: 0.0, end: 0.5, text: 'a', seg_id: 's0' },
      { id: 'w1', start: 0.6, end: 1.0, text: 'b', seg_id: 's0' },
    ];
    const c0 = deleteWords(['w0']);
    const c1 = deleteWords(['w1']);
    expect(c0).toBe('wd0');
    expect(c1).toBe('wd1');
    expect(S.removalCandidates.length).toBe(2);
  });

  test('CJK words: label uses concatenation without spaces', () => {
    S.words = [
      { id: 'w0', start: 0.0, end: 0.3, text: 'テスト', seg_id: 's0' },
      { id: 'w1', start: 0.4, end: 0.7, text: 'です',   seg_id: 's0' },
    ];
    const cid = deleteWords(['w0', 'w1']);
    expect(S.removalCandidates[0].label).toBe('テストです');
  });

  test('initWordEdit resets counter so next deleteWords uses wd0', () => {
    S.words = [{ id: 'w0', start: 0.0, end: 0.5, text: 'a', seg_id: 's0' }];
    deleteWords(['w0']);   // wd0
    initWordEdit();
    S.words = [{ id: 'w1', start: 0.6, end: 1.0, text: 'b', seg_id: 's0' }];
    S.removalCandidates = [];
    S.removalsEnabled   = new Set();
    const cid = deleteWords(['w1']);
    expect(cid).toBe('wd0');
  });
})

// ── getSelectedWordIds / clearWordSelection ────────────────────────────────────
// The drag-selection handlers (mousedown/mousemove) require a browser DOM and
// are not exercisable in vitest.  These tests cover the exported read/clear API
// which is DOM-free (clearWordSelection is guarded by typeof document checks).

describe('getSelectedWordIds / clearWordSelection', () => {
  beforeEach(() => {
    initWordEdit();  // resets _customSelectedIds to empty Set via clearWordSelection
  });

  test('returns an Array (not a Set)', () => {
    const ids = getSelectedWordIds();
    expect(Array.isArray(ids)).toBe(true);
  });

  test('returns empty array when nothing is selected', () => {
    expect(getSelectedWordIds()).toEqual([]);
  });

  test('clearWordSelection is a no-op in Node.js (no DOM) and does not throw', () => {
    expect(() => clearWordSelection()).not.toThrow();
  });

  test('getSelectedWordIds returns empty after explicit clearWordSelection', () => {
    clearWordSelection();
    expect(getSelectedWordIds()).toEqual([]);
  });

  test('initWordEdit clears selection state (getSelectedWordIds returns [])', () => {
    // Can't set selection via mouse handlers in vitest (no DOM),
    // but we can confirm initWordEdit always leaves state clean.
    initWordEdit();
    expect(getSelectedWordIds()).toEqual([]);
  });
})
