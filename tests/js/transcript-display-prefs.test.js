// QA coverage note: renderFullTranscript/_renderWordsIntoSeg are DOM-dependent
// (document.createElement, DocumentFragment) and this project's vitest config
// runs in the `node` environment (no jsdom) — see dom_env_check.test.js for the
// same limitation on telop-overlay.js. The show/hide-deleted-content and
// strikethrough rendering logic was verified manually in a real browser
// (injected S.telopEntries/removalCandidates, confirmed DOM output and computed
// styles) rather than unit-tested here. This file covers the one piece that
// IS DOM-independent: loadTranscriptDisplayPrefs() must never throw even when
// localStorage is unavailable or broken — the exact regression class that
// briefly broke 6/7 test files when state.js read localStorage at module load.

import { describe, test, expect, beforeEach } from 'vitest'
import { S } from '../../src/preprod/static/modules/state.js'
import { loadTranscriptDisplayPrefs } from '../../src/preprod/static/modules/transcript.js'

describe('loadTranscriptDisplayPrefs', () => {
  beforeEach(() => {
    S.showDeletedSilence = false;
    S.showDeletedFillers = false;
  });

  test('does not throw when localStorage is unavailable (node environment)', () => {
    expect(() => loadTranscriptDisplayPrefs()).not.toThrow();
  });

  test('leaves S at its false defaults when localStorage is unavailable', () => {
    loadTranscriptDisplayPrefs();
    expect(S.showDeletedSilence).toBe(false);
    expect(S.showDeletedFillers).toBe(false);
  });

  test('reads persisted "1" values when a working localStorage is present', () => {
    const store = { vt_show_deleted_silence: '1', vt_show_deleted_fillers: '1' };
    globalThis.localStorage = { getItem: k => store[k] ?? null };
    try {
      loadTranscriptDisplayPrefs();
      expect(S.showDeletedSilence).toBe(true);
      expect(S.showDeletedFillers).toBe(true);
    } finally {
      delete globalThis.localStorage;
    }
  });

  test('treats any non-"1" value (including "0") as false', () => {
    const store = { vt_show_deleted_silence: '0' };
    globalThis.localStorage = { getItem: k => store[k] ?? null };
    try {
      loadTranscriptDisplayPrefs();
      expect(S.showDeletedSilence).toBe(false);
    } finally {
      delete globalThis.localStorage;
    }
  });

  test('does not throw when localStorage.getItem itself throws (e.g. private-browsing quota error)', () => {
    globalThis.localStorage = { getItem: () => { throw new Error('quota'); } };
    try {
      expect(() => loadTranscriptDisplayPrefs()).not.toThrow();
    } finally {
      delete globalThis.localStorage;
    }
  });
});
