// Shared application state — import { S } from './state.js'
export const S = {
  filePath:    null,
  media:       null,
  taskId:      null,
  polling:     null,

  waveformData:       [],
  removalCandidates:  [],
  telopEntries:       [],
  words:              [],

  removalsEnabled:    new Set(),
  telopsEnabled:      new Set(),

  history:      [],
  historyIndex: -1,

  whisperAvailable: false,

  zoom: { level: 1, offset: 0 },
  waveformThreshold: 0,
  waveformMaxAmp:    0,

  silPreview: null,   // null = use committed; array = live preview override

  // Transcript display toggles — deleted silence/filler words are hidden by
  // default for a cleaner read; persisted across sessions as a UI preference
  // (independent of the underlying edit — toggling this never changes what
  // actually gets cut, only what the transcript panel displays). Actual
  // persisted value is loaded at startup — see loadTranscriptDisplayPrefs()
  // in transcript.js — to keep this module free of browser-API side effects
  // at import time (state.js is imported by the Node-environment test suite).
  showDeletedSilence: false,
  showDeletedFillers: false,
};

/**
 * Return the Set of word IDs currently marked for deletion.
 * Reads from S.removalCandidates and S.removalsEnabled; must be called
 * after state is fully populated (i.e. after applyResult).
 */
export function collectDeletedWordIds() {
  const ids = new Set();
  S.removalCandidates
    .filter(c => c.type === 'word' && S.removalsEnabled.has(c.id))
    .forEach(c => (c.wordIds || []).forEach(id => ids.add(id)));
  return ids;
}
