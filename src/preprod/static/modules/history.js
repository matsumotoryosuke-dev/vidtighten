import { S } from './state.js';
import { $ } from './utils.js';

let _onHistory = () => {};

export function initHistory({ onHistory }) {
  _onHistory = onHistory;
}

export function snap() {
  return {
    removalsEnabled: [...S.removalsEnabled],
    telopsEnabled:   [...S.telopsEnabled],
    candidates: S.removalCandidates.map(c => ({
      ...c,
      // Deep-clone wordIds[] so undo/redo snapshots are independent.
      // Conditional spread omits the key on non-word candidates entirely
      // (avoids `wordIds: undefined` polluting every filler/silence object).
      ...(c.wordIds && { wordIds: [...c.wordIds] }),
    })),
  };
}

export function pushHistory() {
  S.history = S.history.slice(0, S.historyIndex+1);
  S.history.push(snap());
  S.historyIndex = S.history.length - 1;
  updateUndoRedo();
}

export function applySnap(s) {
  S.removalCandidates = s.candidates.map(c => ({...c}));
  S.removalsEnabled   = new Set(s.removalsEnabled);
  S.telopsEnabled     = new Set(s.telopsEnabled);
  _onHistory();
}

export function undo() {
  if (S.historyIndex > 0) { S.historyIndex--; applySnap(S.history[S.historyIndex]); }
}

export function redo() {
  if (S.historyIndex < S.history.length-1) { S.historyIndex++; applySnap(S.history[S.historyIndex]); }
}

export function updateUndoRedo() {
  $('btn-undo').disabled = S.historyIndex <= 0;
  $('btn-redo').disabled = S.historyIndex >= S.history.length-1;
}
