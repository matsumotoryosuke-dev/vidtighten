import { describe, test, expect, beforeEach } from 'vitest'
import { S, collectDeletedWordIds } from '../../src/preprod/static/modules/state.js'

describe('collectDeletedWordIds', () => {
  beforeEach(() => {
    S.removalCandidates = [];
    S.removalsEnabled   = new Set();
  });

  test('empty state returns empty set', () => {
    expect(collectDeletedWordIds().size).toBe(0);
  });

  test('word-type enabled candidate — wordIds collected', () => {
    S.removalCandidates = [{ id: 'c1', type: 'word', wordIds: ['w1', 'w2'] }];
    S.removalsEnabled   = new Set(['c1']);
    expect([...collectDeletedWordIds()].sort()).toEqual(['w1', 'w2']);
  });

  test('silence-type candidate is ignored even when enabled', () => {
    S.removalCandidates = [
      { id: 'c1', type: 'word',    wordIds: ['w1'] },
      { id: 'c2', type: 'silence', wordIds: ['w2'] },
    ];
    S.removalsEnabled = new Set(['c1', 'c2']);
    // c2 is type='silence' → excluded; only w1 from c1
    expect([...collectDeletedWordIds()]).toEqual(['w1']);
  });

  test('word-type candidate that is NOT enabled is excluded', () => {
    S.removalCandidates = [{ id: 'c1', type: 'word', wordIds: ['w1'] }];
    S.removalsEnabled   = new Set();  // c1 not enabled
    expect(collectDeletedWordIds().size).toBe(0);
  });

  test('candidate with no wordIds property does not throw', () => {
    // wordIds is optional — fall back to [] via `c.wordIds || []`
    S.removalCandidates = [{ id: 'c1', type: 'word' }];
    S.removalsEnabled   = new Set(['c1']);
    expect(collectDeletedWordIds().size).toBe(0);
  });

  test('multiple enabled word candidates — all wordIds merged', () => {
    S.removalCandidates = [
      { id: 'c1', type: 'word', wordIds: ['w1', 'w2'] },
      { id: 'c2', type: 'word', wordIds: ['w3'] },
    ];
    S.removalsEnabled = new Set(['c1', 'c2']);
    expect([...collectDeletedWordIds()].sort()).toEqual(['w1', 'w2', 'w3']);
  });

  test('duplicate wordIds across candidates are deduplicated', () => {
    S.removalCandidates = [
      { id: 'c1', type: 'word', wordIds: ['w1', 'w2'] },
      { id: 'c2', type: 'word', wordIds: ['w2', 'w3'] },
    ];
    S.removalsEnabled = new Set(['c1', 'c2']);
    expect([...collectDeletedWordIds()].sort()).toEqual(['w1', 'w2', 'w3']);
  });

  test('returns a fresh Set each call (not a cached reference)', () => {
    S.removalCandidates = [{ id: 'c1', type: 'word', wordIds: ['w1'] }];
    S.removalsEnabled   = new Set(['c1']);
    const a = collectDeletedWordIds();
    const b = collectDeletedWordIds();
    expect(a).not.toBe(b);   // distinct Set instances
    expect([...a]).toEqual([...b]);
  });
});
