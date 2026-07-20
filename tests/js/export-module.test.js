import { describe, test, expect } from 'vitest'
import { buildSegsJS, rebuildTelopText } from '../../src/preprod/static/modules/export-module.js'

describe('buildSegsJS', () => {
  test('empty removals returns full span', () => {
    expect(buildSegsJS([], 100)).toEqual([[0, 100]])
  })

  test('single middle removal creates two keeps', () => {
    expect(buildSegsJS([[30, 60]], 100)).toEqual([[0, 30], [60, 100]])
  })

  test('removal at start leaves only tail', () => {
    expect(buildSegsJS([[0, 40]], 100)).toEqual([[40, 100]])
  })

  test('removal at end leaves only head', () => {
    expect(buildSegsJS([[80, 100]], 100)).toEqual([[0, 80]])
  })

  test('overlapping removals are merged', () => {
    expect(buildSegsJS([[10, 40], [30, 60]], 100)).toEqual([[0, 10], [60, 100]])
  })

  test('adjacent removals are merged', () => {
    expect(buildSegsJS([[10, 30], [30, 60]], 100)).toEqual([[0, 10], [60, 100]])
  })

  test('non-overlapping removals preserve gap between them', () => {
    expect(buildSegsJS([[10, 20], [50, 60]], 100)).toEqual([[0, 10], [20, 50], [60, 100]])
  })

  test('out-of-order removals are sorted before merging', () => {
    expect(buildSegsJS([[50, 60], [10, 20]], 100)).toEqual([[0, 10], [20, 50], [60, 100]])
  })

  test('padding shrinks the removal region', () => {
    expect(buildSegsJS([[30, 60]], 100, 1000)).toEqual([[0, 31], [59, 100]])
  })

  test('padding so large it eliminates removal returns full span', () => {
    expect(buildSegsJS([[30, 31]], 100, 5000)).toEqual([[0, 100]])
  })

  test('removal spanning full duration returns empty array', () => {
    expect(buildSegsJS([[0, 100]], 100)).toEqual([])
  })

  test('keeps shorter than 0.05s are filtered out', () => {
    // Removal [0.03, 99.97] leaves keeps of 0.03s each — both below the 0.05s threshold
    expect(buildSegsJS([[0.03, 99.97]], 100)).toEqual([])
  })

  test('keep exactly at 0.05s boundary is preserved', () => {
    // Remove [0.05, 100] → leading keep [0, 0.05] is exactly 0.05s (at threshold → kept)
    expect(buildSegsJS([[0.05, 100]], 100)).toEqual([[0, 0.05]])
  })

  test('keep below 0.05s boundary is dropped', () => {
    // Remove [0.04, 100] → leading keep [0, 0.04] is 0.04s (below threshold → dropped)
    expect(buildSegsJS([[0.04, 100]], 100)).toEqual([])
  })

  // ── Word-type padding ──────────────────────────────────────────────────────
  // Word-type regions must use zero padding so short words are never collapsed.

  test('word-type region is not collapsed by padding', () => {
    // 200ms word with 200ms padding — would become 0ms if padded → must survive
    const result = buildSegsJS([[2.0, 2.2, 'word']], 10, 200)
    // Expect two keeps: [0, 2.0] and [2.2, 10]
    expect(result).toEqual([[0, 2.0], [2.2, 10]])
  })

  test('word-type region at exact padding width still survives', () => {
    // 100ms word with 100ms padding each side = collapse if padded — word avoids it
    const result = buildSegsJS([[1.0, 1.1, 'word']], 5, 100)
    expect(result).toEqual([[0, 1.0], [1.1, 5]])
  })

  test('silence-type region with padding still collapses when too short', () => {
    // 300ms silence with 200ms padding collapses → full span returned
    const result = buildSegsJS([[1.0, 1.3, 'silence']], 5, 200)
    expect(result).toEqual([[0, 5]])
  })

  test('mixed word and silence types — word no padding, silence padded', () => {
    // Word [1.0, 1.15]: no padding → kept as-is
    // Silence [5.0, 7.0]: 200ms padding → [5.2, 6.8]
    const result = buildSegsJS(
      [[1.0, 1.15, 'word'], [5.0, 7.0, 'silence']], 10, 200
    )
    // Keeps: [0, 1.0], [1.15, 5.2], [6.8, 10]
    expect(result[0]).toEqual([0, 1.0])
    expect(result[1][0]).toBeCloseTo(1.15, 5)
    expect(result[1][1]).toBeCloseTo(5.2,  5)
    expect(result[2][0]).toBeCloseTo(6.8,  5)
    expect(result[2][1]).toBeCloseTo(10,   5)
  })

  test('word type without padding parameter still works', () => {
    const result = buildSegsJS([[2.0, 3.0, 'word']], 10)
    expect(result).toEqual([[0, 2.0], [3.0, 10]])
  })
})

describe('rebuildTelopText', () => {
  const mkEntry = (id, text) => ({ id, start: 0, end: 5, text })
  const mkWord  = (id, segId, text) => ({ id, seg_id: segId, text })

  test('empty deleted set — returns same entry reference', () => {
    const entry = mkEntry('e1', 'hello world')
    const words = [mkWord('w1', 'e1', 'hello'), mkWord('w2', 'e1', 'world')]
    expect(rebuildTelopText(entry, words, new Set())).toBe(entry)
  })

  test('no words in entry — returns same entry reference unchanged', () => {
    const entry = mkEntry('e1', 'hello')
    expect(rebuildTelopText(entry, [], new Set(['w1']))).toBe(entry)
  })

  test('deleted word not in this entry — returns same entry reference', () => {
    const entry = mkEntry('e1', 'hello world')
    const words = [mkWord('w1', 'e1', 'hello'), mkWord('w2', 'e1', 'world')]
    expect(rebuildTelopText(entry, words, new Set(['w99']))).toBe(entry)
  })

  test('deletes word from start of English entry', () => {
    const entry = mkEntry('e1', 'um hello world')
    const words = [mkWord('w0', 'e1', 'um'), mkWord('w1', 'e1', 'hello'), mkWord('w2', 'e1', 'world')]
    const result = rebuildTelopText(entry, words, new Set(['w0']))
    expect(result.text).toBe('hello world')
  })

  test('deletes word from middle of English entry', () => {
    const entry = mkEntry('e1', 'hello um world')
    const words = [mkWord('w1', 'e1', 'hello'), mkWord('w2', 'e1', 'um'), mkWord('w3', 'e1', 'world')]
    const result = rebuildTelopText(entry, words, new Set(['w2']))
    expect(result.text).toBe('hello world')
  })

  test('deletes filler from Japanese entry — no spaces', () => {
    const entry = mkEntry('e1', 'えーとこれは大事です')
    const words = [
      mkWord('w1', 'e1', 'えーと'),
      mkWord('w2', 'e1', 'これは'),
      mkWord('w3', 'e1', '大事'),
      mkWord('w4', 'e1', 'です'),
    ]
    const result = rebuildTelopText(entry, words, new Set(['w1']))
    expect(result.text).toBe('これは大事です')
  })

  test('mixed JA/EN — deletes English word before CJK', () => {
    const entry = mkEntry('e1', 'Claude Codeを使う')
    const words = [
      mkWord('w1', 'e1', 'Claude'),
      mkWord('w2', 'e1', 'Code'),
      mkWord('w3', 'e1', 'を使う'),
    ]
    const result = rebuildTelopText(entry, words, new Set(['w1']))
    expect(result.text).toBe('Codeを使う')
  })

  test('mixed JA/EN — space preserved between adjacent Latin tokens', () => {
    const entry = mkEntry('e1', 'えーと hello world')
    const words = [
      mkWord('w1', 'e1', 'えーと'),
      mkWord('w2', 'e1', 'hello'),
      mkWord('w3', 'e1', 'world'),
    ]
    const result = rebuildTelopText(entry, words, new Set(['w1']))
    expect(result.text).toBe('hello world')
  })

  test('all words in entry deleted — empty text', () => {
    const entry = mkEntry('e1', 'えー')
    const words = [mkWord('w1', 'e1', 'えー')]
    const result = rebuildTelopText(entry, words, new Set(['w1']))
    expect(result.text).toBe('')
  })

  test('only entry1 words deleted — entry2 words from same list unaffected', () => {
    const entry2 = mkEntry('e2', 'foo bar')
    const words = [
      mkWord('w1', 'e1', 'hello'),
      mkWord('w2', 'e1', 'world'),
      mkWord('w3', 'e2', 'foo'),
      mkWord('w4', 'e2', 'bar'),
    ]
    // Delete w3 belonging to e2 — querying for e1 should be unchanged
    const entry1 = mkEntry('e1', 'hello world')
    const result = rebuildTelopText(entry1, words, new Set(['w3']))
    expect(result).toBe(entry1)
  })

  test('preserves start/end from original entry', () => {
    const entry = { id: 'e1', start: 1.5, end: 4.0, text: 'um yes' }
    const words = [mkWord('w1', 'e1', 'um'), mkWord('w2', 'e1', 'yes')]
    const result = rebuildTelopText(entry, words, new Set(['w1']))
    expect(result.start).toBe(1.5)
    expect(result.end).toBe(4.0)
    expect(result.text).toBe('yes')
  })
})
