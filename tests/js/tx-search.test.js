// QA coverage note: initTxSearch() itself is DOM-dependent (document.querySelector,
// event listeners) and this project's vitest config runs in the `node` environment
// (no jsdom) — see dom_env_check.test.js for the same limitation elsewhere. The pure
// text/word-redistribution logic (_containsQuery, _replaceOccurrences,
// _joinWithOffsets, _applyWordReplace, _applyReplace) is DOM-independent and fully
// covered here; the UI wiring (search bar open/close, highlight classes, keyboard
// shortcuts) was verified manually in a real browser against Rio's real saved
// session data.

import { describe, test, expect } from 'vitest'
import {
  _containsQuery, _replaceOccurrences, _joinWithOffsets,
  _applyWordReplace, _applyReplace,
} from '../../src/preprod/static/modules/tx-search.js'

describe('_containsQuery', () => {
  test('case-insensitive substring match', () => {
    expect(_containsQuery('Hello World', 'world')).toBe(true)
  })
  test('no match returns false', () => {
    expect(_containsQuery('Hello World', 'xyz')).toBe(false)
  })
  test('empty text returns false', () => {
    expect(_containsQuery('', 'a')).toBe(false)
  })
  test('null/undefined text returns false, not throw', () => {
    expect(_containsQuery(null, 'a')).toBe(false)
    expect(_containsQuery(undefined, 'a')).toBe(false)
  })
  test('matches Japanese substrings', () => {
    expect(_containsQuery('空気デザインです', '空気デザイン')).toBe(true)
  })
})

describe('_replaceOccurrences', () => {
  test('replaces a single occurrence and counts it', () => {
    const { text, count } = _replaceOccurrences('foo bar', 'bar', 'baz')
    expect(text).toBe('foo baz')
    expect(count).toBe(1)
  })
  test('replaces all occurrences (global)', () => {
    const { text, count } = _replaceOccurrences('a-a-a', 'a', 'b')
    expect(text).toBe('b-b-b')
    expect(count).toBe(3)
  })
  test('case-insensitive replace', () => {
    const { text, count } = _replaceOccurrences('AIエージェント', 'aiエージェント', 'X')
    expect(text).toBe('X')
    expect(count).toBe(1)
  })
  test('no match leaves text unchanged, count 0', () => {
    const { text, count } = _replaceOccurrences('hello', 'xyz', 'X')
    expect(text).toBe('hello')
    expect(count).toBe(0)
  })
  test('empty text returns count 0 without throwing', () => {
    expect(_replaceOccurrences('', 'a', 'b')).toEqual({ text: '', count: 0 })
  })
  test('query with regex special characters is treated literally', () => {
    const { text, count } = _replaceOccurrences('a.b.c', '.', '-')
    expect(text).toBe('a-b-c')
    expect(count).toBe(2)
  })
  test('brand-name correction example from the feature request', () => {
    const { text, count } = _replaceOccurrences(
      '実際空気デザインというのは', '空気デザイン', 'クウキデザイン'
    )
    expect(text).toBe('実際クウキデザインというのは')
    expect(count).toBe(1)
  })
})

describe('_joinWithOffsets', () => {
  test('no space between two CJK tokens', () => {
    const { text } = _joinWithOffsets([{ text: 'です' }, { text: 'ね' }])
    expect(text).toBe('ですね')
  })
  test('space between two non-CJK (Latin) tokens', () => {
    const { text } = _joinWithOffsets([{ text: 'Claude' }, { text: 'Code' }])
    expect(text).toBe('Claude Code')
  })
  test('no space at a Latin-CJK boundary', () => {
    const { text } = _joinWithOffsets([{ text: 'GitHub' }, { text: 'プロジェクト' }])
    expect(text).toBe('GitHubプロジェクト')
  })
  test('offsets point at each word\'s start position in the joined string', () => {
    const { text, offsets } = _joinWithOffsets([{ text: 'AI' }, { text: 'エージェント' }, { text: 'の' }])
    expect(text).toBe('AIエージェントの')
    expect(offsets).toEqual([0, 2, 8])
    expect(text.slice(offsets[1], offsets[1] + 'エージェント'.length)).toBe('エージェント')
  })
  test('empty-text words (already blanked by an earlier replace) contribute no text and no phantom space', () => {
    // Matches _renderWordsIntoSeg's guard exactly: a blanked word is skipped on
    // both sides, so neighboring non-CJK words end up with no space between them
    // (not a double space, and not a "correct" single space either) — an
    // accepted minor cosmetic quirk of merging a match into one word.
    const { text } = _joinWithOffsets([{ text: 'Claude' }, { text: '' }, { text: 'Code' }])
    expect(text).toBe('ClaudeCode')
  })
})

describe('_applyWordReplace', () => {
  test('replaces within a single word token, others untouched', () => {
    const words = [{ text: '空気デザイン' }, { text: 'です' }]
    _applyWordReplace(words, '空気デザイン', 'クウキデザイン')
    expect(words.map(w => w.text)).toEqual(['クウキデザイン', 'です'])
  })

  test('redistributes a match spanning two word tokens into the first, blanks the second', () => {
    const words = [{ text: 'AI' }, { text: 'エージェント' }, { text: 'の' }]
    _applyWordReplace(words, 'AIエージェント', 'ChatGPT')
    expect(words.map(w => w.text)).toEqual(['ChatGPT', '', 'の'])
  })

  test('preserves an unmatched prefix/suffix on the boundary words', () => {
    // "AIエージェントの" — match "エージェントの" starts mid-second-word... use a
    // case where the match starts inside word 0 and ends inside word 1.
    const words = [{ text: 'ABCエージェント' }, { text: 'です' }]
    _applyWordReplace(words, 'エージェントで', 'X')
    // "ABC" prefix kept from word 0, "す" suffix kept from word 1.
    expect(words.map(w => w.text)).toEqual(['ABCXす', ''])
  })

  test('no match leaves all words untouched', () => {
    const words = [{ text: 'AI' }, { text: 'エージェント' }]
    _applyWordReplace(words, 'nonexistent', 'X')
    expect(words.map(w => w.text)).toEqual(['AI', 'エージェント'])
  })

  test('empty word array is a no-op, does not throw', () => {
    expect(() => _applyWordReplace([], 'a', 'b')).not.toThrow()
  })

  test('multiple occurrences within the same entry all get replaced (right-to-left safety)', () => {
    const words = [{ text: 'a' }, { text: 'a' }, { text: 'b' }, { text: 'a' }]
    _applyWordReplace(words, 'a', 'X')
    expect(words.map(w => w.text)).toEqual(['X', 'X', 'b', 'X'])
  })

  test('word timestamps/ids are never touched, only .text', () => {
    const words = [
      { id: 'w0', start: 1.0, end: 1.2, text: 'AI' },
      { id: 'w1', start: 1.2, end: 1.8, text: 'エージェント' },
    ]
    _applyWordReplace(words, 'AIエージェント', 'ChatGPT')
    expect(words[0]).toMatchObject({ id: 'w0', start: 1.0, end: 1.2 })
    expect(words[1]).toMatchObject({ id: 'w1', start: 1.2, end: 1.8 })
  })
})

describe('_applyReplace', () => {
  test('updates entry.text and returns the occurrence count', () => {
    const entry = { id: 't0', text: 'ほとんどの人はAIエージェントの使い方が' }
    const allWords = [
      { seg_id: 't0', text: 'ほとんどの人は' },
      { seg_id: 't0', text: 'AI' },
      { seg_id: 't0', text: 'エージェント' },
      { seg_id: 't0', text: 'の使い方が' },
    ]
    const count = _applyReplace(entry, allWords, 'AIエージェント', 'ChatGPT')
    expect(count).toBe(1)
    expect(entry.text).toBe('ほとんどの人はChatGPTの使い方が')
    expect(allWords.map(w => w.text)).toEqual(['ほとんどの人は', 'ChatGPT', '', 'の使い方が'])
  })

  test('zero matches: returns 0, mutates nothing', () => {
    const entry = { id: 't0', text: 'hello world' }
    const allWords = [{ seg_id: 't0', text: 'hello' }, { seg_id: 't0', text: 'world' }]
    const count = _applyReplace(entry, allWords, 'xyz', 'X')
    expect(count).toBe(0)
    expect(entry.text).toBe('hello world')
    expect(allWords.map(w => w.text)).toEqual(['hello', 'world'])
  })

  test('only touches words belonging to this entry (filters by seg_id)', () => {
    const entry = { id: 't0', text: 'AIエージェント' }
    const allWords = [
      { seg_id: 't0', text: 'AI' },
      { seg_id: 't0', text: 'エージェント' },
      { seg_id: 't1', text: 'AI' },   // different segment — must stay untouched
      { seg_id: 't1', text: 'エージェント' },
    ]
    _applyReplace(entry, allWords, 'AIエージェント', 'X')
    expect(allWords[2].text).toBe('AI')
    expect(allWords[3].text).toBe('エージェント')
  })

  test('word-less entry (no matching seg_id in allWords) still updates entry.text', () => {
    const entry = { id: 't0', text: '空気デザインです' }
    const count = _applyReplace(entry, [], '空気デザイン', 'クウキデザイン')
    expect(count).toBe(1)
    expect(entry.text).toBe('クウキデザインです')
  })
})
