import { describe, test, expect } from 'vitest'
import {
  _cleanTelopText,
  _charEm,
  _emWidth,
  _breakScore,
  _wrapTelop,
} from '../../src/preprod/static/modules/telop-overlay.js'

describe('_cleanTelopText', () => {
  test('replaces Japanese sentence-ending punctuation with space', () => {
    expect(_cleanTelopText('それはね。すごいよ')).toBe('それはね すごいよ')
  })
  test('removes Japanese bracket pairs', () => {
    expect(_cleanTelopText('「こんにちは」')).toBe('こんにちは')
  })
  test('replaces ASCII comma', () => {
    expect(_cleanTelopText('hello,world')).toBe('hello world')
  })
  test('trims trailing separator', () => {
    expect(_cleanTelopText('end.')).toBe('end')
  })
  test('replaces exclamation point', () => {
    expect(_cleanTelopText('wow!great')).toBe('wow great')
  })
  test('replaces newline with space', () => {
    expect(_cleanTelopText('line1\nline2')).toBe('line1 line2')
  })
  test('collapses consecutive separators to a single space', () => {
    expect(_cleanTelopText('a。。b')).toBe('a b')
  })
  test('all separators returns empty string', () => {
    expect(_cleanTelopText('。、！？')).toBe('')
  })
  test('mixed ASCII and Japanese passes through content words', () => {
    const result = _cleanTelopText('Hello、世界！Great.')
    expect(result).toContain('Hello')
    expect(result).toContain('世界')
    expect(result).toContain('Great')
  })
  test('pure text unchanged', () => {
    expect(_cleanTelopText('ありがとう')).toBe('ありがとう')
  })
  test('trims leading and trailing whitespace', () => {
    expect(_cleanTelopText('  hello  ')).toBe('hello')
  })
  test('removes ASCII double quotes', () => {
    expect(_cleanTelopText('"quoted"')).toBe('quoted')
  })
  test('removes corner bracket annotation', () => {
    expect(_cleanTelopText('【重要】連絡です')).toBe('重要連絡です')
  })
  test('replaces ASCII hyphen with space (parity with Python)', () => {
    expect(_cleanTelopText('a-b')).toBe('a b')
  })
  test('replaces em dash with space', () => {
    expect(_cleanTelopText('a—b')).toBe('a b')
  })
  test('replaces en dash with space', () => {
    expect(_cleanTelopText('a–b')).toBe('a b')
  })
  test('empty string returns empty string', () => {
    expect(_cleanTelopText('')).toBe('')
  })
  test('leading separator is trimmed away (Whisper comma-first pattern)', () => {
    // Whisper sometimes emits '、テキスト'; separator → space → trim → 'テキスト'
    expect(_cleanTelopText('、テキスト')).toBe('テキスト')
  })
  test('whitespace-only string returns empty string', () => {
    expect(_cleanTelopText('   ')).toBe('')
  })
  test('replaces colon with space', () => {
    expect(_cleanTelopText('key:value')).toBe('key value')
  })
  // Separator parity — chars in _SEP_RE not yet individually tested
  test('replaces middle dot ・ with space', () => {
    expect(_cleanTelopText('a・b')).toBe('a b')   // U+30FB
  })
  test('replaces ellipsis … with space', () => {
    expect(_cleanTelopText('a…b')).toBe('a b')   // U+2026
  })
  test('replaces wave dash 〜 with space', () => {
    expect(_cleanTelopText('a〜b')).toBe('a b')   // U+301C
  })
  test('replaces fullwidth tilde ～ with space', () => {
    expect(_cleanTelopText('a～b')).toBe('a b')   // U+FF5E
  })
  test('replaces ASCII question mark with space', () => {
    expect(_cleanTelopText('a?b')).toBe('a b')
  })
  test('replaces ASCII semicolon with space', () => {
    expect(_cleanTelopText('a;b')).toBe('a b')
  })
  test('replaces fullwidth semicolon ；with space', () => {
    expect(_cleanTelopText('a；b')).toBe('a b')   // U+FF1B
  })
  test('replaces fullwidth exclamation ！ with space', () => {
    expect(_cleanTelopText('a！b')).toBe('a b')   // U+FF01 — common in JP Whisper output
  })
  test('replaces fullwidth question mark ？ with space', () => {
    expect(_cleanTelopText('a？b')).toBe('a b')   // U+FF1F
  })
  test('replaces fullwidth colon ： with space', () => {
    expect(_cleanTelopText('a：b')).toBe('a b')   // U+FF1A — e.g. '概要：'
  })
  // Wrapper parity — chars in _WRAP_RE not yet individually tested
  test('removes fullwidth parentheses （）', () => {
    expect(_cleanTelopText('（括弧）')).toBe('括弧')
  })
  test('removes Unicode curly double quotes “”', () => {
    expect(_cleanTelopText('”quoted”')).toBe('quoted')
  })
  test('removes white corner brackets 『』', () => {
    expect(_cleanTelopText('『引用』')).toBe('引用')
  })
  test('removes angle brackets 〈〉', () => {
    expect(_cleanTelopText('〈angle〉')).toBe('angle')    // U+3008/3009
  })
  test('removes double angle brackets 《》', () => {
    expect(_cleanTelopText('《double》')).toBe('double')  // U+300A/300B
  })
  test('removes ASCII parentheses ()', () => {
    expect(_cleanTelopText('(aside)')).toBe('aside')
  })
  test('removes ASCII single quotes', () => {
    expect(_cleanTelopText("'quoted'")).toBe('quoted')
  })
  test('removes Unicode curly single quotes', () => {
    expect(_cleanTelopText('\u2018quoted\u2019')).toBe('quoted')
  })
  test('only wrapper chars produce empty string', () => {
    // Removing all wrappers leaves nothing after trim
    expect(_cleanTelopText('\u300c\u300d')).toBe('')
    expect(_cleanTelopText('\uff08\uff09')).toBe('')
  })
  test('three logical lines joined by newlines become single-spaced', () => {
    // Each \\n \u2192 space; consecutive spaces collapse; trimmed
    expect(_cleanTelopText('line1\nline2\nline3')).toBe('line1 line2 line3')
  })
  test('emoji is not a separator or wrapper and passes through unchanged', () => {
    // \ud83d\ude0a is not in _SEP_RE or _WRAP_RE \u2014 survives cleaning
    expect(_cleanTelopText('\u3042\ud83d\ude0a\u3044')).toBe('\u3042\ud83d\ude0a\u3044')
  })

  // Decimal-point preservation (bug: "GLM5.2" was rendering as "GLM5 2") \u2014 a "."
  // between two digits is a decimal point, not sentence punctuation; only strip
  // it to a space when it isn't touching a digit on either side.
  test('decimal point in a model name is preserved', () => {
    expect(_cleanTelopText('GLM5.2\u3068\u304b\u30c7\u30a3\u30fc\u30d7\u30b7\u30fc\u30af')).toBe('GLM5.2\u3068\u304b\u30c7\u30a3\u30fc\u30d7\u30b7\u30fc\u30af')
  })
  test('decimal point in a standalone number is preserved', () => {
    expect(_cleanTelopText('pi is 3.14159')).toBe('pi is 3.14159')
  })
  test('multiple decimal points are all preserved', () => {
    expect(_cleanTelopText('V4.0\u3068V4.5')).toBe('V4.0\u3068V4.5')
  })
  test('leading decimal point (digit only after) is preserved', () => {
    expect(_cleanTelopText('the score is .5')).toBe('the score is .5')
  })
  test('period touching a digit at string end is preserved', () => {
    expect(_cleanTelopText('version 5.')).toBe('version 5.')
  })
  test('decimal point followed by more text is preserved', () => {
    expect(_cleanTelopText('version 5.0 released')).toBe('version 5.0 released')
  })
  test('English sentence period not touching a digit still becomes a space', () => {
    expect(_cleanTelopText('done. Next sentence')).toBe('done Next sentence')
  })
  test('decimal point survives alongside a real sentence boundary in the same string', () => {
    expect(_cleanTelopText('\u63a1\u7528\u3055\u308c\u305f\u306e\u306fGLM5.2\u3067\u3059\u3002')).toBe('\u63a1\u7528\u3055\u308c\u305f\u306e\u306fGLM5.2\u3067\u3059')
  })
})

describe('_charEm', () => {
  test('hiragana is full width (1.0)', () => {
    expect(_charEm('あ')).toBe(1.0)
  })
  test('kanji is full width (1.0)', () => {
    expect(_charEm('字')).toBe(1.0)
  })
  test('katakana is full width (1.0)', () => {
    expect(_charEm('ア')).toBe(1.0)
  })
  test('ASCII letter is narrow (0.55)', () => {
    expect(_charEm('a')).toBeCloseTo(0.55)
  })
  test('ASCII digit is narrow (0.55)', () => {
    expect(_charEm('1')).toBeCloseTo(0.55)
  })
  test('ASCII space is narrow (0.55)', () => {
    expect(_charEm(' ')).toBeCloseTo(0.55)
  })
  test('fullwidth ASCII letter is full width (1.0)', () => {
    // U+FF21 FULLWIDTH LATIN CAPITAL LETTER A — in the FF01–FF60 range → 1.0 em
    expect(_charEm('Ａ')).toBeCloseTo(1.0)
  })
  // Parity note: Python unicodedata.east_asian_width classifies some characters
  // as "Ambiguous" (A) → 1.0 em (e.g. ° U+00B0, α U+03B1, Greek letters).
  // The JS port omits the A category — those characters return 0.55 in JS.
  // © (U+00A9) is classified 'N' (Narrow) by Python → 0.55 in both environments;
  // it is NOT one of the parity-gap characters.
  test('copyright sign (Narrow in both Python and JS) returns 0.55', () => {
    expect(_charEm('©')).toBeCloseTo(0.55)
  })
  // Degree sign ° (U+00B0) is Ambiguous (A) in Python → 1.0 there, but 0.55 here.
  // This test documents the JS behaviour; the Python equivalent is in test_fcpxml_telop.py.
  test('degree sign (Ambiguous in Python, Narrow in JS port) returns 0.55', () => {
    expect(_charEm('°')).toBeCloseTo(0.55)
  })
  test('Hangul syllable is full width (1.0)', () => {
    // U+AC00 가 — in the AC00–D7FF Hangul Syllables range
    expect(_charEm('가')).toBe(1.0)
  })
  test('fullwidth currency sign ￠ is full width (1.0)', () => {
    // U+FFE0 — in the FFE0–FFE6 Fullwidth signs range
    expect(_charEm('￠')).toBe(1.0)
  })
  test('emoji (above BMP) is narrow (0.55 — not in any CJK range)', () => {
    // U+1F60A 😊 — above U+FFFF, not covered by any range → 0.55
    expect(_charEm('😊')).toBeCloseTo(0.55)
  })
  // Remaining ranges — one representative per untested block
  test('Hangul Jamo (U+1100) is full width (1.0)', () => {
    // ᄀ U+1100 — in the 1100–115F Hangul Jamo range
    expect(_charEm('ᄀ')).toBe(1.0)
  })
  test('CJK Radicals Supplement (U+2E80) is full width (1.0)', () => {
    // ⺀ U+2E80 — in the 2E80–303F CJK Radicals+Symbols range
    expect(_charEm('⺀')).toBe(1.0)
  })
  test('Hangul Jamo Extended-A (U+A960) is full width (1.0)', () => {
    // ꥠ U+A960 — in the A960–A97F range
    expect(_charEm('ꥠ')).toBe(1.0)
  })
  test('CJK Compatibility Ideograph (U+F900) is full width (1.0)', () => {
    // 豈 U+F900 — in the F900–FAFF CJK Compat Ideographs range
    expect(_charEm('豈')).toBe(1.0)
  })
  test('Vertical Presentation Form (U+FE10) is full width (1.0)', () => {
    // ︐ U+FE10 — start of FE10–FE1F Vertical Presentation Forms range
    expect(_charEm('︐')).toBe(1.0)
  })
  test('Combining Half Mark (U+FE20) is narrow (0.55) — Python parity', () => {
    // U+FE20 = COMBINING LIGATURE LEFT HALF — Python east_asian_width → N (Narrow) → 0.55.
    // Historically the JS range was FE10–FE6F which incorrectly included FE20–FE2F.
    // Splitting the range to FE10–FE1F | FE30–FE6F (skipping FE20–FE2F) fixes the gap.
    expect(_charEm('︠')).toBeCloseTo(0.55)
  })
  test('CJK Compat Form (U+FE30) is full width (1.0)', () => {
    // ︰ U+FE30 — start of FE30–FE6F CJK Compatibility Forms range
    expect(_charEm('︰')).toBe(1.0)
  })
  test('half-width katakana (U+FF71) is narrow (0.55) — Python parity', () => {
    // ｱ U+FF71 — HALFWIDTH KATAKANA LETTER A.
    // Python east_asian_width → 'H' (Half-width) → 0.55 em.
    // In JS the FF01–FF60 range covers fullwidth ASCII/symbols;
    // U+FF61–FF9F (half-width katakana) is deliberately excluded → 0.55. ✓
    expect(_charEm('ｱ')).toBeCloseTo(0.55)
  })
  test('half-width katakana at top of range (U+FF9F) is narrow (0.55)', () => {
    // ﾟ U+FF9F — HALFWIDTH KATAKANA VOICED ITERATION MARK (last char before
    // fullwidth currency block) — also EAW 'H' in Python → 0.55.
    expect(_charEm('ﾟ')).toBeCloseTo(0.55)
  })
})

describe('_emWidth', () => {
  test('pure CJK: 5 chars = 5.0 em', () => {
    expect(_emWidth('あいうえお')).toBeCloseTo(5.0)
  })
  test('pure ASCII: 5 chars = 5 × 0.55 em', () => {
    expect(_emWidth('hello')).toBeCloseTo(5 * 0.55)
  })
  test('newline is not counted toward width', () => {
    expect(_emWidth('あ\nい')).toBeCloseTo(2.0)
  })
  test('empty string returns 0.0', () => {
    expect(_emWidth('')).toBeCloseTo(0.0)
  })
  test('mixed CJK + ASCII: correct weighted sum', () => {
    // 3 CJK (3.0) + 3 ASCII (3 × 0.55 = 1.65) = 4.65
    expect(_emWidth('あaいbうc')).toBeCloseTo(3 * 1.0 + 3 * 0.55)
  })
  test('emoji (surrogate pair above U+FFFF) counts as 0.55 em', () => {
    // 😊 U+1F60A is spread to a single code point by [...text] — not in any CJK
    // range → _charEm returns 0.55
    expect(_emWidth('😊')).toBeCloseTo(0.55)
  })
  test('CJK + emoji + CJK gives correct total (surrogate pairs handled)', () => {
    // あ(1.0) + 😊(0.55) + い(1.0) = 2.55 em
    expect(_emWidth('あ😊い')).toBeCloseTo(2.55)
  })
})

describe('_breakScore', () => {
  test('space before position is tier 3 (3.0)', () => {
    expect(_breakScore('a b', 2)).toBeCloseTo(3.0)
  })
  test('は before position is tier 2 (2.0)', () => {
    expect(_breakScore('あはい', 2)).toBeCloseTo(2.0)
  })
  test('が before position is tier 2 (2.0)', () => {
    expect(_breakScore('あがい', 2)).toBeCloseTo(2.0)
  })
  test('で before position is tier 1 (1.0)', () => {
    expect(_breakScore('あでい', 2)).toBeCloseTo(1.0)
  })
  test('て before position is 0.7', () => {
    expect(_breakScore('あてい', 2)).toBeCloseTo(0.7)
  })
  test('plain CJK before position is 0.0', () => {
    expect(_breakScore('あいう', 2)).toBeCloseTo(0.0)
  })
  test('を before position is tier 2 (2.0)', () => {
    expect(_breakScore('あをい', 2)).toBeCloseTo(2.0)
  })
  test('に before position is tier 2 (2.0)', () => {
    expect(_breakScore('あにい', 2)).toBeCloseTo(2.0)
  })
  test('と before position is tier 2 (2.0)', () => {
    expect(_breakScore('あとい', 2)).toBeCloseTo(2.0)
  })
  test('も before position is tier 1 (1.0)', () => {
    expect(_breakScore('あもい', 2)).toBeCloseTo(1.0)
  })
  test('の before position is tier 1 (1.0)', () => {
    expect(_breakScore('あのい', 2)).toBeCloseTo(1.0)
  })
  test('position 0 is always 0.0', () => {
    expect(_breakScore('abc', 0)).toBeCloseTo(0.0)
  })
  test('position at end of string is always 0.0', () => {
    expect(_breakScore('abc', 3)).toBeCloseTo(0.0)
  })
  test('か before position is tier 1 (1.0)', () => {
    expect(_breakScore('あかい', 2)).toBeCloseTo(1.0)
  })
  test('へ before position is tier 1 (1.0)', () => {
    expect(_breakScore('あへい', 2)).toBeCloseTo(1.0)
  })

  // Numeric tokens must never be split (bug: "GPT 5.5" wrapping as "5" / "5")
  test('break right before a decimal point is penalised', () => {
    // "5.5" — pos 1 is "5"|".5", splits the number.
    expect(_breakScore('5.5', 1)).toBeLessThan(0)
  })
  test('break right after a decimal point is penalised', () => {
    // "5.5" — pos 2 is "5."|"5", also splits the number.
    expect(_breakScore('5.5', 2)).toBeLessThan(0)
  })
  test('break between two plain digits is penalised', () => {
    expect(_breakScore('123', 1)).toBeLessThan(0)
    expect(_breakScore('123', 2)).toBeLessThan(0)
  })
  test('break between a letter and a digit is NOT penalised (boundary, not inside a number)', () => {
    // "GPT5" — pos 3 is "T"|"5": not inside a number, keeps its normal (zero) score.
    expect(_breakScore('GPT5', 3)).toBeCloseTo(0.0)
  })
})

describe('_wrapTelop', () => {
  test('short text returned unchanged', () => {
    expect(_wrapTelop('あいう', 10.0)).toBe('あいう')
  })
  test('long text gets exactly one newline', () => {
    const text = 'あ'.repeat(30)
    expect(_wrapTelop(text, 15.0).split('\n').length).toBe(2)
  })
  test('both wrapped lines fit within maxEm', () => {
    const text = 'あ'.repeat(30)
    for (const line of _wrapTelop(text, 16.0).split('\n')) {
      expect(_emWidth(line)).toBeLessThanOrEqual(16.0 + 0.01)
    }
  })
  test('pure CJK wraps near midpoint (balanced lines)', () => {
    const text = 'あ'.repeat(20)
    const [l1, l2] = _wrapTelop(text, 12.0).split('\n')
    expect(Math.abs(_emWidth(l1) - _emWidth(l2))).toBeLessThanOrEqual(2.0)
  })
  test('particle は dominates over raw midpoint', () => {
    const result = _wrapTelop('xxxxはyyyy', 3.5)
    expect(result.split('\n')[0].endsWith('は')).toBe(true)
  })
  test('space dominates over raw midpoint', () => {
    const [l1, l2] = _wrapTelop('aaaa bbb', 4.0).split('\n')
    expect(l1.trimEnd()).toBe('aaaa')
    expect(l2.trimStart()).toBe('bbb')
  })
  test('realistic JP sentence breaks at a particle', () => {
    const text = 'しばらくは完全に本気を出せない'
    const [l1] = _wrapTelop(text, 9.0).split('\n')
    expect('はがをにでとももかのへやよねわて'.includes(l1.at(-1))).toBe(true)
  })
  test('line 2 is hard-truncated when still too long', () => {
    const text = 'あ'.repeat(40)
    expect(_emWidth(_wrapTelop(text, 10.0).split('\n')[1])).toBeLessThanOrEqual(10.0 + 0.01)
  })
  test('never produces more than one newline', () => {
    expect(_wrapTelop('あ'.repeat(100), 10.0).split('\n').length).toBeLessThanOrEqual(2)
  })
  test('no trailing space on line 1 or leading space on line 2 at space break', () => {
    const [l1, l2] = _wrapTelop('hello world', 4.0).split('\n')
    expect(l1.endsWith(' ')).toBe(false)
    expect(l2.startsWith(' ')).toBe(false)
  })
  test('empty string returned as-is (no newline inserted)', () => {
    expect(_wrapTelop('', 10.0)).toBe('')
  })
  test('single character at-boundary is returned unchanged', () => {
    expect(_wrapTelop('あ', 1.0)).toBe('あ')
  })
  test('text exactly at maxEm is returned unchanged', () => {
    // 3 CJK chars = 3.0 em, maxEm = 3.0 → total_em <= maxEm → no wrap
    expect(_wrapTelop('あいう', 3.0)).toBe('あいう')
  })
  test('cleaned comma-space acts as line break (realistic Whisper sentence)', () => {
    // Whisper: 'チャンネル登録、よろしくお願いします' → clean → 'チャンネル登録 よろしくお願いします'
    // The space (former comma) dominates at score 3.0; both lines must fit in 13.44 em.
    const cleaned = _cleanTelopText('チャンネル登録、よろしくお願いします')
    const maxEm = 2560 * 0.42 / 80  // 1440p, font_size=80 → 13.44
    const [l1, l2] = _wrapTelop(cleaned, maxEm).split('\n')
    expect(l1).toBe('チャンネル登録')
    expect(l2).toBe('よろしくお願いします')
  })
  test('string with emoji (surrogate pair) does not crash', () => {
    // Emoji are above U+FFFF — spread via [...text] must handle them as single code points.
    // 😊 is U+1F60A → not in any CJK range → counted as 0.55 em.
    // The key assertion is that no exception is thrown and the result is a string.
    const result = _wrapTelop('こんにちは😊世界', 4.0)
    expect(typeof result).toBe('string')
    expect(result.split('\n').length).toBeLessThanOrEqual(2)
  })
  test('emoji remains intact (not split across surrogate halves) after wrapping', () => {
    // Verifies that [...text] spread treats 😊 as one code point, not two surrogate halves.
    // If the emoji were split, it would appear as two replacement characters in the output.
    const result = _wrapTelop('こんにちは😊世界', 4.0)
    // The emoji must appear intact in the joined output (no surrogate splitting).
    expect(result.replace('\n', '')).toContain('😊')
    // And each line must be a valid string (JSON serialisation would throw on lone surrogates).
    for (const line of result.split('\n')) {
      expect(() => JSON.stringify(line)).not.toThrow()
    }
  })
  test('two-character CJK text splits to one char per line', () => {
    // 'あい' = 2.0 em, maxEm = 1.5 → must wrap; bestPos = max(1, floor(2/2)) = 1
    expect(_wrapTelop('あい', 1.5)).toBe('あ\nい')
  })
  test('both wrapped lines fit within maxEm for a realistic JP sentence', () => {
    // 'しばらくは完全に本気を出せない' (15 em) with maxEm=9.0 → splits at に (pos 8)
    // line1='しばらくは完全に' (8.0 em), line2='本気を出せない' (7.0 em) — both ≤ 9.0
    const text = 'しばらくは完全に本気を出せない'
    const maxEm = 9.0
    for (const line of _wrapTelop(text, maxEm).split('\n')) {
      expect(_emWidth(line)).toBeLessThanOrEqual(maxEm + 0.01)
    }
  })
  test('line 2 exactly at maxEm boundary is not truncated', () => {
    // 'あああああ' = 5 CJK chars (5.0 em total) > maxEm=3.0 → wrapping triggered.
    // Best split at pos=2: line1='ああ' (2.0 em), line2='あああ' (3.0 em == maxEm).
    // _emWidth(line2) = 3.0 ≤ 3.0 → hard-truncation guard `> maxEm` is false
    // → line2 keeps all 3 chars.
    const result = _wrapTelop('あああああ', 3.0)
    expect(result.split('\n').length).toBe(2)
    expect(result.split('\n')[1]).toBe('あああ')
  })
  test('half-width katakana (narrow) does not cause premature wrap', () => {
    // ｱｲｳｴｵ = 5 half-width katakana chars × 0.55 em = 2.75 em ≤ maxEm=3.0 → no wrap
    expect(_wrapTelop('ｱｲｳｴｵ', 3.0)).toBe('ｱｲｳｴｵ')
  })

  // Numeric tokens must never be split across lines (bug: "GPT5.5" -> "GPT5.5"
  // rendered fine, but wrapped as "GPT5" / "5" when the midpoint landed inside it)
  test('decimal point never split across lines in a realistic sentence', () => {
    const text = 'このモデルは本当にすごくてGPT5.5を使うと生産性が劇的に向上します'
    const maxEm = Math.max(8.0, 3840 * 0.42 / 92)
    const lines = _wrapTelop(text, maxEm).split('\n')
    expect(lines.some(l => l.includes('5.5'))).toBe(true)
  })
  test('bare digit run never split when it sits exactly at the forced midpoint', () => {
    // Adversarial: no particles/spaces nearby to steer the break — without the
    // guard, nearest-to-midpoint alone would land inside the digit run.
    const text = 'あいうえおかきくけこ5588さしすせそたちつてと'
    const lines = _wrapTelop(text, 10.0).split('\n')
    expect(lines.some(l => l.includes('5588'))).toBe(true)
  })
  test('multiple decimals in one string all stay whole', () => {
    const text = '最近はGPT5.5とかGemini3.5とか色々出てきて選ぶのが大変ですよね'
    const maxEm = Math.max(8.0, 3840 * 0.42 / 92)
    const lines = _wrapTelop(text, maxEm).split('\n')
    expect(lines.some(l => l.includes('5.5'))).toBe(true)
    expect(lines.some(l => l.includes('3.5'))).toBe(true)
  })
})
