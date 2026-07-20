import { describe, test, expect } from 'vitest'
import { fmt, fmtDur, fmtElapsed, plur, escH, _fmtBytes,
         EVT_FILE_LOADED, EVT_RESULT_APPLIED, EVT_STATE_RESET,
} from '../../src/preprod/static/modules/utils.js'

describe('fmt', () => {
  test('returns em-dash for null', () => {
    expect(fmt(null)).toBe('—')
  })
  test('returns em-dash for undefined (treated as null via ==)', () => {
    expect(fmt(undefined)).toBe('—')
  })
  test('returns em-dash for NaN', () => {
    expect(fmt(NaN)).toBe('—')
  })
  test('formats zero', () => {
    expect(fmt(0)).toBe('0:00.0')
  })
  test('rounds tenths for very small input (0.05)', () => {
    expect(fmt(0.05)).toBe('0:00.1')
  })
  test('formats sub-minute with tenths', () => {
    expect(fmt(65)).toBe('1:05.0')
  })
  test('includes tenths of a second', () => {
    expect(fmt(90.5)).toBe('1:30.5')
  })
  test('omits hours when zero', () => {
    expect(fmt(59)).toBe('0:59.0')
  })
  test('formats hours:minutes:seconds', () => {
    expect(fmt(3661)).toBe('1:01:01')
  })
  test('formats exactly one hour', () => {
    expect(fmt(3600)).toBe('1:00:00')
  })
})

describe('fmtDur', () => {
  test('sub-100ms shows ms', () => {
    expect(fmtDur(0.05)).toBe('50ms')
  })
  test('zero shows 0ms', () => {
    expect(fmtDur(0)).toBe('0ms')
  })
  test('just below 100ms stays in ms', () => {
    expect(fmtDur(0.099)).toBe('99ms')
  })
  test('exactly 0.1s shows seconds', () => {
    expect(fmtDur(0.1)).toBe('0.1s')
  })
  test('formats seconds to one decimal', () => {
    expect(fmtDur(2.5)).toBe('2.5s')
  })
  test('formats whole seconds', () => {
    expect(fmtDur(10)).toBe('10.0s')
  })
})

describe('plur', () => {
  test('singular', () => {
    expect(plur(1, 'cut')).toBe('1 cut')
  })
  test('plural', () => {
    expect(plur(2, 'cut')).toBe('2 cuts')
  })
  test('zero is plural', () => {
    expect(plur(0, 'file')).toBe('0 files')
  })
})

describe('fmtElapsed', () => {
  test('under a minute shows seconds', () => {
    expect(fmtElapsed(30)).toBe('30s')
  })
  test('59s stays in seconds form', () => {
    expect(fmtElapsed(59)).toBe('59s')
  })
  test('exactly 60s flips to minutes', () => {
    expect(fmtElapsed(60)).toBe('1m 0s')
  })
  test('90s shows 1m 30s', () => {
    expect(fmtElapsed(90)).toBe('1m 30s')
  })
})

describe('escH', () => {
  test('escapes angle brackets', () => {
    expect(escH('<b>hi</b>')).toBe('&lt;b&gt;hi&lt;/b&gt;')
  })
  test('escapes ampersand', () => {
    expect(escH('a & b')).toBe('a &amp; b')
  })
  test('escapes double quotes', () => {
    expect(escH('"quoted"')).toBe('&quot;quoted&quot;')
  })
  test('coerces non-string input', () => {
    expect(escH(42)).toBe('42')
  })
  test('no-op on clean string', () => {
    expect(escH('hello world')).toBe('hello world')
  })
})

describe('_fmtBytes', () => {
  test('bytes', () => {
    expect(_fmtBytes(500)).toBe('500 B')
  })
  test('kilobytes', () => {
    expect(_fmtBytes(1536)).toBe('1.5 KB')
  })
  test('exactly 1 MB', () => {
    expect(_fmtBytes(1048576)).toBe('1.0 MB')
  })
  test('gigabytes', () => {
    expect(_fmtBytes(2147483648)).toBe('2.00 GB')
  })
})

describe('EVT_* constants', () => {
  test('all constants are strings', () => {
    expect(typeof EVT_FILE_LOADED).toBe('string')
    expect(typeof EVT_RESULT_APPLIED).toBe('string')
    expect(typeof EVT_STATE_RESET).toBe('string')
  })

  test('all constants are distinct', () => {
    const names = [EVT_FILE_LOADED, EVT_RESULT_APPLIED, EVT_STATE_RESET]
    expect(new Set(names).size).toBe(3)
  })

  test('EVT_FILE_LOADED is analysis:fileLoaded', () => {
    expect(EVT_FILE_LOADED).toBe('analysis:fileLoaded')
  })

  test('EVT_RESULT_APPLIED is analysis:resultApplied', () => {
    expect(EVT_RESULT_APPLIED).toBe('analysis:resultApplied')
  })

  test('EVT_STATE_RESET is analysis:stateReset', () => {
    expect(EVT_STATE_RESET).toBe('analysis:stateReset')
  })
})
