// preflight.js is intentionally DOM-free and import-free (see its header
// comment), so unlike most other modules in tests/js/ (which document a gap
// where DOM-dependent code is verified manually instead — see
// dom_env_check.test.js / tx-search.test.js), every exported function here
// is fully covered under the project's Node-only vitest environment. The DOM
// wiring that CONSUMES these functions (checkCaps() in analysis.js) follows
// the same manual-verification convention as those other modules.

import { describe, test, expect } from 'vitest'
import {
  missingCoreDeps, coreDepBlockerMessage, degradedMessages, formatDegradedToast,
} from '../../src/preprod/static/modules/preflight.js'

describe('missingCoreDeps', () => {
  test('returns [] when ffmpeg and ffprobe are both present', () => {
    expect(missingCoreDeps({ ffmpeg: true, ffprobe: true })).toEqual([])
  })
  test('returns ["ffmpeg"] when only ffmpeg is missing', () => {
    expect(missingCoreDeps({ ffmpeg: false, ffprobe: true })).toEqual(['ffmpeg'])
  })
  test('returns ["ffprobe"] when only ffprobe is missing', () => {
    expect(missingCoreDeps({ ffmpeg: true, ffprobe: false })).toEqual(['ffprobe'])
  })
  test('returns both, ffmpeg first, when both are missing', () => {
    expect(missingCoreDeps({ ffmpeg: false, ffprobe: false })).toEqual(['ffmpeg', 'ffprobe'])
  })
  test('is not affected by unrelated fields on the capabilities object', () => {
    expect(missingCoreDeps({ ffmpeg: true, ffprobe: true, whisperx: false, japanese: false }))
      .toEqual([])
  })
  test('fails closed (treats as missing) when caps is undefined', () => {
    expect(missingCoreDeps(undefined)).toEqual(['ffmpeg', 'ffprobe'])
  })
  test('fails closed (treats as missing) when caps is an empty object', () => {
    expect(missingCoreDeps({})).toEqual(['ffmpeg', 'ffprobe'])
  })
})

describe('coreDepBlockerMessage', () => {
  test('returns empty string when nothing is missing', () => {
    expect(coreDepBlockerMessage([])).toBe('')
  })
  test('mentions ffmpeg and the install link when only ffmpeg is missing', () => {
    const msg = coreDepBlockerMessage(['ffmpeg'])
    expect(msg).toContain('ffmpeg')
    expect(msg).toContain('https://ffmpeg.org')
  })
  test('mentions ffprobe when only ffprobe is missing', () => {
    const msg = coreDepBlockerMessage(['ffprobe'])
    expect(msg).toContain('ffprobe')
    expect(msg).toContain('https://ffmpeg.org')
  })
  test('mentions both names when both are missing', () => {
    const msg = coreDepBlockerMessage(['ffmpeg', 'ffprobe'])
    expect(msg).toContain('ffmpeg and ffprobe')
  })
  test('tells the user to restart the app', () => {
    expect(coreDepBlockerMessage(['ffmpeg'])).toMatch(/restart/i)
  })
})

describe('degradedMessages', () => {
  const fullCaps = { whisper_available: true, whisperx: true, japanese: true }

  test('returns [] when everything is available', () => {
    expect(degradedMessages(fullCaps, true)).toEqual([])
  })

  test('reports missing whisper transcription', () => {
    const msgs = degradedMessages({ ...fullCaps, whisper_available: false }, true)
    expect(msgs.some(m => /transcription/i.test(m))).toBe(true)
  })

  test('does NOT also report whisperx separately when whisper itself is missing', () => {
    const msgs = degradedMessages({ ...fullCaps, whisper_available: false, whisperx: false }, true)
    expect(msgs.some(m => /whisperx/i.test(m))).toBe(false)
    expect(msgs).toHaveLength(1)
  })

  test('reports DTW-only alignment when whisper is present but whisperx is not', () => {
    const msgs = degradedMessages({ ...fullCaps, whisperx: false }, true)
    expect(msgs.some(m => /whisperx/i.test(m) || /DTW/.test(m))).toBe(true)
  })

  test('reports missing japanese phrase-boundary detection', () => {
    const msgs = degradedMessages({ ...fullCaps, japanese: false }, true)
    expect(msgs.some(m => /japanese/i.test(m))).toBe(true)
  })

  test('reports ollama only when explicitly false, phrased as optional', () => {
    const msgs = degradedMessages(fullCaps, false)
    expect(msgs.some(m => /ollama/i.test(m))).toBe(true)
    expect(msgs.some(m => /optional/i.test(m))).toBe(true)
  })

  test('omits ollama when availability is unknown (undefined) rather than guessing', () => {
    const msgs = degradedMessages(fullCaps, undefined)
    expect(msgs.some(m => /ollama/i.test(m))).toBe(false)
  })

  test('omits ollama when availability is true', () => {
    const msgs = degradedMessages(fullCaps, true)
    expect(msgs.some(m => /ollama/i.test(m))).toBe(false)
  })

  test('combines multiple simultaneous gaps', () => {
    const msgs = degradedMessages({ whisper_available: true, whisperx: true, japanese: false }, false)
    expect(msgs).toHaveLength(2)
    expect(msgs.some(m => /japanese/i.test(m))).toBe(true)
    expect(msgs.some(m => /ollama/i.test(m))).toBe(true)
  })

  test('handles a missing/undefined caps object without throwing (fails closed)', () => {
    expect(() => degradedMessages(undefined, undefined)).not.toThrow()
    expect(degradedMessages(undefined, undefined).length).toBeGreaterThan(0)
  })
})

describe('formatDegradedToast', () => {
  test('returns null for an empty list', () => {
    expect(formatDegradedToast([])).toBeNull()
  })
  test('returns null for a null/undefined list', () => {
    expect(formatDegradedToast(null)).toBeNull()
    expect(formatDegradedToast(undefined)).toBeNull()
  })
  test('formats a single message', () => {
    expect(formatDegradedToast(['no transcription'])).toBe('Reduced functionality: no transcription.')
  })
  test('joins multiple messages with a semicolon', () => {
    expect(formatDegradedToast(['a', 'b', 'c'])).toBe('Reduced functionality: a; b; c.')
  })
})
