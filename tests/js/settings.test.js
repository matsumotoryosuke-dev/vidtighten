import { describe, test, expect } from 'vitest'
import { SETTINGS_DEFAULTS } from '../../src/preprod/static/modules/settings.js'

describe('SETTINGS_DEFAULTS', () => {
  test('object is defined and non-empty', () => {
    expect(SETTINGS_DEFAULTS).toBeDefined()
    expect(Object.keys(SETTINGS_DEFAULTS).length).toBeGreaterThan(0)
  })

  test('contains all expected keys', () => {
    const required = [
      'threshold', 'minDur', 'hangover', 'padding', 'model',
      'fEnable', 'fJapanese', 'fEnglish', 'fCustom',
      'tFps', 'tRes', 'tFont', 'tFontSize', 'tFontColor', 'tPosY', 'tLineSpacing',
    ]
    for (const key of required) {
      expect(SETTINGS_DEFAULTS).toHaveProperty(key)
    }
  })

  test('telop color is a valid hex string', () => {
    expect(SETTINGS_DEFAULTS.tFontColor).toMatch(/^#[0-9A-Fa-f]{6}$/)
  })

  test('telop resolution has WxH format', () => {
    expect(SETTINGS_DEFAULTS.tRes).toMatch(/^\d+x\d+$/)
  })

  test('telop fps is a numeric string', () => {
    expect(SETTINGS_DEFAULTS.tFps).toMatch(/^\d+(\.\d+)?$/)
  })

  test('telop font is non-empty string', () => {
    expect(typeof SETTINGS_DEFAULTS.tFont).toBe('string')
    expect(SETTINGS_DEFAULTS.tFont.length).toBeGreaterThan(0)
  })

  test('telop font size parses as a positive integer', () => {
    const sz = parseInt(SETTINGS_DEFAULTS.tFontSize, 10)
    expect(isNaN(sz)).toBe(false)
    expect(sz).toBeGreaterThan(0)
  })

  test('telop posY parses as an integer (can be negative for bottom)', () => {
    const y = parseInt(SETTINGS_DEFAULTS.tPosY, 10)
    expect(isNaN(y)).toBe(false)
  })

  test('telop lineSpacing parses as an integer (negative = tighter)', () => {
    const ls = parseInt(SETTINGS_DEFAULTS.tLineSpacing, 10)
    expect(isNaN(ls)).toBe(false)
    expect(ls).toBeLessThan(0)
  })

  test('filter flags are booleans', () => {
    expect(typeof SETTINGS_DEFAULTS.fEnable).toBe('boolean')
    expect(typeof SETTINGS_DEFAULTS.fJapanese).toBe('boolean')
    expect(typeof SETTINGS_DEFAULTS.fEnglish).toBe('boolean')
  })

  test('threshold parses as a negative number (silence detector dB)', () => {
    const t = parseFloat(SETTINGS_DEFAULTS.threshold)
    expect(isNaN(t)).toBe(false)
    expect(t).toBeLessThan(0)
  })
})
