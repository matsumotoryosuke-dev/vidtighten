// QA coverage note: onTelopSelectChange is the only exported symbol from the
// click-to-select feature that can be exercised without a DOM environment.
// The full selection behaviour (CSS toggling, ESC handler, panel swap) requires
// jsdom or happy-dom (npm i -D jsdom) with the vitest-environment annotation
// on a dedicated DOM test file to cover those paths.

import { describe, test, expect, vi } from 'vitest'
import { onTelopSelectChange } from '../../src/preprod/static/modules/telop-overlay.js'

describe('onTelopSelectChange (node-safe export)', () => {
  test('registers a callback without throwing', () => {
    const cb = vi.fn()
    expect(() => onTelopSelectChange(cb)).not.toThrow()
  })

  test('accepts null to clear the callback', () => {
    expect(() => onTelopSelectChange(null)).not.toThrow()
  })

  test('accepts undefined to clear the callback', () => {
    expect(() => onTelopSelectChange(undefined)).not.toThrow()
  })

  test('replaces a previously registered callback', () => {
    const cb1 = vi.fn()
    const cb2 = vi.fn()
    expect(() => {
      onTelopSelectChange(cb1)
      onTelopSelectChange(cb2)
    }).not.toThrow()
  })
})
