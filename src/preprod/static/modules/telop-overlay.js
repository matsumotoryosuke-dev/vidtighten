/**
 * Telop overlay — renders the active telop entry as a live CSS overlay on the
 * video player so users can preview font, colour, size and position before export.
 *
 * Pure client-side: ports _clean_telop_text, _char_em, _wrap_telop from
 * fcpxml_telop.py so the preview text matches the exported FCPXML exactly.
 */

import { S, collectDeletedWordIds } from './state.js';
import { $ } from './utils.js';
import { rebuildTelopText } from './export-module.js';

// ── Text processors (JS ports of fcpxml_telop.py) ────────────────────────────

// "." is handled separately (_PERIOD_RE) — a bare period in this class would also
// strip decimal points in version numbers/model names ("GLM5.2" -> "GLM5 2").
const _SEP_RE    = /[。、！？・…〜～：；!?,;:\-—–\n]/g;
// A "." only counts as sentence-ending punctuation when it isn't touching a digit
// on either side — preserves decimal points while still splitting English sentences.
const _PERIOD_RE = /(?<!\d)\.(?!\d)/g;
const _WRAP_RE   = /[「」『』【】〈〉《》（）()"'""''“”‘’]/g;

export function _cleanTelopText(text) {
  text = text.replace(_SEP_RE, ' ');
  text = text.replace(_PERIOD_RE, ' ');
  text = text.replace(_WRAP_RE, '');
  return text.replace(/ {2,}/g, ' ').trim();
}

// Known parity notes vs Python unicodedata.east_asian_width():
//   1. "Ambiguous" (A) category — e.g. °, α, Greek letters → 1.0 in Python, 0.55 here.
//      This JS port omits the A category entirely.  Acceptable for JP telop text.
//   2. Emoji above U+FFFF — e.g. 😊 U+1F60A → 1.0 in Python (classified 'W'), 0.55
//      here because the code-point ranges only cover BMP characters up to U+FFE6.
//      Emoji are uncommon in Japanese broadcast subtitles so the gap is acceptable.
//   Note: © (U+00A9) is classified 'N' (Narrow) in Python on most platforms → 0.55
//   in both environments; it is NOT one of the parity-gap characters.
export function _charEm(ch) {
  const cp = ch.codePointAt(0);
  if ((cp >= 0x1100 && cp <= 0x115F) ||   // Hangul Jamo
      (cp >= 0x2E80 && cp <= 0x303F) ||   // CJK Radicals + Symbols
      (cp >= 0x3040 && cp <= 0xA4CF) ||   // Hiragana, Katakana, CJK Unified + ext
      (cp >= 0xA960 && cp <= 0xA97F) ||   // Hangul Jamo Ext-A
      (cp >= 0xAC00 && cp <= 0xD7FF) ||   // Hangul Syllables + Jamo Ext-B
      (cp >= 0xF900 && cp <= 0xFAFF) ||   // CJK Compat Ideographs
      (cp >= 0xFE10 && cp <= 0xFE1F) ||   // Vertical Presentation Forms (Wide/Fullwidth)
      (cp >= 0xFE30 && cp <= 0xFE6F) ||   // CJK Compat Forms + Small Form Variants
      // Note: U+FE20–U+FE2F (Combining Half Marks) are Narrow in Python and skipped here.
      (cp >= 0xFF01 && cp <= 0xFF60) ||   // Fullwidth ASCII / symbols
      (cp >= 0xFFE0 && cp <= 0xFFE6))     // Fullwidth signs
    return 1.0;
  return 0.55;
}

export function _emWidth(text) {
  return [...text].reduce((s, ch) => (ch !== '\n' ? s + _charEm(ch) : s), 0);
}

// True if breaking between prev/cur would split a numeric token ("5.5" -> "5" /
// ".5", or "123" -> "12" / "3"). By the time _wrapTelop runs, _cleanTelopText has
// already converted every non-numeric "." to a space (sentence-ending
// punctuation), so any "." still in the text is guaranteed to be a genuine
// decimal point — a plain adjacency check is sufficient, no wider scan needed.
function _breaksANumber(prev, cur) {
  const isDigitOrDot = ch => /\d/.test(ch) || ch === '.';
  return isDigitOrDot(prev) && isDigitOrDot(cur);
}

export function _breakScore(chars, pos) {
  if (pos <= 0 || pos >= chars.length) return 0.0;
  const prev = chars[pos - 1];
  const cur  = chars[pos];
  if (_breaksANumber(prev, cur)) return -10.0;
  if (prev === ' ') return 3.0;
  if ('はがをにと'.includes(prev)) return 2.0;
  if ('でもかのへ'.includes(prev)) return 1.0;
  if (prev === 'て') return 0.7;
  return 0.0;
}

export function _wrapTelop(text, maxEm) {
  const chars    = [...text];   // code-point safe (handles surrogate pairs)
  const totalEm  = chars.reduce((s, ch) => (ch !== '\n' ? s + _charEm(ch) : s), 0);
  if (totalEm <= maxEm) return text;

  const half = totalEm / 2.0;
  let bestPos      = Math.max(1, Math.floor(chars.length / 2));
  let bestCombined = -Infinity;
  let emBefore     = 0.0;

  for (let pos = 1; pos < chars.length; pos++) {
    emBefore += _charEm(chars[pos - 1]);
    const dist     = Math.abs(emBefore - half);
    const combined = _breakScore(chars, pos) - dist / half;
    if (combined > bestCombined) { bestCombined = combined; bestPos = pos; }
  }

  let line1 = chars.slice(0, bestPos).join('').trimEnd();
  let line2 = chars.slice(bestPos).join('').trimStart();

  // Hard-truncate line 2 if still overflows
  if (_emWidth(line2) > maxEm) {
    const l2chars = [...line2];   // spread once; reuse below
    let c2 = 0, trunc = l2chars.length;
    for (let j = 0; j < l2chars.length; j++) {
      c2 += _charEm(l2chars[j]);
      if (c2 > maxEm) { trunc = j; break; }
    }
    line2 = l2chars.slice(0, trunc).join('');
  }

  return `${line1}\n${line2}`;
}

// ── Text cache ────────────────────────────────────────────────────────────────
// Keyed by entry id; rebuilt whenever entries change or wrap settings change.

const _cache = new Map();   // entry.id → processed display text (with \n if wrapped)

/**
 * Parse an integer from element(id).value; return def when the string is
 * missing, empty, or non-numeric.  Uses isNaN rather than `|| def` so that
 * a valid value of 0 is preserved (e.g. posY=0 = vertical centre).
 */
function _intVal(id, def) {
  const v = parseInt($(id)?.value, 10);
  return isNaN(v) ? def : v;
}

function _rebuild() {
  _cache.clear();
  _naturalTextWidthCache.clear();
  if (!S.telopEntries.length) return;
  const sz    = Math.max(1, _intVal('t-font-size', 80));   // guard: sz=0 → Infinity
  const resW  = parseInt(($('t-res')?.value || '3840x2160').split('x')[0], 10) || 3840;
  const maxEm = Math.max(8.0, resW * 0.42 / sz);

  const deletedWordIds = collectDeletedWordIds();

  for (const e of S.telopEntries) {
    const rebuilt = rebuildTelopText(e, S.words || [], deletedWordIds);
    const cleaned = _cleanTelopText(rebuilt.text || '');
    _cache.set(e.id, cleaned ? _wrapTelop(cleaned, maxEm) : '');
  }
}

// ── Overlay DOM ───────────────────────────────────────────────────────────────

let _el     = null;   // #telop-overlay container div
let _textEl = null;   // absolutely-positioned <span> inside the overlay
let _enabled = false;

// Fix 6: Cache the last active telop entry to short-circuit the O(n) find() each frame.
let _lastActive = null;

// ── Natural line-height measurement ──────────────────────────────────────────
// FCP's lineSpacing is added to the font's own natural line height, which varies
// a lot by font (decorative/display Japanese fonts in particular can be far
// taller than their font-size implies) — a fixed ratio guess doesn't track that,
// so this measures the real value from a hidden probe rendered in the exact same
// font, memoized per fontFamily+fontSize since font/size only change on setting
// edits, not per animation frame.
let _lineHeightProbe = null;
const _naturalLineHeightCache = new Map();

function _naturalLineHeight(fontFamily, fontSize) {
  const key = `${fontFamily}|${fontSize}`;
  const cached = _naturalLineHeightCache.get(key);
  if (cached != null) return cached;
  if (!_lineHeightProbe) {
    _lineHeightProbe = document.createElement('span');
    _lineHeightProbe.style.cssText =
      'position:absolute;visibility:hidden;top:-9999px;left:-9999px;' +
      'font-weight:bold;line-height:normal;white-space:nowrap;';
    _lineHeightProbe.textContent = 'あ';
    document.body.appendChild(_lineHeightProbe);
  }
  _lineHeightProbe.style.fontFamily = fontFamily;
  _lineHeightProbe.style.fontSize   = `${fontSize}px`;
  const h = _lineHeightProbe.getBoundingClientRect().height || fontSize * 1.35;
  _naturalLineHeightCache.set(key, h);
  return h;
}

// _textEl is `position:absolute` with only `left` set (centered via a
// translate(-50%) transform, not `right`), and `width:auto` with only
// `max-width` set does NOT reliably shrink-to-fit its actual multi-line
// content in this configuration — observed rendering at roughly half the
// panel width regardless of content or how generous max-width was made,
// forcing every _wrapTelop-decided line to re-wrap into extra browser-level
// soft-wrapped lines (2026-07-20, live-verified; root cause of the CSS
// auto-width algorithm not fully pinned down, but explicit `width` reliably
// fixes it where max-width alone did not). Same probe technique as
// _naturalLineHeight above — measure the actual widest line in a hidden
// span, then set width explicitly instead of trusting auto-sizing.
let _textWidthProbe = null;
const _naturalTextWidthCache = new Map();

function _naturalTextWidth(fontFamily, fontSize, text) {
  const key = `${fontFamily}|${fontSize}|${text}`;
  const cached = _naturalTextWidthCache.get(key);
  if (cached != null) return cached;
  if (!_textWidthProbe) {
    _textWidthProbe = document.createElement('span');
    _textWidthProbe.style.cssText =
      'position:absolute;visibility:hidden;top:-9999px;left:-9999px;' +
      'font-weight:bold;white-space:pre;';   // pre: respects \n, never auto-wraps
    document.body.appendChild(_textWidthProbe);
  }
  _textWidthProbe.style.fontFamily = fontFamily;
  _textWidthProbe.style.fontSize   = `${fontSize}px`;
  _textWidthProbe.textContent = text;
  const w = _textWidthProbe.getBoundingClientRect().width;
  _naturalTextWidthCache.set(key, w);
  return w;
}

// ── Selection state ───────────────────────────────────────────────────────────

let _selected = false;
let _onSelectChange = null;   // callback(selected: bool)

function _setSelected(on) {
  _selected = on;
  _textEl.classList.toggle('telop-selected', on);
  if (_onSelectChange) _onSelectChange(on);
}

export function onTelopSelectChange(cb) { _onSelectChange = cb; }

export function initTelopOverlay() {
  _el = $('telop-overlay');
  if (!_el) return;   // no overlay div in this page variant — skip silently
  _textEl = document.createElement('span');
  _textEl.id = 'telop-text';
  // Static styles — set once at init; never change regardless of settings or player state.
  Object.assign(_textEl.style, {
    position:   'absolute',
    transform:  'translate(-50%, -50%)',
    fontWeight: 'bold',
    textAlign:  'center',
    whiteSpace: 'pre-line',
    textShadow: '0 1px 6px rgba(0,0,0,.9), 0 0 3px rgba(0,0,0,.7)',
    pointerEvents: 'none',
  });
  _el.appendChild(_textEl);

  // Click on text span → select
  _textEl.addEventListener('click', e => {
    e.stopPropagation();
    if (_textEl.textContent) _setSelected(true);
  });

  // Click on overlay background → deselect
  _el.addEventListener('click', () => _setSelected(false));

  // ESC → deselect
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && _selected) _setSelected(false);
  });
}

// ── Letterbox-aware style ─────────────────────────────────────────────────────

function _applyStyle() {
  const player = $('player');
  if (!player) return;   // page variant without <video id="player">

  let rW, rH, rX, rY, scale;

  if (!player.videoWidth) {
    // Audio-only: no real video frame. Synthesize one at the export resolution's
    // aspect ratio and letterbox it into the panel — same math as the video path
    // below — so telop size/position preview exactly as they will for a video file.
    const panel = $('video-panel') || player.parentElement;
    const cW = panel ? (panel.clientWidth  || 1) : 1;
    const cH = panel ? (panel.clientHeight || 1) : 1;
    if (cW <= 1 && cH <= 1) return;   // not laid out yet

    const res = ($('t-res')?.value || '3840x2160').split('x');
    const vW = parseInt(res[0], 10) || 3840;
    const vH = parseInt(res[1], 10) || 2160;

    if (cW / cH > vW / vH) {   // pillarbox
      rH = cH; rW = cH * vW / vH; rX = (cW - rW) / 2; rY = 0;
    } else {                    // letterbox
      rW = cW; rH = cW * vH / vW; rX = 0; rY = (cH - rH) / 2;
    }
    scale = rH / vH;
  } else {
    const cW = player.clientWidth  || 1;
    const cH = player.clientHeight || 1;
    if (cW <= 1 && cH <= 1) return;   // not laid out yet

    const vW = player.videoWidth;
    const vH = player.videoHeight;

    // Compute rendered video rect inside player (object-fit: contain) — this
    // MUST use the actually-loaded <video>'s own dimensions, since that's
    // what object-fit letterboxes against (proxy or original, same aspect
    // ratio either way).
    if (cW / cH > vW / vH) {   // pillarbox
      rH = cH; rW = cH * vW / vH; rX = (cW - rW) / 2; rY = 0;
    } else {                    // letterbox
      rW = cW; rH = cW * vH / vW; rX = 0; rY = (cH - rH) / 2;
    }
    // scale must convert EXPORT-CANVAS-space pixel values (font-size,
    // position, line-spacing — everything FCP interprets against the
    // project's Position/width/height settings) into rendered-panel pixels,
    // so it has to be relative to the export resolution (t-res), NOT the
    // loaded <video> element's own resolution. Files needing a preview proxy
    // (any source >1920 long edge) load an 800x450 proxy into #player, so
    // player.videoHeight there is the proxy's height, not the export
    // canvas's — using it directly rendered telop text ~4-5x too large
    // (e.g. scaling against a 450px reference instead of 2160px), which
    // then needed far more wrapped lines than the export ever will and
    // overflowed the video panel's clipped bounds entirely.
    const exportH = parseInt(($('t-res')?.value || '3840x2160').split('x')[1], 10) || 2160;
    scale = rH / exportH;
  }
  const posY    = _intVal('t-pos-y',          -420);
  // FCP's Basic Title generator does NOT map its Size/Position fields 1:1 to
  // pixels at the project's native resolution — confirmed by comparing this
  // preview against a real FCP import (Rio, 2026-07-20):
  //  - font_size=92 on a 4K/2160px project rendered text at roughly 8-9% of
  //    frame height in FCP, vs. the ~4.3% this scale alone predicts (92/2160).
  //  - posY=-420 on the same project sat the text at roughly 80% down the
  //    frame in FCP, vs. ~63-65% this scale alone produces — live-tested by
  //    doubling just the offset-from-center term and landing within ~1
  //    percentage point of the target.
  // Both match FCP's well-documented pattern of calibrating its title
  // generator against a reference resolution below the project's actual one
  // — a clean 2x is consistent with FCP referencing an HD-equivalent (1080p)
  // design space even on a 4K timeline (2160/1080 = 2 exactly), and it would
  // be an odd inconsistency for FCP to apply that referencing to Size but not
  // Position within the same generator. lineSp is NOT included here — it
  // hasn't been checked against a real FCP comparison the way size/position
  // just were, so it stays as-is rather than assuming it needs the same fix.
  const FCP_TITLE_COORD_SCALE = 2;
  const fontSz  = Math.max(8, Math.round(Math.max(1, _intVal('t-font-size', 92)) * scale * FCP_TITLE_COORD_SCALE));
  const lineSp  = Math.round(_intVal('t-line-spacing', -65) * scale);

  // FCP Y-axis: positive = up → CSS top: positive = down → negate posY.
  // Only the offset-from-center is scaled by FCP_TITLE_COORD_SCALE, not rH/2
  // itself — rH/2 is a real preview-panel pixel (the vertical center of the
  // rendered video rect), not an FCP coordinate-space value.
  const cssTop  = rY + rH / 2 - posY * scale * FCP_TITLE_COORD_SCALE;
  const cssLeft = rX + rW / 2;

  const fontFamily = `"${$('t-font')?.value ?? 'Hiragino Sans'}", "Hiragino Sans", sans-serif`;
  // FCP lineSpacing is added to the font's own natural line height, not to
  // the raw font size — treating it as `fontSz + lineSp` collapsed multi-line
  // telops to near-zero leading (e.g. 52px font + -29 spacing = 23px line
  // height) and lines rendered on top of each other in the preview, even
  // though the exported FCPXML rendered fine (FCP applies its own natural
  // line height there; this CSS approximation was just wrong). Measuring the
  // real natural height off a hidden probe — rather than guessing a fixed
  // ratio — tracks whatever font is actually active, since decorative/display
  // Japanese fonts can have a natural line height far taller than font-size
  // implies. Floored so aggressive negative spacing can still tighten lines
  // without stacking them.
  const naturalLH = _naturalLineHeight(fontFamily, fontSz);

  // _textEl's <span> needs SOME width constraint, or the browser is free to
  // shrink it to whatever "preferred width" it likes for pre-line-wrapped
  // CJK text (no spaces to hint a narrower wrap point), letting a single
  // _wrapTelop-decided logical line re-wrap into extra visual lines (Round 7,
  // originally fixed with `max-width`). That stopped being reliable once this
  // element's actual configuration was examined closely (2026-07-20): with
  // `width:auto` and only `left` set (not `right`, centered instead via the
  // translate(-50%) transform), the browser's shrink-to-fit resolved to
  // roughly HALF the panel width regardless of content or how generous
  // max-width was made — max-width was never the actual constraint, so no
  // amount of tuning its value (0.42 → 0.9 → 0.98, each tried and verified
  // insufficient in turn) could have fixed it. `width: fit-content` and
  // explicit `display: inline-block` didn't change the outcome either; a
  // literal `width: 400px` did. So: measure the real natural width (same
  // hidden-probe technique as _naturalLineHeight above) and set an explicit
  // width instead of trusting any auto-sizing keyword — capped at the
  // rendered video width as a sane ceiling for pathologically long text.
  const naturalW = _naturalTextWidth(fontFamily, fontSz, _textEl.textContent);
  const widthPx  = Math.min(naturalW, rW * 0.98);

  // Dynamic styles only — static props (fontWeight, textAlign, whiteSpace,
  // textShadow, transform) are applied once in initTelopOverlay.
  Object.assign(_textEl.style, {
    top:        `${cssTop}px`,
    left:       `${cssLeft}px`,
    width:      `${widthPx}px`,
    fontFamily,
    fontSize:   `${fontSz}px`,
    color:      $('t-font-color')?.value ?? '#F3B500',
    lineHeight: `${Math.max(naturalLH + lineSp, fontSz * 0.6)}px`,
  });
}

// ── Public API ────────────────────────────────────────────────────────────────

/** Rebuild the text cache and refresh the display.
 * Call when telopEntries change, when t-font-size / t-res change (max_em
 * depends on both), or when settings are applied/reverted programmatically. */
export function invalidateTelopOverlay() {
  _lastActive = null;   // Fix 6: force re-find after entries or settings change
  _rebuild();
  if (_enabled && _el) {
    _el.style.display = S.telopEntries.length > 0 ? 'block' : 'none';
    const player = $('player');
    if (player) tickTelopOverlay(player.currentTime);
  }
}

/** Called every RAF frame with current playback time. */
export function tickTelopOverlay(ct) {
  if (!_enabled || !S.telopEntries.length) return;

  // Fix 6: Check if the cached entry is still valid before doing a full O(n) find().
  let active;
  if (
    _lastActive &&
    S.telopsEnabled.has(_lastActive.id) &&
    ct >= _lastActive.start &&
    ct < _lastActive.end
  ) {
    active = _lastActive;
  } else {
    active = S.telopEntries.find(
      e => S.telopsEnabled.has(e.id) && ct >= e.start && ct < e.end
    ) ?? null;
    _lastActive = active;
  }

  if (!active) {
    if (_selected) _setSelected(false);   // deselect when entry ends
    _textEl.textContent = '';
    _textEl.style.pointerEvents = 'none';
    return;
  }

  if (!_cache.has(active.id)) _rebuild();   // lazy rebuild on cache miss
  _textEl.textContent = _cache.get(active.id) ?? _cleanTelopText(active.text || '');
  _textEl.style.pointerEvents = 'auto';
  _applyStyle();
}

/** Toggle the overlay on/off. Also updates the button's .active class. */
export function setTelopOverlayEnabled(on) {
  if (!_el) return;   // overlay not in DOM for this page variant — skip silently
  _enabled = on;
  const show = on && S.telopEntries.length > 0;
  _el.style.display = show ? 'block' : 'none';
  if (show) {
    _rebuild();
    const player = $('player');
    if (player) tickTelopOverlay(player.currentTime);
  } else {
    if (_selected) _setSelected(false);
    _textEl.textContent = '';
    _textEl.style.pointerEvents = 'none';
  }
  const btn = $('btn-telop-overlay');
  if (btn) btn.classList.toggle('active', on);
}

/**
 * Show or hide the toggle button.
 * Pass true when a file is loaded (video or audio-only); false when no file is loaded.
 * Also disables the overlay and clears button active state when unavailable.
 */
export function setTelopOverlayAvailable(available) {
  const btn = $('btn-telop-overlay');
  if (btn) btn.style.display = available ? '' : 'none';
  if (!available && _enabled && _el) {
    _enabled = false;
    _el.style.display = 'none';
    if (_selected) _setSelected(false);
    _textEl.textContent = '';
    _textEl.style.pointerEvents = 'none';
    // Remove active state from button so it doesn't appear "on" when a new
    // video file is loaded after the overlay was enabled on the previous file.
    if (btn) btn.classList.remove('active');
  }
}
