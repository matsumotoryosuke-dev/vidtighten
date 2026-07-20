/**
 * First-run dependency preflight — pure decision logic.
 *
 * Consolidates the capability checks the backend already performs
 * (GET /api/capabilities, GET /api/llm/models) into the two severities the
 * UI needs to show:
 *
 *  - CRITICAL: ffmpeg/ffprobe missing → the app cannot function at all.
 *    See missingCoreDeps() / coreDepBlockerMessage().
 *  - DEGRADED: whisper/whisperx/japanese/ollama missing → the app still
 *    works, just with reduced features. See degradedMessages() /
 *    formatDegradedToast().
 *
 * This module is intentionally DOM-free and import-free (no state.js, no
 * fetch) so it can be unit-tested directly under the project's Node-only
 * vitest environment (see tests/js/preflight.test.js) — the same split used
 * by tx-search.js (pure helpers here, DOM wiring lives in analysis.js's
 * checkCaps(), which calls these functions and is verified manually, per the
 * project's established testing convention for DOM-dependent code).
 *
 * Ollama is checked separately from the rest of /api/capabilities (see that
 * route's docstring in web.py) because it requires a live network call to
 * the local Ollama daemon and must never delay the ffmpeg/ffprobe blocker.
 * Callers pass its availability in once resolved, independently of `caps`.
 */

// The only two hard requirements — everything else in the capabilities
// report is optional and degrades gracefully instead of blocking.
const CORE_DEPS = [
  { key: 'ffmpeg',  label: 'ffmpeg' },
  { key: 'ffprobe', label: 'ffprobe' },
];

/**
 * Which CRITICAL dependencies are missing, given an /api/capabilities
 * response. Returns e.g. [] | ['ffmpeg'] | ['ffprobe'] | ['ffmpeg','ffprobe'].
 * A missing/undefined `caps` (e.g. the fetch itself failed) is treated as
 * "everything critical is missing" — fail closed rather than silently
 * assuming the app can run.
 */
export function missingCoreDeps(caps) {
  return CORE_DEPS.filter(d => !caps?.[d.key]).map(d => d.label);
}

/**
 * User-facing text for the ffmpeg/ffprobe hard blocker. ffprobe ships inside
 * the same ffmpeg distribution, so one set of install instructions covers
 * either or both being missing. Mirrors the wording already used deeper in
 * the pipeline (audio.py's _FFMPEG_NOT_FOUND / probe.py's ffprobe message)
 * so the user sees consistent phrasing regardless of where they hit it.
 * Returns '' when nothing is missing.
 */
export function coreDepBlockerMessage(missing) {
  if (!missing || !missing.length) return '';
  const names = missing.join(' and ');
  return `VidTighten can’t run without ${names}. Install ffmpeg from `
    + `https://ffmpeg.org (ffprobe is included in the same download), then `
    + `restart VidTighten.`;
}

/**
 * Non-blocking "what's reduced" messages for optional dependencies, given an
 * /api/capabilities response plus a separately-resolved Ollama availability
 * flag (true/false once known, undefined/null if not checked yet or the
 * check itself failed — in which case Ollama is simply omitted rather than
 * guessed at).
 *
 * whisperx is only reported when whisper IS available — WhisperX only
 * refines an existing faster-whisper transcript, so if there's no
 * transcript at all the whisper message already covers it and a second
 * "no whisperx" message would be redundant/confusing.
 */
export function degradedMessages(caps, ollamaAvailable) {
  const msgs = [];
  if (!caps?.whisper_available) {
    msgs.push('no transcription available (Whisper isn’t installed) — filler-word detection and captions are unavailable');
  } else if (!caps?.whisperx) {
    msgs.push('word alignment uses DTW only (WhisperX isn’t installed) — slightly less precise sync than WhisperX’s forced alignment');
  }
  if (!caps?.japanese) {
    msgs.push('Japanese line breaks use character-count heuristics (fugashi/UniDic isn’t installed) instead of phrase-boundary-aware breaks');
  }
  if (ollamaAvailable === false) {
    msgs.push('local LLM transcript correction isn’t available (Ollama isn’t running) — this feature is optional and off by default anyway');
  }
  return msgs;
}

/**
 * Join degradedMessages() output into a single toast string, or null when
 * there's nothing to report. One call, one message — utils.toast() is a
 * single-slot notification (see utils.js), so firing several toast() calls
 * back to back would just clobber each other rather than queue.
 */
export function formatDegradedToast(messages) {
  if (!messages || !messages.length) return null;
  return `Reduced functionality: ${messages.join('; ')}.`;
}
