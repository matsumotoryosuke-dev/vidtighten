"""Custom-vocabulary corrections for transcription output.

Whisper mis-transcribes domain-specific terms (brand names, proper nouns) as
their homophone kanji.  The canonical example: the channel name "クウキデザイン"
(Kuuki Design) is heard as the kanji "空気デザイン" (literally "air design").

Corrections are applied to the *word token* list right after transcription so
every downstream consumer — telop FCPXML, subtitles, and the word-level editor —
sees the corrected text from a single source of truth.  Telop and subtitle text
are both rebuilt from the word list on export, so correcting words is sufficient.

Keys must be specific enough to avoid false positives: "空気デザイン" is always
the brand, whereas bare "空気" (air) is a real word and must never be touched.
Corrections operate on the token's text only and never alter timestamps, so card
timing is unaffected even when the replacement differs in length (e.g. 空気デザイン
→ クウキデザイン adds one character of display width).
"""

from __future__ import annotations

import json
from pathlib import Path

# Literal replacements, applied longest-key-first so a shorter key never
# pre-empts a longer one that contains it.
BRAND_CORRECTIONS: dict[str, str] = {
    "空気デザイン": "クウキデザイン",
}

# User-grown glossary, appended to over time when Rio approves an LLM brand-noun
# suggestion with "always apply this" (see llm_correct.py). Merged OVER the
# built-in defaults so the shipped source stays fixed while Rio's vocabulary
# grows independently. Missing/broken file → just the built-ins.
_USER_CORRECTIONS_PATH = Path.home() / ".preprod" / "corrections.json"

# Cache the merged view; invalidated on save. Keyed off nothing but process
# lifetime + explicit invalidation — correct_text is called per-segment at
# export, so re-reading the file each call would be wasteful.
_active_cache: dict[str, str] | None = None


def load_user_corrections() -> dict[str, str]:
    """Read the user glossary file. Returns {} for a missing or malformed file
    (a broken glossary must never break transcription/export)."""
    try:
        data = json.loads(_USER_CORRECTIONS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    # Keep only str→str entries; ignore anything malformed.
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str) and k}


def active_corrections() -> dict[str, str]:
    """Built-in defaults merged with the user glossary (user wins). Cached."""
    global _active_cache
    if _active_cache is None:
        _active_cache = {**BRAND_CORRECTIONS, **load_user_corrections()}
    return _active_cache


def invalidate_corrections_cache() -> None:
    """Drop the merged-view cache so the next call re-reads the user glossary."""
    global _active_cache
    _active_cache = None


def save_user_correction(wrong: str, correct: str) -> None:
    """Add (or overwrite) one wrong→correct mapping in the user glossary file,
    then invalidate the cache so it takes effect immediately."""
    if not wrong or not isinstance(wrong, str) or not isinstance(correct, str):
        return
    current = load_user_corrections()
    current[wrong] = correct
    _USER_CORRECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _USER_CORRECTIONS_PATH.write_text(
        json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    invalidate_corrections_cache()


def correct_text(text: str, corrections: dict[str, str] | None = None) -> str:
    """Return *text* with every correction key replaced by its value.

    corrections=None (the default) uses the merged built-in + user glossary via
    active_corrections(); pass an explicit dict to override (tests do this)."""
    if not text:
        return text
    if corrections is None:
        corrections = active_corrections()
    for wrong, right in sorted(corrections.items(), key=lambda kv: -len(kv[0])):
        if wrong in text:
            text = text.replace(wrong, right)
    return text


def correct_words(
    words: list[dict], corrections: dict[str, str] | None = None
) -> int:
    """Apply corrections in-place to each word token's display text.

    Accepts both the 'word' key (raw faster-whisper / openai-whisper output) and
    the 'text' key (grouped tokens).  Returns the number of tokens changed.
    corrections=None uses the merged built-in + user glossary."""
    if corrections is None:
        corrections = active_corrections()
    changed = 0
    for w in words:
        key = "word" if "word" in w else ("text" if "text" in w else None)
        if key is None:
            continue
        new = correct_text(w[key], corrections)
        if new != w[key]:
            w[key] = new
            changed += 1
    return changed
