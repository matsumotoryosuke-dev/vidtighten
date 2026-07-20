"""Japanese phrase-boundary detection via morphological analysis (fugashi/UniDic).

The telop splitter and line-wrapper must cut text only where a 、 or 。 would be
grammatically appropriate — at a 文節 (phrase) boundary — never mid-word and never
mid-phrase.  This module identifies those boundaries.  It does NOT insert any
punctuation: it only reports *where a cut would be natural* and how natural it is.

fugashi is an OPTIONAL dependency.  Every public function degrades gracefully —
returning None — when fugashi/UniDic is not installed, so callers fall back to
their character-level heuristics and nothing breaks.

A 文節 begins with an independent word (自立語) and absorbs the dependent words
(付属語: 助詞/助動詞/接尾辞) that follow it.  A boundary therefore falls *before*
each independent word.  The naturalness score of a boundary is determined by the
LAST morpheme of the phrase that ends there — a phrase ending in a particle is a
natural 、 spot; one ending in a dangling adverb or a verb stem is not.
"""

from __future__ import annotations

import threading
from functools import lru_cache

# Independent words (自立語) — each starts a new 文節.
_INDEP_POS = {
    "名詞", "代名詞", "動詞", "形容詞", "形状詞",
    "副詞", "連体詞", "接続詞", "感動詞",
}
# Dependent words (付属語) — attach to the preceding 文節, never start one.
_DEP_POS = {"助詞", "助動詞", "接尾辞"}


@lru_cache(maxsize=1)
def _tagger():
    """Return a cached fugashi Tagger, or None if fugashi/UniDic is unavailable."""
    try:
        import fugashi  # type: ignore
        return fugashi.Tagger()
    except Exception:
        return None


def available() -> bool:
    """True if morphological analysis is usable."""
    return _tagger() is not None


# fugashi wraps MeCab, whose Tagger keeps a single internal parse buffer that is
# overwritten on every call — the shared instance is NOT thread-safe.  Flask runs
# threaded=True, so concurrent analyze/export requests can interleave parses and
# crash or corrupt output.  Serialize all tagger access through this lock.
_TAGGER_LOCK = threading.Lock()


def _morphemes(text: str):
    """Return [(surface, pos1, pos2, cForm, start_char, end_char), ...] or None.

    start/end are character offsets into *text*.  Offsets are derived by locating
    each surface form in sequence, which keeps them aligned with the original text
    even though UniDic itself does not emit offsets.
    """
    tagger = _tagger()
    if tagger is None or not text:
        return None
    out = []
    pos = 0
    # Hold the lock across the whole parse+read: node attributes (w.surface,
    # w.feature) are read lazily from MeCab's shared buffer, so accessing them
    # after another thread re-parses would return clobbered data.
    with _TAGGER_LOCK:
        for w in tagger(text):
            surf = w.surface
            if not surf:
                continue
            idx = text.find(surf, pos)
            if idx < 0:
                idx = pos                       # fallback: assume contiguous
            f = w.feature
            out.append((
                surf,
                f.pos1,
                getattr(f, "pos2", None),
                getattr(f, "cForm", None),
                idx,
                idx + len(surf),
            ))
            pos = idx + len(surf)
    return out


def _starts_bunsetsu(m, prev) -> bool:
    """True if morpheme *m* begins a new 文節 (given the previous morpheme *prev*)."""
    surf, pos1, pos2, cForm, _s, _e = m
    if pos1 in _DEP_POS:
        return False                        # particle/aux attaches backward
    if pos1 == "接頭辞":
        return False                        # prefix attaches forward (joins next)
    if pos2 == "非自立可能":
        return False                        # 補助用言 (e.g. て+いる) attaches backward
    if pos1 not in _INDEP_POS:
        return False                        # 補助記号, 空白, etc. — not a phrase head
    if prev is not None and prev[1] == "接頭辞":
        return False                        # previous prefix binds to this word
    return True


def _phrase_end_score(last_morpheme) -> float:
    """Naturalness of placing a 、/。 after *last_morpheme* (higher = better cut).

    Mirrors the tiers of fcpxml_telop._break_score but driven by real POS tags.
    """
    surf, pos1, pos2, cForm, _s, _e = last_morpheme
    if pos1 == "助動詞" and cForm and ("終止形" in cForm or "命令形" in cForm):
        return 4.0                          # sentence end → 。
    if pos1 == "助詞":
        if pos2 == "接続助詞":
            return 3.0                      # clause connector (て/から/けど/が) → strong 、
        if pos2 in ("係助詞", "格助詞", "副助詞"):
            return 2.5                      # topic/case/adverbial particle → natural 、
        return 1.5                          # other particle
    if pos1 == "助動詞":
        if cForm and "連体形" in cForm:
            return 0.6                      # modifies the next noun — weak cut
        return 1.5
    if pos1 == "接尾辞":
        return 1.0
    if pos1 == "動詞" and cForm and ("連用形" in cForm or "未然形" in cForm):
        return 0.2                          # verb stem expecting continuation
    if pos1 in ("副詞", "連体詞", "接続詞", "接頭辞"):
        return 0.1                          # dangling modifier — avoid (e.g. "そう")
    return 0.5                              # bare noun / 終止 verb — acceptable but plain


def break_scores(text: str):
    """Return {char_offset: naturalness_score} for every 文節 boundary in *text*.

    Offsets are positions where a cut may occur (0 < offset < len(text)); each is a
    real phrase boundary.  Returns None if morphological analysis is unavailable.
    """
    ms = _morphemes(text)
    if ms is None:
        return None
    scores: dict[int, float] = {}
    prev = None
    for i, m in enumerate(ms):
        if i > 0 and _starts_bunsetsu(m, prev):
            offset = m[4]                   # boundary falls before this phrase head
            if 0 < offset < len(text):
                scores[offset] = _phrase_end_score(ms[i - 1])
        prev = m
    return scores


def crosses_morpheme(prev_text: str, next_text: str) -> bool:
    """True if a single morpheme straddles the boundary between the two texts.

    This is the unambiguous mid-word case (e.g. "元Goo" | "gleの" → "Google" is one
    morpheme spanning the cut).  A word cannot contain a real pause, so callers may
    merge such a boundary regardless of the inter-segment gap.  Returns False when
    analysis is unavailable.
    """
    prev_text = (prev_text or "").rstrip()
    next_text = (next_text or "").lstrip()
    if not prev_text or not next_text:
        return False
    boundary = len(prev_text)
    ms = _morphemes(prev_text + next_text)
    if ms is None:
        return False
    return any(m[4] < boundary < m[5] for m in ms)


def is_continuation(prev_text: str, next_text: str) -> bool:
    """True if the boundary between *prev_text* and *next_text* falls mid-phrase.

    Used to decide whether two adjacent telop segments were split mid-word/mid-
    phrase by Whisper and should be merged before re-splitting.  Returns False when
    analysis is unavailable (caller keeps its own heuristics).

    A boundary is mid-phrase when, analysing the joined text:
    - a morpheme straddles the boundary (mid-word), or
    - the morpheme just after it is a dependent word (付属語/補助用言) that attaches
      backward, or
    - the morpheme just before it is a prefix or a dangling modifier
      (副詞/連体詞/接続詞) that attaches forward (e.g. "そう" | "いった").
    """
    prev_text = (prev_text or "").rstrip()
    next_text = (next_text or "").lstrip()
    if not prev_text or not next_text:
        return False
    boundary = len(prev_text)
    ms = _morphemes(prev_text + next_text)
    if ms is None:
        return False
    before = after = None
    for m in ms:
        s, e = m[4], m[5]
        if s < boundary < e:
            return True                     # morpheme straddles the cut → mid-word
        if e == boundary:
            before = m
        if s == boundary:
            after = m
    if after is not None:
        pos1, pos2 = after[1], after[2]
        if pos1 in _DEP_POS or pos2 == "非自立可能":
            return True                     # next attaches backward
    if before is not None:
        pos1 = before[1]
        if pos1 == "接頭辞" or pos1 in ("副詞", "連体詞", "接続詞"):
            return True                     # prev attaches forward (dangling)
    return False
