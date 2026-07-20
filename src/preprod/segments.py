"""Segment model — invert removal regions into keep-segments with padding."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from preprod import japanese


@dataclass
class Segment:
    source_start: float
    source_end: float

    @property
    def duration(self) -> float:
        """Length of this keep-segment in seconds."""
        return self.source_end - self.source_start


def build_segments(
    removal_regions: list[tuple[float, float]],
    total_duration: float,
    padding_ms: int = 0,
) -> list[Segment]:
    """Invert removal regions into keep-segments.

    padding_ms: amount to shrink each removal on both sides, keeping
    a bit of context audio around each cut.
    """
    if not removal_regions:
        return [Segment(source_start=0.0, source_end=total_duration)]

    padding_sec = padding_ms / 1000.0

    # Sort and merge overlapping removals
    sorted_r = sorted(removal_regions, key=lambda r: r[0])
    merged: list[list[float]] = [list(sorted_r[0])]
    for start, end in sorted_r[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    # Apply padding: shrink each removal inward (keep more context)
    padded = []
    for start, end in merged:
        new_start = start + padding_sec
        new_end = end - padding_sec
        if new_end > new_start:
            padded.append((new_start, new_end))
        # Collapses to nothing → skip (cut removed entirely by padding)

    if not padded:
        return [Segment(source_start=0.0, source_end=total_duration)]

    # Build keep segments (gaps between padded removals)
    raw: list[Segment] = []

    if padded[0][0] > 0.001:
        raw.append(Segment(source_start=0.0, source_end=padded[0][0]))

    for i in range(len(padded) - 1):
        gap_start = padded[i][1]
        gap_end = padded[i + 1][0]
        if gap_end - gap_start > 0.001:
            raw.append(Segment(source_start=gap_start, source_end=gap_end))

    if padded[-1][1] < total_duration - 0.001:
        raw.append(Segment(source_start=padded[-1][1], source_end=total_duration))

    # Clamp to valid range and discard very short segments
    result = []
    for seg in raw:
        seg.source_start = max(0.0, seg.source_start)
        seg.source_end = min(total_duration, seg.source_end)
        if seg.duration >= 0.05:
            result.append(seg)

    return result


def map_time_to_output(t: float, keep_segs: list[Segment]) -> Optional[float]:
    """Map a source timestamp to its position in the output (cut) timeline.

    Returns None if t falls within a removed region.
    """
    output_t = 0.0
    for seg in keep_segs:
        if t <= seg.source_end:
            if t >= seg.source_start:
                return output_t + (t - seg.source_start)
            return None  # in a removed region before this segment
        output_t += seg.duration
    return None  # beyond all segments


def map_span_to_output(
    start: float, end: float, keep_segs: list[Segment]
) -> Optional[tuple[float, float]]:
    """Map a time span to output timeline, clipping at removal boundaries.

    If the span overlaps multiple kept regions (straddles a cut), the output
    span covers from first overlap start to last overlap end, which may
    include a gap (acceptable for title clips that straddle a cut).

    Returns (out_start, out_end) or None if no overlap with kept regions.
    """
    out_start = None
    out_end = None
    output_t = 0.0

    for seg in keep_segs:
        overlap_start = max(start, seg.source_start)
        overlap_end = min(end, seg.source_end)

        if overlap_end > overlap_start:
            seg_out_start = output_t + (overlap_start - seg.source_start)
            seg_out_end = output_t + (overlap_end - seg.source_start)
            if out_start is None:
                out_start = seg_out_start
            out_end = seg_out_end

        output_t += seg.duration

    if out_start is None or out_end is None or out_end <= out_start:
        return None
    return out_start, out_end


def filter_telop_entries(entries: list[dict]) -> list[dict]:
    """Return *entries* that have both ``start`` and ``end`` keys, sorted by start.

    Entries missing either key are silently skipped.  This is the single
    canonical place for this check — app.py, web.py, and fcpxml_telop all
    import and call this instead of inline-filtering with sentinel objects.
    """
    return sorted(
        (e for e in entries if "start" in e and "end" in e),
        key=lambda e: e["start"],
    )


# ── Telop segment splitting ───────────────────────────────────────────────────

# Sentence-ending punctuation that marks a natural subtitle break point.
_SENT_END_RE = re.compile(r"[。！？!?]")

# Conservative defaults matching VidTighten's default telop settings:
#   80px font on 3840px frame → max_em_per_line = 3840 × 0.42 / 80 ≈ 20.2 em
#   Two-line maximum → 40 em total.
_MAX_TELOP_DURATION: float = 6.0   # seconds — broadcast subtitle standard
_MAX_TELOP_EM: float = 40.0        # em units ≈ 2 lines at default settings

# Segment-merge thresholds for _merge_short_adjacent.
# Whisper sometimes puts a segment boundary mid-word (e.g. "フリー" | "ランス")
# or at a semantically empty cut (e.g. "２０２６" | "年に退職して").
# Merging adjacent segments whose inter-segment gap is small and whose combined
# text is still within display limits restores coherence before splitting.
_MERGE_MIN_DUR: float  = 1.0   # merge if either adjacent segment is shorter than this (s)
_MERGE_MAX_GAP: float  = 0.20  # only merge when the gap between them is ≤ 200 ms

# The sokuon (っ/ッ) can NEVER end a Japanese word — it is always followed by a
# consonant — so a segment ending with it was cut mid-word by Whisper
# (e.g. "そういっ" | "た") and the next segment should be merged in even when
# neither side is short.  The prolonged mark ー is deliberately NOT included: it
# legitimately ends many katakana words (フリー, コーヒー), so it is not a
# reliable mid-word signal.
_MIDWORD_TAIL_RE = re.compile(r"[っッ]$")
# Symmetrically, certain characters can NEVER begin a Japanese word: the sokuon,
# the small youon (ゃゅょ), small vowels (ぁぃぅぇぉ), the prolonged mark, and
# iteration marks.  A segment STARTING with one of these means Whisper cut a word
# mid-way (e.g. "そうい" | "った"), so the previous segment should absorb it.
_MIDWORD_HEAD_RE = re.compile(r"^[っッゃゅょゎャュョヮぁぃぅぇぉァィゥェォーゝゞヽヾ々]")
# Very small gap that, combined with a mid-word tail/head, confirms continuous speech.
_MIDWORD_MAX_GAP: float = 0.12


def _telop_char_em(ch: str) -> float:
    """Display width in em units — mirrors fcpxml_telop._char_em."""
    eaw = unicodedata.east_asian_width(ch)
    return 1.0 if eaw in ("W", "F", "A") else 0.55


def _text_em(text: str) -> float:
    return sum(_telop_char_em(c) for c in text if c != "\n")


def _words_for_seg(words: list[dict], seg_start: float, seg_end: float) -> list[dict]:
    """Words whose start falls within [seg_start−0.1, seg_end+0.1)."""
    return [w for w in words if seg_start - 0.1 <= w["start"] < seg_end + 0.1]


def _seg_id_set(seg: dict) -> set:
    """Return the set of Whisper segment ids this (possibly merged) segment spans.

    Merged segments carry a "seg_ids" list (see _merge_short_adjacent); unmerged
    segments carry a scalar "seg_id" stamped by the worker.  Returns an empty set
    when neither is present (e.g. older sessions or hand-built test segments), in
    which case callers fall back to time-window word selection.
    """
    ids = seg.get("seg_ids")
    if ids:
        return {i for i in ids if i is not None}
    sid = seg.get("seg_id")
    return {sid} if sid is not None else set()


def _is_cjk_token(s: str) -> bool:
    """Return True if s contains any CJK Symbols/Punctuation, Hiragana, Katakana,
    CJK Unified Ideographs, or CJK Compatibility Ideographs (U+3000-U+9FFF, U+F900-U+FAFF).

    Starting at U+3000 ensures Japanese punctuation tokens (。、「」 at U+3001-U+303F)
    also trigger the no-space path, matching the _hasCJK rule in transcript.js.
    """
    return any(0x3000 <= ord(c) <= 0x9FFF or 0xF900 <= ord(c) <= 0xFAFF for c in s)


def _halfwidth(s: str) -> str:
    """Convert fullwidth Latin letters and digits to ASCII halfwidth equivalents.

    Whisper sometimes transcribes company names and numbers in Japanese audio using
    fullwidth forms (Ａ-Ｚ U+FF21-FF3A, ａ-ｚ U+FF41-FF5A, ０-９ U+FF10-FF19).
    Professional subtitle convention is halfwidth ASCII; this normalises them.
    Non-alphanumeric fullwidth characters (！、etc.) are left unchanged.
    """
    out: list[str] = []
    for c in s:
        cp = ord(c)
        if 0xFF10 <= cp <= 0xFF19 or 0xFF21 <= cp <= 0xFF3A or 0xFF41 <= cp <= 0xFF5A:
            out.append(chr(cp - 0xFEE0))
        else:
            out.append(c)
    return "".join(out)


# Max inter-token gap below which two Latin/digit fragments are treated as one
# word.  Whisper sometimes splits a single word into chunks ("Goo"+"gle" → Google)
# with no spoken pause between them; a genuine word boundary ("Claude" "Code")
# carries the natural inter-word pause, which is well above this.
_LATIN_CONTIGUOUS_GAP: float = 0.04


def _needs_space(prev_text: str, cur_text: str, gap: Optional[float]) -> bool:
    """Whether a space goes between two reconstructed tokens.

    A space is inserted only between two non-CJK (Latin/digit) tokens, EXCEPT when
    they are fragments of a single word: single-char ASCII runs ('G','o','o'…), a
    numeric suffix ('10'+'%'), or two Latin/digit chunks spoken with no pause
    between them (gap < _LATIN_CONTIGUOUS_GAP — e.g. 'Goo'+'gle' → 'Google').
    """
    if _is_cjk_token(prev_text) or _is_cjk_token(cur_text):
        return False
    is_alnum_run = (
        len(prev_text) == 1 and prev_text.isascii() and prev_text.isalnum()
        and len(cur_text) == 1 and cur_text.isascii() and cur_text.isalnum()
    )
    is_numeric_suffix = prev_text[-1:].isdigit() and cur_text in ("%", "°", "‰")
    is_contiguous = (
        gap is not None and gap < _LATIN_CONTIGUOUS_GAP
        and prev_text[-1:].isascii() and prev_text[-1:].isalnum()
        and cur_text[:1].isascii() and cur_text[:1].isalnum()
    )
    return not (is_alnum_run or is_numeric_suffix or is_contiguous)


def _join_word_texts(words: list[dict]) -> str:
    """Reconstruct display text from a word slice.

    faster-whisper Japanese: character-level tokens → concatenate directly.
    faster-whisper English: stripped word tokens → space-join.
    Mixed Japanese/English: space between adjacent Latin tokens only, unless they
    are time-contiguous fragments of one word (see _needs_space).

    Accepts both 'word' key (raw faster-whisper / openai-whisper) and 'text'
    key (group_word_tokens output) for defensive interoperability.
    """
    toks = [
        (_halfwidth((w.get("word") or w.get("text") or "").strip()),
         w.get("start"), w.get("end"))
        for w in words
    ]
    toks = [t for t in toks if t[0]]   # drop empty tokens
    if not toks:
        return ""
    buf: list[str] = [toks[0][0]]
    prev_text, prev_end = toks[0][0], toks[0][2]
    for cur_text, cur_start, cur_end in toks[1:]:
        gap = (cur_start - prev_end) if (cur_start is not None and prev_end is not None) else None
        if _needs_space(prev_text, cur_text, gap):
            buf.append(" ")
        buf.append(cur_text)
        prev_text, prev_end = cur_text, cur_end
    return "".join(buf)


def _word_char_offsets(words: list[dict]) -> list[int]:
    """Char offset of the start of each word in the text _join_word_texts produces.

    offs[i] is the index, in the joined string, of the first character of word i —
    i.e., the cut position for a split that places words[i:] on a new card.  Mirrors
    the spacing rules of _join_word_texts so the offsets line up exactly with the
    text passed to morphological analysis.
    """
    offs = [0] * len(words)
    buf_len = 0
    prev_part: str | None = None
    prev_end: Optional[float] = None
    for i, w in enumerate(words):
        part = _halfwidth((w.get("word") or w.get("text") or "").strip())
        if not part:
            offs[i] = buf_len               # empty token: zero width
            continue
        if prev_part is not None:
            cur_start = w.get("start")
            gap = (cur_start - prev_end) if (cur_start is not None and prev_end is not None) else None
            if _needs_space(prev_part, part, gap):
                buf_len += 1                 # separating space
        offs[i] = buf_len
        buf_len += len(part)
        prev_part = part
        prev_end = w.get("end")
    return offs


def group_word_tokens(words: list[dict], max_gap_s: float = 0.05) -> list[dict]:
    """Merge contiguous single-CJK-char tokens into single word objects.

    faster-whisper emits Japanese at character level (one token per kana/kanji).
    This groups them into natural words separated by silence gaps, Whisper segment
    boundaries, or non-CJK tokens.

    max_gap_s: max inter-token silence before a new group starts (default 50 ms).
    Intra-word gaps in conversational Japanese are typically < 30 ms; inter-phrase
    gaps are usually ≥ 50 ms, so 50 ms is a natural split point.  Additionally, if
    two adjacent tokens belong to different Whisper segments (different seg_id), they
    are always split regardless of the time gap — Whisper segment boundaries are
    reliable phrase delimiters.

    Non-CJK tokens (English, numbers, etc.) are passed through unchanged.
    """
    if not words:
        return []

    def _is_single_cjk(w: dict) -> bool:
        t = (w.get("word") or w.get("text") or "").strip()
        if len(t) != 1:
            return False
        cp = ord(t)
        # Match CJK Unified Ideographs (kanji) and katakana, but NOT hiragana.
        #
        # All three scripts share east_asian_width="W", so the old check
        # (east_asian_width in {"W","F","A"}) incorrectly merged hiragana with
        # kanji.  This caused entire phrases like "もう一つ余談なんですけれども一回私"
        # to collapse into one span — clicking any character in the span sought to
        # the group start (time of "も"), not to the clicked word.
        #
        # Hiragana (U+3040–U+309F) deliberately excluded: hiragana tokens act as
        # natural phrase separators in Japanese and must NOT merge with adjacent kanji.
        # Katakana (U+30A0–U+30FF) kept: loanwords like コーヒー are emitted char-by-char
        # and need to be grouped just as kanji do.
        return (
            0x4E00 <= cp <= 0x9FFF    # CJK Unified Ideographs (main block, kanji)
            or 0x3400 <= cp <= 0x4DBF   # CJK Extension A
            or 0xF900 <= cp <= 0xFAFF   # CJK Compatibility Ideographs
            or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B (supplementary)
            or 0x30A0 <= cp <= 0x30FF   # Katakana (includes prolonged mark ー U+30FC)
        )

    def _is_single_alnum_char(w: dict) -> bool:
        """Return True if token is a single ASCII or fullwidth alphanumeric character.

        Matches both halfwidth ASCII (A-Z a-z 0-9) and their fullwidth equivalents
        (Ａ-Ｚ ａ-ｚ ０-９, U+FF10-FF5A) so that faster-whisper's character-level
        tokenisation of company names is handled regardless of which form Whisper
        chose to emit.
        """
        t = (w.get("word") or w.get("text") or "").strip()
        if len(t) != 1:
            return False
        cp = ord(t)
        return (
            (t.isascii() and t.isalnum())   # halfwidth ASCII letter or digit
            or 0xFF10 <= cp <= 0xFF19        # fullwidth digit ０-９
            or 0xFF21 <= cp <= 0xFF3A        # fullwidth uppercase Ａ-Ｚ
            or 0xFF41 <= cp <= 0xFF5A        # fullwidth lowercase ａ-ｚ
        )

    def _merge_group(group: list[dict]) -> dict:
        """Build a single merged token dict from a list of consecutive tokens."""
        confs = [
            w.get("confidence") if w.get("confidence") is not None else w.get("score")
            for w in group
        ]
        confs = [c for c in confs if c is not None]
        return {
            "id":         group[0].get("id", ""),   # caller re-assigns IDs
            "start":      group[0]["start"],
            "end":        group[-1]["end"],
            "text":       "".join(
                _halfwidth((w.get("word") or w.get("text") or "").strip()) for w in group
            ),
            "seg_id":     group[0].get("seg_id"),
            "confidence": min(confs) if confs else None,
        }

    result: list[dict] = []
    i = 0
    while i < len(words):
        if _is_single_cjk(words[i]):
            # Start a CJK group — greedily consume contiguous single-CJK tokens
            group = [words[i]]
            j = i + 1
            while j < len(words) and _is_single_cjk(words[j]):
                # Hard break on Whisper segment boundary: seg_id values differ when two
                # tokens belong to different transcription segments (= phrase boundaries).
                seg_cur  = words[j].get("seg_id")
                seg_prev = words[j - 1].get("seg_id")
                if seg_cur is not None and seg_prev is not None and seg_cur != seg_prev:
                    break
                if words[j]["start"] - words[j - 1]["end"] > max_gap_s:
                    break
                group.append(words[j])
                j += 1
            result.append(_merge_group(group))
            i = j
        elif _is_single_alnum_char(words[i]):
            # Start a Latin/digit group (ASCII or fullwidth) — greedily consume
            # contiguous single-alnum tokens.  Fullwidth chars are normalised to
            # halfwidth ASCII in the merged text (Ｇｏｏｇｌｅ → Google).
            group = [words[i]]
            j = i + 1
            while j < len(words) and _is_single_alnum_char(words[j]):
                seg_cur  = words[j].get("seg_id")
                seg_prev = words[j - 1].get("seg_id")
                if seg_cur is not None and seg_prev is not None and seg_cur != seg_prev:
                    break
                if words[j]["start"] - words[j - 1]["end"] > max_gap_s:
                    break
                group.append(words[j])
                j += 1
            result.append(_merge_group(group))
            i = j
        else:
            result.append(words[i])
            i += 1
    return result


def _make_chunk(
    chunk_words: list[dict],
    seg_start: float,
    seg_end: float,
    is_first: bool,
    is_last: bool,
) -> Optional[dict]:
    """Build a telop entry dict from a word slice; returns None if empty."""
    if not chunk_words:
        return None
    text = _join_word_texts(chunk_words)
    if not text:
        return None
    start = seg_start if is_first else chunk_words[0]["start"]
    end = seg_end if is_last else chunk_words[-1]["end"]
    if end <= start:
        return None
    return {"start": start, "end": end, "text": text}


def _build_chunks(
    seg: dict,
    words: list[dict],
    split_indices: list[int],
    max_dur: float,
    max_em: float,
    depth: int = 0,
) -> list[dict]:
    """Emit sub-segments at the given word-list split boundaries.

    A chunk that is STILL too long (e.g. a long sentence with no internal
    punctuation) is re-split via _phrase_split — at a natural 文節 boundary, NOT a
    blind midpoint — so a word like "Google" is never cut.  Recursion depth is
    bounded to avoid infinite loops on degenerate input.
    """
    boundaries = [0] + sorted(set(split_indices)) + [len(words)]
    n = len(boundaries) - 1
    chunks: list[dict] = []

    for k in range(n):
        chunk_words = words[boundaries[k] : boundaries[k + 1]]
        sub = _make_chunk(
            chunk_words,
            seg["start"], seg["end"],
            is_first=(k == 0), is_last=(k == n - 1),
        )
        if sub is None:
            continue

        sub_dur = sub["end"] - sub["start"]
        if (sub_dur > max_dur * 1.5 or _text_em(sub["text"]) > max_em * 1.5) \
                and len(chunk_words) > 2 and depth < 3:
            chunks.extend(_phrase_split(sub, chunk_words, max_dur, max_em, depth + 1))
        else:
            chunks.append(sub)

    return chunks if chunks else [seg]


def _phrase_split(
    seg: dict,
    words: list[dict],
    max_dur: float,
    max_em: float,
    depth: int = 0,
) -> list[dict]:
    """Split an over-long segment by even time-division, snapped to natural phrase
    (文節) boundaries identified by morphological analysis.

    Among candidate boundaries the one nearest each even-time target is chosen,
    pulled toward the more natural boundary (a phrase ending in a particle beats one
    ending in a dangling adverb).  A word is never cut: only real 文節 boundaries are
    candidates when fugashi is available.  Falls back to a character heuristic
    (avoid splitting after a sokuon / before a small kana) when fugashi is absent.
    """
    dur = seg["end"] - seg["start"]
    if len(words) < 2:
        return [seg]
    n = max(2, int(dur / max_dur) + 1)

    text = seg.get("text", "")
    bscores = japanese.break_scores(text) if japanese.available() else None
    fugashi_cands: list[int] = []
    offs: list[int] = []
    if bscores:
        offs = _word_char_offsets(words)
        fugashi_cands = [i for i in range(1, len(words)) if offs[i] in bscores]

    def _ends_midword(idx: int) -> bool:
        prev_tok = words[idx - 1].get("word") or words[idx - 1].get("text") or ""
        next_tok = words[idx].get("word") or words[idx].get("text") or ""
        # Mid-Latin-word: two time-contiguous alnum fragments are one word (G|oogle).
        if (prev_tok[-1:].isascii() and prev_tok[-1:].isalnum()
                and next_tok[:1].isascii() and next_tok[:1].isalnum()):
            gap = words[idx]["start"] - words[idx - 1]["end"]
            if gap < _LATIN_CONTIGUOUS_GAP:
                return True
        return bool(_MIDWORD_TAIL_RE.search(prev_tok)) or bool(_MIDWORD_HEAD_RE.search(next_tok))

    if fugashi_cands:
        candidates = fugashi_cands
        def use_key(i, target):
            # Each naturalness point is worth ~0.6s of time-proximity, so a strong
            # boundary (particle, score 2.5+) wins over a closer weak one.
            return abs(words[i]["start"] - target) - 0.6 * bscores[offs[i]]
    else:
        candidates = [i for i in range(1, len(words)) if not _ends_midword(i)]
        if not candidates:
            candidates = list(range(1, len(words)))   # all mid-word; give up avoiding
        def use_key(i, target):
            return abs(words[i]["start"] - target)

    time_splits: list[int] = []
    for k in range(1, n):
        target = seg["start"] + k * (dur / n)
        best_i = min(candidates, key=lambda i: use_key(i, target))
        if best_i not in time_splits:
            time_splits.append(best_i)

    return _build_chunks(seg, words, time_splits, max_dur, max_em, depth) if time_splits else [seg]


def _split_one(
    seg: dict,
    all_words: list[dict],
    max_dur: float,
    max_em: float,
) -> list[dict]:
    dur = seg["end"] - seg["start"]

    # Fetch words first: needed for both text reconstruction and split logic.
    words = _words_for_seg(all_words, seg["start"], seg["end"])

    # Prefer exact seg_id membership over the time window.  A word's seg_id is the
    # only reliable indicator of which Whisper segment it belongs to: at a segment
    # boundary the previous segment's rounded end time and the next segment's first
    # word's rounded start time can collide to within 1 ms, so the ±100 ms window
    # pulls the neighbour's leading words into this segment — duplicating them
    # across both cards (e.g. "…みたいな" | "いな比較的…").  Filtering by seg_id
    # gives each word to exactly one segment.  Falls back to the full window when
    # seg_id is unavailable (old sessions / hand-built segments).
    seg_ids = _seg_id_set(seg)
    if seg_ids:
        seg_words = [w for w in words if w.get("seg_id") in seg_ids]
        if seg_words:
            words = seg_words

    # Prefer word-level text over raw decoder text.
    #
    # Whisper's decoder and its word aligner can disagree at segment
    # boundaries.  The canonical example: decoder emits "フリー" at the end
    # of one segment while word alignment places the complete token
    # "フリーランス" inside that same segment.  Reconstructing from words
    # corrects the cut-off and eliminates phantom fragments ("ランス") at the
    # start of the following segment.
    #
    # Use a tight 20ms leading window for reconstruction (vs 100ms for split
    # detection) to exclude words that bleed in from the previous segment via
    # the tolerance window — e.g. a trailing "。" whose start time sits just
    # before this segment's start.  (Redundant once seg_id filtering applies,
    # but kept for the fallback path.)
    if words:
        recon_words = [w for w in words if w["start"] >= seg["start"] - 0.02]
        if recon_words:
            reconstructed = _join_word_texts(recon_words)
            if reconstructed:
                seg = {**seg, "text": reconstructed}

    text = seg.get("text", "").strip()

    if dur <= max_dur and _text_em(text) <= max_em:
        return [seg]

    if not words:
        return [seg]   # no word timestamps — can't split

    # ── Strategy 1: split at sentence-ending punctuation ─────────────────────
    # Find word indices AFTER which a sentence boundary occurs.
    sent_splits = [
        i + 1
        for i, w in enumerate(words[:-1])   # never split after the last word
        if _SENT_END_RE.search(w.get("word") or w.get("text") or "")
    ]
    if sent_splits:
        return _build_chunks(seg, words, sent_splits, max_dur, max_em)

    # ── Strategy 2: even time-division, snapped to natural phrase boundaries ──
    return _phrase_split(seg, words, max_dur, max_em)


def _merge_short_adjacent(
    segments: list[dict],
    max_dur: float,
    max_em: float,
    all_words: Optional[list[dict]] = None,
) -> list[dict]:
    """Merge adjacent segments when one is too short to stand alone.

    Whisper's decoder occasionally places a segment boundary mid-word
    (e.g. "フリー" | "ランスのコンサルタント" from "フリーランス") or at a
    semantically empty cut ("２０２６" | "年に退職して").  These micro-segments
    produce nonsensical telop cards.

    Two merge rules, both requiring a small inter-segment gap and no
    sentence-ending punctuation (。！？!?) on the previous segment:

    1. Size rule — at least one segment is shorter than _MERGE_MIN_DUR (1 s),
       the gap is ≤ _MERGE_MAX_GAP (200 ms), AND the combined text still fits
       one card (max_dur, max_em).  Restores short artefacts like "フリー" |
       "ランス" → "フリーランス".

    2. Mid-word-tail rule — the previous segment ends with a sokuon (っ/ッ) or a
       trailing prolonged mark (ー), which can NEVER end a Japanese word, and the
       gap is ≤ _MIDWORD_MAX_GAP (120 ms).  This is an unambiguous mid-word cut
       (e.g. "そういっ" | "た"), so the segments are merged even when both are
       long and the result overflows one card — _split_one re-splits it cleanly
       afterwards, avoiding a split right after the sokuon.

    Merging is left-to-right; the merged segment inherits the previous start and
    the current end, and accumulates both sides' seg_ids so downstream
    reconstruction can select the right words.  Text is concatenated without a
    separator so CJK compound words are restored correctly.
    """
    if len(segments) <= 1:
        return segments

    def _init(seg: dict) -> dict:
        out = dict(seg)
        out["seg_ids"] = sorted(_seg_id_set(seg))
        return out

    result = [_init(segments[0])]
    for seg in segments[1:]:
        prev = result[-1]
        prev_dur = prev["end"] - prev["start"]
        this_dur = seg["end"] - seg["start"]
        gap      = seg["start"] - prev["end"]

        # Japanese: concatenate directly; Latin: add space only if neither
        # side is CJK (same rule as _join_word_texts).
        prev_text = (prev.get("text") or "").rstrip()
        this_text = (seg.get("text")  or "").lstrip()
        if not _is_cjk_token(prev_text[-1:]) and not _is_cjk_token(this_text[:1]):
            combined_text = prev_text + " " + this_text
        else:
            combined_text = prev_text + this_text

        no_sentence_end = not _SENT_END_RE.search(prev_text)
        within_size = (
            (seg["end"] - prev["start"]) <= max_dur
            and _text_em(combined_text) <= max_em
        )
        size_merge = (
            (prev_dur < _MERGE_MIN_DUR or this_dur < _MERGE_MIN_DUR)
            and gap <= _MERGE_MAX_GAP
            and no_sentence_end
            and within_size
        )
        # Mid-word / mid-phrase cut: Whisper split a word or phrase across this
        # boundary.  Primary signal is morphological (fugashi): is_continuation() is
        # True when a morpheme straddles the boundary, the next segment opens with a
        # dependent word, or the previous ends with a dangling modifier
        # (e.g. "…そう" | "いった").  Falls back to the sokuon/small-kana character
        # heuristic when fugashi is unavailable.
        #
        # When the combined card would overflow, merging is only safe if there are
        # ≥2 words in the span for _split_one to re-split on; otherwise we'd emit an
        # oversized, unsplittable card — so in that case we don't bypass the size cap.
        span_words = 0
        if all_words is not None and not within_size:
            span_words = sum(
                1 for w in all_words
                if prev["start"] - 0.1 <= w["start"] < seg["end"] + 0.1
            )
        # A morpheme physically straddling the boundary is an unambiguous mid-word
        # cut (e.g. "元Goo" | "gleの" → "Google") — merge regardless of the gap,
        # since a word cannot contain a real pause (Whisper's English word-timestamps
        # are often coarsely quantised).  Weaker signals (dangling modifier, sokuon,
        # small-kana head) still require a small gap to confirm continuous speech.
        straddles = japanese.crosses_morpheme(prev_text, this_text)
        soft_signal = (
            japanese.is_continuation(prev_text, this_text)
            or bool(_MIDWORD_TAIL_RE.search(prev_text))
            or bool(_MIDWORD_HEAD_RE.search(this_text))
        )
        midword_merge = (
            no_sentence_end
            and (straddles or (gap <= _MIDWORD_MAX_GAP and soft_signal))
            and (within_size or span_words >= 2)
        )

        if size_merge or midword_merge:
            merged_ids = sorted(set(prev.get("seg_ids", [])) | _seg_id_set(seg))
            result[-1] = {
                **prev, "end": seg["end"], "text": combined_text,
                "seg_ids": merged_ids,
            }
        else:
            result.append(_init(seg))

    return result


def split_telop_segments(
    segments: list[dict],
    words: list[dict],
    max_duration: float = _MAX_TELOP_DURATION,
    max_em: float = _MAX_TELOP_EM,
) -> list[dict]:
    """Split Whisper segments that are too long for a single subtitle card.

    Whisper's decoder groups audio into variable-length segments that can span
    15+ seconds and 3–4 display lines.  This function breaks them into
    subtitle-sized chunks (≤ max_duration seconds, ≤ max_em em-units wide).

    Processing order:
    1. Merge adjacent micro-segments (Whisper artefacts, e.g. mid-word cuts).
    2. Split remaining over-length segments at punctuation or time boundaries.

    Split priority (step 2):
    1. Sentence-ending punctuation (。！？ .!?) found in the word sequence.
    2. Even time-divisions at the nearest word boundary.

    Args:
        segments: Raw Whisper segment list [{start, end, text}].
        words:    Whisper word list [{word, start, end}], same order as returned
                  by the transcription worker.
        max_duration: Maximum seconds per output segment (default 6 s).
        max_em:       Maximum em-units of text per segment — approximately two
                      display lines at default settings (default 40 em).

    Returns:
        Flat, in-order list of segment dicts with correct start/end timestamps.
    """
    merged = _merge_short_adjacent(segments, max_duration, max_em, words)
    result: list[dict] = []
    for seg in merged:
        result.extend(_split_one(seg, words, max_duration, max_em))
    return result


def assign_words_to_entries(
    grouped_words: list[dict],
    telop_entries: list[dict],
    tolerance: float = 0.15,
) -> list[str | None]:
    """Return the telop entry ID (or None) for each grouped word.

    Each word is matched to the entry whose start time is closest to the
    word's start time, within a ±tolerance-second window.  The
    "closest-start" strategy correctly handles words that fall at the
    exact boundary between two adjacent entries — a simple "first match"
    would assign such words to the earlier entry even when they are
    semantically the first word of the later one.

    Args:
        grouped_words:  Output of group_word_tokens(); each dict has at
                        least "start", "end", and "text"/"word" keys.
        telop_entries:  Ordered list of telop entry dicts with "id",
                        "start", "end".
        tolerance:      Half-width of the matching window in seconds.
                        Handles Whisper timestamp jitter (typically ≤280 ms
                        but the default 150 ms is the practical match budget).

    Returns:
        List of the same length as grouped_words; each element is the
        matching entry id (str) or None if no entry window contains the word.
    """
    entry_ranges = [(e["start"], e["end"], e["id"]) for e in telop_entries]
    result: list[str | None] = []
    for w in grouped_words:
        ws = float(w.get("start", 0))
        best_id: str | None = None
        best_dist: float = float("inf")
        for es, ee, eid in entry_ranges:
            if es - tolerance > ws:
                break   # entries are time-ordered; no later entry can match
            if ws < ee + tolerance:
                dist = abs(ws - es)
                if dist < best_dist:
                    best_dist = dist
                    best_id = eid
        # Fallback: if no entry matched within tolerance, assign to the nearest entry
        # by start time (unconstrained).  This ensures every word gets a seg_id so the
        # frontend can always seek to the word's timestamp rather than losing the word.
        if best_id is None and entry_ranges:
            best_id = min(entry_ranges, key=lambda t: abs(ws - t[0]))[2]
        result.append(best_id)
    return result
