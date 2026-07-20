"""Telop FCPXML generator — produces title clips time-adjusted to the cut timeline.

Ports the Telop Bridge JS logic to Python. Title positions are mapped from
source timestamps to output (post-cut) timeline positions so the telop FCPXML
aligns with the rough-cut sequence when both are imported into FCP.
"""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Optional

from preprod import japanese
from preprod.corrections import correct_text
from preprod.fcpxml_common import FCP_FORMAT_PREFIX
from preprod.segments import Segment, map_span_to_output, filter_telop_entries


# Punctuation to replace with a space (separates sentence elements in both JP and EN).
# "." is handled separately (see _clean_telop_text) — a bare period in this class would
# also strip decimal points in version numbers/model names ("GLM5.2" -> "GLM5 2").
_SEPARATORS = re.compile(r'[。、！？・…〜～：；!?,;:\-—–\n]')
# A "." only counts as sentence-ending punctuation when it isn't touching a digit on
# either side — preserves decimal points ("5.2", "V4.0") while still splitting English
# sentences ("done. Next" -> "done  Next").
_PERIOD = re.compile(r'(?<!\d)\.(?!\d)')
# Punctuation to remove entirely (wrappers — no space replacement needed)
_WRAPPERS    = re.compile('[「」『』【】〈〉《》（）()"\'"“”‘’]')


def _clean_telop_text(text: str) -> str:
    """Strip punctuation from a telop title card — separator punctuation becomes
    a space so elements remain distinct; wrapper punctuation is removed outright.
    Collapsing runs of whitespace and stripping is the final step.

    Custom-vocabulary corrections (e.g. brand name 空気デザイン → クウキデザイン) are
    applied here too — at the export chokepoint — so the FCPXML is always correct
    regardless of how the entry text was built upstream (frontend rebuild from raw
    words, or an analysis run that predates the correction).
    """
    text = correct_text(text)
    text = _SEPARATORS.sub(" ", text)
    text = _PERIOD.sub(" ", text)
    text = _WRAPPERS.sub("", text)
    return re.sub(r" {2,}", " ", text).strip()


def _char_em(ch: str) -> float:
    """Display width of a character in em units.

    East-Asian Wide / Fullwidth / Ambiguous → 1.0 (full-width CJK and kana).
    Everything else (Latin, digits, spaces) → 0.55.
    """
    eaw = unicodedata.east_asian_width(ch)
    return 1.0 if eaw in ("W", "F", "A") else 0.55


def _em_width(text: str) -> float:
    """Total display width of *text* in em units, ignoring any existing newlines."""
    return sum(_char_em(ch) for ch in text if ch != "\n")


# Two-character sequences that must never be split across a line break: compound
# particles (とか "and such", とは, には, では, から, まで, ては, のに, ので) and
# the quotative って.  Breaking after the first kana orphans the second (e.g.
# "丸と" / "か") and reads as a different word.
_INSEPARABLE_PAIRS: frozenset[str] = frozenset({
    "とか", "とは", "には", "では", "から", "まで", "ては", "のに", "ので",
    "って", "けど", "でも", "ても", "なら",
})

# Small kana that bind to the preceding character to form one mora — breaking
# immediately after one splits a syllable (e.g. "オーディ" | "オ" from オーディオ,
# where ィ is the small i of ディ).  A trailing ー (prolonged mark) on line 2's
# first character is the same problem from the other side.
_SMALL_KANA = "っッぁぃぅぇぉゃゅょゎァィゥェォャュョヮ"


def _breaks_a_number(prev: str, cur: str) -> bool:
    """True if breaking between prev/cur would split a numeric token
    ("5.5" -> "5" / ".5", or "123" -> "12" / "3").

    By the time _wrap_telop runs, _clean_telop_text has already converted every
    non-numeric "." to a space (sentence-ending punctuation) — so any "." still
    in the text at this point is guaranteed to be a genuine decimal point, and a
    plain adjacency check (no wider number-span scan needed) is sufficient.
    """
    return (prev.isdigit() or prev == ".") and (cur.isdigit() or cur == ".")


def _break_score(text: str, pos: int) -> float:
    """Semantic quality of breaking *text* so that line 2 starts at *pos*.

    Examines the character immediately before the proposed break point (text[pos-1]).
    Returns 0.0 for positions with no linguistic signal and a negative penalty for
    positions that would split a word.

    Penalty (−10) – break would land after a sokuon/small kana, before a prolonged
                    mark ー, inside a 2-char inseparable particle, or inside a
                    numeric token ("5.5", "123") — never break.
    Tier 3 (3.0) – space (converted from punctuation): strongest natural pause.
    Tier 2 (2.0) – primary JP phrase-ending particles: は が を に と
    Tier 1 (1.0) – secondary particles: で も か の へ
    Tier 0.7     – te-form clause connector: て (weaker; usually mid-clause)
    """
    if pos <= 0 or pos >= len(text):
        return 0.0
    prev = text[pos - 1]
    cur  = text[pos]
    # Never split a mora, compound particle, or number:
    # - a sokuon (っ/ッ) cannot end line 1 (it never ends a word);
    # - a small kana or ー cannot start line 2 (each binds to the preceding char,
    #   so a break before one splits the syllable, e.g. オーデ|ィオ, コーヒ|ー);
    # - a 2-char inseparable particle (とか, から, って…) must stay intact;
    # - a numeric token ("5.5", "123") must stay on one line — see _breaks_a_number.
    # A non-sokuon small kana ENDING line 1 is fine — its mora is complete there.
    if (prev in "っッ" or cur in _SMALL_KANA or cur == "ー"
            or (prev + cur) in _INSEPARABLE_PAIRS or _breaks_a_number(prev, cur)):
        return -10.0
    if prev == " ":
        return 3.0          # former punctuation → clear breath mark
    if prev in "はがをにと":
        return 2.0          # topic / subject / object / direction markers
    if prev in "でもかのへ":
        return 1.0          # secondary particles
    if prev == "て":
        return 0.7          # te-form clause connector
    return 0.0


def _wrap_telop(text: str, max_em: float) -> str:
    """Insert at most one newline so each line fits within *max_em* em units.

    Scoring per candidate position:
        combined = semantic_score  −  (visual_distance_from_midpoint / half_total_em)

    A particle/space DOMINATES over the raw midpoint unless it is very far from
    centre (dist > semantic_score × half_total_em).  This means:
    - A primary particle (score 2) beats the pure midpoint up to 2× the half-width
      away — i.e., the particle would have to be at a line edge to lose.
    - The pure midpoint (score 0) wins only when no linguistic signal exists.

    After splitting, leading/trailing spaces are stripped from both lines.
    Line 2 is hard-truncated if it still exceeds *max_em*.
    """
    total_em = _em_width(text)
    if total_em <= max_em:
        return text

    half = total_em / 2.0
    best_pos = max(1, len(text) // 2)
    best_combined = float("-inf")

    # Prefer 文節 (phrase) boundaries from morphological analysis — the line break
    # then lands only where a 、 would be grammatically appropriate.  When fugashi
    # is unavailable, fall back to the character-level particle heuristic.
    bscores = japanese.break_scores(text) if japanese.available() else None
    if bscores:
        # Spaces are former punctuation (、/。 converted by _clean_telop_text) or
        # English word boundaries — always strong, valid break points, so add them
        # to the phrase boundaries fugashi found.
        scores = dict(bscores)
        for i in range(1, len(text)):
            if text[i - 1] == " ":
                scores[i] = max(scores.get(i, 0.0), 3.0)
            # Override whatever fugashi scored here — a numeric token ("5.5") is
            # never a valid morpheme boundary, and fugashi's suujoshi handling of
            # a mixed digit-period token like this isn't reliable enough to trust.
            if _breaks_a_number(text[i - 1], text[i]):
                scores[i] = -10.0
        positions = sorted(scores)
        score_at = lambda pos: scores[pos]
    elif bscores is None:
        # No fugashi at all — use the character grammar heuristic (handles hiragana
        # particle/te-form boundaries).
        positions = range(1, len(text))
        score_at = lambda pos: _break_score(text, pos)
    else:
        # fugashi is available but found no phrase boundary — a single long compound
        # (usually katakana, e.g. オーディオビジュアライザー).  Grammatical particle
        # scoring would misfire here (the で of です looks like the particle で), so
        # just split nearest the midpoint while never breaking a mora (small kana /
        # leading ー) or a compound particle.
        positions = range(1, len(text))

        def score_at(pos):
            prev, cur = text[pos - 1], text[pos]
            # A small kana or ー binds to the preceding character, so line 2 must
            # never start with one (that would split the mora, e.g. オーデ|ィオ); a
            # sokuon cannot end line 1 either. A numeric token ("5.5") must also
            # stay whole — see _breaks_a_number.
            if (prev in "っッ" or cur in _SMALL_KANA or cur == "ー"
                    or (prev + cur) in _INSEPARABLE_PAIRS or _breaks_a_number(prev, cur)):
                return -10.0
            return 0.0

    for pos in positions:                        # never split after last character
        dist = abs(_em_width(text[:pos]) - half)
        combined = score_at(pos) - dist / half
        if combined > best_combined:
            best_combined = combined
            best_pos = pos

    line1 = text[:best_pos].rstrip()
    line2 = text[best_pos:].lstrip()

    # Hard-truncate line 2 if it still overflows
    if _em_width(line2) > max_em:
        c2 = 0.0
        trunc = len(line2)
        for j, ch in enumerate(line2):
            c2 += _char_em(ch)
            if c2 > max_em:
                trunc = j
                break
        line2 = line2[:trunc]

    return f"{line1}\n{line2}"


# Known rational frame durations for common frame rates
_FRAME_DURATIONS: dict[str, Fraction] = {
    "24":     Fraction(100, 2400),
    "23.976": Fraction(1001, 24000),
    "25":     Fraction(100, 2500),
    "29.97":  Fraction(1001, 30000),
    "30":     Fraction(100, 3000),
    "50":     Fraction(100, 5000),
    "59.94":  Fraction(1001, 60000),
    "60":     Fraction(100, 6000),
}


def _frame_dur(fps_str: str) -> Fraction:
    return _FRAME_DURATIONS.get(fps_str, Fraction(100, 2400))


def _to_rt(seconds: float, fd: Fraction) -> str:
    """Seconds → rational time string snapped to frame boundary.

    Multiplying an int by a Fraction uses Python's Fraction arithmetic which
    auto-reduces the result, so "24 × 1/24 = 1/1" is written as "1s" not
    "24/24s", consistent with fcpxml_cut._to_rt.
    """
    frame_count = round(seconds / fd)
    rt = frame_count * fd   # Fraction auto-reduces (e.g. 24 × 1/24 = 1/1)
    if rt.denominator == 1:
        return f"{rt.numerator}s"
    return f"{rt.numerator}/{rt.denominator}s"


# FCP's internal format registry uses 4-digit strings for fractional rates.
# "23.976".replace(".", "") = "23976" which doesn't exist in FCP — it must be "2398".
# "29.97" → "2997" and "59.94" → "5994" happen to be correct via the replace trick,
# but list them explicitly here for clarity and future safety.
_FPS_ID: dict[str, str] = {
    "23.976": "2398",
    "29.97":  "2997",
    "59.94":  "5994",
}


def _format_name(width: int, height: int, fps_str: str) -> str:
    fps_id = _FPS_ID.get(fps_str, fps_str.replace(".", ""))
    # FCP_FORMAT_PREFIX imported from fcpxml_common (shared with fcpxml_cut).
    # Fall back to height-only for non-standard resolutions (e.g. 2560×1440).
    prefix = FCP_FORMAT_PREFIX.get((width, height), f"FFVideoFormat{height}p")
    return f"{prefix}{fps_id}"


def _hex_to_fcpxml_color(hex_color: str) -> str:
    """Convert '#RRGGBB' (or '#RGB' shorthand) to 'R G B 1' in 0–1 float range.

    Raises ``ValueError`` with a descriptive message for anything that isn't a
    valid 3- or 6-digit hex color string (with or without leading '#').
    """
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2   # expand shorthand: "F80" → "FF8800"
    if len(h) != 6:
        raise ValueError(
            f"Invalid hex color {hex_color!r}: expected '#RRGGBB' or '#RGB', "
            f"got {len(hex_color.lstrip('#'))}-digit value"
        )
    try:
        r = int(h[0:2], 16) / 255
        g = int(h[2:4], 16) / 255
        b = int(h[4:6], 16) / 255
    except ValueError:
        raise ValueError(
            f"Invalid hex color {hex_color!r}: contains non-hexadecimal characters"
        )
    return f"{r:.4f} {g:.4f} {b:.4f} 1"



def generate_telop_fcpxml(
    telop_entries: list[dict],      # [{id, start, end, text}] — source times
    keep_segments: list[Segment],   # built from removal candidates
    total_source_duration: float,
    settings: dict,                 # fps, width, height, font, font_size, font_color
    stem: str,
    output_path: Path,
    use_source_timing: bool = False,
) -> None:
    """Write a telop FCPXML with Basic Title clips on a gap layer.

    When use_source_timing=False (default), title offsets are remapped to the
    OUTPUT timeline (after cuts) so the telop aligns with the rough-cut sequence.
    When use_source_timing=True, original source timestamps are used unchanged —
    useful when importing alongside uncut footage in FCP.

    settings keys:
        fps           str   e.g. "24", "29.97"
        width         int
        height        int
        font          str
        font_size     int
        font_color    str   hex, e.g. "#F3B500"
        position_y    int   vertical offset in pixels from center (default -420)
        line_spacing  int   line spacing in points, negative tightens (default -65)
    """
    fps_str   = str(settings.get("fps", "24"))
    width     = int(settings.get("width", 3840))
    height    = int(settings.get("height", 2160))
    font      = settings.get("font", "Hiragino Sans")
    font_size = max(1, int(settings.get("font_size", 92)))  # guard: 0 → ZeroDivisionError
    font_color_hex = settings.get("font_color", "#F3B500")
    position_y   = int(settings.get("position_y", -420))
    line_spacing = int(settings.get("line_spacing", -65))

    fd = _frame_dur(fps_str)
    frame_dur_str = f"{fd.numerator}/{fd.denominator}s"
    # Drop-frame timecode applies to 29.97 and 59.94 fps (NTSC colour burst rates).
    # All other frame rates (23.976, 24, 25, 30, 50, 60 …) use non-drop-frame.
    tc_format = "DF" if fps_str in ("29.97", "59.94") else "NDF"
    font_color = _hex_to_fcpxml_color(font_color_hex)

    # Maximum em-units per line.  FCP Basic Title renders font_size in a coordinate
    # space where 1 em ≈ font_size px.  Empirically, ~0.42 of the frame width is the
    # safe text area (leaves ~29% combined margins).  Enforced minimum of 8 em so
    # very small resolutions or large fonts always get at least a few characters.
    max_em_per_line = max(8.0, width * 0.42 / font_size)

    def _process_entry_text(raw: str) -> str:
        text = _clean_telop_text(raw)
        if not text:
            return ""
        return _wrap_telop(text, max_em_per_line)

    if use_source_timing:
        total_out_str = _to_rt(total_source_duration + 1.0, fd)
        entry_spans = [
            (e["start"], e["end"], e.get("text", ""))
            for e in filter_telop_entries(telop_entries)
        ]
    else:
        total_out = sum(s.duration for s in keep_segments) if keep_segments else total_source_duration
        total_out_str = _to_rt(total_out + 1.0, fd)
        entry_spans = []
        for e in filter_telop_entries(telop_entries):
            span = map_span_to_output(e["start"], e["end"], keep_segments)
            if span is not None:
                entry_spans.append((*span, e.get("text", "")))

    mapped = []
    for out_start, out_end, raw_text in entry_spans:
        # Skip entries whose duration rounds to zero frames.
        # The old guard `< 0.02` is frame-rate-agnostic and fails at 24 fps:
        # at 24fps a duration of exactly 0.02s passes the test but
        # round(0.02 / (1/24)) = round(0.48) = 0, producing duration="0s"
        # which FCP rejects.  Compare the *rounded* frame count instead.
        if round((out_end - out_start) / float(fd)) < 1:
            continue
        text = _process_entry_text(raw_text)
        if not text:
            continue
        mapped.append({"out_start": out_start, "out_end": out_end, "text": text})

    # Build XML via ElementTree
    fcpxml = ET.Element("fcpxml", version="1.11")
    resources = ET.SubElement(fcpxml, "resources")

    ET.SubElement(
        resources, "format",
        id="r1",
        name=_format_name(width, height, fps_str),
        width=str(width),
        height=str(height),
        frameDuration=frame_dur_str,
    )
    ET.SubElement(
        resources, "effect",
        id="r2",
        name="Basic Title",
        uid=".../Titles.localized/Bumper:Opener.localized/Basic Title.localized/Basic Title.moti",
    )

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name="Telop Import")
    project = ET.SubElement(event, "project", name=f"{stem}_telop")
    sequence = ET.SubElement(
        project, "sequence",
        format="r1",
        duration=total_out_str,
        tcStart="0s",
        tcFormat=tc_format,
    )
    spine = ET.SubElement(sequence, "spine")
    gap = ET.SubElement(
        spine, "gap",
        name="Gap",
        offset="0s",
        duration=total_out_str,
    )

    for i, entry in enumerate(mapped):
        style_id = f"ts{i + 1}"
        name_attr = entry["text"].replace("\n", " ")[:40]

        title = ET.SubElement(
            gap, "title",
            ref="r2",
            lane="1",
            offset=_to_rt(entry["out_start"], fd),
            duration=_to_rt(entry["out_end"] - entry["out_start"], fd),
            name=name_attr,
        )
        ET.SubElement(
            title, "param",
            name="Position",
            key="9999/999166631/999166633/1/100/101",
            value=f"0 {position_y}",
        )
        text_elem = ET.SubElement(title, "text")
        ts_ref = ET.SubElement(text_elem, "text-style", ref=style_id)
        ts_ref.text = entry["text"]

        ts_def = ET.SubElement(title, "text-style-def", id=style_id)
        ET.SubElement(
            ts_def, "text-style",
            font=font,
            fontSize=str(font_size),
            fontFace="Regular",
            fontColor=font_color,
            bold="1",
            alignment="center",
            lineSpacing=str(line_spacing),
        )

    tree = ET.ElementTree(fcpxml)
    ET.indent(tree, space="    ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE fcpxml>\n')
        tree.write(f, encoding="unicode", xml_declaration=False)
