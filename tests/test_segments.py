"""Tests for segments.py — build_segments(), map_span_to_output(), filter_telop_entries(), split_telop_segments()."""

import pytest
from preprod.segments import Segment, build_segments, map_span_to_output, map_time_to_output, filter_telop_entries, split_telop_segments, group_word_tokens, _join_word_texts, _halfwidth, assign_words_to_entries


# ── _halfwidth() ─────────────────────────────────────────────────────────────

class TestHalfwidth:
    def test_ascii_unchanged(self):
        assert _halfwidth("Google") == "Google"

    def test_fullwidth_uppercase_normalised(self):
        assert _halfwidth("ＡＭＡＺＯＮ") == "AMAZON"

    def test_fullwidth_lowercase_normalised(self):
        assert _halfwidth("ｇｏｏｇｌｅ") == "google"

    def test_fullwidth_digits_normalised(self):
        assert _halfwidth("２０２６") == "2026"

    def test_mixed_fullwidth_and_cjk_unchanged_kanji(self):
        # Kanji must pass through unchanged
        assert _halfwidth("Ａｌ万円") == "Al万円"

    def test_fullwidth_punctuation_unchanged(self):
        # ！ (U+FF01) and ？ (U+FF1F) are NOT in the alphanumeric range → stay fullwidth
        assert _halfwidth("！？") == "！？"

    def test_empty_string(self):
        assert _halfwidth("") == ""


# ── _join_word_texts() ────────────────────────────────────────────────────────

def _jw(*pairs):
    """Build a minimal word list: _jw(("word", start, end), ...) → [{word, start, end}]."""
    return [{"word": p[0], "start": p[1], "end": p[2]} for p in pairs]

class TestJoinWordTexts:
    def test_empty_returns_empty(self):
        assert _join_word_texts([]) == ""

    def test_single_latin_word(self):
        assert _join_word_texts(_jw(("hello", 0, 1))) == "hello"

    def test_english_words_space_joined(self):
        # A real inter-word pause (0.1s gap) keeps the space.
        words = _jw(("Claude", 0, 0.5), ("Code", 0.6, 1.0))
        assert _join_word_texts(words) == "Claude Code"

    def test_contiguous_latin_fragments_no_space(self):
        # One word Whisper split into chunks with no pause → joined without a space.
        words = _jw(("元", 0, 0.3), ("Goo", 0.3, 0.6), ("gle", 0.6, 0.9), ("の", 0.9, 1.0))
        assert _join_word_texts(words) == "元Googleの"

    def test_latin_fragments_with_real_gap_keep_space(self):
        # A genuine word boundary (gap ≥ 0.04s) still gets a space.
        words = _jw(("Claude", 0, 0.5), ("Code", 0.55, 1.0))
        assert _join_word_texts(words) == "Claude Code"

    def test_japanese_chars_concatenated(self):
        # faster-whisper emits char-level tokens for Japanese
        words = _jw(("え", 0, 0.1), ("ー", 0.1, 0.2), ("と", 0.2, 0.3))
        assert _join_word_texts(words) == "えーと"

    def test_mixed_latin_then_cjk_no_space_at_boundary(self):
        # "Code" + "を" — no space between Latin and CJK
        words = _jw(("Code", 0, 0.4), ("を", 0.5, 0.6), ("使", 0.6, 0.7), ("う", 0.7, 0.8))
        assert _join_word_texts(words) == "Codeを使う"

    def test_mixed_cjk_then_latin_no_space_at_boundary(self):
        # Japanese into English: no space at the CJK→Latin boundary
        words = _jw(("使", 0, 0.1), ("う", 0.1, 0.2), ("Claude", 0.3, 0.8))
        assert _join_word_texts(words) == "使うClaude"

    def test_mixed_latin_cjk_latin_full_sentence(self):
        # "Claude Code を使う API" — two Latin words, Japanese verb phrase, Latin noun
        words = _jw(
            ("Claude", 0, 0.5), ("Code", 0.6, 0.9),
            ("を", 1.0, 1.1), ("使", 1.1, 1.2), ("う", 1.2, 1.3),
            ("API", 1.4, 1.8),
        )
        assert _join_word_texts(words) == "Claude Codeを使うAPI"

    def test_text_key_accepted_for_grouped_tokens(self):
        """group_word_tokens uses 'text' key; must not silently produce empty strings."""
        words = [
            {"text": "えーと", "start": 0.0, "end": 0.4},
            {"text": "Claude", "start": 0.5, "end": 0.9},
        ]
        assert _join_word_texts(words) == "えーとClaude"

    def test_empty_tokens_skipped(self):
        # faster-whisper sometimes emits space-only tokens
        words = _jw(("hello", 0, 0.4), (" ", 0.4, 0.5), ("world", 0.5, 0.9))
        assert _join_word_texts(words) == "hello world"

    def test_japanese_punctuation_no_space(self):
        """Ideographic punctuation (。、 at U+3001-U+303F) must not trigger space insertion."""
        # "こんにちは" + "。" — punctuation token adjacent to CJK
        words = [
            {"word": "こんにちは", "start": 0.0, "end": 0.5},
            {"word": "。",         "start": 0.5, "end": 0.55},
        ]
        assert _join_word_texts(words) == "こんにちは。"

    def test_japanese_punctuation_before_latin_no_space(self):
        """。 followed by Latin text — no space inserted between them."""
        words = _jw(("。", 0, 0.1), ("Hello", 0.2, 0.5))
        # 。 is CJK (U+3002 in U+3000-U+9FFF range) → not both non-CJK → no space
        assert _join_word_texts(words) == "。Hello"

    def test_single_char_ascii_run_joined_without_spaces(self):
        """faster-whisper char-level tokens for company names must NOT get spaces."""
        # 'G','o','o','g','l','e' → "Google"
        words = _jw(("G", 0, 0.02), ("o", 0.02, 0.04), ("o", 0.04, 0.06),
                    ("g", 0.06, 0.08), ("l", 0.08, 0.10), ("e", 0.10, 0.12))
        assert _join_word_texts(words) == "Google"

    def test_single_char_digit_run_joined_without_spaces(self):
        """Single-char digit tokens must concatenate — '1','7' → '17', not '1 7'."""
        words = _jw(("1", 0, 0.05), ("7", 0.05, 0.10))
        assert _join_word_texts(words) == "17"

    def test_single_char_run_then_cjk_no_space(self):
        """Single-char alnum run followed by CJK → no space at boundary."""
        words = _jw(("1", 0, 0.05), ("7", 0.05, 0.10), ("年", 0.10, 0.20))
        assert _join_word_texts(words) == "17年"

    def test_fullwidth_latin_normalised_and_joined(self):
        """Fullwidth letters are normalised to ASCII and concatenated."""
        words = _jw(("Ｇ", 0, 0.02), ("ｏ", 0.02, 0.04), ("ｏ", 0.04, 0.06),
                    ("ｇ", 0.06, 0.08), ("ｌ", 0.08, 0.10), ("ｅ", 0.10, 0.12))
        assert _join_word_texts(words) == "Google"

    def test_fullwidth_digits_normalised(self):
        """Fullwidth digit tokens are normalised to ASCII."""
        words = _jw(("２", 0, 0.05), ("０", 0.05, 0.10), ("２", 0.10, 0.15), ("６", 0.15, 0.20))
        assert _join_word_texts(words) == "2026"

    def test_multichar_latin_word_still_gets_space(self):
        """Multi-char tokens are NOT part of the single-char run path — space is still inserted."""
        words = _jw(("Claude", 0, 0.5), ("Code", 0.6, 1.0))
        assert _join_word_texts(words) == "Claude Code"

    def test_digit_then_percent_no_space(self):
        """Whisper may emit '%' as a separate token after digit chars — must join without space."""
        words = _jw(("1", 0, 0.02), ("0", 0.02, 0.04), ("%", 0.04, 0.06))
        assert _join_word_texts(words) == "10%"

    def test_multichar_word_then_percent_keeps_space(self):
        """'%' after a multi-char non-digit word is still space-separated (not numeric context)."""
        words = _jw(("hello", 0, 0.5), ("%", 0.6, 0.7))
        assert _join_word_texts(words) == "hello %"


# ── build_segments() ─────────────────────────────────────────────────────────

class TestBuildSegmentsNoRemovals:
    def test_empty_removal_list_returns_full_duration(self):
        segs = build_segments([], total_duration=60.0)
        assert len(segs) == 1
        assert segs[0].source_start == 0.0
        assert segs[0].source_end == 60.0

    def test_zero_duration_empty_removal(self):
        segs = build_segments([], total_duration=0.0)
        assert len(segs) == 1
        assert segs[0].duration == 0.0


class TestBuildSegmentsSingleRemoval:
    def test_removal_in_middle(self):
        # Remove 10–20 from a 30s clip → keep [0,10] and [20,30]
        segs = build_segments([(10.0, 20.0)], total_duration=30.0)
        assert len(segs) == 2
        assert segs[0].source_start == pytest.approx(0.0)
        assert segs[0].source_end == pytest.approx(10.0)
        assert segs[1].source_start == pytest.approx(20.0)
        assert segs[1].source_end == pytest.approx(30.0)

    def test_removal_at_start(self):
        segs = build_segments([(0.0, 10.0)], total_duration=30.0)
        # First removal starts at 0 — no leading segment
        assert len(segs) == 1
        assert segs[0].source_start == pytest.approx(10.0)
        assert segs[0].source_end == pytest.approx(30.0)

    def test_removal_at_end(self):
        segs = build_segments([(20.0, 30.0)], total_duration=30.0)
        assert len(segs) == 1
        assert segs[0].source_start == pytest.approx(0.0)
        assert segs[0].source_end == pytest.approx(20.0)

    def test_removal_covers_entire_clip(self):
        # When the removal exactly covers [0, total_duration], the padded removal
        # starts at 0 (not > 0.001) and ends at total_duration (not < total_duration - 0.001),
        # so no leading or trailing keep-segment exists → result is empty list.
        segs = build_segments([(0.0, 30.0)], total_duration=30.0)
        assert len(segs) == 0

    def test_removal_covers_entire_clip_with_padding(self):
        # 2 s clip, remove 0–2 with 500 ms padding → padded removal shrinks to 0.5–1.5
        # Gap = [0,0.5] and [1.5,2] both 0.5 s — both should survive
        segs = build_segments([(0.0, 2.0)], total_duration=2.0, padding_ms=500)
        # Leading segment [0, 0.5] should appear
        starts = [s.source_start for s in segs]
        ends = [s.source_end for s in segs]
        assert 0.0 in starts


class TestBuildSegmentsMultipleRemovals:
    def test_two_non_overlapping_removals(self):
        segs = build_segments([(5.0, 10.0), (20.0, 25.0)], total_duration=30.0)
        # Expected keep: [0,5], [10,20], [25,30]
        assert len(segs) == 3
        assert segs[0].source_end == pytest.approx(5.0)
        assert segs[1].source_start == pytest.approx(10.0)
        assert segs[1].source_end == pytest.approx(20.0)
        assert segs[2].source_start == pytest.approx(25.0)

    def test_overlapping_removals_are_merged(self):
        # [5,15] and [10,20] overlap → merged to [5,20]
        segs = build_segments([(5.0, 15.0), (10.0, 20.0)], total_duration=30.0)
        # Expected keep: [0,5] and [20,30]
        assert len(segs) == 2
        assert segs[0].source_end == pytest.approx(5.0)
        assert segs[1].source_start == pytest.approx(20.0)

    def test_adjacent_removals_treated_as_overlapping(self):
        # [5,10] and [10,15] → touching, merged to [5,15]
        segs = build_segments([(5.0, 10.0), (10.0, 15.0)], total_duration=30.0)
        assert len(segs) == 2
        assert segs[0].source_end == pytest.approx(5.0)
        assert segs[1].source_start == pytest.approx(15.0)

    def test_unsorted_removals_are_sorted(self):
        # Providing removals out of order — result should be same as sorted
        segs_unsorted = build_segments([(20.0, 25.0), (5.0, 10.0)], total_duration=30.0)
        segs_sorted = build_segments([(5.0, 10.0), (20.0, 25.0)], total_duration=30.0)
        assert len(segs_unsorted) == len(segs_sorted)
        for a, b in zip(segs_unsorted, segs_sorted):
            assert a.source_start == pytest.approx(b.source_start)
            assert a.source_end == pytest.approx(b.source_end)


class TestBuildSegmentsPadding:
    def test_padding_shrinks_removal_inward(self):
        # Remove 10–20, 500 ms padding → padded removal is 10.5–19.5
        # Keep: [0, 10.5] and [19.5, 30]
        segs = build_segments([(10.0, 20.0)], total_duration=30.0, padding_ms=500)
        assert len(segs) == 2
        assert segs[0].source_end == pytest.approx(10.5)
        assert segs[1].source_start == pytest.approx(19.5)

    def test_padding_larger_than_removal_collapses_to_nothing(self):
        # Remove 10–11 (1 s) with 1000 ms padding → padded window collapses
        # → removal is skipped, full clip returned
        segs = build_segments([(10.0, 11.0)], total_duration=30.0, padding_ms=1000)
        assert len(segs) == 1
        assert segs[0].source_start == 0.0
        assert segs[0].source_end == 30.0

    def test_zero_padding(self):
        segs_no_pad = build_segments([(10.0, 20.0)], total_duration=30.0, padding_ms=0)
        assert segs_no_pad[0].source_end == pytest.approx(10.0)
        assert segs_no_pad[1].source_start == pytest.approx(20.0)

    def test_very_short_gap_dropped(self):
        # Two adjacent removals leave a 30ms gap — below the 50ms minimum → dropped.
        # Remove [0,9.97] and [10.0, 30] leaves a gap of 30ms at [9.97,10.0].
        segs = build_segments([(0.0, 9.97), (10.0, 30.0)], total_duration=30.0)
        assert all(s.duration >= 0.05 for s in segs)

    def test_segment_at_50ms_exactly_is_kept(self):
        # A gap of exactly 50ms is at the boundary — it should be kept (>=0.05).
        # Remove [0,9.95] and [10.0, 30] leaves exactly 50ms at [9.95,10.0].
        segs = build_segments([(0.0, 9.95), (10.0, 30.0)], total_duration=30.0)
        gap_segs = [s for s in segs if abs(s.source_start - 9.95) < 0.01]
        assert len(gap_segs) == 1
        assert gap_segs[0].duration == pytest.approx(0.05, abs=0.001)


class TestSegmentDurationProperty:
    def test_duration_computed_correctly(self):
        seg = Segment(source_start=5.0, source_end=15.0)
        assert seg.duration == pytest.approx(10.0)


# ── map_span_to_output() ─────────────────────────────────────────────────────

class TestMapSpanToOutput:
    def setup_method(self):
        # Simple: keep [0,10] and [20,30] (removal at [10,20])
        self.segs = build_segments([(10.0, 20.0)], total_duration=30.0)

    def test_span_entirely_in_first_keep(self):
        result = map_span_to_output(2.0, 8.0, self.segs)
        assert result == pytest.approx((2.0, 8.0))

    def test_span_entirely_in_second_keep(self):
        # Source [21,25] → output [11,15] (first seg is 10s long)
        result = map_span_to_output(21.0, 25.0, self.segs)
        assert result == pytest.approx((11.0, 15.0))

    def test_span_entirely_in_removed_region(self):
        result = map_span_to_output(12.0, 18.0, self.segs)
        assert result is None

    def test_span_straddles_cut(self):
        # [8, 22] overlaps both kept regions [0,10] and [20,30]
        result = map_span_to_output(8.0, 22.0, self.segs)
        assert result is not None
        out_start, out_end = result
        assert out_start == pytest.approx(8.0)   # 8 in first seg → output 8
        assert out_end == pytest.approx(12.0)     # 22 in second seg → output 10+(22-20)=12

    def test_span_starts_at_segment_boundary(self):
        result = map_span_to_output(0.0, 5.0, self.segs)
        assert result == pytest.approx((0.0, 5.0))

    def test_span_ends_at_segment_boundary(self):
        result = map_span_to_output(5.0, 10.0, self.segs)
        assert result == pytest.approx((5.0, 10.0))

    def test_span_beyond_all_segments_returns_none(self):
        result = map_span_to_output(35.0, 40.0, self.segs)
        assert result is None

    def test_empty_keep_segments_returns_none(self):
        result = map_span_to_output(0.0, 5.0, [])
        assert result is None

    def test_full_clip_no_removal(self):
        segs = build_segments([], total_duration=30.0)
        result = map_span_to_output(0.0, 30.0, segs)
        assert result == pytest.approx((0.0, 30.0))


# ── map_time_to_output() ─────────────────────────────────────────────────────

class TestMapTimeToOutput:
    def setup_method(self):
        self.segs = build_segments([(10.0, 20.0)], total_duration=30.0)

    def test_time_in_first_segment(self):
        assert map_time_to_output(5.0, self.segs) == pytest.approx(5.0)

    def test_time_in_second_segment(self):
        assert map_time_to_output(25.0, self.segs) == pytest.approx(15.0)

    def test_time_in_removal_returns_none(self):
        assert map_time_to_output(15.0, self.segs) is None

    def test_time_at_start(self):
        assert map_time_to_output(0.0, self.segs) == pytest.approx(0.0)

    def test_time_beyond_end_returns_none(self):
        assert map_time_to_output(35.0, self.segs) is None

    def test_empty_segments_returns_none(self):
        assert map_time_to_output(5.0, []) is None

    def test_time_at_segment_boundary_end_of_first(self):
        # t=10.0 is the end of the first keep-segment [0,10] — maps to 10.0 out.
        assert map_time_to_output(10.0, self.segs) == pytest.approx(10.0)

    def test_time_at_segment_boundary_start_of_second(self):
        # t=20.0 is the start of the second keep-segment [20,30] — maps to 10.0 out.
        assert map_time_to_output(20.0, self.segs) == pytest.approx(10.0)


# ── filter_telop_entries() ────────────────────────────────────────────────────

class TestFilterTelopEntries:
    """filter_telop_entries: skip malformed entries, return sorted valid ones."""

    def test_returns_empty_for_empty_list(self):
        assert filter_telop_entries([]) == []

    def test_valid_entries_returned(self):
        entries = [{"start": 1.0, "end": 3.0, "text": "hi"}]
        assert filter_telop_entries(entries) == entries

    def test_entry_missing_start_skipped(self):
        entries = [{"end": 3.0, "text": "no start"}, {"start": 0.0, "end": 2.0, "text": "ok"}]
        result = filter_telop_entries(entries)
        assert len(result) == 1
        assert result[0]["text"] == "ok"

    def test_entry_missing_end_skipped(self):
        entries = [{"start": 1.0, "text": "no end"}, {"start": 2.0, "end": 4.0, "text": "ok"}]
        result = filter_telop_entries(entries)
        assert len(result) == 1

    def test_entry_missing_both_skipped(self):
        assert filter_telop_entries([{"text": "orphan"}]) == []

    def test_results_sorted_by_start(self):
        entries = [
            {"start": 5.0, "end": 7.0, "text": "b"},
            {"start": 1.0, "end": 3.0, "text": "a"},
        ]
        result = filter_telop_entries(entries)
        assert [e["text"] for e in result] == ["a", "b"]

    def test_preserves_extra_fields(self):
        entries = [{"id": "e1", "start": 0.5, "end": 1.5, "text": "x", "extra": "y"}]
        result = filter_telop_entries(entries)
        assert result[0]["extra"] == "y"

    def test_all_malformed_returns_empty(self):
        entries = [{"text": "a"}, {"start": 0.0}, {"end": 1.0}]
        assert filter_telop_entries(entries) == []


# ── split_telop_segments() ────────────────────────────────────────────────────

def _w(word, start, end):
    """Helper: build a word dict."""
    return {"word": word, "start": start, "end": end}

def _seg(text, start, end):
    return {"text": text, "start": start, "end": end}


class TestSplitTelopSegments:

    # ── passthrough ───────────────────────────────────────────────────────────

    def test_short_segment_passes_through_unchanged(self):
        segs   = [_seg("短いテスト", 0.0, 3.0)]
        words  = [_w("短", 0.0, 0.5), _w("い", 0.5, 1.0), _w("テ", 1.0, 1.5),
                  _w("ス", 1.5, 2.0), _w("ト", 2.0, 2.5)]
        result = split_telop_segments(segs, words, max_duration=6.0, max_em=40.0)
        assert len(result) == 1
        assert result[0] == segs[0]

    # ── word-level text reconstruction ────────────────────────────────────────

    def test_word_reconstruction_fixes_mid_word_decoder_cut(self):
        """A word split mid-way across a segment boundary is never broken across cards.

        Regression: Whisper's decoder emitted "フリー" at the end of seg0 and
        "ランス..." at the start of seg1, while word alignment placed the complete
        token "フリーランス" in seg0.  The compound word "フリーランス" must end up
        whole — never as "フリー" | "ランス" across two cards.  (With morphological
        analysis the two fragments are recognised as one word and merged into a
        single clean card; without fugashi, reconstruction still keeps it whole.)
        """
        words = [
            _w("フリーランス", 2.9, 3.5),   # complete word in seg0, extends past seg end
            _w("の",           3.5, 3.7),   # first word of seg1
        ]
        segs = [
            _seg("フリー",  0.0, 3.2),   # decoder text cuts the word short
            _seg("ランスの", 3.2, 6.0),  # decoder starts seg1 with the fragment
        ]
        result = split_telop_segments(segs, words, max_duration=6.0, max_em=40.0)
        joined = "".join(r["text"] for r in result)
        assert "フリーランス" in joined, \
            f"'フリーランス' must stay intact; got {[r['text'] for r in result]!r}"
        # The compound must never be cut as a card ending in "フリー".
        assert not any(r["text"].rstrip().endswith("フリー") for r in result), \
            f"'フリー' must not be a card-final fragment; got {[r['text'] for r in result]!r}"

    def test_leading_boundary_word_excluded_from_reconstruction(self):
        """Word starting > 20ms before segment start is not included in reconstruction.

        The ±100ms tolerance window in _words_for_seg can pull in a trailing
        punctuation mark from the previous segment.  The reconstruction step
        uses a tighter ±20ms window so that only words genuinely belonging to
        this segment are included.
        """
        words = [
            _w("。",   0.85, 0.95),   # ends just before seg start; belongs to prev seg
            _w("か",   1.0,  1.2),    # first real word of this segment
            _w("れ",   1.2,  1.4),
        ]
        segs = [_seg("かれ", 1.0, 4.0)]
        result = split_telop_segments(segs, words, max_duration=6.0, max_em=40.0)
        assert len(result) == 1
        # "。" must NOT appear at the start of the reconstructed text
        assert not result[0]["text"].startswith("。"), \
            f"Bleeding punctuation must be excluded; got {result[0]['text']!r}"
        assert "か" in result[0]["text"]

    def test_empty_segments_list_returns_empty(self):
        assert split_telop_segments([], []) == []

    def test_no_words_passes_segment_through(self):
        segs = [_seg("test", 0.0, 10.0)]
        result = split_telop_segments(segs, [], max_duration=6.0, max_em=40.0)
        assert result == segs

    def test_oversized_chunk_after_sentence_split_not_cut_mid_word(self):
        """Regression: a long segment split at 。 leaves a sub-chunk still over the
        limit; re-splitting it must land on a phrase boundary, never inside a word.

        Real bug: 'Google' (single-char tokens G,o,o,g,l,e in one 12.8s segment)
        was cut 'Goo' | 'gle' because the _build_chunks recursion used a blind
        word-index midpoint.  It now re-splits via _phrase_split (fugashi-aware)."""
        # seg: "<filler>。<…元Google偉い…>" — long enough to need two re-splits.
        pre = list("あいうえおかきくけこさしすせそたちつてとなにぬねの")  # 22 kana
        goog = list("元Google偉い人物だと")
        post = list("みんなが思っているのですけれども実際には違いますよね")
        toks = pre + ["。"] + goog + post
        words = []
        t = 0.0
        for ch in toks:
            words.append(self._wsid(ch, round(t, 3), round(t + 0.25, 3), 0))
            t += 0.25
        segs = [self._ssid("".join(toks), 0.0, round(t, 3), 0)]
        result = split_telop_segments(segs, words, max_duration=6.0, max_em=20.0)
        joined = "".join(r["text"] for r in result)
        assert "Google" in joined, f"Google must stay intact; got {[r['text'] for r in result]!r}"
        # No card may end or start with a fragment of "Google".
        for r in result:
            txt = r["text"].strip()
            for frag in ("Goo", "Goog", "gle", "ogle", "oogle"):
                assert not txt.endswith(frag) and not txt.startswith(frag), \
                    f"'Google' split as fragment {frag!r}: {[x['text'] for x in result]!r}"

    # ── seg_id-based reconstruction (no cross-boundary bleed) ──────────────────

    def _wsid(self, word, start, end, seg_id):
        return {"word": word, "start": start, "end": end, "seg_id": seg_id}

    def _ssid(self, text, start, end, seg_id):
        return {"text": text, "start": start, "end": end, "seg_id": seg_id}

    def test_seg_id_prevents_boundary_word_duplication(self):
        """A word whose start collides with the previous segment's end (rounding)
        must reconstruct into exactly one card, selected by seg_id — not both.

        Real regression: a segment's leading kana bled into the previous card via
        the ±100ms window AND appeared in its own card, duplicating it.  Here the two
        segments form a clean phrase boundary (は particle | 明日 noun) so they do NOT
        merge, isolating the seg_id de-duplication behaviour.  The first word of seg1
        ("明") has a start time that collides (within rounding) with seg0's end.
        """
        words = [
            self._wsid("今", 1.00, 1.40, 0),
            self._wsid("日", 1.40, 1.70, 0),
            self._wsid("は", 1.70, 2.00, 0),   # seg0 ends here
            self._wsid("明", 1.999, 2.40, 1),  # seg1 — start collides with seg0 end
            self._wsid("日", 2.40, 2.70, 1),
            self._wsid("も", 2.70, 3.00, 1),
        ]
        segs = [
            self._ssid("今日は", 1.00, 2.00, 0),
            self._ssid("明日も", 1.999, 3.00, 1),
        ]
        result = split_telop_segments(segs, words, max_duration=6.0, max_em=40.0)
        assert len(result) == 2
        # seg0 gets ONLY its own words; "明" must not bleed in.
        assert result[0]["text"] == "今日は"
        # seg1 keeps its own "明日も"; no duplication of the boundary word.
        assert result[1]["text"] == "明日も"

    def test_seg_id_excludes_next_segment_large_token(self):
        """A large token belonging to the next segment must not bleed in and get
        truncated to a lone leading char (the "比" bug).  Both segs ≥1 s (no merge)."""
        words = [
            self._wsid("な", 0.0, 0.5, 0),
            self._wsid("ん", 0.5, 1.0, 0),   # seg0 ends (dur 1.0s)
            self._wsid("比較的オーガニック", 0.999, 2.50, 1),  # seg1 big token, ~1ms overlap
        ]
        segs = [
            self._ssid("なん", 0.0, 1.0, 0),
            self._ssid("比較的オーガニック", 0.999, 2.50, 1),
        ]
        result = split_telop_segments(segs, words, max_duration=6.0, max_em=40.0)
        assert result[0]["text"] == "なん"          # no trailing "比"
        assert not result[0]["text"].endswith("比")

    def test_missing_seg_id_falls_back_to_time_window(self):
        """Segments/words without seg_id keep the legacy time-window behaviour."""
        words = [_w("か", 1.0, 1.2), _w("れ", 1.2, 1.4)]
        segs = [_seg("かれ", 1.0, 4.0)]
        result = split_telop_segments(segs, words, max_duration=6.0, max_em=40.0)
        assert len(result) == 1
        assert "か" in result[0]["text"]

    # ── sokuon mid-word merge (issue: "そういっ" | "た") ────────────────────────

    def test_sokuon_tail_merges_long_segments(self):
        """A segment ending in っ (sokuon) is mid-word; merge with the next even
        when both are long, then re-split avoiding a break right after っ."""
        # seg0 ends "…そういっ", seg1 starts "た…", tiny gap (continuous speech).
        # Combined overflows max_em, forcing a re-split that must NOT land after っ.
        words = (
            [self._wsid(c, 10.0 + i * 0.1, 10.0 + (i + 1) * 0.1, 0)
             for i, c in enumerate("はなとかそういっ")]
            + [self._wsid(c, 10.81 + i * 0.1, 10.81 + (i + 1) * 0.1, 1)
               for i, c in enumerate("た全然違う見た目")]
        )
        segs = [
            self._ssid("はなとかそういっ", 10.0, 10.8, 0),
            self._ssid("た全然違う見た目", 10.81, 11.61, 1),   # gap 0.01s
        ]
        result = split_telop_segments(segs, words, max_duration=6.0, max_em=8.0)
        # No output card may end with a sokuon (which would orphan the next char).
        for card in result:
            last = card["text"].rstrip()[-1:]
            assert last not in "っッ", f"card ends with sokuon: {card['text']!r}"
        # "た" must not be stranded as the start of a card by itself either; the
        # combined text "…そういった…" should appear contiguously somewhere.
        joined = "".join(c["text"] for c in result)
        assert "そういった" in joined

    def test_midword_head_merges_segment_starting_with_small_kana(self):
        """A segment STARTING with a sokuon/small kana cannot begin a word, so it is
        a mid-word cut (e.g. "そうい" | "った") and must be merged + re-split without
        stranding the fragment."""
        words = (
            [self._wsid(c, 10.0 + i * 0.1, 10.0 + (i + 1) * 0.1, 0)
             for i, c in enumerate("はなとかそうい")]
            + [self._wsid(c, 10.71 + i * 0.1, 10.71 + (i + 1) * 0.1, 1)
               for i, c in enumerate("った全然違う見た目")]
        )
        segs = [
            self._ssid("はなとかそうい", 10.0, 10.7, 0),
            self._ssid("った全然違う見た目", 10.71, 11.51, 1),   # starts with っ, gap 0.01s
        ]
        result = split_telop_segments(segs, words, max_duration=6.0, max_em=8.0)
        # No card may start with a small kana / mark (would be a stranded fragment).
        for card in result:
            first = card["text"].lstrip()[:1]
            assert first not in "っッゃゅょぁぃぅぇぉ", f"card starts mid-word: {card['text']!r}"

    def test_sokuon_merge_not_triggered_across_sentence_end(self):
        """A sentence-ending punctuation blocks the sokuon merge."""
        words = (
            [self._wsid(c, 0.0 + i * 0.1, 0.0 + (i + 1) * 0.1, 0) for i, c in enumerate("あっ。")]
            + [self._wsid(c, 1.0 + i * 0.1, 1.0 + (i + 1) * 0.1, 1) for i, c in enumerate("たいへん")]
        )
        segs = [
            self._ssid("あっ。", 0.0, 0.3, 0),
            self._ssid("たいへん", 1.0, 1.4, 1),
        ]
        result = split_telop_segments(segs, words, max_duration=6.0, max_em=40.0)
        # "。" ends the sentence → must NOT merge into "あっ。たいへん".
        assert len(result) == 2

    # ── sentence-boundary splitting ───────────────────────────────────────────

    def test_splits_at_japanese_maru(self):
        # Two sentences joined: "これは文です。次の文です。" — 14 seconds
        words = [
            _w("こ", 0.0, 0.3), _w("れ", 0.3, 0.6), _w("は", 0.6, 0.9),
            _w("文", 0.9, 1.2), _w("で", 1.2, 1.5), _w("す", 1.5, 1.8),
            _w("。", 1.8, 2.0),   # ← sentence boundary
            _w("次", 7.0, 7.3), _w("の", 7.3, 7.6), _w("文", 7.6, 7.9),
            _w("で", 7.9, 8.2), _w("す", 8.2, 8.5), _w("。", 8.5, 8.7),
        ]
        seg = _seg("これは文です。次の文です。", 0.0, 14.0)
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=40.0)
        assert len(result) == 2
        # First chunk uses seg start; second chunk uses seg end
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == pytest.approx(2.0, abs=0.05)
        assert result[1]["start"] == pytest.approx(7.0, abs=0.05)
        assert result[1]["end"] == 14.0

    def test_splits_at_exclamation_mark(self):
        words = [
            _w("Hello", 0.0, 0.5), _w("world", 0.5, 1.0), _w("!", 1.0, 1.1),
            _w("How", 5.0, 5.3), _w("are", 5.3, 5.6), _w("you", 5.6, 5.9),
            _w("?", 5.9, 6.0),
        ]
        seg = _seg("Hello world! How are you?", 0.0, 12.0)
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=40.0)
        assert len(result) == 2

    def test_splits_at_question_mark_japanese(self):
        words = [
            _w("何", 0.0, 0.5), _w("で", 0.5, 0.8), _w("す", 0.8, 1.0), _w("か", 1.0, 1.2),
            _w("？", 1.2, 1.4),  # ← full-width question mark
            _w("そ", 5.0, 5.3), _w("れ", 5.3, 5.6), _w("は", 5.6, 5.9), _w("正", 5.9, 6.2),
            _w("し", 6.2, 6.5), _w("い", 6.5, 6.8),
        ]
        seg = _seg("何ですか？それは正しい", 0.0, 13.0)
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=40.0)
        assert len(result) == 2

    def test_multiple_sentence_boundaries_produce_multiple_chunks(self):
        # Three sentences, each followed by 。
        words = (
            [_w(c, i * 0.5, (i + 1) * 0.5) for i, c in enumerate("AAAA。")] +
            [_w(c, 7 + i * 0.5, 7 + (i + 1) * 0.5) for i, c in enumerate("BBBB。")] +
            [_w(c, 14 + i * 0.5, 14 + (i + 1) * 0.5) for i, c in enumerate("CCCC。")]
        )
        seg = _seg("AAAA。BBBB。CCCC。", 0.0, 21.0)
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=40.0)
        assert len(result) == 3

    # ── time-division fallback ────────────────────────────────────────────────

    def test_time_split_when_no_sentence_boundary(self):
        # 12-second segment, no punctuation — should split into ~2 chunks
        words = [_w(f"w{i}", i * 1.0, (i + 1) * 1.0) for i in range(12)]
        seg = _seg(" ".join(f"w{i}" for i in range(12)), 0.0, 12.0)
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=40.0)
        assert len(result) >= 2
        # All chunks within [0, 12]
        assert result[0]["start"] == 0.0
        assert result[-1]["end"] == 12.0

    def test_time_split_timestamps_are_contiguous(self):
        words = [_w(f"w{i}", i * 1.0, (i + 1) * 1.0) for i in range(15)]
        seg = _seg(" ".join(f"w{i}" for i in range(15)), 0.0, 15.0)
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=40.0)
        # No gaps — each chunk's end <= next chunk's start (word boundary rounding allows small overlap)
        for i in range(len(result) - 1):
            assert result[i]["end"] <= result[i + 1]["start"] + 0.15

    # ── timing correctness ────────────────────────────────────────────────────

    def test_first_chunk_uses_original_seg_start(self):
        words = [
            _w("A", 0.1, 0.5), _w("B", 0.5, 1.0), _w("。", 1.0, 1.1),
            _w("C", 7.0, 7.5), _w("D", 7.5, 8.0),
        ]
        seg = _seg("AB。CD", 0.0, 14.0)  # seg start=0.0 but first word at 0.1
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=40.0)
        # First chunk must use the segment's original start (0.0), not word start (0.1)
        assert result[0]["start"] == 0.0

    def test_last_chunk_uses_original_seg_end(self):
        words = [
            _w("A", 0.0, 0.5), _w("B", 0.5, 1.0), _w("。", 1.0, 1.1),
            _w("C", 7.0, 7.5), _w("D", 7.5, 7.9),  # word ends at 7.9 but seg ends at 8.0
        ]
        seg = _seg("AB。CD", 0.0, 8.0)
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=40.0)
        assert result[-1]["end"] == 8.0

    # ── em-width splitting ────────────────────────────────────────────────────

    def test_wide_text_triggers_split_even_if_short_duration(self):
        # 5-second segment but text is very wide (> 40 em of full-width CJK chars)
        long_text = "あ" * 50   # 50 full-width chars = 50 em >> 40 em threshold
        words = [_w("あ" * 10, i * 1.0, (i + 1) * 1.0) for i in range(5)]
        seg = _seg(long_text, 0.0, 5.0)
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=40.0)
        # Should be split (even though duration is only 5 s)
        assert len(result) >= 2

    def test_max_em_controls_split_threshold(self):
        # With a very large max_em, no split occurs even for wide text
        long_text = "あ" * 20   # 20 em — fits within max_em=100
        words = [_w("あ", i * 0.2, (i + 1) * 0.2) for i in range(20)]
        seg = _seg(long_text, 0.0, 5.0)
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=100.0)
        assert len(result) == 1

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_single_word_segment_not_split(self):
        words = [_w("一つ", 0.0, 0.5)]
        seg = _seg("一つ", 0.0, 10.0)
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=40.0)
        # Only one word — can't split; falls through as-is
        assert len(result) == 1

    def test_multiple_segments_each_processed_independently(self):
        segs = [
            _seg("短い", 0.0, 2.0),
            _seg("これは長い文章です。次もあります。", 3.0, 17.0),
        ]
        words = (
            [_w("短", 0.0, 1.0), _w("い", 1.0, 2.0)] +
            [_w(c, 3.0 + i * 0.5, 3.0 + (i + 1) * 0.5) for i, c in enumerate("これは長い文章です。次もあります。")]
        )
        result = split_telop_segments(segs, words, max_duration=6.0, max_em=40.0)
        # First seg stays, second seg is split
        assert result[0]["text"] == "短い"
        assert len(result) >= 3

    def test_zero_duration_chunk_skipped(self):
        # Degenerate: word start == word end at a sentence boundary
        words = [
            _w("A", 0.0, 0.5), _w("。", 0.5, 0.5),   # zero-duration boundary word
            _w("B", 5.0, 5.5),
        ]
        seg = _seg("A。B", 0.0, 10.0)
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=40.0)
        # Should not crash; zero-duration chunks are skipped
        assert all(r["end"] > r["start"] for r in result)

    def test_output_is_in_chronological_order(self):
        words = [_w(f"w{i}", i * 0.5, (i + 1) * 0.5) for i in range(20)]
        seg = _seg(" ".join(f"w{i}" for i in range(20)), 0.0, 10.0)
        result = split_telop_segments([seg], words, max_duration=6.0, max_em=40.0)
        starts = [r["start"] for r in result]
        assert starts == sorted(starts)

    def test_grouped_words_text_key_accepted(self):
        """Words with 'text' key (group_word_tokens output) work in split logic."""
        # group_word_tokens produces dicts with 'text' not 'word'; _join_word_texts
        # must handle both keys so that a caller who pre-groups tokens can still
        # pass grouped words to split_telop_segments without silent empty-text bugs.
        # Simulate: "えーと" was grouped from 3 CJK tokens into one dict with 'text'.
        words_with_text_key = [
            {"text": "えーと", "start": 0.0, "end": 0.4},
            {"text": "これは", "start": 0.45, "end": 0.8},
            {"text": "テスト", "start": 0.85, "end": 1.2},
        ]
        seg = _seg("えーとこれはテスト", 0.0, 10.0)
        result = split_telop_segments([seg], words_with_text_key, max_duration=6.0, max_em=40.0)
        # Should not produce empty-text segments
        assert all(r.get("text") for r in result)
        # Text should contain the grouped word content
        combined = "".join(r["text"] for r in result)
        assert "えーと" in combined or "えーとこれはテスト" == combined


# ── _merge_short_adjacent / split_telop_segments merge behaviour ─────────────

class TestMergeShortAdjacent:
    """Unit tests for the micro-segment merge pre-pass inside split_telop_segments.

    Whisper sometimes puts a segment boundary mid-word ("フリー" | "ランス")
    or at a semantically meaningless cut ("２０２６" | "年に退職して").
    _merge_short_adjacent must stitch these back together before splitting.
    """

    def _segs(self, *items):
        """Build segments from (text, start, end) tuples."""
        return [_seg(t, s, e) for t, s, e in items]

    def test_katakana_mid_word_merged(self):
        """'フリー' (0.5s) | gap 0ms | 'ランスのコンサルタント' → merged."""
        segs = self._segs(
            ("退職後は独立してフリー",   0.0,  2.5),   # ends with short 0.5s part
            ("ランスのコンサルタントと", 2.5,  6.0),
        )
        # Make segment 0 short so it triggers merge.
        segs[0]["end"] = 1.0   # 1.0s duration → < _MERGE_MIN_DUR (1.0s limit: equal, not <)
        segs[0]["start"] = 0.0
        # Adjust: use a segment clearly short: 0.4s
        segs = [
            _seg("フリー",                    0.0, 0.4),   # 0.4s < 1.0 → candidate
            _seg("ランスのコンサルタントと",  0.4, 4.0),
        ]
        result = split_telop_segments(segs, [], max_duration=6.0, max_em=40.0)
        assert len(result) == 1, f"Expected 1 merged card, got {len(result)}: {[r['text'] for r in result]}"
        assert result[0]["text"] == "フリーランスのコンサルタントと"
        assert result[0]["start"] == 0.0
        assert result[0]["end"]   == pytest.approx(4.0)

    def test_numeric_year_merged(self):
        """'私は２０２６' (0.3s) | gap 0ms | '年に退職して' → merged."""
        segs = [
            _seg("私は２０２６",      0.0, 0.3),   # 0.3s < 1.0 → candidate
            _seg("年に退職して",      0.3, 3.0),
        ]
        result = split_telop_segments(segs, [], max_duration=6.0, max_em=40.0)
        assert len(result) == 1
        assert result[0]["text"] == "私は２０２６年に退職して"

    def test_sentence_end_prevents_merge(self):
        """Segment ending with 。 must never be merged even when short."""
        segs = [
            _seg("ありがとう。",  0.0, 0.5),   # 0.5s, ends with 。
            _seg("次の話題です。", 0.5, 4.0),
        ]
        result = split_telop_segments(segs, [], max_duration=6.0, max_em=40.0)
        assert len(result) == 2, "Sentence-ending segment must not be merged"
        assert result[0]["text"] == "ありがとう。"
        assert result[1]["text"] == "次の話題です。"

    def test_large_gap_prevents_merge(self):
        """Gap > 200ms between segments: do not merge even if segments are short."""
        segs = [
            _seg("えー",     0.0, 0.5),   # 0.5s short
            _seg("次の話題", 0.8, 3.5),   # gap = 0.3s > _MERGE_MAX_GAP
        ]
        result = split_telop_segments(segs, [], max_duration=6.0, max_em=40.0)
        assert len(result) == 2, "Large gap must prevent merge"

    def test_combined_too_long_prevents_merge(self):
        """Combined text exceeding max_em: do not merge."""
        # 40 em ≈ 40 full-width chars; make combined text > 40 chars
        long_next = "あ" * 38   # 38 em alone is fine; combined with prev makes 41+
        segs = [
            _seg("あああ",  0.0, 0.5),   # 3 em
            _seg(long_next, 0.5, 4.0),   # 38 em → combined 41 em > 40
        ]
        result = split_telop_segments(segs, [], max_duration=6.0, max_em=40.0)
        assert len(result) == 2, "Over-limit combined text must not be merged"

    def test_both_segments_normal_length_not_merged(self):
        """Both segments ≥ 1s: leave them as-is."""
        segs = [
            _seg("これは普通の長さのセグメントです",  0.0, 2.0),   # 2.0s ≥ 1.0 → not short
            _seg("次のセグメントも普通の長さです",   2.0, 4.5),   # 2.5s ≥ 1.0 → not short
        ]
        result = split_telop_segments(segs, [], max_duration=6.0, max_em=40.0)
        assert len(result) == 2, "Normal-length segments must not be merged"

    def test_english_mid_word_merged_with_space(self):
        """Latin mid-word split: 'free' + 'lance' → 'free lance' (space inserted)."""
        segs = [
            _seg("free",  0.0, 0.3),
            _seg("lance", 0.3, 2.0),
        ]
        result = split_telop_segments(segs, [], max_duration=6.0, max_em=40.0)
        assert len(result) == 1
        assert result[0]["text"] == "free lance"

    def test_three_short_segments_all_merged(self):
        """Chained cascade: A(0.3s)+B(0.4s)+C(0.3s) all within gap → one card."""
        segs = [
            _seg("フリー",    0.0, 0.3),   # 0.3s < 1.0
            _seg("ランス",    0.3, 0.7),   # 0.4s < 1.0 — merges with AB
            _seg("のコンサル", 0.7, 1.0),  # 0.3s < 1.0 — merges into ABC
        ]
        result = split_telop_segments(segs, [], max_duration=6.0, max_em=40.0)
        assert len(result) == 1, f"Expected 1 merged card, got {len(result)}: {[r['text'] for r in result]}"
        assert result[0]["text"] == "フリーランスのコンサル"
        assert result[0]["start"] == pytest.approx(0.0)
        assert result[0]["end"]   == pytest.approx(1.0)

    def test_exactly_1s_duration_is_not_short(self):
        """A 1.0s segment is NOT below _MERGE_MIN_DUR (strictly <) — must not merge."""
        segs = [
            _seg("これは",   0.0, 1.0),   # exactly 1.0s — not < _MERGE_MIN_DUR
            _seg("テストです", 1.0, 3.0),
        ]
        result = split_telop_segments(segs, [], max_duration=6.0, max_em=40.0)
        assert len(result) == 2, "Exactly 1.0s segment must not trigger merge"


# ── group_word_tokens() ───────────────────────────────────────────────────────

def _wt(text, start, end, confidence=None):
    """Minimal word dict for group_word_tokens tests (uses 'text' key)."""
    return {"id": "", "word": text, "text": text, "start": start, "end": end,
            "seg_id": None, "confidence": confidence}


class TestGroupWordTokens:

    def test_empty_list_returns_empty(self):
        assert group_word_tokens([]) == []

    def test_english_words_pass_through_unchanged(self):
        words = [_wt("Hello", 0.0, 0.4), _wt("world", 0.5, 0.9)]
        result = group_word_tokens(words)
        assert len(result) == 2
        assert result[0]["text"] == "Hello"
        assert result[1]["text"] == "world"

    def test_cjk_chars_merged_within_gap(self):
        # 今 日 本 — three kanji tokens within 100ms gap
        words = [_wt("今", 0.0, 0.1), _wt("日", 0.11, 0.2), _wt("本", 0.21, 0.3)]
        result = group_word_tokens(words, max_gap_s=0.10)
        assert len(result) == 1
        assert result[0]["text"] == "今日本"
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 0.3

    def test_cjk_split_by_large_gap(self):
        # Large gap (>100ms) between kanji tokens should split groups
        words = [_wt("今", 0.0, 0.1), _wt("日", 0.3, 0.4), _wt("本", 0.41, 0.5)]
        result = group_word_tokens(words, max_gap_s=0.10)
        assert len(result) == 2
        assert result[0]["text"] == "今"
        assert result[1]["text"] == "日本"

    def test_hiragana_not_merged(self):
        """Hiragana single characters must NOT be treated as mergeable.

        Before this fix, east_asian_width="W" caused hiragana to merge with adjacent
        kanji, collapsing phrases like "もう一つ余談なんですけれども一回私" into one span.
        Clicking any character in that span sought to the group start, not the clicked word.
        """
        words = [_wt("も", 0.0, 0.02), _wt("う", 0.02, 0.04), _wt("一", 0.05, 0.07)]
        result = group_word_tokens(words)
        # "も" and "う" must NOT merge with "一" (or each other)
        assert len(result) == 3, "hiragana must pass through as individual tokens"
        assert result[2]["text"] == "一"

    def test_mixed_cjk_and_latin(self):
        # "今日 is nice" — 今 日 are CJK, " is", " nice" are Latin
        words = [
            _wt("今", 0.0, 0.1), _wt("日", 0.11, 0.2),
            _wt("is",  0.5, 0.7), _wt("nice", 0.8, 1.0),
        ]
        result = group_word_tokens(words, max_gap_s=0.10)
        assert len(result) == 3
        assert result[0]["text"] == "今日"
        assert result[1]["text"] == "is"
        assert result[2]["text"] == "nice"

    def test_confidence_takes_minimum(self):
        words = [
            _wt("今", 0.0, 0.1, confidence=0.9),
            _wt("日", 0.11, 0.2, confidence=0.4),
            _wt("本", 0.21, 0.3, confidence=0.8),
        ]
        result = group_word_tokens(words)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.4

    def test_single_cjk_token_not_artificially_grouped(self):
        # A lone CJK character with a large gap before the next one stays alone
        words = [_wt("日", 0.0, 0.1), _wt("本", 0.5, 0.6)]
        result = group_word_tokens(words, max_gap_s=0.10)
        assert len(result) == 2

    def test_multi_char_cjk_token_passes_through(self):
        # WhisperX may produce grouped tokens already; these must not be re-split
        words = [_wt("今日", 0.0, 0.3), _wt("は", 0.31, 0.4)]
        result = group_word_tokens(words, max_gap_s=0.10)
        # "今日" (len=2) is not single-CJK — passes through unchanged
        assert result[0]["text"] == "今日"
        assert result[1]["text"] == "は"

    def test_japanese_punctuation_not_merged(self):
        # 、。「」 have east_asian_width "W" but are punctuation — must not be merged
        words = [
            _wt("今", 0.0, 0.1), _wt("日", 0.11, 0.2),
            _wt("、", 0.21, 0.22),   # punctuation: should break the group
            _wt("は", 0.23, 0.33),
        ]
        result = group_word_tokens(words, max_gap_s=0.10)
        assert len(result) == 3
        assert result[0]["text"] == "今日"
        assert result[1]["text"] == "、"
        assert result[2]["text"] == "は"

    def test_exclamation_not_merged(self):
        # ！ is fullwidth punctuation (U+FF01) — must not be merged with adjacent kanji
        words = [_wt("凄", 0.0, 0.1), _wt("技", 0.11, 0.2), _wt("！", 0.21, 0.25)]
        result = group_word_tokens(words, max_gap_s=0.10)
        assert len(result) == 2
        assert result[0]["text"] == "凄技"
        assert result[1]["text"] == "！"

    def test_katakana_prolonged_mark_merged(self):
        # ー (U+30FC, east_asian_width W, category Lm) must merge with adjacent kana.
        # faster-whisper may emit コ + ー + ヒ + ー as four separate tokens.
        words = [_wt("コ", 0.0, 0.08), _wt("ー", 0.08, 0.15),
                 _wt("ヒ", 0.16, 0.22), _wt("ー", 0.22, 0.28)]
        result = group_word_tokens(words, max_gap_s=0.10)
        assert len(result) == 1
        assert result[0]["text"] == "コーヒー"

    def test_boundary_timestamps_preserved(self):
        """Grouped word uses first token's start and last token's end."""
        words = [_wt("修", 1.0, 1.1), _wt("正", 1.1, 1.2), _wt("案", 1.2, 1.3), _wt("作", 1.3, 1.4), _wt("成", 1.4, 1.5)]
        result = group_word_tokens(words)
        assert len(result) == 1
        assert result[0]["start"] == pytest.approx(1.0)
        assert result[0]["end"]   == pytest.approx(1.5)

    def test_score_key_used_when_confidence_absent(self):
        """faster-whisper uses 'score' key; group_word_tokens must pick it up."""
        # Words with "score" instead of "confidence" (faster-whisper format)
        words = [
            {"id": "", "word": "今", "start": 0.0, "end": 0.1, "seg_id": None, "score": 0.95},
            {"id": "", "word": "日", "start": 0.1, "end": 0.2, "seg_id": None, "score": 0.92},
            {"id": "", "word": "本", "start": 0.2, "end": 0.3, "seg_id": None, "score": 0.40},
        ]
        result = group_word_tokens(words)
        assert len(result) == 1
        # confidence = min(0.95, 0.92, 0.40) = 0.40
        assert result[0]["confidence"] == pytest.approx(0.40)

    def test_confidence_preferred_over_score(self):
        """When both keys are present, 'confidence' takes priority."""
        words = [
            {"id": "", "word": "コ", "start": 0.0, "end": 0.08, "seg_id": None,
             "confidence": 0.80, "score": 0.60},
            {"id": "", "word": "ー", "start": 0.08, "end": 0.15, "seg_id": None,
             "confidence": 0.70, "score": 0.90},
        ]
        result = group_word_tokens(words)
        assert len(result) == 1
        # confidence key wins; min(0.80, 0.70) = 0.70
        assert result[0]["confidence"] == pytest.approx(0.70)

    def test_zero_confidence_kept_not_treated_as_absent(self):
        """confidence=0.0 is falsy but present — must not be swapped for 'score'.

        The guard uses `is not None` (not a truthiness check) so 0.0 is correctly
        used even when a 'score' key is also present.
        """
        words = [
            {"id": "", "word": "今", "start": 0.0, "end": 0.1, "seg_id": None,
             "confidence": 0.0, "score": 0.9},
            {"id": "", "word": "日", "start": 0.1, "end": 0.2, "seg_id": None,
             "confidence": 0.5, "score": 0.8},
        ]
        result = group_word_tokens(words)
        assert len(result) == 1
        # confidence wins (is not None), min(0.0, 0.5) = 0.0
        assert result[0]["confidence"] == pytest.approx(0.0)

    def test_seg_id_boundary_never_merged(self):
        """Tokens from different Whisper segments must never be merged, even < 50ms apart.

        Whisper segment boundaries are reliable phrase delimiters.  The real-world
        failure was: 'る範囲で修正案を書' — characters from two separate Whisper
        segments ('できる範囲で' and '修正案を書いて') merged into one chunk because
        the inter-phrase gap was < 100ms.  With the seg_id hard break this cannot
        happen regardless of the time gap.

        Both tokens must be kanji (single-CJK) for the seg_id check to apply;
        hiragana tokens pass through without merging regardless.
        """
        words = [
            {"id": "", "word": "範", "start": 1.0, "end": 1.02,
             "seg_id": "s0", "confidence": None},
            {"id": "", "word": "修", "start": 1.04, "end": 1.06,
             "seg_id": "s1", "confidence": None},  # different segment, only 20ms gap
        ]
        result = group_word_tokens(words)
        assert len(result) == 2, "tokens from different seg_ids must not be merged"
        assert result[0]["text"] == "範"
        assert result[1]["text"] == "修"

    def test_same_seg_id_still_merged_within_gap(self):
        """Tokens with the same seg_id and gap < 50ms are still merged normally."""
        words = [
            {"id": "", "word": "今", "start": 0.0,  "end": 0.05,
             "seg_id": "s0", "confidence": None},
            {"id": "", "word": "日", "start": 0.08, "end": 0.13,
             "seg_id": "s0", "confidence": None},  # same seg, 30ms gap — should merge
        ]
        result = group_word_tokens(words)
        assert len(result) == 1
        assert result[0]["text"] == "今日"

    def test_none_seg_id_falls_back_to_gap_only(self):
        """When seg_id is None on either token, fall back to gap threshold."""
        # seg_id=None means seg_id was not assigned — use gap (50ms default)
        words = [
            {"id": "", "word": "今", "start": 0.0, "end": 0.02,
             "seg_id": None, "confidence": None},
            {"id": "", "word": "日", "start": 0.04, "end": 0.06,
             "seg_id": None, "confidence": None},  # 20ms gap, no seg_id → merge
        ]
        result = group_word_tokens(words)
        assert len(result) == 1
        assert result[0]["text"] == "今日"

    # ── Latin / digit grouping ────────────────────────────────────────────────

    def test_single_char_ascii_letters_grouped(self):
        """Character-level ASCII tokens for company names are merged — G,o,o,g,l,e → Google."""
        words = [_wt(c, i * 0.02, (i + 1) * 0.02) for i, c in enumerate("Google")]
        result = group_word_tokens(words, max_gap_s=0.05)
        assert len(result) == 1
        assert result[0]["text"] == "Google"
        assert result[0]["start"] == pytest.approx(0.0)
        assert result[0]["end"]   == pytest.approx(0.12)

    def test_single_char_digits_grouped(self):
        """Single-char digit tokens are merged — '1','7' → '17'."""
        words = [_wt("1", 0.0, 0.05), _wt("7", 0.05, 0.10)]
        result = group_word_tokens(words, max_gap_s=0.05)
        assert len(result) == 1
        assert result[0]["text"] == "17"

    def test_single_char_alnum_split_by_large_gap(self):
        """A gap > max_gap_s breaks a Latin/digit group."""
        # 'A' then 'I' but 200ms apart — they must NOT merge
        words = [_wt("A", 0.0, 0.05), _wt("I", 0.25, 0.30)]
        result = group_word_tokens(words, max_gap_s=0.05)
        assert len(result) == 2
        assert result[0]["text"] == "A"
        assert result[1]["text"] == "I"

    def test_multichar_latin_token_not_affected(self):
        """Multi-char tokens ('Hello') are NOT touched by the single-char grouping."""
        words = [_wt("Hello", 0.0, 0.4), _wt("world", 0.5, 0.9)]
        result = group_word_tokens(words)
        assert len(result) == 2
        assert result[0]["text"] == "Hello"
        assert result[1]["text"] == "world"

    def test_fullwidth_letters_grouped_and_normalised(self):
        """Fullwidth letter tokens (Ａ-Ｚ) are grouped AND normalised to ASCII."""
        words = [_wt("Ａ", 0.0, 0.02), _wt("Ｉ", 0.02, 0.04)]
        result = group_word_tokens(words, max_gap_s=0.05)
        assert len(result) == 1
        assert result[0]["text"] == "AI"   # normalised to halfwidth

    def test_fullwidth_amazon_grouped_and_normalised(self):
        """ＡＭＡＺＯＮ as individual fullwidth tokens groups and normalises to AMAZON."""
        chars = list("ＡＭＡＺＯＮ")
        words = [_wt(c, i * 0.02, (i + 1) * 0.02) for i, c in enumerate(chars)]
        result = group_word_tokens(words, max_gap_s=0.05)
        assert len(result) == 1
        assert result[0]["text"] == "AMAZON"

    def test_fullwidth_digits_grouped_and_normalised(self):
        """Fullwidth digit tokens (０-９) are grouped and normalised."""
        words = [_wt("２", 0.0, 0.05), _wt("０", 0.05, 0.10),
                 _wt("２", 0.10, 0.15), _wt("６", 0.15, 0.20)]
        result = group_word_tokens(words, max_gap_s=0.05)
        assert len(result) == 1
        assert result[0]["text"] == "2026"

    def test_alnum_group_adjacent_to_cjk(self):
        """Single-char alnum group followed by CJK group — both handled correctly."""
        # '1','7' then '年','に' (kanji+hiragana)
        words = [
            _wt("1", 0.0, 0.02), _wt("7", 0.02, 0.04),
            _wt("年", 0.05, 0.10), _wt("に", 0.11, 0.15),
        ]
        result = group_word_tokens(words, max_gap_s=0.05)
        # '1','7' → "17"; '年' groups as kanji; 'に' is hiragana → passthrough
        assert result[0]["text"] == "17"
        assert result[1]["text"] == "年"
        assert result[2]["text"] == "に"


# ── assign_words_to_entries() ─────────────────────────────────────────────────

def _ent(id_, start, end):
    return {"id": id_, "start": start, "end": end, "text": ""}

def _wrd(start, end, text="x"):
    return {"text": text, "start": start, "end": end}

class TestAssignWordsToEntries:
    """assign_words_to_entries assigns each word to the telop entry whose
    start time is closest (within ±tolerance), fixing the 'first match'
    boundary bug that assigned first-words of later entries to earlier ones.
    """

    def test_word_entirely_inside_entry_assigned_to_it(self):
        words = [_wrd(1.0, 1.5)]
        entries = [_ent("t0", 0.0, 2.0)]
        result = assign_words_to_entries(words, entries)
        assert result == ["t0"]

    def test_two_entries_mid_word_goes_to_first(self):
        """Word equidistant from two entry starts goes to the first (iteration order)."""
        words = [_wrd(0.5, 0.8)]
        entries = [_ent("t0", 0.0, 1.0), _ent("t1", 1.0, 2.0)]
        result = assign_words_to_entries(words, entries)
        # dist(t0)=|0.5-0.0|=0.5, dist(t1)=|0.5-1.0|=0.5 — equal distance;
        # t0 wins because it appears first in the ordered entry list.
        assert result == ["t0"]

    def test_boundary_word_goes_to_later_entry_not_earlier(self):
        """Core regression: a word at exactly t1.start must go to t1, not t0.

        With the old 'first match' logic, a word at T=1.0 would match t0
        (end=1.0 → 1.0 < 1.0+0.15=1.15) before reaching t1.  closest-start
        gives dist(t0)=|1.0-0.0|=1.0 vs dist(t1)=|1.0-1.0|=0.0 → t1 wins.
        """
        words = [_wrd(1.0, 1.3)]
        entries = [_ent("t0", 0.0, 1.0), _ent("t1", 1.0, 2.0)]
        result = assign_words_to_entries(words, entries)
        assert result == ["t1"], (
            "Word starting at entry boundary must be assigned to the LATER entry "
            "(closest-start wins over first-match)"
        )

    def test_word_close_to_next_entry_start_assigned_to_next_entry(self):
        """Word at 0.92 is physically inside t0 but only 0.08s from t1's start.

        Closest-start gives t1: dist(t0)=0.92, dist(t1)=0.08.  This is correct —
        Whisper timestamp jitter means the word may actually belong to t1.
        """
        words = [_wrd(0.92, 1.1)]
        entries = [_ent("t0", 0.0, 1.0), _ent("t1", 1.0, 2.0)]
        result = assign_words_to_entries(words, entries)
        assert result == ["t1"]

    def test_word_clearly_inside_first_entry_assigned_to_it(self):
        """Word at 0.4 with t0=(0,1) and t1=(1,2): dist=0.4 vs 0.6 → t0 wins."""
        words = [_wrd(0.4, 0.7)]
        entries = [_ent("t0", 0.0, 1.0), _ent("t1", 1.0, 2.0)]
        result = assign_words_to_entries(words, entries)
        assert result == ["t0"]

    def test_word_outside_all_entries_falls_back_to_nearest(self):
        """Word start 5.0 with no entry within tolerance falls back to nearest entry.

        Fix 4: instead of returning None, assign to the nearest entry by start time.
        This ensures every word gets a seg_id so the frontend can always seek to it.
        With entries t0(start=0) and t1(start=1): |5.0-1.0|=4.0 < |5.0-0.0|=5.0 → t1.
        """
        words = [_wrd(5.0, 5.3)]
        entries = [_ent("t0", 0.0, 1.0), _ent("t1", 1.0, 2.0)]
        result = assign_words_to_entries(words, entries)
        assert result == ["t1"]

    def test_empty_words_returns_empty_list(self):
        entries = [_ent("t0", 0.0, 2.0)]
        assert assign_words_to_entries([], entries) == []

    def test_empty_entries_all_none(self):
        words = [_wrd(1.0, 1.5), _wrd(2.0, 2.5)]
        assert assign_words_to_entries(words, []) == [None, None]

    def test_three_entries_each_word_to_correct_entry(self):
        words = [_wrd(0.3, 0.6), _wrd(1.1, 1.5), _wrd(2.2, 2.7)]
        entries = [_ent("t0", 0.0, 1.0), _ent("t1", 1.0, 2.0), _ent("t2", 2.0, 3.0)]
        result = assign_words_to_entries(words, entries)
        assert result == ["t0", "t1", "t2"]

    def test_multiple_boundary_words_each_to_later_entry(self):
        """Multiple words at adjacent entry boundaries each go to the correct entry."""
        words = [_wrd(0.0, 0.4), _wrd(1.0, 1.4), _wrd(2.0, 2.4)]
        entries = [_ent("t0", 0.0, 1.0), _ent("t1", 1.0, 2.0), _ent("t2", 2.0, 3.0)]
        result = assign_words_to_entries(words, entries)
        # word at 0.0: dist(t0)=0.0 → t0; word at 1.0: dist(t1)=0.0 → t1; etc.
        assert result == ["t0", "t1", "t2"]

    def test_custom_tolerance_respected(self):
        """With tolerance=0.10, a word 0.12s before entry start is not matched."""
        words = [_wrd(0.88, 1.0)]
        entries = [_ent("t0", 0.0, 1.0), _ent("t1", 1.0, 2.0)]
        # With tolerance=0.10: t1 window is [0.9, 2.1) → 0.88 < 0.9 → t1 excluded
        # t0 window is [-0.10, 1.1) → 0.88 ∈ [-0.10, 1.1) → t0 matches
        result = assign_words_to_entries(words, entries, tolerance=0.10)
        assert result == ["t0"]

    def test_far_outside_word_assigned_to_nearest_not_none(self):
        """Fix 4: word far past all entries still gets a seg_id (nearest fallback).

        A word at 10.0 with a single entry ending at 2.0 would return None with the
        old logic.  After Fix 4 it falls back to the nearest entry regardless of distance.
        """
        words = [_wrd(10.0, 10.3)]
        entries = [_ent("t0", 0.0, 2.0)]
        result = assign_words_to_entries(words, entries)
        assert result == ["t0"], (
            "Fix 4: a word beyond all entry windows should fall back to the nearest "
            "entry rather than returning None"
        )

    def test_multiple_far_words_all_get_nearest_entry(self):
        """Fix 4: multiple out-of-tolerance words each get the nearest entry."""
        words = [_wrd(50.0, 50.5), _wrd(51.0, 51.5)]
        entries = [_ent("t0", 0.0, 5.0), _ent("t1", 5.0, 10.0)]
        result = assign_words_to_entries(words, entries)
        # Both words are far past t1 (start=5.0); nearest is t1 for both.
        assert result == ["t1", "t1"], (
            f"Fix 4: far words should both fall back to t1 (nearest); got {result}"
        )


class TestPhraseSplitDepthGuard:
    """T0289: _build_chunks recursion is bounded (depth < 3). Degenerate oversized
    input must terminate and the chunk must be returned intact — never mid-word cut,
    never truncated, never lost."""

    @staticmethod
    def _words(n, ch="あ"):
        return [{"word": ch, "start": float(i), "end": float(i) + 1.0, "seg_id": 0}
                for i in range(n)]

    def test_depth_limit_returns_oversized_chunk_intact(self):
        from preprod.segments import _build_chunks
        words = self._words(6)
        seg = {"start": 0.0, "end": 10.0, "text": "あ" * 6, "seg_id": 0}
        # Oversized (dur 10 >> max_dur; em 6 > max_em) AND splittable (>2 words),
        # so only the depth guard can stop recursion at depth 3.
        out = _build_chunks(seg, words, [], max_dur=0.5, max_em=2.0, depth=3)
        assert len(out) == 1                        # not recursed past the guard
        assert out[0]["text"] == "あ" * 6           # intact — not truncated/cut
        assert out[0]["start"] == 0.0 and out[0]["end"] == 10.0

    def test_recursion_from_top_terminates_without_losing_text(self):
        from preprod.segments import _build_chunks
        words = self._words(6)
        seg = {"start": 0.0, "end": 10.0, "text": "あ" * 6, "seg_id": 0}
        # From depth 0 this recurses; it must terminate (pytest-timeout guards
        # infinite recursion) and preserve every character across the chunks.
        out = _build_chunks(seg, words, [], max_dur=0.5, max_em=2.0, depth=0)
        assert out                                  # non-empty
        assert "".join(c["text"] for c in out) == "あ" * 6   # no loss, no mid-word drop
