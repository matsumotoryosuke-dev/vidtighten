"""Tests for fcpxml_telop.py — generate_telop_fcpxml()."""

import pytest
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from preprod.segments import build_segments
from preprod.fcpxml_telop import (
    generate_telop_fcpxml,
    _to_rt,
    _hex_to_fcpxml_color,
    _clean_telop_text,
    _char_em,
    _em_width,
    _break_score,
    _wrap_telop,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _default_settings():
    return {
        "fps": "24",
        "width": 1920,
        "height": 1080,
        "font": "Helvetica",
        "font_size": 80,
        "font_color": "#FFFFFF",
    }


def _make_entries(raw):
    return [{"id": f"t{i}", "start": s, "end": e, "text": t}
            for i, (s, e, t) in enumerate(raw)]


def _write_and_parse(tmp_path, entries, segs, settings=None, duration=30.0, stem="test"):
    settings = settings or _default_settings()
    out = tmp_path / "out.fcpxml"
    generate_telop_fcpxml(entries, segs, duration, settings, stem, out)
    return out, ET.parse(out).getroot()


def _rt_to_seconds(rt: str) -> float:
    """Parse a FCP rational-time string ("N/Ds" or "Ns") → float seconds."""
    raw = rt.rstrip("s")
    if "/" in raw:
        n, d = raw.split("/")
        return float(n) / float(d)
    return float(raw)


# ── XML structure ─────────────────────────────────────────────────────────────

class TestFcpxmlStructure:
    def test_output_file_created(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Hello")])
        out, root = _write_and_parse(tmp_path, entries, segs)
        assert out.exists()

    def test_root_element_is_fcpxml(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        assert root.tag == "fcpxml"

    def test_fcpxml_version_attribute(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        assert root.get("version") == "1.11"

    def test_resources_present(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        assert root.find("resources") is not None

    def test_format_element_present(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        fmt = root.find(".//format")
        assert fmt is not None
        assert fmt.get("id") == "r1"

    def test_format_width_height(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        settings = _default_settings()
        settings["width"] = 3840
        settings["height"] = 2160
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs, settings)
        fmt = root.find(".//format")
        assert fmt.get("width") == "3840"
        assert fmt.get("height") == "2160"

    def test_format_name_24fps(self, tmp_path):
        """1920×1080 must use height-only name (FFVideoFormat1080p24), not WxH form."""
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        assert root.find(".//format").get("name") == "FFVideoFormat1080p24"

    def test_format_name_23976fps(self, tmp_path):
        """1080p 23.976fps: height-only name + '2398' fps-id."""
        segs = build_segments([], total_duration=30.0)
        s = _default_settings()
        s["fps"] = "23.976"
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs, s)
        assert root.find(".//format").get("name") == "FFVideoFormat1080p2398"

    def test_format_name_2997fps(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        s = _default_settings()
        s["fps"] = "29.97"
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs, s)
        assert root.find(".//format").get("name") == "FFVideoFormat1080p2997"

    def test_format_name_5994fps(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        s = _default_settings()
        s["fps"] = "59.94"
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs, s)
        assert root.find(".//format").get("name") == "FFVideoFormat1080p5994"

    def test_format_name_4k_23976fps(self, tmp_path):
        """Regression: 4K 23.976fps must use '2398' in format name, not '23976'."""
        segs = build_segments([], total_duration=30.0)
        s = _default_settings()
        s["fps"] = "23.976"; s["width"] = 3840; s["height"] = 2160
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs, s)
        assert root.find(".//format").get("name") == "FFVideoFormat3840x2160p2398"

    def test_format_name_25fps_pal(self, tmp_path):
        """25fps (PAL): not in _FPS_ID dict → integer str fallback → 'p25'."""
        segs = build_segments([], total_duration=30.0)
        s = _default_settings()
        s["fps"] = "25"
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs, s)
        assert root.find(".//format").get("name") == "FFVideoFormat1080p25"

    def test_format_name_50fps_pal(self, tmp_path):
        """50fps PAL HD: integer str fallback → 'p50'."""
        segs = build_segments([], total_duration=30.0)
        s = _default_settings()
        s["fps"] = "50"
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs, s)
        assert root.find(".//format").get("name") == "FFVideoFormat1080p50"

    def test_format_name_vertical_9x16(self, tmp_path):
        """Vertical 9:16 (1080×1920): not in FCP_FORMAT_PREFIX → height-only fallback
        → 'FFVideoFormat1920p30'. FCP creates a Custom sequence from width/height attrs."""
        segs = build_segments([], total_duration=30.0)
        s = {**_default_settings(), "fps": "30", "width": 1080, "height": 1920}
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "テスト")]), segs, s)
        assert root.find(".//format").get("name") == "FFVideoFormat1920p30"
        # Width and height attributes must also be correct so FCP can fall back to them
        fmt = root.find(".//format")
        assert fmt.get("width") == "1080"
        assert fmt.get("height") == "1920"

    def test_format_name_4x5(self, tmp_path):
        """4:5 portrait (1080×1350): height-only fallback → 'FFVideoFormat1350p30'."""
        segs = build_segments([], total_duration=30.0)
        s = {**_default_settings(), "fps": "30", "width": 1080, "height": 1350}
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "テスト")]), segs, s)
        assert root.find(".//format").get("name") == "FFVideoFormat1350p30"
        fmt = root.find(".//format")
        assert fmt.get("width") == "1080"
        assert fmt.get("height") == "1350"

    def test_format_frame_duration_24fps(self, tmp_path):
        """frameDuration must match the fps — FCP uses it to snap all time values."""
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        assert root.find(".//format").get("frameDuration") == "1/24s"

    def test_format_frame_duration_23976fps(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        s = _default_settings(); s["fps"] = "23.976"
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs, s)
        assert root.find(".//format").get("frameDuration") == "1001/24000s"

    def test_format_frame_duration_2997fps(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        s = _default_settings(); s["fps"] = "29.97"
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs, s)
        assert root.find(".//format").get("frameDuration") == "1001/30000s"

    def test_format_frame_duration_30fps(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        s = _default_settings(); s["fps"] = "30"
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs, s)
        assert root.find(".//format").get("frameDuration") == "1/30s"

    def test_unknown_fps_falls_back_to_24fps_frame_duration(self, tmp_path):
        """_frame_dur falls back to Fraction(100, 2400) = 1/24s for unknown fps strings.

        Passing an unrecognised fps string (e.g. '15') must not crash; the
        FCPXML is written with frameDuration='1/24s' (the 24fps default).
        """
        segs = build_segments([], total_duration=30.0)
        s = {**_default_settings(), "fps": "15"}   # not in _FRAME_DURATIONS
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs, s)
        assert root.find(".//format").get("frameDuration") == "1/24s", (
            "Unknown fps string should fall back to 24fps (Fraction(100,2400)=1/24s)"
        )

    def test_effect_element_present(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        eff = root.find(".//effect")
        assert eff is not None
        assert eff.get("id") == "r2"
        assert "Basic Title" in eff.get("name", "")

    def test_effect_uid_is_exact_basic_title_path(self, tmp_path):
        """The effect UID is the path FCP uses to locate the Basic Title .moti bundle.
        If this string is wrong, FCP cannot resolve the effect and every imported
        title clip appears as a missing/offline effect.  Pin the exact value."""
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        eff = root.find(".//effect")
        expected_uid = (
            ".../Titles.localized/Bumper:Opener.localized/"
            "Basic Title.localized/Basic Title.moti"
        )
        assert eff.get("uid") == expected_uid, (
            f"effect uid must exactly match the FCP bundle path; got {eff.get('uid')!r}"
        )

    def test_position_param_key_is_exact_fcp_key(self, tmp_path):
        """The Position param key is a hard-coded FCP internal key path.
        If this key changes, the Y-offset is silently ignored and all telop
        clips render at the default vertical position.  Pin the exact value."""
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        param = root.find(".//param[@name='Position']")
        expected_key = "9999/999166631/999166633/1/100/101"
        assert param is not None
        assert param.get("key") == expected_key, (
            f"Position param key must exactly match the FCP internal key; got {param.get('key')!r}"
        )

    def test_fcpxml_version_is_1_11(self, tmp_path):
        """fcpxml version='1.11' is the format version targeting FCP 10.7+.
        Pin this so a future upgrade is a deliberate, visible diff."""
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        assert root.get("version") == "1.11", (
            f"Expected fcpxml version='1.11'; got {root.get('version')!r}"
        )

    def test_library_event_project_spine_hierarchy(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        library = root.find("library")
        assert library is not None
        event = library.find("event")
        assert event is not None
        project = event.find("project")
        assert project is not None
        seq = project.find("sequence")
        assert seq is not None
        spine = seq.find("spine")
        assert spine is not None

    def test_project_name_uses_stem(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs, stem="myfile")
        project = root.find(".//project")
        assert "myfile" in project.get("name", "")

    def test_xml_declaration_in_file(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        out, _ = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        content = out.read_text(encoding="utf-8")
        assert content.startswith("<?xml")

    def test_doctype_in_file(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        out, _ = _write_and_parse(tmp_path, _make_entries([(1.0, 3.0, "Hi")]), segs)
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE fcpxml>" in content

    def test_gap_duration_equals_sequence_duration(self, tmp_path):
        """Gap element duration attribute must match the sequence duration.

        The gap is the sole spine element and spans the whole timeline, so its
        duration string must be identical to the sequence's duration string.
        Any mismatch would cause FCP to reject the import.
        """
        segs = build_segments([(5.0, 15.0)], total_duration=30.0)   # 20s kept
        entries = _make_entries([(1.0, 3.0, "Hi")])
        _, root = _write_and_parse(tmp_path, entries, segs, duration=30.0)
        seq = root.find(".//sequence")
        gap = root.find(".//gap")
        assert seq is not None and gap is not None
        assert seq.get("duration") == gap.get("duration"), (
            f"sequence duration {seq.get('duration')!r} != "
            f"gap duration {gap.get('duration')!r}"
        )

    def test_gap_starts_at_zero(self, tmp_path):
        """Gap element must have offset='0s'.

        FCP requires the gap to start at the beginning of the timeline.
        A non-zero offset would misalign all title clips relative to the video.
        """
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Hi")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        gap = root.find(".//gap")
        assert gap is not None
        assert gap.get("offset") == "0s", (
            f"Gap offset must be '0s', got {gap.get('offset')!r}"
        )

    def test_sequence_tcstart_is_zero(self, tmp_path):
        """sequence tcStart must be '0s' — FCPXML timecode origin.

        A non-zero tcStart would shift every title clip's display time in FCP.
        """
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Hi")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        seq = root.find(".//sequence")
        assert seq is not None
        assert seq.get("tcStart") == "0s", (
            f"sequence tcStart must be '0s', got {seq.get('tcStart')!r}"
        )

    def test_sequence_tc_format_ndf_for_24fps(self, tmp_path):
        """24 fps uses non-drop-frame timecode (NDF)."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Hi")])
        settings = {**_default_settings(), "fps": "24"}
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        seq = root.find(".//sequence")
        assert seq.get("tcFormat") == "NDF"

    def test_sequence_tc_format_ndf_for_23976fps(self, tmp_path):
        """23.976 fps uses non-drop-frame timecode (NDF)."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Hi")])
        settings = {**_default_settings(), "fps": "23.976"}
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        seq = root.find(".//sequence")
        assert seq.get("tcFormat") == "NDF"

    def test_sequence_tc_format_df_for_2997fps(self, tmp_path):
        """29.97 fps (NTSC colour burst rate) uses drop-frame timecode (DF).

        DF is required for 29.97 so that timecode display stays in sync with
        wall-clock time over long durations (non-drop drifts ~2 frames/min).
        """
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Hi")])
        settings = {**_default_settings(), "fps": "29.97"}
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        seq = root.find(".//sequence")
        assert seq.get("tcFormat") == "DF", (
            f"29.97 fps must use DF timecode; got {seq.get('tcFormat')!r}"
        )

    def test_sequence_tc_format_df_for_5994fps(self, tmp_path):
        """59.94 fps (4K NTSC colour burst rate) uses drop-frame timecode (DF)."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Hi")])
        settings = {**_default_settings(), "fps": "59.94"}
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        seq = root.find(".//sequence")
        assert seq.get("tcFormat") == "DF", (
            f"59.94 fps must use DF timecode; got {seq.get('tcFormat')!r}"
        )

    def test_sequence_tc_format_ndf_for_30fps(self, tmp_path):
        """30 fps (not 29.97 NTSC) uses non-drop-frame timecode."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Hi")])
        settings = {**_default_settings(), "fps": "30"}
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        seq = root.find(".//sequence")
        assert seq.get("tcFormat") == "NDF"

    def test_sequence_tc_format_ndf_for_60fps(self, tmp_path):
        """60 fps (not 59.94 NTSC) uses non-drop-frame timecode."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Hi")])
        settings = {**_default_settings(), "fps": "60"}
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        seq = root.find(".//sequence")
        assert seq.get("tcFormat") == "NDF"


# ── Title clips ───────────────────────────────────────────────────────────────

class TestFcpxmlTitleClips:
    def test_title_elements_created(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "A"), (5.0, 7.0, "B")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        titles = root.findall(".//title")
        assert len(titles) == 2

    def test_title_text_content(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Hello World")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        ts_ref = root.find(".//text/text-style")
        assert ts_ref is not None
        assert ts_ref.text == "Hello World"

    def test_title_has_lane_1(self, tmp_path):
        """title/@lane must be '1' — FCP uses this to place clips on the correct layer.

        Without lane='1', FCP either rejects the import or places the title on a
        different connected-clip lane than expected, breaking the telop layout.
        """
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Test")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        title = root.find(".//title")
        assert title is not None
        assert title.get("lane") == "1", (
            f"title/@lane must be '1', got {title.get('lane')!r}"
        )

    def test_title_ref_points_to_basic_title_effect(self, tmp_path):
        """title/@ref must match the 'Basic Title' effect resource id ('r2').

        The ref attribute links each title clip to the Basic Title motion template.
        An incorrect ref would cause FCP to fail to render the title text.
        """
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Test")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        title = root.find(".//title")
        assert title is not None
        assert title.get("ref") == "r2", (
            f"title/@ref must be 'r2' (Basic Title effect), got {title.get('ref')!r}"
        )

    def test_entry_in_removed_region_dropped(self, tmp_path):
        segs = build_segments([(5.0, 15.0)], total_duration=30.0)
        entries = _make_entries([(7.0, 12.0, "Dropped")])
        _, root = _write_and_parse(tmp_path, entries, segs, duration=30.0)
        titles = root.findall(".//title")
        assert len(titles) == 0

    def test_entry_time_remapped_after_cut(self, tmp_path):
        # Remove 10–20, entry at 21–25 → output offset ~11s
        segs = build_segments([(10.0, 20.0)], total_duration=30.0)
        entries = _make_entries([(21.0, 25.0, "After cut")])
        _, root = _write_and_parse(tmp_path, entries, segs, duration=30.0)
        title = root.find(".//title")
        assert title is not None
        assert title.get("offset") is not None
        assert _rt_to_seconds(title.get("offset")) == pytest.approx(11.0, abs=1.0 / 24)

    def test_entry_straddling_cut_truncated_not_dropped(self, tmp_path):
        # Entry 3–7s straddles a cut that removes 5–10s.
        # map_span_to_output should truncate it to 3–5s (output), not drop it.
        segs = build_segments([(5.0, 10.0)], total_duration=20.0)
        entries = _make_entries([(3.0, 7.0, "Straddle")])
        _, root = _write_and_parse(tmp_path, entries, segs, duration=20.0)
        title = root.find(".//title")
        assert title is not None, "entry straddling a cut should be kept (truncated), not dropped"
        # offset ≈ 3.0s (before the cut, no remapping needed)
        assert _rt_to_seconds(title.get("offset", "0s")) == pytest.approx(3.0, abs=1.0 / 24)
        # duration ≈ 2.0s (truncated at cut start 5.0 − entry start 3.0)
        assert _rt_to_seconds(title.get("duration", "0s")) == pytest.approx(2.0, abs=1.0 / 24)

    def test_straddling_entry_too_short_after_truncation_dropped(self, tmp_path):
        # Entry 4.99–7.0s straddles a cut at 5.0; truncated to 4.99–5.0 = 0.01s.
        # round(0.01 / (1/24)) = round(0.24) = 0 frames → must be dropped.
        segs = build_segments([(5.0, 10.0)], total_duration=20.0)
        entries = _make_entries([(4.99, 7.0, "Sliver")])
        _, root = _write_and_parse(tmp_path, entries, segs, duration=20.0)
        titles = root.findall(".//title")
        assert len(titles) == 0, "sub-frame sliver after truncation must be dropped"

    def test_entry_at_exactly_0p02s_dropped_at_24fps(self, tmp_path):
        """Regression: duration=0.02s at 24fps rounds to 0 frames → must be dropped.

        With the old hardcoded ``< 0.02`` threshold, an entry with exactly 0.02s
        duration would pass the filter.  ``round(0.02 × 24) = round(0.48) = 0``,
        so _to_rt produced ``duration="0s"`` — an invalid FCP clip that would
        cause the import to fail silently.

        The new frame-boundary-aware guard uses
        ``round((end - start) / float(fd)) < 1`` which correctly drops this entry.
        """
        segs = build_segments([], total_duration=10.0)
        # Construct an entry whose source span is exactly 0.02s.
        entries = _make_entries([(1.0, 1.02, "Zero-frame")])
        _, root = _write_and_parse(tmp_path, entries, segs, duration=10.0)
        titles = root.findall(".//title")
        assert len(titles) == 0, (
            "0.02s at 24fps rounds to 0 frames and must be dropped to avoid a "
            "'duration=0s' FCP clip"
        )

    def test_entry_at_one_frame_duration_kept(self, tmp_path):
        """One full frame (1/24s ≈ 0.04167s) at 24fps must produce a valid clip.

        Ensures the frame-aware guard does not over-filter: a duration that rounds
        to exactly 1 frame is the minimum valid FCP clip and must be kept.
        """
        from fractions import Fraction
        one_frame = float(Fraction(1, 24))   # ≈ 0.04167s
        segs = build_segments([], total_duration=10.0)
        entries = _make_entries([(1.0, 1.0 + one_frame, "One frame")])
        _, root = _write_and_parse(tmp_path, entries, segs, duration=10.0)
        titles = root.findall(".//title")
        assert len(titles) == 1, "one-frame duration must survive the minimum filter"
        assert titles[0].get("duration") == "1/24s", (
            f"Expected '1/24s', got {titles[0].get('duration')!r}"
        )

    def test_mixed_entries_correct_count(self, tmp_path):
        segs = build_segments([(5.0, 15.0)], total_duration=30.0)
        entries = _make_entries([
            (1.0, 3.0, "Keep"),
            (7.0, 12.0, "Drop"),
            (18.0, 22.0, "Keep too"),
        ])
        _, root = _write_and_parse(tmp_path, entries, segs, duration=30.0)
        titles = root.findall(".//title")
        assert len(titles) == 2

    def test_font_applied_to_text_style_def(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        settings = _default_settings()
        settings["font"] = "Arial"
        entries = _make_entries([(1.0, 3.0, "Test")])
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        ts_def = root.find(".//text-style-def/text-style")
        assert ts_def is not None
        assert ts_def.get("font") == "Arial"

    def test_font_color_converted_to_fcpxml_format(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        settings = _default_settings()
        settings["font_color"] = "#FF0000"
        entries = _make_entries([(1.0, 3.0, "Red")])
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        ts_def = root.find(".//text-style-def/text-style")
        color = ts_def.get("fontColor")
        assert color is not None
        parts = color.split()
        assert float(parts[0]) == pytest.approx(1.0, abs=0.001)
        assert float(parts[1]) == pytest.approx(0.0, abs=0.001)
        assert float(parts[2]) == pytest.approx(0.0, abs=0.001)
        assert parts[3] == "1"

    def test_no_entries_produces_valid_xml(self, tmp_path):
        segs = build_segments([], total_duration=30.0)
        out, root = _write_and_parse(tmp_path, [], segs)
        assert root.tag == "fcpxml"
        titles = root.findall(".//title")
        assert len(titles) == 0

    def test_zero_font_size_does_not_crash(self, tmp_path):
        """font_size=0 must not raise ZeroDivisionError (guarded by max(1, ...))."""
        segs = build_segments([], total_duration=10.0)
        entries = _make_entries([(1.0, 3.0, "テスト")])
        settings = {**_default_settings(), "font_size": 0}
        # Must complete without exception and produce a valid FCPXML root.
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(entries, segs, 10.0, settings, "test", out)
        assert ET.parse(out).getroot().tag == "fcpxml"

    def test_default_position_y_is_minus_420(self, tmp_path):
        """Default Y offset is -420 when not specified in settings."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Test")])
        settings = _default_settings()
        settings.pop("position_y", None)  # ensure not set
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        param = root.find(".//param[@name='Position']")
        assert param is not None
        assert param.get("value") == "0 -420"

    def test_line_spacing_written_to_text_style(self, tmp_path):
        """lineSpacing attribute is written to text-style-def."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Test")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        ts_def = root.find(".//text-style-def/text-style")
        assert ts_def is not None
        assert ts_def.get("lineSpacing") == "-65"

    def test_custom_line_spacing_applied(self, tmp_path):
        """Custom line_spacing setting is reflected in exported XML."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Test")])
        settings = _default_settings()
        settings["line_spacing"] = -10
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        ts_def = root.find(".//text-style-def/text-style")
        assert ts_def.get("lineSpacing") == "-10"

    def test_title_name_attribute_truncated_to_40_chars(self, tmp_path):
        """name attribute on <title> is capped at 40 characters for FCP compatibility."""
        segs = build_segments([], total_duration=10.0)
        entries = _make_entries([(1.0, 3.0, "あ" * 50)])
        _, root = _write_and_parse(tmp_path, entries, segs)
        title = root.find(".//title")
        assert title is not None
        assert len(title.get("name", "")) <= 40

    def test_title_name_has_no_newline(self, tmp_path):
        """\\n from word-wrap is replaced with space in the title's name attribute."""
        segs = build_segments([], total_duration=10.0)
        # 1920px wide, font_size=80 → max_em≈10.08; 20 CJK chars (20 em) will wrap.
        settings = {**_default_settings(), "width": 1920, "font_size": 80}
        entries = _make_entries([(1.0, 3.0, "あ" * 20)])
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        title = root.find(".//title")
        assert title is not None
        assert "\n" not in title.get("name", "")

    def test_multiple_entries_have_unique_style_ids(self, tmp_path):
        """Every title entry gets its own text-style-def id (ts1, ts2, …) — FCP
        will silently merge styles that share an id, producing wrong colours/fonts."""
        segs = build_segments([], total_duration=10.0)
        entries = _make_entries([
            (1.0, 2.0, "First"),
            (3.0, 4.0, "Second"),
            (5.0, 6.0, "Third"),
        ])
        _, root = _write_and_parse(tmp_path, entries, segs)
        ids = [e.get("id") for e in root.findall(".//text-style-def")]
        assert ids == ["ts1", "ts2", "ts3"]

    def test_custom_position_y_applied(self, tmp_path):
        """Custom position_y=0 (vertical centre) must produce '0 0' in the Position param."""
        segs = build_segments([], total_duration=10.0)
        entries = _make_entries([(1.0, 3.0, "Test")])
        settings = {**_default_settings(), "position_y": 0}
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        param = root.find(".//param[@name='Position']")
        assert param is not None
        assert param.get("value") == "0 0", (
            f"Expected '0 0' for position_y=0, got {param.get('value')!r}"
        )

    def test_positive_position_y_applied(self, tmp_path):
        """Positive position_y (above centre) writes the correct value."""
        segs = build_segments([], total_duration=10.0)
        entries = _make_entries([(1.0, 3.0, "Test")])
        settings = {**_default_settings(), "position_y": 200}
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        param = root.find(".//param[@name='Position']")
        assert param is not None
        assert param.get("value") == "0 200"

    # ── text-style-def attribute contracts ────────────────────────────────────

    def test_text_style_bold_is_1(self, tmp_path):
        """bold='1' must be present — FCP renders non-bold by default; omitting
        this attribute would silently produce regular-weight subtitles."""
        segs = build_segments([], total_duration=10.0)
        entries = _make_entries([(1.0, 3.0, "Test")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        ts = root.find(".//text-style-def/text-style")
        assert ts is not None
        assert ts.get("bold") == "1", (
            f"Expected bold='1' on text-style; got {ts.get('bold')!r}"
        )

    def test_text_style_alignment_center(self, tmp_path):
        """alignment='center' must be present — without it FCP defaults to left
        alignment, mis-positioning multi-line telop text."""
        segs = build_segments([], total_duration=10.0)
        entries = _make_entries([(1.0, 3.0, "Test")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        ts = root.find(".//text-style-def/text-style")
        assert ts is not None
        assert ts.get("alignment") == "center", (
            f"Expected alignment='center' on text-style; got {ts.get('alignment')!r}"
        )

    def test_text_style_font_face_regular(self, tmp_path):
        """fontFace='Regular' must be present — required by FCP's Basic Title
        effect; omitting it can cause the title to fall back to an unexpected face."""
        segs = build_segments([], total_duration=10.0)
        entries = _make_entries([(1.0, 3.0, "Test")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        ts = root.find(".//text-style-def/text-style")
        assert ts is not None
        assert ts.get("fontFace") == "Regular", (
            f"Expected fontFace='Regular' on text-style; got {ts.get('fontFace')!r}"
        )

    def test_text_style_font_size_matches_setting(self, tmp_path):
        """fontSize on text-style-def must match the font_size setting."""
        segs = build_segments([], total_duration=10.0)
        entries = _make_entries([(1.0, 3.0, "Test")])
        settings = {**_default_settings(), "font_size": 120}
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        ts = root.find(".//text-style-def/text-style")
        assert ts is not None
        assert ts.get("fontSize") == "120", (
            f"Expected fontSize='120'; got {ts.get('fontSize')!r}"
        )


# ── use_source_timing ─────────────────────────────────────────────────────────

class TestSourceTiming:
    def _write_source(self, tmp_path, entries, segs, duration=30.0):
        settings = _default_settings()
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(entries, segs, duration, settings, "test", out,
                              use_source_timing=True)
        return out, ET.parse(out).getroot()

    def _rt_to_seconds(self, rt: str) -> float:
        return _rt_to_seconds(rt)

    def test_source_timing_preserves_original_offset(self, tmp_path):
        """use_source_timing=True: title offset matches the raw entry start time."""
        segs = build_segments([(5.0, 15.0)], total_duration=30.0)
        entries = _make_entries([(20.0, 25.0, "After cut")])
        _, root = self._write_source(tmp_path, entries, segs, duration=30.0)
        title = root.find(".//title")
        assert title is not None
        t = self._rt_to_seconds(title.get("offset"))
        assert t == pytest.approx(20.0, abs=1.0 / 24)

    def test_source_timing_includes_removed_region_entries(self, tmp_path):
        """use_source_timing=True: entries inside removed regions are NOT dropped."""
        segs = build_segments([(5.0, 15.0)], total_duration=30.0)
        entries = _make_entries([(7.0, 12.0, "Inside cut")])
        _, root = self._write_source(tmp_path, entries, segs, duration=30.0)
        titles = root.findall(".//title")
        assert len(titles) == 1

    def test_source_timing_sequence_uses_full_duration(self, tmp_path):
        """Sequence duration reflects total source duration, not cut duration."""
        segs = build_segments([(0.0, 10.0)], total_duration=30.0)  # 20s kept
        entries = _make_entries([(1.0, 3.0, "Hi")])
        _, root = self._write_source(tmp_path, entries, segs, duration=30.0)
        seq = root.find(".//sequence")
        seq_dur = self._rt_to_seconds(seq.get("duration"))
        # Should be ~31s (30 + 1s buffer), NOT ~21s (cut duration + buffer)
        assert seq_dur == pytest.approx(31.0, abs=1.0)

    def test_edited_timing_drops_removed_region_entries(self, tmp_path):
        """Control: use_source_timing=False (default) still drops removed entries."""
        segs = build_segments([(5.0, 15.0)], total_duration=30.0)
        entries = _make_entries([(7.0, 12.0, "Inside cut")])
        settings = _default_settings()
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(entries, segs, 30.0, settings, "test", out,
                              use_source_timing=False)
        root = ET.parse(out).getroot()
        titles = root.findall(".//title")
        assert len(titles) == 0

    def test_source_timing_clean_text_still_applied(self, tmp_path):
        """Punctuation stripping still runs with source timing enabled."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Hello、World。")])
        _, root = self._write_source(tmp_path, entries, segs)
        ts_ref = root.find(".//text/text-style")
        assert "、" not in ts_ref.text
        assert "。" not in ts_ref.text

    def test_edited_timing_sequence_uses_cut_duration(self, tmp_path):
        """Sequence duration with edited timing reflects post-cut length, not source.

        Complement to test_source_timing_sequence_uses_full_duration:
        use_source_timing=False (default) should produce a sequence whose
        duration is the total kept duration plus the 1s buffer — NOT the source
        duration.  Here: 30s source − 10s removed = 20s kept → ~21s sequence.
        """
        segs = build_segments([(10.0, 20.0)], total_duration=30.0)   # 20s kept
        entries = _make_entries([(1.0, 3.0, "Hi")])
        settings = _default_settings()
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(
            entries, segs, 30.0, settings, "test", out,
            use_source_timing=False,
        )
        root = ET.parse(out).getroot()
        seq = root.find(".//sequence")
        seq_dur = _rt_to_seconds(seq.get("duration"))
        # 20s kept + 1s buffer ≈ 21s, NOT ≈ 31s (full source)
        assert seq_dur == pytest.approx(21.0, abs=1.0), (
            f"Expected ≈21s (cut+buffer), got {seq_dur:.4f}s"
        )

    def test_entries_out_of_order_sorted_in_output(self, tmp_path):
        """Entries passed out of chronological order must be sorted by start time.

        filter_telop_entries() sorts by start, so the FCPXML title clips must
        appear in temporal order regardless of the input order.
        """
        segs = build_segments([], total_duration=30.0)
        # Deliberately reversed: later entry first in the list
        entries = [
            {"id": "b", "start": 10.0, "end": 14.0, "text": "Second"},
            {"id": "a", "start": 1.0,  "end": 5.0,  "text": "First"},
        ]
        settings = _default_settings()
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(entries, segs, 30.0, settings, "test", out,
                              use_source_timing=True)
        root = ET.parse(out).getroot()
        titles = root.findall(".//title")
        assert len(titles) == 2
        offsets = [_rt_to_seconds(t.get("offset")) for t in titles]
        assert offsets[0] < offsets[1], (
            f"Titles should be in temporal order; got offsets {offsets}"
        )

    def test_empty_keep_segments_uses_source_duration(self, tmp_path):
        """Passing keep_segments=[] with use_source_timing=False must use
        total_source_duration as the timeline length, not 0.

        ``sum(s.duration for s in [])`` = 0, but ``if keep_segments`` is False
        for an empty list, so the guard falls back to total_source_duration.
        Without that guard, the sequence would be only 1s (0+1 buffer) long.
        """
        entries = _make_entries([(1.0, 3.0, "Hi")])
        settings = _default_settings()
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(
            entries, [], 30.0, settings, "test", out,
            use_source_timing=False,
        )
        root = ET.parse(out).getroot()
        seq = root.find(".//sequence")
        seq_dur = _rt_to_seconds(seq.get("duration"))
        # Without the guard: 0+1 = 1s.  With the guard: 30+1 = 31s.
        assert seq_dur == pytest.approx(31.0, abs=1.0), (
            f"Expected ≈31s (source+buffer fallback), got {seq_dur:.4f}s"
        )


# ── _char_em / _em_width / _wrap_telop ───────────────────────────────────────

class TestCharEm:
    def test_cjk_is_full_width(self):
        assert _char_em("あ") == 1.0
        assert _char_em("字") == 1.0
        assert _char_em("ア") == 1.0

    def test_ascii_is_narrow(self):
        assert _char_em("a") == pytest.approx(0.55)
        assert _char_em("1") == pytest.approx(0.55)
        assert _char_em(" ") == pytest.approx(0.55)

    def test_fullwidth_ascii_is_full_width(self):
        """Fullwidth ASCII (U+FF01–U+FF60) must be classified as 1.0 em (Wide/Fullwidth)."""
        assert _char_em("Ａ") == pytest.approx(1.0)   # U+FF21 FULLWIDTH LATIN CAPITAL LETTER A

    def test_ambiguous_degree_sign_is_full_width(self):
        """° (U+00B0) has east_asian_width='A' (Ambiguous) → Python returns 1.0.

        Parity note: the JS port omits the Ambiguous (A) category entirely, so °
        falls through to the 0.55 default.  Both behaviours are intentional and
        documented; this test pins the Python 1.0 value.
        """
        assert _char_em("°") == pytest.approx(1.0)

    def test_hangul_syllable_is_full_width(self):
        """가 (U+AC00) is in the AC00–D7FF Hangul Syllables range → 1.0 em."""
        assert _char_em("가") == pytest.approx(1.0)

    def test_fullwidth_currency_sign_is_full_width(self):
        """￠ (U+FFE0) is in the FFE0–FFE6 Fullwidth Signs range → 1.0 em."""
        assert _char_em("￠") == pytest.approx(1.0)

    # ── Remaining ranges — one representative per block ──────────────────────

    def test_hangul_jamo_is_full_width(self):
        """ᄀ (U+1100) is in the 1100–115F Hangul Jamo range → 1.0 em."""
        assert _char_em("ᄀ") == pytest.approx(1.0)

    def test_cjk_radicals_supplement_is_full_width(self):
        """⺀ (U+2E80) is in the 2E80–303F CJK Radicals+Symbols range → 1.0 em."""
        assert _char_em("⺀") == pytest.approx(1.0)

    def test_hangul_jamo_ext_a_is_full_width(self):
        """ꥠ (U+A960) is in the A960–A97F Hangul Jamo Extended-A range → 1.0 em."""
        assert _char_em("ꥠ") == pytest.approx(1.0)

    def test_cjk_compat_ideograph_is_full_width(self):
        """豈 (U+F900) is in the F900–FAFF CJK Compat Ideographs range → 1.0 em."""
        assert _char_em("豈") == pytest.approx(1.0)

    def test_vertical_presentation_form_is_full_width(self):
        """︐ (U+FE10) is in the FE10–FE6F Vertical/CJK Compat Forms range → 1.0 em."""
        assert _char_em("︐") == pytest.approx(1.0)

    def test_emoji_is_full_width_in_python(self):
        """😊 (U+1F60A) has east_asian_width='W' (Wide) in Python → 1.0 em.

        Parity note: the JS port uses explicit code-point ranges that only cover
        BMP characters up to U+FFE6.  Emoji above U+FFFF fall through to the 0.55
        default in JS.  Python and JS therefore disagree on emoji width.
        This test pins the Python 1.0 value; see the JS 'emoji' test for 0.55.
        """
        assert _char_em("😊") == pytest.approx(1.0)

    def test_half_width_katakana_is_narrow(self):
        """ｱ (U+FF71) has east_asian_width='H' (Half-width) in Python → 0.55 em.

        Half-width katakana (U+FF61–FF9F) looks like katakana but is narrow.
        Parity: the JS port's FF01–FF60 range deliberately excludes FF61–FF9F,
        so both environments return 0.55 em.  Important because Whisper
        occasionally outputs half-width kana via transliteration.
        """
        assert _char_em("ｱ") == pytest.approx(0.55)   # U+FF71 HALFWIDTH KATAKANA LETTER A
        assert _char_em("ｦ") == pytest.approx(0.55)   # U+FF66 HALFWIDTH KATAKANA LETTER WO
        assert _char_em("ﾟ") == pytest.approx(0.55)   # U+FF9F HALFWIDTH KATAKANA VOICED MARK

    def test_em_width_pure_cjk(self):
        # 5 CJK chars → 5.0 em
        assert _em_width("あいうえお") == pytest.approx(5.0)

    def test_em_width_pure_ascii(self):
        assert _em_width("hello") == pytest.approx(5 * 0.55)

    def test_em_width_ignores_newline(self):
        assert _em_width("あ\nい") == pytest.approx(2.0)

    def test_em_width_empty_string(self):
        assert _em_width("") == pytest.approx(0.0)

    def test_em_width_mixed_cjk_and_ascii(self):
        # 3 CJK (3.0) + 3 ASCII (3 × 0.55 = 1.65) = 4.65
        assert _em_width("あaいbうc") == pytest.approx(3 * 1.0 + 3 * 0.55)


class TestBreakScore:
    def test_space_is_tier3(self):
        assert _break_score("a b", 2) == pytest.approx(3.0)

    def test_primary_particle_wa_is_tier2(self):
        assert _break_score("あはい", 2) == pytest.approx(2.0)

    def test_primary_particle_ga_is_tier2(self):
        assert _break_score("あがい", 2) == pytest.approx(2.0)

    def test_secondary_particle_de_is_tier1(self):
        assert _break_score("あでい", 2) == pytest.approx(1.0)

    def test_te_form_is_tier07(self):
        assert _break_score("あてい", 2) == pytest.approx(0.7)

    def test_plain_cjk_is_zero(self):
        assert _break_score("あいう", 2) == pytest.approx(0.0)

    def test_boundary_positions_are_zero(self):
        assert _break_score("abc", 0) == pytest.approx(0.0)
        assert _break_score("abc", 3) == pytest.approx(0.0)

    # Remaining tier-2 particles — all should score 2.0
    def test_primary_particle_wo_is_tier2(self):
        assert _break_score("あをい", 2) == pytest.approx(2.0)

    def test_primary_particle_ni_is_tier2(self):
        assert _break_score("あにい", 2) == pytest.approx(2.0)

    def test_primary_particle_to_is_tier2(self):
        assert _break_score("あとい", 2) == pytest.approx(2.0)

    # Remaining tier-1 particles
    def test_secondary_particle_mo_is_tier1(self):
        assert _break_score("あもい", 2) == pytest.approx(1.0)

    def test_secondary_particle_no_is_tier1(self):
        assert _break_score("あのい", 2) == pytest.approx(1.0)

    def test_secondary_particle_he_is_tier1(self):
        assert _break_score("あへい", 2) == pytest.approx(1.0)

    def test_secondary_particle_ka_is_tier1(self):
        assert _break_score("あかい", 2) == pytest.approx(1.0)

    # ── mid-word penalties (never break here) ─────────────────────────────────

    def test_break_after_sokuon_penalised(self):
        # Breaking after っ would orphan the following kana — strongly avoided.
        assert _break_score("あった", 2) < 0

    def test_break_after_katakana_sokuon_penalised(self):
        assert _break_score("アッタ", 2) < 0

    def test_compound_particle_toka_penalised(self):
        # "丸とか" — breaking between と and か splits the compound particle とか.
        assert _break_score("丸とか", 2) < 0

    def test_compound_particle_kara_penalised(self):
        assert _break_score("だから", 2) < 0

    def test_compound_particle_tte_penalised(self):
        assert _break_score("だって", 2) < 0

    def test_break_after_prolonged_mark_ok(self):
        # Breaking AFTER コーヒー (pos 4, コーヒー|本) is fine — ー ends the word.
        assert _break_score("コーヒー本", 4) >= 0

    def test_break_before_prolonged_mark_penalised(self):
        # Breaking BEFORE a ー (pos 3, コーヒ|ー本) strands the long-vowel mark on
        # line 2, splitting the syllable — must be avoided.
        assert _break_score("コーヒー本", 3) < 0

    def test_break_before_small_kana_penalised(self):
        # Line 2 must not start with a small kana (splits a mora): オーデ|ィオ.
        assert _break_score("オーディオ", 3) < 0

    def test_to_followed_by_non_compound_still_tier2(self):
        # "と" before a non-compound char keeps its tier-2 score.
        assert _break_score("あとあ", 2) == pytest.approx(2.0)

    # ── Numeric tokens must never be split (bug: "GPT 5.5" wrapping as "5" / "5") ──

    def test_break_before_decimal_point_penalised(self):
        # "5.5" — breaking right before the "." (pos 1, "5"|".5") splits the number.
        assert _break_score("5.5", 1) < 0

    def test_break_after_decimal_point_penalised(self):
        # "5.5" — breaking right after the "." (pos 2, "5."|"5") also splits it.
        assert _break_score("5.5", 2) < 0

    def test_break_between_two_digits_penalised(self):
        # "123" — no decimal point at all, still must not split a plain digit run.
        assert _break_score("123", 1) < 0
        assert _break_score("123", 2) < 0

    def test_break_before_digit_after_letter_not_penalised(self):
        # "GPT5" — breaking between "T" and "5" doesn't split a NUMBER (prev isn't
        # a digit or "."), so this position keeps its normal (zero) score.
        assert _break_score("GPT5", 3) == pytest.approx(0.0)


class TestWrapTelop:
    def test_short_text_unchanged(self):
        assert _wrap_telop("あいう", max_em=10.0) == "あいう"

    def test_long_text_gets_newline(self):
        text = "あ" * 30
        result = _wrap_telop(text, max_em=15.0)
        assert result.count("\n") == 1

    def test_two_lines_each_within_max_em(self):
        text = "あ" * 30
        result = _wrap_telop(text, max_em=16.0)
        lines = result.split("\n")
        assert len(lines) == 2
        for line in lines:
            assert _em_width(line) <= 16.0 + 0.01

    def test_balanced_lines_no_particles(self):
        """Pure CJK with no particles falls back to midpoint — near-equal lines."""
        text = "あ" * 20
        result = _wrap_telop(text, max_em=12.0)
        lines = result.split("\n")
        diff = abs(_em_width(lines[0]) - _em_width(lines[1]))
        assert diff <= 2.0

    def test_particle_dominates_over_midpoint(self):
        """は near the centre wins over the raw midpoint (no は)."""
        # "xxxxはyyyy" total em ≈ 5.4, max_em=3.5 forces split.
        # は at cumulative 3.2 em has: combined = 2 - |3.2-2.7|/2.7 = 2 - 0.185 = 1.815
        # Next char 'y' at 3.75 em: combined = 0 - 0.19 = -0.19  → は wins decisively.
        text = "xxxxはyyyy"
        result = _wrap_telop(text, max_em=3.5)
        assert result.split("\n")[0].endswith("は")

    def test_space_dominates_over_midpoint(self):
        """A space (former punctuation) wins over the raw midpoint."""
        # "aaaa bbb" — space at position 4, raw midpoint around 4.
        text = "aaaa bbb"
        result = _wrap_telop(text, max_em=4.0)
        lines = result.split("\n")
        # Line 1 should not have a trailing space (stripped), line 2 starts with 'b'
        assert lines[0].rstrip() == "aaaa"
        assert lines[1].lstrip() == "bbb"

    def test_realistic_jp_sentence_breaks_at_particle(self):
        """Realistic Japanese sentence breaks at a particle, not mid-word.

        "しばらくは完全に本気を出せない":
        - は at pos 5 (score 1.667) and に at pos 8 (score 1.933, closer to midpoint)
        - Algorithm picks に — closer to centre AND still a particle. Both are valid;
          the important check is that the break falls on SOME particle, not mid-word.
        """
        text = "しばらくは完全に本気を出せない"
        result = _wrap_telop(text, max_em=9.0)
        lines = result.split("\n")
        # The break must be after a particle (は, が, を, に, で, と, …), not mid-kanji.
        assert lines[0][-1] in "はがをにでとももかのへやよねわて", (
            f"Expected break after a particle, got: {lines}"
        )

    def test_line2_truncated_when_still_too_long(self):
        text = "あ" * 40
        result = _wrap_telop(text, max_em=10.0)
        assert _em_width(result.split("\n")[1]) <= 10.0 + 0.01

    def test_at_most_two_lines(self):
        text = "あ" * 100
        assert _wrap_telop(text, max_em=10.0).count("\n") <= 1

    def test_strips_space_at_break_boundary(self):
        """Space at the break is consumed — no trailing/leading spaces in output."""
        text = "hello world"
        result = _wrap_telop(text, max_em=4.0)
        lines = result.split("\n")
        assert not lines[0].endswith(" ")
        assert not lines[1].startswith(" ")

    def test_empty_string_returned_unchanged(self):
        """Empty string has total_em=0 which is ≤ any maxEm — returned as-is."""
        assert _wrap_telop("", max_em=10.0) == ""

    def test_single_character_at_boundary_unchanged(self):
        """Single CJK char at maxEm=1.0: total_em==maxEm → no wrap."""
        assert _wrap_telop("あ", max_em=1.0) == "あ"

    def test_text_exactly_at_max_em_unchanged(self):
        """Text whose total_em equals maxEm exactly is returned without a newline."""
        # 3 CJK chars × 1.0 em = 3.0 em; max_em=3.0 → total_em <= max_em → no wrap
        assert _wrap_telop("あいう", max_em=3.0) == "あいう"

    def test_two_char_cjk_wraps_at_midpoint(self):
        """2-char CJK string with max_em < total_em splits to one char per line.
        bestPos = max(1, floor(2/2)) = 1; only one loop iteration → pos=1 wins."""
        result = _wrap_telop("あい", max_em=1.5)   # total 2.0 em > 1.5 → wrap
        assert result == "あ\nい"

    def test_both_wrapped_lines_fit_within_max_em(self):
        """After wrapping a realistic JP sentence, each output line is ≤ max_em.
        'しばらくは完全に本気を出せない' (15 em) with max_em=9.0 splits at に
        (pos 8, score 1.933) → line1='しばらくは完全に' (8.0 em),
        line2='本気を出せない' (7.0 em) — both within the 9.0 em limit."""
        text = "しばらくは完全に本気を出せない"
        max_em = 9.0
        result = _wrap_telop(text, max_em)
        for line in result.split("\n"):
            assert _em_width(line) <= max_em + 0.01, (
                f"Line {line!r} exceeds max_em {max_em}: {_em_width(line):.2f} em"
            )

    def test_emoji_preserved_intact_after_wrap(self):
        """Emoji (multibyte Unicode scalar) must survive wrapping without corruption.

        Python iterates over Unicode scalars natively, so the emoji stays intact.
        This test mirrors the JS 'emoji remains intact' test to ensure parity.
        """
        text = "こんにちは😊世界"
        result = _wrap_telop(text, max_em=4.0)
        # The emoji must appear in the output — not replaced by ? or corrupted.
        assert "😊" in result, f"Emoji should pass through intact; got: {result!r}"
        # Result must have at most one newline (at most 2 lines).
        assert result.count("\n") <= 1

    def test_line2_exactly_at_max_em_not_truncated(self):
        """Line 2 width == maxEm must NOT trigger hard-truncation (condition is >).

        'ああ あああ' (7 em incl. space) > maxEm=3.0 → wrapping triggered.  The space
        is the strongest break point, so it splits there:
          line1 = 'ああ' (2.0 em), line2 = 'あああ' (3.0 em == maxEm).
        _em_width(line2) = 3.0 ≤ 3.0 → truncation guard `> max_em` is false
        → line2 is left intact (3 chars, not 2).
        """
        result = _wrap_telop("ああ あああ", max_em=3.0)
        assert "\n" in result, f"text (>3.0 em) should wrap at maxEm=3.0; got: {result!r}"
        line2 = result.split("\n")[1]
        assert len(line2) == 3, f"Line 2 should have 3 chars (3.0 em == maxEm, not truncated); got: {line2!r}"

    def test_half_width_katakana_not_wrapped_below_max_em(self):
        """Half-width katakana (0.55 em each) should not trigger wrapping when total
        width is within maxEm.

        'ｱｲｳｴｵ' = 5 half-width chars × 0.55 em = 2.75 em ≤ maxEm=3.0 → no wrap.
        This guards against a regression where half-width kana is mis-classified as
        full-width (1.0 em), which would make the total 5.0 em and trigger wrapping.
        """
        result = _wrap_telop("ｱｲｳｴｵ", max_em=3.0)
        assert "\n" not in result, f"Half-width kana (2.75 em) should not wrap at maxEm=3.0; got: {result!r}"

    def test_compound_particle_toka_not_split_across_lines(self):
        """Regression: '不完全な丸とかブレた形とかが好きなんで' must not break the
        compound particle とか (was '…丸と' / 'かブレた…')."""
        text = "私は不完全な丸とかブレた形とかが好きなんで"
        result = _wrap_telop(text, max_em=20.16)   # 3840 × 0.42 / 80
        lines = result.split("\n")
        for i, ln in enumerate(lines[:-1]):
            nxt = lines[i + 1]
            assert not (ln.endswith("と") and nxt.startswith("か")), \
                f"とか split across lines: {result!r}"

    def test_sokuon_never_at_line_end(self):
        """A line must never end with a sokuon (っ/ッ) — it would orphan the next kana."""
        text = "ちょっとだけまってそういったことなんですよね"
        result = _wrap_telop(text, max_em=10.0)
        for ln in result.split("\n")[:-1]:
            assert ln[-1:] not in "っッ", f"line ends with sokuon: {result!r}"

    def test_line2_never_starts_with_prolonged_mark_or_small_kana(self):
        """A katakana compound must not be wrapped so line 2 starts with ー or a
        small kana (would split a mora, e.g. コーヒ|ー or オーデ|ィオ)."""
        for text in ["コーヒーをいっぱい飲んだcomment", "オーディオビジュアライザー機能です"]:
            result = _wrap_telop(text, max_em=8.0)
            for ln in result.split("\n")[1:]:
                assert ln[:1] not in "ーっッぁぃぅぇぉゃゅょァィゥェォャュョ", \
                    f"line 2 starts mid-mora: {result!r}"

    def test_decimal_point_never_split_across_lines(self):
        """Regression: 'GPT5.5' rendering as 'GPT5' / '5' (or '.5') when the
        wrap's midpoint calculation happened to land inside the number."""
        text = "このモデルは本当にすごくてGPT5.5を使うと生産性が劇的に向上します"
        result = _wrap_telop(text, max_em=17.53)   # 3840 × 0.42 / 92
        lines = result.split("\n")
        assert any("5.5" in ln for ln in lines), f"decimal point split across lines: {result!r}"

    def test_digit_run_never_split_at_forced_midpoint(self):
        """Adversarial case: a bare digit run sits exactly at the em-width midpoint
        with no particles/spaces nearby to steer the break elsewhere — without the
        guard, nearest-to-midpoint alone would land inside the number."""
        text = "あいうえおかきくけこ5588さしすせそたちつてと"
        result = _wrap_telop(text, max_em=10.0)
        lines = result.split("\n")
        assert any("5588" in ln for ln in lines), f"digit run split across lines: {result!r}"

    def test_multiple_decimals_in_one_string_all_stay_whole(self):
        text = "最近はGPT5.5とかGemini3.5とか色々出てきて選ぶのが大変ですよね"
        result = _wrap_telop(text, max_em=17.53)
        lines = result.split("\n")
        assert any("5.5" in ln for ln in lines), f"5.5 split: {result!r}"
        assert any("3.5" in ln for ln in lines), f"3.5 split: {result!r}"


class TestWrapTelopIntegration:
    def test_long_jp_sentence_wraps_in_fcpxml(self, tmp_path):
        """A 25-char CJK sentence (25.0 em > max_em 20.16) wraps to exactly 2 lines."""
        segs = build_segments([], total_duration=30.0)
        long_text = "あ" * 25  # 25 CJK chars = 25.0 em, clearly above max_em=20.16
        entries = _make_entries([(1.0, 5.0, long_text)])
        # 3840px wide, font_size=80 → max_em = 3840*0.42/80 ≈ 20.16
        settings = {**_default_settings(), "font_size": 80, "width": 3840}
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(entries, segs, 30.0, settings, "test", out)
        root = ET.parse(out).getroot()
        ts_ref = root.find(".//text/text-style")
        assert ts_ref is not None
        assert "\n" in ts_ref.text, "25 CJK chars (25.0 em > max_em 20.16) must wrap"
        assert ts_ref.text.count("\n") == 1, "must wrap to exactly 2 lines"

    def test_short_entry_stays_single_line(self, tmp_path):
        """A short entry (< max_em) is NOT split."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "ありがとう")])  # 5 CJK chars
        settings = {**_default_settings(), "font_size": 80, "width": 3840}
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(entries, segs, 30.0, settings, "test", out)
        root = ET.parse(out).getroot()
        ts_ref = root.find(".//text/text-style")
        assert "\n" not in ts_ref.text

    def test_mixed_english_japanese_space_breaks_early_line2_truncated(self, tmp_path):
        """When ASCII text precedes Japanese, a space between English words dominates
        as a tier-3 break even if it leaves a short line1.  Line2 is then hard-
        truncated if it exceeds max_em.  This is the known algorithm limitation for
        mixed-script text like 'Kuuki Designのウェブサイトをご覧ください'."""
        text = "Kuuki Designのウェブサイトをご覧ください"
        # 1440p, font_size=80 → max_em = 2560*0.42/80 = 13.44
        max_em = 2560 * 0.42 / 80
        result = _wrap_telop(_clean_telop_text(text), max_em)
        lines = result.split("\n")
        assert len(lines) == 2, "mixed text above max_em must wrap"
        l1, l2 = lines
        # Line 1: "Kuuki" (space was tier-3 break winner)
        assert l1 == "Kuuki"
        # Line 2: hard-truncated to ≤ max_em
        assert _em_width(l2) <= max_em + 0.01

    def test_vertical_video_max_em_floor_wraps_9_chars(self, tmp_path):
        """Vertical 9:16 (1080px wide) with font_size=80: computed max_em = 1080*0.42/80
        = 5.67, which is below the 8.0 minimum → clamped to 8.0.
        9 CJK chars (9.0 em > 8.0) must be wrapped to 2 lines."""
        segs = build_segments([], total_duration=10.0)
        settings = {**_default_settings(), "width": 1080, "height": 1920, "font_size": 80}
        entries = _make_entries([(1.0, 3.0, "あ" * 9)])
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(entries, segs, 10.0, settings, "test", out)
        root = ET.parse(out).getroot()
        ts_ref = root.find(".//text/text-style")
        assert ts_ref is not None
        assert "\n" in ts_ref.text, "9 CJK chars (9.0 em) must wrap when max_em floor is 8.0"

    def test_vertical_video_max_em_floor_no_wrap_8_chars(self, tmp_path):
        """Vertical 9:16 (1080px wide) with font_size=80: max_em clamped to 8.0.
        8 CJK chars (8.0 em == max_em) must NOT wrap (_wrap_telop returns early when
        total_em <= max_em)."""
        segs = build_segments([], total_duration=10.0)
        settings = {**_default_settings(), "width": 1080, "height": 1920, "font_size": 80}
        entries = _make_entries([(1.0, 3.0, "あ" * 8)])
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(entries, segs, 10.0, settings, "test", out)
        root = ET.parse(out).getroot()
        ts_ref = root.find(".//text/text-style")
        assert ts_ref is not None
        assert "\n" not in ts_ref.text, "8 CJK chars (8.0 em == max_em) must NOT wrap"

    def test_large_font_size_max_em_at_boundary(self, tmp_path):
        """3840px wide, font_size=160: max_em = 3840*0.42/160 = 10.08.
        10 CJK chars (10.0 em ≤ 10.08) must NOT wrap."""
        segs = build_segments([], total_duration=10.0)
        settings = {**_default_settings(), "width": 3840, "height": 2160, "font_size": 160}
        entries = _make_entries([(1.0, 3.0, "あ" * 10)])
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(entries, segs, 10.0, settings, "test", out)
        root = ET.parse(out).getroot()
        ts_ref = root.find(".//text/text-style")
        assert ts_ref is not None
        assert "\n" not in ts_ref.text, "10 CJK chars (10.0 em ≤ max_em 10.08) must NOT wrap"

    def test_large_font_size_wraps_when_exceeds_max_em(self, tmp_path):
        """3840px wide, font_size=160: max_em = 10.08.
        11 CJK chars (11.0 em > 10.08) must wrap to 2 lines."""
        segs = build_segments([], total_duration=10.0)
        settings = {**_default_settings(), "width": 3840, "height": 2160, "font_size": 160}
        entries = _make_entries([(1.0, 3.0, "あ" * 11)])
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(entries, segs, 10.0, settings, "test", out)
        root = ET.parse(out).getroot()
        ts_ref = root.find(".//text/text-style")
        assert ts_ref is not None
        assert "\n" in ts_ref.text, "11 CJK chars (11.0 em > max_em 10.08) must wrap"
        assert ts_ref.text.count("\n") == 1, "must wrap to exactly 2 lines"


# ── _clean_telop_text() ───────────────────────────────────────────────────────

class TestCleanTelopText:
    def test_jp_separators_become_spaces(self):
        assert _clean_telop_text("それはね。すごいよ") == "それはね すごいよ"

    def test_jp_wrappers_removed(self):
        assert _clean_telop_text("「こんにちは」") == "こんにちは"

    def test_en_comma_becomes_space(self):
        # comma → space; adjacent space collapsed → single space
        assert _clean_telop_text("hello,world") == "hello world"

    def test_en_period_becomes_space(self):
        assert _clean_telop_text("end.") == "end"

    def test_exclamation_becomes_space(self):
        assert _clean_telop_text("wow!great") == "wow great"

    def test_newline_becomes_space(self):
        assert _clean_telop_text("line1\nline2") == "line1 line2"

    def test_multiple_separators_collapse(self):
        # Each 。 becomes a space; consecutive spaces are collapsed to one.
        assert _clean_telop_text("a。。b") == "a b"

    def test_pure_punctuation_returns_empty(self):
        assert _clean_telop_text("。、！？") == ""

    def test_mixed_jp_en(self):
        result = _clean_telop_text("Hello、世界！Great.")
        assert "、" not in result
        assert "！" not in result
        assert "." not in result

    def test_no_punctuation_unchanged(self):
        assert _clean_telop_text("ありがとう") == "ありがとう"

    def test_leading_trailing_whitespace_stripped(self):
        assert _clean_telop_text("  hello  ") == "hello"

    def test_quotes_removed(self):
        assert _clean_telop_text('"quoted"') == "quoted"

    def test_jp_brackets_removed(self):
        # wrappers removed entirely — no space added between adjacent chars
        assert _clean_telop_text("【重要】連絡です") == "重要連絡です"

    def test_brand_correction_applied_at_export(self):
        """Brand name is corrected at the export chokepoint so the FCPXML is right
        even if the entry text was built before the analysis-time correction."""
        assert _clean_telop_text("実際空気デザインのロゴ") == "実際クウキデザインのロゴ"

    def test_bare_air_word_not_corrected(self):
        assert _clean_telop_text("空気がきれい") == "空気がきれい"

    # ── Separator parity with JS _cleanTelopText tests ────────────────────────

    def test_ascii_hyphen_becomes_space(self):
        assert _clean_telop_text("a-b") == "a b"

    def test_em_dash_becomes_space(self):
        assert _clean_telop_text("a—b") == "a b"      # U+2014 —

    def test_en_dash_becomes_space(self):
        assert _clean_telop_text("a–b") == "a b"      # U+2013 –

    def test_ascii_colon_becomes_space(self):
        assert _clean_telop_text("key:value") == "key value"

    def test_middle_dot_becomes_space(self):
        assert _clean_telop_text("a・b") == "a b"          # U+30FB ・

    def test_ellipsis_becomes_space(self):
        assert _clean_telop_text("a…b") == "a b"           # U+2026 …

    def test_wave_dash_becomes_space(self):
        assert _clean_telop_text("a〜b") == "a b"          # U+301C 〜

    def test_fullwidth_tilde_becomes_space(self):
        assert _clean_telop_text("a～b") == "a b"          # U+FF5E ～

    def test_ascii_question_mark_becomes_space(self):
        assert _clean_telop_text("a?b") == "a b"

    def test_ascii_semicolon_becomes_space(self):
        assert _clean_telop_text("a;b") == "a b"

    def test_fullwidth_semicolon_becomes_space(self):
        assert _clean_telop_text("a；b") == "a b"          # U+FF1B ；

    def test_fullwidth_exclamation_becomes_space(self):
        """！ (U+FF01 fullwidth exclamation) — appears in Japanese Whisper output."""
        assert _clean_telop_text("a！b") == "a b"          # U+FF01 ！

    def test_fullwidth_question_mark_becomes_space(self):
        """？ (U+FF1F fullwidth question mark) — common in Japanese text."""
        assert _clean_telop_text("a？b") == "a b"          # U+FF1F ？

    def test_fullwidth_colon_becomes_space(self):
        """： (U+FF1A fullwidth colon) — used in Japanese labels like '概要：'."""
        assert _clean_telop_text("a：b") == "a b"          # U+FF1A ：

    def test_curly_double_quotes_removed(self):
        """Unicode curly/smart double quotes “” are removed (Whisper uses them)."""
        assert _clean_telop_text("“quoted”") == "quoted"

    def test_curly_single_quotes_removed(self):
        """Unicode curly/smart single quotes ‘’ are removed."""
        assert _clean_telop_text("’quoted’") == "quoted"

    # ── Wrapper parity — remaining chars in _WRAP_RE not yet individually tested

    def test_angle_brackets_removed(self):
        """〈〉 (U+3008/3009) are removed by _WRAP_RE."""
        assert _clean_telop_text("〈angle〉") == "angle"

    def test_double_angle_brackets_removed(self):
        """《》 (U+300A/300B) are removed by _WRAP_RE."""
        assert _clean_telop_text("《double》") == "double"

    def test_ascii_parentheses_removed(self):
        """ASCII () are removed by _WRAP_RE (Whisper occasionally wraps asides)."""
        assert _clean_telop_text("(aside)") == "aside"

    def test_ascii_single_quotes_removed(self):
        """ASCII ‘’ are removed by _WRAP_RE."""
        assert _clean_telop_text("’quoted’") == "quoted"

    def test_white_corner_brackets_removed(self):
        """『』 (U+300E/300F white corner brackets) — used for quoted titles in JP."""
        assert _clean_telop_text("『引用』") == "引用"

    def test_fullwidth_parentheses_removed(self):
        """（） (U+FF08/FF09 fullwidth parens) — common in JP speech asides."""
        assert _clean_telop_text("（括弧）") == "括弧"

    def test_leading_separator_trimmed(self):
        """Whisper sometimes emits '、テキスト'; separator → space → trim → 'テキスト'."""
        assert _clean_telop_text("、テキスト") == "テキスト"

    def test_whitespace_only_returns_empty(self):
        assert _clean_telop_text("   ") == ""

    def test_empty_string_returns_empty(self):
        assert _clean_telop_text("") == ""

    def test_only_wrappers_returns_empty(self):
        """Removing all wrapper chars leaves nothing — result is empty."""
        assert _clean_telop_text("「」") == ""
        assert _clean_telop_text("（）") == ""

    def test_multi_line_three_parts_collapses(self):
        """Three logical lines joined by \\n each become a space, then collapsed."""
        assert _clean_telop_text("line1\nline2\nline3") == "line1 line2 line3"

    def test_emoji_passes_through_unchanged(self):
        """😊 is not in _SEP_RE or _WRAP_RE — it survives cleaning untouched."""
        assert _clean_telop_text("あ😊い") == "あ😊い"

    # ── Decimal-point preservation (T-bug: "GLM5.2" was rendering as "GLM5 2") ──
    # A "." between two digits is a decimal point, not sentence punctuation — only
    # strip it to a space when it ISN'T touching a digit on either side.

    def test_decimal_point_in_model_name_preserved(self):
        assert _clean_telop_text("GLM5.2とかディープシーク") == "GLM5.2とかディープシーク"

    def test_decimal_point_standalone_number_preserved(self):
        assert _clean_telop_text("pi is 3.14159") == "pi is 3.14159"

    def test_multiple_decimal_points_all_preserved(self):
        assert _clean_telop_text("V4.0とV4.5") == "V4.0とV4.5"

    def test_leading_decimal_point_preserved(self):
        """A period with a digit only on one side (".5") is still a decimal, not
        sentence punctuation — the fix only strips periods touching NO digit."""
        assert _clean_telop_text("the score is .5") == "the score is .5"

    def test_period_touching_a_digit_at_string_end_preserved(self):
        """A period right after a digit is ambiguous at string end (could be a cut-off
        decimal) — the conservative rule keeps it since it touches a digit at all,
        rather than guessing it's sentence-ending punctuation."""
        assert _clean_telop_text("version 5.") == "version 5."

    def test_decimal_point_before_more_text_preserved(self):
        assert _clean_telop_text("version 5.0 released") == "version 5.0 released"

    def test_english_sentence_period_still_becomes_space(self):
        """Regular English sentence-ending punctuation (not touching a digit) must
        still split — the fix is scoped to digit-adjacent periods only."""
        assert _clean_telop_text("done. Next sentence") == "done Next sentence"

    def test_period_at_string_end_with_no_digit_still_stripped(self):
        assert _clean_telop_text("end.") == "end"

    def test_decimal_point_mixed_with_real_sentence_boundary(self):
        assert _clean_telop_text("採用されたのはGLM5.2です。") == "採用されたのはGLM5.2です"


class TestCleanTelopTextIntegration:
    def test_pure_punctuation_entry_skipped(self, tmp_path):
        """An entry whose text reduces to empty after cleaning is dropped."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "。、！")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        titles = root.findall(".//title")
        assert len(titles) == 0

    def test_punctuation_stripped_in_output(self, tmp_path):
        """Punctuation in telop text is cleaned in the written FCPXML."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "Hello、World。")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        ts_ref = root.find(".//text/text-style")
        assert ts_ref is not None
        assert "、" not in ts_ref.text
        assert "。" not in ts_ref.text
        assert "Hello" in ts_ref.text
        assert "World" in ts_ref.text

    def test_jp_wrapper_stripped_in_output(self, tmp_path):
        """Japanese bracket wrappers are removed from written FCPXML."""
        segs = build_segments([], total_duration=30.0)
        entries = _make_entries([(1.0, 3.0, "「挨拶」")])
        _, root = _write_and_parse(tmp_path, entries, segs)
        ts_ref = root.find(".//text/text-style")
        assert "「" not in ts_ref.text
        assert "」" not in ts_ref.text
        assert "挨拶" in ts_ref.text

    def test_comma_becomes_wrap_break_at_1440p(self, tmp_path):
        """Regression: Whisper-style comma mid-sentence → space after cleaning →
        space dominates as line break (score 3.0) over particles.

        'チャンネル登録、よろしくお願いします' has a comma that becomes the
        natural sentence break.  At 1440p font_size=80 (max_em≈13.44) the cleaned
        text 'チャンネル登録 よろしくお願いします' (21 em) must wrap and the
        split must fall at the space (former comma) rather than mid-word.
        """
        raw = "チャンネル登録、よろしくお願いします"
        segs = build_segments([], total_duration=30.0)
        settings = {"fps": "60", "width": 2560, "height": 1440,
                    "font": "Hiragino Sans", "font_size": 80,
                    "font_color": "#CCA806", "position_y": -400, "line_spacing": -44}
        entries = _make_entries([(1.0, 5.0, raw)])
        _, root = _write_and_parse(tmp_path, entries, segs, settings)
        ts_ref = root.find(".//text/text-style")
        assert ts_ref is not None
        # Must wrap (21 em > 13.44)
        assert "\n" in ts_ref.text, f"Expected wrap, got: {ts_ref.text!r}"
        line1, line2 = ts_ref.text.split("\n", 1)
        # Line 1 must end with 登録 (the space/comma is consumed by the strip)
        assert line1 == "チャンネル登録", (
            f"Expected line1='チャンネル登録', got {line1!r}"
        )
        # Line 2 must start with よろしく
        assert line2.startswith("よろしく"), (
            f"Expected line2 starting with 'よろしく', got {line2!r}"
        )


# ── _to_rt() helper ───────────────────────────────────────────────────────────

class TestToRt:
    def test_24fps_1second(self):
        # Fraction(100, 2400) auto-simplifies to 1/24; 24 × 1/24 = 1 → "1s"
        fd = Fraction(100, 2400)
        result = _to_rt(1.0, fd)
        assert result == "1s"

    def test_24fps_zero(self):
        fd = Fraction(100, 2400)
        result = _to_rt(0.0, fd)
        assert result == "0s"

    def test_30fps_1second(self):
        # Fraction(100, 3000) auto-simplifies to 1/30; 30 × 1/30 = 1 → "1s"
        fd = Fraction(100, 3000)
        result = _to_rt(1.0, fd)
        assert result == "1s"

    def test_integer_denominator_omits_slash(self):
        # When denominator simplifies to 1, output is bare "Ns"
        fd = Fraction(1, 1)
        result = _to_rt(5.0, fd)
        assert result == "5s"

    def test_23976fps_1second_reduced(self):
        # 24 × (1001/24000) = 24024/24000 = 1001/1000 → "1001/1000s"
        fd = Fraction(1001, 24000)
        assert _to_rt(1.0, fd) == "1001/1000s"

    def test_2997fps_1second_reduced(self):
        fd = Fraction(1001, 30000)
        # 30 × (1001/30000) = 30030/30000 = 1001/1000 → "1001/1000s"
        assert _to_rt(1.0, fd) == "1001/1000s"

    def test_24fps_sub_second_snaps_to_frame(self):
        """0.5s at 24fps = 12 frames × 1/24 = 12/24 = 1/2 → '1/2s'."""
        fd = Fraction(100, 2400)   # 1/24s
        assert _to_rt(0.5, fd) == "1/2s"

    def test_24fps_non_frame_boundary_rounds(self):
        """0.1s at 24fps ≈ 2.4 frames → rounds to 2 frames → 2/24 = 1/12s."""
        fd = Fraction(100, 2400)   # 1/24s
        assert _to_rt(0.1, fd) == "1/12s"

    def test_60fps_sub_second(self):
        """0.5s at 60fps = 30 frames × 1/60 = 1/2 → '1/2s'."""
        fd = Fraction(100, 6000)   # 1/60s
        assert _to_rt(0.5, fd) == "1/2s"

    def test_25fps_1second(self):
        """1s at 25fps (PAL) = 25 frames × 1/25 = 1 → '1s'."""
        fd = Fraction(100, 2500)   # 1/25s
        assert _to_rt(1.0, fd) == "1s"

    def test_50fps_1second(self):
        """1s at 50fps (PAL HD) = 50 frames × 1/50 = 1 → '1s'."""
        fd = Fraction(100, 5000)   # 1/50s
        assert _to_rt(1.0, fd) == "1s"

    def test_5994fps_1second_reduced(self):
        """1s at 59.94fps: 60 × (1001/60000) = 1001/1000 → '1001/1000s'."""
        fd = Fraction(1001, 60000)
        assert _to_rt(1.0, fd) == "1001/1000s"


# ── _hex_to_fcpxml_color() ────────────────────────────────────────────────────

class TestHexToFcpxmlColor:
    def test_white(self):
        result = _hex_to_fcpxml_color("#FFFFFF")
        assert result == "1.0000 1.0000 1.0000 1"

    def test_black(self):
        result = _hex_to_fcpxml_color("#000000")
        assert result == "0.0000 0.0000 0.0000 1"

    def test_red(self):
        result = _hex_to_fcpxml_color("#FF0000")
        parts = result.split()
        assert float(parts[0]) == pytest.approx(1.0, abs=0.001)
        assert float(parts[1]) == pytest.approx(0.0, abs=0.001)
        assert float(parts[2]) == pytest.approx(0.0, abs=0.001)
        assert parts[3] == "1"

    def test_always_has_alpha_1(self):
        result = _hex_to_fcpxml_color("#CCA806")
        assert result.endswith(" 1")

    # ── Shorthand + validation ────────────────────────────────────────────────

    def test_rgb_shorthand_expanded(self):
        """3-digit '#RGB' shorthand is expanded to 6-digit '#RRGGBB'."""
        assert _hex_to_fcpxml_color("#FFF") == _hex_to_fcpxml_color("#FFFFFF")

    def test_rgb_shorthand_mixed(self):
        assert _hex_to_fcpxml_color("#F80") == _hex_to_fcpxml_color("#FF8800")

    def test_no_hash_prefix_accepted(self):
        """Leading '#' is optional."""
        assert _hex_to_fcpxml_color("FFFFFF") == "1.0000 1.0000 1.0000 1"

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid hex color"):
            _hex_to_fcpxml_color("")

    def test_short_hex_raises_value_error(self):
        """A 4-digit string is not a valid 3- or 6-digit hex color."""
        with pytest.raises(ValueError, match="Invalid hex color"):
            _hex_to_fcpxml_color("#FFFF")

    def test_non_hex_chars_raise_value_error(self):
        with pytest.raises(ValueError, match="Invalid hex color"):
            _hex_to_fcpxml_color("#GGHHII")

    def test_named_color_raises_value_error(self):
        """Named CSS colors like 'red' are not accepted."""
        with pytest.raises(ValueError, match="Invalid hex color"):
            _hex_to_fcpxml_color("red")

    def test_lowercase_6digit_hex_accepted(self):
        """Lowercase hex digits are accepted (int('#ff...', 16) works)."""
        assert _hex_to_fcpxml_color("#ffffff") == "1.0000 1.0000 1.0000 1"

    def test_lowercase_shorthand_accepted(self):
        """3-digit lowercase shorthand '#fff' → '#ffffff'."""
        assert _hex_to_fcpxml_color("#fff") == "1.0000 1.0000 1.0000 1"

    def test_input_whitespace_is_stripped(self):
        """Leading/trailing whitespace around the hex string is stripped."""
        assert _hex_to_fcpxml_color("  #FFFFFF  ") == "1.0000 1.0000 1.0000 1"

    def test_mixed_case_hex_accepted(self):
        """Mixed-case digits work (e.g., browser color pickers)."""
        assert _hex_to_fcpxml_color("#FfFfFf") == "1.0000 1.0000 1.0000 1"


# ── generate_telop_fcpxml — malformed entry robustness ───────────────────────

class TestMalformedEntries:
    """Entries missing 'start' or 'end' keys must be silently skipped."""

    def _settings(self):
        return {
            "fps": "24", "width": 1920, "height": 1080,
            "font": "Hiragino Sans", "font_size": 80,
            "font_color": "#FFFFFF", "position_y": -400, "line_spacing": -44,
        }

    def test_entry_missing_start_skipped(self, tmp_path):
        out = tmp_path / "out.fcpxml"
        bad = {"end": 5.0, "text": "no start"}
        good = {"id": "e1", "start": 0.0, "end": 3.0, "text": "good"}
        generate_telop_fcpxml(
            [bad, good], [], 10.0, self._settings(), "stem", out,
            use_source_timing=True,
        )
        tree = ET.parse(out)
        titles = tree.findall(".//title")
        assert len(titles) == 1
        assert titles[0].get("name", "").startswith("good")

    def test_entry_missing_end_skipped(self, tmp_path):
        out = tmp_path / "out.fcpxml"
        bad = {"id": "e1", "start": 0.0, "text": "no end"}
        good = {"id": "e2", "start": 1.0, "end": 4.0, "text": "good entry"}
        generate_telop_fcpxml(
            [bad, good], [], 10.0, self._settings(), "stem", out,
            use_source_timing=True,
        )
        tree = ET.parse(out)
        titles = tree.findall(".//title")
        assert len(titles) == 1

    def test_all_entries_malformed_produces_empty_spine(self, tmp_path):
        out = tmp_path / "out.fcpxml"
        generate_telop_fcpxml(
            [{"text": "no times"}, {"id": "x"}], [], 10.0,
            self._settings(), "stem", out,
            use_source_timing=True,
        )
        tree = ET.parse(out)
        titles = tree.findall(".//title")
        assert titles == []

    def test_malformed_entries_skipped_with_source_timing(self, tmp_path):
        out = tmp_path / "out.fcpxml"
        bad = {"text": "missing both"}
        good = {"id": "e1", "start": 0.0, "end": 2.0, "text": "ok"}
        generate_telop_fcpxml(
            [bad, good], [], 10.0, self._settings(), "stem", out,
            use_source_timing=True,
        )
        tree = ET.parse(out)
        titles = tree.findall(".//title")
        assert len(titles) == 1

    def test_malformed_entries_skipped_with_edited_timing(self, tmp_path):
        """Edited-timing path also skips entries without start/end keys."""
        from preprod.segments import Segment
        seg = Segment(source_start=0.0, source_end=10.0)
        out = tmp_path / "out.fcpxml"
        bad = {"text": "no times at all"}
        good = {"id": "e2", "start": 1.0, "end": 4.0, "text": "valid entry"}
        generate_telop_fcpxml(
            [bad, good], [seg], 10.0, self._settings(), "stem", out,
        )
        tree = ET.parse(out)
        titles = tree.findall(".//title")
        assert len(titles) == 1

    def test_inverted_timestamps_skipped(self, tmp_path):
        """Entry where end < start produces negative duration — must be silently skipped."""
        out = tmp_path / "out.fcpxml"
        inverted = {"id": "e1", "start": 5.0, "end": 2.0, "text": "backwards"}
        good     = {"id": "e2", "start": 1.0, "end": 4.0, "text": "normal"}
        generate_telop_fcpxml(
            [inverted, good], [], 10.0, self._settings(), "stem", out,
            use_source_timing=True,
        )
        tree = ET.parse(out)
        titles = tree.findall(".//title")
        assert len(titles) == 1
        assert titles[0].get("name", "").startswith("normal")

    def test_too_short_entry_skipped(self, tmp_path):
        """Entry whose duration < 20ms (two frames at most fps) must be skipped."""
        out = tmp_path / "out.fcpxml"
        tiny   = {"id": "e1", "start": 1.0, "end": 1.015, "text": "blink"}  # 15ms < 20ms
        normal = {"id": "e2", "start": 2.0, "end": 5.0,   "text": "normal"}
        generate_telop_fcpxml(
            [tiny, normal], [], 10.0, self._settings(), "stem", out,
            use_source_timing=True,
        )
        tree = ET.parse(out)
        titles = tree.findall(".//title")
        assert len(titles) == 1
        assert titles[0].get("name", "").startswith("normal")

    def test_entry_at_one_frame_duration_kept(self, tmp_path):
        """An entry with exactly 1 frame duration at 24fps must be kept.

        1/24s ≈ 0.04167s → round(0.04167 × 24) = round(1.0) = 1 ≥ 1 → kept.
        Ensures the frame-boundary guard does not over-filter the minimum
        valid FCP title duration.
        """
        from fractions import Fraction
        one_frame = float(Fraction(1, 24))   # ≈ 0.04167s
        out = tmp_path / "out.fcpxml"
        threshold = {"id": "e1", "start": 1.0, "end": 1.0 + one_frame, "text": "one-frame"}
        generate_telop_fcpxml(
            [threshold], [], 10.0, self._settings(), "stem", out,
            use_source_timing=True,
        )
        tree = ET.parse(out)
        titles = tree.findall(".//title")
        assert len(titles) == 1

    def test_inverted_timestamps_skipped_edited_path(self, tmp_path):
        """Inverted timestamps (end < start) produce no overlap in map_span_to_output
        and must be silently skipped in the use_source_timing=False (edited) path."""
        from preprod.segments import Segment
        seg = Segment(source_start=0.0, source_end=10.0)
        out = tmp_path / "out.fcpxml"
        inverted = {"id": "e1", "start": 5.0, "end": 2.0, "text": "backwards"}
        good     = {"id": "e2", "start": 1.0, "end": 4.0, "text": "normal"}
        generate_telop_fcpxml(
            [inverted, good], [seg], 10.0, self._settings(), "stem", out,
        )
        tree = ET.parse(out)
        titles = tree.findall(".//title")
        assert len(titles) == 1
        assert titles[0].get("name", "").startswith("normal")

    def test_too_short_entry_skipped_edited_path(self, tmp_path):
        """Entry < 20ms duration is skipped via the out_end - out_start < 0.02
        guard in the use_source_timing=False (edited) path."""
        from preprod.segments import Segment
        seg = Segment(source_start=0.0, source_end=10.0)
        out = tmp_path / "out.fcpxml"
        tiny   = {"id": "e1", "start": 1.0, "end": 1.015, "text": "blink"}
        normal = {"id": "e2", "start": 2.0, "end": 5.0,   "text": "normal"}
        generate_telop_fcpxml(
            [tiny, normal], [seg], 10.0, self._settings(), "stem", out,
        )
        tree = ET.parse(out)
        titles = tree.findall(".//title")
        assert len(titles) == 1
        assert titles[0].get("name", "").startswith("normal")


# ── Integration test against real sample.mov ──────────────────────────────────

SAMPLE_MOV = Path(__file__).parent / "fixtures" / "sample.mov"


@pytest.mark.skipif(not SAMPLE_MOV.exists(), reason="sample.mov fixture not present")
class TestTelopRealSampleMov:
    """Probe sample.mov (2560×1440, 60fps) → generate_telop_fcpxml → validate FCPXML.

    Exercises the full pipeline with real metadata, catching regressions that
    unit tests with hard-coded settings might miss.
    """

    def _settings(self):
        """Settings matching what the frontend sends for sample.mov's resolution."""
        return {
            "fps": "60",
            "width": 2560,
            "height": 1440,
            "font": "Hiragino Sans",
            "font_size": 80,
            "font_color": "#CCA806",
            "position_y": -400,
            "line_spacing": -44,
        }

    def test_format_name_for_1440p_60fps(self, tmp_path):
        """2560×1440 60fps must produce FFVideoFormat1440p60 in telop FCPXML."""
        entries = [{"id": "e1", "start": 1.0, "end": 5.0, "text": "テスト"}]
        out = tmp_path / "telop.fcpxml"
        generate_telop_fcpxml(
            entries, [], 10.0, self._settings(), "test", out,
            use_source_timing=True,
        )
        root = ET.parse(out).getroot()
        fmt_name = root.find("resources/format").get("name")
        assert fmt_name == "FFVideoFormat1440p60", (
            f"Expected FFVideoFormat1440p60, got {fmt_name!r}"
        )

    def test_frame_duration_for_60fps(self, tmp_path):
        """frameDuration for 60fps must be '1/60s'."""
        entries = [{"id": "e1", "start": 1.0, "end": 5.0, "text": "テスト"}]
        out = tmp_path / "telop.fcpxml"
        generate_telop_fcpxml(
            entries, [], 10.0, self._settings(), "test", out,
            use_source_timing=True,
        )
        root = ET.parse(out).getroot()
        frame_dur = root.find("resources/format").get("frameDuration")
        assert frame_dur == "1/60s", f"Expected 1/60s for 60fps, got {frame_dur!r}"

    def test_title_timing_matches_entry(self, tmp_path):
        """Title offset and duration must correctly reflect the entry timestamps."""
        entries = [{"id": "e1", "start": 2.0, "end": 7.0, "text": "テスト"}]
        out = tmp_path / "telop.fcpxml"
        generate_telop_fcpxml(
            entries, [], 30.0, self._settings(), "test", out,
            use_source_timing=True,
        )
        root = ET.parse(out).getroot()
        title = root.find(".//title")
        assert title is not None
        # 2s @ 60fps = 120 frames × 1/60s = 2s → "2s"
        assert title.get("offset") == "2s"
        # duration 5s @ 60fps = 300 frames × 1/60s = 5s → "5s"
        assert title.get("duration") == "5s"

    def test_long_jp_text_wraps_for_1440p(self, tmp_path):
        """2560×1440, font_size=80 → max_em≈13.44; 15 CJK chars (15.0 em) must wrap."""
        text = "あ" * 15   # 15.0 em > 13.44 max_em
        entries = [{"id": "e1", "start": 1.0, "end": 5.0, "text": text}]
        out = tmp_path / "telop.fcpxml"
        generate_telop_fcpxml(
            entries, [], 10.0, self._settings(), "test", out,
            use_source_timing=True,
        )
        root = ET.parse(out).getroot()
        ts_ref = root.find(".//text/text-style")
        assert ts_ref is not None
        assert "\n" in ts_ref.text

    def test_short_jp_text_not_wrapped_for_1440p(self, tmp_path):
        """2560×1440, font_size=80 → max_em≈13.44; 13 CJK chars (13.0 em) must NOT wrap."""
        text = "あ" * 13   # 13.0 em < 13.44 max_em
        entries = [{"id": "e1", "start": 1.0, "end": 5.0, "text": text}]
        out = tmp_path / "telop.fcpxml"
        generate_telop_fcpxml(
            entries, [], 10.0, self._settings(), "test", out,
            use_source_timing=True,
        )
        root = ET.parse(out).getroot()
        ts_ref = root.find(".//text/text-style")
        assert ts_ref is not None
        assert "\n" not in ts_ref.text

    def test_realistic_jp_sentence_breaks_at_particle_for_1440p(self, tmp_path):
        """Realistic JP sentence wraps at a particle with 1440p/60fps settings.

        'しばらくは完全に本気を出せない' (15 chars = 15.0 em) exceeds max_em≈13.44
        and must split after a linguistic particle, not mid-word.
        """
        text = "しばらくは完全に本気を出せない"
        entries = [{"id": "e1", "start": 1.0, "end": 5.0, "text": text}]
        out = tmp_path / "telop.fcpxml"
        generate_telop_fcpxml(
            entries, [], 10.0, self._settings(), "test", out,
            use_source_timing=True,
        )
        root = ET.parse(out).getroot()
        ts_ref = root.find(".//text/text-style")
        assert ts_ref is not None
        assert "\n" in ts_ref.text
        line1 = ts_ref.text.split("\n")[0]
        assert line1[-1] in "はがをにでとももかのへやよねわて", (
            f"Expected wrap after a particle, got first line: {line1!r}"
        )

    def test_cut_remapping_with_1440p_settings(self, tmp_path):
        """Entry after a removed segment is correctly remapped in the output timeline.

        Remove 5–10s (5s cut); entry at 12–16s source → offset 7s in output.
        Uses use_source_timing=False (the default export path).
        """
        segs = build_segments([(5.0, 10.0)], total_duration=30.0)
        entries = [{"id": "e1", "start": 12.0, "end": 16.0, "text": "テスト"}]
        out = tmp_path / "telop.fcpxml"
        generate_telop_fcpxml(
            entries, segs, 30.0, self._settings(), "test", out,
            use_source_timing=False,
        )
        root = ET.parse(out).getroot()
        title = root.find(".//title")
        assert title is not None, "entry after the cut must not be dropped"
        # Source 12s − 5s cut = 7s output offset
        got = _rt_to_seconds(title.get("offset"))
        assert got == pytest.approx(7.0, abs=1.0 / 60), (
            f"Expected offset ≈ 7.0s after remapping, got {got:.4f}s"
        )
        # Duration unchanged: 4s
        dur = _rt_to_seconds(title.get("duration"))
        assert dur == pytest.approx(4.0, abs=1.0 / 60)

    def test_two_cuts_entry_after_both_remapped_correctly(self, tmp_path):
        """Entry after two removed segments is shifted by the combined cut duration.

        Remove 2–4s (2s) and 8–12s (4s); entry at 14–18s source →
        output offset = 14 − 2 − 4 = 8s.
        """
        segs = build_segments([(2.0, 4.0), (8.0, 12.0)], total_duration=30.0)
        entries = [{"id": "e1", "start": 14.0, "end": 18.0, "text": "二回カット"}]
        out = tmp_path / "telop.fcpxml"
        generate_telop_fcpxml(
            entries, segs, 30.0, self._settings(), "test", out,
            use_source_timing=False,
        )
        root = ET.parse(out).getroot()
        title = root.find(".//title")
        assert title is not None, "entry after two cuts must not be dropped"
        got = _rt_to_seconds(title.get("offset"))
        assert got == pytest.approx(8.0, abs=1.0 / 60), (
            f"Expected offset ≈ 8.0s after two cuts, got {got:.4f}s"
        )

    def test_full_pipeline_with_probed_metadata(self, tmp_path):
        """probe_media(sample.mov) → generate_telop_fcpxml → valid FCPXML.

        Anchors the class's hardcoded assumptions to the real fixture file, then
        exercises the complete pipeline: probe → settings → generate → parse.
        Unique value: settings are derived from probed metadata, not hardcoded,
        so any fixture change that breaks the assumptions is caught immediately.
        """
        from preprod.probe import probe_media

        media = probe_media(SAMPLE_MOV)

        # Verify the fixture is still 2560×1440 @ 60fps.  If sample.mov is ever
        # replaced with a different file, these assertions catch the mismatch
        # before the rest of the class silently tests the wrong resolution.
        assert media.video_width == 2560
        assert media.video_height == 1440
        assert media.frame_rate == 60   # Fraction(60,1) == int 60 in Python

        # Build settings from probed metadata — mirrors what the frontend sends.
        settings = {
            "fps":          str(int(media.frame_rate)),
            "width":        media.video_width,
            "height":       media.video_height,
            "font":         "Hiragino Sans",
            "font_size":    80,
            "font_color":   "#CCA806",
            "position_y":   -400,
            "line_spacing": -44,
        }

        # max_em = 2560 × 0.42 / 80 = 13.44
        # long_text: 15 CJK chars = 15.0 em → must wrap
        # short_text: 5 CJK chars =  5.0 em → must NOT wrap
        long_text  = "しばらくは完全に本気を出せない"   # 15 chars, realistic JP subtitle
        short_text = "短いテスト"                       # 5 chars

        entries = [
            {"id": "e1", "start": 10.0, "end": 15.0, "text": long_text},
            {"id": "e2", "start": 20.0, "end": 23.0, "text": short_text},
        ]
        out = tmp_path / "telop_integration.fcpxml"
        generate_telop_fcpxml(
            entries, [], media.duration, settings, "sample", out,
            use_source_timing=True,
        )
        root = ET.parse(out).getroot()

        # Format element reflects probed resolution and fps.
        fmt = root.find("resources/format")
        assert fmt.get("name") == "FFVideoFormat1440p60"
        assert fmt.get("frameDuration") == "1/60s"
        assert fmt.get("width")  == "2560"
        assert fmt.get("height") == "1440"

        titles = root.findall(".//title")
        assert len(titles) == 2, f"Expected 2 title clips, got {len(titles)}"

        # Long entry: must have been wrapped (15 em > max_em 13.44).
        ts1 = titles[0].find("text/text-style")
        assert "\n" in ts1.text, f"Long text should wrap; got {ts1.text!r}"

        # Short entry: must NOT be wrapped (5 em ≤ max_em 13.44).
        ts2 = titles[1].find("text/text-style")
        assert "\n" not in ts2.text, f"Short text should not wrap; got {ts2.text!r}"

        # Both titles carry the position param at position_y=-400.
        for title in titles:
            pos = title.find("param[@name='Position']")
            assert pos is not None, "Position param missing from title"
            assert pos.get("value") == "0 -400", (
                f"Expected position '0 -400', got {pos.get('value')!r}"
            )

        # Title 1 offset: 10s @ 60fps = 600 frames × 1/60s = "10s".
        assert titles[0].get("offset") == "10s"
        # Title 2 offset: 20s @ 60fps = 1200 frames × 1/60s = "20s".
        assert titles[1].get("offset") == "20s"
