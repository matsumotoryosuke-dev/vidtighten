"""Tests for fcpxml_cut.py — generate_roughcut_fcpxml() and helpers."""

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import pytest

from preprod.fcpxml_cut import (
    _file_url,
    _format_name,
    _frac_rt,
    _frame_dur_str,
    _to_rt,
    generate_roughcut_fcpxml,
)
from preprod.probe import MediaInfo
from preprod.segments import Segment, build_segments


# ── helpers ───────────────────────────────────────────────────────────────────

def _media(
    path: Path,
    duration: float = 30.0,
    has_video: bool = True,
    has_audio: bool = True,
    width: int = 1920,
    height: int = 1080,
    fps: Fraction = Fraction(24, 1),
    sample_rate: int = 48000,
    audio_channels: Optional[int] = None,
) -> MediaInfo:
    return MediaInfo(
        path=path,
        duration=duration,
        has_video=has_video,
        has_audio=has_audio,
        video_width=width,
        video_height=height,
        frame_rate=fps,
        sample_rate=sample_rate,
        audio_channels=audio_channels,
        codec_video="h264" if has_video else None,
        codec_audio="aac" if has_audio else None,
    )


def _write_and_parse(tmp_path, segments, media):
    out = tmp_path / "cut.fcpxml"
    generate_roughcut_fcpxml(segments, media, out)
    return out, ET.parse(out).getroot()


# ── _to_rt ────────────────────────────────────────────────────────────────────

class TestToRt:
    FPS24 = Fraction(24, 1)
    FPS30 = Fraction(30, 1)
    FPS2997 = Fraction(30000, 1001)

    def test_zero_seconds_is_0s(self):
        assert _to_rt(0.0, self.FPS24) == "0s"

    def test_one_second_at_24fps(self):
        # 1s = 24 frames × (1/24)s = 1s  →  "1s"
        assert _to_rt(1.0, self.FPS24) == "1s"

    def test_two_seconds_at_24fps(self):
        assert _to_rt(2.0, self.FPS24) == "2s"

    def test_half_second_at_24fps(self):
        # 0.5s = 12 frames × (1/24)s = 12/24 = 1/2
        # denominator != 1 → "1/2s"
        assert _to_rt(0.5, self.FPS24) == "1/2s"

    def test_snapped_to_frame_at_30fps(self):
        # 1.0s = 30 frames at 30fps → "1s"
        assert _to_rt(1.0, self.FPS30) == "1s"

    def test_fractional_seconds_at_2997fps(self):
        # 1.0s at 29.97fps ≈ 29.97 frames → rounds to 30 frames
        # 30 frames × (1001/30000)s = 30030/30000 = 1001/1000s
        result = _to_rt(1.0, self.FPS2997)
        assert result.endswith("s")
        # Parse and verify it's close to 1s
        num, rest = result.rstrip("s").split("/") if "/" in result else (result.rstrip("s"), "1")
        value = float(num) / float(rest)
        assert value == pytest.approx(1.0, abs=0.01)


# ── _frac_rt ─────────────────────────────────────────────────────────────────

class TestFracRt:
    """_frac_rt formats an exact Fraction as an FCPXML rational time string."""

    def test_zero_is_0s(self):
        # Fraction(0).denominator = 1 → integer form
        assert _frac_rt(Fraction(0)) == "0s"

    def test_integer_fraction(self):
        assert _frac_rt(Fraction(2)) == "2s"

    def test_one_as_fraction(self):
        assert _frac_rt(Fraction(1)) == "1s"

    def test_fractional_value(self):
        # 1/2 s → "1/2s"
        assert _frac_rt(Fraction(1, 2)) == "1/2s"

    def test_already_reduced_fraction(self):
        # Fraction(2, 4) auto-reduces to 1/2
        assert _frac_rt(Fraction(2, 4)) == "1/2s"

    def test_canon_timecode_at_2398(self):
        # Timecode 08:39:17:08 → Fraction(11695684, 375) at 23.976fps
        assert _frac_rt(Fraction(11695684, 375)) == "11695684/375s"

    def test_segment_offset_at_2397(self):
        # 10s at 23.976fps = 240 frames × (1001/24000) = 1001/100s
        assert _frac_rt(Fraction(1001, 100)) == "1001/100s"


# ── _format_name ─────────────────────────────────────────────────────────────

class TestFormatName:
    def test_video_1080p24(self, tmp_path):
        m = _media(tmp_path / "v.mp4", width=1920, height=1080, fps=Fraction(24, 1))
        assert _format_name(m, m.frame_rate) == "FFVideoFormat1080p24"

    def test_video_4k_uhd_30(self, tmp_path):
        # 3840×2160 must use FCP's explicit WxH preset name, not height-only.
        m = _media(tmp_path / "v.mp4", width=3840, height=2160, fps=Fraction(30, 1))
        assert _format_name(m, m.frame_rate) == "FFVideoFormat3840x2160p30"

    def test_video_4k_uhd_2398(self, tmp_path):
        # Canonical Canon 4K HEVC case — 3840×2160 @ 23.976fps.
        m = _media(tmp_path / "v.mp4", width=3840, height=2160, fps=Fraction(24000, 1001))
        assert _format_name(m, m.frame_rate) == "FFVideoFormat3840x2160p2398"

    def test_video_4k_dci_24(self, tmp_path):
        # 4096×2160 DCI 4K uses explicit WxH name.
        m = _media(tmp_path / "v.mp4", width=4096, height=2160, fps=Fraction(24, 1))
        assert _format_name(m, m.frame_rate) == "FFVideoFormat4096x2160p24"

    def test_video_1440p_custom_fallback(self, tmp_path):
        # 2560×1440 is not a built-in FCP preset; height-only fallback creates
        # a Custom sequence which FCP accepts without error.
        m = _media(tmp_path / "v.mp4", width=2560, height=1440, fps=Fraction(60, 1))
        assert _format_name(m, m.frame_rate) == "FFVideoFormat1440p60"

    def test_video_2997fps(self, tmp_path):
        m = _media(tmp_path / "v.mp4", width=1920, height=1080, fps=Fraction(30000, 1001))
        assert _format_name(m, m.frame_rate) == "FFVideoFormat1080p2997"

    def test_video_1080p25_pal(self, tmp_path):
        # 25fps is common in PAL regions; not in _FPS_ID → uses integer str path
        m = _media(tmp_path / "v.mp4", width=1920, height=1080, fps=Fraction(25, 1))
        assert _format_name(m, m.frame_rate) == "FFVideoFormat1080p25"

    def test_video_1080p50_pal(self, tmp_path):
        # 50fps PAL — also uses the integer str fallback
        m = _media(tmp_path / "v.mp4", width=1920, height=1080, fps=Fraction(50, 1))
        assert _format_name(m, m.frame_rate) == "FFVideoFormat1080p50"

    def test_audio_only_returns_undefined(self, tmp_path):
        m = _media(tmp_path / "a.aac", has_video=False, has_audio=True)
        assert _format_name(m, None) == "FFVideoFormatRateUndefined"

    def test_no_frame_rate_returns_undefined(self, tmp_path):
        m = _media(tmp_path / "v.mp4")
        assert _format_name(m, None) == "FFVideoFormatRateUndefined"


# ── _frame_dur_str ────────────────────────────────────────────────────────────

class TestFrameDurStr:
    """frameDuration attribute written to <format> — must match FCP's registry."""

    def test_24fps(self):
        assert _frame_dur_str(Fraction(24, 1)) == "1/24s"

    def test_23976fps(self):
        # 1001/24000 is the exact frame duration for 23.976fps
        assert _frame_dur_str(Fraction(24000, 1001)) == "1001/24000s"

    def test_30fps(self):
        assert _frame_dur_str(Fraction(30, 1)) == "1/30s"

    def test_2997fps(self):
        assert _frame_dur_str(Fraction(30000, 1001)) == "1001/30000s"

    def test_25fps(self):
        assert _frame_dur_str(Fraction(25, 1)) == "1/25s"

    def test_50fps(self):
        assert _frame_dur_str(Fraction(50, 1)) == "1/50s"

    def test_60fps(self):
        assert _frame_dur_str(Fraction(60, 1)) == "1/60s"

    def test_format_element_has_correct_frame_duration(self, tmp_path):
        """Integration: <format frameDuration> in generated FCPXML matches fps."""
        m = _media(tmp_path / "v.mp4", fps=Fraction(24000, 1001))
        segs = build_segments([], total_duration=m.duration)
        _, root = _write_and_parse(tmp_path, segs, m)
        assert root.find(".//format").get("frameDuration") == "1001/24000s"


# ── _file_url ─────────────────────────────────────────────────────────────────

class TestFileUrl:
    def test_starts_with_file_scheme(self, tmp_path):
        p = tmp_path / "v.mp4"
        p.write_bytes(b"\x00")
        assert _file_url(p).startswith("file://")

    def test_spaces_encoded(self, tmp_path):
        p = tmp_path / "my video.mp4"
        p.write_bytes(b"\x00")
        url = _file_url(p)
        assert " " not in url
        assert "%20" in url

    def test_japanese_filename_encoded(self, tmp_path):
        """Non-ASCII (CJK) characters in filenames must be percent-encoded."""
        p = tmp_path / "動画ファイル.mp4"
        p.write_bytes(b"\x00")
        url = _file_url(p)
        # URL must contain only ASCII — no raw CJK bytes
        assert all(ord(c) < 128 for c in url)
        assert url.startswith("file://")


# ── generate_roughcut_fcpxml — file creation ──────────────────────────────────

class TestGenerateRoughcutFcpxml:
    def test_output_file_created(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([], total_duration=30.0)
        out, _ = _write_and_parse(tmp_path, segs, _media(src))
        assert out.exists()

    def test_xml_declaration_present(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([], total_duration=30.0)
        out = tmp_path / "cut.fcpxml"
        generate_roughcut_fcpxml(segs, _media(src), out)
        content = out.read_text()
        assert "<?xml" in content

    def test_doctype_present(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([], total_duration=30.0)
        out = tmp_path / "cut.fcpxml"
        generate_roughcut_fcpxml(segs, _media(src), out)
        content = out.read_text()
        assert "<!DOCTYPE fcpxml>" in content


# ── XML structure ─────────────────────────────────────────────────────────────

class TestXmlStructure:
    def _src_and_segs(self, tmp_path, removal=None, duration=30.0):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments(removal or [], total_duration=duration)
        return src, segs

    def test_root_element_is_fcpxml(self, tmp_path):
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        assert root.tag == "fcpxml"

    def test_fcpxml_version_is_1_11(self, tmp_path):
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        assert root.get("version") == "1.11"

    def test_resources_element_present(self, tmp_path):
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        assert root.find("resources") is not None

    def test_format_element_in_resources(self, tmp_path):
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        assert root.find("resources/format") is not None

    def test_asset_element_in_resources(self, tmp_path):
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        assert root.find("resources/asset") is not None

    def test_asset_has_media_rep(self, tmp_path):
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        assert root.find("resources/asset/media-rep") is not None

    def test_media_rep_src_starts_with_file_scheme(self, tmp_path):
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        rep = root.find("resources/asset/media-rep")
        assert rep.get("src").startswith("file://")

    def test_library_event_project_present(self, tmp_path):
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        assert root.find("library/event/project") is not None

    def test_sequence_in_project(self, tmp_path):
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        assert root.find("library/event/project/sequence") is not None

    def test_spine_in_sequence(self, tmp_path):
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        assert root.find("library/event/project/sequence/spine") is not None

    # ── timecode format (drop-frame vs non-drop-frame) ────────────────────────

    def test_tc_format_ndf_for_24fps(self, tmp_path):
        """24 fps uses NDF timecode on both sequence and asset-clips."""
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src, fps=Fraction(24, 1)))
        seq = root.find(".//sequence")
        assert seq.get("tcFormat") == "NDF"
        for clip in root.findall(".//asset-clip"):
            assert clip.get("tcFormat") == "NDF"

    def test_tc_format_df_for_2997fps(self, tmp_path):
        """29.97 fps (30000/1001) uses DF timecode on sequence and asset-clips."""
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src, fps=Fraction(30000, 1001)))
        seq = root.find(".//sequence")
        assert seq.get("tcFormat") == "DF", (
            f"29.97 fps sequence must use DF; got {seq.get('tcFormat')!r}"
        )
        for clip in root.findall(".//asset-clip"):
            assert clip.get("tcFormat") == "DF"

    def test_tc_format_df_for_5994fps(self, tmp_path):
        """59.94 fps (60000/1001) uses DF timecode on sequence and asset-clips."""
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src, fps=Fraction(60000, 1001)))
        seq = root.find(".//sequence")
        assert seq.get("tcFormat") == "DF"

    def test_tc_format_ndf_for_30fps(self, tmp_path):
        """30 fps (not 29.97 NTSC) uses NDF timecode."""
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src, fps=Fraction(30, 1)))
        assert root.find(".//sequence").get("tcFormat") == "NDF"

    def test_tc_format_ndf_for_23976fps(self, tmp_path):
        """23.976 fps uses NDF timecode (not drop-frame)."""
        src, segs = self._src_and_segs(tmp_path)
        _, root = _write_and_parse(tmp_path, segs, _media(src, fps=Fraction(24000, 1001)))
        assert root.find(".//sequence").get("tcFormat") == "NDF"


# ── Asset-clip count and positioning ─────────────────────────────────────────

class TestAssetClips:
    def test_no_removals_produces_one_clip(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        clips = root.findall("library/event/project/sequence/spine/asset-clip")
        assert len(clips) == 1

    def test_one_removal_produces_two_clips(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        # Remove 10–20 s → keep 0–10 and 20–30
        segs = build_segments([(10.0, 20.0)], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        clips = root.findall("library/event/project/sequence/spine/asset-clip")
        assert len(clips) == 2

    def test_two_removals_produces_three_clips(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([(5.0, 10.0), (20.0, 25.0)], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        clips = root.findall("library/event/project/sequence/spine/asset-clip")
        assert len(clips) == 3

    def test_clips_have_ref_to_asset(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        clips = root.findall("library/event/project/sequence/spine/asset-clip")
        assert all(c.get("ref") == "r2" for c in clips)

    def test_first_clip_starts_at_source_zero(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        clip = root.find("library/event/project/sequence/spine/asset-clip")
        assert clip.get("start") == "0s"

    def test_first_clip_offset_is_zero(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        clip = root.find("library/event/project/sequence/spine/asset-clip")
        assert clip.get("offset") == "0s"

    def test_second_clip_offset_equals_first_clip_duration(self, tmp_path):
        """After a removal, clip 2's timeline offset must equal clip 1's duration.

        Remove 10–20 s from a 30 s clip at 24 fps:
          clip 1 → source 0–10 s, duration 10 s, offset 0 s
          clip 2 → source 20–30 s, duration 10 s, offset 10 s
        """
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([(10.0, 20.0)], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        clips = root.findall("library/event/project/sequence/spine/asset-clip")
        assert len(clips) == 2
        # Clip 1: 10 s at 24fps → Fraction(240,24) reduces to Fraction(10,1);
        # denominator == 1 → _frac_rt formats it as "10s"
        assert clips[0].get("offset") == "0s"
        assert clips[1].get("offset") == "10s"

    def test_second_clip_start_reflects_removal_boundary(self, tmp_path):
        """Clip 2's source start must point to the end of the removed region.

        Remove 10–20 s from a 30 s clip at 24 fps, no embedded timecode:
          clip 2 → start in source = 20 s → "20s"
        """
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([(10.0, 20.0)], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        clips = root.findall("library/event/project/sequence/spine/asset-clip")
        assert len(clips) == 2
        assert clips[1].get("start") == "20s"


# ── Asset video/audio source attributes ──────────────────────────────────────

class TestAssetSourceAttrs:
    """Regression: FCP rejects assets without videoSources/audioSources attrs.

    Commit 048f651 added these to fix 'Invalid edit with no respective media'.
    """

    def _root(self, tmp_path, **media_kwargs):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        m = _media(src, **media_kwargs)
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        return root

    def test_video_asset_has_videoSources_1(self, tmp_path):
        asset = self._root(tmp_path).find("resources/asset")
        assert asset.get("videoSources") == "1"

    def test_video_asset_has_audioSources_1(self, tmp_path):
        asset = self._root(tmp_path).find("resources/asset")
        assert asset.get("audioSources") == "1"

    def test_video_asset_has_audioChannels(self, tmp_path):
        asset = self._root(tmp_path).find("resources/asset")
        assert asset.get("audioChannels") is not None

    def test_video_asset_audioChannels_defaults_to_2(self, tmp_path):
        # When MediaInfo.audio_channels is None (probe didn't report it),
        # fcpxml_cut falls back to 2 (standard stereo).
        asset = self._root(tmp_path).find("resources/asset")  # audio_channels=None
        assert asset.get("audioChannels") == "2"

    def test_video_asset_audioChannels_explicit_stereo(self, tmp_path):
        asset = self._root(tmp_path, audio_channels=2).find("resources/asset")
        assert asset.get("audioChannels") == "2"

    def test_video_asset_audioChannels_explicit_surround(self, tmp_path):
        # 5.1 audio: 6 channels
        asset = self._root(tmp_path, audio_channels=6).find("resources/asset")
        assert asset.get("audioChannels") == "6"

    def test_video_asset_audioChannels_multi_track_four(self, tmp_path):
        # 4K camera with two stereo tracks → 4 channels total
        asset = self._root(tmp_path, audio_channels=4).find("resources/asset")
        assert asset.get("audioChannels") == "4"

    def test_video_asset_audioRate_matches_sample_rate(self, tmp_path):
        asset = self._root(tmp_path, sample_rate=48000).find("resources/asset")
        assert asset.get("audioRate") == "48000"

    def test_audio_only_asset_no_videoSources(self, tmp_path):
        root = self._root(tmp_path, has_video=False, has_audio=True)
        asset = root.find("resources/asset")
        assert asset.get("videoSources") is None

    def test_audio_only_asset_has_audioSources(self, tmp_path):
        root = self._root(tmp_path, has_video=False, has_audio=True)
        asset = root.find("resources/asset")
        assert asset.get("audioSources") == "1"


# ── Project element must not have format attribute ────────────────────────────

class TestProjectElementNoFormat:
    """Regression: FCPXML DTD rejects format attr on <project> element.

    Commit 9404f69 removed format='r1' from <project>. This test ensures
    the attr stays absent so FCP can import without DTD validation errors.
    """

    def test_project_has_no_format_attr(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        project = root.find("library/event/project")
        assert project is not None
        assert project.get("format") is None, (
            "<project> must not have a format attribute — it belongs on <sequence>, "
            "not <project>. FCP DTD validation rejects it."
        )

    def test_sequence_has_format_attr(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        seq = root.find("library/event/project/sequence")
        assert seq.get("format") == "r1", (
            "<sequence> must carry format='r1' — it was moved here from <project>"
        )


# ── Asset must NOT carry a format attribute ───────────────────────────────────

class TestAssetNoFormatAttr:
    """Regression: <asset format="r1"> causes FCP to validate the codec name
    against our declared <format name="FFVideoFormat...">.  Non-standard profiles
    (HEVC Rext 4:2:2 10-bit) don't match that name and FCP reports
    "Invalid edit with no respective media" for every clip.  The fix is to omit
    format= on <asset> so FCP auto-detects it from the file, while <sequence>
    still carries format="r1" for rendering settings.
    """

    def test_asset_has_no_format_attr(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        asset = root.find("resources/asset")
        assert asset is not None
        assert asset.get("format") is None, (
            "<asset> must not carry format='r1' — it makes FCP validate the codec "
            "name against our declared <format>, which fails for HEVC Rext and other "
            "non-standard profiles.  FCP auto-detects the asset format from the file."
        )

    def test_sequence_still_has_format_attr(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, _media(src))
        seq = root.find("library/event/project/sequence")
        assert seq.get("format") == "r1", (
            "<sequence> must keep format='r1' for rendering/export settings"
        )


# ── Audio-only source ─────────────────────────────────────────────────────────

class TestAudioOnly:
    def test_audio_only_generates_valid_fcpxml(self, tmp_path):
        src = tmp_path / "audio.aac"
        src.write_bytes(b"\x00")
        m = _media(src, has_video=False, has_audio=True)
        m.frame_rate = None
        segs = build_segments([], total_duration=30.0)
        out = tmp_path / "cut.fcpxml"
        # Should not raise
        generate_roughcut_fcpxml(segs, m, out)
        assert out.exists()

    def test_audio_only_format_name_undefined(self, tmp_path):
        src = tmp_path / "audio.aac"
        src.write_bytes(b"\x00")
        m = _media(src, has_video=False, has_audio=True)
        m.frame_rate = None
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        fmt = root.find("resources/format")
        assert "Undefined" in fmt.get("name", "")


# ── Embedded timecode (Canon / camera files) ─────────────────────────────────

class TestTimecodeStart:
    """Regression: Canon camera files embed a non-zero timecode (e.g. 08:39:17:08).
    FCP uses the timecode as the asset's time origin; if <asset start> and the
    asset-clip <start> are both "0s", FCP reports "Invalid edit with no
    respective media" because time 0 is before the asset's timecode start.
    """

    TC = Fraction(11695684, 375)  # 08:39:17:08 at 23.976fps NDF

    def _media_with_tc(self, path, tc_start: Fraction):
        m = _media(path, fps=Fraction(24000, 1001))
        m.timecode_start = tc_start
        return m

    def test_asset_start_reflects_timecode(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        m = self._media_with_tc(src, self.TC)
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        asset = root.find("resources/asset")
        assert asset.get("start") == "11695684/375s"

    def test_asset_clip_start_reflects_timecode(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        m = self._media_with_tc(src, self.TC)
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        clip = root.find("library/event/project/sequence/spine/asset-clip")
        assert clip.get("start") == "11695684/375s"

    def test_asset_clip_start_includes_segment_offset(self, tmp_path):
        # Segment starting at 10s into the source → start = TC + 10s
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        m = self._media_with_tc(src, self.TC)
        # Remove 0–10s; keep 10–30s
        segs = build_segments([(0.0, 10.0)], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        clip = root.find("library/event/project/sequence/spine/asset-clip")
        # 10s at 23.976fps = 240 frames × (1001/24000) = 240240/24000 = 1001/100s
        # clip start = TC + 1001/100 = 11695684/375 + 1001/100
        expected = Fraction(11695684, 375) + Fraction(1001, 100)
        assert clip.get("start") == _frac_rt(expected)

    def test_no_timecode_defaults_to_zero(self, tmp_path):
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        m = _media(src)  # no timecode_start
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        asset = root.find("resources/asset")
        assert asset.get("start") == "0s"
        clip = root.find("library/event/project/sequence/spine/asset-clip")
        assert clip.get("start") == "0s"

    def test_midnight_timecode_explicit_zero_produces_0s(self, tmp_path):
        """Regression: timecode_start=Fraction(0) (midnight TC) must produce
        start='0s', not be masked by the `or Fraction(0)` falsy-bug.
        The fix uses `is not None` so this path is distinct from None.
        """
        src = tmp_path / "source.mp4"
        src.write_bytes(b"\x00")
        m = _media(src)
        m.timecode_start = Fraction(0)  # explicit midnight timecode
        segs = build_segments([], total_duration=30.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        asset = root.find("resources/asset")
        assert asset.get("start") == "0s"
        clip = root.find("library/event/project/sequence/spine/asset-clip")
        assert clip.get("start") == "0s"


# ── colorSpace attribute (10-bit / HDR content) ───────────────────────────────

class TestColorSpace:
    """generate_roughcut_fcpxml must set colorSpace on <format> when the source
    has a known color_primaries value.  FCP requires this for 10-bit and HDR
    footage; omitting it causes incorrect display in the viewer.
    """

    def _media_with_primaries(self, path: Path, primaries: str) -> "MediaInfo":
        m = _media(path)
        m.color_primaries = primaries
        return m

    def test_bt709_sets_rec709_colorspace(self, tmp_path):
        src = tmp_path / "v.mp4"
        src.write_bytes(b"\x00")
        m = self._media_with_primaries(src, "bt709")
        segs = build_segments([], total_duration=10.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        fmt = root.find("resources/format")
        assert fmt.get("colorSpace") == "1-1-1 (Rec. 709)"

    def test_bt2020_sets_hlg_colorspace(self, tmp_path):
        src = tmp_path / "v.mp4"
        src.write_bytes(b"\x00")
        m = self._media_with_primaries(src, "bt2020")
        segs = build_segments([], total_duration=10.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        fmt = root.find("resources/format")
        assert fmt.get("colorSpace") == "9-18-9 (Rec. 2020 HLG)"

    def test_bt470bg_sets_pal_colorspace(self, tmp_path):
        src = tmp_path / "v.mp4"
        src.write_bytes(b"\x00")
        m = self._media_with_primaries(src, "bt470bg")
        segs = build_segments([], total_duration=10.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        fmt = root.find("resources/format")
        assert fmt.get("colorSpace") == "5-1-6 (Rec. 601 PAL)"

    def test_smpte170m_sets_ntsc_colorspace(self, tmp_path):
        src = tmp_path / "v.mp4"
        src.write_bytes(b"\x00")
        m = self._media_with_primaries(src, "smpte170m")
        segs = build_segments([], total_duration=10.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        fmt = root.find("resources/format")
        assert fmt.get("colorSpace") == "6-1-6 (Rec. 601 NTSC)"

    def test_unknown_primaries_no_colorspace_attr(self, tmp_path):
        """An unknown color_primaries string must not produce a colorSpace attr."""
        src = tmp_path / "v.mp4"
        src.write_bytes(b"\x00")
        m = self._media_with_primaries(src, "xyz-nonstandard")
        segs = build_segments([], total_duration=10.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        fmt = root.find("resources/format")
        assert fmt.get("colorSpace") is None

    def test_no_color_primaries_no_colorspace_attr(self, tmp_path):
        """When color_primaries is None (SDR without tag), no colorSpace attr."""
        src = tmp_path / "v.mp4"
        src.write_bytes(b"\x00")
        m = _media(src)   # color_primaries defaults to None
        segs = build_segments([], total_duration=10.0)
        _, root = _write_and_parse(tmp_path, segs, m)
        fmt = root.find("resources/format")
        assert fmt.get("colorSpace") is None


# ── Integration test against real sample.mov ──────────────────────────────────

SAMPLE_MOV = Path(__file__).parent / "fixtures" / "sample.mov"


@pytest.mark.skipif(not SAMPLE_MOV.exists(), reason="sample.mov fixture not present")
class TestRealSampleMov:
    """End-to-end: probe sample.mov → generate_roughcut_fcpxml → validate FCPXML.

    Catches format-name and colorSpace regressions that unit tests (with mocked
    MediaInfo) might miss when a real ffprobe is involved.

    sample.mov is 2560×1440 @ 60fps bt709, which exercises:
      - non-standard resolution → height-only fallback (FFVideoFormat1440p60)
      - bt709 → colorSpace="1-1-1 (Rec. 709)"
      - no embedded timecode → asset start="0s"
    """

    def test_format_name_height_only_fallback(self, tmp_path):
        from preprod.probe import probe_media
        media = probe_media(SAMPLE_MOV)
        segs = build_segments([], total_duration=media.duration)
        out = tmp_path / "cut.fcpxml"
        generate_roughcut_fcpxml(segs, media, out)
        root = ET.parse(out).getroot()
        fmt_name = root.find("resources/format").get("name")
        # 2560×1440 is not a built-in FCP preset → height-only fallback
        assert fmt_name == "FFVideoFormat1440p60", (
            f"Expected FFVideoFormat1440p60 for 2560×1440 60fps, got {fmt_name!r}"
        )

    def test_bt709_colorspace_set(self, tmp_path):
        from preprod.probe import probe_media
        media = probe_media(SAMPLE_MOV)
        segs = build_segments([], total_duration=media.duration)
        out = tmp_path / "cut.fcpxml"
        generate_roughcut_fcpxml(segs, media, out)
        root = ET.parse(out).getroot()
        color_space = root.find("resources/format").get("colorSpace")
        assert color_space == "1-1-1 (Rec. 709)", (
            f"Expected Rec. 709 colorSpace for bt709 sample, got {color_space!r}"
        )

    def test_no_timecode_asset_starts_at_0s(self, tmp_path):
        from preprod.probe import probe_media
        media = probe_media(SAMPLE_MOV)
        segs = build_segments([], total_duration=media.duration)
        out = tmp_path / "cut.fcpxml"
        generate_roughcut_fcpxml(segs, media, out)
        root = ET.parse(out).getroot()
        asset = root.find("resources/asset")
        assert asset.get("start") == "0s"
