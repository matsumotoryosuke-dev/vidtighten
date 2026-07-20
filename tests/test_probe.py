"""Tests for probe.py — ffprobe wrapper and MediaInfo construction."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from preprod.probe import MediaInfo, _parse_timecode, probe_media


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ffprobe_output(
    duration: float = 10.0,
    video: bool = True,
    audio: bool = True,
    fps: str = "24/1",
    width: int = 1920,
    height: int = 1080,
    sample_rate: str = "48000",
    codec_video: str = "h264",
    codec_audio: str = "aac",
    format_duration: Optional[float] = None,
    timecode: Optional[str] = None,   # e.g. "08:39:17:08"
    tmcd_stream: bool = False,         # add a tmcd stream with the timecode
    channels: int = 2,                 # audio channel count for the primary audio stream
    color_primaries: Optional[str] = None,  # e.g. "bt709", "bt2020"
    extra_audio_streams: Optional[list] = None,  # additional audio streams [{channels: N}]
) -> str:
    """Build a fake ffprobe JSON payload."""
    streams = []
    if video:
        vid: dict = {
            "codec_type": "video",
            "codec_name": codec_video,
            "width": width,
            "height": height,
            "r_frame_rate": fps,
            "duration": str(duration),
        }
        if timecode and not tmcd_stream:
            vid["tags"] = {"timecode": timecode}
        if color_primaries is not None:
            vid["color_primaries"] = color_primaries
        streams.append(vid)
    if audio:
        streams.append({
            "codec_type": "audio",
            "codec_name": codec_audio,
            "sample_rate": sample_rate,
            "channels": channels,
            "duration": str(duration),
        })
    if extra_audio_streams:
        for extra in extra_audio_streams:
            streams.append({
                "codec_type": "audio",
                "codec_name": codec_audio,
                "sample_rate": sample_rate,
                "channels": extra.get("channels", 2),
                "duration": str(duration),
            })
    if timecode and tmcd_stream:
        streams.append({
            "codec_type": "data",
            "codec_tag_string": "tmcd",
            "duration": str(duration),
            "tags": {"timecode": timecode},
        })
    fmt_duration = format_duration if format_duration is not None else duration
    return json.dumps({
        "streams": streams,
        "format": {"duration": str(fmt_duration)},
    })


def _run_probe(fake_json: str, fake_path: Path) -> MediaInfo:
    """Call probe_media() with subprocess.run returning fake ffprobe JSON."""
    mock_proc = MagicMock()
    mock_proc.stdout = fake_json
    with patch("preprod.probe.subprocess.run", return_value=mock_proc) as mock_run:
        result = probe_media(fake_path)
    return result


# ── Basic parsing ─────────────────────────────────────────────────────────────

class TestProbeMediaBasic:
    def test_returns_media_info_instance(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(), f)
        assert isinstance(result, MediaInfo)

    def test_path_resolved(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(), f)
        assert result.path == f.resolve()

    def test_duration_from_format(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(format_duration=42.5), f)
        assert result.duration == pytest.approx(42.5)

    def test_has_video_true(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(video=True, audio=True), f)
        assert result.has_video is True

    def test_has_audio_true(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(video=True, audio=True), f)
        assert result.has_audio is True

    def test_video_dimensions(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(width=3840, height=2160), f)
        assert result.video_width == 3840
        assert result.video_height == 2160

    def test_sample_rate_parsed(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(sample_rate="44100"), f)
        assert result.sample_rate == 44100

    def test_codec_names(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(codec_video="hevc", codec_audio="aac"), f)
        assert result.codec_video == "hevc"
        assert result.codec_audio == "aac"


# ── Frame rate ────────────────────────────────────────────────────────────────

class TestFrameRate:
    def test_integer_fps_24(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(fps="24/1"), f)
        assert result.frame_rate == Fraction(24, 1)

    def test_integer_fps_30(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(fps="30/1"), f)
        assert result.frame_rate == Fraction(30, 1)

    def test_drop_frame_fps_2997(self, tmp_path):
        """29.97 drop-frame is represented as 30000/1001."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(fps="30000/1001"), f)
        assert result.frame_rate == Fraction(30000, 1001)

    def test_zero_denominator_falls_back_to_30fps(self, tmp_path):
        """r_frame_rate="0/0" is malformed — probe should fall back to 30fps."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(fps="30/0"), f)
        assert result.frame_rate == Fraction(30, 1)

    def test_bare_integer_fps_string(self, tmp_path):
        """Some containers emit r_frame_rate as a bare integer (e.g. "24", no slash)."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(fps="24"), f)
        assert result.frame_rate == Fraction(24, 1)

    def test_malformed_fps_string_falls_back_to_30fps(self, tmp_path):
        """An unrecognised r_frame_rate string falls back to 30fps instead of crashing."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(fps="ntsc"), f)
        assert result.frame_rate == Fraction(30, 1)

    def test_audio_only_file_has_no_frame_rate(self, tmp_path):
        f = tmp_path / "a.aac"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(video=False, audio=True), f)
        assert result.frame_rate is None


# ── Duration fallback logic ───────────────────────────────────────────────────

class TestDurationFallback:
    def test_duration_falls_back_to_video_stream_when_format_zero(self, tmp_path):
        """If format.duration is 0, fall back to the video stream's duration."""
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        streams = [
            {"codec_type": "video", "codec_name": "h264",
             "width": 1920, "height": 1080, "r_frame_rate": "24/1",
             "duration": "55.0"},
        ]
        fake_json = json.dumps({"streams": streams, "format": {"duration": "0"}})
        result = _run_probe(fake_json, f)
        assert result.duration == pytest.approx(55.0)

    def test_duration_falls_back_to_audio_stream_when_no_video(self, tmp_path):
        """Audio-only file: no video stream, format duration may also be 0."""
        f = tmp_path / "a.aac"
        f.write_bytes(b"\x00")
        streams = [
            {"codec_type": "audio", "codec_name": "aac",
             "sample_rate": "44100", "duration": "90.0"},
        ]
        fake_json = json.dumps({"streams": streams, "format": {"duration": "0"}})
        result = _run_probe(fake_json, f)
        assert result.duration == pytest.approx(90.0)


# ── Audio-only and video-only files ──────────────────────────────────────────

class TestStreamTypes:
    def test_audio_only_file(self, tmp_path):
        f = tmp_path / "a.mp3"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(video=False, audio=True), f)
        assert result.has_video is False
        assert result.has_audio is True
        assert result.video_width is None
        assert result.video_height is None
        assert result.frame_rate is None
        assert result.codec_video is None

    def test_video_only_file(self, tmp_path):
        f = tmp_path / "v.mov"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(video=True, audio=False), f)
        assert result.has_video is True
        assert result.has_audio is False
        assert result.sample_rate is None
        assert result.codec_audio is None


# ── Error handling ────────────────────────────────────────────────────────────

class TestProbeErrors:
    def test_ffprobe_not_found_raises_file_not_found_error(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        with patch("preprod.probe.subprocess.run",
                   side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError, match="ffprobe"):
                probe_media(f)

    def test_ffprobe_nonzero_exit_raises_runtime_error(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        exc = subprocess.CalledProcessError(1, "ffprobe", stderr="bad file")
        with patch("preprod.probe.subprocess.run", side_effect=exc):
            with pytest.raises(RuntimeError, match="ffprobe"):
                probe_media(f)


# ── _parse_timecode ───────────────────────────────────────────────────────────

class TestParseTimecode:
    """Unit tests for the _parse_timecode() helper."""

    FPS_2398 = Fraction(24000, 1001)
    FPS_24   = Fraction(24, 1)
    FPS_2997 = Fraction(30000, 1001)

    def test_midnight_is_zero(self):
        assert _parse_timecode("00:00:00:00", self.FPS_24) == Fraction(0)

    def test_one_frame_at_24fps(self):
        # 1 frame at 24fps = 1/24 s
        assert _parse_timecode("00:00:00:01", self.FPS_24) == Fraction(1, 24)

    def test_one_second_at_24fps(self):
        # 24 frames at 24fps = 1s
        assert _parse_timecode("00:00:01:00", self.FPS_24) == Fraction(1, 1)

    def test_canon_timecode_2398(self):
        # 08:39:17:08 at 23.976fps NDF = 11695684/375s (confirmed by test I)
        result = _parse_timecode("08:39:17:08", self.FPS_2398)
        assert result == Fraction(11695684, 375)

    def test_ndf_29_97(self):
        # 00:01:00:00 NDF at 29.97 = 60×30=1800 frames × (1001/30000) = 1801800/30000 = 1001/16.67...
        # = 1800 * 1001/30000 = 1801800/30000 = simplified: 60060/1000 = 6006/100 = 3003/50
        result = _parse_timecode("00:01:00:00", self.FPS_2997)
        assert result == Fraction(3003, 50)

    def test_df_separator_handled(self):
        # DF notation: semicolon before frames field — should not raise
        result = _parse_timecode("00:00:01;00", self.FPS_2997)
        assert result is not None

    def test_invalid_string_returns_none(self):
        assert _parse_timecode("not-a-timecode", self.FPS_24) is None

    def test_wrong_field_count_returns_none(self):
        assert _parse_timecode("01:00:00", self.FPS_24) is None


# ── Timecode extraction in probe_media ───────────────────────────────────────

class TestTimecodeProbe:
    """Regression: FCP uses the embedded timecode as the asset's time origin."""

    def test_no_timecode_gives_none(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(), f)
        assert result.timecode_start is None

    def test_timecode_from_video_stream_tag(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(
            _make_ffprobe_output(fps="24000/1001", timecode="08:39:17:08"),
            f,
        )
        assert result.timecode_start == Fraction(11695684, 375)

    def test_timecode_from_tmcd_stream(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(
            _make_ffprobe_output(fps="24000/1001", timecode="08:39:17:08", tmcd_stream=True),
            f,
        )
        assert result.timecode_start == Fraction(11695684, 375)

    def test_midnight_timecode_gives_zero(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(
            _make_ffprobe_output(fps="24/1", timecode="00:00:00:00"),
            f,
        )
        assert result.timecode_start == Fraction(0)

    def test_timecode_ignored_for_audio_only(self, tmp_path):
        """Audio-only files have no frame_rate so timecode can't be parsed."""
        f = tmp_path / "a.aac"
        f.write_bytes(b"\x00")
        # No video stream → frame_rate=None → timecode_start stays None
        result = _run_probe(_make_ffprobe_output(video=False, audio=True), f)
        assert result.timecode_start is None


# ── audio_channels ────────────────────────────────────────────────────────────

class TestAudioChannels:
    """probe_media() sums channels across all audio streams."""

    def test_single_stereo_stream(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(channels=2), f)
        assert result.audio_channels == 2

    def test_mono_stream(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(channels=1), f)
        assert result.audio_channels == 1

    def test_surround_six_channels(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(channels=6), f)
        assert result.audio_channels == 6

    def test_multi_track_sums_channels(self, tmp_path):
        # Two stereo tracks (common in 4K cinema cameras) → 4 total
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(
            _make_ffprobe_output(channels=2, extra_audio_streams=[{"channels": 2}]),
            f,
        )
        assert result.audio_channels == 4

    def test_multi_track_mixed_channels(self, tmp_path):
        # Stereo + mono (e.g. main mix + guide track)
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(
            _make_ffprobe_output(channels=2, extra_audio_streams=[{"channels": 1}]),
            f,
        )
        assert result.audio_channels == 3

    def test_audio_channels_none_for_video_only(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(video=True, audio=False), f)
        assert result.audio_channels is None


# ── color_primaries ───────────────────────────────────────────────────────────

class TestColorPrimaries:
    """probe_media() passes through the color_primaries field from the video stream."""

    def test_bt709_populated(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(color_primaries="bt709"), f)
        assert result.color_primaries == "bt709"

    def test_bt2020_populated(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(color_primaries="bt2020"), f)
        assert result.color_primaries == "bt2020"

    def test_absent_is_none(self, tmp_path):
        # _make_ffprobe_output without color_primaries → no key in stream → None
        f = tmp_path / "v.mp4"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(), f)
        assert result.color_primaries is None

    def test_color_primaries_none_for_audio_only(self, tmp_path):
        f = tmp_path / "a.mp3"
        f.write_bytes(b"\x00")
        result = _run_probe(_make_ffprobe_output(video=False, audio=True), f)
        assert result.color_primaries is None


# ── _parse_timecode — drop-frame at 59.94 fps ─────────────────────────────────

class TestParseTimecodeDF5994:
    """Drop-frame arithmetic for 59.94 fps (nominal 60) — drop_per_min = 4."""

    FPS_5994 = Fraction(60000, 1001)

    def test_midnight_df_5994_is_zero(self):
        # 00:00:00;00 DF at 59.94: no frames, no drops → 0
        result = _parse_timecode("00:00:00;00", self.FPS_5994)
        assert result == Fraction(0)

    def test_one_second_df_5994(self):
        # 00:00:01;00 DF at 59.94: 60 frames at second 0 (no drops before 1-minute mark)
        # total_frames = 1*60 = 60; rt = 60 * 1001/60000 = 60060/60000 = 1001/1000
        result = _parse_timecode("00:00:01;00", self.FPS_5994)
        assert result == Fraction(1001, 1000)

    def test_one_minute_df_5994_drops_four_frames(self):
        # 00:01:00;00 DF at 59.94: frames 0-3 of the first second of each non-10-minute
        # are dropped. total_minutes=1, dropped=4
        # total_frames = (60)*60 + 0 - 4 = 3596
        # rt = 3596 * 1001/60000 = 3599596/60000
        result = _parse_timecode("00:01:00;00", self.FPS_5994)
        assert result == Fraction(3596 * 1001, 60000)
