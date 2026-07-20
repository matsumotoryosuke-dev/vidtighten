"""ffprobe wrapper — extract media metadata."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional


@dataclass
class MediaInfo:
    path: Path
    duration: float
    has_video: bool
    has_audio: bool
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    frame_rate: Optional[Fraction] = None
    sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    color_primaries: Optional[str] = None   # e.g. "bt709", "bt2020"
    codec_video: Optional[str] = None
    codec_audio: Optional[str] = None
    # Embedded timecode of the first frame (from tmcd track or stream tag).
    # None means no timecode → FCPXML asset start defaults to 0s.
    # FCP uses this as the asset's timecode origin; asset-clips must use the
    # same value as their `start` attribute or FCP reports "Invalid edit with
    # no respective media".
    timecode_start: Optional[Fraction] = None


def _parse_timecode(tc: str, frame_rate: Fraction) -> Optional[Fraction]:
    """Convert an embedded timecode string to a rational time Fraction.

    Supports both NDF (``HH:MM:SS:FF``) and DF (``HH:MM:SS;FF``) notation.
    Returns ``None`` if the string cannot be parsed.

    For NDF the frame count from midnight is:
        (h×3600 + m×60 + s) × nominal_fps + f
    where *nominal_fps* is the nearest integer to the true frame rate.
    For DF a standard drop-frame adjustment is applied (only 29.97 and
    59.94 are commonly DF; other rates fall back to NDF arithmetic).
    """
    is_df = ";" in tc
    parts = tc.replace(";", ":").split(":")
    if len(parts) != 4:
        return None
    try:
        h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None

    # Use pure-integer rounding to avoid float imprecision at boundary values.
    nominal_fps = int(frame_rate + Fraction(1, 2))

    if is_df and nominal_fps in (30, 60):
        # SMPTE drop-frame: frames 0 and 1 are dropped at the start of every
        # minute except multiples of 10.
        drop_per_min = nominal_fps // 15  # 2 for 29.97, 4 for 59.94
        total_minutes = 60 * h + m
        dropped = drop_per_min * (total_minutes - total_minutes // 10)
        total_frames = (h * 3600 + m * 60 + s) * nominal_fps + f - dropped
    else:
        total_frames = (h * 3600 + m * 60 + s) * nominal_fps + f

    rt = Fraction(total_frames * frame_rate.denominator, frame_rate.numerator)
    return rt


def probe_media(path: Path) -> MediaInfo:
    """Run ffprobe and parse media metadata."""
    path = path.resolve()
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "ffprobe not found on PATH. Install ffmpeg from https://ffmpeg.org"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe timed out after 30s for {path}")
    except subprocess.CalledProcessError as e:
        stderr_snippet = (e.stderr or "")[:200]
        raise RuntimeError(f"ffprobe failed for {path}: {stderr_snippet}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )
    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"), None
    )

    duration = float(fmt.get("duration", 0))
    if duration == 0 and video_stream:
        duration = float(video_stream.get("duration", 0))
    if duration == 0 and audio_stream:
        duration = float(audio_stream.get("duration", 0))

    frame_rate = None
    if video_stream:
        rate_str = video_stream.get("r_frame_rate", "0/1")
        try:
            if "/" in rate_str:
                num_s, den_s = rate_str.split("/", 1)
                num, den = int(num_s), int(den_s)
                frame_rate = Fraction(num, den) if den != 0 else Fraction(30, 1)
            else:
                # Some containers emit a bare integer fps string (e.g. "24")
                fps_int = int(rate_str)
                frame_rate = Fraction(fps_int, 1) if fps_int > 0 else Fraction(30, 1)
        except (ValueError, ZeroDivisionError):
            frame_rate = Fraction(30, 1)

    # Sum channels across ALL audio streams so multi-track 4K files
    # (e.g. 2 × stereo = 4 ch) report the correct total to FCPXML.
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    total_channels = sum(int(s.get("channels", 2)) for s in audio_streams) if audio_streams else None

    color_primaries = video_stream.get("color_primaries") if video_stream else None

    # Extract embedded timecode. Prefer the dedicated tmcd stream; fall back
    # to the video stream's timecode tag.  FCP uses this as the asset's start
    # time in FCPXML — without it every asset-clip references a time before
    # the declared asset start, causing "Invalid edit with no respective media".
    timecode_start: Optional[Fraction] = None
    if frame_rate:
        raw_tc: Optional[str] = None
        tmcd_stream = next(
            (s for s in streams if s.get("codec_tag_string") == "tmcd"), None
        )
        if tmcd_stream:
            raw_tc = tmcd_stream.get("tags", {}).get("timecode")
        if not raw_tc and video_stream:
            raw_tc = video_stream.get("tags", {}).get("timecode")
        if raw_tc:
            timecode_start = _parse_timecode(raw_tc, frame_rate)

    return MediaInfo(
        path=path,
        duration=duration,
        has_video=video_stream is not None,
        has_audio=audio_stream is not None,
        video_width=int(video_stream["width"]) if video_stream else None,
        video_height=int(video_stream["height"]) if video_stream else None,
        frame_rate=frame_rate,
        sample_rate=int(audio_stream["sample_rate"]) if audio_stream else None,
        audio_channels=total_channels,
        color_primaries=color_primaries,
        codec_video=video_stream.get("codec_name") if video_stream else None,
        codec_audio=audio_stream.get("codec_name") if audio_stream else None,
        timecode_start=timecode_start,
    )
