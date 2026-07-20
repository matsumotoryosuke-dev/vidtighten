"""Rough-cut FCPXML v1.11 generator — keeps non-removed segments on a timeline."""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path

from preprod.fcpxml_common import FCP_FORMAT_PREFIX
from preprod.probe import MediaInfo
from preprod.segments import Segment

# color_primaries value from ffprobe → FCP colorSpace attribute string.
# FCP requires this for 10-bit and HDR content; safe to include for SDR too.
_COLOR_SPACE: dict[str, str] = {
    "bt709":  "1-1-1 (Rec. 709)",
    "bt470bg": "5-1-6 (Rec. 601 PAL)",    # bt601 PAL
    "smpte170m": "6-1-6 (Rec. 601 NTSC)", # bt601 NTSC
    "bt2020": "9-18-9 (Rec. 2020 HLG)",   # default bt2020 to HLG; HDR10 handled separately
}

# Canonical frame-rate → FCP fps-id string (matches FCP internal format names).
# FCP uses 4-digit strings for fractional rates (2398, 2997, 5994).
_FPS_ID: dict[Fraction, str] = {
    Fraction(24000, 1001): "2398",
    Fraction(24, 1):       "24",
    Fraction(25, 1):       "25",
    Fraction(30000, 1001): "2997",
    Fraction(30, 1):       "30",
    Fraction(50, 1):       "50",
    Fraction(60000, 1001): "5994",
    Fraction(60, 1):       "60",
}


def _to_rt(seconds: float, frame_rate: Fraction) -> str:
    """Convert seconds to FCPXML rational time, snapped to frame boundary."""
    frame_dur = Fraction(frame_rate.denominator, frame_rate.numerator)
    frame_count = round(Fraction(seconds) / frame_dur)
    rt = frame_count * frame_dur
    if rt.denominator == 1:
        return f"{rt.numerator}s"
    return f"{rt.numerator}/{rt.denominator}s"


def _frame_dur_str(frame_rate: Fraction) -> str:
    fd = Fraction(frame_rate.denominator, frame_rate.numerator)
    if fd.denominator == 1:
        return f"{fd.numerator}s"
    return f"{fd.numerator}/{fd.denominator}s"


def _frac_rt(rt: Fraction) -> str:
    """Format an exact Fraction as an FCPXML rational time string.

    Unlike _to_rt (which takes float seconds and snaps to a frame boundary),
    this function accepts a pre-computed Fraction and formats it verbatim.
    Used for timecode-derived start/offset attributes where the exact rational
    value must be preserved without additional frame-snapping.
    """
    if rt.denominator == 1:
        return f"{rt.numerator}s"
    return f"{rt.numerator}/{rt.denominator}s"


def _file_url(path: Path) -> str:
    encoded = urllib.parse.quote(str(path.resolve()), safe="/:")
    return f"file://{encoded}"


def _format_name(media: MediaInfo, frame_rate: Fraction | None) -> str:
    if not media.has_video or not media.video_height or not frame_rate:
        return "FFVideoFormatRateUndefined"
    fps_id = _FPS_ID.get(frame_rate)
    if fps_id is None:
        fps = float(frame_rate)
        fps_id = str(int(fps)) if fps == int(fps) else f"{fps * 100:.0f}"[:4]
    # Look up by exact resolution first; fall back to height-only for
    # non-standard sizes (e.g. 2560×1440) where FCP creates a Custom sequence.
    key = (media.video_width, media.video_height) if media.video_width else None
    prefix = FCP_FORMAT_PREFIX.get(key) if key else None
    if prefix is None:
        prefix = f"FFVideoFormat{media.video_height}p"
    return f"{prefix}{fps_id}"


def generate_roughcut_fcpxml(
    segments: list[Segment],
    media: MediaInfo,
    output_path: Path,
) -> None:
    """Write FCPXML v1.11 referencing the original source file.

    Creates one <asset-clip> per kept segment, positioned consecutively
    on a spine. FCP uses full-quality original media — no proxy needed.
    """
    frame_rate = media.frame_rate or Fraction(30, 1)
    frame_dur = Fraction(frame_rate.denominator, frame_rate.numerator)
    # Drop-frame timecode applies to 29.97 fps (30000/1001) and 59.94 fps (60000/1001).
    _DROP_FRAME = frozenset({Fraction(30000, 1001), Fraction(60000, 1001)})
    tc_format = "DF" if frame_rate in _DROP_FRAME else "NDF"

    fcpxml = ET.Element("fcpxml", version="1.11")
    resources = ET.SubElement(fcpxml, "resources")

    fmt_attrs: dict[str, str] = {
        "id":            "r1",
        "name":          _format_name(media, frame_rate),
        "frameDuration": _frame_dur_str(frame_rate),
    }
    if media.has_video and media.video_width and media.video_height:
        fmt_attrs["width"]  = str(media.video_width)
        fmt_attrs["height"] = str(media.video_height)
    if media.color_primaries and media.color_primaries in _COLOR_SPACE:
        fmt_attrs["colorSpace"] = _COLOR_SPACE[media.color_primaries]
    ET.SubElement(resources, "format", **fmt_attrs)

    # Embedded timecode is the asset's time origin (e.g. 08:39:17:08 on Canon
    # files).  FCP uses this as the asset's start in FCPXML; asset-clips must
    # use the same origin or FCP reports "Invalid edit with no respective media".
    # Use `is not None` so a genuine midnight timecode (Fraction(0)) is kept —
    # Fraction(0) is falsy and `or Fraction(0)` would silently discard it.
    tc_start: Fraction = media.timecode_start if media.timecode_start is not None else Fraction(0)

    asset_attrs: dict[str, str] = {
        "id":       "r2",
        "name":     media.path.stem,
        "start":    _frac_rt(tc_start),
        "duration": _to_rt(media.duration, frame_rate),
        "hasVideo": "1" if media.has_video else "0",
        "hasAudio": "1" if media.has_audio else "0",
        # Omit format= on <asset> intentionally: linking the asset to our declared
        # <format> makes FCP validate the codec name, which fails for non-standard
        # profiles (e.g. HEVC Rext 4:2:2 10-bit). Without it, FCP auto-detects the
        # asset's format from the file; the sequence still uses format="r1".
    }
    if media.has_video:
        asset_attrs["videoSources"] = "1"
    if media.has_audio:
        asset_attrs["audioSources"]  = "1"
        asset_attrs["audioChannels"] = str(media.audio_channels or 2)
        if media.sample_rate:
            asset_attrs["audioRate"] = str(media.sample_rate)
    asset = ET.SubElement(resources, "asset", **asset_attrs)
    ET.SubElement(asset, "media-rep", kind="original-media", src=_file_url(media.path))

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name="VidTighten")

    total_out = sum(s.duration for s in segments)
    project = ET.SubElement(
        event, "project",
        name=f"{media.path.stem}_cut",
    )
    sequence = ET.SubElement(
        project, "sequence",
        format="r1",
        duration=_to_rt(total_out, frame_rate),
        tcStart="0s",
        tcFormat=tc_format,
        audioLayout="stereo",
        audioRate="48k",
    )
    spine = ET.SubElement(sequence, "spine")

    timeline_offset = Fraction(0)
    for i, seg in enumerate(segments):
        # Source start in the asset's timecode space = tc_start + segment offset.
        # Snap to frame boundary first, then add the timecode origin.
        src_frames = round(Fraction(seg.source_start) / frame_dur)
        clip_start = _frac_rt(tc_start + src_frames * frame_dur)
        ET.SubElement(
            spine, "asset-clip",
            ref="r2",
            # timeline_offset is already frame-snapped (accumulated as
            # frame_count × frame_dur), so use _frac_rt directly to avoid
            # a float round-trip that could drift on very long timelines.
            offset=_frac_rt(timeline_offset),
            name=f"{media.path.stem} {i + 1}",
            start=clip_start,
            duration=_to_rt(seg.duration, frame_rate),
            tcFormat=tc_format,
        )
        frame_count = round(Fraction(seg.duration) / frame_dur)
        timeline_offset += frame_count * frame_dur

    tree = ET.ElementTree(fcpxml)
    ET.indent(tree, space="    ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE fcpxml>\n')
        tree.write(f, encoding="unicode", xml_declaration=False)
