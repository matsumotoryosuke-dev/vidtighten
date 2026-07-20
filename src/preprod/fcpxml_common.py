"""Shared FCPXML constants used by both fcpxml_cut and fcpxml_telop.

Centralising these here prevents the two generators from drifting out of sync
when Apple adds new format entries to FCP's internal Flexo.framework registry.
"""

from __future__ import annotations

# FCP canonical format-name prefixes keyed by (width, height).
# Standard HD (1280×720, 1920×1080) use height-only names; FCP has no 1440p
# preset so 2560×1440 falls through to the height-only fallback (creating a
# "Custom" sequence which FCP accepts). All 4K+ use explicit WxH names that
# match FCP's internal Flexo.framework format registry.
FCP_FORMAT_PREFIX: dict[tuple[int, int], str] = {
    # Standard HD — height-only (FCP convention, matches internal presets)
    (1280,  720): "FFVideoFormat720p",
    (1920, 1080): "FFVideoFormat1080p",
    # 4K UHD / DCI / 5K / 6K / 8K — explicit WxH (required by FCP format registry)
    (3840, 2160): "FFVideoFormat3840x2160p",
    (4096, 2048): "FFVideoFormat4096x2048p",
    (4096, 2160): "FFVideoFormat4096x2160p",
    (4096, 2304): "FFVideoFormat4096x2304p",
    (4096, 3112): "FFVideoFormat4096x3112p",
    (5120, 2160): "FFVideoFormat5120x2160p",
    (5120, 2560): "FFVideoFormat5120x2560p",
    (5120, 2700): "FFVideoFormat5120x2700p",
    (5760, 2880): "FFVideoFormat5760x2880p",
    (7680, 3840): "FFVideoFormat7680x3840p",
    (7680, 4320): "FFVideoFormat7680x4320p",
    (8192, 4320): "FFVideoFormat8192x4320p",
}
