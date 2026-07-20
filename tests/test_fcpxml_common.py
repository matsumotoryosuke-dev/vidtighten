"""Tests for fcpxml_common.py — FCP_FORMAT_PREFIX shared constant.

These tests verify two things:
  1. The dict itself is structurally correct and contains the expected entries.
  2. Both generators (fcpxml_cut and fcpxml_telop) produce identical format-name
     *prefixes* for the same (width, height) resolution — i.e. they truly share
     the same table and can't silently drift out of sync.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from preprod.fcpxml_common import FCP_FORMAT_PREFIX


# ── Structure ─────────────────────────────────────────────────────────────────

class TestFcpFormatPrefixStructure:
    def test_is_not_empty(self):
        assert len(FCP_FORMAT_PREFIX) > 0

    def test_keys_are_width_height_int_tuples(self):
        for k in FCP_FORMAT_PREFIX:
            assert isinstance(k, tuple) and len(k) == 2
            assert isinstance(k[0], int) and isinstance(k[1], int)

    def test_values_are_nonempty_strings(self):
        for v in FCP_FORMAT_PREFIX.values():
            assert isinstance(v, str) and v

    def test_all_prefixes_start_with_FFVideoFormat(self):
        for v in FCP_FORMAT_PREFIX.values():
            assert v.startswith("FFVideoFormat"), f"{v!r} doesn't start with FFVideoFormat"

    def test_no_duplicate_values(self):
        """Each resolution maps to a unique prefix string."""
        values = list(FCP_FORMAT_PREFIX.values())
        assert len(values) == len(set(values)), "Duplicate prefix found in FCP_FORMAT_PREFIX"


# ── Key resolution entries ────────────────────────────────────────────────────

class TestFcpFormatPrefixEntries:
    """Spot-check the resolutions that matter most in practice."""

    def test_720p_uses_height_only(self):
        assert FCP_FORMAT_PREFIX[(1280, 720)] == "FFVideoFormat720p"

    def test_1080p_uses_height_only(self):
        # FCP's internal preset for standard HD is height-only — never WxH.
        assert FCP_FORMAT_PREFIX[(1920, 1080)] == "FFVideoFormat1080p"

    def test_4k_uhd_uses_wxh(self):
        # 3840×2160 UHD must use the explicit WxH name from FCP's registry.
        assert FCP_FORMAT_PREFIX[(3840, 2160)] == "FFVideoFormat3840x2160p"

    def test_dci_4k_2160_uses_wxh(self):
        assert FCP_FORMAT_PREFIX[(4096, 2160)] == "FFVideoFormat4096x2160p"

    def test_8k_uhd_uses_wxh(self):
        assert FCP_FORMAT_PREFIX[(7680, 4320)] == "FFVideoFormat7680x4320p"

    def test_1440p_not_in_dict(self):
        # 2560×1440 is NOT a built-in FCP preset; it should fall through to the
        # height-only fallback in _format_name, creating a "Custom" sequence.
        assert (2560, 1440) not in FCP_FORMAT_PREFIX

    def test_4k_all_entries_use_wxh_naming(self):
        """Any resolution whose smaller dimension is ≥2048 should use WxH form."""
        for (w, h), prefix in FCP_FORMAT_PREFIX.items():
            if h >= 2048:
                assert f"{w}x{h}" in prefix, (
                    f"Expected WxH naming for {w}x{h} but got {prefix!r}"
                )

    def test_standard_hd_uses_height_only(self):
        """Standard HD resolutions (720p, 1080p) must use height-only names."""
        for (w, h), prefix in FCP_FORMAT_PREFIX.items():
            if h in (720, 1080):
                assert f"{w}x{h}" not in prefix, (
                    f"Unexpected WxH naming for standard HD {w}x{h}: {prefix!r}"
                )


# ── Cross-module parity ───────────────────────────────────────────────────────

class TestCrossModuleParity:
    """Both generators must produce identical format-name *prefixes* for the same
    resolution.  A mismatch means the two modules have silently diverged despite
    both importing from fcpxml_common — catching that here avoids silent FCP errors.
    """

    @pytest.fixture()
    def tmp_path_src(self, tmp_path):
        """Create a placeholder source file for fcpxml_cut."""
        src = tmp_path / "v.mp4"
        src.write_bytes(b"\x00")
        return src

    def _cut_format_prefix(self, width: int, height: int, fps: Fraction, tmp_path_src: Path) -> str:
        """Return the prefix portion of the format name from fcpxml_cut."""
        from preprod.fcpxml_cut import _format_name as cut_fmt
        from preprod.probe import MediaInfo
        m = MediaInfo(
            path=tmp_path_src,
            duration=10.0,
            has_video=True,
            has_audio=False,
            video_width=width,
            video_height=height,
            frame_rate=fps,
        )
        name = cut_fmt(m, fps)
        # Strip the trailing fps-id suffix — we compare the resolution prefix only.
        for fps_id in ("2398", "2997", "5994", "24", "25", "30", "50", "60"):
            if name.endswith(fps_id):
                return name[: -len(fps_id)]
        return name  # fallback: whole name if no known fps-id suffix

    def _telop_format_prefix(self, width: int, height: int, fps_str: str) -> str:
        """Return the prefix portion of the format name from fcpxml_telop."""
        from preprod.fcpxml_telop import _format_name as telop_fmt
        name = telop_fmt(width, height, fps_str)
        for fps_id in ("2398", "2997", "5994", "24", "25", "30", "50", "60"):
            if name.endswith(fps_id):
                return name[: -len(fps_id)]
        return name

    @pytest.mark.parametrize("width,height,fps_frac,fps_str", [
        (1280,  720,  Fraction(24, 1),       "24"),
        (1920, 1080,  Fraction(24, 1),       "24"),
        (1920, 1080,  Fraction(24000, 1001), "23.976"),
        (1920, 1080,  Fraction(30000, 1001), "29.97"),
        (3840, 2160,  Fraction(24, 1),       "24"),
        (3840, 2160,  Fraction(24000, 1001), "23.976"),
        (4096, 2160,  Fraction(24, 1),       "24"),
        (2560, 1440,  Fraction(60, 1),       "60"),   # non-standard → both fall back to height
    ])
    def test_prefix_matches_between_generators(
        self, width, height, fps_frac, fps_str, tmp_path_src
    ):
        cut_prefix   = self._cut_format_prefix(width, height, fps_frac, tmp_path_src)
        telop_prefix = self._telop_format_prefix(width, height, fps_str)
        assert cut_prefix == telop_prefix, (
            f"{width}x{height} @ {fps_str}: "
            f"fcpxml_cut={cut_prefix!r} vs fcpxml_telop={telop_prefix!r}"
        )
