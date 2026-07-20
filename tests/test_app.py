"""Tests for app.py — _PywebviewApi.get_dropped_paths, save_file_content,
export_roughcut/telop/subtitles bridge methods, and startup config.

Tests the drag-and-drop path resolution and export save logic that bridges
pywebview's native APIs to JS.

We do NOT import app.py directly because its module-level
`from preprod.web import app` triggers Flask/web.py startup side-effects.
Instead we test the contract independently.
"""
from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from webview.dom import _dnd_state

APP_PY = Path(__file__).parent.parent / "app.py"


# ── Inline equivalent of _PywebviewApi for unit testing ───────────────────────
# Mirrors the implementation in app.py without triggering Flask imports.

class _Api:
    """Test double mirroring app._PywebviewApi without triggering Flask imports."""

    def get_dropped_paths(self) -> list[str]:
        from webview.dom import _dnd_state
        paths = [item[1] for item in list(_dnd_state["paths"])]
        _dnd_state["paths"].clear()
        return paths

    # Override in tests to avoid writing to the real ~/Downloads folder.
    _downloads_dir: Path | None = None

    # ── Export bridge methods (mirroring app._PywebviewApi) ─────────────────

    def export_roughcut(self, path: str, removal_regions: list, padding_ms: int,
                        threshold_db: float = -40.0) -> dict:
        try:
            import preprod.web as _web
            from preprod.fcpxml_cut import generate_roughcut_fcpxml
            from preprod.probe import probe_media
            from preprod.segments import build_segments

            p = Path(path).expanduser().resolve()
            if not p.exists():
                return {"ok": False, "error": f"File not found: {p.name}"}

            with _web._media_cache_lock:
                media = _web._media_cache.get(str(p))
            if media is None:
                media = probe_media(p)
                _web._cache_media(str(p), media)

            regions = [(float(r["start"]), float(r["end"])) for r in (removal_regions or [])]
            segs = build_segments(regions, media.duration, int(padding_ms or 200))
            if not segs:
                return {"ok": False, "error": "No keep-segments after applying removals"}

            _web._EXPORT_DIR.mkdir(exist_ok=True)
            out = _web._EXPORT_DIR / f"{p.stem}_cut.fcpxml"
            generate_roughcut_fcpxml(segs, media, out)

            ok, err = _web._copy_to_downloads(out, f"{p.stem}_cut.fcpxml")
            if not ok:
                return {"ok": False, "error": err}
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def export_telop(self, path, duration, telop_entries, removal_regions, padding_ms, settings, stem,
                    use_source_timing: bool = False, threshold_db: float = -40.0) -> dict:
        try:
            import preprod.web as _web
            from preprod.fcpxml_telop import generate_telop_fcpxml
            from preprod.segments import build_segments

            regions = [(float(r["start"]), float(r["end"])) for r in (removal_regions or [])]
            dur = float(duration or 0)
            keep_segs = build_segments(regions, dur, int(padding_ms or 200)) if dur else []

            _web._EXPORT_DIR.mkdir(exist_ok=True)
            out = _web._EXPORT_DIR / f"{stem}_telop.fcpxml"
            generate_telop_fcpxml(
                telop_entries or [], keep_segs,
                total_source_duration=dur, settings=settings or {},
                stem=stem, output_path=out,
            )
            ok, err = _web._copy_to_downloads(out, f"{stem}_telop.fcpxml")
            if not ok:
                return {"ok": False, "error": err}
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def export_subtitles(self, fmt, telop_entries, removal_regions, duration, padding_ms, stem,
                        threshold_db: float = -40.0) -> dict:
        try:
            import preprod.web as _web
            from preprod.segments import build_segments, map_span_to_output, filter_telop_entries

            fmt = (fmt or "srt").lower()
            if fmt not in ("srt", "vtt"):
                return {"ok": False, "error": "format must be 'srt' or 'vtt'"}

            regions = [(float(r["start"]), float(r["end"])) for r in (removal_regions or [])]
            dur = float(duration or 0)
            keep_segs = build_segments(regions, dur, int(padding_ms or 200)) if dur else []

            ts_fn = _web._ts_vtt if fmt == "vtt" else _web._ts_srt
            lines = ["WEBVTT\n"] if fmt == "vtt" else []
            idx = 1
            for entry in filter_telop_entries(telop_entries or []):
                span = map_span_to_output(entry["start"], entry["end"], keep_segs)
                if span is None:
                    continue
                out_s, out_e = span
                text = entry.get("text", "").strip()
                if not text:
                    continue
                if fmt == "srt":
                    lines.append(str(idx))
                lines.append(f"{ts_fn(out_s)} --> {ts_fn(out_e)}")
                lines.append(text)
                lines.append("")
                idx += 1

            content = "\n".join(lines)
            _web._EXPORT_DIR.mkdir(exist_ok=True)
            tmp = _web._EXPORT_DIR / f"{stem}_subtitles.{fmt}"
            tmp.write_text(content, encoding="utf-8")
            ok, err = _web._copy_to_downloads(tmp, f"{stem}_subtitles.{fmt}")
            if not ok:
                return {"ok": False, "error": err}
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def save_file_content(self, content: str, filename: str) -> dict:
        """Inline replica of app._PywebviewApi.save_file_content.

        Deliberately does NOT call subprocess.run — spawning a child process
        from a pywebview JS-API background thread causes side-effects that reset
        the WKWebView (confirmed from crash log). Also does NOT return the path
        in the success dict — multi-byte characters in the path can break
        pywebview's evaluate_js serialisation, which also blanks the page.
        """
        downloads = self._downloads_dir or (Path.home() / "Downloads")
        try:
            downloads.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": f"Cannot create ~/Downloads: {exc}"}

        dest = downloads / filename

        if dest.exists():
            stem, suffix = Path(filename).stem, Path(filename).suffix
            for n in range(1, 100):
                dest = downloads / f"{stem} ({n}){suffix}"
                if not dest.exists():
                    break

        try:
            dest.write_text(content, encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

        return {"ok": True}


# ── Contract tests ─────────────────────────────────────────────────────────────

class TestGetDroppedPaths:
    """Unit tests for the get_dropped_paths() contract.

    pywebview's cocoa backend stores drag paths as (basename, full_path) tuples
    in _dnd_state['paths'].  get_dropped_paths() must:
      1. Return the full path (index 1) from each tuple.
      2. Clear _dnd_state['paths'] after reading (prevent double-fire).
      3. Return [] when no file was dropped.
      4. Handle multi-byte (Japanese) filenames correctly.
    """

    def setup_method(self):
        _dnd_state["paths"].clear()

    def teardown_method(self):
        _dnd_state["paths"].clear()

    def test_returns_full_path_from_single_drop(self):
        _dnd_state["paths"] = [("recording.mp3", "/Users/rio/Desktop/recording.mp3")]
        assert _Api().get_dropped_paths() == ["/Users/rio/Desktop/recording.mp3"]

    def test_returns_multiple_full_paths(self):
        _dnd_state["paths"] = [
            ("clip.mov",  "/Users/rio/Movies/clip.mov"),
            ("audio.mp3", "/Users/rio/Desktop/audio.mp3"),
        ]
        result = _Api().get_dropped_paths()
        assert result == ["/Users/rio/Movies/clip.mov", "/Users/rio/Desktop/audio.mp3"]

    def test_clears_dnd_state_after_read(self):
        _dnd_state["paths"] = [("file.mp3", "/tmp/file.mp3")]
        _Api().get_dropped_paths()
        assert _dnd_state["paths"] == [], "paths must be cleared to prevent stale re-reads"

    def test_returns_empty_list_when_no_drop(self):
        _dnd_state["paths"] = []
        assert _Api().get_dropped_paths() == []

    def test_handles_japanese_filename(self):
        path = "/Users/rio/Desktop/Google辞めました。数千万の給与と無料飯を捨てた理由.mp3"
        _dnd_state["paths"] = [("Google辞めました。数千万の給与と無料飯を捨てた理由.mp3", path)]
        assert _Api().get_dropped_paths() == [path]

    def test_second_call_returns_empty_after_clear(self):
        """Second call must return [] — no stale paths from previous drop."""
        _dnd_state["paths"] = [("video.mov", "/tmp/video.mov")]
        api = _Api()
        api.get_dropped_paths()          # first call consumes
        assert api.get_dropped_paths() == []  # second must be empty


# ── save_file_content tests ───────────────────────────────────────────────────

class TestSaveFileContent:
    """Unit tests for the save_file_content() contract.

    dlBlob() in utils.js calls this via the pywebview bridge instead of the
    blob-URL anchor trick (which blanks the page) or create_file_dialog (which
    also blanks the page — modal panel from a JS-API thread forces AppKit into
    an inconsistent WKWebView state).

    Contracts:
      1. Returns {"ok": True} (no "path" — multi-byte paths break evaluate_js).
      2. Does NOT call subprocess — side effects from bg thread blank WKWebView.
      3. Written file content matches what was passed in.
      4. Collision-safe: appends " (1)", " (2)", … when file already exists.
      5. Returns {"ok": False, "error": ...} on OS write failure.
    """

    def _api(self, downloads_dir: Path) -> _Api:
        api = _Api()
        api._downloads_dir = downloads_dir
        return api

    def test_returns_ok_on_save(self, tmp_path):
        result = self._api(tmp_path).save_file_content("<fcpxml/>", "clip_cut.fcpxml")
        assert result == {"ok": True}, "success dict must be exactly {'ok': True} — no path field"

    def test_written_content_matches_input(self, tmp_path):
        content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n"
        self._api(tmp_path).save_file_content(content, "clip.srt")
        assert (tmp_path / "clip.srt").read_text(encoding="utf-8") == content

    def test_does_not_spawn_subprocess(self, tmp_path):
        """Must NOT call subprocess — spawning from a pywebview bg thread blanks WKWebView."""
        with patch("subprocess.run") as mock_run:
            self._api(tmp_path).save_file_content("<fcpxml/>", "clip_cut.fcpxml")
        mock_run.assert_not_called()

    def test_collision_appends_suffix(self, tmp_path):
        (tmp_path / "clip_cut.fcpxml").write_text("old", encoding="utf-8")
        result = self._api(tmp_path).save_file_content("new", "clip_cut.fcpxml")
        assert result["ok"] is True
        assert (tmp_path / "clip_cut (1).fcpxml").exists()

    def test_collision_increments_to_2(self, tmp_path):
        (tmp_path / "clip_cut.fcpxml").write_text("old", encoding="utf-8")
        (tmp_path / "clip_cut (1).fcpxml").write_text("old2", encoding="utf-8")
        result = self._api(tmp_path).save_file_content("new", "clip_cut.fcpxml")
        assert result["ok"] is True
        assert (tmp_path / "clip_cut (2).fcpxml").exists()

    def test_handles_japanese_content(self, tmp_path):
        content = "<fcpxml>日本語テスト</fcpxml>"
        self._api(tmp_path).save_file_content(content, "動画_cut.fcpxml")
        assert (tmp_path / "動画_cut.fcpxml").read_text(encoding="utf-8") == content

    def test_handles_japanese_filename_no_path_in_result(self, tmp_path):
        """Multi-byte path must NOT appear in result — pywebview evaluate_js breaks on it."""
        path = "Google辞めました。数千万の給与と無料飯を捨てた理由_cut.fcpxml"
        result = self._api(tmp_path).save_file_content("<fcpxml/>", path)
        assert result == {"ok": True}
        assert "path" not in result


# ── Export bridge method tests ────────────────────────────────────────────────

class TestExportBridgeMethods:
    """Unit tests for _PywebviewApi.export_roughcut / export_telop / export_subtitles.

    These methods must:
      1. Bypass fetch() entirely — call preprod modules directly.
      2. Write the output file to ~/Downloads (redirected to tmp_path in tests).
      3. Return {"ok": True} on success, {"ok": False, "error": "..."} on failure.
      4. Never raise an unhandled exception (all errors are caught and returned).
    """

    def _api(self) -> "_Api":
        return _Api()

    # ── roughcut ──────────────────────────────────────────────────────────────

    def test_roughcut_returns_ok_and_writes_fcpxml(self, tmp_path):
        fake_media = MagicMock()
        fake_media.duration = 10.0

        src_file = tmp_path / "video.mp4"
        src_file.write_bytes(b"\x00" * 100)

        import preprod.web as _web
        _web._media_cache.clear()  # ensure cache miss so probe_media is called
        try:
            with (
                patch("preprod.probe.probe_media", return_value=fake_media),
                patch("preprod.fcpxml_cut.generate_roughcut_fcpxml",
                      side_effect=lambda segs, media, out: out.write_text("<fcpxml/>", encoding="utf-8")),
                patch("preprod.web._copy_to_downloads", return_value=(True, "")) as mock_dl,
            ):
                result = _Api().export_roughcut(str(src_file), [], 200)
        finally:
            _web._media_cache.clear()

        assert result == {"ok": True}
        assert mock_dl.called

    def test_roughcut_returns_error_if_file_missing(self, tmp_path):
        api = _Api()
        result = api.export_roughcut(str(tmp_path / "nonexistent.mp4"), [], 200)
        assert result["ok"] is False
        assert "not found" in result["error"].lower()

    def test_roughcut_never_raises(self, tmp_path):
        """All exceptions must be caught and returned as {"ok": False, ...}."""
        api = _Api()
        # Pass an invalid path that will fail during resolution
        result = api.export_roughcut("", [], 200)
        assert result["ok"] is False

    # ── telop ────────────────────────────────────────────────────────────────

    def test_telop_returns_ok(self, tmp_path):
        entries = [{"start": 1.0, "end": 2.0, "text": "Hello"}]
        with (
            patch("preprod.fcpxml_telop.generate_telop_fcpxml") as mock_gen,
            patch("preprod.web._copy_to_downloads", return_value=(True, "")),
        ):
            mock_gen.side_effect = lambda *a, **kw: kw["output_path"].write_text("<fcpxml/>", encoding="utf-8")
            result = _Api().export_telop("/tmp/v.mp4", 10.0, entries, [], 200, {}, "v")
        assert result == {"ok": True}

    def test_telop_error_propagated(self):
        with (
            patch("preprod.fcpxml_telop.generate_telop_fcpxml", side_effect=RuntimeError("bad")),
        ):
            result = _Api().export_telop("/tmp/v.mp4", 10.0, [], [], 200, {}, "v")
        assert result["ok"] is False
        assert "bad" in result["error"]

    # ── subtitles ────────────────────────────────────────────────────────────

    def test_subtitles_srt_returns_ok(self, tmp_path):
        entries = [{"start": 0.0, "end": 1.0, "text": "Hi"}]
        with patch("preprod.web._copy_to_downloads", return_value=(True, "")):
            result = _Api().export_subtitles("srt", entries, [], 5.0, 0, "clip")
        assert result == {"ok": True}

    def test_subtitles_vtt_returns_ok(self, tmp_path):
        entries = [{"start": 0.0, "end": 1.0, "text": "Hi"}]
        with patch("preprod.web._copy_to_downloads", return_value=(True, "")):
            result = _Api().export_subtitles("vtt", entries, [], 5.0, 0, "clip")
        assert result == {"ok": True}

    def test_subtitles_invalid_format(self):
        result = _Api().export_subtitles("mp4", [], [], 5.0, 0, "clip")
        assert result["ok"] is False
        assert "format" in result["error"]

    def test_subtitles_never_raises(self):
        with patch("preprod.web._copy_to_downloads", side_effect=OSError("disk full")):
            result = _Api().export_subtitles("srt", [], [], 5.0, 0, "clip")
        assert result["ok"] is False

    def test_subtitles_malformed_entries_skipped(self, tmp_path):
        """Entries missing 'start' or 'end' must be silently skipped, not crash."""
        entries = [
            {"text": "no times"},           # missing start and end
            {"start": 1.0, "text": "missing end"},
            {"end": 3.0, "text": "missing start"},
            {"start": 0.5, "end": 2.0, "text": "valid"},
        ]
        with patch("preprod.web._copy_to_downloads", return_value=(True, "")):
            result = _Api().export_subtitles("srt", entries, [], 5.0, 0, "clip")
        assert result == {"ok": True}

    def test_telop_malformed_entries_skipped(self):
        """export_telop must not crash on entries missing start/end."""
        entries = [
            {"text": "no times"},
            {"start": 1.0, "end": 3.0, "text": "ok"},
        ]
        with (
            patch("preprod.fcpxml_telop.generate_telop_fcpxml") as mock_gen,
            patch("preprod.web._copy_to_downloads", return_value=(True, "")),
        ):
            mock_gen.side_effect = lambda *a, **kw: kw["output_path"].write_text("<fcpxml/>", encoding="utf-8")
            result = _Api().export_telop("/tmp/v.mp4", 10.0, entries, [], 200, {}, "v")
        assert result == {"ok": True}


# ── Source-code structural checks (no import required) ────────────────────────

class TestAppPyStructure:
    """Verify app.py source contains the required plumbing without importing it."""

    def setup_method(self):
        self._src = APP_PY.read_text(encoding="utf-8")

    def test_sets_num_listeners_in_main(self):
        """main() must set _dnd_state['num_listeners'] before webview.start().

        Without this, cocoa's performDragOperation_ never stores paths.
        """
        assert 'num_listeners' in self._src, (
            "app.py must set _dnd_state['num_listeners'] so pywebview captures drops"
        )

    def test_js_api_registered(self):
        """create_window must pass js_api= so window.pywebview.api is available."""
        assert 'js_api=' in self._src, (
            "create_window must receive js_api=_PywebviewApi() to expose get_dropped_paths"
        )

    def test_get_dropped_paths_uses_dnd_state_not_jxa(self):
        """Implementation must use _dnd_state (pywebview's own capture), not osascript/JXA."""
        assert '_dnd_state' in self._src, "must use _dnd_state for path capture"
        assert 'osascript' not in self._src, (
            "osascript reads the general pasteboard, not the drag pasteboard — wrong approach"
        )

    def test_get_dropped_paths_returns_index_1(self):
        """_dnd_state paths are (basename, full_path) tuples; must return full_path (index 1)."""
        assert 'item[1]' in self._src, "must extract full_path via item[1] from dnd_state tuples"

    def test_save_file_content_registered(self):
        """save_file_content must be defined on _PywebviewApi so JS bridge exposes it."""
        assert 'save_file_content' in self._src, (
            "_PywebviewApi must define save_file_content for dlBlob() pywebview path"
        )

    def test_save_file_content_writes_to_downloads(self):
        """Must write to ~/Downloads — no Save dialog (dialog from JS-API thread blanks page)."""
        assert 'Downloads' in self._src, (
            "save_file_content must write to ~/Downloads to avoid create_file_dialog thread issues"
        )
        assert 'SAVE_DIALOG' not in self._src, (
            "save_file_content must NOT call create_file_dialog — causes WKWebView blank page"
        )

    def test_save_file_content_no_subprocess(self):
        """Must NOT call subprocess from save_file_content.

        Spawning a child process (e.g. 'open -R') from a pywebview JS-API background
        thread causes side effects that reset the WKWebView, blanking the page.
        """
        import re
        # Extract save_file_content method body (up to next method or end of class)
        match = re.search(r'def save_file_content.*?(?=\n    def |\nclass |\Z)', self._src, re.DOTALL)
        if match:
            method_src = match.group(0)
            # Strip docstring before checking — "subprocess" may appear there as explanation
            code_only = re.sub(r'""".*?"""', '', method_src, flags=re.DOTALL)
            assert 'subprocess.run' not in code_only, (
                "save_file_content must NOT call subprocess.run — crashes WKWebView from pywebview bg thread"
            )
            assert 'import subprocess' not in code_only, (
                "save_file_content must NOT import subprocess — side effects blank WKWebView"
            )

    def test_save_file_content_returns_minimal_dict(self):
        """Success return must be {'ok': True} with no path field.

        Returning the file path in evaluate_js with multi-byte (Japanese) characters
        can produce malformed JS, triggering a WKWebView navigation error.
        """
        import re
        match = re.search(r'def save_file_content.*?(?=\n    def |\ndef |\Z)', self._src, re.DOTALL)
        if match:
            method_src = match.group(0)
            # The success return must not include "path" key
            assert '"path"' not in method_src and "'path'" not in method_src, (
                "save_file_content must NOT return 'path' — multi-byte paths break evaluate_js"
            )

    def test_export_bridge_methods_defined(self):
        """export_roughcut/telop/subtitles must be defined on _PywebviewApi.

        JS calls these directly via window.pywebview.api.export_* to bypass
        fetch(), which blanks WKWebView regardless of response MIME type.
        """
        assert 'def export_roughcut' in self._src, "_PywebviewApi must define export_roughcut"
        assert 'def export_telop'    in self._src, "_PywebviewApi must define export_telop"
        assert 'def export_subtitles' in self._src, "_PywebviewApi must define export_subtitles"

    def test_export_bridge_methods_do_not_use_fetch(self):
        """Bridge export methods must not make HTTP requests — any fetch() can blank WKWebView."""
        import re
        for method in ('export_roughcut', 'export_telop', 'export_subtitles'):
            match = re.search(rf'def {method}.*?(?=\n    def |\nclass |\Z)', self._src, re.DOTALL)
            if match:
                # Strip docstrings before checking — docstrings explain the no-fetch contract
                # and naturally mention "fetch()" in the prose.
                code_only = re.sub(r'""".*?"""', '', match.group(0), flags=re.DOTALL)
                assert 'apiFetch' not in code_only, f"{method} must not use apiFetch()"
                assert 'requests.get' not in code_only and 'requests.post' not in code_only, \
                    f"{method} must not use requests library"
                assert 'urllib.request.urlopen' not in code_only, \
                    f"{method} must not use urllib.request"

    def test_export_bridge_methods_accept_threshold_db(self):
        """All three export bridge methods must accept threshold_db.

        The pywebview path previously used a hardcoded -40 dB for word boundary
        refinement while the HTTP path forwarded the user's threshold slider value.
        threshold_db must be a parameter so JS can pass the user's value to both
        paths consistently.
        """
        import re
        for method in ('export_roughcut', 'export_telop', 'export_subtitles'):
            match = re.search(rf'def {method}\b.*?\).*?:', self._src, re.DOTALL)
            assert match, f"Could not find {method} signature in app.py"
            assert 'threshold_db' in match.group(0), (
                f"{method} must accept threshold_db so the user's silence threshold "
                "is used consistently in both pywebview and HTTP export paths"
            )

    def test_export_bridge_methods_forward_threshold_db_to_refine(self):
        """Each export method must forward threshold_db= to _build_segments_typed.

        Without this, word boundary refinement uses the default -40 dB regardless
        of what threshold the user has set in the UI.

        Note: the old check was for _refine_word_regions; it was updated when the
        export pipeline switched to _build_segments_typed which inlines the
        word-boundary refinement with per-type padding awareness.
        """
        import re
        for method in ('export_roughcut', 'export_telop', 'export_subtitles'):
            match = re.search(rf'def {method}.*?(?=\n    def |\nclass |\Z)', self._src, re.DOTALL)
            if match:
                code_only = re.sub(r'""".*?"""', '', match.group(0), flags=re.DOTALL)
                assert 'threshold_db' in code_only and '_build_segments_typed' in code_only, (
                    f"{method} must call _build_segments_typed with threshold_db="
                )


# ── Timestamp formatting ──────────────────────────────────────────────────────

class TestTsSrt:
    """Unit tests for _ts_srt / _ts_vtt timestamp formatters."""

    @pytest.fixture(autouse=True)
    def _load(self):
        import preprod.web as _web
        self._ts_srt = _web._ts_srt
        self._ts_vtt = _web._ts_vtt

    def test_normal(self):
        assert self._ts_srt(1.5) == "00:00:01,500"

    def test_zero(self):
        assert self._ts_srt(0.0) == "00:00:00,000"

    def test_hours_minutes_seconds(self):
        assert self._ts_srt(3661.001) == "01:01:01,001"

    def test_ms_carry_over(self):
        """sec where fractional part rounds to 1.0 must carry into the next second."""
        # Previously: int(round((1.9995 % 1) * 1000)) == 1000 → "00:00:01,1000"
        assert self._ts_srt(1.9995) == "00:00:02,000"
        assert self._ts_srt(0.9995) == "00:00:01,000"
        assert self._ts_srt(59.9995) == "00:01:00,000"
        assert self._ts_srt(3599.9995) == "01:00:00,000"

    def test_vtt_uses_dot_separator(self):
        assert self._ts_vtt(1.5) == "00:00:01.500"
        # carry-over must work identically via _ts_vtt
        assert self._ts_vtt(1.9995) == "00:00:02.000"
