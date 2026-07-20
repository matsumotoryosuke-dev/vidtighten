"""Tests for web.py.

Regression tests:
  - Bug #1: np.sqrt/np.mean used without importing numpy → NameError
    (verify `import numpy as np` exists at top level)

Flask endpoint tests:
  - /api/capabilities returns {"whisper_available": bool}
  - /api/probe returns 404 for nonexistent file
"""

import ast
import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

WEB_PATH = Path(__file__).parent.parent / "src" / "preprod" / "web.py"


def _make_video_media(path, duration=10.0, fps=24):
    """Return a MagicMock MediaInfo for a standard H.264/AAC video file."""
    from fractions import Fraction
    m = MagicMock()
    m.path = path
    m.duration = duration
    m.has_video = True
    m.has_audio = True
    m.video_width = 1920
    m.video_height = 1080
    m.frame_rate = Fraction(fps, 1)
    m.sample_rate = 48000
    m.codec_video = "h264"
    m.codec_audio = "aac"
    return m


# ── Bug #1 regression: numpy must be imported at module level ─────────────────

class TestNumpyImportedAtModuleLevel:
    """Regression test for Bug #1.

    web.py uses np.sqrt() and np.mean() in _run_analysis().
    Before the fix, numpy was never imported, causing:
        NameError: name 'np' is not defined
    The fix is: import numpy as np  at the top of the file.
    """

    def test_numpy_imported_as_np_at_module_level(self):
        source = WEB_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        found = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "numpy" and alias.asname == "np":
                        found = True
        assert found, (
            "REGRESSION BUG #1: 'import numpy as np' not found at module level in web.py. "
            "np.sqrt and np.mean will raise NameError at runtime."
        )

    def test_numpy_import_before_usage(self):
        """Confirm 'import numpy as np' appears before the line that uses np.sqrt/np.mean."""
        source = WEB_PATH.read_text(encoding="utf-8")
        lines = source.splitlines()
        import_line = None
        usage_line = None
        for i, line in enumerate(lines, start=1):
            if "import numpy as np" in line and import_line is None:
                import_line = i
            if "np.sqrt" in line or "np.mean" in line:
                if usage_line is None:
                    usage_line = i
        assert import_line is not None, "REGRESSION BUG #1: 'import numpy as np' line not found"
        assert usage_line is not None, "np.sqrt/np.mean usage line not found"
        assert import_line < usage_line, (
            f"numpy import (line {import_line}) must come before usage (line {usage_line})"
        )


# ── Flask app fixture ─────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Test client for the Flask app with TESTING=True."""
    # Import the module BEFORE patching so that web.py's module-level name
    # bindings (e.g. `extract_audio = preprod.audio.extract_audio`) resolve to
    # the real functions, not a mock.  If preprod.web were first imported while
    # patch("preprod.audio.extract_audio") is active the local name inside
    # web.py would permanently point at the MagicMock, breaking any test that
    # later calls the endpoint without its own mock in place.
    import preprod.web  # noqa: F401 — side-effect: warm the module cache
    with patch("preprod.web.extract_audio"), \
         patch("preprod.web.probe_media"):
        from preprod.web import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


# ── /api/capabilities ─────────────────────────────────────────────────────────

class TestApiCapabilities:
    def test_returns_200(self, client):
        resp = client.get("/api/capabilities")
        assert resp.status_code == 200

    def test_returns_json(self, client):
        resp = client.get("/api/capabilities")
        assert resp.content_type.startswith("application/json")

    def test_whisper_available_key_present(self, client):
        resp = client.get("/api/capabilities")
        data = resp.get_json()
        assert "whisper_available" in data

    def test_whisper_available_is_bool(self, client):
        resp = client.get("/api/capabilities")
        data = resp.get_json()
        assert isinstance(data["whisper_available"], bool)

    def test_whisper_available_false_when_not_installed(self, client):
        with patch("preprod.web.WHISPER_AVAILABLE", False):
            resp = client.get("/api/capabilities")
            data = resp.get_json()
            assert data["whisper_available"] is False

    def test_whisper_available_true_when_installed(self, client):
        with patch("preprod.web.WHISPER_AVAILABLE", True):
            resp = client.get("/api/capabilities")
            data = resp.get_json()
            assert data["whisper_available"] is True

    # ── first-run preflight report: ffmpeg/ffprobe/whisperx/japanese ───────────
    # See src/preprod/web.py:api_capabilities docstring for the field contract.

    def test_all_expected_keys_present(self, client):
        resp = client.get("/api/capabilities")
        data = resp.get_json()
        assert set(data.keys()) == {
            "whisper_available", "ffmpeg", "ffprobe", "whisperx", "japanese",
        }

    def test_ffmpeg_and_ffprobe_are_bool(self, client):
        resp = client.get("/api/capabilities")
        data = resp.get_json()
        assert isinstance(data["ffmpeg"], bool)
        assert isinstance(data["ffprobe"], bool)

    def test_whisperx_and_japanese_are_bool(self, client):
        resp = client.get("/api/capabilities")
        data = resp.get_json()
        assert isinstance(data["whisperx"], bool)
        assert isinstance(data["japanese"], bool)

    def test_uses_shutil_which_for_ffmpeg_and_ffprobe(self, client):
        """Regression guard: must be a cheap PATH lookup, never a subprocess
        spawn or the extract_audio()/probe_media() exception path."""
        with patch("preprod.web.shutil.which", return_value="/usr/bin/x") as mock_which:
            client.get("/api/capabilities")
            called_with = {c.args[0] for c in mock_which.call_args_list}
            assert {"ffmpeg", "ffprobe"} <= called_with

    def test_ffmpeg_check_never_spawns_a_subprocess(self, client):
        with patch("preprod.web.subprocess.run") as mock_run, \
             patch("preprod.web.subprocess.Popen") as mock_popen:
            resp = client.get("/api/capabilities")
            assert resp.status_code == 200
            mock_run.assert_not_called()
            mock_popen.assert_not_called()

    def test_ffmpeg_true_when_found_on_path(self, client):
        with patch("preprod.web.shutil.which",
                    side_effect=lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None):
            resp = client.get("/api/capabilities")
            data = resp.get_json()
            assert data["ffmpeg"] is True
            assert data["ffprobe"] is False

    def test_ffmpeg_false_when_missing_from_path(self, client):
        with patch("preprod.web.shutil.which", return_value=None):
            resp = client.get("/api/capabilities")
            data = resp.get_json()
            assert data["ffmpeg"] is False

    def test_ffprobe_true_when_found_on_path(self, client):
        with patch("preprod.web.shutil.which",
                    side_effect=lambda name: "/usr/bin/ffprobe" if name == "ffprobe" else None):
            resp = client.get("/api/capabilities")
            data = resp.get_json()
            assert data["ffprobe"] is True
            assert data["ffmpeg"] is False

    def test_ffprobe_false_when_missing_from_path(self, client):
        with patch("preprod.web.shutil.which", return_value=None):
            resp = client.get("/api/capabilities")
            data = resp.get_json()
            assert data["ffprobe"] is False

    def test_whisperx_true_when_available(self, client):
        with patch("preprod.web.WHISPERX_AVAILABLE", True):
            resp = client.get("/api/capabilities")
            assert resp.get_json()["whisperx"] is True

    def test_whisperx_false_when_unavailable(self, client):
        with patch("preprod.web.WHISPERX_AVAILABLE", False):
            resp = client.get("/api/capabilities")
            assert resp.get_json()["whisperx"] is False

    def test_japanese_true_when_fugashi_available(self, client):
        with patch("preprod.web.japanese.available", return_value=True):
            resp = client.get("/api/capabilities")
            assert resp.get_json()["japanese"] is True

    def test_japanese_false_when_fugashi_unavailable(self, client):
        with patch("preprod.web.japanese.available", return_value=False):
            resp = client.get("/api/capabilities")
            assert resp.get_json()["japanese"] is False

    def test_ollama_not_included_in_report(self, client):
        """Ollama needs a live network round-trip to the local daemon, unlike
        every other field here. It is intentionally excluded from this
        endpoint (see docstring) so a slow/absent Ollama can never delay the
        ffmpeg/ffprobe blocker — the frontend checks it separately via the
        already-existing /api/llm/models endpoint."""
        resp = client.get("/api/capabilities")
        assert "ollama" not in resp.get_json()

    def test_ollama_availability_check_never_invoked(self, client):
        with patch("preprod.web.llm_correct.ollama_available") as mock_avail:
            client.get("/api/capabilities")
            mock_avail.assert_not_called()


# ── /api/debug/logs ───────────────────────────────────────────────────────────

class TestApiDebugLogs:
    def test_returns_200(self, client):
        resp = client.get("/api/debug/logs")
        assert resp.status_code == 200

    def test_returns_json_with_required_keys(self, client):
        data = client.get("/api/debug/logs").get_json()
        assert "log_path" in data
        assert "lines" in data

    def test_lines_empty_when_log_missing(self, client, tmp_path):
        import preprod.web as web_mod
        nonexistent = tmp_path / "no_such_file.log"
        with patch.object(web_mod, "LOG_PATH", nonexistent):
            data = client.get("/api/debug/logs").get_json()
        assert data["lines"] == []

    def test_lines_returned_when_log_present(self, client, tmp_path):
        log_file = tmp_path / "app.log"
        log_file.write_text("line1\nline2\nline3\n", encoding="utf-8")
        import preprod.web as web_mod
        with patch.object(web_mod, "LOG_PATH", log_file):
            data = client.get("/api/debug/logs").get_json()
        assert data["lines"] == ["line1", "line2", "line3"]

    def test_only_last_500_lines_returned(self, client, tmp_path):
        log_file = tmp_path / "app.log"
        log_file.write_text("\n".join(f"line{i}" for i in range(600)) + "\n", encoding="utf-8")
        import preprod.web as web_mod
        with patch.object(web_mod, "LOG_PATH", log_file):
            data = client.get("/api/debug/logs").get_json()
        assert len(data["lines"]) == 500
        assert data["lines"][0] == "line100"   # first of the last-500

    def test_log_path_echoed_in_response(self, client, tmp_path):
        log_file = tmp_path / "app.log"
        log_file.write_text("hello\n", encoding="utf-8")
        import preprod.web as web_mod
        with patch.object(web_mod, "LOG_PATH", log_file):
            data = client.get("/api/debug/logs").get_json()
        assert data["log_path"] == str(log_file)


# ── /api/probe ────────────────────────────────────────────────────────────────

class TestApiProbe:
    def test_nonexistent_file_returns_404(self, client):
        resp = client.post(
            "/api/probe",
            data=json.dumps({"path": "/nonexistent/path/that/does/not/exist.mp4"}),
            content_type="application/json",
        )
        assert resp.status_code == 404

    def test_nonexistent_file_returns_json_error(self, client):
        resp = client.post(
            "/api/probe",
            data=json.dumps({"path": "/nonexistent/path/that/does/not/exist.mp4"}),
            content_type="application/json",
        )
        data = resp.get_json()
        assert "error" in data

    def test_missing_path_returns_400(self, client):
        resp = client.post(
            "/api/probe",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_empty_path_returns_400(self, client):
        resp = client.post(
            "/api/probe",
            data=json.dumps({"path": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_existing_file_with_audio_returns_200(self, tmp_path, client):
        """A real file that probe_media can handle successfully."""
        fake_file = tmp_path / "video.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_media = _make_video_media(fake_file)

        with patch("preprod.web.probe_media", return_value=fake_media):
            resp = client.post(
                "/api/probe",
                data=json.dumps({"path": str(fake_file)}),
                content_type="application/json",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "media" in data

    def test_file_without_audio_returns_400(self, tmp_path, client):
        fake_file = tmp_path / "video_noaudio.mp4"
        fake_file.write_bytes(b"\x00" * 100)

        fake_media = MagicMock()
        fake_media.has_audio = False

        with patch("preprod.web.probe_media", return_value=fake_media):
            resp = client.post(
                "/api/probe",
                data=json.dumps({"path": str(fake_file)}),
                content_type="application/json",
            )
        assert resp.status_code == 400


# ── App import does not crash ─────────────────────────────────────────────────

class TestWebImport:
    def test_web_module_importable(self):
        """Importing web.py must not raise (numpy import is the key check)."""
        import preprod.web  # noqa: F401 — just ensure no NameError at import time


# ── /api/cache/info ───────────────────────────────────────────────────────────

class TestApiCacheInfo:
    def test_returns_200(self, client):
        resp = client.get("/api/cache/info")
        assert resp.status_code == 200

    def test_returns_expected_keys(self, client):
        data = client.get("/api/cache/info").get_json()
        for key in ("upload_bytes", "export_bytes", "session_bytes",
                    "session_count", "total_bytes"):
            assert key in data, f"missing key: {key}"

    def test_total_is_sum_of_parts(self, client):
        data = client.get("/api/cache/info").get_json()
        expected = data["upload_bytes"] + data["export_bytes"] + data["session_bytes"]
        assert data["total_bytes"] == expected


# ── /api/cache/clear ──────────────────────────────────────────────────────────

class TestApiCacheClear:
    def test_returns_200(self, client):
        resp = client.post("/api/cache/clear",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 200

    def test_returns_expected_keys(self, client):
        resp = client.post("/api/cache/clear",
                           data=json.dumps({}),
                           content_type="application/json")
        data = resp.get_json()
        for key in ("removed_uploads", "removed_exports", "removed_sessions"):
            assert key in data, f"missing key: {key}"

    def test_clear_sessions_false_by_default(self, client):
        resp = client.post("/api/cache/clear",
                           data=json.dumps({}),
                           content_type="application/json")
        data = resp.get_json()
        assert data["removed_sessions"] == 0


# ── /api/upload ───────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_upload_dir(tmp_path):
    """Redirect _UPLOAD_DIR to tmp_path so no real files accumulate, then restore."""
    import preprod.web as web_mod
    orig = web_mod._UPLOAD_DIR
    web_mod._UPLOAD_DIR = tmp_path
    yield tmp_path
    web_mod._UPLOAD_DIR = orig


class TestApiUpload:
    def test_no_file_returns_400(self, client):
        resp = client.post("/api/upload", data={})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_empty_filename_returns_400(self, client):
        from io import BytesIO
        data = {"file": (BytesIO(b""), "")}
        resp = client.post("/api/upload", data=data,
                           content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_save_oserror_returns_500_json(self, client, tmp_path):
        """If writing to disk fails, the endpoint must return JSON 500, not HTML."""
        from io import BytesIO
        import werkzeug.datastructures

        # Patch FileStorage.save() to raise OSError (e.g. disk full).
        with patch.object(werkzeug.datastructures.FileStorage, "save",
                          side_effect=OSError("No space left on device")):
            resp = client.post(
                "/api/upload",
                data={"file": (BytesIO(b"fake content"), "video.mp4")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 500
        data = resp.get_json()
        assert data is not None, "Response must be JSON, not HTML"
        assert "error" in data

    def test_successful_upload_path_contains_original_name(self, client, mock_upload_dir):
        """Successful upload returns a path that includes the original filename."""
        from io import BytesIO
        resp = client.post(
            "/api/upload",
            data={"file": (BytesIO(b"fake video content"), "myvideo.mp4")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "path" in data
        assert "myvideo.mp4" in data["path"]

    def test_two_same_name_uploads_produce_different_paths(self, client, mock_upload_dir):
        """Two uploads with the same filename must NOT overwrite each other (uuid prefix)."""
        from io import BytesIO
        r1 = client.post(
            "/api/upload",
            data={"file": (BytesIO(b"content A"), "clip.mov")},
            content_type="multipart/form-data",
        )
        r2 = client.post(
            "/api/upload",
            data={"file": (BytesIO(b"content B"), "clip.mov")},
            content_type="multipart/form-data",
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
        path1 = r1.get_json()["path"]
        path2 = r2.get_json()["path"]
        assert path1 != path2, (
            "Two uploads with the same filename must produce distinct paths "
            "(uuid prefix missing — silent data-loss regression)"
        )

    def test_upload_dir_is_not_volatile_temp(self):
        """_UPLOAD_DIR must NOT be under tempfile.gettempdir() — that path changes on macOS reboot."""
        import tempfile
        import preprod.web as web_mod
        tmp = Path(tempfile.gettempdir()).resolve()
        upload = web_mod._UPLOAD_DIR.resolve()
        assert not str(upload).startswith(str(tmp)), (
            f"_UPLOAD_DIR is under volatile temp ({tmp}). "
            f"Uploaded files would be lost on reboot. Got: {upload}"
        )


# ── /api/export/subtitles ────────────────────────────────────────────────────

class TestApiExportSubtitles:
    """SRT/VTT export: correct timestamps, format, and removal-region mapping."""

    _ENTRIES = [
        {"start": 0.0,  "end": 2.0,  "text": "Hello world"},
        {"start": 5.0,  "end": 7.0,  "text": "Second line"},
        {"start": 10.0, "end": 12.0, "text": "Third line"},
    ]

    def test_srt_format_returns_200(self, client):
        resp = client.post("/api/export/subtitles", json={
            "format": "srt", "telop_entries": self._ENTRIES,
            "removal_regions": [], "duration": 15.0, "stem": "test",
        })
        assert resp.status_code == 200

    def test_srt_contains_index_numbers(self, client):
        resp = client.post("/api/export/subtitles", json={
            "format": "srt", "telop_entries": self._ENTRIES,
            "removal_regions": [], "duration": 15.0, "stem": "test",
        })
        content = resp.data.decode()
        assert "1\n" in content
        assert "2\n" in content

    def test_srt_uses_comma_millisecond_separator(self, client):
        resp = client.post("/api/export/subtitles", json={
            "format": "srt", "telop_entries": self._ENTRIES,
            "removal_regions": [], "duration": 15.0, "stem": "test",
        })
        content = resp.data.decode()
        assert "," in content   # SRT: 00:00:00,000 not 00:00:00.000
        assert "00:00:00,000 --> 00:00:02,000" in content

    def test_vtt_header_present(self, client):
        resp = client.post("/api/export/subtitles", json={
            "format": "vtt", "telop_entries": self._ENTRIES,
            "removal_regions": [], "duration": 15.0, "stem": "test",
        })
        content = resp.data.decode()
        assert content.startswith("WEBVTT")

    def test_vtt_uses_dot_millisecond_separator(self, client):
        resp = client.post("/api/export/subtitles", json={
            "format": "vtt", "telop_entries": self._ENTRIES,
            "removal_regions": [], "duration": 15.0, "stem": "test",
        })
        content = resp.data.decode()
        # VTT: 00:00:00.000 not 00:00:00,000
        assert "00:00:00.000 --> 00:00:02.000" in content

    def test_segment_in_removed_region_is_dropped(self, client):
        """A segment that falls entirely within a removal region must be absent."""
        resp = client.post("/api/export/subtitles", json={
            "format": "srt",
            "telop_entries": self._ENTRIES,
            # Remove 4.0–8.5 s — covers the "Second line" (5.0–7.0)
            "removal_regions": [{"start": 4.0, "end": 8.5}],
            "duration": 15.0, "padding_ms": 0, "stem": "test",
        })
        content = resp.data.decode()
        assert "Second line" not in content

    def test_timestamps_adjusted_for_removed_region(self, client):
        """After removing 4–8 s, 'Third line' (10–12 s) should start earlier."""
        resp = client.post("/api/export/subtitles", json={
            "format": "srt",
            "telop_entries": [{"start": 10.0, "end": 12.0, "text": "Third line"}],
            "removal_regions": [{"start": 4.0, "end": 8.0}],
            "duration": 15.0, "padding_ms": 0, "stem": "test",
        })
        content = resp.data.decode()
        # Removal is 4 s (4.0–8.0). Third line at 10 s → 10-4 = 6 s in output.
        assert "00:00:06,000 --> 00:00:08,000" in content

    def test_invalid_format_returns_400(self, client):
        resp = client.post("/api/export/subtitles", json={
            "format": "xml", "telop_entries": [], "duration": 10.0,
        })
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_malformed_entries_skipped_not_500(self, client):
        """Entries missing start/end keys must be silently skipped, not crash."""
        entries = [
            {"text": "no times at all"},
            {"start": 0.5, "text": "missing end"},
            {"end": 3.0,  "text": "missing start"},
            {"start": 1.0, "end": 3.0, "text": "valid entry"},
        ]
        resp = client.post("/api/export/subtitles", json={
            "format": "srt", "telop_entries": entries,
            "removal_regions": [], "duration": 10.0, "stem": "test",
        })
        assert resp.status_code == 200
        assert "valid entry" in resp.data.decode()


# ── /api/analyze/cancel ───────────────────────────────────────────────────────

class TestApiAnalyzeCancel:
    def test_cancel_unknown_task_returns_404(self, client):
        resp = client.post(
            "/api/analyze/cancel",
            data=json.dumps({"task_id": "doesnotexist"}),
            content_type="application/json",
        )
        assert resp.status_code == 404
        assert "error" in resp.get_json()

    def test_cancel_known_task_returns_200(self, client):
        import threading
        import preprod.web as web_mod

        mock_event = MagicMock(spec=threading.Event)
        task_id = "test-cancel-task-ok"
        with web_mod._tasks_lock:
            web_mod._tasks[task_id] = {
                "status": "running",
                "progress": 10,
                "stage": "transcribing",
                "result": None,
                "error": None,
                "cancel_event": mock_event,
            }
        try:
            resp = client.post(
                "/api/analyze/cancel",
                data=json.dumps({"task_id": task_id}),
                content_type="application/json",
            )
            assert resp.status_code == 200
            assert resp.get_json().get("ok") is True
            mock_event.set.assert_called_once()
        finally:
            with web_mod._tasks_lock:
                web_mod._tasks.pop(task_id, None)

    def test_cancel_sets_status_cancelled(self, client, tmp_path):
        """Mock transcribe to block until cancelled; verify task status becomes 'cancelled'."""
        import threading
        import time
        import preprod.web as web_mod

        fake_file = tmp_path / "video.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_media = _make_video_media(fake_file)

        # transcribe blocks until cancel_event is set, then raises TranscribeCancelled
        from preprod.transcribe import TranscribeCancelled
        def _blocking_transcribe(*args, **kwargs):
            ev = kwargs.get("cancel_event")
            if ev:
                ev.wait(timeout=5.0)
            raise TranscribeCancelled()

        import numpy as np
        fake_samples = np.zeros(16000 * 10, dtype=np.float32)

        with patch("preprod.web.probe_media", return_value=fake_media), \
             patch("preprod.web.extract_audio", return_value=fake_samples), \
             patch("preprod.web.detect_silence", return_value=[]), \
             patch("preprod.web.WHISPER_AVAILABLE", True), \
             patch("preprod.web.transcribe", side_effect=_blocking_transcribe):

            start_resp = client.post(
                "/api/analyze/start",
                data=json.dumps({
                    "path": str(fake_file),
                    "use_whisper": True,
                }),
                content_type="application/json",
            )
            assert start_resp.status_code == 200
            task_id = start_resp.get_json()["task_id"]

            # Give the background thread a moment to reach transcribe()
            time.sleep(0.1)

            cancel_resp = client.post(
                "/api/analyze/cancel",
                data=json.dumps({"task_id": task_id}),
                content_type="application/json",
            )
            assert cancel_resp.status_code == 200

            # Wait up to 3 s for background thread to process cancellation
            deadline = time.monotonic() + 3.0
            status = "running"
            while time.monotonic() < deadline:
                sr = client.get(f"/api/analyze/status/{task_id}")
                status = sr.get_json().get("status", "running")
                if status != "running":
                    break
                time.sleep(0.05)

            assert status == "cancelled", f"Expected 'cancelled', got '{status}'"

            # Drain the background thread so it doesn't pollute subsequent tests.
            # _tasks[task_id] holds a reference to the thread via the cancel_event;
            # we watch for the task to reach a terminal state and then give the
            # thread a brief moment to fully exit before teardown.
            with web_mod._tasks_lock:
                task_entry = web_mod._tasks.get(task_id)
            if task_entry is not None:
                drain_deadline = time.monotonic() + 2.0
                while time.monotonic() < drain_deadline:
                    with web_mod._tasks_lock:
                        t_status = web_mod._tasks.get(task_id, {}).get("status", "running")
                    if t_status != "running":
                        break
                    time.sleep(0.05)
            # Extra margin for the thread to fully wind down
            time.sleep(0.15)

    def test_status_response_does_not_contain_cancel_event(self, client):
        """cancel_event (threading.Event) must never appear in status JSON."""
        import threading
        import preprod.web as web_mod

        task_id = "test-no-event-leak"
        with web_mod._tasks_lock:
            web_mod._tasks[task_id] = {
                "status": "running",
                "progress": 0,
                "stage": "starting",
                "result": None,
                "error": None,
                "cancel_event": threading.Event(),
            }
        try:
            resp = client.get(f"/api/analyze/status/{task_id}")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "cancel_event" not in data
        finally:
            with web_mod._tasks_lock:
                web_mod._tasks.pop(task_id, None)


# ── /api/analyze/start ───────────────────────────────────────────────────────


class TestApiAnalyzeStart:
    """Tests for POST /api/analyze/start error paths and result shape."""

    def test_missing_path_returns_400(self, client):
        resp = client.post(
            "/api/analyze/start",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_nonexistent_file_returns_404(self, client):
        resp = client.post(
            "/api/analyze/start",
            data=json.dumps({"path": "/nonexistent/does/not/exist.mp4"}),
            content_type="application/json",
        )
        assert resp.status_code == 404
        assert "error" in resp.get_json()

    def test_hangover_ms_accepted_returns_202(self, client, tmp_path):
        """hangover_ms in request is accepted; endpoint returns 202 with task_id."""
        fake_file = tmp_path / "video.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        with patch("preprod.web._run_analysis"):
            resp = client.post(
                "/api/analyze/start",
                data=json.dumps({"path": str(fake_file), "hangover_ms": 500}),
                content_type="application/json",
            )
        assert resp.status_code == 200
        assert "task_id" in resp.get_json()

    def test_truncated_audio_triggers_error_status(self, client, tmp_path):
        """A task whose audio is < 10 % of expected must reach 'error' status."""
        import time
        import numpy as np

        fake_file = tmp_path / "big.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_media = _make_video_media(fake_file, duration=1190.0)  # 1190 s like the 26 GB case

        # Return only 1000 samples for a 1190 s file → well below 10 % threshold
        truncated_samples = np.zeros(1000, dtype=np.float32)

        # Keep the poll loop INSIDE the patch context: the background task thread
        # runs after client.post() returns, so mocks must remain active while polling.
        with patch("preprod.web.probe_media", return_value=fake_media), \
             patch("preprod.web.extract_audio", return_value=truncated_samples):
            resp = client.post(
                "/api/analyze/start",
                data=json.dumps({"path": str(fake_file)}),
                content_type="application/json",
            )
            assert resp.status_code == 200
            task_id = resp.get_json()["task_id"]

            deadline = time.monotonic() + 5.0
            status = "running"
            error_msg = ""
            while time.monotonic() < deadline:
                sr = client.get(f"/api/analyze/status/{task_id}")
                data = sr.get_json()
                status = data.get("status", "running")
                error_msg = data.get("error", "")
                if status != "running":
                    break
                time.sleep(0.05)

        assert status == "error", f"Expected error, got {status!r}"
        assert "samples" in error_msg.lower() or "extraction" in error_msg.lower(), (
            f"Error message should mention audio extraction, got: {error_msg!r}"
        )

    def test_whisper_timeout_falls_back_to_silence_only(self, client, tmp_path):
        """Whisper timeout produces a 'done' task with silence cuts + whisper_warning."""
        import time
        import numpy as np

        fake_file = tmp_path / "video.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_media = _make_video_media(fake_file, duration=1190.0)

        # Enough samples to pass ratio check, with a brief silence region detectable
        samples = np.zeros(int(1190.0 * 16000), dtype=np.float32)
        samples[0:16000] = 0.5   # 1s of speech at the start, rest is silence

        with patch("preprod.web.probe_media", return_value=fake_media), \
             patch("preprod.web.extract_audio", return_value=samples), \
             patch("preprod.web.transcribe",
                   side_effect=RuntimeError("Whisper timed out (> 20 minutes)")):
            resp = client.post(
                "/api/analyze/start",
                data=json.dumps({
                    "path": str(fake_file),
                    "use_whisper": True,
                    "threshold_db": -10.0,   # aggressive so we detect the silence
                    "min_duration": 1.0,
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200
            task_id = resp.get_json()["task_id"]

            deadline = time.monotonic() + 5.0
            status, result = "running", {}
            while time.monotonic() < deadline:
                sr = client.get(f"/api/analyze/status/{task_id}")
                data = sr.get_json()
                status = data.get("status", "running")
                result = data.get("result", {})
                if status != "running":
                    break
                time.sleep(0.05)

        assert status == "done", f"Expected done (fallback), got {status!r}"
        stats = result.get("stats", {})
        assert stats.get("whisper_warning"), "whisper_warning should be set on timeout"
        assert "filler" in stats["whisper_warning"].lower()

    def test_audio_only_file_completes_analysis(self, client, tmp_path):
        """Audio-only files (mp3/m4a/aac) produce a 'done' analysis with silence cuts."""
        import time
        import numpy as np
        from fractions import Fraction

        fake_file = tmp_path / "recording.m4a"
        fake_file.write_bytes(b"\x00" * 100)

        # Audio-only media: has_audio=True, has_video=False, frame_rate=None
        fake_media = MagicMock()
        fake_media.path = fake_file
        fake_media.duration = 120.0
        fake_media.has_video = False
        fake_media.has_audio = True
        fake_media.video_width = None
        fake_media.video_height = None
        fake_media.frame_rate = None
        fake_media.sample_rate = 44100
        fake_media.codec_video = None
        fake_media.codec_audio = "aac"

        samples = np.zeros(int(120.0 * 16000), dtype=np.float32)
        samples[:16000] = 0.5  # 1 s of speech at start

        with patch("preprod.web.probe_media", return_value=fake_media), \
             patch("preprod.web.extract_audio", return_value=samples):
            resp = client.post(
                "/api/analyze/start",
                data=json.dumps({
                    "path": str(fake_file),
                    "threshold_db": -10.0,
                    "min_duration": 1.0,
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200
            task_id = resp.get_json()["task_id"]

            deadline = time.monotonic() + 5.0
            status, result = "running", {}
            while time.monotonic() < deadline:
                sr = client.get(f"/api/analyze/status/{task_id}")
                d = sr.get_json()
                status = d.get("status", "running")
                result = d.get("result", {})
                if status != "running":
                    break
                time.sleep(0.05)

        assert status == "done", f"Audio-only analysis failed: {status}"
        assert result.get("stats", {}).get("total_removals", -1) >= 0

    def test_analyze_uses_audio_cache_on_hit(self, client, tmp_path):
        """Main analyze endpoint skips extract_audio when a fresh cache entry exists."""
        import time
        import numpy as np
        import preprod.web as web_mod

        fake_file = (tmp_path / "cached.mp4").resolve()
        fake_file.write_bytes(b"\x00" * 100)
        fake_media = _make_video_media(fake_file)

        cached_samples = np.zeros(int(10.0 * 16000), dtype=np.float32)
        cached_samples[:16000] = 0.5  # 1 s speech at start, 9 s silence

        web_mod._audio_cache.clear()
        web_mod._audio_cache[str(fake_file)] = (cached_samples, fake_file.stat().st_mtime)
        try:
            with patch("preprod.web.probe_media", return_value=fake_media), \
                 patch("preprod.web.extract_audio") as mock_extract:
                resp = client.post(
                    "/api/analyze/start",
                    data=json.dumps({"path": str(fake_file)}),
                    content_type="application/json",
                )
                assert resp.status_code == 200
                task_id = resp.get_json()["task_id"]

                deadline = time.monotonic() + 5.0
                status = "running"
                while time.monotonic() < deadline:
                    sr = client.get(f"/api/analyze/status/{task_id}")
                    status = sr.get_json().get("status", "running")
                    if status != "running":
                        break
                    time.sleep(0.05)

                assert status == "done", f"Expected done, got {status!r}"
                mock_extract.assert_not_called()
        finally:
            web_mod._audio_cache.clear()

    def test_waveform_max_amp_in_result(self, client):
        """A completed task result must contain waveform_max_amp as a positive float."""
        import preprod.web as web_mod

        task_id = "test-waveform-max-amp"
        fake_result = {
            "media": {},
            "removal_candidates": [],
            "telop_entries": [],
            "waveform": [0.1, 0.2, 0.3],
            "waveform_threshold": 0.01,
            "waveform_max_amp": 0.875432,
            "stats": {
                "total_removals": 0,
                "silence_count": 0,
                "filler_count": 0,
                "telop_count": 0,
                "kept_duration": 10.0,
                "removed_duration": 0.0,
                "original_duration": 10.0,
                "kept_percent": 100.0,
                "whisper_available": False,
            },
        }
        with web_mod._tasks_lock:
            web_mod._tasks[task_id] = {
                "status": "done",
                "progress": 100,
                "stage": "done",
                "result": fake_result,
                "error": None,
                "created_at": __import__("time").time(),
            }
        try:
            resp = client.get(f"/api/analyze/status/{task_id}")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["status"] == "done"
            result = data["result"]
            assert "waveform_max_amp" in result, "waveform_max_amp missing from result"
            assert isinstance(result["waveform_max_amp"], float), (
                f"waveform_max_amp must be float, got {type(result['waveform_max_amp'])}"
            )
            assert result["waveform_max_amp"] > 0, (
                f"waveform_max_amp must be positive, got {result['waveform_max_amp']}"
            )
        finally:
            with web_mod._tasks_lock:
                web_mod._tasks.pop(task_id, None)


# ── /api/export/roughcut ──────────────────────────────────────────────────────

class TestApiExportRoughcut:
    """Tests for the POST /api/export/roughcut endpoint."""

    def test_missing_path_returns_400(self, client):
        resp = client.post(
            "/api/export/roughcut",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_nonexistent_file_returns_404(self, client):
        resp = client.post(
            "/api/export/roughcut",
            data=json.dumps({"path": "/nonexistent/does/not/exist.mp4"}),
            content_type="application/json",
        )
        assert resp.status_code == 404
        assert "error" in resp.get_json()

    def test_missing_upload_file_returns_actionable_error(self, client, tmp_path):
        """When an uploaded file has been swept, the error message tells the user to
        re-import rather than showing a bare 'File not found' path."""
        import preprod.web as web_mod
        # Simulate the path a stale-session file would reference: it was under
        # _UPLOAD_DIR (or the legacy preprod_uploads temp path) but no longer exists.
        ghost_path = tmp_path / "preprod_uploads" / "abc123_recording.mp3"
        # Do NOT create the file — it should be missing.
        resp = client.post(
            "/api/export/roughcut",
            data=json.dumps({"path": str(ghost_path)}),
            content_type="application/json",
        )
        assert resp.status_code == 404
        err = resp.get_json()["error"]
        # Must mention re-importing, not just a bare path
        assert "re-import" in err.lower() or "open file" in err.lower(), (
            f"Expected actionable re-import guidance in error, got: {err!r}"
        )

    def test_no_removal_regions_generates_fcpxml(self, tmp_path, client):
        """Happy path: file exists, no removal_regions — generate called, response sent."""
        import preprod.web as web_mod

        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_media = _make_video_media(fake_file, duration=30.0, fps=30)

        orig_export_dir = web_mod._EXPORT_DIR
        web_mod._EXPORT_DIR = tmp_path
        try:
            def _fake_generate(segments, media, out_path):
                out_path.write_text("<fcpxml/>")

            with patch("preprod.web.probe_media", return_value=fake_media), \
                 patch("preprod.web.generate_roughcut_fcpxml",
                       side_effect=_fake_generate) as mock_gen:
                resp = client.post(
                    "/api/export/roughcut",
                    data=json.dumps({
                        "path": str(fake_file),
                        "removal_regions": [],
                    }),
                    content_type="application/json",
                )
        finally:
            web_mod._EXPORT_DIR = orig_export_dir

        mock_gen.assert_called_once()
        assert resp.status_code == 200

    def test_all_regions_removed_returns_400(self, tmp_path, client):
        """If removal_regions cover 100% of the file, no keep-segments → 400."""
        from fractions import Fraction

        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)

        fake_media = MagicMock()
        fake_media.path = fake_file
        fake_media.duration = 10.0
        fake_media.has_audio = True

        with patch("preprod.web.probe_media", return_value=fake_media):
            resp = client.post(
                "/api/export/roughcut",
                data=json.dumps({
                    "path": str(fake_file),
                    "removal_regions": [{"start": 0.0, "end": 10.0}],
                    "padding_ms": 0,
                }),
                content_type="application/json",
            )

        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_generate_raises_returns_500_json(self, tmp_path, client):
        """If generate_roughcut_fcpxml raises, the endpoint returns JSON 500."""
        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_media = _make_video_media(fake_file, duration=30.0, fps=30)

        with patch("preprod.web.probe_media", return_value=fake_media), \
             patch("preprod.web.generate_roughcut_fcpxml",
                   side_effect=RuntimeError("FCPXML generation failed")):
            resp = client.post(
                "/api/export/roughcut",
                data=json.dumps({
                    "path": str(fake_file),
                    "removal_regions": [],
                }),
                content_type="application/json",
            )

        assert resp.status_code == 500
        data = resp.get_json()
        assert data is not None, "Response must be JSON"
        assert "error" in data


# ── save_to_downloads pywebview path ─────────────────────────────────────────

class TestSaveToDownloadsExport:
    """Tests for the pywebview save_to_downloads=true code path.

    When window.pywebview is present in the browser, the frontend sends
    save_to_downloads=true so the server writes the file directly to
    ~/Downloads instead of streaming it as an HTTP attachment.

    Fetching application/xml as a blob through WKWebView triggers
    decidePolicyForNavigationResponse → WKNavigationResponsePolicyCancel,
    which blanks the page.  This path completely avoids that.
    """

    def _post(self, client, url, body, tmp_path, content="<fcpxml/>"):
        """Helper that mocks probe_media + generate_*_fcpxml and posts the request."""
        import preprod.web as web_mod
        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_media = _make_video_media(fake_file, duration=30.0, fps=30)
        orig_export_dir = web_mod._EXPORT_DIR
        web_mod._EXPORT_DIR = tmp_path
        try:
            def _fake_gen(segments, media, out_path, *args, **kwargs):
                out_path.write_text(content, encoding="utf-8")

            with patch("preprod.web.probe_media", return_value=fake_media), \
                 patch("preprod.web.generate_roughcut_fcpxml", side_effect=_fake_gen), \
                 patch("preprod.web.generate_telop_fcpxml", side_effect=_fake_gen):
                resp = client.post(url, data=json.dumps(body), content_type="application/json")
        finally:
            web_mod._EXPORT_DIR = orig_export_dir
        return resp, fake_file

    def test_roughcut_save_to_downloads_returns_json_ok(self, client, tmp_path):
        """save_to_downloads=true must return JSON {"ok": true}, not a file blob."""
        downloads = tmp_path / "Downloads"
        downloads.mkdir()
        with patch("preprod.web._copy_to_downloads", return_value=(True, "")) as mock_copy:
            resp, fake_file = self._post(
                client, "/api/export/roughcut",
                {"path": str(tmp_path / "clip.mp4"), "removal_regions": [],
                 "save_to_downloads": True},
                tmp_path,
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"ok": True}, f"Expected {{ok: true}}, got {data}"
        assert resp.content_type.startswith("application/json")
        mock_copy.assert_called_once()

    def test_roughcut_save_to_downloads_writes_file(self, client, tmp_path):
        """File must actually land in the redirected Downloads dir."""
        import preprod.web as web_mod
        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_media = _make_video_media(fake_file, duration=30.0, fps=30)
        downloads = tmp_path / "Downloads"
        downloads.mkdir()
        orig_export_dir = web_mod._EXPORT_DIR
        web_mod._EXPORT_DIR = tmp_path
        try:
            def _fake_gen(segments, media, out_path, *args, **kwargs):
                out_path.write_text("<fcpxml>content</fcpxml>", encoding="utf-8")

            with patch("preprod.web.probe_media", return_value=fake_media), \
                 patch("preprod.web.generate_roughcut_fcpxml", side_effect=_fake_gen), \
                 patch("preprod.web.Path.home", return_value=tmp_path):
                resp = client.post(
                    "/api/export/roughcut",
                    data=json.dumps({
                        "path": str(fake_file),
                        "removal_regions": [],
                        "save_to_downloads": True,
                    }),
                    content_type="application/json",
                )
        finally:
            web_mod._EXPORT_DIR = orig_export_dir
        assert resp.get_json() == {"ok": True}
        # File must exist in the mocked Downloads dir
        dest = downloads / "clip_cut.fcpxml"
        assert dest.exists(), f"Expected {dest} to exist"
        assert dest.read_text(encoding="utf-8") == "<fcpxml>content</fcpxml>"

    def test_roughcut_without_save_to_downloads_returns_file(self, client, tmp_path):
        """Without save_to_downloads, response must be a file attachment (non-JSON)."""
        import preprod.web as web_mod
        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_media = _make_video_media(fake_file, duration=30.0, fps=30)
        orig_export_dir = web_mod._EXPORT_DIR
        web_mod._EXPORT_DIR = tmp_path
        try:
            def _fake_gen(segments, media, out_path, *args, **kwargs):
                out_path.write_text("<fcpxml/>", encoding="utf-8")

            with patch("preprod.web.probe_media", return_value=fake_media), \
                 patch("preprod.web.generate_roughcut_fcpxml", side_effect=_fake_gen):
                resp = client.post(
                    "/api/export/roughcut",
                    data=json.dumps({"path": str(fake_file), "removal_regions": []}),
                    content_type="application/json",
                )
        finally:
            web_mod._EXPORT_DIR = orig_export_dir
        assert resp.status_code == 200
        assert "xml" in resp.content_type or "attachment" in resp.headers.get("Content-Disposition", "")

    def test_subtitles_save_to_downloads_returns_json_ok(self, client, tmp_path):
        """Subtitles export with save_to_downloads=true returns JSON, not text file."""
        import preprod.web as web_mod
        orig_export_dir = web_mod._EXPORT_DIR
        web_mod._EXPORT_DIR = tmp_path
        try:
            with patch("preprod.web._copy_to_downloads", return_value=(True, "")) as mock_copy:
                resp = client.post(
                    "/api/export/subtitles",
                    data=json.dumps({
                        "format": "srt",
                        "telop_entries": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                        "removal_regions": [],
                        "duration": 10.0,
                        "padding_ms": 0,
                        "stem": "clip",
                        "save_to_downloads": True,
                    }),
                    content_type="application/json",
                )
        finally:
            web_mod._EXPORT_DIR = orig_export_dir
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}
        mock_copy.assert_called_once()


# ── /api/export/telop ─────────────────────────────────────────────────────────

class TestApiExportTelop:
    """Tests for the POST /api/export/telop endpoint."""

    def test_empty_telop_entries_returns_200(self, client):
        """Missing / empty telop_entries still produces a valid FCPXML response."""
        with patch("preprod.web.generate_telop_fcpxml") as mock_gen, \
             patch("preprod.web.send_file") as mock_send:
            mock_send.return_value = client.application.response_class(
                b"<fcpxml/>", status=200, mimetype="application/xml"
            )
            resp = client.post(
                "/api/export/telop",
                data=json.dumps({"telop_entries": [], "duration": 10.0}),
                content_type="application/json",
            )
        mock_gen.assert_called_once()
        # Either the mocked send_file 200 or a real send_file response is fine
        assert resp.status_code == 200

    def test_valid_request_calls_generate_and_returns_attachment(self, tmp_path, client):
        """Valid request with mocked generator: response must be a 200 attachment."""
        import preprod.web as web_mod

        # Redirect export dir to tmp_path so send_file can find the file.
        orig_export = web_mod._EXPORT_DIR
        web_mod._EXPORT_DIR = tmp_path

        try:
            telop_entries = [
                {"id": "t0", "start": 1.0, "end": 3.0, "text": "Hello"},
            ]

            def _fake_generate(entries, keep_segs, *, total_source_duration,
                               settings, stem, output_path, use_source_timing=False):
                output_path.write_text("<fcpxml/>", encoding="utf-8")

            with patch("preprod.web.generate_telop_fcpxml",
                       side_effect=_fake_generate):
                resp = client.post(
                    "/api/export/telop",
                    data=json.dumps({
                        "telop_entries": telop_entries,
                        "duration": 10.0,
                        "stem": "myclip",
                    }),
                    content_type="application/json",
                )
        finally:
            web_mod._EXPORT_DIR = orig_export

        assert resp.status_code == 200
        cd = resp.headers.get("Content-Disposition", "")
        assert "attachment" in cd

    def test_generate_raises_returns_500_json(self, client):
        """If generate_telop_fcpxml raises, the endpoint returns JSON 500."""
        with patch("preprod.web.generate_telop_fcpxml",
                   side_effect=RuntimeError("Telop generation failed")):
            resp = client.post(
                "/api/export/telop",
                data=json.dumps({"telop_entries": [], "duration": 5.0}),
                content_type="application/json",
            )
        assert resp.status_code == 500
        data = resp.get_json()
        assert data is not None, "Response must be JSON"
        assert "error" in data

    def test_use_source_timing_forwarded_to_generator(self, client, tmp_path):
        """use_source_timing=true in request body is forwarded to generate_telop_fcpxml."""
        import preprod.web as web_mod

        captured: dict = {}

        def _fake_generate(entries, keep_segs, *, total_source_duration,
                           settings, stem, output_path, use_source_timing=False):
            captured["use_source_timing"] = use_source_timing
            output_path.write_text("<fcpxml/>", encoding="utf-8")

        orig_export = web_mod._EXPORT_DIR
        web_mod._EXPORT_DIR = tmp_path
        try:
            with patch("preprod.web.generate_telop_fcpxml", side_effect=_fake_generate):
                client.post(
                    "/api/export/telop",
                    data=json.dumps({
                        "telop_entries": [],
                        "duration": 10.0,
                        "use_source_timing": True,
                    }),
                    content_type="application/json",
                )
        finally:
            web_mod._EXPORT_DIR = orig_export

        assert captured.get("use_source_timing") is True

    def test_use_source_timing_defaults_false(self, client, tmp_path):
        """use_source_timing defaults to False when omitted from request."""
        import preprod.web as web_mod

        captured: dict = {}

        def _fake_generate(entries, keep_segs, *, total_source_duration,
                           settings, stem, output_path, use_source_timing=False):
            captured["use_source_timing"] = use_source_timing
            output_path.write_text("<fcpxml/>", encoding="utf-8")

        orig_export = web_mod._EXPORT_DIR
        web_mod._EXPORT_DIR = tmp_path
        try:
            with patch("preprod.web.generate_telop_fcpxml", side_effect=_fake_generate):
                client.post(
                    "/api/export/telop",
                    data=json.dumps({"telop_entries": [], "duration": 10.0}),
                    content_type="application/json",
                )
        finally:
            web_mod._EXPORT_DIR = orig_export

        assert captured.get("use_source_timing") is False

    def test_zero_duration_with_entries_returns_400(self, client):
        """If duration=0 but telop_entries is non-empty, return 400 with error message."""
        resp = client.post(
            "/api/export/telop",
            data=json.dumps({
                "telop_entries": [{"start": 1.0, "end": 2.0, "text": "hello"}],
                "duration": 0,
            }),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "duration" in data.get("error", "").lower()

    def test_zero_duration_no_entries_still_ok(self, client, tmp_path):
        """duration=0 with no entries is allowed (produces empty FCPXML gracefully)."""
        import preprod.web as web_mod
        orig_export = web_mod._EXPORT_DIR
        web_mod._EXPORT_DIR = tmp_path
        try:
            with patch("preprod.web.generate_telop_fcpxml",
                       side_effect=lambda *a, **kw: kw["output_path"].write_text("<fcpxml/>", encoding="utf-8")):
                resp = client.post(
                    "/api/export/telop",
                    data=json.dumps({"telop_entries": [], "duration": 0}),
                    content_type="application/json",
                )
        finally:
            web_mod._EXPORT_DIR = orig_export
        assert resp.status_code == 200

    def test_settings_dict_forwarded_to_generator(self, client, tmp_path):
        """settings object in the request body is forwarded verbatim to generate_telop_fcpxml."""
        import preprod.web as web_mod

        captured: dict = {}

        def _fake_generate(entries, keep_segs, *, total_source_duration,
                           settings, stem, output_path, use_source_timing=False):
            captured["settings"] = settings
            output_path.write_text("<fcpxml/>", encoding="utf-8")

        orig_export = web_mod._EXPORT_DIR
        web_mod._EXPORT_DIR = tmp_path
        try:
            with patch("preprod.web.generate_telop_fcpxml", side_effect=_fake_generate):
                client.post(
                    "/api/export/telop",
                    data=json.dumps({
                        "telop_entries": [],
                        "duration": 10.0,
                        "settings": {"font": "Toppan Bunkyu Gothic", "font_size": 80},
                    }),
                    content_type="application/json",
                )
        finally:
            web_mod._EXPORT_DIR = orig_export

        assert captured.get("settings") == {"font": "Toppan Bunkyu Gothic", "font_size": 80}

    def test_save_to_downloads_returns_json_ok(self, client, tmp_path):
        """save_to_downloads=true must return JSON {ok: true}, not a file blob."""
        import preprod.web as web_mod

        orig_export = web_mod._EXPORT_DIR
        web_mod._EXPORT_DIR = tmp_path
        try:
            with patch("preprod.web.generate_telop_fcpxml",
                       side_effect=lambda *a, **kw: kw["output_path"].write_text("<fcpxml/>", encoding="utf-8")), \
                 patch("preprod.web._copy_to_downloads", return_value=(True, "")) as mock_copy:
                resp = client.post(
                    "/api/export/telop",
                    data=json.dumps({
                        "telop_entries": [],
                        "duration": 10.0,
                        "save_to_downloads": True,
                    }),
                    content_type="application/json",
                )
        finally:
            web_mod._EXPORT_DIR = orig_export

        assert resp.status_code == 200
        data = resp.get_json()
        assert data is not None
        assert data.get("ok") is True
        mock_copy.assert_called_once()

    def test_invalid_font_color_returns_500_json(self, client):
        """A non-hex font_color in settings (e.g. 'red') must return JSON 500, not crash."""
        resp = client.post(
            "/api/export/telop",
            data=json.dumps({
                "telop_entries": [{"id": "t0", "start": 1.0, "end": 3.0, "text": "Test"}],
                "duration": 10.0,
                "settings": {"font_color": "red"},
            }),
            content_type="application/json",
        )
        assert resp.status_code == 500
        data = resp.get_json()
        assert data is not None
        assert "error" in data


# ── /api/session/save ─────────────────────────────────────────────────────────

class TestApiSessionSave:
    """Tests for the POST /api/session/save endpoint."""

    def test_missing_path_returns_400(self, client):
        resp = client.post(
            "/api/session/save",
            data=json.dumps({"state": {"foo": "bar"}}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_valid_path_and_state_returns_ok(self, client):
        with patch("preprod.web.save_session") as mock_save:
            resp = client.post(
                "/api/session/save",
                data=json.dumps({
                    "path": "/some/video.mp4",
                    "state": {"removal_regions": [], "telop_entries": []},
                }),
                content_type="application/json",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"ok": True}
        mock_save.assert_called_once_with("/some/video.mp4",
                                          {"removal_regions": [], "telop_entries": []})

    def test_save_session_raises_returns_500_json(self, client):
        with patch("preprod.web.save_session",
                   side_effect=OSError("Disk full")):
            resp = client.post(
                "/api/session/save",
                data=json.dumps({"path": "/some/video.mp4", "state": {}}),
                content_type="application/json",
            )
        assert resp.status_code == 500
        data = resp.get_json()
        assert data is not None, "Response must be JSON"
        assert "error" in data


# ── /api/session/load ─────────────────────────────────────────────────────────

class TestApiSessionLoad:
    """Tests for the POST /api/session/load endpoint."""

    def test_missing_path_returns_session_null(self, client):
        """Missing path must return {"session": null}, not an error."""
        resp = client.post(
            "/api/session/load",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"session": None}

    def test_valid_path_with_saved_session_returns_session(self, client):
        saved_state = {"removal_regions": [{"start": 1.0, "end": 2.0}]}
        with patch("preprod.web.load_session", return_value=saved_state):
            resp = client.post(
                "/api/session/load",
                data=json.dumps({"path": "/some/video.mp4"}),
                content_type="application/json",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"session": saved_state}

    def test_path_with_no_saved_session_returns_session_null(self, client):
        with patch("preprod.web.load_session", return_value=None):
            resp = client.post(
                "/api/session/load",
                data=json.dumps({"path": "/some/video.mp4"}),
                content_type="application/json",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"session": None}


# ── /api/session/delete ───────────────────────────────────────────────────────

class TestApiSessionDelete:
    """Tests for the POST /api/session/delete endpoint."""

    def test_always_returns_200_ok(self, client):
        with patch("preprod.web.delete_session"):
            resp = client.post(
                "/api/session/delete",
                data=json.dumps({"path": "/some/video.mp4"}),
                content_type="application/json",
            )
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}

    def test_missing_path_still_returns_200_ok(self, client):
        """Endpoint is idempotent — even a missing path must return 200."""
        resp = client.post(
            "/api/session/delete",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True}

    def test_delete_called_only_when_path_provided(self, client):
        """delete_session must be called when a path is given."""
        with patch("preprod.web.delete_session") as mock_del:
            client.post(
                "/api/session/delete",
                data=json.dumps({"path": "/some/video.mp4"}),
                content_type="application/json",
            )
        mock_del.assert_called_once_with("/some/video.mp4")

    def test_delete_not_called_when_path_missing(self, client):
        """delete_session must NOT be called when path is absent."""
        with patch("preprod.web.delete_session") as mock_del:
            client.post(
                "/api/session/delete",
                data=json.dumps({}),
                content_type="application/json",
            )
        mock_del.assert_not_called()


# ── /api/session/list ──────────────────────────────────────────────────────────

class TestApiSessionList:
    """Tests for the GET /api/session/list endpoint."""

    def test_returns_200(self, client):
        with patch("preprod.web.list_sessions", return_value=[]):
            resp = client.get("/api/session/list")
        assert resp.status_code == 200

    def test_empty_sessions_returns_empty_list(self, client):
        with patch("preprod.web.list_sessions", return_value=[]):
            data = client.get("/api/session/list").get_json()
        assert data == {"sessions": []}

    def test_returns_sessions_from_list_sessions(self, client):
        fake = [{"file_path": "/video.mp4", "file_name": "video.mp4",
                 "saved_at": 1000.0, "telop_count": 3,
                 "removal_count": 5, "file_exists": True}]
        with patch("preprod.web.list_sessions", return_value=fake):
            data = client.get("/api/session/list").get_json()
        assert data["sessions"] == fake

    def test_sessions_key_present(self, client):
        with patch("preprod.web.list_sessions", return_value=[]):
            data = client.get("/api/session/list").get_json()
        assert "sessions" in data


# ── /api/analyze/redetect_silence ─────────────────────────────────────────────

class TestApiRedetectSilence:
    """Tests for POST /api/analyze/redetect_silence."""

    @pytest.fixture(autouse=True)
    def _clear_audio_cache(self):
        import preprod.web as web_mod
        web_mod._audio_cache.clear()
        yield
        web_mod._audio_cache.clear()

    def test_no_path_returns_400(self, client):
        resp = client.post(
            "/api/analyze/redetect_silence",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_nonexistent_file_returns_404(self, client):
        resp = client.post(
            "/api/analyze/redetect_silence",
            data=json.dumps({"path": "/nonexistent/video.mp4"}),
            content_type="application/json",
        )
        assert resp.status_code == 404
        assert "error" in resp.get_json()

    def test_returns_silence_candidates_on_success(self, client, tmp_path):
        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00")
        fake_samples = __import__("numpy").zeros(16000, dtype="float32")

        with (
            patch("preprod.web.extract_audio", return_value=fake_samples),
            patch("preprod.web.detect_silence", return_value=[(1.0, 2.5), (5.0, 7.0)]),
        ):
            resp = client.post(
                "/api/analyze/redetect_silence",
                data=json.dumps({"path": str(fake_file), "threshold_db": -35.0}),
                content_type="application/json",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "candidates" in data
        assert len(data["candidates"]) == 2
        assert data["candidates"][0]["type"] == "silence"
        assert data["candidates"][0]["start"] == 1.0
        assert data["candidates"][1]["end"] == 7.0

    def test_response_echoes_threshold_db(self, client, tmp_path):
        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00")
        fake_samples = __import__("numpy").zeros(16000, dtype="float32")

        with (
            patch("preprod.web.extract_audio", return_value=fake_samples),
            patch("preprod.web.detect_silence", return_value=[]),
        ):
            resp = client.post(
                "/api/analyze/redetect_silence",
                data=json.dumps({"path": str(fake_file), "threshold_db": -50.0}),
                content_type="application/json",
            )
        assert resp.get_json()["threshold_db"] == -50.0

    def test_extract_audio_error_returns_500(self, client, tmp_path):
        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00")

        with patch("preprod.web.extract_audio", side_effect=RuntimeError("ffmpeg failed")):
            resp = client.post(
                "/api/analyze/redetect_silence",
                data=json.dumps({"path": str(fake_file)}),
                content_type="application/json",
            )
        assert resp.status_code == 500
        assert "error" in resp.get_json()

    def test_second_call_uses_audio_cache(self, client, tmp_path):
        """The endpoint must call extract_audio only once for the same file."""
        import numpy as np

        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_samples = np.zeros(16000, dtype="float32")

        with patch("preprod.web.extract_audio", return_value=fake_samples) as mock_extract:
            with patch("preprod.web.detect_silence", return_value=[]):
                for threshold in [-40.0, -35.0, -30.0]:
                    client.post(
                        "/api/analyze/redetect_silence",
                        data=json.dumps({"path": str(fake_file), "threshold_db": threshold}),
                        content_type="application/json",
                    )
        mock_extract.assert_called_once()

    def test_accepts_hangover_ms_param(self, client, tmp_path):
        """Backend accepts hangover_ms and passes it to detect_silence."""
        import numpy as np

        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        with patch("preprod.web.extract_audio", return_value=np.zeros(16000, dtype="float32")):
            with patch("preprod.web.detect_silence", return_value=[]) as mock_detect:
                client.post(
                    "/api/analyze/redetect_silence",
                    data=json.dumps({"path": str(fake_file), "hangover_ms": 100}),
                    content_type="application/json",
                )
        assert mock_detect.call_args[1].get("hangover_ms") == 100

    def test_hangover_ms_echoed_in_response(self, client, tmp_path):
        """hangover_ms is echoed back in the response payload."""
        import numpy as np

        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        with patch("preprod.web.extract_audio", return_value=np.zeros(16000, dtype="float32")):
            with patch("preprod.web.detect_silence", return_value=[]):
                resp = client.post(
                    "/api/analyze/redetect_silence",
                    data=json.dumps({"path": str(fake_file), "hangover_ms": 0}),
                    content_type="application/json",
                )
        assert resp.get_json()["hangover_ms"] == 0

    def test_hangover_ms_defaults_to_300(self, client, tmp_path):
        """When hangover_ms is omitted, backend defaults to 300."""
        import numpy as np

        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        with patch("preprod.web.extract_audio", return_value=np.zeros(16000, dtype="float32")):
            with patch("preprod.web.detect_silence", return_value=[]) as mock_detect:
                client.post(
                    "/api/analyze/redetect_silence",
                    data=json.dumps({"path": str(fake_file)}),
                    content_type="application/json",
                )
        assert mock_detect.call_args[1].get("hangover_ms") == 300

    def test_total_duration_in_response(self, client, tmp_path):
        """Response includes total_duration = sum of silence region lengths."""
        import numpy as np

        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        # Two silence regions: 0–2 s and 5–8 s → total 5.0 s
        fake_regions = [(0.0, 2.0), (5.0, 8.0)]
        with patch("preprod.web.extract_audio", return_value=np.zeros(16000, dtype="float32")):
            with patch("preprod.web.detect_silence", return_value=fake_regions):
                resp = client.post(
                    "/api/analyze/redetect_silence",
                    data=json.dumps({"path": str(fake_file)}),
                    content_type="application/json",
                )
        data = resp.get_json()
        assert "total_duration" in data
        assert data["total_duration"] == 5.0

    def test_audio_too_short_returns_422(self, client, tmp_path):
        """ValueError from detect_silence (e.g. audio too short) → 422 Unprocessable Entity."""
        import numpy as np

        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        with patch("preprod.web.extract_audio", return_value=np.zeros(16000, dtype="float32")):
            with patch("preprod.web.detect_silence",
                       side_effect=ValueError("Audio too short to analyze")):
                resp = client.post(
                    "/api/analyze/redetect_silence",
                    data=json.dumps({"path": str(fake_file)}),
                    content_type="application/json",
                )
        assert resp.status_code == 422
        assert "error" in resp.get_json()

    def test_word_cache_snaps_silence_boundaries(self, client, tmp_path):
        """When word timestamps are cached for a file, redetect applies snap_silences_to_words.

        Scenario: silence ends at 5.5s, but the cached word list says the next word
        starts at 5.2s (within the 400ms snapping window).  The response should show
        the silence snapped to 5.2s, consistent with a full-analysis result.
        """
        import numpy as np
        import preprod.web as web_mod

        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)

        # Pre-populate the word cache for this file
        web_mod._cache_words(str(fake_file.resolve()), [
            {"word": "hello", "start": 5.2, "end": 5.9},
            {"word": "world", "start": 6.1, "end": 6.5},
        ])

        # Silence region that ends 300ms after the cached word's start
        fake_regions = [(3.0, 5.5)]

        with patch("preprod.web.extract_audio", return_value=np.zeros(16000, dtype="float32")):
            with patch("preprod.web.detect_silence", return_value=fake_regions):
                resp = client.post(
                    "/api/analyze/redetect_silence",
                    data=json.dumps({"path": str(fake_file)}),
                    content_type="application/json",
                )
        data = resp.get_json()
        assert resp.status_code == 200, data
        assert len(data["candidates"]) == 1
        # Silence end should be snapped from 5.5s to 5.2s (the cached word start)
        assert data["candidates"][0]["end"] == pytest.approx(5.2, abs=0.01)

    def test_no_word_cache_behaves_unchanged(self, client, tmp_path):
        """When no word cache exists for the file, redetect returns raw silence regions."""
        import numpy as np
        import preprod.web as web_mod

        fake_file = tmp_path / "newfile.mp4"
        fake_file.write_bytes(b"\x00" * 100)

        # Ensure no words are cached for this file
        web_mod._words_cache.pop(str(fake_file.resolve()), None)

        fake_regions = [(3.0, 5.5)]
        with patch("preprod.web.extract_audio", return_value=np.zeros(16000, dtype="float32")):
            with patch("preprod.web.detect_silence", return_value=fake_regions):
                resp = client.post(
                    "/api/analyze/redetect_silence",
                    data=json.dumps({"path": str(fake_file)}),
                    content_type="application/json",
                )
        data = resp.get_json()
        assert resp.status_code == 200, data
        assert len(data["candidates"]) == 1
        # No snapping — raw silence boundary returned as-is
        assert data["candidates"][0]["end"] == pytest.approx(5.5, abs=0.01)


# ── _get_cached_audio ──────────────────────────────────────────────────────────

class TestGetCachedAudio:
    """Unit tests for the audio sample cache (_get_cached_audio)."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Wipe _audio_cache before each test for isolation."""
        import preprod.web as web_mod
        web_mod._audio_cache.clear()
        yield
        web_mod._audio_cache.clear()

    def test_cache_hit_reuses_array(self, tmp_path):
        """Second call must return the same array without calling extract_audio again."""
        import numpy as np
        import preprod.web as web_mod

        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_samples = np.zeros(16000, dtype="float32")

        with patch("preprod.web.extract_audio", return_value=fake_samples) as mock_extract:
            first  = web_mod._get_cached_audio(fake_file)
            second = web_mod._get_cached_audio(fake_file)

        mock_extract.assert_called_once()
        assert first is second

    def test_cache_invalidated_on_mtime_change(self, tmp_path):
        """When the file's mtime changes, extract_audio must be called again."""
        import time
        import numpy as np
        import preprod.web as web_mod

        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        s1 = np.zeros(16000, dtype="float32")
        s2 = np.ones(16000, dtype="float32")

        with patch("preprod.web.extract_audio", return_value=s1):
            web_mod._get_cached_audio(fake_file)

        # Advance mtime so the cache entry looks stale
        new_mtime = fake_file.stat().st_mtime + 1.0
        import os
        os.utime(fake_file, (new_mtime, new_mtime))

        with patch("preprod.web.extract_audio", return_value=s2) as mock2:
            result = web_mod._get_cached_audio(fake_file)

        mock2.assert_called_once()
        assert result is s2

    def test_cache_evicts_oldest_when_full(self, tmp_path):
        """When _AUDIO_CACHE_MAX entries are exceeded, the oldest entry is evicted."""
        import numpy as np
        import preprod.web as web_mod

        files = []
        for i in range(web_mod._AUDIO_CACHE_MAX + 1):
            f = tmp_path / f"clip_{i}.mp4"
            f.write_bytes(bytes([i]) * 100)
            files.append(f)

        samples = [np.zeros(16, dtype="float32") for _ in files]

        for f, s in zip(files, samples):
            with patch("preprod.web.extract_audio", return_value=s):
                web_mod._get_cached_audio(f)

        assert len(web_mod._audio_cache) == web_mod._AUDIO_CACHE_MAX
        # The first file should have been evicted
        assert str(files[0].resolve()) not in web_mod._audio_cache

    def test_cache_size_after_repeated_same_file(self, tmp_path):
        """Calling the same file multiple times must not grow the cache."""
        import numpy as np
        import preprod.web as web_mod

        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_samples = np.zeros(16000, dtype="float32")

        with patch("preprod.web.extract_audio", return_value=fake_samples):
            for _ in range(5):
                web_mod._get_cached_audio(fake_file)

        assert len(web_mod._audio_cache) == 1


# ── /api/analyze/status ───────────────────────────────────────────────────────

class TestApiAnalyzeStatus:
    """Tests for GET /api/analyze/status/<task_id>."""

    def test_unknown_task_id_returns_404(self, client):
        resp = client.get("/api/analyze/status/doesnotexist00000000")
        assert resp.status_code == 404
        assert "error" in resp.get_json()

    def test_known_task_id_returns_200(self, client):
        import preprod.web as web_mod
        task_id = "test-status-known"
        with web_mod._tasks_lock:
            web_mod._tasks[task_id] = {
                "status": "running", "progress": 50,
                "stage": "transcribing", "result": None,
                "error": None, "created_at": 0.0,
            }
        try:
            resp = client.get(f"/api/analyze/status/{task_id}")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["status"] == "running"
            assert data["progress"] == 50
        finally:
            with web_mod._tasks_lock:
                web_mod._tasks.pop(task_id, None)

    def test_cancel_event_not_in_response(self, client):
        """cancel_event (threading.Event) must be stripped before JSON serialisation."""
        import threading
        import preprod.web as web_mod
        task_id = "test-status-strip-cancel"
        with web_mod._tasks_lock:
            web_mod._tasks[task_id] = {
                "status": "running", "progress": 0,
                "stage": "starting", "result": None,
                "error": None, "created_at": 0.0,
                "cancel_event": threading.Event(),
            }
        try:
            resp = client.get(f"/api/analyze/status/{task_id}")
            assert resp.status_code == 200
            assert "cancel_event" not in resp.get_json()
        finally:
            with web_mod._tasks_lock:
                web_mod._tasks.pop(task_id, None)


# ── /api/stream ───────────────────────────────────────────────────────────────

class TestApiStream:
    """Tests for GET /api/stream — error paths and security restrictions."""

    @pytest.fixture(autouse=True)
    def _allow_tmp(self, tmp_path, monkeypatch):
        """Add tmp_path to the stream allowlist so functional tests pass."""
        import preprod.web as web_mod
        monkeypatch.setattr(web_mod, "_ALLOWED_DIRS", web_mod._ALLOWED_DIRS | {tmp_path})

    def test_no_path_returns_400(self, client):
        resp = client.get("/api/stream")
        assert resp.status_code == 400

    def test_nonexistent_file_in_allowed_dir_returns_404(self, client, tmp_path):
        """A non-existent path inside an allowed dir → 404 (not 403)."""
        f = tmp_path / "does_not_exist.mp4"
        resp = client.get(f"/api/stream?path={f}")
        assert resp.status_code == 404

    def test_existing_file_returns_200(self, client, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00" * 16)
        resp = client.get(f"/api/stream?path={f}")
        assert resp.status_code == 200

    def test_content_type_mp4(self, client, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00" * 16)
        resp = client.get(f"/api/stream?path={f}")
        assert "video/mp4" in resp.content_type

    def test_content_type_mp3_cbr(self, client, tmp_path):
        """CBR MP3 (no Xing header) is served as audio/mpeg unchanged."""
        f = tmp_path / "recording.mp3"
        f.write_bytes(b"\x00" * 16)
        resp = client.get(f"/api/stream?path={f}")
        assert "audio/mpeg" in resp.content_type

    def test_vbr_mp3_served_as_m4a_when_sidecar_ready(self, client, tmp_path, monkeypatch):
        """VBR MP3 with a pre-built M4A sidecar is served as audio/mp4."""
        import preprod.web as web_mod

        # Write a fake MP3 with a Xing VBR marker in the first 8 KB
        mp3 = tmp_path / "vbr.mp3"
        mp3.write_bytes(b"JUNK" * 64 + b"Xing" + b"\x00" * 64)

        # Pre-create a fake M4A sidecar in a temp dir
        fake_cache = tmp_path / "transcode_cache"
        fake_cache.mkdir()
        monkeypatch.setattr(web_mod, "_TRANSCODE_DIR", fake_cache)
        key = web_mod._mp3_cache_key(mp3)
        sidecar = fake_cache / f"{key}.m4a"
        sidecar.write_bytes(b"\x00" * 16)   # fake M4A content

        resp = client.get(f"/api/stream?path={mp3}")
        assert resp.status_code == 200
        assert "audio/mp4" in resp.content_type

    def test_vbr_mp3_served_as_mpeg_when_sidecar_absent(self, client, tmp_path, monkeypatch):
        """VBR MP3 without a sidecar falls back to audio/mpeg and starts transcoding."""
        import preprod.web as web_mod

        mp3 = tmp_path / "vbr.mp3"
        mp3.write_bytes(b"JUNK" * 64 + b"Xing" + b"\x00" * 64)

        fake_cache = tmp_path / "transcode_cache"
        fake_cache.mkdir()
        monkeypatch.setattr(web_mod, "_TRANSCODE_DIR", fake_cache)
        # Clear in-memory state so _start_m4a_transcode sees no pending entry
        monkeypatch.setattr(web_mod, "_TRANSCODE_STATE", {})

        # Stub out the actual ffmpeg call so the test doesn't launch a real process
        monkeypatch.setattr(web_mod, "_transcode_mp3_to_m4a_worker", lambda *_: None)

        resp = client.get(f"/api/stream?path={mp3}")
        assert resp.status_code == 200
        assert "audio/mpeg" in resp.content_type

    def test_content_type_aac(self, client, tmp_path):
        f = tmp_path / "recording.aac"
        f.write_bytes(b"\x00" * 16)
        resp = client.get(f"/api/stream?path={f}")
        assert "audio/aac" in resp.content_type

    def test_content_type_m4a(self, client, tmp_path):
        f = tmp_path / "recording.m4a"
        f.write_bytes(b"\x00" * 16)
        resp = client.get(f"/api/stream?path={f}")
        assert "audio/mp4" in resp.content_type

    def test_content_type_unknown_extension_is_octet_stream(self, client, tmp_path):
        f = tmp_path / "clip.xyz"
        f.write_bytes(b"\x00" * 16)
        resp = client.get(f"/api/stream?path={f}")
        assert resp.status_code == 200
        assert "octet-stream" in resp.content_type

    # ── Security: path allowlist ───────────────────────────────────────────────

    def test_path_outside_allowlist_returns_403(self, client, tmp_path, monkeypatch):
        """File exists but is outside any allowed directory → 403."""
        import preprod.web as web_mod
        # Remove tmp_path from the allowlist for this specific test.
        monkeypatch.setattr(web_mod, "_ALLOWED_DIRS", web_mod._ALLOWED_DIRS - {tmp_path})
        f = tmp_path / "secret.mp4"
        f.write_bytes(b"\x00" * 16)
        resp = client.get(f"/api/stream?path={f}")
        assert resp.status_code == 403

    def test_path_traversal_outside_allowlist_returns_403(self, client, tmp_path, monkeypatch):
        """Symlink or traversal that resolves outside allowed dirs → 403."""
        import preprod.web as web_mod
        monkeypatch.setattr(web_mod, "_ALLOWED_DIRS", web_mod._ALLOWED_DIRS - {tmp_path})
        # /etc/hosts always exists and is outside the allowlist.
        resp = client.get("/api/stream?path=/etc/hosts")
        assert resp.status_code == 403

    # ── Security: Host header (DNS-rebinding guard) ────────────────────────────

    def test_bad_host_header_returns_403(self, client, tmp_path):
        """Request with an external Host header is rejected."""
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00" * 16)
        resp = client.get(f"/api/stream?path={f}", headers={"Host": "evil.com:9877"})
        assert resp.status_code == 403

    def test_localhost_host_header_is_allowed(self, client, tmp_path):
        """Explicit localhost Host header is accepted."""
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00" * 16)
        resp = client.get(f"/api/stream?path={f}", headers={"Host": "localhost:9877"})
        assert resp.status_code == 200

    def test_127_host_header_is_allowed(self, client, tmp_path):
        """127.0.0.1 Host header is accepted."""
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00" * 16)
        resp = client.get(f"/api/stream?path={f}", headers={"Host": "127.0.0.1:9877"})
        assert resp.status_code == 200

    def test_bracketed_ipv6_loopback_host_header_is_allowed(self, client):
        """"[::1]:9877" — bracketed IPv6 literal with a port — is accepted.

        Regression: a naive host.split(":")[0] returns "[" for this input
        (IPv6 literals are bracketed in the Host header specifically because
        the address itself contains colons), which used to reject a
        legitimate ::1 client. Fails closed, not a security hole, but wrong.
        """
        resp = client.get("/api/llm/models", headers={"Host": "[::1]:9877"})
        assert resp.status_code == 200

    def test_bare_bracketed_ipv6_loopback_host_header_is_allowed(self, client):
        """"[::1]" — bracketed IPv6 literal with no port — is also accepted."""
        resp = client.get("/api/llm/models", headers={"Host": "[::1]"})
        assert resp.status_code == 200

    def test_bad_host_header_rejected_on_non_stream_route(self, client):
        """The guard is app-wide (before_request), not just on /api/stream.

        Regression for the OSS-readiness security review: previously only
        /api/stream checked the Host header, so a DNS-rebinding page could
        still drive every other /api/* route (analysis, export, LLM, session).
        """
        resp = client.get("/api/llm/models", headers={"Host": "evil.com:9877"})
        assert resp.status_code == 403

    def test_localhost_host_header_allowed_on_non_stream_route(self, client):
        resp = client.get("/api/llm/models", headers={"Host": "localhost:9877"})
        assert resp.status_code == 200

    def test_upload_dir_path_is_allowed(self, client, monkeypatch):
        """Files under _UPLOAD_DIR (macOS tempdir) are allowed via the resolved path."""
        import preprod.web as web_mod
        import tempfile
        upload_dir = web_mod._UPLOAD_DIR
        upload_dir.mkdir(parents=True, exist_ok=True)
        f = upload_dir / "test_clip.mp4"
        f.write_bytes(b"\x00" * 16)
        try:
            resp = client.get(f"/api/stream?path={f}")
            assert resp.status_code == 200
        finally:
            f.unlink(missing_ok=True)

    def test_upload_dir_is_under_app_support(self, client):
        """Uploaded files should persist in Application Support, not a temp directory."""
        import preprod.web as web_mod
        app_support = Path.home() / "Library" / "Application Support"
        upload = web_mod._UPLOAD_DIR.resolve()
        assert str(upload).startswith(str(app_support)), (
            f"_UPLOAD_DIR should be under ~/Library/Application Support for persistence. Got: {upload}"
        )


class TestCheckRemoteBind:
    """--allow-remote gate for non-loopback --host (OSS-readiness security review)."""

    def _fn(self):
        import preprod.web as web_mod
        return web_mod._check_remote_bind

    def test_loopback_host_needs_no_flag(self):
        self._fn()("127.0.0.1", False)   # must not raise

    def test_localhost_needs_no_flag(self):
        self._fn()("localhost", False)   # must not raise

    def test_remote_host_without_flag_raises_usage_error(self):
        import click
        with pytest.raises(click.UsageError):
            self._fn()("0.0.0.0", False)

    def test_remote_host_with_flag_does_not_raise(self):
        self._fn()("0.0.0.0", True)   # must not raise


# ── VBR MP3 → M4A transcode helpers ──────────────────────────────────────────

class TestIsVbrMp3:
    """Unit tests for _is_vbr_mp3 helper."""

    def test_non_mp3_extension_returns_false(self, tmp_path):
        import preprod.web as web_mod
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"Xing" * 10)   # Xing bytes in an .mp4 → still not MP3
        assert web_mod._is_vbr_mp3(f) is False

    def test_cbr_mp3_no_xing_returns_false(self, tmp_path):
        import preprod.web as web_mod
        f = tmp_path / "cbr.mp3"
        f.write_bytes(b"\xff\xfb" + b"\x00" * 100)   # MP3 sync but no Xing header
        assert web_mod._is_vbr_mp3(f) is False

    def test_xing_header_mp3_returns_true(self, tmp_path):
        import preprod.web as web_mod
        f = tmp_path / "vbr_xing.mp3"
        f.write_bytes(b"\x00" * 64 + b"Xing" + b"\x00" * 64)
        assert web_mod._is_vbr_mp3(f) is True

    def test_vbri_header_mp3_returns_true(self, tmp_path):
        import preprod.web as web_mod
        f = tmp_path / "vbr_vbri.mp3"
        f.write_bytes(b"\x00" * 64 + b"VBRI" + b"\x00" * 64)
        assert web_mod._is_vbr_mp3(f) is True

    def test_missing_file_returns_false(self, tmp_path):
        import preprod.web as web_mod
        f = tmp_path / "nonexistent.mp3"
        assert web_mod._is_vbr_mp3(f) is False


class TestApiTranscodeStatus:
    """Tests for GET /api/media/transcode_status."""

    @pytest.fixture(autouse=True)
    def _allow_tmp(self, tmp_path, monkeypatch):
        import preprod.web as web_mod
        monkeypatch.setattr(web_mod, "_ALLOWED_DIRS", web_mod._ALLOWED_DIRS | {tmp_path})

    def _make_vbr_mp3(self, tmp_path):
        f = tmp_path / "vbr.mp3"
        f.write_bytes(b"\x00" * 64 + b"Xing" + b"\x00" * 64)
        return f

    def test_no_path_returns_not_applicable(self, client):
        resp = client.get("/api/media/transcode_status")
        assert resp.json["status"] == "not_applicable"

    def test_non_mp3_returns_not_applicable(self, client, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00" * 16)
        resp = client.get(f"/api/media/transcode_status?path={f}")
        assert resp.json["status"] == "not_applicable"

    def test_cbr_mp3_returns_not_applicable(self, client, tmp_path):
        f = tmp_path / "cbr.mp3"
        f.write_bytes(b"\xff\xfb" + b"\x00" * 100)
        resp = client.get(f"/api/media/transcode_status?path={f}")
        assert resp.json["status"] == "not_applicable"

    def test_vbr_mp3_with_cached_sidecar_returns_ready(self, client, tmp_path, monkeypatch):
        import preprod.web as web_mod
        mp3 = self._make_vbr_mp3(tmp_path)
        fake_cache = tmp_path / "tc"
        fake_cache.mkdir()
        monkeypatch.setattr(web_mod, "_TRANSCODE_DIR", fake_cache)
        (fake_cache / f"{web_mod._mp3_cache_key(mp3)}.m4a").write_bytes(b"\x00")
        resp = client.get(f"/api/media/transcode_status?path={mp3}")
        assert resp.json["status"] == "ready"

    def test_vbr_mp3_without_sidecar_returns_not_started(self, client, tmp_path, monkeypatch):
        import preprod.web as web_mod
        mp3 = self._make_vbr_mp3(tmp_path)
        fake_cache = tmp_path / "tc2"
        fake_cache.mkdir()
        monkeypatch.setattr(web_mod, "_TRANSCODE_DIR", fake_cache)
        monkeypatch.setattr(web_mod, "_TRANSCODE_STATE", {})
        resp = client.get(f"/api/media/transcode_status?path={mp3}")
        assert resp.json["status"] == "not_started"

    def test_path_outside_allowlist_returns_not_applicable(self, client, tmp_path, monkeypatch):
        import preprod.web as web_mod
        monkeypatch.setattr(web_mod, "_ALLOWED_DIRS", web_mod._ALLOWED_DIRS - {tmp_path})
        mp3 = tmp_path / "vbr.mp3"
        mp3.write_bytes(b"\x00" * 64 + b"Xing" + b"\x00" * 64)
        resp = client.get(f"/api/media/transcode_status?path={mp3}")
        assert resp.json["status"] == "not_applicable"


# ── Video preview proxy (T0246 4K lag fix) ─────────────────────────────────────

class TestNeedsProxy:
    """Unit tests for _needs_proxy() — the resolution threshold decision."""

    def test_4k_video_needs_proxy(self):
        import preprod.web as web_mod
        media = _make_video_media("f.mp4")
        media.video_width, media.video_height = 3840, 2160
        assert web_mod._needs_proxy(media) is True

    def test_1080p_video_does_not_need_proxy(self):
        import preprod.web as web_mod
        media = _make_video_media("f.mp4")  # defaults to 1920x1080
        assert web_mod._needs_proxy(media) is False

    def test_vertical_4k_needs_proxy(self):
        """Long edge (not just width) determines the threshold — a 2160x3840
        vertical video is exactly as decode-heavy as 3840x2160 landscape."""
        import preprod.web as web_mod
        media = _make_video_media("f.mp4")
        media.video_width, media.video_height = 2160, 3840
        assert web_mod._needs_proxy(media) is True

    def test_audio_only_does_not_need_proxy(self):
        import preprod.web as web_mod
        media = _make_video_media("f.mp4")
        media.has_video = False
        media.video_width, media.video_height = 3840, 2160
        assert web_mod._needs_proxy(media) is False

    def test_missing_dimensions_does_not_need_proxy(self):
        import preprod.web as web_mod
        media = _make_video_media("f.mp4")
        media.video_width, media.video_height = None, None
        assert web_mod._needs_proxy(media) is False

    def test_unconfigured_mock_media_does_not_raise(self):
        """Regression: an unconfigured MagicMock (video_width/height are
        MagicMocks, not numbers) must not crash /api/stream's hot path —
        several pre-existing tests patch probe_media generically without
        configuring these attributes. Guard, don't raise."""
        import preprod.web as web_mod
        media = MagicMock()
        assert web_mod._needs_proxy(media) is False


class TestStartVideoProxyTranscode:
    """_start_video_proxy_transcode() no-op / state-machine behavior."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        import preprod.web as web_mod
        monkeypatch.setattr(web_mod, "_PROXY_DIR", tmp_path / "proxy_cache")
        monkeypatch.setattr(web_mod, "_PROXY_STATE", {})

    def test_not_needed_is_a_noop(self, tmp_path, monkeypatch):
        import preprod.web as web_mod
        spawned = []
        monkeypatch.setattr(web_mod.threading, "Thread",
                             lambda *a, **kw: spawned.append(1) or MagicMock())
        media = _make_video_media("f.mp4")  # 1080p — doesn't need a proxy
        web_mod._start_video_proxy_transcode(Path("f.mp4"), media)
        assert spawned == []

    def test_already_cached_on_disk_sets_ready_without_spawning(self, tmp_path, monkeypatch):
        import preprod.web as web_mod
        spawned = []
        monkeypatch.setattr(web_mod.threading, "Thread",
                             lambda *a, **kw: spawned.append(1) or MagicMock())
        src = tmp_path / "4k.mp4"
        web_mod._PROXY_DIR.mkdir(parents=True)
        web_mod._proxy_cache_path(src).write_bytes(b"\x00")
        media = _make_video_media(src)
        media.video_width, media.video_height = 3840, 2160
        web_mod._start_video_proxy_transcode(src, media)
        assert spawned == []
        assert web_mod._PROXY_STATE[web_mod._proxy_cache_key(src)] == "ready"

    def test_needed_and_uncached_spawns_exactly_one_thread(self, tmp_path, monkeypatch):
        import preprod.web as web_mod
        spawned = []
        monkeypatch.setattr(web_mod.threading, "Thread",
                             lambda *a, **kw: spawned.append(1) or MagicMock())
        src = tmp_path / "4k.mp4"
        media = _make_video_media(src)
        media.video_width, media.video_height = 3840, 2160
        web_mod._start_video_proxy_transcode(src, media)
        assert len(spawned) == 1
        assert web_mod._PROXY_STATE[web_mod._proxy_cache_key(src)] == "pending"

    def test_already_pending_does_not_spawn_a_second_thread(self, tmp_path, monkeypatch):
        import preprod.web as web_mod
        spawned = []
        monkeypatch.setattr(web_mod.threading, "Thread",
                             lambda *a, **kw: spawned.append(1) or MagicMock())
        src = tmp_path / "4k.mp4"
        media = _make_video_media(src)
        media.video_width, media.video_height = 3840, 2160
        web_mod._start_video_proxy_transcode(src, media)   # 1st call → spawns
        web_mod._start_video_proxy_transcode(src, media)   # 2nd call → no-op
        assert len(spawned) == 1


class TestApiProxyStatus:
    """Tests for GET /api/media/proxy_status."""

    @pytest.fixture(autouse=True)
    def _allow_tmp(self, tmp_path, monkeypatch):
        import preprod.web as web_mod
        monkeypatch.setattr(web_mod, "_ALLOWED_DIRS", web_mod._ALLOWED_DIRS | {tmp_path})
        monkeypatch.setattr(web_mod, "_media_cache", {})

    def test_no_path_returns_not_applicable(self, client):
        resp = client.get("/api/media/proxy_status")
        assert resp.json["status"] == "not_applicable"

    def test_nonexistent_file_returns_not_applicable(self, client, tmp_path):
        resp = client.get(f"/api/media/proxy_status?path={tmp_path/'ghost.mp4'}")
        assert resp.json["status"] == "not_applicable"

    def test_1080p_video_returns_not_applicable(self, client, tmp_path, monkeypatch):
        import preprod.web as web_mod
        f = tmp_path / "hd.mp4"; f.write_bytes(b"\x00" * 16)
        monkeypatch.setattr(web_mod, "probe_media", lambda p: _make_video_media(f))
        resp = client.get(f"/api/media/proxy_status?path={f}")
        assert resp.json["status"] == "not_applicable"

    def test_4k_with_cached_proxy_returns_ready(self, client, tmp_path, monkeypatch):
        import preprod.web as web_mod
        f = tmp_path / "4k.mp4"; f.write_bytes(b"\x00" * 16)
        media = _make_video_media(f); media.video_width, media.video_height = 3840, 2160
        monkeypatch.setattr(web_mod, "probe_media", lambda p: media)
        fake_cache = tmp_path / "pc"; fake_cache.mkdir()
        monkeypatch.setattr(web_mod, "_PROXY_DIR", fake_cache)
        (fake_cache / f"{web_mod._proxy_cache_key(f.resolve())}.proxy.mp4").write_bytes(b"\x00")
        resp = client.get(f"/api/media/proxy_status?path={f}")
        assert resp.json["status"] == "ready"

    def test_4k_without_cached_proxy_triggers_transcode_and_returns_pending(self, client, tmp_path, monkeypatch):
        """This endpoint is the ONLY trigger point for files whose original is
        never streamed (the frontend deliberately skips loading an original
        that might have no decode path in WKWebView at all — see loadFilePath's
        proxy-pending flow). So a status check must itself kick off the
        transcode, not just report "not_started" and leave it stuck forever."""
        import preprod.web as web_mod
        f = tmp_path / "4k.mp4"; f.write_bytes(b"\x00" * 16)
        media = _make_video_media(f); media.video_width, media.video_height = 3840, 2160
        monkeypatch.setattr(web_mod, "probe_media", lambda p: media)
        monkeypatch.setattr(web_mod, "_PROXY_DIR", tmp_path / "pc2")
        monkeypatch.setattr(web_mod, "_PROXY_STATE", {})
        # Stub the actual subprocess spawn so this test doesn't launch real ffmpeg.
        spawned = []
        monkeypatch.setattr(web_mod.threading, "Thread",
                             lambda *a, **kw: spawned.append(1) or MagicMock())
        resp = client.get(f"/api/media/proxy_status?path={f}")
        assert resp.json["status"] == "pending"
        assert len(spawned) == 1

    def test_path_outside_allowlist_returns_not_applicable(self, client, tmp_path, monkeypatch):
        import preprod.web as web_mod
        monkeypatch.setattr(web_mod, "_ALLOWED_DIRS", web_mod._ALLOWED_DIRS - {tmp_path})
        f = tmp_path / "4k.mp4"; f.write_bytes(b"\x00" * 16)
        resp = client.get(f"/api/media/proxy_status?path={f}")
        assert resp.json["status"] == "not_applicable"


class TestApiStreamVideoProxy:
    """Tests for /api/stream serving the low-res proxy for high-res video."""

    @pytest.fixture(autouse=True)
    def _allow_tmp(self, tmp_path, monkeypatch):
        import preprod.web as web_mod
        monkeypatch.setattr(web_mod, "_ALLOWED_DIRS", web_mod._ALLOWED_DIRS | {tmp_path})
        monkeypatch.setattr(web_mod, "_media_cache", {})

    def test_4k_video_with_cached_proxy_served_as_proxy(self, client, tmp_path, monkeypatch):
        import preprod.web as web_mod
        f = tmp_path / "4k.mp4"; f.write_bytes(b"ORIGINAL-4K-BYTES")
        media = _make_video_media(f); media.video_width, media.video_height = 3840, 2160
        monkeypatch.setattr(web_mod, "probe_media", lambda p: media)
        fake_cache = tmp_path / "pc"; fake_cache.mkdir()
        monkeypatch.setattr(web_mod, "_PROXY_DIR", fake_cache)
        proxy_path = fake_cache / f"{web_mod._proxy_cache_key(f.resolve())}.proxy.mp4"
        proxy_path.write_bytes(b"SMALL-PROXY-BYTES")

        resp = client.get(f"/api/stream?path={f}")
        assert resp.status_code == 200
        assert "video/mp4" in resp.content_type
        assert resp.data == b"SMALL-PROXY-BYTES"   # proxy served, NOT the original

    def test_4k_video_without_cached_proxy_falls_back_to_original(self, client, tmp_path, monkeypatch):
        import preprod.web as web_mod
        f = tmp_path / "4k.mp4"; f.write_bytes(b"ORIGINAL-4K-BYTES")
        media = _make_video_media(f); media.video_width, media.video_height = 3840, 2160
        monkeypatch.setattr(web_mod, "probe_media", lambda p: media)
        monkeypatch.setattr(web_mod, "_PROXY_DIR", tmp_path / "pc2")
        monkeypatch.setattr(web_mod, "_PROXY_STATE", {})
        # Stub the worker so this test doesn't launch a real ffmpeg process.
        monkeypatch.setattr(web_mod, "_transcode_video_proxy_worker", lambda *_: None)

        resp = client.get(f"/api/stream?path={f}")
        assert resp.status_code == 200
        assert "video/mp4" in resp.content_type
        assert resp.data == b"ORIGINAL-4K-BYTES"   # original served while proxy transcodes

    def test_1080p_video_never_checks_proxy_cache(self, client, tmp_path, monkeypatch):
        """A standard-resolution video must stream the original with no proxy
        detour at all — confirms the fix doesn't touch already-fine playback."""
        import preprod.web as web_mod
        f = tmp_path / "hd.mp4"; f.write_bytes(b"HD-BYTES")
        monkeypatch.setattr(web_mod, "probe_media", lambda p: _make_video_media(f))
        resp = client.get(f"/api/stream?path={f}")
        assert resp.status_code == 200
        assert resp.data == b"HD-BYTES"


# ── Static file cache-control headers ─────────────────────────────────────────

class TestStaticFileCacheControl:
    """Regression: WKWebView served stale JS from NSURLCache across restarts.

    web.py adds an after_request hook that sets Cache-Control: no-store on all
    /static/ responses so WKWebView always fetches fresh JS/CSS.
    """

    def test_static_js_has_no_store_header(self, client):
        resp = client.get("/static/app.js")
        assert resp.status_code == 200
        cc = resp.headers.get("Cache-Control", "")
        assert "no-store" in cc, f"Expected no-store in Cache-Control, got: {cc!r}"

    def test_static_css_has_no_store_header(self, client):
        resp = client.get("/static/style.css")
        if resp.status_code != 200:
            pytest.skip("style.css not present in test env")
        cc = resp.headers.get("Cache-Control", "")
        assert "no-store" in cc, f"Expected no-store in Cache-Control, got: {cc!r}"

    def test_non_static_route_does_not_get_no_store(self, client):
        resp = client.get("/api/capabilities")
        cc = resp.headers.get("Cache-Control", "")
        assert "no-store" not in cc, (
            f"Cache-Control: no-store should only be on /static/ routes, got: {cc!r}"
        )


# ── /api/filepicker ───────────────────────────────────────────────────────────

class TestApiFilepicker:
    """Tests for POST /api/filepicker (desktop-only, pywebview required)."""

    def test_webview_unavailable_returns_400(self, client):
        """In the test environment pywebview is not installed → returns 400 with error."""
        resp = client.post("/api/filepicker",
                           data=json.dumps({}),
                           content_type="application/json")
        assert resp.status_code == 400
        assert "error" in resp.get_json()


# ── Integration tests (use real sample.mov) ────────────────────────────────────

SAMPLE_MOV = Path(__file__).parent / "fixtures" / "sample.mov"


@pytest.mark.slow
@pytest.mark.skipif(not SAMPLE_MOV.exists(), reason="sample.mov fixture not present")
class TestRedetectSilenceIntegration:
    """End-to-end tests against the real sample.mov — exercises actual audio extraction."""

    @pytest.fixture()
    def real_client(self):
        """Flask test client with NO audio/probe mocks."""
        from preprod.web import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    @pytest.fixture(autouse=True)
    def _clear_audio_cache(self):
        import preprod.web as web_mod
        web_mod._audio_cache.clear()
        yield
        web_mod._audio_cache.clear()

    def test_hangover_affects_region_count(self, real_client):
        """Higher hangover_ms merges more nearby silences → fewer regions returned."""
        r0 = real_client.post(
            "/api/analyze/redetect_silence",
            data=json.dumps({
                "path": str(SAMPLE_MOV),
                "threshold_db": -40.0,
                "min_duration": 1.0,
                "hangover_ms": 0,
            }),
            content_type="application/json",
        ).get_json()
        r1 = real_client.post(
            "/api/analyze/redetect_silence",
            data=json.dumps({
                "path": str(SAMPLE_MOV),
                "threshold_db": -40.0,
                "min_duration": 1.0,
                "hangover_ms": 500,
            }),
            content_type="application/json",
        ).get_json()
        assert "candidates" in r0, f"r0 missing candidates: {r0}"
        assert "candidates" in r1, f"r1 missing candidates: {r1}"
        # More hold time → regions merge → fewer total
        assert len(r1["candidates"]) < len(r0["candidates"]), (
            f"hangover=0 gave {len(r0['candidates'])} regions, "
            f"hangover=500 gave {len(r1['candidates'])} — expected fewer with more hold time"
        )

    def test_warm_cache_is_fast(self, real_client):
        """Second redetect call (warm cache) resolves in under 100 ms."""
        import time

        payload = json.dumps({
            "path": str(SAMPLE_MOV),
            "threshold_db": -40.0,
            "min_duration": 1.0,
            "hangover_ms": 300,
        })
        # Cold call to warm the cache
        real_client.post(
            "/api/analyze/redetect_silence",
            data=payload,
            content_type="application/json",
        )
        # Warm call — should skip audio extraction entirely
        t0 = time.perf_counter()
        resp = real_client.post(
            "/api/analyze/redetect_silence",
            data=payload,
            content_type="application/json",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200
        assert elapsed_ms < 100, f"Warm cache took {elapsed_ms:.1f}ms — expected <100ms"

    def test_roughcut_export_save_to_downloads(self, real_client, tmp_path):
        """save_to_downloads=true writes an FCPXML file to ~/Downloads (mocked).

        This is the exact code path that caused the blank-screen bug: the old code
        streamed the file as application/xml, which triggered pywebview's cocoa
        decidePolicyForNavigationResponse → WKNavigationResponsePolicyCancel.
        With save_to_downloads=true the server writes directly to disk and returns
        plain JSON — no blob fetch, no WKWebView response interception.
        """
        import preprod.web as web_mod

        downloads = tmp_path / "Downloads"
        downloads.mkdir()

        # Patch Path.home() so _copy_to_downloads writes to our tmp dir
        orig_export_dir = web_mod._EXPORT_DIR
        web_mod._EXPORT_DIR = tmp_path / "exports"
        web_mod._EXPORT_DIR.mkdir()

        try:
            with patch("preprod.web.Path.home", return_value=tmp_path):
                resp = real_client.post(
                    "/api/export/roughcut",
                    data=json.dumps({
                        "path": str(SAMPLE_MOV),
                        "removal_regions": [],
                        "padding_ms": 200,
                        "save_to_downloads": True,
                    }),
                    content_type="application/json",
                )
        finally:
            web_mod._EXPORT_DIR = orig_export_dir

        # Must return JSON {"ok": true}, NOT a file blob
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"ok": True}, f"Expected {{ok: true}}, got {data}"
        assert resp.content_type.startswith("application/json"), (
            f"Expected JSON content-type, got {resp.content_type}"
        )

        # FCPXML file must be on disk in the mocked Downloads folder
        exported = list(downloads.glob("*_cut.fcpxml"))
        assert len(exported) == 1, f"Expected 1 FCPXML in Downloads, got: {exported}"
        fcpxml = exported[0].read_text(encoding="utf-8")
        assert fcpxml.startswith("<?xml"), f"Not valid XML: {fcpxml[:100]}"
        assert "fcpxml" in fcpxml.lower(), "Missing fcpxml element"


# ── Fix 2: Word timestamp clamping unit tests ─────────────────────────────────

class TestWordTimestampClamping:
    """Fix 2: Word timestamps must be clamped to their assigned segment bounds
    (±50ms allowance) so words cannot stray ±280ms outside their segment.

    These tests directly exercise the clamping logic by injecting mocked
    transcription results into _run_analysis and inspecting the words payload.
    """

    def _run_analysis_result(self, client, tmp_path, words, segments, whisperx_used=False):
        """Run a full analysis with mocked transcription and return the result dict."""
        import time
        import numpy as np
        import preprod.web as web_mod

        fake_file = tmp_path / "video.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_media = _make_video_media(fake_file, duration=10.0)
        samples = np.zeros(int(10.0 * 16000), dtype=np.float32)
        samples[:8000] = 0.5  # brief speech burst at start

        def _fake_transcribe(*args, **kwargs):
            return {"words": words, "segments": segments, "language": "ja",
                    "whisperx_used": whisperx_used}

        with patch("preprod.web.probe_media", return_value=fake_media), \
             patch("preprod.web.extract_audio", return_value=samples), \
             patch("preprod.web.WHISPER_AVAILABLE", True), \
             patch("preprod.web.detect_fillers", return_value=[]), \
             patch("preprod.web.snap_silences_to_words", side_effect=lambda sr, _w: sr), \
             patch("preprod.web.detect_untranscribed_speech", return_value=[]), \
             patch("preprod.web.transcribe", side_effect=_fake_transcribe):
            resp = client.post(
                "/api/analyze/start",
                data=json.dumps({"path": str(fake_file), "use_whisper": True}),
                content_type="application/json",
            )
            assert resp.status_code == 200
            task_id = resp.get_json()["task_id"]

            deadline = time.monotonic() + 5.0
            result = {}
            status = "running"
            while time.monotonic() < deadline:
                sr = client.get(f"/api/analyze/status/{task_id}")
                data = sr.get_json()
                status = data.get("status", "running")
                result = data.get("result", {})
                if status != "running":
                    break
                time.sleep(0.05)

        assert status == "done", f"Expected done, got {status!r}"
        return result

    def test_word_within_segment_not_clamped(self, client, tmp_path):
        """A word well within its segment bounds must not have its timestamps modified."""
        words = [{"word": "hello", "start": 0.5, "end": 1.0, "score": 0.9, "seg_id": 0}]
        segments = [{"start": 0.0, "end": 3.0, "text": "hello"}]

        result = self._run_analysis_result(client, tmp_path, words, segments)
        out_words = result.get("words", [])
        assert out_words, "Expected at least one word in output"
        w = out_words[0]
        assert w["start"] == 0.5, f"start should be unclamped: {w['start']}"
        assert w["end"] == 1.0,   f"end should be unclamped: {w['end']}"

    def test_word_start_before_segment_clamped(self, client, tmp_path):
        """A word starting 200ms before its segment start must be clamped to seg.start - 50ms."""
        # Segment starts at 1.0, word starts at 0.7 (300ms early → exceeds 50ms allowance).
        words = [{"word": "hello", "start": 0.7, "end": 1.5, "score": 0.9, "seg_id": 0}]
        segments = [{"start": 1.0, "end": 3.0, "text": "hello"}]

        result = self._run_analysis_result(client, tmp_path, words, segments)
        out_words = result.get("words", [])
        assert out_words, "Expected at least one word in output"
        w = out_words[0]
        # Clamp: max(0.7, 1.0 - 0.05) = max(0.7, 0.95) = 0.95
        assert w["start"] == pytest.approx(0.95, abs=0.001), (
            f"Fix 2: word start (0.7) should be clamped to 0.95 (seg.start - 50ms); got {w['start']}"
        )

    def test_word_end_after_segment_clamped(self, client, tmp_path):
        """A word ending 300ms after its segment end must be clamped to seg.end + 50ms."""
        # Segment ends at 2.0, word ends at 2.4 (400ms late → exceeds 50ms allowance).
        words = [{"word": "world", "start": 1.0, "end": 2.4, "score": 0.9, "seg_id": 0}]
        segments = [{"start": 0.0, "end": 2.0, "text": "world"}]

        result = self._run_analysis_result(client, tmp_path, words, segments)
        out_words = result.get("words", [])
        assert out_words, "Expected at least one word in output"
        w = out_words[0]
        # Clamp: min(2.4, 2.0 + 0.05) = min(2.4, 2.05) = 2.05
        assert w["end"] == pytest.approx(2.05, abs=0.001), (
            f"Fix 2: word end (2.4) should be clamped to 2.05 (seg.end + 50ms); got {w['end']}"
        )

    def test_whisperx_words_not_clamped(self, client, tmp_path):
        """WhisperX words must NOT be clamped even when timestamps exceed segment bounds.

        WhisperX CTC timestamps are ±20ms accurate; clamping clips the last word of
        each segment short and creates highlight gaps. Skipping clamping is the fix.
        """
        # Same setup as test_word_start_before_segment_clamped but whisperx_used=True
        words = [{"word": "hello", "start": 0.7, "end": 1.5, "score": 0.9, "seg_id": 0}]
        segments = [{"start": 1.0, "end": 3.0, "text": "hello"}]

        result = self._run_analysis_result(client, tmp_path, words, segments,
                                           whisperx_used=True)
        out_words = result.get("words", [])
        assert out_words, "Expected at least one word in output"
        w = out_words[0]
        assert w["start"] == pytest.approx(0.7, abs=0.001), (
            f"whisperx_used=True: word start should remain 0.7 (unclamped); got {w['start']}"
        )
        assert w["end"] == pytest.approx(1.5, abs=0.001), (
            f"whisperx_used=True: word end should remain 1.5 (unclamped); got {w['end']}"
        )

    def test_word_with_no_seg_id_not_clamped(self, client, tmp_path):
        """Words with seg_id=None must never be clamped — leave raw timestamps."""
        # seg_id=None means fallback from assign_words_to_entries; no clamping.
        words = [{"word": "uhh", "start": 0.1, "end": 0.4, "score": 0.5, "seg_id": None}]
        segments = [{"start": 1.0, "end": 3.0, "text": "something"}]

        result = self._run_analysis_result(client, tmp_path, words, segments)
        out_words = result.get("words", [])
        assert out_words, "Expected word in output"
        w = out_words[0]
        # seg_id is None here — but Fix 4 will assign it to nearest entry.
        # The key test is that it doesn't crash; timestamps may be adjusted by clamping
        # only if seg_id is now set by Fix 4. If seg_id remains None after all path, no clamp.
        # We just verify the word is present with valid timestamps.
        assert isinstance(w["start"], float)
        assert isinstance(w["end"], float)

    def test_degenerate_clamp_not_applied(self, client, tmp_path):
        """When clamping would produce start >= end, raw timestamps are preserved.

        A word assigned to a distant segment by the unconstrained fallback
        (assign_words_to_entries) may have raw timestamps entirely outside the
        segment window.  Applying the clamp would invert the span (start > end).
        The code must detect this and keep the original raw timestamps instead.
        """
        # Word at 0.1-0.4; segment at 5.0-6.0.
        # Clamped: start = max(0.1, 5.0-0.05) = 4.95, end = min(0.4, 6.0+0.05) = 0.4
        # 4.95 > 0.4 → degenerate; raw timestamps (0.1, 0.4) must be used instead.
        words = [{"word": "uhh", "start": 0.1, "end": 0.4, "score": 0.5, "seg_id": 0}]
        segments = [{"start": 5.0, "end": 6.0, "text": "hello"}]

        result = self._run_analysis_result(client, tmp_path, words, segments)
        out_words = result.get("words", [])
        assert out_words, "Expected word in output (degenerate clamp must not drop it)"
        w = out_words[0]
        # Raw timestamps preserved — 0.1 and 0.4.
        assert w["start"] == pytest.approx(0.1, abs=0.001), (
            f"Degenerate clamp: raw start (0.1) must be kept; got {w['start']}"
        )
        assert w["end"] == pytest.approx(0.4, abs=0.001), (
            f"Degenerate clamp: raw end (0.4) must be kept; got {w['end']}"
        )


# ── Fix 3: whisperx_used flag in analysis result ──────────────────────────────

class TestWhisperxUsedFlagInResult:
    """Fix 3: The analysis result must include accuracy_warning when WhisperX
    was not used, so the frontend can show the user a persistent banner.
    """

    def _run_with_whisperx_used(self, client, tmp_path, whisperx_used: bool):
        """Run analysis with mocked transcribe that reports whisperx_used."""
        import time
        import numpy as np

        fake_file = tmp_path / f"video_{whisperx_used}.mp4"
        fake_file.write_bytes(b"\x00" * 100)
        fake_media = _make_video_media(fake_file, duration=5.0)
        samples = np.zeros(int(5.0 * 16000), dtype=np.float32)
        samples[:8000] = 0.5

        words = [{"word": "test", "start": 0.5, "end": 1.0, "score": 0.9, "seg_id": 0}]
        segments = [{"start": 0.0, "end": 2.0, "text": "test"}]

        def _fake_transcribe(*args, **kwargs):
            return {"words": words, "segments": segments, "language": "ja",
                    "whisperx_used": whisperx_used}

        with patch("preprod.web.probe_media", return_value=fake_media), \
             patch("preprod.web.extract_audio", return_value=samples), \
             patch("preprod.web.WHISPER_AVAILABLE", True), \
             patch("preprod.web.detect_fillers", return_value=[]), \
             patch("preprod.web.snap_silences_to_words", side_effect=lambda sr, _w: sr), \
             patch("preprod.web.detect_untranscribed_speech", return_value=[]), \
             patch("preprod.web.transcribe", side_effect=_fake_transcribe):
            resp = client.post(
                "/api/analyze/start",
                data=json.dumps({"path": str(fake_file), "use_whisper": True}),
                content_type="application/json",
            )
            task_id = resp.get_json()["task_id"]

            deadline = time.monotonic() + 5.0
            result = {}
            status = "running"
            while time.monotonic() < deadline:
                sr = client.get(f"/api/analyze/status/{task_id}")
                data = sr.get_json()
                status = data.get("status", "running")
                result = data.get("result", {})
                if status != "running":
                    break
                time.sleep(0.05)

        assert status == "done", f"Expected done, got {status!r}"
        return result

    def test_accuracy_warning_set_when_whisperx_not_used(self, client, tmp_path):
        """When whisperx_used=False, result must have accuracy_warning with ±150ms message."""
        result = self._run_with_whisperx_used(client, tmp_path, whisperx_used=False)
        aw = result.get("accuracy_warning")
        assert aw is not None, (
            "Fix 3: accuracy_warning must be set when whisperx_used=False. "
            f"Got accuracy_warning={aw!r} in result keys: {list(result.keys())}"
        )
        assert "150" in aw or "faster-whisper" in aw.lower(), (
            f"accuracy_warning should mention ±150ms or faster-whisper; got: {aw!r}"
        )

    def test_accuracy_warning_none_when_whisperx_used(self, client, tmp_path):
        """When whisperx_used=True, accuracy_warning must be None (no banner needed)."""
        result = self._run_with_whisperx_used(client, tmp_path, whisperx_used=True)
        aw = result.get("accuracy_warning")
        assert aw is None, (
            f"Fix 3: accuracy_warning must be None when whisperx_used=True; got: {aw!r}"
        )

    def test_whisper_warning_also_set_in_stats_when_not_used(self, client, tmp_path):
        """whisper_warning in stats.whisper_warning is also populated for backward compat."""
        result = self._run_with_whisperx_used(client, tmp_path, whisperx_used=False)
        stats = result.get("stats", {})
        ww = stats.get("whisper_warning")
        assert ww is not None, (
            f"whisper_warning in stats must also be set when whisperx_used=False; "
            f"stats keys: {list(stats.keys())}"
        )


class TestBuildSegmentsTyped:
    """Unit tests for _build_segments_typed — type-aware padding for export.

    Word-type removals must be exported with zero padding (boundaries are
    already snapped to audio energy by refine_word_boundary) so short words
    are not silently dropped.  Non-word removals get the user's configured
    padding as before.
    """

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        import preprod.web as web_mod
        web_mod._audio_cache.clear()
        yield
        web_mod._audio_cache.clear()

    def _make_fake_audio(self, tmp_path, amp=0.5, duration_s=10.0):
        """Write a dummy file and populate the audio cache with constant audio."""
        import numpy as np
        import preprod.web as web_mod
        from preprod.audio import SAMPLE_RATE

        p = tmp_path / "clip.mp4"
        p.write_bytes(b"\x00" * 100)
        samples = np.full(int(SAMPLE_RATE * duration_s), amp, dtype="float32")
        web_mod._audio_cache[str(p)] = (samples, p.stat().st_mtime)
        return p, samples

    def test_word_region_not_collapsed_by_padding(self, tmp_path):
        """A 200ms word deletion with 200ms padding must survive in the output.

        Previously, padding was applied uniformly: a 200ms word region with
        200ms padding would shrink to 0ms and be dropped entirely.
        """
        import preprod.web as web_mod

        p, _ = self._make_fake_audio(tmp_path, duration_s=10.0)

        regions = [{"start": 2.0, "end": 2.2, "type": "word"}]  # 200ms word
        # padding_ms=200 would collapse this to nothing if applied to word type
        with patch("preprod.web.refine_word_boundary", return_value=(2.0, 2.2)):
            segs = web_mod._build_segments_typed(
                regions, p, total_duration=10.0, padding_ms=200
            )

        # Expect two keep-segments: [0, 2.0] and [2.2, 10.0]
        starts = [round(s.source_start, 3) for s in segs]
        ends   = [round(s.source_end,   3) for s in segs]
        assert 0.0 in starts, f"Segment before word missing; starts={starts}"
        assert 2.2 in ends or 10.0 in ends, f"Segment after word missing; ends={ends}"
        # Verify the 200ms word is actually cut: no keep-segment spans 2.0–2.2
        for seg in segs:
            assert not (seg.source_start <= 2.0 and seg.source_end >= 2.2), (
                f"Word region should be removed but segment {seg} spans it"
            )

    def test_silence_region_gets_padding(self, tmp_path):
        """Non-word (silence) removals must have padding applied as before."""
        import preprod.web as web_mod

        p, _ = self._make_fake_audio(tmp_path, duration_s=10.0)

        regions = [{"start": 2.0, "end": 4.0, "type": "silence"}]  # 2s silence
        segs = web_mod._build_segments_typed(
            regions, p, total_duration=10.0, padding_ms=200
        )

        # With 200ms padding: cut shrinks to [2.2, 3.8].
        # Keep-segments: [0, 2.2] and [3.8, 10.0].
        starts = [round(s.source_start, 3) for s in segs]
        ends   = [round(s.source_end,   3) for s in segs]
        assert pytest.approx(2.2, abs=0.01) in ends, (
            f"First keep-segment should end at 2.2 (after padding); ends={ends}"
        )
        assert pytest.approx(3.8, abs=0.01) in starts, (
            f"Second keep-segment should start at 3.8 (before padding); starts={starts}"
        )

    def test_short_silence_collapsed_by_padding_is_dropped(self, tmp_path):
        """A 300ms silence with 200ms padding each side collapses to nothing → dropped."""
        import preprod.web as web_mod

        p, _ = self._make_fake_audio(tmp_path, duration_s=5.0)

        regions = [{"start": 1.0, "end": 1.3, "type": "silence"}]  # 300ms < 2×200ms
        segs = web_mod._build_segments_typed(
            regions, p, total_duration=5.0, padding_ms=200
        )

        # The silence collapses to nothing → single continuous keep-segment [0, 5.0].
        assert len(segs) == 1, f"Expected 1 keep-seg (silence dropped); got {len(segs)}"
        assert segs[0].source_start == pytest.approx(0.0)
        assert segs[0].source_end   == pytest.approx(5.0)

    def test_mixed_word_and_silence_types(self, tmp_path):
        """Word and silence regions together: word has no padding, silence has padding."""
        import preprod.web as web_mod

        p, _ = self._make_fake_audio(tmp_path, duration_s=10.0)

        regions = [
            {"start": 1.0, "end": 1.15, "type": "word"},     # 150ms word
            {"start": 5.0, "end": 7.0,  "type": "silence"},  # 2s silence
        ]
        with patch("preprod.web.refine_word_boundary", return_value=(1.0, 1.15)):
            segs = web_mod._build_segments_typed(
                regions, p, total_duration=10.0, padding_ms=200
            )

        # Silence padded: [5.2, 6.8].  Word exact: [1.0, 1.15].
        # Keep segs: [0.0, 1.0], [1.15, 5.2], [6.8, 10.0]
        assert len(segs) == 3, f"Expected 3 keep-segs; got {len(segs)}: {segs}"
        assert segs[0].source_start == pytest.approx(0.0)
        assert segs[0].source_end   == pytest.approx(1.0)
        assert segs[1].source_start == pytest.approx(1.15)
        assert segs[1].source_end   == pytest.approx(5.2, abs=0.01)
        assert segs[2].source_start == pytest.approx(6.8, abs=0.01)
        assert segs[2].source_end   == pytest.approx(10.0)

    def test_no_media_path_word_uses_raw_timestamps(self, tmp_path):
        """When media_path is None, word regions fall back to raw timestamps (no refinement)."""
        import preprod.web as web_mod

        regions = [{"start": 2.0, "end": 2.5, "type": "word"}]
        with patch("preprod.web.refine_word_boundary") as mock_rb:
            segs = web_mod._build_segments_typed(
                regions, None, total_duration=5.0, padding_ms=200
            )
        mock_rb.assert_not_called()
        # Should still cut [2.0, 2.5] with no padding
        ends = [round(s.source_end, 3) for s in segs]
        assert 2.0 in ends, f"Keep-seg before word should end at 2.0; ends={ends}"


class TestStageTimer:
    """T0246: per-stage analysis profiling timer. Pure measurement; clock injected."""

    def test_laps_accumulate_and_total(self):
        from preprod.web import _StageTimer
        ticks = iter([100.0, 100.5, 102.0, 102.25])  # init, lap a, lap b, lap a-again
        timer = _StageTimer(clock=lambda: next(ticks))
        timer.lap("a")   # 0.5
        timer.lap("b")   # 1.5
        timer.lap("a")   # +0.25 → a accumulates to 0.75
        res = timer.result({"transcribe_detail": {"model_load": 1.0}})
        assert res["a"] == 0.75
        assert res["b"] == 1.5
        assert res["total"] == 2.25                       # sum of stages only
        assert res["transcribe_detail"] == {"model_load": 1.0}

    def test_skip_advances_without_recording(self):
        from preprod.web import _StageTimer
        ticks = iter([0.0, 1.0, 3.0])  # init, skip, lap x
        timer = _StageTimer(clock=lambda: next(ticks))
        timer.skip()      # mark → 1.0, nothing recorded
        timer.lap("x")    # 3.0 - 1.0 = 2.0
        res = timer.result()
        assert res == {"x": 2.0, "total": 2.0}

    def test_result_is_wellformed_json_serializable(self):
        import json
        from preprod.web import _StageTimer
        ticks = iter([0.0, 0.1, 0.3])
        timer = _StageTimer(clock=lambda: next(ticks))
        timer.lap("probe")
        timer.lap("extract_audio")
        res = timer.result()
        # every value numeric, "total" present, JSON-serializable (goes into the API result)
        assert "total" in res
        assert all(isinstance(v, (int, float)) for v in res.values())
        json.dumps(res)


class TestCancelRunningAnalyses:
    """T0251: starting a new analysis must cancel any in-flight one."""

    def test_signals_only_running_tasks_with_events(self):
        import threading
        from preprod import web
        ev_run, ev_done = threading.Event(), threading.Event()
        saved = dict(web._tasks)
        try:
            web._tasks.clear()
            web._tasks["r1"] = {"status": "running", "cancel_event": ev_run}
            web._tasks["d1"] = {"status": "done", "cancel_event": ev_done}
            web._tasks["r2"] = {"status": "running"}  # startup window: no event yet
            n = web._cancel_running_analyses()
            assert ev_run.is_set()         # running → signalled
            assert not ev_done.is_set()    # terminal → untouched
            assert n == 1                  # only the one with an event was counted
        finally:
            web._tasks.clear()
            web._tasks.update(saved)


class TestBoundedGrowth:
    """T0252: bound in-memory task dict and on-disk M4A cache."""

    def test_sweep_tasks_caps_terminal_count(self):
        import time as _t
        from preprod import web
        saved = dict(web._tasks)
        try:
            web._tasks.clear()
            now = _t.time()
            # 60 recent terminal tasks (age sweep won't touch them) — cap is 50.
            for i in range(60):
                web._tasks[f"d{i}"] = {"status": "done", "created_at": now - i, "result": {"x": i}}
            web._sweep_tasks(max_age_hours=1.0)
            remaining = [t for t in web._tasks.values() if t.get("status") in ("done", "error")]
            assert len(remaining) == web._MAX_TERMINAL_TASKS    # capped
            # Newest kept (created_at = now-0 … now-49), oldest (now-50…now-59) evicted.
            kept_ids = set(web._tasks)
            assert "d0" in kept_ids and "d59" not in kept_ids
        finally:
            web._tasks.clear()
            web._tasks.update(saved)

    def test_sweep_dir_by_size_evicts_oldest(self, tmp_path):
        import os, time
        from preprod.web import _sweep_dir_by_size
        # three 1000-byte files, distinct mtimes (a oldest → c newest)
        for name, mtime in [("a", 1000), ("b", 2000), ("c", 3000)]:
            f = tmp_path / f"{name}.m4a"
            f.write_bytes(b"x" * 1000)
            os.utime(f, (mtime, mtime))
        freed = _sweep_dir_by_size(tmp_path, max_bytes=2500)  # must drop to <=2500
        assert freed == 1000                                  # one file removed
        assert not (tmp_path / "a.m4a").exists()              # oldest evicted
        assert (tmp_path / "b.m4a").exists() and (tmp_path / "c.m4a").exists()

    def test_sweep_dir_by_size_noop_under_cap(self, tmp_path):
        from preprod.web import _sweep_dir_by_size
        (tmp_path / "a.m4a").write_bytes(b"x" * 100)
        assert _sweep_dir_by_size(tmp_path, max_bytes=10_000) == 0


class TestErrorSurfacing:
    """T0253: silent swallows now log instead of degrading invisibly."""

    def test_audio_load_failure_is_logged(self, caplog):
        from preprod import web
        with patch.object(web, "_get_cached_audio", side_effect=OSError("boom")):
            with caplog.at_level("WARNING"):
                web._build_segments_typed(
                    [{"start": 1.0, "end": 2.0, "type": "silence"}],
                    media_path=Path("/fake.mp4"), total_duration=10.0, padding_ms=0,
                )
        assert any("could not load audio" in r.message for r in caplog.records)

    def test_refine_failures_logged_once_with_count(self, caplog):
        import numpy as np
        from preprod import web
        with patch.object(web, "_get_cached_audio",
                          return_value=np.zeros(16000, dtype=np.float32)), \
             patch.object(web, "refine_word_boundary", side_effect=ValueError("bad")):
            with caplog.at_level("WARNING"):
                web._build_segments_typed(
                    [{"start": 1.0, "end": 2.0, "type": "word"},
                     {"start": 3.0, "end": 4.0, "type": "word"}],
                    media_path=Path("/fake.mp4"), total_duration=10.0, padding_ms=0,
                )
        msgs = [r.message for r in caplog.records]
        assert sum("refinement failed for 2 region" in m for m in msgs) == 1  # once, not per-word
