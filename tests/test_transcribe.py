"""Tests for transcribe.py.

Regression tests:
  - Bug #3: whisper/torch must NOT be imported at module level (libomp SIGSEGV fix)
  - transcribe() raises RuntimeError when whisper absent
  - subprocess is called with correct JSON args
"""

import ast
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

TRANSCRIBE_PATH = Path(__file__).parent.parent / "src" / "preprod" / "transcribe.py"


# ── Bug #3 regression: no top-level whisper/torch import ──────────────────────

class TestNoTopLevelWhisperImport:
    """Verify transcribe.py does NOT import whisper or torch at module level.

    The fix for the libomp SIGSEGV was to use importlib.util.find_spec()
    instead of a direct import. A direct `import whisper` or `import torch`
    at module level would bring torch's libomp into the main process alongside
    numpy's OpenBLAS libomp, causing SIGSEGV on macOS Apple Silicon.
    """

    def test_whisper_not_imported_at_module_level(self):
        source = TRANSCRIBE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):  # only direct children = module level
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "whisper", (
                        "REGRESSION BUG #3: 'import whisper' found at module level in transcribe.py"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "whisper", (
                    "REGRESSION BUG #3: 'from whisper import ...' found at module level"
                )

    def test_torch_not_imported_at_module_level(self):
        source = TRANSCRIBE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "torch", (
                        "REGRESSION BUG #3: 'import torch' found at module level in transcribe.py"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "torch", (
                    "REGRESSION BUG #3: 'from torch import ...' found at module level"
                )

    def test_find_spec_used_for_whisper_detection(self):
        source = TRANSCRIBE_PATH.read_text(encoding="utf-8")
        assert "find_spec" in source, (
            "REGRESSION BUG #3: expected importlib.util.find_spec() for whisper detection"
        )

    def test_importlib_util_imported_not_whisper(self):
        """The availability check block must reference importlib.util, not whisper directly."""
        source = TRANSCRIBE_PATH.read_text(encoding="utf-8")
        assert "importlib.util" in source or "importlib" in source, (
            "importlib not used in transcribe.py — find_spec check is missing"
        )


# ── WHISPER_AVAILABLE detection ───────────────────────────────────────────────

class TestWhisperAvailable:
    def test_whisper_available_is_bool(self):
        from preprod.transcribe import WHISPER_AVAILABLE
        assert isinstance(WHISPER_AVAILABLE, bool)

    def test_whisper_available_false_when_find_spec_returns_none(self):
        """Reimport with find_spec mocked to return None → WHISPER_AVAILABLE must be False."""
        mod_name = "preprod.transcribe"
        saved = sys.modules.pop(mod_name, None)
        try:
            with patch("importlib.util.find_spec", return_value=None):
                import preprod.transcribe as t
                assert t.WHISPER_AVAILABLE is False
        finally:
            # Restore original module state
            sys.modules.pop(mod_name, None)
            if saved is not None:
                sys.modules[mod_name] = saved

    def test_whisper_available_true_when_find_spec_returns_spec(self):
        """Reimport with find_spec mocked to return a fake spec → WHISPER_AVAILABLE must be True."""
        fake_spec = MagicMock()
        mod_name = "preprod.transcribe"
        saved = sys.modules.pop(mod_name, None)
        try:
            with patch("importlib.util.find_spec", return_value=fake_spec):
                import preprod.transcribe as t
                assert t.WHISPER_AVAILABLE is True
        finally:
            sys.modules.pop(mod_name, None)
            if saved is not None:
                sys.modules[mod_name] = saved


# ── WHISPERX_AVAILABLE detection ────────────────────────────────────────────
# WhisperX is a separate, heavier optional dependency from faster-whisper/
# openai-whisper (see whisper_worker.py:_run_whisperx_alignment) — it only
# refines word timestamps and is never required for transcription itself.

class TestWhisperxAvailable:
    def test_whisperx_available_is_bool(self):
        from preprod.transcribe import WHISPERX_AVAILABLE
        assert isinstance(WHISPERX_AVAILABLE, bool)

    def test_whisperx_not_imported_at_module_level(self):
        """Mirrors the whisper/torch regression guard above: find_spec() must
        be used so importing transcribe.py never imports whisperx (and
        therefore never its own torch/pyannote deps) into the main process."""
        source = TRANSCRIBE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "whisperx", (
                        "'import whisperx' found at module level in transcribe.py — "
                        "use importlib.util.find_spec() instead"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "whisperx", (
                    "'from whisperx import ...' found at module level in transcribe.py"
                )

    def test_whisperx_available_false_when_find_spec_returns_none(self):
        """Reimport with find_spec mocked to return None for every module name
        (including whisper/faster_whisper) → WHISPERX_AVAILABLE must be False."""
        mod_name = "preprod.transcribe"
        saved = sys.modules.pop(mod_name, None)
        try:
            with patch("importlib.util.find_spec", return_value=None):
                import preprod.transcribe as t
                assert t.WHISPERX_AVAILABLE is False
        finally:
            sys.modules.pop(mod_name, None)
            if saved is not None:
                sys.modules[mod_name] = saved

    def test_whisperx_available_true_when_find_spec_returns_spec(self):
        """Reimport with find_spec mocked to return a fake spec for every
        module name → WHISPERX_AVAILABLE must be True."""
        fake_spec = MagicMock()
        mod_name = "preprod.transcribe"
        saved = sys.modules.pop(mod_name, None)
        try:
            with patch("importlib.util.find_spec", return_value=fake_spec):
                import preprod.transcribe as t
                assert t.WHISPERX_AVAILABLE is True
        finally:
            sys.modules.pop(mod_name, None)
            if saved is not None:
                sys.modules[mod_name] = saved

    def test_whisperx_available_independent_of_whisper_available(self):
        """The realistic case this feature exists for: faster-whisper IS
        installed but whisperx is NOT — the two flags must move independently,
        not track the same find_spec() call."""
        mod_name = "preprod.transcribe"
        saved = sys.modules.pop(mod_name, None)

        def _fake_find_spec(name, *a, **kw):
            if name == "whisperx":
                return None
            return MagicMock()  # whisper / faster_whisper both "installed"

        try:
            with patch("importlib.util.find_spec", side_effect=_fake_find_spec):
                import preprod.transcribe as t
                assert t.WHISPER_AVAILABLE is True
                assert t.WHISPERX_AVAILABLE is False
        finally:
            sys.modules.pop(mod_name, None)
            if saved is not None:
                sys.modules[mod_name] = saved


# ── transcribe() raises RuntimeError when whisper absent ─────────────────────

class TestTranscribeRaisesWhenWhisperAbsent:
    def test_runtime_error_when_whisper_not_available(self, tmp_path):
        """Patch WHISPER_AVAILABLE on the already-imported module."""
        from preprod import transcribe as t
        with patch.object(t, "WHISPER_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="openai-whisper"):
                t.transcribe(tmp_path / "fake.mp4")

    def test_error_message_mentions_pip_install(self, tmp_path):
        from preprod import transcribe as t
        with patch.object(t, "WHISPER_AVAILABLE", False):
            with pytest.raises(RuntimeError) as exc_info:
                t.transcribe(tmp_path / "fake.mp4")
            assert "pip install" in str(exc_info.value).lower() or \
                   "openai-whisper" in str(exc_info.value)


# ── transcribe() subprocess call ─────────────────────────────────────────────

def _make_fake_popen(returncode: int, stdout: str, stderr: str) -> MagicMock:
    """Build a Popen mock compatible with the streaming approach in transcribe().

    The code:
    - Drains proc.stdout in a background thread via read(65536) chunks until EOF
    - Iterates proc.stderr line-by-line in a background thread
    - Uses proc.wait(timeout=...) in the polling loop
    """
    fake_proc = MagicMock()
    fake_proc.returncode = returncode
    fake_proc.wait.return_value = returncode   # completes immediately on first poll
    # io.StringIO for both pipes: read(n) returns up to n bytes, then '' on EOF.
    # Using return_value=stdout would make every read() call return the full string
    # and the drain loop would spin forever.
    fake_proc.stdout = io.StringIO(stdout)
    fake_proc.stderr = io.StringIO(stderr)
    return fake_proc


class TestTranscribeSubprocessCall:
    """All tests patch WHISPER_AVAILABLE=True directly on the module object."""

    def _make_good_result(self):
        return {
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
            "words": [{"word": "Hello", "start": 0.0, "end": 0.5}],
            "language": "en",
        }

    def test_subprocess_called_with_json_args(self, tmp_path):
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"\x00" * 100)

        fake_proc = _make_fake_popen(0, json.dumps(self._make_good_result()), "")

        from preprod import transcribe as t

        with patch.object(t, "WHISPER_AVAILABLE", True), \
             patch.object(t.subprocess, "Popen", return_value=fake_proc) as mock_popen:
            result = t.transcribe(audio_file, model_size="base", language="en")

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]

        # Must invoke python + the worker script + a JSON arg
        assert str(t._WORKER) in cmd[1]

        # The JSON arg must be valid and contain the right keys
        args_json = json.loads(cmd[2])
        assert args_json["path"] == str(audio_file)
        assert args_json["model_size"] == "base"
        assert args_json["language"] == "en"

        # Result should pass through correctly
        assert result["language"] == "en"
        assert len(result["segments"]) == 1

    def test_subprocess_failure_raises_runtime_error(self, tmp_path):
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"\x00" * 100)

        fake_proc = _make_fake_popen(1, "", "worker crashed")

        from preprod import transcribe as t

        with patch.object(t, "WHISPER_AVAILABLE", True), \
             patch.object(t.subprocess, "Popen", return_value=fake_proc):
            with pytest.raises(RuntimeError, match="worker crashed"):
                t.transcribe(audio_file)

    def test_invalid_json_from_worker_raises_runtime_error(self, tmp_path):
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"\x00" * 100)

        fake_proc = _make_fake_popen(0, "this is not json", "")

        from preprod import transcribe as t

        with patch.object(t, "WHISPER_AVAILABLE", True), \
             patch.object(t.subprocess, "Popen", return_value=fake_proc):
            with pytest.raises(RuntimeError, match="invalid JSON"):
                t.transcribe(audio_file)

    def test_timeout_raises_runtime_error(self, tmp_path):
        """When proc.wait() always times out and deadline is exceeded, RuntimeError is raised."""
        import subprocess as sp
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"\x00" * 100)

        from preprod import transcribe as t

        # Popen that always raises TimeoutExpired on wait()
        fake_proc = MagicMock()
        fake_proc.wait.side_effect = sp.TimeoutExpired(cmd="x", timeout=1.0)
        fake_proc.kill.return_value = None
        # Give the drain threads proper EOF so they don't spin forever in background
        fake_proc.stdout = io.StringIO("")
        fake_proc.stderr = iter([])

        # Patch time.monotonic so deadline is immediately exceeded on first iteration
        import time
        call_count = [0]
        original_monotonic = time.monotonic

        def _fast_deadline():
            call_count[0] += 1
            # First call (deadline = now + 1200) returns a normal time;
            # subsequent calls return a value > deadline so the loop exits.
            if call_count[0] == 1:
                return 0.0   # "now" when deadline is set
            return 2000.0   # well past deadline

        with patch.object(t, "WHISPER_AVAILABLE", True), \
             patch.object(t.subprocess, "Popen", return_value=fake_proc), \
             patch.object(t.time, "monotonic", side_effect=_fast_deadline):
            with pytest.raises(RuntimeError, match="timed out"):
                t.transcribe(audio_file)

    def test_progress_callback_called(self, tmp_path):
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"\x00" * 100)

        fake_proc = _make_fake_popen(0, json.dumps({
            "segments": [], "words": [], "language": "en"
        }), "")

        progress_calls = []

        from preprod import transcribe as t

        with patch.object(t, "WHISPER_AVAILABLE", True), \
             patch.object(t.subprocess, "Popen", return_value=fake_proc):
            t.transcribe(audio_file, progress_callback=progress_calls.append)

        assert len(progress_calls) >= 1

    def test_worker_script_path_is_correct(self):
        from preprod import transcribe as t
        assert t._WORKER.exists(), f"whisper_worker.py not found at {t._WORKER}"
        assert t._WORKER.name == "whisper_worker.py"

    def test_kmp_duplicate_lib_ok_set_in_env(self, tmp_path):
        """KMP_DUPLICATE_LIB_OK=TRUE must be in the subprocess env to suppress OMP warning."""
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"\x00" * 100)

        fake_proc = _make_fake_popen(0, json.dumps({"segments": [], "words": [], "language": "en"}), "")

        from preprod import transcribe as t

        with patch.object(t, "WHISPER_AVAILABLE", True), \
             patch.object(t.subprocess, "Popen", return_value=fake_proc) as mock_popen:
            t.transcribe(audio_file)

        call_kwargs = mock_popen.call_args[1]
        env = call_kwargs.get("env", {})
        assert env.get("KMP_DUPLICATE_LIB_OK") == "TRUE", (
            "KMP_DUPLICATE_LIB_OK=TRUE must be set in subprocess env to suppress OMP warning"
        )

    def test_large_stdout_does_not_deadlock(self, tmp_path):
        """stdout larger than the OS pipe buffer (~64 KB) must not cause a deadlock.

        Previously transcribe() called proc.stdout.read() AFTER proc.wait(), so
        when the child wrote > 64 KB the child blocked on write and the parent
        blocked on wait → deadlock.  The fix drains stdout concurrently.
        """
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"\x00" * 100)

        from preprod import transcribe as t

        # Build a payload larger than 65 536 bytes (one OS pipe buffer).
        many_words = [{"word": f"word{i}", "start": float(i), "end": float(i) + 0.5}
                      for i in range(2000)]
        big_payload = {"segments": [], "words": many_words, "language": "ja"}
        big_json = json.dumps(big_payload)
        assert len(big_json.encode()) > 65536, "test payload must exceed pipe buffer"

        fake_proc = _make_fake_popen(0, big_json, "")

        with patch.object(t, "WHISPER_AVAILABLE", True), \
             patch.object(t.subprocess, "Popen", return_value=fake_proc):
            result = t.transcribe(audio_file, duration=100.0)

        assert len(result["words"]) == 2000

    def test_cancel_event_set_terminates_process(self, tmp_path):
        """When cancel_event is set during a wait() timeout, process is terminated and
        RuntimeError('Cancelled') is raised."""
        import subprocess as sp
        import threading

        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"\x00" * 100)

        from preprod import transcribe as t

        cancel_event = threading.Event()
        cancel_event.set()   # pre-set so it triggers on first timeout check

        fake_proc = MagicMock()
        fake_proc.terminate.return_value = None
        # Second proc.wait() (for graceful shutdown) completes normally
        fake_proc.wait.side_effect = [
            sp.TimeoutExpired(cmd="x", timeout=1.0),  # first call in loop → triggers cancel
            None,                                       # second call (5s grace period) → ok
        ]
        # Give the drain threads proper EOF so they don't spin forever in background
        fake_proc.stdout = io.StringIO("")
        fake_proc.stderr = iter([])

        # Always get TranscribeCancelled from the same module object as `t`.
        # TestWhisperAvailable reimports transcribe and restores sys.modules but
        # not preprod.transcribe's package attribute, so the two can diverge.
        TranscribeCancelled = t.TranscribeCancelled
        with patch.object(t, "WHISPER_AVAILABLE", True), \
             patch.object(t.subprocess, "Popen", return_value=fake_proc):
            with pytest.raises(TranscribeCancelled):
                t.transcribe(audio_file, cancel_event=cancel_event)


# ── detect_fillers — orphan/isolation detection ───────────────────────────────

class TestDetectFillers:
    """Tests for the isolation-gap filler detection logic.

    Key invariant: a candidate filler word is only flagged when it is
    'orphaned' — i.e. has a pause of >= isolation_gap on at least one side.
    Words tightly connected to adjacent speech must NOT be flagged.
    """

    from preprod.transcribe import detect_fillers  # imported at class scope

    def _w(self, word, start, end):
        return {"word": word, "start": start, "end": end}

    # ── basic English fillers ──────────────────────────────────────

    def test_isolated_english_filler_flagged(self):
        from preprod.transcribe import detect_fillers
        # "um" with 300ms gap after → should be flagged
        words = [
            self._w("So",   0.0,  0.3),
            self._w("um",   0.6,  0.8),   # gap_before=0.3, gap_after=0.5
            self._w("yeah", 1.3,  1.6),
        ]
        result = detect_fillers(words, en=True, ja=False)
        assert len(result) == 1
        assert result[0][2].lower() == "um"

    def test_tight_english_word_not_flagged(self):
        from preprod.transcribe import detect_fillers
        # "like" immediately followed by next word (gap_after = 20ms) → NOT a filler
        words = [
            self._w("it's",  0.0, 0.2),
            self._w("like",  0.22, 0.40),  # gap_before=0.02, gap_after=0.02
            self._w("this",  0.42, 0.6),
        ]
        result = detect_fillers(words, en=True, ja=False)
        assert result == []

    # ── Japanese demonstrative/filler ambiguity ────────────────────

    def test_ano_as_filler_flagged_when_isolated(self):
        from preprod.transcribe import detect_fillers
        # "あの" followed by 300ms pause → filler
        words = [
            self._w("あの",    0.0,  0.3),   # gap_before=inf (first word), gap_after=0.3
            self._w("AIに",   0.6,  1.0),
        ]
        result = detect_fillers(words, en=False, ja=True)
        assert len(result) == 1

    def test_sono_never_flagged(self):
        from preprod.transcribe import detect_fillers
        # "その" (bare demonstrative) is NOT in the filler list — never flagged
        words = [
            self._w("基本的に",  0.0,  0.5),
            self._w("その",     1.0,  1.3),   # large gap_before=0.5 — still not flagged
            self._w("チャット", 2.0,  2.4),
        ]
        result = detect_fillers(words, en=False, ja=True)
        assert result == [], "その is not in the filler list and should never be flagged"

    def test_sono_elongated_flagged_when_isolated(self):
        from preprod.transcribe import detect_fillers
        # "そのー" (elongated filler form) IS in the filler list and should be flagged
        words = [
            self._w("では",      0.0,  0.3),
            self._w("そのー",    0.6,  0.9),   # gap_before=0.3, gap_after=0.4
            self._w("次のステップは", 1.3, 1.8),
        ]
        result = detect_fillers(words, en=False, ja=True)
        assert len(result) == 1, "そのー is an unambiguous filler and should be flagged"

    def test_nnn_elongated_flagged_when_isolated(self):
        from preprod.transcribe import detect_fillers
        # "んー" (nasal hesitation) is in the filler list
        words = [
            self._w("では",   0.0,  0.3),
            self._w("んー",   0.6,  0.8),
            self._w("続けて",  1.1,  1.5),
        ]
        result = detect_fillers(words, en=False, ja=True)
        assert len(result) == 1

    def test_eto_flagged_when_isolated(self):
        from preprod.transcribe import detect_fillers
        # "えと" (short form of えーと) is in the filler list
        words = [self._w("えと", 0.0, 0.3), self._w("そう", 0.6, 0.9)]
        result = detect_fillers(words, en=False, ja=True)
        assert len(result) == 1

    def test_un_never_flagged(self):
        from preprod.transcribe import detect_fillers
        # "うん" is NOT in the filler list — never flagged regardless of gaps
        words = [
            self._w("思う",      0.0,  0.4),
            self._w("うん",      0.7,  0.9),   # gap_before=0.3, gap_after=0.4 — still not flagged
            self._w("ですけど",  1.3,  1.8),
        ]
        result = detect_fillers(words, en=False, ja=True)
        assert result == [], "うん is not in the filler list and should never be flagged"

    def test_nanka_isolated_flagged(self):
        from preprod.transcribe import detect_fillers
        # "なんか" IS in the filler list — flagged when isolated with a pause on either side
        words = [
            self._w("それで",  0.0, 0.4),
            self._w("なんか",  0.7, 1.0),  # 0.3s gap before, 0.5s gap after ≥ isolation_gap
            self._w("こう",    1.5, 1.8),
        ]
        result = detect_fillers(words, en=False, ja=True)
        assert len(result) == 1, "なんか should be flagged when isolated by pauses"
        assert result[0][2] == "なんか"

    def test_hora_never_flagged(self):
        from preprod.transcribe import detect_fillers
        words = [self._w("ほら", 0.0, 0.3), self._w("ね", 0.6, 0.8)]
        result = detect_fillers(words, en=False, ja=True)
        assert result == [], "ほら is not in the filler list"

    def test_iya_never_flagged(self):
        from preprod.transcribe import detect_fillers
        words = [self._w("いや", 0.0, 0.3), self._w("でも", 0.6, 0.9)]
        result = detect_fillers(words, en=False, ja=True)
        assert result == [], "いや is not in the filler list"

    # ── boundary conditions ────────────────────────────────────────

    def test_first_word_filler_flagged(self):
        from preprod.transcribe import detect_fillers
        # First word in list → gap_before = inf → always qualifies
        words = [
            self._w("えー",   0.0,  0.3),   # gap_before=inf
            self._w("そうです", 0.35, 0.8),
        ]
        result = detect_fillers(words, en=False, ja=True)
        assert len(result) == 1

    def test_last_word_filler_flagged(self):
        from preprod.transcribe import detect_fillers
        # Last word in list → gap_after = inf → always qualifies
        words = [
            self._w("ですね",  0.0, 0.4),
            self._w("まあ",    0.6, 0.9),   # gap_before=0.2 (>=0.2), gap_after=inf
        ]
        result = detect_fillers(words, en=False, ja=True)
        assert len(result) == 1

    def test_single_word_list_flagged(self):
        from preprod.transcribe import detect_fillers
        words = [self._w("えっと", 0.0, 0.5)]
        result = detect_fillers(words, en=False, ja=True)
        assert len(result) == 1

    # ── custom filler list ─────────────────────────────────────────

    def test_custom_filler_isolated(self):
        from preprod.transcribe import detect_fillers
        words = [
            self._w("so",    0.0,  0.2),
            self._w("ganz",  0.5,  0.7),   # gap_before=0.3 → isolated
            self._w("okay",  1.0,  1.2),
        ]
        result = detect_fillers(words, en=False, ja=False, custom=["ganz"])
        assert len(result) == 1

    def test_custom_filler_tight_not_flagged(self):
        from preprod.transcribe import detect_fillers
        words = [
            self._w("ganz",  0.0,  0.2),   # gap_before=inf, gap_after=0.02 → one side OK
            self._w("genau", 0.22, 0.5),
        ]
        # gap_before=inf (first word) → flagged even though gap_after is small
        result = detect_fillers(words, en=False, ja=False, custom=["ganz"])
        assert len(result) == 1  # boundary word always qualifies

    # ── isolation_gap threshold ────────────────────────────────────

    def test_custom_isolation_gap(self):
        from preprod.transcribe import detect_fillers
        # "あの" surrounded by words, gap 0.15s on both sides
        words = [
            self._w("まず",   0.0,  0.3),
            self._w("あの",   0.45, 0.75),   # gap_before=0.15, gap_after=0.15
            self._w("です",   0.90, 1.2),
        ]
        assert detect_fillers(words, en=False, ja=True, isolation_gap=0.20) == [], \
            "gap of 0.15 should NOT trigger at default 0.20 threshold"
        assert len(detect_fillers(words, en=False, ja=True, isolation_gap=0.10)) == 1, \
            "gap of 0.15 SHOULD trigger at 0.10 threshold"

    # ── disabled language flags ────────────────────────────────────

    def test_japanese_disabled_skips_ja_fillers(self):
        from preprod.transcribe import detect_fillers
        words = [self._w("えー", 0.0, 0.5)]
        assert detect_fillers(words, en=True, ja=False) == []

    def test_english_disabled_skips_en_fillers(self):
        from preprod.transcribe import detect_fillers
        words = [self._w("um", 0.0, 0.3)]
        assert detect_fillers(words, en=False, ja=True) == []

    def test_well_isolated_flagged(self):
        from preprod.transcribe import detect_fillers
        # "well" at sentence start with pause before → isolated filler
        words = [self._w("well", 0.0, 0.3), self._w("I", 0.7, 0.8)]
        result = detect_fillers(words, en=True, ja=False)
        assert len(result) == 1 and result[0][2].lower() == "well"

    def test_well_embedded_not_flagged(self):
        from preprod.transcribe import detect_fillers
        # "well" embedded in "did well at" → no isolation → not flagged
        words = [
            self._w("did",  0.0, 0.2),
            self._w("well", 0.2, 0.4),
            self._w("at",   0.4, 0.6),
        ]
        assert detect_fillers(words, en=True, ja=False) == []

    def test_empty_word_list(self):
        from preprod.transcribe import detect_fillers
        assert detect_fillers([], en=True, ja=True) == []


# ── detect_fillers — audio-based isolation path ───────────────────────────────

class TestDetectFillersAudioPath:
    """Tests for the audio-based RMS isolation path (samples= provided).

    The audio path is the primary production code path; timestamp-based
    isolation is only a fallback. These tests were absent from the original
    test suite — gap identified in the pipeline audit.
    """

    import numpy as np

    SR = 16000

    def _make_samples(self, duration: float, amplitude: float = 0.0) -> "np.ndarray":
        """Flat audio at given amplitude (in linear scale, not dB)."""
        import numpy as np
        n = int(duration * self.SR)
        return np.full(n, amplitude, dtype=np.float32)

    def _make_speech_silence(
        self,
        speech_amp: float,
        silence_amp: float,
        pattern: list[tuple[float, float]],
        total: float,
    ) -> "np.ndarray":
        """Build audio: pattern is [(start, end, is_speech), ...].
        Between pattern entries, fill with silence_amp."""
        import numpy as np
        n_total = int(total * self.SR)
        buf = np.full(n_total, silence_amp, dtype=np.float32)
        for start_t, end_t in pattern:
            i0 = int(start_t * self.SR)
            i1 = min(n_total, int(end_t * self.SR))
            buf[i0:i1] = speech_amp
        return buf

    def _w(self, word, start, end):
        return {"word": word, "start": start, "end": end}

    def test_audio_isolated_filler_flagged(self):
        """Audio silence before and after the word → flagged via audio path."""
        from preprod.transcribe import detect_fillers
        import numpy as np
        # Pattern: [0.0-0.3s speech] [0.3-0.6s silence] [um 0.6-0.8s] [0.8-1.1s silence] [1.1-1.4s speech]
        samples = self._make_speech_silence(
            speech_amp=0.1,
            silence_amp=0.0001,
            pattern=[(0.0, 0.3), (0.6, 0.8), (1.1, 1.4)],
            total=1.4,
        )
        words = [
            self._w("So",   0.0,  0.3),
            self._w("um",   0.6,  0.8),
            self._w("yeah", 1.1,  1.4),
        ]
        result = detect_fillers(words, en=True, ja=False,
                                samples=samples, sample_rate=self.SR,
                                threshold_db=-35.0)
        assert len(result) == 1
        assert result[0][2].lower() == "um"

    def test_audio_tight_word_not_flagged(self):
        """Continuous speech around a filler candidate → not flagged via audio path."""
        from preprod.transcribe import detect_fillers
        import numpy as np
        # All speech, no silence anywhere — "like" is surrounded by continuous audio
        samples = np.full(int(1.0 * self.SR), 0.1, dtype=np.float32)
        words = [
            self._w("it's",  0.0,  0.2),
            self._w("like",  0.2,  0.4),
            self._w("this",  0.4,  0.6),
        ]
        result = detect_fillers(words, en=True, ja=False,
                                samples=samples, sample_rate=self.SR,
                                threshold_db=-35.0)
        assert result == [], "Continuous speech should not trigger filler detection"

    def test_audio_path_takes_priority_over_timestamps(self):
        """When samples are provided, audio RMS is used (not timestamp gaps).

        Middle word has large timestamp gaps (≥isolation_gap on both sides),
        but continuous speech in the audio → audio path suppresses the filler.
        """
        from preprod.transcribe import detect_fillers
        import numpy as np
        # Timestamps have a 300ms gap on each side (would be flagged by timestamp path),
        # but audio is continuous speech → should NOT be flagged via audio path.
        samples = np.full(int(2.0 * self.SR), 0.1, dtype=np.float32)
        words = [
            self._w("hello",  0.0,  0.3),
            self._w("right",  0.6,  0.9),   # 300ms timestamp gap on both sides
            self._w("okay",   1.2,  1.5),
        ]
        # "right" is in the filler list; with continuous audio (RMS >> threshold),
        # neither window before nor after should be below threshold → not flagged
        result = detect_fillers(words, en=True, ja=False,
                                samples=samples, sample_rate=self.SR,
                                threshold_db=-35.0)
        assert result == [], (
            "Audio-based path should suppress filler when audio is continuous "
            "even if timestamp gap is large"
        )

    def test_audio_first_word_always_isolated_before(self):
        """First word is always has_gap_before=True (file-start counts as silence)."""
        from preprod.transcribe import detect_fillers
        import numpy as np
        # Word starts immediately at t=0, followed by silence → should be flagged
        samples = self._make_speech_silence(
            speech_amp=0.1,
            silence_amp=0.0001,
            pattern=[(0.0, 0.3)],
            total=0.8,
        )
        words = [self._w("um", 0.0, 0.3)]
        result = detect_fillers(words, en=True, ja=False,
                                samples=samples, sample_rate=self.SR,
                                threshold_db=-35.0)
        assert len(result) == 1

    def test_audio_empty_samples_falls_back_to_timestamps(self):
        """Empty samples array → falls back to timestamp-based path."""
        from preprod.transcribe import detect_fillers
        import numpy as np
        samples = np.array([], dtype=np.float32)
        words = [
            self._w("So",   0.0,  0.3),
            self._w("um",   0.6,  0.8),   # 300ms timestamp gap
            self._w("yeah", 1.3,  1.6),
        ]
        result = detect_fillers(words, en=True, ja=False,
                                samples=samples, sample_rate=self.SR,
                                threshold_db=-35.0)
        # Empty samples triggers timestamp fallback; 300ms gap >= 200ms isolation_gap
        assert len(result) == 1

    def test_rms_window_negative_start_clamped(self):
        """RMS window with negative start (word near t=0) should not raise."""
        from preprod.transcribe import detect_fillers
        import numpy as np
        samples = np.full(int(1.0 * self.SR), 0.0001, dtype=np.float32)
        words = [
            self._w("um",   0.05, 0.2),   # start - isolation_gap = -0.15 → clamped to 0
            self._w("yeah", 0.8,  1.0),
        ]
        # Should not raise; returns result based on audio RMS
        result = detect_fillers(words, en=True, ja=False,
                                samples=samples, sample_rate=self.SR,
                                threshold_db=-35.0)
        assert isinstance(result, list)  # no exception

# ── detect_fillers — multi-token Japanese grouping (faster-whisper char-level) ─

class TestDetectFillersMultiToken:
    """Multi-token Japanese filler grouping for faster-whisper character-level tokens.

    faster-whisper tokenizes Japanese at character level.  'えー' becomes
    ['え', 'ー'] with near-zero inner gap.  The second pass in detect_fillers
    must compose these tokens and match the composed form against JAPANESE_FILLERS.
    """

    SR = 16000

    def _w(self, word, start, end):
        return {"word": word, "start": start, "end": end}

    def _silence_samples(self, total: float):
        """Near-zero amplitude audio (below any reasonable threshold)."""
        import numpy as np
        return np.full(int(total * self.SR), 1e-5, dtype=np.float32)

    def test_two_char_filler_detected(self):
        """'え'+'ー' with 20 ms inner gap → composed 'えー' in filler_set → detected."""
        from preprod.transcribe import detect_fillers
        import numpy as np
        samples = self._silence_samples(1.0)
        words = [
            self._w("え",  0.3, 0.45),
            self._w("ー",  0.47, 0.6),   # 20 ms inner gap
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert len(result) == 1
        assert result[0][2] == "えー"

    def test_three_char_filler_detected(self):
        """'え'+'ー'+'と' → 'えーと' → detected."""
        from preprod.transcribe import detect_fillers
        import numpy as np
        samples = self._silence_samples(1.0)
        words = [
            self._w("え",  0.1, 0.2),
            self._w("ー",  0.22, 0.32),
            self._w("と",  0.34, 0.44),
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert len(result) == 1
        assert result[0][2] == "えーと"

    def test_longest_span_preferred(self):
        """When both 'えー' (span=2) and 'えーと' (span=3) match, 'えーと' wins."""
        from preprod.transcribe import detect_fillers
        import numpy as np
        samples = self._silence_samples(1.0)
        words = [
            self._w("え",  0.1, 0.2),
            self._w("ー",  0.22, 0.32),
            self._w("と",  0.34, 0.44),
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert len(result) == 1
        assert result[0][2] == "えーと"  # longer match wins

    def test_large_inner_gap_breaks_grouping(self):
        """Inner gap > 100 ms prevents grouping: 'え' + (200 ms gap) + 'ー' → no match."""
        from preprod.transcribe import detect_fillers
        import numpy as np
        samples = self._silence_samples(1.0)
        words = [
            self._w("え",  0.3, 0.45),
            self._w("ー",  0.65, 0.8),   # 200 ms inner gap — exceeds 100 ms limit
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert result == [], "Large inner gap must prevent multi-token grouping"

    def test_eighty_ms_gap_still_detected(self):
        """Inner gap of 80 ms must still group under the 100 ms threshold: 'え'+'ー' → 'えー'."""
        from preprod.transcribe import detect_fillers
        samples = self._silence_samples(1.0)
        words = [
            self._w("え",  0.3,  0.40),
            self._w("ー",  0.48, 0.60),  # 80 ms gap — was broken with old 50 ms limit
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert len(result) == 1, f"80 ms gap must be accepted at 100 ms threshold; got {result}"

    def test_hundred_ten_ms_gap_breaks_grouping(self):
        """Inner gap of 110 ms (just above 100 ms) must prevent grouping: 'え'+'ー' → no match."""
        from preprod.transcribe import detect_fillers
        samples = self._silence_samples(1.0)
        words = [
            self._w("え",  0.3,  0.40),
            self._w("ー",  0.51, 0.62),  # 110 ms gap — exceeds 100 ms limit
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert result == [], f"110 ms gap must break grouping at 100 ms threshold; got {result}"

    def test_grouped_tokens_not_double_counted(self):
        """Tokens consumed by multi-token match are excluded from further matching."""
        from preprod.transcribe import detect_fillers
        import numpy as np
        samples = self._silence_samples(1.0)
        # え+ー = えー (filler); don't also match え alone as a filler (え ∉ filler_set anyway)
        words = [
            self._w("え",  0.3, 0.45),
            self._w("ー",  0.47, 0.6),
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert len(result) == 1  # exactly one match, not two

    def test_japanese_disabled_skips_multi_token(self):
        """Multi-token pass is skipped when ja=False."""
        from preprod.transcribe import detect_fillers
        import numpy as np
        samples = self._silence_samples(1.0)
        words = [
            self._w("え",  0.3, 0.45),
            self._w("ー",  0.47, 0.6),
        ]
        result = detect_fillers(words, en=True, ja=False,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert result == []

    def test_speech_around_multi_token_not_flagged(self):
        """Multi-token filler surrounded by continuous speech → isolation check blocks it."""
        from preprod.transcribe import detect_fillers
        import numpy as np
        # Fill ALL audio with speech-level amplitude — nothing is silence
        samples = np.full(int(1.5 * self.SR), 0.1, dtype=np.float32)
        words = [
            self._w("で",   0.0,  0.1),
            self._w("は",   0.1,  0.2),
            self._w("え",   0.2,  0.35),
            self._w("ー",   0.37, 0.5),   # えー in filler_set but not isolated
            self._w("こ",   0.5,  0.6),   # additional words AFTER so last-word shortcut doesn't fire
            self._w("と",   0.6,  0.7),
            self._w("で",   0.7,  0.8),
            self._w("す",   0.8,  0.9),
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert result == [], "Continuous speech should prevent multi-token filler from being flagged"

    def test_uu_n_three_token_filler_detected(self):
        """'う'+'ー'+'ん' (3 char-level tokens) → composed 'うーん' in filler_set → detected."""
        from preprod.transcribe import detect_fillers
        samples = self._silence_samples(2.0)
        words = [
            self._w("う",  0.5,  0.65),
            self._w("ー",  0.66, 0.80),
            self._w("ん",  0.81, 0.95),
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert len(result) == 1
        assert result[0][2] == "うーん", f"expected 'うーん', got {result[0][2]!r}"

    def test_ee_tto_four_token_filler_detected(self):
        """'え'+'ー'+'っ'+'と' (4 tokens) → composed 'えーっと' in filler_set → detected."""
        from preprod.transcribe import detect_fillers
        samples = self._silence_samples(2.0)
        words = [
            self._w("え",  0.3,  0.45),
            self._w("ー",  0.46, 0.60),
            self._w("っ",  0.61, 0.70),
            self._w("と",  0.71, 0.85),
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert len(result) == 1
        assert result[0][2] == "えーっと", f"expected 'えーっと', got {result[0][2]!r}"

    def test_etto_three_token_filler_detected(self):
        """'え'+'っ'+'と' (3 tokens without elongation) → 'えっと' → detected."""
        from preprod.transcribe import detect_fillers
        samples = self._silence_samples(2.0)
        words = [
            self._w("え",  0.3,  0.45),
            self._w("っ",  0.46, 0.55),
            self._w("と",  0.56, 0.70),
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert len(result) == 1
        assert result[0][2] == "えっと", f"expected 'えっと', got {result[0][2]!r}"

    def test_anoo_three_token_filler_detected(self):
        """'あ'+'の'+'ー' (3 tokens) → 'あのー' → detected."""
        from preprod.transcribe import detect_fillers
        samples = self._silence_samples(2.0)
        words = [
            self._w("あ",  0.3,  0.42),
            self._w("の",  0.43, 0.55),
            self._w("ー",  0.56, 0.70),
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert len(result) == 1
        assert result[0][2] == "あのー", f"expected 'あのー', got {result[0][2]!r}"

    def test_nanka_three_token_filler_detected(self):
        """'な'+'ん'+'か' (3 tokens) → 'なんか' → detected as filler when isolated."""
        from preprod.transcribe import detect_fillers
        samples = self._silence_samples(1.0)
        words = [
            self._w("な",  0.3,  0.40),
            self._w("ん",  0.41, 0.50),
            self._w("か",  0.51, 0.60),
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert len(result) == 1, f"'なんか' should be detected as filler; got {result}"
        assert result[0][2] == "なんか"

    def test_nanka_not_flagged_in_tight_speech(self):
        """'なんか' immediately surrounded by speech must not be flagged (non-filler use).

        Surrounding speech words are placed in the word list (indices 0 and 4) so
        that idx_start != 0 and idx_end != n-1, forcing the audio-based RMS check
        rather than the boundary shortcircuit.
        """
        from preprod.transcribe import detect_fillers
        import numpy as np
        sr = self.SR
        total_dur = 1.5
        amp = 10.0 ** (-20.0 / 20.0)   # −20 dB — well above −35 dB threshold
        # Entire audio is loud speech — no silence window anywhere
        samples = (np.random.rand(int(total_dur * sr)).astype(np.float32) - 0.5) * 2 * amp
        words = [
            self._w("それ",  0.0,  0.20),   # idx 0 — speech before
            self._w("な",    0.21, 0.31),   # idx 1 — start of なんか
            self._w("ん",    0.32, 0.42),   # idx 2
            self._w("か",    0.43, 0.53),   # idx 3 — end of なんか (idx 3 != n-1=4)
            self._w("あった", 0.54, 0.90),  # idx 4 — speech after
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=sr, threshold_db=-35.0)
        assert result == [], f"'なんか' in continuous speech must NOT be flagged; got {result}"

    def test_nankaー_four_token_filler_detected(self):
        """'な'+'ん'+'か'+'ー' (4 tokens) → composed 'なんかー' → detected when isolated.

        'なんかー' is a distinct elongated variant in JAPANESE_FILLERS and is the
        only 4-character filler that is NOT also a prefix of a longer JAPANESE_FILLER
        entry — making this a unique regression test for the 4-token span path.
        """
        from preprod.transcribe import detect_fillers
        samples = self._silence_samples(2.0)
        words = [
            self._w("な",  0.3,  0.40),
            self._w("ん",  0.41, 0.50),
            self._w("か",  0.51, 0.60),
            self._w("ー",  0.61, 0.75),
        ]
        result = detect_fillers(words, en=False, ja=True,
                                samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert len(result) == 1, f"'なんかー' (4 tokens) should be detected; got {result}"
        assert result[0][2] == "なんかー", f"expected 'なんかー', got {result[0][2]!r}"

    def test_custom_multi_token_filler_detected(self):
        """Custom filler not in JAPANESE_FILLERS is caught by multi-token pass.

        Regression: Pass 2 previously checked JAPANESE_FILLERS instead of filler_set,
        so custom compound Japanese fillers were silently missed even when ja=True.

        "むー" is used as the custom filler because neither "む" nor "むー" appears
        in the built-in JAPANESE_FILLERS set, so no prefix can be matched without
        the custom argument.
        """
        from preprod.transcribe import detect_fillers
        samples = self._silence_samples(2.0)
        # "むー" is NOT in the built-in JAPANESE_FILLERS set.
        # faster-whisper would tokenize it as ['む', 'ー'] with a near-zero gap.
        words = [
            self._w("む",  0.3,  0.45),
            self._w("ー",  0.46, 0.60),
        ]
        # Without custom=, no match in any set → not detected.
        result_no_custom = detect_fillers(words, en=False, ja=True,
                                          samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert result_no_custom == [], "Not in built-in set without custom arg"

        # With custom=["むー"], Pass 2 must compose ['む','ー'] → 'むー' and find it in filler_set.
        result_with_custom = detect_fillers(words, en=False, ja=True,
                                            custom=["むー"],
                                            samples=samples, sample_rate=self.SR, threshold_db=-35.0)
        assert len(result_with_custom) == 1, \
            f"Custom 2-token filler 'むー' should be detected; got {result_with_custom}"
        assert result_with_custom[0][2] == "むー"
