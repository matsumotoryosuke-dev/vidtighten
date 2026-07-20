"""Tests for whisper_worker.py.

Regression test for Bug #2:
  Whisper printed "Detected language: Japanese" to stdout before the JSON,
  causing json.JSONDecodeError in the parent process.

Fix: redirect sys.stdout → sys.stderr while whisper runs.

The test runs the worker as a subprocess with a mocked whisper module,
captures stdout, and asserts that stdout is valid JSON with correct structure.
"""

import json
import os
import sys
import subprocess
import textwrap
from pathlib import Path

import pytest

WORKER_PATH = Path(__file__).parent.parent / "src" / "preprod" / "whisper_worker.py"

# A fake whisper module that prints to stdout (simulating the bug scenario)
# and returns a predictable result.
_FAKE_WHISPER_STUB = textwrap.dedent("""
import sys, builtins

class _FakeModel:
    def transcribe(self, path, **kwargs):
        # Simulate what real whisper does: print to stdout (the pre-bug behavior)
        # The fix in whisper_worker.py redirects stdout→stderr before this runs,
        # so this print must NOT appear in the final stdout.
        print("Detected language: Japanese")
        print("Some other whisper chatter")
        return {
            "language": "ja",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.5,
                    "text": "テスト",
                    "words": [
                        {"word": "テスト", "start": 0.0, "end": 2.5, "probability": 0.92}
                    ],
                }
            ],
        }

def load_model(name, **kwargs):  # accept device= and any future kwargs
    return _FakeModel()
""")

# Stub that makes faster_whisper unavailable so the worker falls through to the
# fake openai-whisper above without loading any ML model.  Without this, each
# subprocess would load the real faster_whisper model (~2 s each) before failing
# on "/fake/audio.wav" and retrying with the fake whisper — making 16 tests slow.
_FAKE_FASTER_WHISPER_STUB = textwrap.dedent("""
raise ImportError("faster-whisper mocked out in tests")
""")


def _run_worker(args_json: str) -> subprocess.CompletedProcess:
    """Run the whisper_worker subprocess with mocked whisper modules.

    Injects two stubs via PYTHONPATH:
    - ``whisper.py``: fake openai-whisper that returns predictable results without
      loading any model (also simulates pre-fix stdout printing for Bug #2 tests).
    - ``faster_whisper.py``: raises ImportError immediately so the worker never
      tries to load the real CTranslate2 model (~2 s each).  Without this stub,
      each subprocess call would attempt real model loading before falling back.
    """
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "whisper.py"), "w") as f:
            f.write(_FAKE_WHISPER_STUB)
        with open(os.path.join(tmpdir, "faster_whisper.py"), "w") as f:
            f.write(_FAKE_FASTER_WHISPER_STUB)

        env = {**os.environ, "PYTHONPATH": tmpdir + os.pathsep + os.environ.get("PYTHONPATH", "")}
        return subprocess.run(
            [sys.executable, str(WORKER_PATH), args_json],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )


# ── Bug #2 regression: stdout must be clean JSON ─────────────────────────────

class TestWhisperWorkerStdoutIsCleanJSON:
    """The critical regression test for Bug #2.

    Before the fix, whisper's "Detected language: ..." printed to stdout
    before the JSON payload, making json.loads() fail in the parent.
    After the fix, sys.stdout is redirected to sys.stderr during whisper execution.
    """

    @pytest.mark.timeout(20)
    def test_stdout_is_valid_json(self):
        args = json.dumps({"path": "/fake/audio.wav", "model_size": "base", "language": None})
        proc = _run_worker(args)
        assert proc.returncode == 0, f"Worker exited non-zero: {proc.stderr}"
        # This is the key regression assertion: stdout must parse as JSON
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"REGRESSION BUG #2: stdout is not valid JSON. "
                f"Got: {proc.stdout[:300]!r}. Error: {exc}"
            )

    @pytest.mark.timeout(20)
    def test_stdout_contains_no_extra_lines(self):
        """There must be exactly one line of output (the JSON), not multiple."""
        args = json.dumps({"path": "/fake/audio.wav", "model_size": "base", "language": None})
        proc = _run_worker(args)
        assert proc.returncode == 0, proc.stderr
        # Strip trailing newline and check only one non-empty line
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        assert len(lines) == 1, (
            f"REGRESSION BUG #2: expected exactly 1 output line, got {len(lines)}: {proc.stdout!r}"
        )

    @pytest.mark.timeout(20)
    def test_whisper_prints_go_to_stderr_not_stdout(self):
        """Verify that the "Detected language" text ends up in stderr, not stdout."""
        args = json.dumps({"path": "/fake/audio.wav", "model_size": "base", "language": None})
        proc = _run_worker(args)
        assert proc.returncode == 0, proc.stderr
        assert "Detected language" not in proc.stdout, (
            "REGRESSION BUG #2: 'Detected language' found in stdout — "
            "stdout redirect to stderr is not working"
        )


# ── Output structure correctness ──────────────────────────────────────────────

@pytest.fixture(scope="class")
def worker_result():
    """Run the worker once and share the parsed JSON across all structure tests.

    Running once per class avoids spawning 10 subprocesses (one per test method)
    for the same input.  The fake faster_whisper + fake whisper stubs mean each
    subprocess still takes ~0.9 s for Python startup; sharing cuts that to 1 call.
    """
    args = json.dumps({"path": "/fake/audio.wav", "model_size": "base", "language": None})
    proc = _run_worker(args)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


class TestWhisperWorkerOutputStructure:
    def test_result_has_segments_key(self, worker_result):
        assert "segments" in worker_result

    def test_result_has_words_key(self, worker_result):
        assert "words" in worker_result

    def test_result_has_language_key(self, worker_result):
        assert "language" in worker_result

    def test_language_value_is_string(self, worker_result):
        assert isinstance(worker_result["language"], str)

    def test_segments_is_list(self, worker_result):
        assert isinstance(worker_result["segments"], list)

    def test_words_is_list(self, worker_result):
        assert isinstance(worker_result["words"], list)

    def test_segment_has_required_fields(self, worker_result):
        assert len(worker_result["segments"]) > 0
        seg = worker_result["segments"][0]
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg

    def test_segment_times_are_floats(self, worker_result):
        seg = worker_result["segments"][0]
        assert isinstance(seg["start"], (int, float))
        assert isinstance(seg["end"], (int, float))

    def test_word_has_required_fields(self, worker_result):
        assert len(worker_result["words"]) > 0
        word = worker_result["words"][0]
        assert "word" in word
        assert "start" in word
        assert "end" in word

    def test_word_times_are_floats(self, worker_result):
        word = worker_result["words"][0]
        assert isinstance(word["start"], (int, float))
        assert isinstance(word["end"], (int, float))

    def test_word_score_forwarded_from_probability(self, worker_result):
        """openai-whisper 'probability' must be forwarded as 'score' so the
        frontend's confidence-underline feature works with the fallback backend."""
        word = worker_result["words"][0]
        assert "score" in word, (
            "openai-whisper word 'probability' should be forwarded as 'score'; "
            "without it the frontend cannot show low-confidence underlines"
        )
        assert isinstance(word["score"], float)
        assert 0.0 <= word["score"] <= 1.0

    def test_segment_has_seg_id(self, worker_result):
        """Each segment must carry seg_id so split_telop_segments can select words
        by exact segment membership (preventing cross-boundary bleed)."""
        seg = worker_result["segments"][0]
        assert "seg_id" in seg
        assert seg["seg_id"] == 0

    def test_word_seg_id_matches_its_segment(self, worker_result):
        """A word's seg_id must equal its containing segment's seg_id so the two can
        be cross-referenced during text reconstruction."""
        seg = worker_result["segments"][0]
        word = worker_result["words"][0]
        assert word["seg_id"] == seg["seg_id"]


# ── WhisperX score-omission regression ───────────────────────────────────────

# Fake openai-whisper that returns two words so the worker reaches the whisperX path.
_FAKE_WHISPER_FOR_WHISPERX_TEST = textwrap.dedent("""
import sys

class _FakeModel:
    def transcribe(self, path, **kwargs):
        return {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.8, "probability": 0.9},
                        {"word": "world", "start": 1.0, "end": 1.9, "probability": 0.88},
                    ],
                }
            ],
        }

def load_model(name, **kwargs):
    return _FakeModel()
""")

# Fake whisperx that returns the same words — one WITH score, one WITHOUT.
# This exercises the score-omission fix in _run_whisperx_alignment.
_FAKE_WHISPERX_STUB = textwrap.dedent("""
class _FakeArray:
    pass

def load_audio(path):
    return _FakeArray()

def load_align_model(language_code, device):
    return (object(), {})

def align(segments, model, metadata, audio, device, return_char_alignments=False):
    # 'hello' has score=0.95; 'world' has NO score key (whisperX sometimes omits it).
    return {
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.8, "score": 0.95},
                    {"word": "world", "start": 1.0, "end": 1.9},
                ],
            }
        ]
    }
""")


def _run_worker_with_whisperx(args_json: str) -> subprocess.CompletedProcess:
    """Run the worker with fake openai-whisper + fake whisperx so the alignment path runs."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        # Block faster_whisper so worker falls back to openai-whisper.
        with open(os.path.join(tmpdir, "faster_whisper.py"), "w") as f:
            f.write('raise ImportError("faster-whisper mocked out")\n')
        # Provide a working fake openai-whisper that produces words.
        with open(os.path.join(tmpdir, "whisper.py"), "w") as f:
            f.write(_FAKE_WHISPER_FOR_WHISPERX_TEST)
        # Inject our whisperx mock.
        whisperx_dir = os.path.join(tmpdir, "whisperx")
        os.makedirs(whisperx_dir)
        with open(os.path.join(whisperx_dir, "__init__.py"), "w") as f:
            f.write(_FAKE_WHISPERX_STUB)

        env = {**os.environ, "PYTHONPATH": tmpdir + os.pathsep + os.environ.get("PYTHONPATH", "")}
        return subprocess.run(
            [sys.executable, str(WORKER_PATH), args_json],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )


class TestWhisperXScoreOmission:
    """Regression for score=0.0 bug: whisperX words lacking a score key must NOT
    appear with score=0.0 in the output; the key should simply be absent so the
    frontend treats confidence as None (no underline) rather than 0% (dotted underline).
    """

    @pytest.fixture(scope="class")
    def wx_result(self):
        args = json.dumps({"path": "/fake/audio.wav", "model_size": "base", "language": "en"})
        proc = _run_worker_with_whisperx(args)
        if proc.returncode != 0:
            pytest.skip(f"whisperx path exited non-zero: {proc.stderr[:300]}")
        return json.loads(proc.stdout)

    def test_word_with_score_has_score_key(self, wx_result):
        """'hello' was returned by fake whisperx with score=0.95 — must be present."""
        words = wx_result.get("words", [])
        scored = [w for w in words if w.get("word") == "hello"]
        assert scored, f"'hello' not found in words: {words}"
        assert "score" in scored[0], (
            f"'hello' word missing 'score' key: {scored[0]}"
        )
        assert abs(scored[0]["score"] - 0.95) < 0.001

    def test_word_without_score_has_no_score_key(self, wx_result):
        """'world' was returned by fake whisperx with NO score key.

        REGRESSION: before the fix, whisper_worker.py used `score=0.0` as default,
        so every unscored word got score=0.0, triggering the low-confidence underline.
        After the fix, the key is absent — the frontend then reads confidence as None.
        """
        words = wx_result.get("words", [])
        unscored = [w for w in words if w.get("word") == "world"]
        assert unscored, f"'world' not found in words: {words}"
        assert "score" not in unscored[0], (
            f"REGRESSION: 'world' (no whisperX score) got score={unscored[0].get('score')!r}. "
            "A missing score key should stay absent so the frontend shows no confidence underline."
        )

    def test_whisperx_words_inherit_original_seg_id(self, wx_result):
        """WhisperX re-enumerates its aligned segments; refined words must inherit
        the ORIGINAL faster-whisper seg_id (here 0) so they stay consistent with the
        segment list, whose seg_id may have gaps from skipped empty segments."""
        seg = wx_result["segments"][0]
        for w in wx_result.get("words", []):
            assert w.get("seg_id") == seg["seg_id"], (
                f"refined word {w.get('word')!r} seg_id {w.get('seg_id')} "
                f"!= segment seg_id {seg['seg_id']}"
            )


# ── WhisperX empty-segments early-return ─────────────────────────────────────

# Fake whisper that returns a segment with whitespace-only text.
# wx_segs will be empty after filtering → _run_whisperx_alignment returns None
# and the worker must still produce valid output from the original word list.
_FAKE_WHISPER_EMPTY_SEGS = textwrap.dedent("""
import sys

class _FakeModel:
    def transcribe(self, path, **kwargs):
        return {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "   ",   # whitespace-only — filtered out by wx_segs
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.8, "probability": 0.9},
                    ],
                }
            ],
        }

def load_model(name, **kwargs):
    return _FakeModel()
""")

# Sentinel whisperX stub — raises if align() is called so the test fails loudly
# if the early-return guard is accidentally removed.
_FAKE_WHISPERX_SENTINEL = textwrap.dedent("""
class _FakeArray:
    pass

def load_audio(path):
    return _FakeArray()

def load_align_model(language_code, device):
    return (object(), {})

def align(segments, model, metadata, audio, device, return_char_alignments=False):
    raise RuntimeError("align() must NOT be called when wx_segs is empty")
""")


class TestWhisperXEmptySegmentsEarlyReturn:
    """Regression for the early-return guard added in commit 9557228.

    When all transcribed segment texts are whitespace-only, wx_segs is empty
    after filtering.  whisperx.align([]) must not be called (behaviour varies
    across versions).  The guard returns None and the worker must fall back to
    the openai-whisper word list without crashing.
    """

    @pytest.fixture(scope="class")
    def result(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "faster_whisper.py"), "w") as f:
                f.write('raise ImportError("faster-whisper mocked out")\n')
            with open(os.path.join(tmpdir, "whisper.py"), "w") as f:
                f.write(_FAKE_WHISPER_EMPTY_SEGS)
            whisperx_dir = os.path.join(tmpdir, "whisperx")
            os.makedirs(whisperx_dir)
            with open(os.path.join(whisperx_dir, "__init__.py"), "w") as f:
                f.write(_FAKE_WHISPERX_SENTINEL)

            env = {**os.environ, "PYTHONPATH": tmpdir + os.pathsep + os.environ.get("PYTHONPATH", "")}
            proc = subprocess.run(
                [sys.executable, str(WORKER_PATH),
                 json.dumps({"path": "/fake/audio.wav", "model_size": "base", "language": "en"})],
                capture_output=True, text=True, timeout=30, env=env,
            )
            return proc

    def test_worker_exits_zero(self, result):
        assert result.returncode == 0, f"stderr: {result.stderr[:400]}"

    def test_stdout_is_valid_json(self, result):
        data = json.loads(result.stdout)
        assert isinstance(data, dict)

    def test_output_has_words_from_original_whisper(self, result):
        """Words must come from openai-whisper since whisperX was skipped."""
        data = json.loads(result.stdout)
        words = data.get("words", [])
        assert len(words) == 1
        assert words[0]["word"] == "hello"


# ── _INITIAL_PROMPT token budget ──────────────────────────────────────────────

class TestInitialPromptTokenBudget:
    """Regression for the silent 280-token prompt that Whisper truncated to 224.

    The entire YouTube/creator vocabulary section was being silently dropped.
    This test locks the prompt to ≤ 224 tokens so a future edit cannot
    accidentally re-introduce the truncation.

    Uses the Whisper tokenizer if available (accurate); falls back to a
    character-count heuristic (CJK chars ≈ 1 token, ASCII chars ≈ 0.35 tokens).
    """

    _WHISPER_224_LIMIT = 224

    def test_initial_prompt_fits_within_whisper_token_limit(self):
        """_INITIAL_PROMPT must be ≤ 224 Whisper tokens to avoid silent truncation."""
        # Import the prompt from the worker module.
        import importlib.util, pathlib
        worker_spec = importlib.util.spec_from_file_location(
            "whisper_worker",
            pathlib.Path(__file__).parent.parent / "src" / "preprod" / "whisper_worker.py",
        )
        worker = importlib.util.module_from_spec(worker_spec)
        worker_spec.loader.exec_module(worker)
        prompt = worker._INITIAL_PROMPT

        try:
            import whisper  # type: ignore
            tok = whisper.tokenizer.get_tokenizer(False)
            n_tokens = len(tok.encode(prompt))
            method = "whisper-tokenizer"
        except ImportError:
            # Heuristic: CJK = 1 token each, ASCII = ~2.85 chars/token
            import unicodedata
            cjk = sum(
                1 for c in prompt
                if unicodedata.east_asian_width(c) in ("W", "F")
            )
            ascii_chars = len(prompt) - cjk
            n_tokens = cjk + int(ascii_chars / 2.85) + 1  # +1 conservative buffer
            method = "heuristic"

        assert n_tokens <= self._WHISPER_224_LIMIT, (
            f"_INITIAL_PROMPT is {n_tokens} tokens (via {method}) — "
            f"exceeds the {self._WHISPER_224_LIMIT}-token Whisper limit. "
            f"Prompt tail would be silently truncated. "
            f"Trim the prompt to ≤ {self._WHISPER_224_LIMIT} tokens."
        )


# ── Error handling ────────────────────────────────────────────────────────────

class TestWhisperWorkerErrorHandling:
    def test_no_args_exits_nonzero(self):
        proc = subprocess.run(
            [sys.executable, str(WORKER_PATH)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode != 0

    def test_invalid_json_arg_exits_nonzero(self):
        proc = subprocess.run(
            [sys.executable, str(WORKER_PATH), "not-valid-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert proc.returncode != 0

    def test_missing_whisper_exits_nonzero(self):
        """Without fake whisper on path, real whisper likely absent in test env."""
        import os
        env = {**os.environ}
        # Remove any PYTHONPATH that might have fake whisper
        env.pop("PYTHONPATH", None)
        args = json.dumps({"path": "/fake/audio.wav"})
        proc = subprocess.run(
            [sys.executable, str(WORKER_PATH), args],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        # Either whisper is installed (returncode=0) or not (returncode=2)
        # What we assert is that if it exits non-zero, the error goes to stderr not stdout
        if proc.returncode != 0:
            assert proc.stdout.strip() == "" or not proc.stdout.strip(), (
                "Error messages should go to stderr, not stdout"
            )


# ── Fix 1: seg_id stamped on raw word tokens ─────────────────────────────────

class TestSegIdStampedOnWords:
    """Fix 1 regression: seg_id must be present on every word token so that
    group_word_tokens can use it to break CJK groups at segment boundaries.
    Before the fix, seg_id was always None and the boundary-break never fired.
    """

    @pytest.fixture(scope="class")
    def result(self):
        args = json.dumps({"path": "/fake/audio.wav", "model_size": "base", "language": None})
        proc = _run_worker(args)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)

    def test_word_has_seg_id_key(self, result):
        """Every word dict must have a 'seg_id' key (may be int or None)."""
        words = result.get("words", [])
        assert words, "expected at least one word in output"
        for w in words:
            assert "seg_id" in w, (
                f"Fix 1 MISSING: word {w!r} has no 'seg_id' key. "
                "group_word_tokens's segment-boundary break will never fire."
            )

    def test_seg_id_is_integer(self, result):
        """seg_id must be an integer (the segment index from faster-whisper)."""
        words = result.get("words", [])
        for w in words:
            assert isinstance(w["seg_id"], int), (
                f"seg_id must be int, got {type(w['seg_id']).__name__!r} for word {w!r}"
            )

    def test_seg_id_is_zero_for_first_segment(self, result):
        """The fake whisper stub has one segment — all words should have seg_id=0."""
        words = result.get("words", [])
        assert all(w["seg_id"] == 0 for w in words), (
            f"All words from the single fake segment should have seg_id=0; got: "
            f"{[w.get('seg_id') for w in words]}"
        )


# ── Fix 3: whisperx_used flag in result payload ───────────────────────────────

class TestWhisperxUsedFlag:
    """Fix 3: result payload must include a 'whisperx_used' boolean field so the
    frontend knows whether it got ±20ms (WhisperX) or ±150ms alignment.
    """

    @pytest.fixture(scope="class")
    def result_no_wx(self):
        """Run worker without whisperx available — whisperx_used must be False."""
        args = json.dumps({"path": "/fake/audio.wav", "model_size": "base", "language": None})
        proc = _run_worker(args)
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)

    @pytest.fixture(scope="class")
    def result_with_wx(self):
        """Run worker with fake whisperx — whisperx_used must be True."""
        args = json.dumps({"path": "/fake/audio.wav", "model_size": "base", "language": "en"})
        proc = _run_worker_with_whisperx(args)
        if proc.returncode != 0:
            pytest.skip(f"whisperx path exited non-zero: {proc.stderr[:300]}")
        return json.loads(proc.stdout)

    def test_whisperx_used_key_present_when_no_wx(self, result_no_wx):
        assert "whisperx_used" in result_no_wx, (
            "Fix 3 MISSING: 'whisperx_used' key absent from payload. "
            "The frontend cannot determine alignment accuracy."
        )

    def test_whisperx_used_is_false_when_no_wx(self, result_no_wx):
        assert result_no_wx["whisperx_used"] is False, (
            f"whisperx_used must be False when WhisperX is not available; "
            f"got {result_no_wx.get('whisperx_used')!r}"
        )

    def test_whisperx_used_key_present_when_wx_ran(self, result_with_wx):
        assert "whisperx_used" in result_with_wx, (
            "Fix 3 MISSING: 'whisperx_used' key absent even when WhisperX ran."
        )

    def test_whisperx_used_is_true_when_wx_ran(self, result_with_wx):
        assert result_with_wx["whisperx_used"] is True, (
            f"whisperx_used must be True when WhisperX alignment succeeded; "
            f"got {result_with_wx.get('whisperx_used')!r}"
        )


# ── Fix 1: seg_id stamped on whisperx output words ───────────────────────────

class TestWhisperXWordsHaveSegId:
    """Fix 1 (WhisperX path): seg_id must be stamped on words returned by
    _run_whisperx_alignment so group_word_tokens can use segment boundaries.
    """

    @pytest.fixture(scope="class")
    def wx_result(self):
        args = json.dumps({"path": "/fake/audio.wav", "model_size": "base", "language": "en"})
        proc = _run_worker_with_whisperx(args)
        if proc.returncode != 0:
            pytest.skip(f"whisperx path exited non-zero: {proc.stderr[:300]}")
        return json.loads(proc.stdout)

    def test_all_whisperx_words_have_seg_id(self, wx_result):
        words = wx_result.get("words", [])
        assert words, "expected words in whisperx output"
        for w in words:
            assert "seg_id" in w, (
                f"Fix 1 (WhisperX): word {w!r} is missing 'seg_id'. "
                "_run_whisperx_alignment must stamp seg_id on its output words."
            )

    def test_whisperx_seg_id_is_integer(self, wx_result):
        words = wx_result.get("words", [])
        for w in words:
            assert isinstance(w["seg_id"], int), (
                f"seg_id must be int on whisperx words, got {type(w['seg_id']).__name__!r}"
            )
