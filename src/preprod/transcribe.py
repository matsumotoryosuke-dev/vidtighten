"""Whisper transcription and filler-word detection.

Requires openai-whisper: pip install preprod[whisper]
If not installed, only silence detection is available.

Whisper is run in a subprocess so that torch's bundled libomp never
shares a process with numpy's OpenBLAS libomp — two concurrent OMP
runtimes cause a SIGSEGV on macOS Apple Silicon.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from preprod.audio import terminate_and_reap

# Check availability without importing torch/whisper into the main process.
# Either faster-whisper (preferred) or openai-whisper satisfies the requirement.
try:
    import importlib.util
    _spec_ow = importlib.util.find_spec("whisper")
    _spec_fw = importlib.util.find_spec("faster_whisper")
    WHISPER_AVAILABLE = _spec_ow is not None or _spec_fw is not None
except Exception:
    WHISPER_AVAILABLE = False

# WhisperX is a separate, heavier optional dependency used only for forced-
# alignment refinement (see whisper_worker.py:_run_whisperx_alignment) — it is
# NOT required for transcription itself, just for tighter word timestamps.
# find_spec() only locates the module; it never imports whisperx (or its own
# torch/pyannote dependencies) into the main process, keeping this check cheap
# and libomp-safe just like the WHISPER_AVAILABLE check above.
try:
    WHISPERX_AVAILABLE = importlib.util.find_spec("whisperx") is not None
except Exception:
    WHISPERX_AVAILABLE = False

_WORKER = Path(__file__).parent / "whisper_worker.py"


class TranscribeCancelled(RuntimeError):
    """Raised when transcription is cancelled via a cancel_event."""


# ── Filler word lists ───────────────────────────────────────────────────────

JAPANESE_FILLERS: set[str] = {
    # ── えー family (most common Japanese hesitation marker) ─────────────────
    "えー", "えっと", "えーと", "えーっと", "えぇ", "えと",
    # ── あの family ────────────────────────────────────────────────────────────
    "あの", "あのー", "あのう", "あのぉ",
    # ── まあ family ────────────────────────────────────────────────────────────
    "まあ", "まー", "まぁ", "まぁー",
    # ── あー / うー (pure hesitation — elongated vowels are unambiguous fillers)
    "あー", "あぁ", "ああ",
    "うー", "うぅ",
    # ── うーん (thinking/hesitation hmm — elongated form only; bare "うん" is
    #    the "yes" acknowledgment marker and must not be flagged)
    "うーん",
    # ── んー family (nasal hesitation — elongated/repeated form only; bare "ん"
    #    is excluded as it also serves as the "yes" acknowledgment marker)
    "んー", "んん",
    # ── そのー family — elongated form only; bare "その" is a real demonstrative
    "そのー", "そのう",
    # ── なんか family — hedge/filler use only; isolation check (≥200 ms audio
    #    silence on at least one side) prevents flagging meaningful "something"
    #    uses embedded in fast speech (e.g. "なんかあった" stays unflagged).
    "なんか", "なんかー",
}

ENGLISH_FILLERS: set[str] = {
    "um", "uh", "uhh", "umm",
    "hmm", "hm", "er", "err",
    "ah", "ahh",
    # ── discourse markers (flagged only when isolated — isolation check prevents
    #    flagging real uses embedded in sentences without adjacent pauses)
    "well",
    "like",
    "literally",
    "basically",
    "actually",
    "right",
}


# ── Main API ────────────────────────────────────────────────────────────────

def transcribe(
    input_path: Path,
    model_size: str = "base",
    language: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    cancel_event: Optional[threading.Event] = None,
    duration: Optional[float] = None,
) -> dict:
    """Transcribe audio with Whisper via a subprocess.

    Running Whisper in a subprocess keeps torch's libomp isolated from
    numpy's OpenBLAS libomp, preventing the SIGSEGV that occurs when both
    OMP runtimes are loaded in the same process on macOS Apple Silicon.

    Returns:
        {
            "segments": [{"start", "end", "text"}, ...],
            "words":    [{"word", "start", "end"}, ...],
            "language": detected language code,
        }

    Raises RuntimeError if whisper is not installed or transcription fails.
    """
    if not WHISPER_AVAILABLE:
        raise RuntimeError(
            "Neither faster-whisper nor openai-whisper is installed.\n"
            "Run: pip install faster-whisper  (recommended, 4–8× faster)\n"
            "  or: pip install openai-whisper"
        )

    if progress_callback:
        progress_callback("loading model")

    args_json = json.dumps({
        "path":       str(input_path),
        "model_size": model_size,
        "language":   language,
        "duration":   duration,
    })

    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}

    proc = subprocess.Popen(
        [sys.executable, str(_WORKER), args_json],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )

    # Drain both pipes in background threads.
    #
    # stdout: the worker writes a single JSON line (words + segments for the
    # whole file).  For a 26-min file this can easily reach 200-300 KB, well
    # above the OS pipe buffer (~64 KB on macOS/Linux).  If we wait for the
    # process to exit before reading stdout, the child blocks on its write
    # while the parent blocks on proc.wait() → deadlock.  Reading concurrently
    # avoids the buffer fill entirely.
    #
    # stderr: progress JSON lines + any error text.  Same buffer-fill risk for
    # very verbose output; already drained asynchronously.
    _stdout_chunks: list[str] = []

    def _drain_stdout() -> None:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            _stdout_chunks.append(chunk)

    _stderr_lines: list[str] = []

    def _drain_stderr() -> None:
        for line in proc.stderr:
            stripped = line.rstrip("\n")
            if progress_callback and stripped.startswith("{"):
                try:
                    data = json.loads(stripped)
                    pct  = data.get("progress")
                    if isinstance(pct, int):
                        progress_callback(pct)
                        continue   # don't include in error accumulator
                except (json.JSONDecodeError, ValueError):
                    pass
            _stderr_lines.append(stripped)

    stdout_thread = threading.Thread(target=_drain_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    # Allow 10× the audio duration, minimum 20 min.
    # large-v3-turbo: ~1–3× real-time on Apple Silicon MPS, ~5–10× on CPU.
    # 10× covers a 1-hour pre-edit clip even on pure CPU (~10 h worst case).
    # The cancel button is the user's escape hatch; this is just a hung-process
    # safety net so a zombie worker never runs forever.
    _timeout_s = max(1200, int((duration or 0) * 10))
    deadline = time.monotonic() + _timeout_s
    try:
        while True:
            try:
                proc.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                if cancel_event and cancel_event.is_set():
                    raise TranscribeCancelled()
                if time.monotonic() > deadline:
                    _timeout_min = _timeout_s // 60
                    raise RuntimeError(f"Whisper timed out (> {_timeout_min} minutes)")

        stdout_thread.join(timeout=10)
        stderr_thread.join(timeout=10)
        stdout = "".join(_stdout_chunks)
        stderr = "\n".join(_stderr_lines)

        if proc.returncode != 0:
            err = stderr.strip() or f"exit code {proc.returncode}"
            raise RuntimeError(f"Whisper worker failed: {err}")

        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Whisper worker returned invalid JSON: {exc}\n{stdout[:200]}")
    finally:
        # Guarantee the worker is terminated AND reaped on every exit path —
        # cancel, timeout, worker-error, bad JSON, or normal return (no-op when
        # it already exited).  A bare kill() without wait() leaves a zombie, and
        # an un-cancelled worker would keep burning CPU after the caller moved on.
        terminate_and_reap(proc)
        # Drain threads so pipe FDs close and buffered stderr isn't lost.
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)


def detect_fillers(
    words: list[dict],
    en: bool = True,
    ja: bool = True,
    custom: Optional[list[str]] = None,
    isolation_gap: float = 0.20,
    samples=None,           # np.ndarray — audio at 16 kHz float32
    sample_rate: int = 16000,
    threshold_db: float = -35.0,
) -> list[tuple[float, float, str]]:
    """Return (start, end, word) for each detected filler word.

    Isolation detection: a candidate word is only flagged if there is a pause
    on at least one side.  Two strategies are used depending on what is available:

    Audio-based (preferred): if `samples` is provided, the actual RMS energy in
    a window of `isolation_gap` seconds before and after the word is measured.
    A window whose average RMS is below `threshold_db` counts as a pause.
    This handles the common case where Whisper's word timestamps report 0 ms
    gaps even though real silence is present in the audio.

    Timestamp-based (fallback): if no samples are given, inter-word gaps from
    Whisper's word list are used.  These are often 0 ms due to quantisation, so
    this strategy catches fewer fillers.

    Matching is case-insensitive; trailing punctuation stripped.

    Multi-token Japanese grouping: faster-whisper tokenizes Japanese at the
    character level (e.g. 'えー' → 'え', 'ー').  A second pass combines up to
    5 consecutive tokens whose inter-token gaps are ≤ 100 ms and checks whether
    the composed string matches a Japanese filler.  Tokens already matched in
    the single-token pass are excluded.  Longest span wins.  The 100 ms limit
    matches group_word_tokens() so any filler that renders as one word is also
    detected.
    """
    import numpy as np

    filler_set: set[str] = set()
    if en:
        filler_set |= ENGLISH_FILLERS
    if ja:
        filler_set |= JAPANESE_FILLERS
    if custom:
        filler_set |= {w.lower().strip() for w in custom if w.strip()}

    use_audio = samples is not None and len(samples) > 0
    threshold_amp = 10.0 ** (threshold_db / 20.0)

    def _rms_window(t_start: float, t_end: float) -> float:
        i0 = max(0, int(t_start * sample_rate))
        i1 = min(len(samples), int(t_end * sample_rate))
        if i1 <= i0:
            return 0.0
        return float(np.sqrt(np.mean(samples[i0:i1] ** 2)))

    def _isolated(idx_start: int, idx_end: int, w_start: float, w_end: float) -> bool:
        """Return True if the span [w_start, w_end] has a pause on at least one side."""
        n = len(words)
        if use_audio:
            before_rms = _rms_window(w_start - isolation_gap, w_start)
            after_rms  = _rms_window(w_end,                   w_end + isolation_gap)
            return (idx_start == 0 or before_rms < threshold_amp) or \
                   (idx_end == n - 1 or after_rms < threshold_amp)
        else:
            gap_before = w_start - words[idx_start - 1]["end"] if idx_start > 0 else float("inf")
            gap_after  = words[idx_end + 1]["start"] - w_end   if idx_end < n - 1 else float("inf")
            return gap_before >= isolation_gap or gap_after >= isolation_gap

    matched: set[int] = set()
    results: list[tuple[float, float, str]] = []
    n = len(words)

    def _word_text(w: dict) -> str:
        """Return the word's display text regardless of whether it uses 'word' or 'text' key."""
        return (w.get("word") or w.get("text") or "").strip()

    # Pass 1: single-token matches (English + any full Japanese tokens)
    for i, w in enumerate(words):
        wt = _word_text(w)
        normalized = wt.lower().rstrip(".,!?、。")
        if normalized not in filler_set:
            continue
        if _isolated(i, i, w["start"], w["end"]):
            results.append((w["start"], w["end"], wt))
            matched.add(i)

    # Pass 2: multi-token Japanese composites (fastest-whisper char-level tokenization).
    # For each starting position, find the longest contiguous run (up to 5 tokens,
    # all inner gaps ≤ 100 ms) whose concatenated text is in JAPANESE_FILLERS.
    # 100 ms is intentionally more permissive than group_word_tokens (50 ms) so that
    # fillers with slightly large intra-token gaps (e.g. えーと from a slow speaker)
    # are still detected.  Any filler that group_word_tokens groups as one display word
    # (≤ 50 ms gaps) is a subset of what this pass detects.
    if ja:
        for i in range(n):
            if i in matched:
                continue
            # Build valid prefix lengths while consecutive gaps stay ≤ 100 ms
            valid_end_indices: list[int] = []
            for k in range(1, min(5, n - i)):
                if words[i + k]["start"] - words[i + k - 1]["end"] > 0.10:
                    break   # gap too large; no longer span possible from i
                valid_end_indices.append(i + k)
            if not valid_end_indices:
                continue
            # Try longest span first (prefer えーと over えー)
            for j in reversed(valid_end_indices):
                if any(m in matched for m in range(i, j + 1)):
                    continue  # don't overlap an already-matched token
                composed = "".join(_word_text(words[m]) for m in range(i, j + 1))
                if composed.lower().rstrip(".,!?、。") not in filler_set:
                    continue
                if _isolated(i, j, words[i]["start"], words[j]["end"]):
                    results.append((words[i]["start"], words[j]["end"], composed))
                    matched.update(range(i, j + 1))
                    break   # longest match consumed; move to next starting position

    return results
