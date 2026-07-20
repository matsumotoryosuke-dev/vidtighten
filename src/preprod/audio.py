"""Audio extraction and RMS-based silence detection."""

from __future__ import annotations

import bisect
import io
import logging
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_MS = 20
REFINE_WINDOW_MS = 10   # smaller window for precise boundary refinement

# ── detect_untranscribed_speech tunables ─────────────────────────────────────
# Exposed as module constants so they can be adjusted without hunting through
# the function body (eng-director recommendation, 2026-04).
#
# UTS = Untranscribed Speech
UTS_MIN_ISLAND_S       = 0.10   # minimum island duration to consider
                                 # (was 0.15 — lowered to catch 100–150ms
                                 #  Japanese hesitation particles like え/ん/あ)
UTS_MIN_ADJACENT_LARGE = 0.25   # the *larger* adjacent silence must be ≥ this
                                 # (one solid pause required on at least one side)
UTS_MIN_ADJACENT_SMALL = 0.15   # the *smaller* adjacent silence must be ≥ this
                                 # (floor that prevents mouth-clicks/breaths in
                                 #  narrow gaps from being flagged)

# Even a 6-hour file produces <500 MB of f32le audio at 16 kHz.
# If ffmpeg is still running after this, the process is stuck.
_FFMPEG_TIMEOUT_S = 600
_FFMPEG_NOT_FOUND     = "ffmpeg not found on PATH. Install ffmpeg from https://ffmpeg.org"
_FFMPEG_TIMEOUT_MSG   = (
    f"ffmpeg audio extraction timed out after {_FFMPEG_TIMEOUT_S // 60} minutes. "
    "The file may be on a slow/disconnected network drive, or ffmpeg encountered "
    "an unrecoverable codec hang. Try moving the file to a local drive."
)
_FFMPEG_NO_DETAILS    = (
    " (ffmpeg gave no details — file may be corrupted, "
    "DRM-protected, or an unsupported format)"
)


def terminate_and_reap(proc, *, term_timeout: float = 5.0) -> None:
    """Terminate a subprocess and REAP it so no zombie or orphan remains.

    Idempotent and safe when the process has already exited (no-op). Sends
    SIGTERM first, then SIGKILL, waiting after each so the OS releases the PID.
    Used on every cleanup/error path for the ffmpeg and Whisper-worker children
    (T0251) — a bare kill() without a following wait() leaves a zombie.
    """
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=term_timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.kill()
    try:
        proc.wait(timeout=term_timeout)
    except subprocess.TimeoutExpired:
        pass


def extract_audio(
    input_path: Path,
    sample_rate: int = SAMPLE_RATE,
    progress_callback: Optional[Callable[[float], None]] = None,
    total_duration: Optional[float] = None,
) -> np.ndarray:
    """Extract mono audio as float32 samples via ffmpeg pipe (no temp files).

    When ``progress_callback`` and ``total_duration`` are supplied, ffmpeg's
    ``-progress pipe:2`` output is parsed in a background thread and the
    callback is called with a 0–99 float as extraction progresses.  This is
    important for large 4K video files (10–50 GB) where extraction can take
    several minutes; without it the UI stalls silently at "extracting audio".

    When ``progress_callback`` is None (default), a simpler ``subprocess.run``
    path is used — identical behaviour to the original implementation.

    Memory note — streaming deferred by design
    -------------------------------------------
    A 30-minute 4K video at 16 kHz mono float32 produces ~28 M samples = ~110 MB.
    That is well within safe limits for the target hardware (M-series Mac, ≥16 GB RAM).
    Re-evaluate if files routinely exceed 2 h (>400 MB).
    """
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vn",              # skip video — reduces overhead for large 4K files
        "-map", "0:a:0",    # first audio stream explicitly
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
    ]

    if progress_callback is None:
        # Simple blocking path — existing callers and all tests use this.
        cmd += ["-v", "error", "pipe:1"]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, check=True, timeout=_FFMPEG_TIMEOUT_S,
            )
        except FileNotFoundError:
            raise FileNotFoundError(_FFMPEG_NOT_FOUND)
        except subprocess.TimeoutExpired:
            raise RuntimeError(_FFMPEG_TIMEOUT_MSG)
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr.decode(errors="replace").strip() if e.stderr else ""
            raise RuntimeError(
                f"ffmpeg audio extraction failed{': ' + stderr_msg if stderr_msg else _FFMPEG_NO_DETAILS}"
            )
        raw_bytes = proc.stdout
    else:
        # Progress-reporting path: ffmpeg emits key=value lines to stderr via
        # -progress pipe:2; we parse out_time_us to derive 0–99 % completion.
        cmd += ["-progress", "pipe:2", "-loglevel", "error", "pipe:1"]
        try:
            proc_p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise FileNotFoundError(_FFMPEG_NOT_FOUND)

        stdout_buf = io.BytesIO()
        stderr_acc: list[str] = []   # accumulate all stderr lines (errors + progress)
        total_dur_us = int(total_duration * 1_000_000) if total_duration else None

        def _drain_stdout() -> None:
            while True:
                chunk = proc_p.stdout.read(65536)
                if not chunk:
                    break
                stdout_buf.write(chunk)

        def _drain_stderr() -> None:
            last_pct = -1
            for raw in proc_p.stderr:
                line = raw.decode("utf-8", errors="replace").strip()
                stderr_acc.append(line)    # keep for error reporting
                if line.startswith("out_time_us=") and total_dur_us:
                    try:
                        time_us = int(line.split("=", 1)[1])
                        if time_us > 0:
                            pct = min(99, int(time_us / total_dur_us * 100))
                            if pct != last_pct:
                                progress_callback(float(pct))
                                last_pct = pct
                    except (ValueError, ZeroDivisionError):
                        pass

        t_out = threading.Thread(target=_drain_stdout, daemon=True)
        t_err = threading.Thread(target=_drain_stderr, daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc_p.wait(timeout=_FFMPEG_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise RuntimeError(_FFMPEG_TIMEOUT_MSG)
        finally:
            # Always reap ffmpeg — on timeout (wait raised) or any error below —
            # so a killed/abandoned ffmpeg never lingers as a zombie (T0251).
            terminate_and_reap(proc_p)
            t_out.join(timeout=10)
            t_err.join(timeout=10)

        if proc_p.returncode != 0:
            # Use lines accumulated by _drain_stderr — pipe is already at EOF.
            leftover = "\n".join(stderr_acc).strip()
            raise RuntimeError(
                f"ffmpeg audio extraction failed{': ' + leftover if leftover else _FFMPEG_NO_DETAILS}"
            )
        raw_bytes = stdout_buf.getvalue()

    if not raw_bytes:
        raise RuntimeError(
            f"ffmpeg produced no audio output for {input_path}. "
            "The file may have no readable audio track or be corrupted."
        )

    samples = np.frombuffer(raw_bytes, dtype=np.float32)
    log.info(
        "extract_audio: %d samples (%.1fs) from %s",
        len(samples), len(samples) / sample_rate, input_path.name,
    )
    return samples


def refine_word_boundary(
    samples: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    threshold_db: float = -35.0,
    search_s: float = 0.3,
    window_ms: int = REFINE_WINDOW_MS,
) -> tuple[float, float]:
    """Snap a word's (start, end) to the nearest audio energy transitions.

    Whisper's word timestamps have ~50–200ms jitter.  This function walks
    the actual audio to find:
      - refined_start: the frame right after the last *silent* frame before
        Whisper's start (i.e., where the word audio actually begins).
      - refined_end:   the frame right before the first *silent* frame after
        Whisper's end (i.e., where the word audio actually ends).

    If no silence is found within search_s seconds, the original value is
    kept.  The region is never collapsed to zero.
    """
    n = len(samples)
    if n == 0:
        return start, end

    window_size = max(1, int(sample_rate * window_ms / 1000))
    threshold_amp = 10.0 ** (threshold_db / 20.0)

    def _rms(frame: int) -> float:
        i0 = frame * window_size
        i1 = min(i0 + window_size, n)
        if i1 <= i0:
            return 0.0
        return float(np.sqrt(np.mean(samples[i0:i1] ** 2)))

    # ── refined start ────────────────────────────────────────────
    # Scan backward from Whisper start; find the last silent frame.
    # Refined start = one frame after that silent frame.
    start_frame = int(start * sample_rate) // window_size
    min_frame   = max(0, int((start - search_s) * sample_rate) // window_size)
    refined_start = start
    for f in range(start_frame, min_frame - 1, -1):
        if _rms(f) < threshold_amp:
            refined_start = (f + 1) * window_size / sample_rate
            break

    # ── refined end ──────────────────────────────────────────────
    # Scan forward from Whisper end; find the first silent frame.
    # Refined end = that silent frame's start time.
    end_frame   = int(end * sample_rate) // window_size
    max_frame   = min(
        n // window_size - 1,
        int((end + search_s) * sample_rate) // window_size,
    )
    refined_end = end
    for f in range(end_frame, max_frame + 1):
        if _rms(f) < threshold_amp:
            refined_end = f * window_size / sample_rate
            break

    # Safety: never collapse or invert the region
    if refined_end <= refined_start:
        return start, end

    return refined_start, refined_end


def detect_untranscribed_speech(
    total_duration: float,
    silence_regions: list[tuple[float, float]],
    transcript_segments: list[dict],
    max_duration: float = 1.2,
    min_adjacent_large: float = UTS_MIN_ADJACENT_LARGE,
    min_adjacent_small: float = UTS_MIN_ADJACENT_SMALL,
    min_island_duration: float = UTS_MIN_ISLAND_S,
    transcript_words: list[dict] | None = None,
) -> list[tuple[float, float]]:
    """Find short speech bursts that audio confirms exist but Whisper skipped.

    Whisper routinely ignores filler vocalizations (えー、うー、hmm、uh…)
    — they never appear in the word list at all.  This function finds them
    as *speech islands*: gaps between silence regions with no word coverage.

    Key design decisions:
    - `silence_regions` should come from a short-duration pass (min_duration≈0.1s,
      hangover_ms=0) so that brief pauses around fillers create proper island
      boundaries.  Using the coarse 1s-min pass merges fillers into large speech
      blocks that Whisper's segment timestamps then cover entirely.
    - Overlap is checked against *word*-level timestamps (transcript_words) when
      available.  Whisper's segment timestamps can span over untranscribed sounds,
      giving false "covered" results.  Word timestamps don't span them.
    - Adjacent-silence requirement is two-sided: the *larger* of the two flanking
      silences must be ≥ min_adjacent_large AND the *smaller* must be ≥
      min_adjacent_small.  This prevents mouth-clicks/breaths in narrow inter-word
      gaps from being flagged while still catching genuinely isolated vocalizations.
      (file-start and file-end count as infinite silence.)

    Returns list of (start_sec, end_sec) for each candidate.
    """
    if total_duration <= 0 or not silence_regions:
        return []

    silences = sorted(silence_regions, key=lambda x: x[0])

    # Build (start, end, sil_before_dur, sil_after_dur) for each speech island
    islands: list[tuple[float, float, float, float]] = []
    prev_end = 0.0
    prev_sil_dur = float("inf")   # treat file-start as infinite prior silence

    for s_start, s_end in silences:
        # 5 ms gap minimum: sub-5ms speech bursts are indistinguishable from
        # quantisation artefacts at 20ms frame boundaries and are not flagged.
        if s_start > prev_end + 0.005:
            sil_after_dur = s_end - s_start
            islands.append((prev_end, s_start, prev_sil_dur, sil_after_dur))
        prev_sil_dur = s_end - s_start
        prev_end = max(prev_end, s_end)

    # Speech after the last silence
    # Same 5 ms minimum applies to the trailing island.
    if prev_end < total_duration - 0.005:
        islands.append((prev_end, total_duration, prev_sil_dur, float("inf")))

    # Prefer word-level timestamps: a word "covers" an island only when its own
    # timestamps overlap the island.  Segment timestamps span entire sentences and
    # swallow the fillers that appear at sentence boundaries.
    if transcript_words is not None:
        lookup = sorted(transcript_words, key=lambda w: w["start"])
        _w_starts = [w["start"] for w in lookup]

        def overlaps_transcript(start: float, end: float) -> bool:
            # Binary-search to bound the scan to words near the island.
            # hi: first word starting at or after island end → skip all later words.
            # lo: words starting 1 s before island start → covers any word long
            #     enough to overlap (typical word duration ≤ 0.5 s; 1 s is safe).
            hi = bisect.bisect_left(_w_starts, end)
            lo = bisect.bisect_left(_w_starts, max(0.0, start - 1.0))
            for i in range(lo, hi):
                # 20 ms tolerance: a word that ends ≤20ms into the island is not
                # considered to "cover" it — that's within Whisper's timestamp jitter.
                if lookup[i]["end"] > start + 0.02:
                    return True
            return False
    else:
        trans = sorted(transcript_segments, key=lambda s: s["start"])
        _s_starts = [s["start"] for s in trans]

        def overlaps_transcript(start: float, end: float) -> bool:  # type: ignore[misc]
            # Segments can be long (up to 6 s after split_telop_segments); use 7 s lookback.
            hi = bisect.bisect_left(_s_starts, end)
            lo = bisect.bisect_left(_s_starts, max(0.0, start - 7.0))
            for i in range(lo, hi):
                if trans[i]["end"] > start + 0.02:
                    return True
            return False

    candidates = []
    n_rejected_duration   = 0
    n_rejected_transcript = 0
    n_rejected_isolation  = 0

    # 1 ns tolerance: frame timestamps are integer multiples of 0.02 s (20 ms),
    # and 0.02 is not exactly representable in binary float.  Multiplying by
    # integers accumulates a representational error (e.g. 110 * 0.02 − 50 * 0.02
    # = 1.2000000000000002, not 1.2).  The epsilon prevents a 60-frame island
    # from being incorrectly rejected when max_duration is exactly 1.2 s.
    _DUR_EPS = 1e-9
    for sp_start, sp_end, sil_before, sil_after in islands:
        duration = sp_end - sp_start
        if duration < min_island_duration - _DUR_EPS or duration > max_duration + _DUR_EPS:
            n_rejected_duration += 1
            continue
        if overlaps_transcript(sp_start, sp_end):
            n_rejected_transcript += 1
            continue
        # Two-sided isolation check: the larger adjacent silence must meet the
        # primary threshold; the smaller must clear a secondary floor.  This
        # prevents narrow inter-word clips (mouth clicks, breaths) from passing
        # while still catching vocalizations flanked by genuine pauses.
        sil_large = max(sil_before, sil_after)
        sil_small = min(sil_before, sil_after)
        if sil_large < min_adjacent_large or sil_small < min_adjacent_small:
            n_rejected_isolation += 1
            continue
        candidates.append((sp_start, sp_end))

    log.debug(
        "detect_untranscribed_speech: %d islands → %d candidates "
        "(rejected: %d duration, %d transcript-overlap, %d isolation)",
        len(islands),
        len(candidates),
        n_rejected_duration,
        n_rejected_transcript,
        n_rejected_isolation,
    )
    return candidates


def snap_silences_to_words(
    silence_regions: list[tuple[float, float]],
    words: list[dict],
    tolerance: float = 0.40,
) -> list[tuple[float, float]]:
    """Conservatively snap silence boundaries to Whisper word timestamps.

    Whisper timestamps have ~50–280 ms jitter.  A silence that ends at 58.38 s
    but Whisper places the next word at 58.10 s will appear to "eat" the first
    40 ms of that word in the UI — even though the silence is acoustically
    correct.

    This function reconciles the two sources of truth by only *shrinking*
    silence regions, never widening them:

    • END snapping: if a word starts within *tolerance* seconds BEFORE the
      silence end, snap the silence end to that word's start time.  The word
      is preserved and the silence region becomes slightly shorter.

    • START snapping: if a word ends within *tolerance* seconds AFTER the
      silence start, snap the silence start to that word's end time.  Again
      the region only shrinks.

    A region that would collapse to ≤ 0 s after snapping is dropped entirely.

    Args:
        silence_regions: list of (start_sec, end_sec) from detect_silence.
        words: list of dicts with "start" and "end" keys (Whisper word list).
        tolerance: maximum distance (seconds) a word timestamp can be from a
            silence boundary for snapping to occur.  Default 0.40 s is enough
            to cover Whisper's worst-case jitter while avoiding false snaps
            when the nearest word is genuinely far away.

    Returns:
        Filtered, snapped list of (start_sec, end_sec).
    """
    if not words or not silence_regions:
        return list(silence_regions)

    # Pre-sort by start so bisect can index into the word list directly.
    ws = sorted(words, key=lambda w: w["start"])
    word_starts = [w["start"] for w in ws]   # parallel list for bisect

    snapped: list[tuple[float, float]] = []
    for sil_start, sil_end in silence_regions:
        new_start = sil_start
        new_end   = sil_end

        # ── snap END to nearest word-start before silence-end ────────────
        # Find the maximum word-start in [sil_end - tolerance, sil_end).
        # bisect gives the half-open window directly; last element is the max.
        lo = bisect.bisect_left(word_starts, sil_end - tolerance)
        hi = bisect.bisect_left(word_starts, sil_end)
        if lo < hi:
            new_end = word_starts[hi - 1]

        # ── snap START to nearest word-end after silence-start ────────────
        # Find the minimum word-end in (sil_start, sil_start + tolerance].
        # ws is sorted by *start*, not end, so we scan a narrow start-window
        # (words starting ≤ 1 s before the silence can still end inside it).
        lo2 = bisect.bisect_left(word_starts, max(0.0, sil_start - 1.0))
        hi2 = bisect.bisect_right(word_starts, sil_start + tolerance)
        best_word_end: float | None = None
        for i in range(lo2, hi2):
            we = ws[i]["end"]
            if sil_start < we <= sil_start + tolerance:
                if best_word_end is None or we < best_word_end:
                    best_word_end = we
        if best_word_end is not None:
            new_start = best_word_end

        # Only keep regions that remain positive after snapping.
        if new_end > new_start:
            snapped.append((new_start, new_end))

    return snapped


def detect_silence(
    samples: np.ndarray,
    sample_rate: int,
    threshold_db: float,
    min_duration: float,
    window_ms: int = WINDOW_MS,
    hangover_ms: int = 300,
) -> list[tuple[float, float]]:
    """Detect silence regions using windowed RMS analysis.

    Returns list of (start_sec, end_sec) for each silence region
    that meets or exceeds min_duration.

    hangover_ms: after audio drops below threshold, continue treating it as
    speech for this many ms.  This prevents the detector from triggering on
    the natural trailing-edge decay of words (e.g. Japanese "ます", "です",
    "す" are soft and fade out before the actual pause begins).
    """
    window_size = int(sample_rate * window_ms / 1000)
    if window_size == 0 or len(samples) < window_size:
        raise ValueError("Audio too short to analyze")

    n_frames = len(samples) // window_size
    trimmed = samples[: n_frames * window_size].reshape(n_frames, window_size)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1))

    threshold_amp = 10.0 ** (threshold_db / 20.0)
    is_silent = rms < threshold_amp

    # Speech hangover: extend each speech region forward by hangover_ms.
    # Any frame within hangover_frames of a preceding speech frame is kept
    # as speech, shifting the silence-start point past word trailing edges.
    hangover_frames = max(0, int(hangover_ms / window_ms))
    if hangover_frames > 0:
        is_speech = (~is_silent).astype(np.uint8)
        kernel = np.ones(hangover_frames + 1, dtype=np.uint8)
        extended = np.convolve(is_speech, kernel, mode='full')[: len(is_speech)]
        is_silent = extended == 0

    # Vectorised region extraction — ~37× faster than a Python for-loop.
    # np.diff with prepend=0,append=0 finds every silence-start (+1) and
    # silence-end (−1) in a single pass; slicing by min_duration then
    # produces the final list without any Python-level iteration.
    frame_duration = window_ms / 1000.0
    d = np.diff(is_silent.astype(np.int8), prepend=0, append=0)
    # Shape of d: len(is_silent) + 1 (one extra element from append)
    starts_f = np.where(d == 1)[0]   # frame indices where silence begins
    ends_f   = np.where(d == -1)[0]  # frame indices where silence ends
    durs     = (ends_f - starts_f) * frame_duration
    mask     = durs >= min_duration
    return list(zip(
        (starts_f[mask] * frame_duration).tolist(),
        (ends_f[mask]   * frame_duration).tolist(),
    ))
