"""Tests for audio.py — silence detection with hangover, boundary refinement."""

import subprocess
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from preprod.audio import detect_silence, extract_audio, refine_word_boundary, detect_untranscribed_speech, snap_silences_to_words

SAMPLE_RATE = 16000
WINDOW_MS = 20


class TestExtractAudio:
    """Unit tests for the extract_audio() ffmpeg wrapper."""

    def test_ffmpeg_timeout_raises_runtime_error(self, tmp_path):
        """subprocess.TimeoutExpired is converted to a descriptive RuntimeError."""
        fake = tmp_path / "video.mp4"
        fake.write_bytes(b"\x00" * 100)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=600)):
            with pytest.raises(RuntimeError, match="timed out"):
                extract_audio(fake)

    def test_ffmpeg_not_found_raises_file_not_found(self, tmp_path):
        """Missing ffmpeg binary surfaces a FileNotFoundError with install hint."""
        fake = tmp_path / "video.mp4"
        fake.write_bytes(b"\x00" * 100)
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError, match="ffmpeg not found"):
                extract_audio(fake)

    def test_ffmpeg_nonzero_exit_raises_runtime_error(self, tmp_path):
        """A non-zero ffmpeg exit code is surfaced as RuntimeError."""
        fake = tmp_path / "video.mp4"
        fake.write_bytes(b"\x00" * 100)
        err = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"Invalid data found")
        with patch("subprocess.run", side_effect=err):
            with pytest.raises(RuntimeError, match="ffmpeg audio extraction failed"):
                extract_audio(fake)

    def test_empty_stdout_raises_runtime_error(self, tmp_path):
        """ffmpeg exit 0 but empty stdout is treated as an extraction failure."""
        fake = tmp_path / "video.mp4"
        fake.write_bytes(b"\x00" * 100)
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
        with patch("subprocess.run", return_value=result):
            with pytest.raises(RuntimeError, match="no audio output"):
                extract_audio(fake)

    def test_valid_output_returns_float32_array(self, tmp_path):
        """Valid f32le stdout is returned as a float32 numpy array."""
        fake = tmp_path / "video.mp4"
        fake.write_bytes(b"\x00" * 100)
        raw = np.ones(1600, dtype=np.float32).tobytes()
        result = subprocess.CompletedProcess(args=[], returncode=0, stdout=raw, stderr=b"")
        with patch("subprocess.run", return_value=result):
            samples = extract_audio(fake)
        assert samples.dtype == np.float32
        assert len(samples) == 1600

    # ── Progress-callback (Popen) path ────────────────────────────────────────

    @staticmethod
    def _make_popen_mock(pcm_bytes: bytes, stderr_lines: list[bytes], returncode: int = 0):
        """Build a Popen mock that streams pcm_bytes on stdout and stderr_lines on stderr.

        stderr is a MagicMock whose __iter__ yields the given byte lines AND
        whose .read() is also available (needed by the non-zero exit code path).
        """
        stdout_mock = MagicMock()
        stdout_mock.read.side_effect = [pcm_bytes, b""]  # data then EOF
        stderr_mock = MagicMock()
        stderr_mock.__iter__ = MagicMock(return_value=iter(stderr_lines))
        stderr_mock.read.return_value = b""  # leftover after iteration (overridable)
        proc = MagicMock()
        proc.stdout = stdout_mock
        proc.stderr = stderr_mock
        proc.returncode = returncode
        proc.wait.return_value = None
        return proc

    def test_progress_callback_called_with_increasing_values(self, tmp_path):
        """progress_callback receives increasing 0-99 float values as ffmpeg progresses."""
        fake = tmp_path / "video.mp4"
        fake.write_bytes(b"\x00" * 100)
        raw_pcm = np.ones(1600, dtype=np.float32).tobytes()
        stderr_lines = [
            b"out_time_us=1000000\n",   # 1s of 10s → 10%
            b"out_time_us=5000000\n",   # 5s of 10s → 50%
            b"out_time_us=9000000\n",   # 9s of 10s → 90%
            b"progress=end\n",          # non-progress line — ignored
        ]
        proc_mock = self._make_popen_mock(raw_pcm, stderr_lines)
        received: list[float] = []
        with patch("subprocess.Popen", return_value=proc_mock):
            samples = extract_audio(fake, progress_callback=received.append, total_duration=10.0)
        assert samples.dtype == np.float32
        assert len(samples) == 1600
        assert received == sorted(received), "Callback values must be non-decreasing"
        assert 10.0 in received
        assert 50.0 in received
        assert 90.0 in received

    def test_progress_path_nonzero_exit_raises_runtime_error(self, tmp_path):
        """Non-zero Popen returncode raises RuntimeError with the stderr error text."""
        fake = tmp_path / "video.mp4"
        fake.write_bytes(b"\x00" * 100)
        # Stderr includes a mix of -progress lines and a real error message.
        stderr_with_error = [
            b"frame=0 fps=0.0 q=0.0 size=0kB time=00:00:00.00\n",
            b"Invalid data found when processing input\n",
        ]
        proc_mock = self._make_popen_mock(b"", stderr_with_error, returncode=1)
        with patch("subprocess.Popen", return_value=proc_mock):
            with pytest.raises(RuntimeError, match="Invalid data found"):
                extract_audio(fake, progress_callback=lambda _: None, total_duration=10.0)


def _make_samples(timeline: list[tuple[float, float, float]], total: float) -> np.ndarray:
    """Build a synthetic audio array from (start, end, amplitude) segments.

    Regions not covered by any segment are silent (amplitude 0).
    """
    n = int(total * SAMPLE_RATE)
    samples = np.zeros(n, dtype=np.float32)
    for start, end, amp in timeline:
        i0 = int(start * SAMPLE_RATE)
        i1 = int(end * SAMPLE_RATE)
        samples[i0:i1] = amp
    return samples


class TestDetectSilenceBasic:
    """Basic detection — no hangover (hangover_ms=0)."""

    def _detect(self, samples, threshold_db=-30.0, min_duration=0.5):
        return detect_silence(samples, SAMPLE_RATE, threshold_db, min_duration, hangover_ms=0)

    def test_fully_silent_returns_one_region(self):
        samples = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
        regions = self._detect(samples, min_duration=0.5)
        assert len(regions) == 1
        assert regions[0][0] == pytest.approx(0.0, abs=0.05)
        assert regions[0][1] == pytest.approx(3.0, abs=0.05)

    def test_continuous_speech_returns_no_regions(self):
        amp = 10.0 ** (-20.0 / 20.0)  # −20 dB — above −30 dB threshold
        samples = np.full(SAMPLE_RATE * 3, amp, dtype=np.float32)
        regions = self._detect(samples)
        assert regions == []

    def test_single_silence_gap_detected(self):
        amp = 10.0 ** (-20.0 / 20.0)
        # Speech 0–1s, silence 1–2.5s, speech 2.5–4s
        samples = _make_samples([(0, 1.0, amp), (2.5, 4.0, amp)], total=4.0)
        regions = self._detect(samples, min_duration=0.5)
        assert len(regions) == 1
        s, e = regions[0]
        assert s == pytest.approx(1.0, abs=0.05)
        assert e == pytest.approx(2.5, abs=0.05)

    def test_short_silence_below_min_duration_skipped(self):
        amp = 10.0 ** (-20.0 / 20.0)
        # Speech, 0.3s silence (< 0.5s min), speech
        samples = _make_samples([(0, 1.0, amp), (1.3, 3.0, amp)], total=3.0)
        regions = self._detect(samples, min_duration=0.5)
        assert regions == []

    def test_multiple_silence_regions(self):
        amp = 10.0 ** (-20.0 / 20.0)
        samples = _make_samples([
            (0.0, 1.0, amp),
            (2.0, 3.0, amp),
            (4.0, 5.0, amp),
        ], total=5.0)
        regions = self._detect(samples, min_duration=0.5)
        assert len(regions) == 2

    def test_trailing_silence_detected(self):
        amp = 10.0 ** (-20.0 / 20.0)
        samples = _make_samples([(0.0, 1.0, amp)], total=3.0)
        regions = self._detect(samples, min_duration=0.5)
        assert len(regions) == 1
        assert regions[0][0] == pytest.approx(1.0, abs=0.05)

    def test_empty_samples_raises_value_error(self):
        with pytest.raises(ValueError, match="Audio too short to analyze"):
            detect_silence(np.array([], dtype=np.float32), SAMPLE_RATE, -30, 0.5)

    def test_threshold_respected(self):
        # Amplitude at exactly −25 dB
        amp = 10.0 ** (-25.0 / 20.0)
        samples = np.full(SAMPLE_RATE * 2, amp, dtype=np.float32)
        # With −20 dB threshold: signal is below threshold → all silent
        regions_strict = self._detect(samples, threshold_db=-20.0, min_duration=0.5)
        # With −30 dB threshold: signal is above threshold → no silence
        regions_loose  = self._detect(samples, threshold_db=-30.0, min_duration=0.5)
        assert len(regions_strict) == 1
        assert regions_loose == []


class TestDetectSilenceHangover:
    """Hangover prevents the detector from firing on word trailing edges."""

    def _detect(self, samples, threshold_db=-30.0, min_duration=0.5, hangover_ms=300):
        return detect_silence(samples, SAMPLE_RATE, threshold_db, min_duration,
                              hangover_ms=hangover_ms)

    def test_hangover_shifts_silence_start(self):
        """Audio drops below threshold 200ms before word ends — hangover of 300ms
        should push the silence-start past the word end."""
        amp = 10.0 ** (-20.0 / 20.0)   # loud speech
        fade = 10.0 ** (-40.0 / 20.0)  # below −35 dB threshold (word trailing edge)

        # Word at full volume 0–1.8s, then fades 1.8–2.0s, then real silence 2.0–5.0s
        samples = _make_samples([
            (0.0, 1.8, amp),
            (1.8, 2.0, fade),   # trailing edge — below threshold but still "word"
        ], total=5.0)

        regions_no_ho = self._detect(samples, threshold_db=-35.0, min_duration=0.5, hangover_ms=0)
        regions_ho    = self._detect(samples, threshold_db=-35.0, min_duration=0.5, hangover_ms=300)

        # Without hangover: silence starts at 1.8s (during trailing edge)
        assert regions_no_ho[0][0] == pytest.approx(1.8, abs=0.05)

        # With hangover: silence start pushed forward by ~300ms → ≥ 2.0s
        assert regions_ho[0][0] >= 1.95

    def test_hangover_does_not_affect_clearly_separated_speech(self):
        """A long clean silence is still detected correctly with hangover enabled."""
        amp = 10.0 ** (-20.0 / 20.0)
        # Speech 0–1s, 3-second silence, speech 4–6s
        samples = _make_samples([(0.0, 1.0, amp), (4.0, 6.0, amp)], total=6.0)
        regions = self._detect(samples, threshold_db=-30.0, min_duration=0.5, hangover_ms=300)
        assert len(regions) == 1
        # Silence start shifts forward 300ms, but it's still a long silence
        s, e = regions[0]
        assert s >= 0.95                  # shifted forward from 1.0 by hangover
        assert s <= 1.35                  # no more than 300ms shift
        assert e == pytest.approx(4.0, abs=0.05)

    def test_hangover_zero_equals_no_hangover(self):
        """hangover_ms=0 should behave identically to the baseline."""
        amp = 10.0 ** (-20.0 / 20.0)
        samples = _make_samples([(0.0, 1.0, amp), (2.0, 4.0, amp)], total=4.0)
        r0 = detect_silence(samples, SAMPLE_RATE, -30.0, 0.5, hangover_ms=0)
        r1 = detect_silence(samples, SAMPLE_RATE, -30.0, 0.5, hangover_ms=0)
        assert r0 == r1

    def test_trailing_edge_shorter_than_hangover_removed(self):
        """A 100ms trailing edge with 300ms hangover: the entire trailing dip
        disappears into the hangover window — silence starts at the real end."""
        amp  = 10.0 ** (-20.0 / 20.0)
        fade = 10.0 ** (-40.0 / 20.0)
        # Word 0–2.0s, 100ms fade 2.0–2.1s, silence 2.1–5.0s
        samples = _make_samples([
            (0.0, 2.0, amp),
            (2.0, 2.1, fade),
        ], total=5.0)
        regions = self._detect(samples, threshold_db=-35.0, min_duration=0.5, hangover_ms=300)
        assert len(regions) == 1
        # With 300ms hangover starting at 2.0s → silence start pushed to ~2.3s
        assert regions[0][0] >= 1.95


class TestRefineWordBoundary:
    """refine_word_boundary snaps Whisper timestamps to audio energy transitions."""

    THR = -30.0  # threshold dB used throughout
    AMP = 10.0 ** (-20.0 / 20.0)  # −20 dB — well above threshold

    def _refine(self, samples, start, end, threshold_db=None, search_s=0.3):
        thr = threshold_db if threshold_db is not None else self.THR
        return refine_word_boundary(samples, SAMPLE_RATE, start, end,
                                    threshold_db=thr, search_s=search_s)

    def test_perfect_timestamps_unchanged(self):
        """If Whisper timestamps already sit exactly at silence boundaries,
        refinement should return values very close to the originals."""
        # Silence 0–1s, word 1–1.5s, silence 1.5–3s
        samples = _make_samples([(1.0, 1.5, self.AMP)], total=3.0)
        rs, re = self._refine(samples, 1.0, 1.5)
        assert rs == pytest.approx(1.0, abs=0.03)
        assert re == pytest.approx(1.5, abs=0.03)

    def test_start_snapped_earlier_when_whisper_is_late(self):
        """Whisper start is 150ms into the word — refinement pulls it back."""
        # Word actually starts at 1.0s, Whisper says 1.15s
        samples = _make_samples([(1.0, 1.5, self.AMP)], total=3.0)
        rs, re = self._refine(samples, start=1.15, end=1.5)
        assert rs < 1.15   # snapped earlier
        assert rs == pytest.approx(1.0, abs=0.03)

    def test_end_snapped_later_when_whisper_is_early(self):
        """Whisper end is 150ms before the word finishes — refinement extends it."""
        # Word actually ends at 1.5s, Whisper says 1.35s
        samples = _make_samples([(1.0, 1.5, self.AMP)], total=3.0)
        rs, re = self._refine(samples, start=1.0, end=1.35)
        assert re > 1.35   # snapped later
        assert re == pytest.approx(1.5, abs=0.03)

    def test_both_boundaries_refined_simultaneously(self):
        """Both start and end are off; both get corrected."""
        # Word 1.0–1.6s, Whisper says 1.15–1.45
        samples = _make_samples([(1.0, 1.6, self.AMP)], total=3.0)
        rs, re = self._refine(samples, start=1.15, end=1.45)
        assert rs < 1.15
        assert re > 1.45

    def test_never_collapses_region(self):
        """If refinement can't find silence, original bounds are preserved."""
        # Continuous speech — no silence near the word at all
        samples = np.full(SAMPLE_RATE * 3, self.AMP, dtype=np.float32)
        rs, re = self._refine(samples, start=1.0, end=1.5)
        assert rs <= re   # must not invert
        # Can't find silence, so originals are returned
        assert rs == pytest.approx(1.0, abs=0.05)
        assert re == pytest.approx(1.5, abs=0.05)

    def test_empty_samples_returns_original(self):
        """Empty audio array doesn't crash — returns (start, end) unchanged."""
        rs, re = self._refine(np.array([], dtype=np.float32), 1.0, 1.5)
        assert rs == 1.0
        assert re == 1.5

    def test_search_window_limits_how_far_refinement_travels(self):
        """With a tiny search window, refinement stops early."""
        # Word 1.0–1.5s, preceding silence from 0.0–1.0s
        samples = _make_samples([(1.0, 1.5, self.AMP)], total=3.0)
        # search_s=0.05 (50ms) — can barely reach back to 0.95s
        rs_narrow, _ = self._refine(samples, start=1.3, end=1.5, search_s=0.05)
        rs_wide,   _ = self._refine(samples, start=1.3, end=1.5, search_s=0.5)
        # Wide search reaches further back
        assert rs_wide <= rs_narrow


class TestDetectUntranscribedSpeech:
    """detect_untranscribed_speech finds speech islands Whisper skipped."""

    def _run(self, total, silence_regions, segments, **kw):
        return detect_untranscribed_speech(total, silence_regions, segments, **kw)

    def test_basic_missed_vocalization(self):
        """Short speech island with no transcript segment is flagged."""
        # Silence 0–1s, vocalization 1–1.4s, silence 1.4–2s, speech 2–4s
        silence = [(0.0, 1.0), (1.4, 2.0)]
        segments = [{"start": 2.0, "end": 4.0}]
        result = self._run(4.0, silence, segments)
        assert len(result) == 1
        s, e = result[0]
        assert s == pytest.approx(1.0, abs=0.02)
        assert e == pytest.approx(1.4, abs=0.02)

    def test_transcribed_island_not_flagged(self):
        """Island that overlaps a Whisper segment is ignored."""
        silence = [(0.0, 1.0), (2.0, 3.0)]
        segments = [{"start": 1.0, "end": 2.0}]
        result = self._run(3.0, silence, segments)
        assert result == []

    def test_long_island_not_flagged(self):
        """Islands longer than max_duration are not filler candidates."""
        silence = [(0.0, 0.5), (3.0, 4.0)]
        segments = []
        # Island is 2.5s long — too long to be a filler
        result = self._run(4.0, silence, segments, max_duration=1.2)
        assert result == []

    def test_multiple_missed_vocalizations(self):
        """Multiple short untranscribed islands are all returned."""
        silence = [(0.0, 1.0), (1.3, 2.0), (2.4, 3.0), (3.3, 4.0)]
        # No transcript at all
        result = self._run(4.0, silence, segments=[])
        assert len(result) == 3   # islands: 1–1.3, 2–2.4, 3–3.3

    def test_empty_silence_returns_empty(self):
        """No silence regions → no islands → empty result."""
        result = self._run(3.0, silence_regions=[], segments=[])
        assert result == []

    def test_zero_duration_returns_empty(self):
        result = self._run(0.0, [(0.0, 1.0)], [])
        assert result == []

    def test_tiny_sliver_below_min_duration_skipped(self):
        """Islands < UTS_MIN_ISLAND_S (100ms) are noise artefacts — ignored."""
        silence = [(0.0, 1.0), (1.02, 2.0)]   # 20ms island
        result = self._run(2.0, silence, segments=[])
        assert result == []

    def test_insufficient_adjacent_silence_skipped(self):
        """Island whose surrounding silences are both too short is ignored."""
        # Two short silences (50ms each) surround the target island.
        # Cover the leading island with a transcript segment so it's excluded.
        # Set total_duration = end of last silence so no trailing island forms.
        silence = [(0.5, 0.55), (0.9, 0.95)]
        # Island (0.0, 0.5): overlaps transcript → excluded
        # Island (0.55, 0.9): sil_before=0.05, sil_after=0.05 →
        #   max(0.05, 0.05)=0.05 < min_adjacent_large=0.1 → excluded
        segments = [{"start": 0.0, "end": 0.5}]
        result = self._run(0.95, silence, segments,
                           min_adjacent_large=0.1, min_adjacent_small=0.0)
        assert result == []

    def test_sufficient_silence_on_both_sides_passes(self):
        """Island with long silence on one side and ≥floor on the other passes."""
        # sil_before = 1.0s, sil_after = 0.20s; both meet their respective thresholds.
        silence = [(0.0, 1.0), (1.4, 1.60)]
        result = self._run(1.60, silence, segments=[])
        # Island (1.0, 1.4): max=1.0 ≥ 0.25, min=0.20 ≥ 0.15 → passes
        assert len(result) == 1

    def test_small_side_below_floor_rejected(self):
        """Island with large silence on one side but tiny silence on the other is rejected.

        This is the new two-sided isolation requirement (eng-director, 2026-04):
        having one long pause is no longer sufficient if the other side is < 0.15s,
        because mouth-clicks and breaths in narrow inter-word gaps would pass.
        """
        # sil_before = 1.0s (large side OK), sil_after = 0.08s (below 0.15s floor)
        silence = [(0.0, 1.0), (1.4, 1.48)]
        result = self._run(1.48, silence, segments=[])
        # Island (1.0, 1.4): max=1.0 ≥ 0.25 ✓, min=0.08 < 0.15 ✗ → rejected
        assert result == []

    def test_island_exactly_at_max_duration_is_kept(self):
        """Island whose duration equals max_duration (within float epsilon) is kept.

        Frame timestamps are integer multiples of 0.02 s (20 ms window), but 0.02
        is not exactly representable in binary float.  E.g. 110 * 0.02 − 50 * 0.02
        = 1.2000000000000002, not 1.2.  The 1 ns epsilon in the duration filter
        ensures such boundary islands are correctly accepted.
        """
        # Island (1.0, 2.2) = 1.2000000000000002s in float, max_duration=1.2.
        # Before the epsilon fix this was incorrectly rejected.
        silence = [(0.0, 1.0), (2.2, 3.0)]
        result = self._run(3.0, silence, segments=[], max_duration=1.2)
        assert len(result) == 1

    def test_island_just_above_max_duration_rejected(self):
        """Island clearly over max_duration is rejected."""
        # Island (1.0, 2.5) = 1.5s > max_duration=1.2 → rejected
        silence = [(0.0, 1.0), (2.5, 4.0)]
        result = self._run(4.0, silence, segments=[], max_duration=1.2)
        assert result == []

    def test_island_exactly_at_min_duration_is_kept(self):
        """Island whose duration equals min_island_duration (within float epsilon) is kept.

        Python's float arithmetic: 0.3 - 0.2 = 0.09999999999999998, which is
        strictly less than 0.1 (= UTS_MIN_ISLAND_S).  This mirrors real-world
        frame timestamps (frame 10 → 0.2 s, frame 15 → 0.3 s via n * 0.02).
        Without the 1 ns epsilon the island would be incorrectly rejected.
        """
        # Island (0.2, 0.3): duration = 0.3 - 0.2 = 0.09999999999999998 < 0.1 in float.
        # The epsilon fix ensures 0.09999999999999998 >= 0.1 - 1e-9 → island is kept.
        silence = [(0.0, 0.2), (0.3, 1.5)]
        result = self._run(1.5, silence, segments=[])
        assert len(result) == 1

    def test_word_ending_just_outside_jitter_does_not_cover_island(self):
        """Word ending at island_start + 0.01 (< 0.02 jitter threshold) does NOT cover it.

        The overlaps_transcript check uses `end > start + 0.02` so that words
        whose timestamps barely reach into an island are treated as Whisper jitter.
        """
        # Island (1.0, 1.3): word ends at 1.01 — only 10ms into the island
        # 1.01 <= 1.0 + 0.02 = 1.02 → NOT covered → island should be flagged
        words = [{"word": "x", "start": 0.5, "end": 1.01}]
        silence = [(0.0, 1.0), (1.3, 2.0)]
        result = detect_untranscribed_speech(
            total_duration=2.0,
            silence_regions=silence,
            transcript_segments=[],
            transcript_words=words,
        )
        # Island (1.0, 1.3) is 0.3s, well-isolated, word only 10ms overlap → flagged
        assert len(result) == 1

    def test_word_ending_past_jitter_covers_island(self):
        """Word ending at island_start + 0.03 (> 0.02 jitter) DOES cover the island."""
        # Island (1.0, 1.3): word ends at 1.03 — 30ms into the island
        # 1.03 > 1.0 + 0.02 = 1.02 → covered → island should NOT be flagged
        words = [{"word": "x", "start": 0.5, "end": 1.03}]
        silence = [(0.0, 1.0), (1.3, 2.0)]
        result = detect_untranscribed_speech(
            total_duration=2.0,
            silence_regions=silence,
            transcript_segments=[],
            transcript_words=words,
        )
        assert result == []


class TestSnapSilencesToWords:
    """snap_silences_to_words conservatively snaps boundaries to word timestamps."""

    # Helper: build a word dict
    @staticmethod
    def _w(text: str, start: float, end: float) -> dict:
        return {"word": text, "start": start, "end": end}

    # ── END snapping ────────────────────────────────────────────────────────

    def test_end_snapped_to_preceding_word_start(self):
        """Classic Whisper jitter: silence ends at 58.38, word starts at 58.10.
        Silence end should snap left to 58.10 (word is preserved)."""
        silences = [(54.76, 58.38)]
        words = [self._w("はい", 58.10, 58.50)]
        result = snap_silences_to_words(silences, words)
        assert len(result) == 1
        s, e = result[0]
        assert s == pytest.approx(54.76)
        assert e == pytest.approx(58.10)

    def test_end_not_snapped_when_word_outside_tolerance(self):
        """Word starts 0.60s before silence end — beyond default 0.40s tolerance."""
        silences = [(10.0, 15.0)]
        words = [self._w("foo", 14.35, 15.2)]   # 15.0 - 14.35 = 0.65 > 0.40
        result = snap_silences_to_words(silences, words)
        assert len(result) == 1
        s, e = result[0]
        assert e == pytest.approx(15.0)   # unchanged

    def test_end_snapped_when_word_just_within_tolerance(self):
        """Word starts exactly at tolerance boundary → snapped."""
        silences = [(10.0, 15.0)]
        words = [self._w("foo", 14.65, 15.2)]   # 15.0 - 14.65 = 0.35 < 0.40
        result = snap_silences_to_words(silences, words)
        s, e = result[0]
        assert e == pytest.approx(14.65)

    def test_end_picks_closest_word_when_multiple_candidates(self):
        """Two words both start inside the silence end's tolerance window.
        The one whose start is closest (latest) to the silence end is chosen."""
        silences = [(10.0, 15.0)]
        # Both 14.70 and 14.85 are within 0.40 of 15.0; 14.85 is closer.
        words = [
            self._w("a", 14.70, 15.1),
            self._w("b", 14.85, 15.3),
        ]
        result = snap_silences_to_words(silences, words)
        s, e = result[0]
        assert e == pytest.approx(14.85)

    # ── START snapping ──────────────────────────────────────────────────────

    def test_start_snapped_to_following_word_end(self):
        """Silence starts at 5.00, but preceding word ends at 5.22 (overrun).
        Silence start snaps right to 5.22."""
        silences = [(5.00, 8.00)]
        words = [self._w("ます", 4.70, 5.22)]
        result = snap_silences_to_words(silences, words)
        s, e = result[0]
        assert s == pytest.approx(5.22)
        assert e == pytest.approx(8.00)

    def test_start_not_snapped_when_word_end_outside_tolerance(self):
        """Word end is 0.50s past silence start — beyond default 0.40s tolerance."""
        silences = [(5.00, 8.00)]
        words = [self._w("foo", 4.0, 5.55)]  # 5.55 - 5.00 = 0.55 > 0.40
        result = snap_silences_to_words(silences, words)
        s, e = result[0]
        assert s == pytest.approx(5.00)   # unchanged

    # ── Region preservation / collapse ────────────────────────────────────

    def test_region_dropped_if_start_snaps_past_end(self):
        """If snapping collapses a region to ≤ 0 s, it is removed entirely."""
        # 200ms silence [5.00, 5.20]; word ends at 5.15 (start snap)
        # AND another word starts at 5.05 (end snap).
        # After snapping: start=5.15, end=5.05 → inverted → dropped.
        silences = [(5.00, 5.20)]
        words = [
            self._w("A", 4.80, 5.15),  # end=5.15, within 0.40 of start 5.00
            self._w("B", 5.05, 5.40),  # start=5.05, within 0.40 of end 5.20
        ]
        result = snap_silences_to_words(silences, words)
        assert result == []

    def test_multiple_silences_processed_independently(self):
        """Each silence region is snapped independently."""
        silences = [(2.0, 4.0), (6.0, 8.0)]
        words = [
            self._w("A", 3.85, 4.3),   # end=4.3 → snaps end of first silence 4.0→3.85
            # Word B ends at 6.50 — that's 0.50s past silence start (6.0),
            # which exceeds the default 0.40s tolerance, so no start snap.
            self._w("B", 5.85, 6.50),
        ]
        result = snap_silences_to_words(silences, words)
        assert len(result) == 2
        s0, e0 = result[0]
        assert e0 == pytest.approx(3.85)   # first silence end snapped
        s1, e1 = result[1]
        assert s1 == pytest.approx(6.0)    # second silence start unchanged (word B end outside tolerance)

    # ── Edge cases ────────────────────────────────────────────────────────

    def test_empty_words_returns_silences_unchanged(self):
        silences = [(1.0, 3.0), (5.0, 7.0)]
        result = snap_silences_to_words(silences, [])
        assert result == silences

    def test_empty_silences_returns_empty(self):
        words = [self._w("hello", 1.0, 1.5)]
        assert snap_silences_to_words([], words) == []

    def test_both_empty(self):
        assert snap_silences_to_words([], []) == []

    def test_word_after_silence_does_not_trigger_end_snap(self):
        """A word that starts AFTER the silence end should not affect end snapping."""
        silences = [(2.0, 4.0)]
        words = [self._w("after", 4.5, 5.0)]
        result = snap_silences_to_words(silences, words)
        s, e = result[0]
        assert e == pytest.approx(4.0)   # unchanged


# ── Bisect optimization regression tests ────────────────────────────────────

class TestSnapSilencesToWordsBisect:
    """snap_silences_to_words gives correct results with a large word list
    (bisect implementation must jump past many irrelevant words)."""

    @staticmethod
    def _w(text: str, start: float, end: float) -> dict:
        return {"word": text, "start": start, "end": end}

    def test_end_snap_with_many_preceding_words(self):
        """Silence near end of a 100-word list snaps to the nearby word-start."""
        # 100 words at 0..99, each 0.3 s; then a silence at [99.8, 100.5]
        # Word[99] starts at 99.0 → within tolerance (0.4 s) of silence end 100.5
        words = [self._w(f"w{i}", float(i), float(i) + 0.3) for i in range(100)]
        silences = [(99.8, 100.5)]
        result = snap_silences_to_words(silences, words, tolerance=0.40)
        assert len(result) == 1
        # end should snap to 99.0 (closest word-start before 100.5, within tolerance 0.4)
        # 100.5 - 0.4 = 100.1; word starts in [100.1, 100.5): none (last start is 99.0 < 100.1)
        # So no snap should occur; end stays 100.5
        # But wait: 100.5 - 99.0 = 1.5 > 0.4, so word[99] is NOT in the tolerance window
        # → end unchanged = 100.5, start unchanged = 99.8
        assert result[0] == pytest.approx((99.8, 100.5))

    def test_end_snap_word_within_tolerance(self):
        """Word starting 0.3 s before silence end snaps the end."""
        words = [self._w(f"w{i}", float(i), float(i) + 0.3) for i in range(100)]
        # Silence ends at 99.4; word[99] starts at 99.0 → gap = 0.4 ≤ tolerance ✓
        silences = [(98.0, 99.4)]
        result = snap_silences_to_words(silences, words, tolerance=0.40)
        assert len(result) == 1
        _, e = result[0]
        assert e == pytest.approx(99.0)  # snapped to word[99].start

    def test_start_snap_with_many_preceding_words(self):
        """Word ending 0.2 s after silence start snaps the silence start forward."""
        words = [self._w(f"w{i}", float(i), float(i) + 0.5) for i in range(80)]
        # Silence starts at 49.6; word[49] ends at 49.5 → end=49.5 is NOT > 49.6
        # word[50] starts at 50.0, ends at 50.5 → end=50.5 > 49.6, gap = 50.5-49.6=0.9 > 0.4
        # So no start snap
        silences = [(49.6, 51.0)]
        result = snap_silences_to_words(silences, words, tolerance=0.40)
        assert len(result) == 1
        s, _ = result[0]
        assert s == pytest.approx(49.6)  # unchanged


class TestOverlapsTranscriptBisect:
    """detect_untranscribed_speech bisect path: island detection remains correct
    when the word list is large (bisect skips most words)."""

    @staticmethod
    def _w(start: float, end: float) -> dict:
        return {"word": "x", "start": start, "end": end}

    def test_island_at_end_of_long_transcript_flagged(self):
        """Untranscribed island near minute-2 is found even after 100 preceding words."""
        # 100 words: [i, i+0.3] for i=0..99; gaps between words form silence regions
        words = [self._w(float(i), float(i) + 0.3) for i in range(100)]
        # Silences between each word (0.7 s each), plus one enclosing the target island
        silence = [(float(i) + 0.3, float(i + 1)) for i in range(100)]
        silence.append((100.5, 101.5))   # island = [100.0, 100.5], no word coverage
        result = detect_untranscribed_speech(
            total_duration=101.5,
            silence_regions=silence,
            transcript_segments=[],
            transcript_words=words,
            min_adjacent_large=0.25,
            min_adjacent_small=0.15,
        )
        # Island [100.0, 100.5]: 0.5 s, not covered → should appear
        island_starts = [r[0] for r in result]
        assert any(abs(s - 100.0) < 0.05 for s in island_starts)

    def test_covered_island_far_in_word_list_not_flagged(self):
        """An island covered by word[50] in a 100-word list is excluded."""
        # All words [i, i+0.3] for i=0..99; silences between them
        words = [self._w(float(i), float(i) + 0.3) for i in range(100)]
        # Create an explicit silence that frames island [50.0, 50.3] (= word[50])
        silence = [(49.5, 50.0), (50.3, 51.0)]
        result = detect_untranscribed_speech(
            total_duration=51.5,
            silence_regions=silence,
            transcript_segments=[],
            transcript_words=words,
            min_adjacent_large=0.25,
            min_adjacent_small=0.15,
        )
        # Island [50.0, 50.3] is exactly covered by word[50] → must NOT be flagged
        island_starts = [r[0] for r in result]
        assert not any(abs(s - 50.0) < 0.05 for s in island_starts)


class TestTerminateAndReap:
    """T0251: terminate_and_reap must kill AND reap subprocesses (no zombies)."""

    def test_kills_and_reaps_running_process(self):
        import sys
        from preprod.audio import terminate_and_reap
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        assert proc.poll() is None              # it's running
        terminate_and_reap(proc)
        assert proc.poll() is not None          # reaped — returncode set, no zombie

    def test_noop_on_already_exited_process(self):
        import sys
        from preprod.audio import terminate_and_reap
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()                             # let it exit on its own
        rc = proc.returncode
        terminate_and_reap(proc)                # must not raise or change state
        assert proc.returncode == rc

    def test_handles_none(self):
        from preprod.audio import terminate_and_reap
        terminate_and_reap(None)                # must be a safe no-op
