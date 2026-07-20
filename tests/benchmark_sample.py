"""Benchmark analysis against the reference sample file.

Run with:
    python3 tests/benchmark_sample.py [path/to/file.mov] [--no-whisper]

If no path is given, uses tests/fixtures/sample.mov.
Pass --no-whisper to skip transcription (silence detection only).

Outputs a human-readable report of silence + filler detection quality.
Use this after any changes to audio.py / transcribe.py / web.py to catch
regressions on real-world content.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from preprod.audio import (
    detect_silence,
    detect_untranscribed_speech,
    extract_audio,
    refine_word_boundary,
)
from preprod.probe import probe_media
from preprod.transcribe import WHISPER_AVAILABLE, detect_fillers, transcribe

DEFAULT_SAMPLE = Path(__file__).parent / "fixtures" / "sample.mov"
THRESHOLD_DB = -35.0
MIN_SILENCE   = 1.0
HANGOVER_MS   = 300
WHISPER_MODEL = "large-v3-turbo"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark silence + filler detection on a video file.")
    parser.add_argument("file", nargs="?", type=Path, default=DEFAULT_SAMPLE,
                        help="Video file to analyse (default: tests/fixtures/sample.mov)")
    parser.add_argument("--no-whisper", action="store_true",
                        help="Skip Whisper transcription (silence detection only)")
    parser.add_argument("--threshold-db", type=float, default=THRESHOLD_DB,
                        metavar="DB", help=f"Silence threshold in dBFS (default: {THRESHOLD_DB})")
    parser.add_argument("--min-dur", type=float, default=MIN_SILENCE,
                        metavar="S", help=f"Minimum silence duration in seconds (default: {MIN_SILENCE})")
    parser.add_argument("--hangover-ms", type=int, default=HANGOVER_MS,
                        metavar="MS", help=f"Hangover hold time in ms (default: {HANGOVER_MS})")
    parser.add_argument("--model", default=WHISPER_MODEL,
                        help=f"Whisper model size (default: {WHISPER_MODEL})")
    parser.add_argument("--language", default="ja",
                        help="Language code for Whisper, or 'auto' for auto-detection (default: ja)")
    parser.add_argument("--max-shown", type=int, default=15, metavar="N",
                        help="Max untranscribed entries to print in detail (default: 15, 0 = all)")
    args = parser.parse_args()

    SAMPLE        = args.file
    skip_whisper  = args.no_whisper
    threshold_db  = args.threshold_db
    min_silence   = args.min_dur
    hangover_ms   = args.hangover_ms
    whisper_model = args.model
    # "auto" is a user-friendly alias for None (faster-whisper auto-detects when language is None)
    language      = None if args.language.lower() == "auto" else args.language
    max_shown     = args.max_shown if args.max_shown > 0 else float("inf")

    if not SAMPLE.exists():
        print(f"ERROR: file not found: {SAMPLE}")
        sys.exit(1)

    print(f"Sample   : {SAMPLE}")
    media = probe_media(SAMPLE)
    print(f"Duration : {media.duration:.1f}s ({media.duration / 60:.1f}min)")
    print(f"Settings : threshold={threshold_db} dBFS  min_dur={min_silence}s  hangover={hangover_ms}ms\n")

    print("Extracting audio…", end=" ", flush=True)
    t0 = time.perf_counter()

    def _progress(pct: float) -> None:
        print(f"\rExtracting audio… {int(pct):3d}%", end="", flush=True)

    samples = extract_audio(SAMPLE, progress_callback=_progress, total_duration=media.duration)
    elapsed = time.perf_counter() - t0
    print(f"\rExtracted audio  ({elapsed:.1f}s, {len(samples)/16000:.1f}s @ 16 kHz)")

    # ── Audio level stats ─────────────────────────────────────────
    # Compute RMS in 50ms windows; show p95 ("speech level") and overall.
    # If threshold_db is within 3 dB of p95, the threshold is probably too tight.
    _win = int(16000 * 0.05)   # 50ms window
    if len(samples) >= _win:
        _arr = samples[:len(samples) - len(samples) % _win].reshape(-1, _win)
        _rms = np.sqrt(np.mean(_arr ** 2, axis=1))
        _rms_nz = _rms[_rms > 0]
        if len(_rms_nz) > 0:
            _p95_db = float(20 * np.log10(np.percentile(_rms_nz, 95)))
            _max_db = float(20 * np.log10(_rms_nz.max()))
            _gap    = _p95_db - threshold_db
            if _gap < 0:
                _warn = "  ⚠ threshold ABOVE speech level — try lowering threshold"
            elif _gap < 3:
                _warn = "  ⚠ threshold close to speech level — risk of cutting speech"
            else:
                _warn = ""
            print(f"Audio level      : p95={_p95_db:.1f} dBFS  peak={_max_db:.1f} dBFS"
                  f"  (threshold {threshold_db:+.0f}, gap={_gap:.1f} dB){_warn}")

    # ── Silence detection ────────────────────────────────────────
    silence_regions = detect_silence(
        samples, 16000, threshold_db, min_silence, hangover_ms=hangover_ms
    )
    total_sil = sum(e - s for s, e in silence_regions)
    pct_sil = total_sil / media.duration * 100 if media.duration else 0
    if silence_regions:
        durs = [e - s for s, e in silence_regions]
        print(f"Silence regions  : {len(silence_regions)}  ({total_sil:.1f}s, {pct_sil:.1f}%)"
              f"  dur min={min(durs):.1f}s mean={sum(durs)/len(durs):.1f}s max={max(durs):.1f}s")
    else:
        print(f"Silence regions  : 0  (no silence detected)")

    # ── Hangover sensitivity table ───────────────────────────────
    print("\nHangover sensitivity (same threshold + min-dur):")
    _base_n = len(silence_regions)
    for h_ms in (0, 100, 200, 300, 500, 750, 1000):
        regs = detect_silence(samples, 16000, threshold_db, min_silence, hangover_ms=h_ms)
        n = len(regs)
        marker = " ← current" if h_ms == hangover_ms else ""
        if _base_n > 0 and h_ms != hangover_ms:
            delta = (n - _base_n) / _base_n * 100
            delta_str = f"  ({delta:+.0f}%)"
        else:
            delta_str = ""
        print(f"  {h_ms:>4d} ms → {n:3d} regions{delta_str}{marker}")

    if not WHISPER_AVAILABLE:
        print("\nWhisper not installed — skipping filler detection.")
        print("Install with: pip install faster-whisper")
    if skip_whisper or not WHISPER_AVAILABLE:
        return

    # ── Whisper transcription ────────────────────────────────────
    print(f"\nTranscribing with {whisper_model}…", flush=True)
    t1 = time.perf_counter()
    result = transcribe(SAMPLE, model_size=whisper_model, language=language,
                        duration=media.duration)
    t_elapsed = time.perf_counter() - t1
    words    = result["words"]
    segments = result["segments"]
    rtf = t_elapsed / media.duration if media.duration else 0
    print(f"Transcribed      : {len(segments)} segments  {len(words)} words  lang={result['language']}  RTF={rtf:.2f}x")

    def _context(t_start: float, t_end: float, window: float = 3.0) -> str:
        """Return transcript text surrounding [t_start, t_end], using segment-level text."""
        # Use segments for readable text (segments have de-tokenized text; word tokens are
        # character-level for Japanese in faster-whisper and look fragmented in output).
        # before: segments whose end falls within [t_start-window, t_start]
        # (checking end, not start, captures segments that began before the window)
        before_segs = [s["text"].strip() for s in segments
                       if s["end"] > t_start - window and s["end"] <= t_start]
        after_segs  = [s["text"].strip() for s in segments
                       if s["start"] >= t_end and s["start"] < t_end + window]
        parts = []
        if before_segs:
            parts.append("".join(before_segs))
        parts.append("[FILLER]")
        if after_segs:
            parts.append("".join(after_segs))
        return "  …" + " · ".join(p for p in parts if p) + "…"

    # How many verbose entries to print before truncating (set via --max-shown).
    # Text fillers are rarely more than ~10 so always show them in full.
    MAX_SHOWN = max_shown

    # ── Text-based filler detection (audio-isolated) ─────────────
    fillers = detect_fillers(
        words, en=True, ja=True,
        samples=samples, sample_rate=16000, threshold_db=threshold_db,
    )
    if fillers:
        from collections import Counter
        _word_counts = Counter(w for _, _, w in fillers)
        _word_summary = "  " + "  ".join(
            f"\"{w}\" ×{n}" if n > 1 else f"\"{w}\""
            for w, n in _word_counts.most_common()
        )
    else:
        _word_summary = ""
    print(f"\nText fillers     : {len(fillers)}{_word_summary}")
    for s, e, w in fillers:
        rs, re = refine_word_boundary(samples, 16000, s, e, threshold_db=threshold_db)
        print(f"  [{s:.2f}–{e:.2f}] → [{rs:.2f}–{re:.2f}]  \"{w}\"")
        print(_context(s, e))

    # ── Untranscribed vocalizations ──────────────────────────────
    fine_silences = detect_silence(
        samples, 16000, threshold_db, min_duration=0.1, hangover_ms=0
    )
    untranscribed = detect_untranscribed_speech(
        total_duration=media.duration,
        silence_regions=fine_silences,
        transcript_segments=segments,
        transcript_words=words,
    )
    n_ut = len(untranscribed)
    rate_per_min = n_ut / (media.duration / 60) if media.duration > 0 else 0
    print(f"\nUntranscribed    : {n_ut}  ({rate_per_min:.1f}/min)")

    if n_ut:
        # Duration distribution: ≤100ms / 101–250ms / 251–500ms / >500ms
        _durs = [re - rs
                 for s, e in untranscribed
                 for rs, re in [refine_word_boundary(samples, 16000, s, e,
                                                     threshold_db=threshold_db)]]
        _buckets = [
            ("<= 100ms",  sum(1 for d in _durs if d <= 0.100)),
            ("101–250ms", sum(1 for d in _durs if 0.100 < d <= 0.250)),
            ("251–500ms", sum(1 for d in _durs if 0.250 < d <= 0.500)),
            (">  500ms",  sum(1 for d in _durs if d > 0.500)),
        ]
        print("  dur distribution: " + "  ".join(
            f"{label}: {cnt}" for label, cnt in _buckets if cnt
        ))
        # Print first MAX_SHOWN entries; summarise the rest
        shown = 0
        for idx, (s, e) in enumerate(untranscribed):
            rs, re = refine_word_boundary(samples, 16000, s, e, threshold_db=threshold_db)
            if shown < MAX_SHOWN:
                print(f"  [{s:.2f}–{e:.2f}] → [{rs:.2f}–{re:.2f}]  dur={re - rs:.3f}s")
                print(_context(s, e))
                shown += 1
            elif shown == MAX_SHOWN:
                remaining = n_ut - MAX_SHOWN
                print(f"  … and {remaining} more (set MAX_SHOWN to see all)")
                break

    total = len(fillers) + n_ut
    rate_total = total / (media.duration / 60) if media.duration > 0 else 0
    print(f"\nTOTAL filler candidates : {total}  ({rate_total:.1f}/min)")
    print(f"  Text-matched  : {len(fillers)}")
    print(f"  Audio-only    : {n_ut}")


if __name__ == "__main__":
    main()
