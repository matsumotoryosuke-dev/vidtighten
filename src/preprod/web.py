"""VidTighten web backend — Flask app with all API endpoints."""

from __future__ import annotations

import collections
import hashlib
import io
import logging
import logging.handlers
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

import numpy as np
import click
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from preprod.audio import SAMPLE_RATE, detect_silence, extract_audio, refine_word_boundary, detect_untranscribed_speech, snap_silences_to_words
from preprod.fcpxml_cut import generate_roughcut_fcpxml
from preprod.fcpxml_telop import generate_telop_fcpxml
from preprod.probe import MediaInfo, probe_media
from preprod.segments import (
    Segment, build_segments, map_span_to_output,
    filter_telop_entries, split_telop_segments, group_word_tokens,
    assign_words_to_entries,
)
from preprod.corrections import (
    correct_text, correct_words, active_corrections, save_user_correction,
)
from preprod import japanese, llm_correct
from preprod.session import delete_session, list_sessions, load_session, save_session
from preprod.transcribe import (
    WHISPER_AVAILABLE, WHISPERX_AVAILABLE, TranscribeCancelled, detect_fillers, transcribe,
)

app = Flask(__name__, static_folder="static", static_url_path="/static")
# No hard upload cap — large video files (25 GB+) must be accepted.
# The real protection is _is_path_allowed on /api/stream, not upload size.
app.config["MAX_CONTENT_LENGTH"] = None
# Disable static file HTTP caching so WKWebView always fetches fresh JS/CSS.
# Without this, Flask sends Cache-Control: max-age=43200 and WKWebView's
# NSURLCache serves the stale on-disk body even after the file changes.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Force WKWebView to never serve stale JS/CSS from its NSURLCache.
# SEND_FILE_MAX_AGE_DEFAULT=0 only omits the header; WKWebView then uses
# heuristic caching that can keep old files for hours.  Explicit no-store
# is the only reliable way to bust the on-disk NSURLCache across restarts.
@app.after_request
def _no_cache_static(response):
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# DNS-rebinding guard, app-wide: there is no auth anywhere in this app — every
# route trusts "the client is the user, on this machine" instead.  That trust
# only holds if we also check the client arrived via a localhost Host header.
# Without this, a malicious page open in the user's *browser* (not the app's
# WKWebView) can resolve a DNS name to 127.0.0.1 and drive any /api/* route
# cross-origin (browsers permit this; only the Host header changes). Was
# previously enforced only on /api/stream (T040) — every other route, including
# analysis/export/session/LLM endpoints, was unguarded.
_ALLOWED_HOSTS = ("127.0.0.1", "localhost", "::1")


def _host_without_port(host: str) -> str:
    """Strip the :port suffix from an HTTP Host header value.

    IPv6 literals are bracketed per RFC 7230 (e.g. "[::1]:9877" or
    bare "[::1]") specifically so their own colons aren't ambiguous
    with the port separator — a plain host.split(":")[0] would return
    "[" for "[::1]:9877" and reject a legitimate ::1 client.
    """
    if host.startswith("["):
        return host[1:host.index("]")] if "]" in host else host[1:]
    return host.split(":")[0]


@app.before_request
def _require_localhost_host():
    if _host_without_port(request.host) not in _ALLOWED_HOSTS:
        return "Forbidden", 403

# ── Analysis task registry ──────────────────────────────────────────────────
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()

# ── Directory layout ─────────────────────────────────────────────────────────
# Uploads: persistent ~/Library/Application Support/VidTighten/uploads.
# Using tempfile.gettempdir() was wrong: on macOS the TMPDIR path contains a
# per-boot hash that changes on every restart, so sessions that referenced an
# uploaded file failed with "File not found" after a reboot.  The 24-hour sweep
# also silently deleted files when a new upload triggered it, breaking export
# for sessions created the previous day.
#
# Exports: short-lived; stay in the system temp dir (one-shot downloads that
# don't need to survive past the current session).
_UPLOAD_DIR = (
    Path.home() / "Library" / "Application Support" / "VidTighten" / "uploads"
)
_EXPORT_DIR  = Path(tempfile.gettempdir()) / "preprod_exports"

# Days to keep uploaded files before sweeping.  Matches session retention so
# that a session and its source file always expire together.
_UPLOAD_KEEP_DAYS = 30


def _resolve_log_path() -> Path:
    """Return the path of the active RotatingFileHandler, or the default."""
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            return Path(h.baseFilename)
    return Path.home() / ".preprod" / "logs" / "app.log"

LOG_PATH = _resolve_log_path()

# ── Stream path allowlist (DNS-rebinding + path-traversal protection) ────────
# Only files under these directories may be served by /api/stream.
# Mutating this set in tests is acceptable; do not do so in production code.
# /Volumes is included so that files on external drives, SD cards, and
# network-attached storage work out of the box on macOS.
_ALLOWED_DIRS: set[Path] = {
    Path.home() / "Movies",
    Path.home() / "Desktop",
    Path.home() / "Downloads",
    Path.home() / "Documents",
    Path("/Volumes"),          # external drives, SD cards, NAS mounts on macOS
    # resolve() needed: macOS /var → /private/var symlink breaks is_relative_to otherwise
    _UPLOAD_DIR.resolve(),
}


def _is_path_allowed(p: Path) -> bool:
    """Return True if resolved path p is under one of the allowed directories."""
    return any(p.is_relative_to(d) for d in _ALLOWED_DIRS)


# ── M4A sidecar transcode cache ──────────────────────────────────────────────
# VBR MP3 files are transcoded to M4A (CBR AAC) for browser playback.
#
# Why: AVFoundation computes player.currentTime from byte offsets in a VBR MP3
# (using the Xing/VBRI TOC), which diverges from actual sample position for long
# files.  Observed drift: ~1 s after 23 minutes for a 259 kbps VBR MP3 encoded
# by Final Cut Pro.  M4A/AAC is a block codec with sample-accurate timestamps
# stored in the container — player.currentTime matches decoded samples exactly,
# so transcript and telop overlays stay in sync throughout the file.
#
# Transcoding runs once in a background thread when the file is first streamed,
# then the M4A is served from cache on all subsequent requests.

_TRANSCODE_DIR = Path.home() / ".preprod" / "transcode_cache"
_M4A_CACHE_MAX_BYTES = 3 * 1024 ** 3   # 3 GB cap on M4A sidecars (size bound)
# Map from cache key → "pending" | "ready" | "failed"
_TRANSCODE_STATE: dict[str, str] = {}
_TRANSCODE_STATE_LOCK = threading.Lock()


def _mp3_cache_key(mp3_path: Path) -> str:
    """16-hex-char stable key for a given MP3 path."""
    return hashlib.sha1(str(mp3_path).encode()).hexdigest()[:16]


def _m4a_cache_path(mp3_path: Path) -> Path:
    return _TRANSCODE_DIR / f"{_mp3_cache_key(mp3_path)}.m4a"


def _is_vbr_mp3(p: Path) -> bool:
    """Return True when the file is an MP3 with a Xing/VBRI VBR header."""
    if p.suffix.lower() != ".mp3":
        return False
    try:
        with open(p, "rb") as f:
            header = f.read(8192)
        return b"Xing" in header or b"VBRI" in header
    except OSError:
        return False


def _transcode_mp3_to_m4a_worker(mp3_path: Path) -> None:
    """Background worker: ffmpeg MP3 → M4A AAC with +faststart (seekable)."""
    key = _mp3_cache_key(mp3_path)
    out = _m4a_cache_path(mp3_path)
    _TRANSCODE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(mp3_path),
                "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart",
                str(out),
            ],
            capture_output=True,
            timeout=600,
        )
        if r.returncode == 0 and out.exists():
            with _TRANSCODE_STATE_LOCK:
                _TRANSCODE_STATE[key] = "ready"
            log.info("M4A transcode ready: %s", out)
        else:
            with _TRANSCODE_STATE_LOCK:
                _TRANSCODE_STATE[key] = "failed"
            log.warning("M4A transcode failed for %s: %s", mp3_path,
                        r.stderr.decode(errors="replace")[:300])
    except Exception as exc:
        with _TRANSCODE_STATE_LOCK:
            _TRANSCODE_STATE[key] = "failed"
        log.warning("M4A transcode error for %s: %s", mp3_path, exc)


def _start_m4a_transcode(mp3_path: Path) -> None:
    """Start background M4A transcoding if not already started/done.

    Also triggers a rate-limited cache sweep to remove M4A sidecars whose
    source MP3 has not been accessed for 30 days, preventing unbounded
    disk growth when many MP3 files are opened over time.
    """
    # Rate-limited cleanup: sweep sidecars older than 30 days (once per 5 min),
    # then enforce a hard total-size cap so many recent files can't fill the disk.
    _sweep_temp_dir(_TRANSCODE_DIR, max_age_hours=30 * 24)
    _sweep_dir_by_size(_TRANSCODE_DIR, _M4A_CACHE_MAX_BYTES)

    key = _mp3_cache_key(mp3_path)
    # Check on-disk cache first (survives process restarts)
    if _m4a_cache_path(mp3_path).exists():
        with _TRANSCODE_STATE_LOCK:
            _TRANSCODE_STATE[key] = "ready"
        return
    with _TRANSCODE_STATE_LOCK:
        if _TRANSCODE_STATE.get(key) in ("pending", "ready"):
            return
        _TRANSCODE_STATE[key] = "pending"
    t = threading.Thread(
        target=_transcode_mp3_to_m4a_worker, args=(mp3_path,), daemon=True
    )
    t.start()


# ── Video preview proxy cache ─────────────────────────────────────────────────
# High-resolution footage (e.g. 4K 10-bit 4:2:2 HEVC "Rext" profile, common from
# prosumer/cinema cameras) frequently has no hardware decode path in WKWebView,
# forcing a slow software decode that makes scrubbing/playback extremely laggy
# during editing — independent of anything in this app's JS.  Standard editorial
# workaround: generate a small proxy (low resolution, fast-decode codec) for
# on-screen PREVIEW ONLY.  The original file stays canonical for audio
# extraction, transcription, and FCPXML/export — only /api/stream (the <video>
# element's source) ever serves the proxy.
#
# Transcoding runs once in a background thread when the file is first streamed,
# then the proxy is served from cache on all subsequent requests.

_PROXY_DIR = Path.home() / ".preprod" / "proxy_cache"
_PROXY_CACHE_MAX_BYTES = 8 * 1024 ** 3   # 8 GB cap (long 4K timelines add up)
_PROXY_MIN_LONG_EDGE = 1920              # generate a proxy above this resolution
_PROXY_WIDTH = 800                       # target width; height auto (-2 keeps aspect + even)
# Map from cache key → "pending" | "ready" | "failed"
_PROXY_STATE: dict[str, str] = {}
_PROXY_STATE_LOCK = threading.Lock()


def _proxy_cache_key(path: Path) -> str:
    """16-hex-char stable key for a given source path."""
    return hashlib.sha1(str(path).encode()).hexdigest()[:16]


def _proxy_cache_path(path: Path) -> Path:
    return _PROXY_DIR / f"{_proxy_cache_key(path)}.proxy.mp4"


def _needs_proxy(media: MediaInfo) -> bool:
    if not media.has_video:
        return False
    w, h = media.video_width, media.video_height
    # Defensive: video_width/height should be int|None per MediaInfo, but guard
    # against any unexpected type reaching here (e.g. a probe result that wasn't
    # fully populated) rather than raising out of the /api/stream hot path.
    if not isinstance(w, (int, float)) or not isinstance(h, (int, float)):
        return False
    return max(w, h) > _PROXY_MIN_LONG_EDGE


def _transcode_video_proxy_worker(path: Path) -> None:
    """Background worker: ffmpeg downscale to a fast-decode preview proxy."""
    key = _proxy_cache_key(path)
    out = _proxy_cache_path(path)
    _PROXY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(path),
                "-vf", f"scale={_PROXY_WIDTH}:-2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                str(out),
            ],
            capture_output=True,
            timeout=1800,
        )
        if r.returncode == 0 and out.exists():
            with _PROXY_STATE_LOCK:
                _PROXY_STATE[key] = "ready"
            log.info("Video proxy ready: %s", out)
        else:
            with _PROXY_STATE_LOCK:
                _PROXY_STATE[key] = "failed"
            log.warning("Video proxy transcode failed for %s: %s", path,
                        r.stderr.decode(errors="replace")[:300])
    except Exception as exc:
        with _PROXY_STATE_LOCK:
            _PROXY_STATE[key] = "failed"
        log.warning("Video proxy transcode error for %s: %s", path, exc)


def _start_video_proxy_transcode(path: Path, media: MediaInfo) -> None:
    """Start background proxy transcoding if not already started/done and needed.

    Also triggers rate-limited cleanup: sweep proxies whose source hasn't been
    accessed in 30 days, then enforce a hard total-size cap.
    """
    if not _needs_proxy(media):
        return

    _sweep_temp_dir(_PROXY_DIR, max_age_hours=30 * 24)
    _sweep_dir_by_size(_PROXY_DIR, _PROXY_CACHE_MAX_BYTES)

    key = _proxy_cache_key(path)
    # Check on-disk cache first (survives process restarts)
    if _proxy_cache_path(path).exists():
        with _PROXY_STATE_LOCK:
            _PROXY_STATE[key] = "ready"
        return
    with _PROXY_STATE_LOCK:
        if _PROXY_STATE.get(key) in ("pending", "ready"):
            return
        _PROXY_STATE[key] = "pending"
    t = threading.Thread(
        target=_transcode_video_proxy_worker, args=(path,), daemon=True
    )
    t.start()


# ── Media info cache (path → MediaInfo, LRU-bounded) ────────────────────────
# Bounded at 200 entries to prevent unbounded growth in long-running sessions.
# _media_cache_lock serialises writes from background analysis threads and
# reads from request threads — necessary because dict iteration + pop is not
# atomic even under the GIL when multiple threads interleave.
# OrderedDict enables true LRU: move_to_end on hit, popitem(last=False) on evict.
_media_cache: collections.OrderedDict = collections.OrderedDict()
_media_cache_lock = threading.Lock()
_MEDIA_CACHE_MAX = 200


def _cache_media(path: str, media: MediaInfo) -> None:
    """Insert into the media cache, evicting the LRU entry when full."""
    with _media_cache_lock:
        if path in _media_cache:
            _media_cache.move_to_end(path)
        _media_cache[path] = media
        if len(_media_cache) > _MEDIA_CACHE_MAX:
            _media_cache.popitem(last=False)  # evict least-recently-used


# ── Audio sample cache (path → (samples, mtime), LRU-bounded) ───────────────
# Caches decoded PCM arrays so repeated redetect_silence calls (threshold
# slider adjustments) skip the ~1.7s extract_audio step after the first hit.
# Bounded at 3 entries (~200–300 MB for 26-min files) to cap memory use.
# Keyed by resolved path string; invalidated when mtime changes.
# OrderedDict enables true LRU: move_to_end on hit, popitem(last=False) on evict.
_audio_cache: collections.OrderedDict = collections.OrderedDict()
_audio_cache_lock = threading.Lock()
_AUDIO_CACHE_MAX = 3


def _get_cached_audio(
    p: Path,
    progress_callback=None,
    total_duration: float = 0.0,
    expected_samples: int = 0,
) -> np.ndarray:
    """Return decoded PCM samples for p, using a mtime-keyed in-process cache.

    The lock is released before extract_audio so slow IO doesn't stall other
    requests. Two concurrent misses for the same file both extract and the last
    writer wins — identical content, so this is safe.

    If expected_samples > 0, validates that extraction produced at least 10 % of
    the expected count (guards against ffmpeg silent decode failures on large files).
    """
    key = str(p)
    mtime = p.stat().st_mtime
    with _audio_cache_lock:
        entry = _audio_cache.get(key)
        if entry is not None and entry[1] == mtime:
            _audio_cache.move_to_end(key)  # mark as recently used
            return entry[0]

    samples = extract_audio(p, progress_callback=progress_callback, total_duration=total_duration)

    # ffmpeg occasionally exits 0 while producing far fewer samples than expected
    # (codec decode error — emits stderr but doesn't always set a non-zero exit code).
    # Threshold: < 10 % of expected counts as a failed extraction.
    if expected_samples > 0:
        _ratio = len(samples) / expected_samples
        if _ratio < 0.10:
            raise RuntimeError(
                f"Audio extraction produced only {len(samples):,} samples "
                f"({_ratio*100:.1f}% of expected {expected_samples:,} for a "
                f"{total_duration:.0f}s file). "
                "The file may use an unsupported codec, be partially corrupted, "
                "or be DRM-protected. Try re-encoding with HandBrake or ffmpeg."
            )

    with _audio_cache_lock:
        if key in _audio_cache:
            _audio_cache.move_to_end(key)
        _audio_cache[key] = (samples, mtime)
        if len(_audio_cache) > _AUDIO_CACHE_MAX:
            _audio_cache.popitem(last=False)  # evict least-recently-used
    return samples


# ── Word timestamp cache (path → words list, bounded LRU) ───────────────────
# Stores the raw Whisper word list from the last successful full analysis for
# each media file.  Used by api_analyze_redetect_silence to apply
# snap_silences_to_words when words are available, making threshold-slider
# adjustments consistent with a full analysis.
# Bounded at 5 entries (word lists are small — ~1 KB per minute of audio).
_words_cache: collections.OrderedDict = collections.OrderedDict()
_words_cache_lock = threading.Lock()
_WORDS_CACHE_MAX = 5


def _cache_words(path: str, words: list[dict]) -> None:
    """Insert word list into the per-path word cache, evicting LRU when full."""
    with _words_cache_lock:
        if path in _words_cache:
            _words_cache.move_to_end(path)
        _words_cache[path] = words
        if len(_words_cache) > _WORDS_CACHE_MAX:
            _words_cache.popitem(last=False)


def _get_cached_words(path: str) -> list[dict] | None:
    """Return the cached word list for path, or None if not cached."""
    with _words_cache_lock:
        words = _words_cache.get(path)
        if words is not None:
            _words_cache.move_to_end(path)
        return words


# ── Helpers ─────────────────────────────────────────────────────────────────

def _media_to_dict(m: MediaInfo) -> dict:
    return {
        "path":        str(m.path),
        "duration":    m.duration,
        "has_video":   m.has_video,
        "has_audio":   m.has_audio,
        "video_width": m.video_width,
        "video_height": m.video_height,
        "frame_rate":  float(m.frame_rate) if m.frame_rate else None,
        "frame_rate_num": m.frame_rate.numerator if m.frame_rate else 30,
        "frame_rate_den": m.frame_rate.denominator if m.frame_rate else 1,
        "sample_rate": m.sample_rate,
        "codec_video": m.codec_video,
        "codec_audio": m.codec_audio,
    }


def _task_update(task_id: str, **kwargs) -> None:
    with _tasks_lock:
        _tasks[task_id].update(kwargs)


def _silence_candidates(silence_regions: list) -> list[dict]:
    """Convert a list of (start, end) silence tuples to removal-candidate dicts."""
    return [
        {
            "id":    f"s{i}",
            "start": round(s, 3),
            "end":   round(e, 3),
            "type":  "silence",
            "label": "silence",
        }
        for i, (s, e) in enumerate(silence_regions)
    ]


class _StageTimer:
    """Accumulates wall-clock durations between lap() marks for pipeline profiling (T0246).

    Pure measurement — no behaviour change. `clock` is injectable for tests.
    """

    def __init__(self, clock=time.perf_counter):
        self._clock = clock
        self._stages: dict[str, float] = {}
        self._mark = clock()

    def lap(self, name: str) -> None:
        """Record time since the previous mark under `name` (accumulates if repeated)."""
        now = self._clock()
        self._stages[name] = round(self._stages.get(name, 0.0) + (now - self._mark), 3)
        self._mark = now

    def skip(self) -> None:
        """Advance the mark without recording — for stretches we don't attribute."""
        self._mark = self._clock()

    def result(self, extra: dict | None = None) -> dict:
        """Return {stage: seconds, ..., total} plus any `extra` (e.g. worker sub-timings)."""
        out = dict(self._stages)
        out["total"] = round(sum(self._stages.values()), 3)
        if extra:
            out.update(extra)
        return out


def _run_analysis(task_id: str, path: str, params: dict, hangover_ms: int = 300) -> None:
    """Background thread: silence detection + optional Whisper transcription."""
    log.info(
        "Analysis %s started — path=%s threshold_db=%s min_dur=%s"
        " hangover_ms=%s use_whisper=%s",
        task_id, path,
        params.get("threshold_db", -40.0),
        params.get("min_duration", 1.0),
        hangover_ms,
        bool(params.get("use_whisper", False)),
    )
    cancel_event = threading.Event()
    with _tasks_lock:
        _tasks[task_id]["cancel_event"] = cancel_event

    try:
        _timer = _StageTimer()          # T0246: per-stage profiling (no behaviour change)
        _worker_timings: dict | None = None

        p = Path(path).resolve()

        _task_update(task_id, progress=5, stage="probing")
        media = probe_media(p)
        _cache_media(str(p), media)  # key by resolved path for cache consistency
        _timer.lap("probe")
        log.info("Analysis %s probed — duration=%.1fs has_audio=%s", task_id, media.duration, media.has_audio)

        if not media.has_audio:
            raise RuntimeError("File has no audio track.")

        _task_update(task_id, progress=15, stage="extracting audio")

        def _extraction_progress(pct: float) -> None:
            # Map ffmpeg 0-99 → task progress 15-29 % (14-point band before silence detection at 30)
            _task_update(task_id, progress=15 + round(pct / 99 * 14), stage="extracting audio")

        samples = _get_cached_audio(
            p,
            progress_callback=_extraction_progress,
            total_duration=media.duration,
            expected_samples=int(media.duration * SAMPLE_RATE),
        )
        _timer.lap("extract_audio")

        # Extract analysis params once — default matches the UI's SETTINGS_DEFAULTS.
        threshold_db = float(params.get("threshold_db", -40.0))
        min_duration = float(params.get("min_duration", 1.0))

        _task_update(task_id, progress=30, stage="detecting silence")
        silence_regions = detect_silence(
            samples,
            sample_rate=16000,
            threshold_db=threshold_db,
            min_duration=min_duration,
            hangover_ms=hangover_ms,
        )
        _timer.lap("silence_coarse")

        removal_candidates = _silence_candidates(silence_regions)

        telop_entries:  list[dict] = []
        _words_payload: list[dict] = []  # structured words for the frontend
        whisper_warning: str | None = None   # set if Whisper ran but had a non-fatal problem

        # Optional Whisper transcription + filler detection
        use_whisper = params.get("use_whisper", False) and WHISPER_AVAILABLE
        if use_whisper:
          try:
            # Whisper progress is mapped into the 46–78 % band.
            # The callback receives either a string stage ("loading model",
            # "transcribing") or an int 0-99 emitted per decoded chunk.
            _WH_LOW, _WH_HIGH = 46, 78

            def _progress(value) -> None:
                if isinstance(value, int):
                    # Per-chunk progress: scale 0-99 → _WH_LOW–_WH_HIGH
                    pct   = _WH_LOW + round(value / 99 * (_WH_HIGH - _WH_LOW))
                    pct   = min(_WH_HIGH, pct)
                    stage = "loading Whisper model" if value < 5 else f"Whisper {value}%"
                else:
                    pct   = _WH_LOW   if "load" in str(value) else _WH_HIGH
                    stage = str(value)
                _task_update(task_id, progress=pct, stage=stage)

            _task_update(task_id, progress=45, stage="loading Whisper model")
            result = transcribe(
                p,
                model_size=params.get("whisper_model", "large-v3-turbo"),
                language=params.get("language") or None,
                progress_callback=_progress,
                cancel_event=cancel_event,
                duration=float(media.duration),
            )
            _timer.lap("transcribe")
            _worker_timings = result.get("timings")
            words = result["words"]
            segments_raw = result["segments"]
            # Apply custom-vocabulary corrections (e.g. brand name "空気デザイン" →
            # "クウキデザイン") to the word list — the single source of truth from
            # which telop and subtitle text are rebuilt — and to raw segment text.
            correct_words(words)
            for _seg in segments_raw:
                if _seg.get("text"):
                    _seg["text"] = correct_text(_seg["text"])
            # Surface WhisperX alignment status to the user.  whisperx_used=False means
            # the worker ran but WhisperX was unavailable or failed — word timestamps are
            # ±150ms instead of ±20ms.
            _whisperx_used: bool = bool(result.get("whisperx_used"))
            if not _whisperx_used:
                whisper_warning = (
                    "Word timestamps use faster-whisper alignment (±150ms). "
                    "Install whisperx for ±20ms accuracy."
                )

            # Cache word timestamps so redetect_silence can apply snapping later.
            _cache_words(str(p), words)

            # Snap silence boundaries to word timestamps.
            # Whisper's word timestamps can be up to ~280 ms ahead of the
            # actual audio energy onset.  Without this, a silence that ends
            # at 58.38 s appears to "eat" the word Whisper reports at 58.10 s.
            # snap_silences_to_words only shrinks regions (never widens them)
            # so it cannot create false positives.
            silence_regions = snap_silences_to_words(silence_regions, words)
            # Rebuild removal_candidates from the snapped silence list.
            removal_candidates = _silence_candidates(silence_regions)
            _timer.lap("correct_snap")  # brand corrections + silence-to-word snapping

            _task_update(task_id, progress=80, stage="detecting fillers")
            # Filler isolation uses a floor of -35 dB regardless of the user's
            # silence threshold.  The silence detector may use a strict -40 dB
            # threshold (catching only very quiet regions), but typical inter-word
            # pauses in speech sit around -36 to -30 dB — still clearly quieter
            # than speech but not absolute silence.  A -40 dB floor for the filler
            # isolation check would cause too many real fillers to fail the isolation
            # test (nothing counts as "quiet enough" around the word).
            # Floor at -35 dB for all word-level boundary operations: filler
            # isolation, refine_word_boundary, and untranscribed-speech refinement.
            # Typical inter-word pauses sit at -38 to -30 dB — stricter thresholds
            # (e.g. -40 dB) prevent these operations from seeing any "silence" at all.
            _refine_threshold_db = max(threshold_db, -35.0)
            fillers = detect_fillers(
                words,
                en=bool(params.get("fillers_english", True)),
                ja=bool(params.get("fillers_japanese", True)),
                custom=params.get("fillers_custom", []),
                samples=samples,
                sample_rate=16000,
                threshold_db=_refine_threshold_db,
            )
            for i, (start, end, word) in enumerate(fillers):
                rs, re = refine_word_boundary(
                    samples, 16000, start, end, threshold_db=_refine_threshold_db
                )
                removal_candidates.append({
                    "id":    f"f{i}",
                    "start": round(rs, 3),
                    "end":   round(re, 3),
                    "type":  "filler",
                    "label": word,
                })
            _timer.lap("fillers")

            # Split long Whisper segments into subtitle-sized chunks.
            # Whisper may produce 15-second segments spanning 3–4 display lines;
            # split_telop_segments breaks them at sentence boundaries (preferring
            # 。！？ punctuation) or at word boundaries if no punctuation is found.
            # max_em: derive from telop settings when available, fall back to default.
            _t_font_sz  = max(1, int(params.get("font_size", 92)))
            _t_width    = int(params.get("width", 3840))
            _max_em     = max(8.0, _t_width * 0.42 / _t_font_sz) * 2   # 2-line budget
            split_segs  = split_telop_segments(segments_raw, words, max_em=_max_em)
            for i, seg in enumerate(split_segs):
                telop_entries.append({
                    "id":    f"t{i}",
                    "start": seg["start"],
                    "end":   seg["end"],
                    "text":  seg["text"],
                })
            _timer.lap("telop_split")   # fugashi phrase-boundary splitting

            # Merge contiguous single-CJK-char tokens (faster-whisper Japanese
            # character-level tokenisation) into single word objects for the
            # frontend payload.  The raw `words` list is kept intact for
            # detect_untranscribed_speech below (acoustic analysis benefits
            # from the original per-character timestamps).
            _grouped_words = group_word_tokens(words)

            # Build flat word payload for the frontend word-level editor.
            # Assign each grouped word to its containing telop entry; uses the
            # closest-start strategy to correctly handle boundary words.
            _nonempty_grouped = [
                _w for _w in _grouped_words
                if (_w.get("word") or _w.get("text") or "").strip()
            ]
            _seg_ids = assign_words_to_entries(_nonempty_grouped, telop_entries)
            # Build O(1) lookup dict so word-clamping doesn't require a linear scan per word.
            _entry_by_id: dict[str, dict] = {e["id"]: e for e in telop_entries}
            _wi_counter = 0
            for _w, _seg_id in zip(_nonempty_grouped, _seg_ids):
                _ws = float(_w.get("start", 0))
                _we = float(_w.get("end",   0))
                _wt = (_w.get("word") or _w.get("text") or "").strip()
                # faster-whisper uses "score"; group_word_tokens uses "confidence"
                _conf = _w.get("score") if _w.get("score") is not None else _w.get("confidence")
                # Clamp word timestamps to segment bounds (±50ms allowance for Whisper's
                # systematic ~50ms early-start bias) — but ONLY for faster-whisper output.
                # WhisperX CTC-aligned timestamps are ±20ms accurate and don't need
                # segment-level clamping; clamping them would clip the last word of each
                # segment short (e.g. 5.286s→5.05s) and cause highlight gaps during playback.
                # Unassigned words (seg_id=None) keep raw timestamps regardless.
                if _seg_id is not None and not _whisperx_used:
                    _entry = _entry_by_id.get(_seg_id)
                    if _entry is not None:
                        _ws_c = max(_ws, _entry["start"] - 0.05)
                        _we_c = min(_we, _entry["end"]   + 0.05)
                        # Only apply the clamp when it doesn't invert the span.
                        # A word assigned to a distant segment by the unconstrained
                        # fallback may have raw timestamps that land outside the
                        # segment window; clamping them would give start > end.
                        if _ws_c < _we_c:
                            _ws, _we = _ws_c, _we_c
                _words_payload.append({
                    "id":         f"w{_wi_counter}",
                    "start":      round(_ws, 3),
                    "end":        round(_we, 3),
                    "text":       _wt,
                    "seg_id":     _seg_id,
                    "confidence": round(float(_conf), 3) if _conf is not None else None,
                })
                _wi_counter += 1
            _timer.lap("word_payload")  # group_word_tokens + entry assignment + clamping

            # Detect vocalizations Whisper skipped (えー、うー、hmm, uh…).
            # Run a fine-grained silence pass (0.1s min, no hangover) to get
            # precise island boundaries; use word timestamps (not segment
            # timestamps) for overlap — Whisper segments span over fillers.
            _task_update(task_id, progress=83, stage="detecting missed vocalizations")
            fine_silences = detect_silence(
                samples, 16000,
                threshold_db=threshold_db,
                min_duration=0.1,
                hangover_ms=0,
            )
            untranscribed = detect_untranscribed_speech(
                total_duration=media.duration,
                silence_regions=fine_silences,
                transcript_segments=segments_raw,
                transcript_words=words,
            )
            for i, (vs, ve) in enumerate(untranscribed):
                rs, re = refine_word_boundary(
                    samples, 16000, vs, ve,
                    threshold_db=_refine_threshold_db,
                )
                removal_candidates.append({
                    "id":    f"v{i}",
                    "start": round(rs, 3),
                    "end":   round(re, 3),
                    "type":  "filler",
                    "label": "〜",
                })
            _timer.lap("untranscribed")  # fine silence pass + missed-vocalization detection

          except TranscribeCancelled:
            raise  # let the outer handler mark as cancelled
          except RuntimeError as _wh_exc:
            # Whisper timed out or crashed — fall back to silence-only results.
            # removal_candidates already has the silence cuts from before the
            # Whisper block, so the user still gets a useful analysis.
            log.warning("Analysis %s: Whisper failed, falling back to silence-only — %s", task_id, _wh_exc)
            whisper_warning = (
                "Filler detection timed out — silence cuts are included below. "
                "To avoid this, disable Filler Word Detection in Settings, "
                "or switch to a faster Whisper model (e.g. turbo)."
            )

        # Sort removal candidates by start time
        removal_candidates.sort(key=lambda c: c["start"])

        # Compute waveform (normalized RMS amplitude, adaptive resolution).
        # 30 pts/sec gives fine detail for short files; capped at 3000 so
        # even a 90-min recording stays manageable (~0.56 pts/sec at cap).
        # Floor at 1500 so short clips still have a full-width waveform.
        _task_update(task_id, progress=88, stage="computing waveform")
        WF_POINTS = min(3000, max(1500, int(media.duration * 30)))
        if len(samples) >= WF_POINTS:
            chunk = len(samples) // WF_POINTS
            arr = samples[:chunk * WF_POINTS].reshape(WF_POINTS, chunk)
            rms = np.sqrt(np.mean(arr ** 2, axis=1))
        else:
            rms = np.abs(samples)
        max_amp = float(rms.max()) if rms.size > 0 else 1.0
        if max_amp > 0:
            rms = rms / max_amp
        waveform = [round(float(v), 3) for v in rms]

        # Threshold fraction: where the silence threshold sits on the normalized scale
        threshold_amp = 10.0 ** (threshold_db / 20.0)
        waveform_threshold = round(threshold_amp / max_amp, 4) if max_amp > 0 else 0.0
        _timer.lap("waveform")

        # T0246: assemble per-stage timings (worker sub-timings nested under transcribe_detail).
        _timings = _timer.result(
            {"transcribe_detail": _worker_timings} if _worker_timings else None
        )
        log.info("Analysis %s timings(s): %s", task_id, _timings)

        _task_update(task_id, progress=95, stage="finalizing")

        kept = _kept_duration(
            silence_regions, media.duration,
            padding_ms=int(params.get("padding_ms", 200))
        )
        removed = media.duration - kept
        pct = (kept / media.duration * 100) if media.duration > 0 else 100.0

        # accuracy_warning is set when WhisperX was unavailable (word timestamps ±150ms).
        # Stored separately from whisper_warning so the frontend can apply a distinct
        # visual treatment (e.g. a persistent banner vs a transient toast).
        _accuracy_warning: str | None = (
            whisper_warning
            if whisper_warning and "faster-whisper alignment" in whisper_warning
            else None
        )

        result_data = {
            "media": _media_to_dict(media),
            "removal_candidates": removal_candidates,
            "telop_entries": telop_entries,
            "words": _words_payload,
            "waveform": waveform,
            "waveform_threshold": waveform_threshold,
            "waveform_max_amp": round(max_amp, 6),
            "accuracy_warning": _accuracy_warning,
            "timings": _timings,   # T0246: per-stage profiling (seconds)
            "stats": {
                "total_removals": len(removal_candidates),
                "silence_count": len(silence_regions),
                "filler_count": len([c for c in removal_candidates if c["type"] == "filler"]),
                "telop_count": len(telop_entries),
                "kept_duration": round(kept, 2),
                "removed_duration": round(removed, 2),
                "original_duration": round(media.duration, 2),
                "kept_percent": round(pct, 1),
                "whisper_available": WHISPER_AVAILABLE,
                "whisper_warning": whisper_warning,
            },
        }

        with _tasks_lock:
            _tasks[task_id].update({
                "status":   "done",
                "progress": 100,
                "stage":    "done",
                "result":   result_data,
            })

    except Exception as exc:
        if isinstance(exc, TranscribeCancelled):
            log.info("Analysis task %s cancelled by user", task_id)
            with _tasks_lock:
                _tasks[task_id].update({
                    "status": "cancelled",
                    "error":  None,
                })
        else:
            log.error("Analysis task %s failed: %s", task_id, exc, exc_info=True)
            with _tasks_lock:
                _tasks[task_id].update({
                    "status": "error",
                    "error":  str(exc),
                })


def _build_segments_typed(
    removal_data: list[dict],
    media_path: "Path | None",
    total_duration: float,
    padding_ms: int,
    threshold_db: float = -40.0,
) -> "list[Segment]":
    """Build keep-segments with per-type padding awareness.

    Word-type regions are refined to actual audio boundaries (via
    refine_word_boundary) and exported with zero padding — their boundaries
    are already snapped to audio energy transitions, and applying the normal
    silence padding would collapse short words (< 2×padding_ms) to nothing.

    Non-word regions (silence, manual, filler) receive the user's configured
    padding so context audio is preserved around each cut, matching the
    existing behaviour.

    Returns the same list[Segment] as build_segments().
    """
    try:
        samples = _get_cached_audio(media_path) if media_path else None
    except Exception as exc:
        log.warning("Word-boundary refinement disabled — could not load audio for %s: %s",
                    media_path, exc)
        samples = None

    padding_sec = padding_ms / 1000.0
    # Use -35 dB floor for word boundary refinement.  Typical indoor noise sits
    # around -38 to -36 dB, so a raw -40 dB user threshold prevents
    # refine_word_boundary from finding the frame right before the word
    # onset — leaving boundaries at Whisper's ±150 ms precision instead of ±10 ms.
    _refine_db = max(threshold_db, -35.0)
    combined: list[tuple[float, float]] = []
    _refine_failures = 0

    for r in removal_data:
        start, end = float(r["start"]), float(r["end"])
        if r.get("type") == "word":
            # Refine boundary to actual audio energy; zero padding so short
            # words aren't collapsed by the padding subtraction.
            if samples is not None:
                try:
                    start, end = refine_word_boundary(
                        samples, SAMPLE_RATE, start, end, threshold_db=_refine_db,
                    )
                except Exception:
                    _refine_failures += 1   # raw timestamps; summarized after loop
            if end > start:
                combined.append((start, end))
        else:
            # Silence / manual / filler: shrink inward to keep context audio.
            ns, ne = start + padding_sec, end - padding_sec
            if ne > ns:
                combined.append((ns, ne))
            # else: removal shrinks to nothing under padding — intentionally dropped.

    # Surface systematic refinement failures once (per-word logging would spam).
    if _refine_failures:
        log.warning("Word-boundary refinement failed for %d region(s); used raw timestamps.",
                    _refine_failures)

    # build_segments sorts, merges, and inverts into keep-segments.
    # padding_ms=0 because per-type padding was already applied above.
    return build_segments(combined, total_duration, padding_ms=0)


def _kept_duration(silence_regions, total_duration, padding_ms):
    segs = build_segments(
        [(s, e) for s, e in silence_regions],
        total_duration,
        padding_ms,
    )
    return sum(s.duration for s in segs)


_sweep_temp_dir_last_ts: dict[Path, float] = {}
_SWEEP_TEMP_DIR_INTERVAL = 300.0  # seconds — maximum sweep frequency per directory


def _sweep_temp_dir(directory: Path, max_age_hours: float = 24.0, force: bool = False) -> int:
    """Delete files in *directory* older than *max_age_hours*. Returns number removed.

    Calls are rate-limited to at most once per _SWEEP_TEMP_DIR_INTERVAL seconds per
    directory to avoid O(n) filesystem I/O on every upload or export.  Pass
    force=True to bypass the rate limit (used at startup).
    """
    now = time.time()
    if not force and now - _sweep_temp_dir_last_ts.get(directory, 0.0) < _SWEEP_TEMP_DIR_INTERVAL:
        return 0
    _sweep_temp_dir_last_ts[directory] = now

    if not directory.exists():
        return 0
    cutoff = now - max_age_hours * 3600
    removed = 0
    for f in directory.iterdir():
        try:
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def _sweep_dir_by_size(directory: Path, max_bytes: int) -> int:
    """Delete oldest files until *directory* totals ≤ max_bytes. Returns bytes freed.

    Age-based sweeping alone can't bound disk: opening many large files within the
    retention window grows the cache unbounded. This caps total size, evicting the
    least-recently-modified files first.
    """
    if not directory.exists():
        return 0
    try:
        files = [(f.stat().st_mtime, f.stat().st_size, f)
                 for f in directory.iterdir() if f.is_file()]
    except OSError:
        return 0
    total = sum(size for _, size, _ in files)
    if total <= max_bytes:
        return 0
    files.sort()  # oldest mtime first
    freed = 0
    for _, size, f in files:
        if total - freed <= max_bytes:
            break
        try:
            f.unlink()
            freed += size
        except OSError:
            pass
    return freed


_RUNNING_TASK_MAX_AGE_S = 6 * 3600  # orphaned "running" tasks older than this are evicted
_MAX_TERMINAL_TASKS = 50            # hard cap on retained done/error tasks (memory bound)


def _sweep_tasks(max_age_hours: float = 1.0) -> None:
    """Remove terminal and orphaned-running tasks from _tasks.

    Terminal tasks (done/error) older than *max_age_hours* are removed.
    Running tasks older than _RUNNING_TASK_MAX_AGE_S are also evicted to
    reclaim memory from client disconnects and browser refreshes that left
    a task in the "running" state indefinitely.
    """
    now = time.time()
    cutoff = now - max_age_hours * 3600
    with _tasks_lock:
        stale = [
            tid for tid, t in _tasks.items()
            if (t.get("status") in ("done", "error") and t.get("created_at", 0) < cutoff)
            or (t.get("status") == "running" and now - t.get("started_at", now) > _RUNNING_TASK_MAX_AGE_S)
        ]
        for tid in stale:
            del _tasks[tid]

        # Hard cap on terminal tasks regardless of age — a burst of analyses
        # (each result holds the waveform + word list, ~MBs) must not grow
        # unbounded between hourly age sweeps. Evict the oldest past the cap.
        terminal = sorted(
            (t.get("created_at", 0), tid)
            for tid, t in _tasks.items()
            if t.get("status") in ("done", "error")
        )
        for _, tid in terminal[: max(0, len(terminal) - _MAX_TERMINAL_TASKS)]:
            del _tasks[tid]


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/capabilities")
def api_capabilities():
    """First-run dependency preflight report.

    Flat, cheap, synchronous checks only — no model loads, no network calls —
    so the frontend can call this once at launch and get an answer instantly:

        {
          "whisper_available": bool,  # faster-whisper or openai-whisper installed
          "ffmpeg":   bool,           # required — media extraction
          "ffprobe":  bool,           # required — media metadata
          "whisperx": bool,           # optional — tighter word-alignment refinement
          "japanese": bool,           # optional — fugashi/UniDic phrase-boundary breaks
        }

    ffmpeg/ffprobe existence is checked via shutil.which() rather than by
    triggering extract_audio()/probe_media()'s own exception paths, so a
    missing binary is reported instantly instead of only being discovered
    mid-analysis.

    Ollama is intentionally NOT reported here. Unlike the fields above, it
    requires a live network round-trip to the local Ollama daemon
    (llm_correct.ollama_available()) and a slow/absent Ollama must never
    delay this report — especially the ffmpeg/ffprobe check, which gates a
    hard "this app can't run" blocker that has to appear immediately. The
    LLM-correction feature is already fully opt-in/manually-triggered, so its
    existing lazy check (GET /api/llm/models, already polled at startup by
    the LLM-correct UI) is the right place for that — see checkCaps() in
    static/modules/analysis.js, which fetches both and never lets the Ollama
    fetch block the ffmpeg/ffprobe blocker from rendering.
    """
    return jsonify({
        "whisper_available": WHISPER_AVAILABLE,
        "ffmpeg":   shutil.which("ffmpeg") is not None,
        "ffprobe":  shutil.which("ffprobe") is not None,
        "whisperx": WHISPERX_AVAILABLE,
        "japanese": japanese.available(),
    })


@app.route("/api/debug/logs", methods=["GET"])
def api_debug_logs():
    """Return the last 500 lines from the rotating log file.

    Always returns 200. ``lines`` is empty if the log file doesn't exist yet.
    """
    lines: list[str] = []
    try:
        with LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip("\n") for l in collections.deque(f, maxlen=500)]
    except (OSError, FileNotFoundError):
        pass  # log not written yet — return empty list
    return jsonify({"log_path": str(LOG_PATH), "lines": lines})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accept a multipart file upload, save it to _UPLOAD_DIR, and return its path.

    Stale uploads (>24 h) are swept opportunistically on each call.
    Returns: {"path": "/tmp/preprod_uploads/<filename>"}  on success
             {"error": "..."} with 400/500  on failure.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Sweep stale uploads opportunistically on each new upload.
    _sweep_temp_dir(_UPLOAD_DIR, max_age_hours=_UPLOAD_KEEP_DAYS * 24)
    # Prefix with a short UUID so two files named "recording.mov" never
    # silently overwrite each other — a safety-critical fix for data loss.
    dest = _UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{f.filename}"
    try:
        f.save(str(dest))
    except OSError as exc:
        log.error("Upload save failed for %s: %s", dest, exc)
        return jsonify({"error": f"Could not save uploaded file: {exc}"}), 500
    return jsonify({"path": str(dest)})


@app.route("/api/filepicker", methods=["POST"])
def api_filepicker():
    """Open a native macOS file-open dialog via pywebview. Desktop app only."""
    try:
        import webview  # type: ignore
        wins = webview.windows
        if not wins:
            return jsonify({"error": "No webview window"}), 400
        result = wins[0].create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=(
                # First entry is the default — include both video and audio so
                # MP3/AAC/WAV are visible without changing the filter dropdown.
                "All Media (*.mp4;*.mov;*.m4v;*.mxf;*.avi;*.mkv;*.mp3;*.aac;*.wav;*.m4a;*.flac)",
                "Video Files (*.mp4;*.mov;*.m4v;*.mxf;*.avi;*.mkv)",
                "Audio Files (*.mp3;*.aac;*.wav;*.m4a;*.flac)",
                "All Files (*.*)",
            ),
        )
        if result:
            return jsonify({"path": result[0]})
        return jsonify({"path": None})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/probe", methods=["POST"])
def api_probe():
    data = request.get_json() or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "No path"}), 400
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return jsonify({"error": f"File not found: {p}"}), 404
    try:
        media = probe_media(p)
        _cache_media(str(p), media)  # key by resolved path for cache consistency
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    if not media.has_audio:
        return jsonify({"error": "File has no audio track"}), 400
    return jsonify({"media": _media_to_dict(media)})


def _cancel_running_analyses() -> int:
    """Signal every in-flight analysis to cancel; return how many were signalled.

    Starting a new analysis (loading a new file or re-analyzing) supersedes any
    previous one — the old Whisper worker should stop, not keep burning CPU and
    competing with the new run (T0251). The worker checks its cancel_event each
    second and is then terminated + reaped by transcribe()'s cleanup.
    """
    n = 0
    with _tasks_lock:
        for t in _tasks.values():
            if t.get("status") == "running":
                ev = t.get("cancel_event")
                if ev is not None and not ev.is_set():
                    ev.set()
                    n += 1
    return n


@app.route("/api/analyze/start", methods=["POST"])
def api_analyze_start():
    """Start background analysis. Returns task_id for polling."""
    data = request.get_json() or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "No path"}), 400
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return jsonify({"error": f"File not found: {p}"}), 404

    # Supersede any analysis already in flight so it stops instead of running on.
    _cancel_running_analyses()

    task_id = uuid.uuid4().hex
    with _tasks_lock:
        _tasks[task_id] = {
            "status":     "running",
            "progress":   0,
            "stage":      "starting",
            "result":     None,
            "error":      None,
            "created_at": time.time(),
            "started_at": time.time(),
        }

    # Purge terminal tasks older than 1 hour to prevent unbounded memory growth.
    _sweep_tasks(max_age_hours=1.0)

    hangover_ms = int(data.get("hangover_ms", 300))

    thread = threading.Thread(
        target=_run_analysis,
        args=(task_id, str(p), data),
        kwargs={"hangover_ms": hangover_ms},
        daemon=True,
    )
    thread.start()

    return jsonify({"task_id": task_id})


@app.route("/api/analyze/status/<task_id>")
def api_analyze_status(task_id: str):
    """Poll the status of a background analysis task.

    Returns the full task dict: {status, progress, stage, result, error}.
    status is one of: "running", "done", "error".
    Returns 404 if the task_id is unknown (e.g. already swept by _sweep_tasks).
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        return jsonify({"error": "Unknown task"}), 404
    # Strip non-serialisable cancel_event before returning to client
    safe_task = {k: v for k, v in task.items() if k != "cancel_event"}
    return jsonify(safe_task)


@app.route("/api/analyze/cancel", methods=["POST"])
def api_analyze_cancel():
    """Signal a running analysis task to cancel.

    Body JSON: {"task_id": "<uuid>"}
    Returns {"ok": true} on success, 404 if the task_id is unknown.
    """
    data = request.get_json() or {}
    task_id = data.get("task_id", "")
    with _tasks_lock:
        task = _tasks.get(task_id)
    if task is None:
        return jsonify({"error": "Unknown task"}), 404
    ev = task.get("cancel_event")
    if ev:
        ev.set()
    return jsonify({"ok": True})


@app.route("/api/analyze/redetect_silence", methods=["POST"])
def api_analyze_redetect_silence():
    """Re-run only the RMS silence detector with new threshold/min-duration.

    Whisper is not re-run.  If a previous full analysis was run for this file,
    word-boundary snapping is applied using the cached word timestamps — making
    threshold-slider adjustments consistent with the full-analysis result.
    Fillers are not touched — the caller merges the new silence candidates back
    with existing state.
    """
    data = request.get_json() or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "path required"}), 400
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return jsonify({"error": f"File not found: {p}"}), 404

    threshold_db  = float(data.get("threshold_db", -40.0))
    min_duration  = float(data.get("min_duration",  1.0))
    hangover_ms   = int(data.get("hangover_ms", 300))

    try:
        samples = _get_cached_audio(p)
        silence_regions = detect_silence(
            samples,
            sample_rate=16000,
            threshold_db=threshold_db,
            min_duration=min_duration,
            hangover_ms=hangover_ms,
        )
        # Apply word-boundary snapping when a previous analysis cached the words.
        # This keeps redetect boundaries consistent with full-analysis boundaries.
        cached_words = _get_cached_words(str(p))
        if cached_words:
            silence_regions = snap_silences_to_words(silence_regions, cached_words)
        total_duration = round(sum(e - s for s, e in silence_regions), 2)
        return jsonify({
            "candidates":     _silence_candidates(silence_regions),
            "threshold_db":   threshold_db,
            "hangover_ms":    hangover_ms,
            "total_duration": total_duration,
        })
    except ValueError as exc:
        log.warning("redetect_silence rejected: %s", exc)
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        log.exception("redetect_silence failed")
        return jsonify({"error": str(exc)}), 500


# ── Local-LLM transcript correction (optional, manually triggered) ───────────
# See src/preprod/llm_correct.py and 04_Context/vidtighten-llm-correct.md. The
# backend only PROPOSES brand-noun fixes; the frontend reviews and applies them.

@app.route("/api/llm/models")
def api_llm_models():
    """List installed Ollama models + a default pick, for the model selector.

    {available: bool, models: [{name, loaded}], default: name|null}
    available=false means Ollama isn't reachable — the UI disables the feature.
    """
    available = llm_correct.ollama_available()
    models = llm_correct.list_models() if available else []
    return jsonify({
        "available": available,
        "models": models,
        "default": llm_correct.pick_default_model(models),
    })


def _run_llm_suggest(task_id: str, transcript_text: str, model: str) -> None:
    """Background worker: ask the local LLM for brand-noun corrections.

    Seeds the model with the active glossary (built-in defaults + user glossary)
    so fix direction is reliable — the correct names are ground truth, the known
    wrong→right pairs are few-shot examples. Result is anchor-validated inside
    suggest_brand_corrections before it lands in the task result.
    """
    try:
        corr = active_corrections()
        known_names = sorted(set(corr.values()))
        known_examples = [{"wrong": w, "correct": r} for w, r in corr.items()]
        result = llm_correct.suggest_brand_corrections(
            transcript_text, known_names, known_examples, model,
        )
        with _tasks_lock:
            if task_id in _tasks:
                _tasks[task_id].update({
                    "status": "done" if result["status"] == "ok" else "error",
                    "result": result,
                    "error": result.get("error"),
                })
    except Exception as exc:  # never let the worker thread die silently
        log.exception("llm suggest failed")
        with _tasks_lock:
            if task_id in _tasks:
                _tasks[task_id].update({"status": "error", "error": str(exc)})


@app.route("/api/llm/suggest", methods=["POST"])
def api_llm_suggest():
    """Start a background brand-noun correction pass. Returns {task_id}.

    Body: {transcript_text: str, model: str}. Poll /api/analyze/status/<task_id>;
    the result field holds {status, fixes: [{wrong, correct, count}], error}.
    """
    data = request.get_json() or {}
    transcript_text = (data.get("transcript_text") or "").strip()
    model = (data.get("model") or "").strip()
    if not transcript_text:
        return jsonify({"error": "transcript_text required"}), 400
    if not model:
        return jsonify({"error": "model required"}), 400

    task_id = uuid.uuid4().hex
    with _tasks_lock:
        _tasks[task_id] = {
            "status": "running", "progress": 0, "stage": "llm_correct",
            "result": None, "error": None,
            "created_at": time.time(), "started_at": time.time(),
        }
    _sweep_tasks(max_age_hours=1.0)
    threading.Thread(
        target=_run_llm_suggest, args=(task_id, transcript_text, model), daemon=True,
    ).start()
    return jsonify({"task_id": task_id})


@app.route("/api/llm/glossary/add", methods=["POST"])
def api_llm_glossary_add():
    """Persist an approved brand correction to the user glossary so future
    transcriptions fix it deterministically (the "always apply this" action).

    Body: {wrong: str, correct: str}. Idempotent.
    """
    data = request.get_json() or {}
    wrong = (data.get("wrong") or "").strip()
    correct = (data.get("correct") or "").strip()
    if not wrong or not correct:
        return jsonify({"error": "wrong and correct required"}), 400
    save_user_correction(wrong, correct)
    return jsonify({"ok": True})


@app.route("/api/stream")
def api_stream():
    """Stream a local media file with Range request support for seeking."""
    # DNS-rebinding guard now applies app-wide via _require_localhost_host().
    path = request.args.get("path", "").strip()
    if not path:
        return "No path", 400
    p = Path(path).resolve()

    # Allowlist before existence check: avoids leaking file existence outside
    # approved dirs (attacker distinguishing 404 vs 403 is information).
    if not _is_path_allowed(p):
        return "Forbidden", 403
    if not p.exists():
        return "Not found", 404

    mime_map = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".m4v": "video/x-m4v",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".mxf": "application/mxf",
        ".mp3": "audio/mpeg",
        ".aac": "audio/aac",
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
    }
    # VBR MP3 → serve M4A sidecar when ready for sample-accurate currentTime.
    # _start_m4a_transcode is a no-op once the sidecar is cached on disk.
    if _is_vbr_mp3(p):
        _start_m4a_transcode(p)
        m4a = _m4a_cache_path(p)
        if m4a.exists():
            return send_file(m4a, mimetype="audio/mp4", conditional=True)

    # High-resolution video → serve a low-res preview proxy when ready (this
    # endpoint is the ONLY consumer swapped; audio analysis and export always
    # use the original path directly, never /api/stream).
    # _start_video_proxy_transcode is a no-op once cached or already running.
    with _media_cache_lock:
        media = _media_cache.get(str(p))
    if media is None:
        try:
            media = probe_media(p)
            _cache_media(str(p), media)
        except Exception:
            media = None   # probing failed — fall through to serving the original
    if media is not None and media.has_video:
        _start_video_proxy_transcode(p, media)
        proxy = _proxy_cache_path(p)
        if proxy.exists():
            return send_file(proxy, mimetype="video/mp4", conditional=True)

    mime = mime_map.get(p.suffix.lower(), "application/octet-stream")
    return send_file(p, mimetype=mime, conditional=True)


@app.route("/api/media/transcode_status")
def api_transcode_status():
    """Return M4A background-transcode status for a VBR MP3 source file.

    Response JSON:
        {"status": "ready" | "pending" | "failed" | "not_applicable" | "not_started"}

    "ready"          — M4A sidecar exists on disk; next /api/stream call serves it.
    "pending"        — ffmpeg is currently transcoding.
    "failed"         — ffmpeg exited non-zero; original MP3 will always be served.
    "not_applicable" — file is not a VBR MP3 (video, WAV, CBR MP3 …).
    "not_started"    — VBR MP3 detected but transcoding hasn't been triggered yet.
    """
    path = request.args.get("path", "").strip()
    if not path:
        return jsonify({"status": "not_applicable"})
    p = Path(path).resolve()
    if not _is_path_allowed(p):
        return jsonify({"status": "not_applicable"})
    if not _is_vbr_mp3(p):
        return jsonify({"status": "not_applicable"})
    # On-disk check is authoritative (survives process restarts)
    if _m4a_cache_path(p).exists():
        with _TRANSCODE_STATE_LOCK:
            _TRANSCODE_STATE[_mp3_cache_key(p)] = "ready"
        return jsonify({"status": "ready"})
    with _TRANSCODE_STATE_LOCK:
        state = _TRANSCODE_STATE.get(_mp3_cache_key(p), "not_started")
    return jsonify({"status": state})


@app.route("/api/media/proxy_status")
def api_proxy_status():
    """Return low-res preview-proxy background-transcode status for a video file.

    Response JSON:
        {"status": "ready" | "pending" | "failed" | "not_applicable" | "not_started"}

    "ready"          — proxy exists on disk; next /api/stream call serves it.
    "pending"        — ffmpeg is currently transcoding.
    "failed"         — ffmpeg exited non-zero; original file will always be served.
    "not_applicable" — file has no video track, or resolution is already small
                        enough that hardware/software decode isn't a bottleneck.
    "not_started"    — proxy needed but transcoding hasn't been triggered yet.
    """
    path = request.args.get("path", "").strip()
    if not path:
        return jsonify({"status": "not_applicable"})
    p = Path(path).resolve()
    if not _is_path_allowed(p) or not p.exists():
        return jsonify({"status": "not_applicable"})
    with _media_cache_lock:
        media = _media_cache.get(str(p))
    if media is None:
        try:
            media = probe_media(p)
            _cache_media(str(p), media)
        except Exception:
            return jsonify({"status": "not_applicable"})
    if not _needs_proxy(media):
        return jsonify({"status": "not_applicable"})
    # Ensure the transcode is actually running — this endpoint is now the ONLY
    # trigger point for files whose original is never streamed (the frontend
    # deliberately skips loading an original that might have no decode path
    # in WKWebView at all; see loadFilePath's proxy-pending flow). No-op if
    # already cached/pending, matching _start_m4a_transcode's idempotency.
    _start_video_proxy_transcode(p, media)
    # On-disk check is authoritative (survives process restarts)
    if _proxy_cache_path(p).exists():
        with _PROXY_STATE_LOCK:
            _PROXY_STATE[_proxy_cache_key(p)] = "ready"
        return jsonify({"status": "ready"})
    with _PROXY_STATE_LOCK:
        state = _PROXY_STATE.get(_proxy_cache_key(p), "not_started")
    return jsonify({"status": state})


def _ts_srt(sec: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm).

    Rounds to the nearest millisecond first and derives all fields from the
    integer total — avoids "00:00:01,1000" when sec%1 rounds to 1.0.
    """
    total_ms = int(round(sec * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(sec: float) -> str:
    """Format seconds as WebVTT timestamp (HH:MM:SS.mmm)."""
    return _ts_srt(sec).replace(",", ".")


def _copy_to_downloads(src: Path, filename: str) -> tuple[bool, str]:
    """Copy *src* to ~/Downloads with collision-safe naming.

    Used by export endpoints when the pywebview client sets save_to_downloads=true.
    Returning the file as an HTTP attachment would trigger pywebview's cocoa
    WKNavigationResponsePolicyCancel handler (application/xml can't be rendered),
    which blanks the WKWebView page.  Writing server-side bypasses that path entirely.

    Returns (ok, error_message).  error_message is empty on success.
    """
    downloads = Path.home() / "Downloads"
    try:
        downloads.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"Cannot create ~/Downloads: {exc}"

    dest = downloads / filename
    if dest.exists():
        stem, suffix = Path(filename).stem, Path(filename).suffix
        for n in range(1, 100):
            dest = downloads / f"{stem} ({n}){suffix}"
            if not dest.exists():
                break

    try:
        shutil.copy2(src, dest)
    except OSError as exc:
        return False, str(exc)

    return True, ""


@app.route("/api/export/roughcut", methods=["POST"])
def api_export_roughcut():
    """Generate rough-cut FCPXML and return it as a file download."""
    data = request.get_json() or {}
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "No path"}), 400

    p = Path(path).expanduser().resolve()
    if not p.exists():
        # Give actionable guidance when an uploaded file has been swept or the
        # session was created before the persistent-upload-dir migration.
        was_upload = (
            str(p).startswith(str(_UPLOAD_DIR))
            # legacy: old sessions reference the volatile tempfile.gettempdir() path
            or "preprod_uploads" in str(p)
        )
        if was_upload:
            return jsonify({
                "error": (
                    f"The uploaded file is no longer available: {p.name}\n\n"
                    "Uploaded files are stored temporarily. "
                    "Please click \u201cOpen File\u201d and re-import the file to export."
                )
            }), 404
        return jsonify({"error": f"File not found: {p}"}), 404

    # Re-use cached media info or probe fresh.
    # Always look up by resolved path — matches how _run_analysis stores it.
    with _media_cache_lock:
        media = _media_cache.get(str(p))
    if media is None:
        try:
            media = probe_media(p)
            _cache_media(str(p), media)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    padding_ms = int(data.get("padding_ms", 200))

    # Word-type regions: refine boundaries + zero padding (already snapped to audio).
    # Non-word regions: normal padding to preserve context audio around cuts.
    segments = _build_segments_typed(
        data.get("removal_regions", []), p, media.duration, padding_ms,
        threshold_db=float(data.get("threshold_db", -40.0)),
    )
    if not segments:
        return jsonify({"error": "No keep-segments after applying removals"}), 400

    _EXPORT_DIR.mkdir(exist_ok=True)
    _sweep_temp_dir(_EXPORT_DIR, max_age_hours=24)
    out_path = _EXPORT_DIR / f"{p.stem}_cut.fcpxml"

    try:
        generate_roughcut_fcpxml(segments, media, out_path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    # pywebview path: write directly to ~/Downloads so the JS side never has to
    # fetch a blob.  Fetching application/xml through WKWebView triggers
    # decidePolicyForNavigationResponse → WKNavigationResponsePolicyCancel,
    # which blanks the page.
    if data.get("save_to_downloads"):
        ok, err = _copy_to_downloads(out_path, f"{p.stem}_cut.fcpxml")
        if not ok:
            return jsonify({"ok": False, "error": err}), 500
        return jsonify({"ok": True})

    return send_file(
        out_path,
        as_attachment=True,
        download_name=f"{p.stem}_cut.fcpxml",
        mimetype="application/xml",
    )


@app.route("/api/export/telop", methods=["POST"])
def api_export_telop():
    """Generate telop FCPXML with time-adjusted title positions."""
    data = request.get_json() or {}
    path = data.get("path", "").strip()

    # Duration needed for time mapping
    duration = float(data.get("duration", 0))
    telop_entries_raw = data.get("telop_entries", [])
    if duration <= 0 and telop_entries_raw:
        return jsonify({"error": "duration must be > 0 when telop entries are present"}), 400

    _telop_path = Path(path) if path else None
    padding_ms = int(data.get("padding_ms", 200))
    # Word-type regions: refine + zero padding; non-word: normal padding.
    keep_segs = _build_segments_typed(
        data.get("removal_regions", []),
        _telop_path if (_telop_path and _telop_path.exists()) else None,
        duration, padding_ms,
        threshold_db=float(data.get("threshold_db", -40.0)),
    ) if duration else []

    telop_entries = telop_entries_raw
    settings = data.get("settings", {})
    stem = data.get("stem") or (Path(path).stem if path else "telop")
    use_source_timing = bool(data.get("use_source_timing", False))

    _EXPORT_DIR.mkdir(exist_ok=True)
    _sweep_temp_dir(_EXPORT_DIR, max_age_hours=24)
    out_path = _EXPORT_DIR / f"{stem}_telop.fcpxml"

    try:
        generate_telop_fcpxml(
            telop_entries,
            keep_segs,
            total_source_duration=duration,
            settings=settings,
            stem=stem,
            output_path=out_path,
            use_source_timing=use_source_timing,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    if data.get("save_to_downloads"):
        ok, err = _copy_to_downloads(out_path, f"{stem}_telop.fcpxml")
        if not ok:
            return jsonify({"ok": False, "error": err}), 500
        return jsonify({"ok": True})

    return send_file(
        out_path,
        as_attachment=True,
        download_name=f"{stem}_telop.fcpxml",
        mimetype="application/xml",
    )


@app.route("/api/export/subtitles", methods=["POST"])
def api_export_subtitles():
    """Export transcript as SRT or VTT subtitle file, time-adjusted for cuts.

    Timestamps are mapped from source (Whisper) to output (post-cut) timeline
    using the same map_span_to_output logic as the telop FCPXML export, so
    subtitles align with the cut video.

    Body JSON:
        format           "srt" | "vtt"   (default "srt")
        telop_entries    [{start, end, text}]  — source-time Whisper segments
        removal_regions  [{start, end}]  — regions being removed
        duration         float  — source total duration (seconds)
        padding_ms       int    — padding applied to removals
        stem             str    — base filename for download

    Returns subtitle file as attachment.
    """
    data = request.get_json() or {}
    fmt = data.get("format", "srt").lower()
    if fmt not in ("srt", "vtt"):
        return jsonify({"error": "format must be 'srt' or 'vtt'"}), 400

    telop_entries: list[dict] = data.get("telop_entries", [])
    duration = float(data.get("duration", 0))
    padding_ms = int(data.get("padding_ms", 200))

    _sub_path_str = data.get("path", "").strip()
    _sub_path = Path(_sub_path_str) if _sub_path_str else None
    stem = data.get("stem") or "subtitles"

    # Word-type regions: refine + zero padding; non-word: normal padding.
    keep_segs = _build_segments_typed(
        data.get("removal_regions", []),
        _sub_path if (_sub_path and _sub_path.exists()) else None,
        duration, padding_ms,
        threshold_db=float(data.get("threshold_db", -40.0)),
    ) if duration else []

    lines: list[str] = []
    if fmt == "vtt":
        lines.append("WEBVTT\n")

    idx = 1
    for entry in filter_telop_entries(telop_entries):
        span = map_span_to_output(entry["start"], entry["end"], keep_segs)
        if span is None:
            continue  # segment falls entirely within a removed region
        out_s, out_e = span
        # Apply custom-vocabulary corrections at the export chokepoint so subtitles
        # are correct regardless of how the entry text was built upstream.
        text = correct_text(entry.get("text", "")).strip()
        if not text:
            continue

        ts_fn = _ts_vtt if fmt == "vtt" else _ts_srt
        if fmt == "srt":
            lines.append(str(idx))
        lines.append(f"{ts_fn(out_s)} --> {ts_fn(out_e)}")
        lines.append(text)
        lines.append("")
        idx += 1

    content = "\n".join(lines)
    ext = fmt
    mime = "text/vtt" if fmt == "vtt" else "application/x-subrip"

    if data.get("save_to_downloads"):
        # Write to a temp file first, then copy to ~/Downloads.
        _EXPORT_DIR.mkdir(exist_ok=True)
        tmp = _EXPORT_DIR / f"{stem}_subtitles.{ext}"
        try:
            tmp.write_text(content, encoding="utf-8")
        except OSError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
        ok, err = _copy_to_downloads(tmp, f"{stem}_subtitles.{ext}")
        if not ok:
            return jsonify({"ok": False, "error": err}), 500
        return jsonify({"ok": True})

    return Response(
        content,
        mimetype=mime,
        headers={
            "Content-Disposition": f'attachment; filename="{stem}_subtitles.{ext}"',
        },
    )


@app.route("/api/cache/info")
def api_cache_info():
    """Return sizes of upload, export, and session caches."""
    from preprod.session import sessions_size_bytes, sessions_count

    def _dir_bytes(d: Path) -> int:
        if not d.exists():
            return 0
        return sum(f.stat().st_size for f in d.iterdir() if f.is_file())

    upload_bytes = _dir_bytes(_UPLOAD_DIR)
    export_bytes = _dir_bytes(_EXPORT_DIR)
    session_bytes = sessions_size_bytes()
    session_count = sessions_count()
    total = upload_bytes + export_bytes + session_bytes
    return jsonify({
        "upload_bytes":   upload_bytes,
        "export_bytes":   export_bytes,
        "session_bytes":  session_bytes,
        "session_count":  session_count,
        "total_bytes":    total,
    })


@app.route("/api/cache/clear", methods=["POST"])
def api_cache_clear():
    """Delete all temp uploads, exports, and (optionally) session files."""
    from preprod.session import clear_all_sessions

    data = request.get_json() or {}
    clear_sessions = bool(data.get("sessions", False))

    removed_uploads = 0
    removed_exports = 0
    for d in (_UPLOAD_DIR, _EXPORT_DIR):
        if d.exists():
            for f in list(d.iterdir()):
                try:
                    if f.is_file():
                        f.unlink()
                        if d == _UPLOAD_DIR:
                            removed_uploads += 1
                        else:
                            removed_exports += 1
                except OSError:
                    pass

    removed_sessions = 0
    if clear_sessions:
        removed_sessions = clear_all_sessions()

    return jsonify({
        "removed_uploads":  removed_uploads,
        "removed_exports":  removed_exports,
        "removed_sessions": removed_sessions,
    })


@app.route("/api/session/list")
def api_session_list():
    """Return the last 20 saved sessions, newest first."""
    return jsonify({"sessions": list_sessions(limit=20)})


@app.route("/api/session/save", methods=["POST"])
def api_session_save():
    data = request.get_json() or {}
    path = data.get("path", "")
    state = data.get("state", {})
    if not path:
        return jsonify({"ok": False}), 400
    try:
        save_session(path, state)
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/session/load", methods=["POST"])
def api_session_load():
    data = request.get_json() or {}
    path = data.get("path", "")
    if not path:
        return jsonify({"session": None})
    session = load_session(path)
    return jsonify({"session": session})


@app.route("/api/session/delete", methods=["POST"])
def api_session_delete():
    data = request.get_json() or {}
    path = data.get("path", "")
    if path:
        delete_session(path)
    return jsonify({"ok": True})


# ── CLI entry point ──────────────────────────────────────────────────────────

def _check_remote_bind(host: str, allow_remote: bool) -> None:
    """Enforce the --allow-remote gate for non-loopback --host values.

    There is no authentication anywhere in this app — every route trusts "the
    client is the user, on this machine." Binding beyond loopback turns that
    into "anyone who can reach this host:port has full file access," so it
    needs an explicit, loud opt-in rather than a bare --host flag. Raises
    click.UsageError (clean CLI exit) when the gate isn't satisfied; otherwise
    prints a loud warning for a remote bind and returns.
    """
    if host in _ALLOWED_HOSTS:
        return
    if not allow_remote:
        raise click.UsageError(
            f"Refusing to bind to {host!r} without --allow-remote: this app has no "
            "login or access control, so anyone who can reach this host:port would "
            "have full read/write access to your files. Pass --allow-remote to "
            "acknowledge and proceed."
        )
    click.secho(
        f"⚠  Binding to {host} with NO AUTHENTICATION. Anyone on this "
        "network who can reach this address has full access to your files.",
        fg="red", bold=True, err=True,
    )


@click.command("preprod-web")
@click.option("--port", default=9877, help="Port to run on.")
@click.option("--host", default="127.0.0.1", help="Host to bind to.")
@click.option(
    "--allow-remote", is_flag=True, default=False,
    help="Required to bind --host to anything other than localhost.",
)
def run_web(port: int, host: str, allow_remote: bool) -> None:
    """Start the VidTighten web UI."""
    _check_remote_bind(host, allow_remote)

    # Sweep stale files on startup so old uploads/exports don't accumulate.
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    _sweep_temp_dir(_UPLOAD_DIR, max_age_hours=_UPLOAD_KEEP_DAYS * 24, force=True)
    _sweep_temp_dir(_EXPORT_DIR, max_age_hours=24, force=True)

    from preprod.session import expire_old_sessions
    expire_old_sessions(days=30)

    click.echo(f"VidTighten: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
