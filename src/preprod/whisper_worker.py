#!/usr/bin/env python3
"""Whisper worker — executed as a subprocess to keep torch/libomp isolated
from the main process (which has numpy/OpenBLAS libomp already loaded).

Usage:
    python whisper_worker.py '<json-args>'

Args JSON keys:
    path        str   — audio/video file path
    model_size  str   — e.g. "turbo", "base", "large-v3"
    language    str|null — BCP-47 language code or null for auto-detect
    duration    float — file duration in seconds (used for progress tracking)

Output: single JSON line on stdout:
    {"segments": [...], "words": [...], "language": "..."}

Progress: JSON lines on stderr during transcription:
    {"progress": N}  (N in 0–99)

Errors: non-zero exit code + message on stderr.

Backend selection:
    1. faster-whisper (CTranslate2 INT8) — 4–8× faster on CPU, no MPS sparse-tensor crash
    2. openai-whisper — fallback when faster-whisper is absent or fails
"""

from __future__ import annotations

import json
import os
import sys
import time

# Suppress the OMP duplicate warning (the subprocess has only one libomp)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Bilingual prompt seeds the decoder for mixed Japanese/English tech + UX/design content.
# Domain vocabulary prevents proper nouns and specialised terms from being mis-transcribed
# as phonetic gibberish.  Well under the 224-token limit for both backends.
_INITIAL_PROMPT = (
    "以下は日本語と英語が混在した会話の文字起こしです。"
    # AI / dev terms — proper nouns Whisper may mis-transcribe as phonetics.
    # NOTE: prompt is capped at 224 Whisper tokens; common English terms
    # (Python, JS, React, Docker, OpenAI, GPT, etc.) are omitted because
    # Whisper's training data already covers them — keeping only rarer terms
    # and Japanese-context words that risk phonetic mis-transcription.
    "Claude, Claude Code, Anthropic, API, TypeScript, Next.js, VS Code, LLM, GitHub, Cursor, Copilot, "
    "デプロイ, リポジトリ, プロンプト, 生成AI, "
    # UX/design terms
    "Figma, Notion, ユーザー, プロトタイプ, "
    "ワイヤーフレーム, コンポーネント, デザインシステム, "
    "アクセシビリティ, フィードバック, ユーザビリティ, "
    # Creator/VidTighten terms
    "YouTube, サムネイル, チャンネル, 登録者, 動画, 編集, 字幕"
)

# openai-whisper calls the "turbo" (large-v3-turbo) model just "turbo";
# faster-whisper uses the full name.
_FW_MODEL_MAP: dict[str, str] = {
    "turbo": "large-v3-turbo",
}


def _run_faster_whisper(
    path: str,
    model_size: str,
    language: str | None,
    duration: float | None,
) -> dict:
    """Run transcription with faster-whisper (CTranslate2 INT8 backend).

    Uses CPU int8 — avoids the MPS sparse-tensor crash that makes openai-whisper
    always fall back to slow CPU anyway.  CTranslate2 int8 is 4–8× faster than
    openai-whisper on CPU for the same model size.

    Emits {"progress": N} JSON lines on stderr as segments are decoded.
    """
    from faster_whisper import WhisperModel  # type: ignore

    fw_model = _FW_MODEL_MAP.get(model_size, model_size)
    print(f"faster-whisper: loading {fw_model!r} (cpu, int8)", file=sys.stderr)
    _t_load0 = time.perf_counter()
    model = WhisperModel(fw_model, device="cpu", compute_type="int8")
    _t_model_load = time.perf_counter() - _t_load0

    segments_gen, info = model.transcribe(
        str(path),
        language=language,
        word_timestamps=True,
        # Prevent hallucination loops on long silences
        condition_on_previous_text=False,
        initial_prompt=_INITIAL_PROMPT,
        # Raise no-speech gate from default 0.6 → 0.7 to reduce phantom segments
        no_speech_threshold=0.7,
        # VAD filter suppresses transcription on silent sections — equivalent to
        # openai-whisper's hallucination_silence_threshold
        vad_filter=True,
    )

    total_dur = float(duration or info.duration or 1.0)
    segments: list[dict] = []
    words: list[dict] = []
    last_pct = -1

    # faster-whisper decodes lazily — the real inference work happens while
    # iterating the generator, not in the model.transcribe() call above.
    _t_infer0 = time.perf_counter()
    for seg_idx, seg in enumerate(segments_gen):
        text = seg.text.strip()
        if text:
            segments.append({
                "start":  round(float(seg.start), 3),
                "end":    round(float(seg.end),   3),
                "text":   text,
                "seg_id": seg_idx,
            })
        if seg.words:
            for w in seg.words:
                word_text = w.word.strip()
                if word_text:
                    words.append({
                        "word":   word_text,
                        "start":  round(float(w.start), 3),
                        "end":    round(float(w.end),   3),
                        "score":  round(float(w.probability), 3),
                        "seg_id": seg_idx,
                    })
        pct = min(99, int(seg.end / total_dur * 100))
        if pct != last_pct:
            print(json.dumps({"progress": pct}), file=sys.stderr, flush=True)
            last_pct = pct

    return {
        "segments": segments,
        "words":    words,
        "language": info.language or "",
        "timings": {
            "model_load": round(_t_model_load, 3),
            "inference":  round(time.perf_counter() - _t_infer0, 3),
        },
    }


def _run_openai_whisper(
    path: str,
    model_size: str,
    language: str | None,
    duration: float | None,
) -> dict:
    """Run transcription with openai-whisper (fallback backend).

    Attempts MPS first (Apple Silicon), retries on CPU when MPS raises
    sparse-tensor errors.  Emits {"progress": N} JSON lines on stderr.
    """
    import inspect as _inspect
    import math as _math
    import torch
    import whisper  # type: ignore

    # hallucination_silence_threshold was added in openai-whisper 20250625.
    # Probe the signature so older versions (or test stubs) don't crash with TypeError.
    try:
        _has_hal_thresh = (
            "hallucination_silence_threshold"
            in _inspect.signature(whisper.transcribe).parameters
        )
    except AttributeError:
        # Module-level transcribe() absent (e.g. test stub or very old build).
        _has_hal_thresh = False
    if not _has_hal_thresh:
        print(
            "openai-whisper < 20250625: hallucination_silence_threshold unavailable;"
            " silent-section hallucinations may occur",
            file=sys.stderr,
        )

    transcribe_kwargs = dict(
        word_timestamps=True,
        language=language,
        verbose=False,
        # Prevent hallucination loops on silent regions
        condition_on_previous_text=False,
        initial_prompt=_INITIAL_PROMPT,
        # Raise no-speech gate from default 0.6 → 0.7
        no_speech_threshold=0.7,
        # Skip decoding on 2+ second silent segments — requires Whisper ≥ 20250625
        **({"hallucination_silence_threshold": 2.0} if _has_hal_thresh else {}),
    )

    _n_chunks   = max(1, _math.ceil(float(duration) / 30.0)) if duration else None
    _decode_cnt = [0]
    _stderr     = sys.stderr

    def _apply_progress_patch(m) -> None:
        if _n_chunks is None:
            return
        _orig = m.decode
        def _patched(mel, *a, **kw):
            r = _orig(mel, *a, **kw)
            _decode_cnt[0] += 1
            pct = min(99, int(100 * _decode_cnt[0] / _n_chunks))
            print(json.dumps({"progress": pct}), file=_stderr, flush=True)
            return r
        m.decode = _patched

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        model = whisper.load_model(model_size, device=device)
        _apply_progress_patch(model)
        result = model.transcribe(str(path), **transcribe_kwargs)
    except Exception as _err:
        if device != "cpu":
            print(
                f"openai-whisper on {device} failed ({type(_err).__name__}: {_err}), "
                "retrying on CPU",
                file=sys.stderr,
            )
            device = "cpu"
            _decode_cnt[0] = 0
            model = whisper.load_model(model_size, device=device)
            _apply_progress_patch(model)
            result = model.transcribe(str(path), **transcribe_kwargs)
        else:
            raise

    segments: list[dict] = []
    words: list[dict] = []

    for seg_idx, seg in enumerate(result.get("segments", [])):
        text = seg.get("text", "").strip()
        if text:
            segments.append({
                "start":  round(float(seg["start"]), 3),
                "end":    round(float(seg["end"]),   3),
                "text":   text,
                "seg_id": seg_idx,
            })
        for w in seg.get("words", []):
            word_text = w.get("word", "").strip()
            if word_text:
                prob = w.get("probability")
                words.append({
                    "word":   word_text,
                    "start":  round(float(w["start"]), 3),
                    "end":    round(float(w["end"]),   3),
                    # openai-whisper uses "probability"; forward as "score" to
                    # match faster-whisper's field name so web.py's confidence
                    # extraction works for both backends.
                    **({"score": round(float(prob), 3)} if prob is not None else {}),
                    "seg_id": seg_idx,
                })

    return {
        "segments": segments,
        "words":    words,
        "language": result.get("language", ""),
    }


def _run_whisperx_alignment(
    path: str,
    segments: list[dict],
    detected_language: str | None,
) -> list[dict] | None:
    """Refine word timestamps using WhisperX forced CTC alignment.

    Returns flat list of {word, start, end, score} dicts, or None if
    WhisperX is unavailable or alignment fails (caller falls back to
    faster-whisper timestamps with score/confidence=None).
    """
    try:
        import whisperx  # type: ignore
    except ImportError:
        return None

    try:
        audio = whisperx.load_audio(str(path))

        # Normalise to 2-char BCP-47 root; fall back to 'en' for unsupported.
        lang = ((detected_language or "ja").split("-")[0].lower())
        _SUPPORTED = {"en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt"}
        if lang not in _SUPPORTED:
            lang = "en"

        print(f"whisperx: loading alignment model for '{lang}'", file=sys.stderr)
        align_model, metadata = whisperx.load_align_model(
            language_code=lang, device="cpu"
        )

        _wx_pairs = [
            (s, s.get("seg_id"))
            for s in segments
            if s.get("text", "").strip()
        ]
        wx_segs = [
            {"start": s["start"], "end": s["end"], "text": s["text"]}
            for s, _ in _wx_pairs
        ]
        # Parallel seg_id list: whisperx.align preserves input order 1:1, so the
        # k-th aligned segment maps to wx_seg_ids[k].  Inheriting the ORIGINAL
        # seg_id (rather than re-enumerating) keeps refined words consistent with
        # the faster-whisper segment list, whose seg_id may have gaps from
        # skipped empty-text segments.
        wx_seg_ids = [sid for _, sid in _wx_pairs]
        if not wx_segs:
            return None

        aligned = whisperx.align(
            wx_segs, align_model, metadata, audio, "cpu",
            return_char_alignments=False,
        )

        refined: list[dict] = []
        for seg_idx, seg in enumerate(aligned.get("segments", [])):
            orig_seg_id = wx_seg_ids[seg_idx] if seg_idx < len(wx_seg_ids) else seg_idx
            for w in seg.get("words", []):
                word_text = (w.get("word") or "").strip()
                if not word_text or "start" not in w or "end" not in w:
                    continue
                score_val = w.get("score")
                refined.append({
                    "word":   word_text,
                    "start":  round(float(w["start"]), 3),
                    "end":    round(float(w["end"]),   3),
                    # Omit score when absent — a missing key becomes confidence=None
                    # in the frontend, suppressing the low-confidence underline.
                    # score=0.0 would incorrectly mark every unscored word as uncertain.
                    **({"score": round(float(score_val), 3)} if score_val is not None else {}),
                    "seg_id": orig_seg_id,
                })
        return refined if refined else None

    except Exception as exc:
        print(
            f"whisperx alignment failed ({type(exc).__name__}: {exc}), "
            "using faster-whisper timestamps",
            file=sys.stderr,
        )
        return None


def main() -> None:
    """Run Whisper transcription and print a JSON result to stdout.

    Tries faster-whisper first; falls back to openai-whisper.
    """
    if len(sys.argv) < 2:
        print("Usage: whisper_worker.py '<json>'", file=sys.stderr)
        sys.exit(1)

    try:
        args = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        print(f"Bad args JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    path       = args["path"]
    model_size = args.get("model_size", "base")
    language   = args.get("language") or None
    duration   = args.get("duration")

    # Redirect stdout → stderr so any print() during model loading cannot
    # corrupt the JSON output channel (openai-whisper logs to stdout).
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        # Primary: faster-whisper (CTranslate2 INT8)
        payload: dict | None = None
        fw_error: str | None = None
        try:
            payload = _run_faster_whisper(path, model_size, language, duration)
        except ImportError:
            fw_error = "not installed"
        except Exception as exc:
            fw_error = f"{type(exc).__name__}: {exc}"
            print(f"faster-whisper failed: {fw_error}", file=sys.stderr)

        # Fallback: openai-whisper
        if payload is None:
            if fw_error:
                print(
                    f"[faster-whisper {fw_error}] — falling back to openai-whisper",
                    file=sys.stderr,
                )
            try:
                payload = _run_openai_whisper(path, model_size, language, duration)
            except ImportError:
                sys.stdout = _real_stdout
                print(
                    "Neither faster-whisper nor openai-whisper is installed.\n"
                    "Run: pip install faster-whisper",
                    file=sys.stderr,
                )
                sys.exit(2)
    finally:
        sys.stdout = _real_stdout

    # Optional WhisperX forced alignment — refines word timestamps from
    # ±150 ms (faster-whisper) to ±20 ms and adds per-word confidence scores.
    whisperx_used = False
    _t_align = 0.0
    if payload and payload.get("words"):
        _ta0 = time.perf_counter()
        refined = _run_whisperx_alignment(
            path, payload["segments"], payload.get("language")
        )
        _t_align = time.perf_counter() - _ta0
        if refined:
            payload["words"] = refined
            whisperx_used = True
            print(
                f"whisperx: {len(refined)} words aligned (±20 ms)",
                file=sys.stderr,
            )

    if payload is not None:
        payload["whisperx_used"] = whisperx_used
        payload.setdefault("timings", {})["whisperx_align"] = round(_t_align, 3)

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
