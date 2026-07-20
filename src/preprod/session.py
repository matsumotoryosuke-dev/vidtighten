"""Session persistence — autosave and restore user work between launches."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional


_SESSION_DIR = Path.home() / ".preprod" / "sessions"
_SESSION_TTL_DAYS = 30


def _session_path(file_path: str) -> Path:
    h = hashlib.sha256(file_path.encode()).hexdigest()[:16]
    return _SESSION_DIR / f"{h}.json"


def save_session(file_path: str, state: dict) -> None:
    """Persist session state keyed by file path hash.

    Embeds _file_path and _saved_at at the top level so list_sessions() can
    reconstruct the recent-sessions index without a separate manifest file.
    The UI ignores underscore-prefixed keys.
    """
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    p = _session_path(file_path)
    data = {
        "_file_path": file_path,
        "_saved_at": time.time(),
        **state,
    }
    # Compact separators (no indent/newlines): sessions embed large arrays
    # (waveform ~3000 floats, words ~thousands) that must persist for restore —
    # pretty-printing them roughly doubled file size for zero benefit (T0252).
    p.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def list_sessions(limit: int = 20) -> list[dict]:
    """Return recent sessions sorted newest-first.

    Only includes sessions that have a readable ``_file_path`` field (i.e. were
    saved with the current version of save_session).  Old sessions without the
    metadata field are silently skipped — they will appear once re-saved.

    Returns a list of dicts with keys:
        file_path     str    — absolute path of the original media file
        file_name     str    — basename only, for display
        saved_at      float  — Unix timestamp of last save
        telop_count   int    — number of transcript segments
        removal_count int    — number of cut candidates
        file_exists   bool   — whether the file is still on disk
    """
    if not _SESSION_DIR.exists():
        return []
    results = []
    for f in _SESSION_DIR.iterdir():
        if not (f.is_file() and f.suffix == ".json"):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            file_path = data.get("_file_path")
            if not file_path:
                continue  # pre-metadata session — skip until re-saved
            saved_at      = data.get("_saved_at", f.stat().st_mtime)
            telop_count   = len(data.get("telopEntries", []))
            removal_count = len(data.get("removalCandidates", []))
            results.append({
                "file_path":    file_path,
                "file_name":    Path(file_path).name,
                "saved_at":     saved_at,
                "telop_count":  telop_count,
                "removal_count": removal_count,
                "file_exists":  Path(file_path).exists(),
            })
        except (json.JSONDecodeError, OSError):
            continue
    results.sort(key=lambda x: x["saved_at"], reverse=True)
    return results[:limit]


def load_session(file_path: str) -> Optional[dict]:
    """Load a previously saved session for this file path. Returns None if not found.

    Strips underscore-prefixed internal meta keys (_file_path, _saved_at) so
    callers receive only application state regardless of which version saved it.
    """
    p = _session_path(file_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (json.JSONDecodeError, OSError):
        return None


def delete_session(file_path: str) -> None:
    """Remove saved session for this file."""
    p = _session_path(file_path)
    if p.exists():
        p.unlink()


def expire_old_sessions(days: int = _SESSION_TTL_DAYS) -> int:
    """Delete session files not accessed in *days* days. Returns count removed."""
    if not _SESSION_DIR.exists():
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for f in _SESSION_DIR.iterdir():
        try:
            if f.is_file() and f.suffix == ".json" and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def sessions_size_bytes() -> int:
    """Return total disk bytes used by all session files."""
    if not _SESSION_DIR.exists():
        return 0
    total = 0
    for f in _SESSION_DIR.iterdir():
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def sessions_count() -> int:
    """Return number of saved session files."""
    if not _SESSION_DIR.exists():
        return 0
    return sum(1 for f in _SESSION_DIR.iterdir() if f.is_file() and f.suffix == ".json")


def clear_all_sessions() -> int:
    """Delete all session files. Returns count removed."""
    if not _SESSION_DIR.exists():
        return 0
    removed = 0
    for f in _SESSION_DIR.iterdir():
        try:
            if f.is_file():
                f.unlink()
                removed += 1
        except OSError:
            pass
    return removed
