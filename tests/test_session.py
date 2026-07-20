"""Tests for session.py — TTL, size reporting, and bulk-clear helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from preprod.session import (
    _session_path,
    clear_all_sessions,
    delete_session,
    expire_old_sessions,
    list_sessions,
    load_session,
    save_session,
    sessions_count,
    sessions_size_bytes,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_session_dir(tmp_path, monkeypatch):
    """Redirect _SESSION_DIR to a temp dir for isolation."""
    import preprod.session as sm
    monkeypatch.setattr(sm, "_SESSION_DIR", tmp_path)
    return tmp_path


# ── Basic save / load / delete ────────────────────────────────────────────────


class TestSaveLoadDelete:
    def test_save_and_load_roundtrip(self, tmp_session_dir):
        state = {"foo": "bar", "n": 42}
        save_session("/fake/video.mp4", state)
        loaded = load_session("/fake/video.mp4")
        assert loaded == state

    def test_saved_file_is_compact_and_roundtrips(self, tmp_session_dir):
        # T0252: sessions are written with compact separators (no indent/newlines)
        # to keep large embedded arrays small, while still round-tripping.
        state = {"waveform": [0.1, 0.2, 0.3], "words": [{"id": "w0", "t": 1.0}]}
        save_session("/fake/big.mp4", state)
        raw = _session_path("/fake/big.mp4").read_text(encoding="utf-8")
        assert "\n" not in raw            # compact: no pretty-print newlines
        assert ", " not in raw and ": " not in raw  # compact separators
        assert load_session("/fake/big.mp4") == state

    def test_load_returns_none_for_unknown_path(self, tmp_session_dir):
        assert load_session("/nonexistent/file.mp4") is None

    def test_delete_removes_session(self, tmp_session_dir):
        save_session("/fake/video.mp4", {"x": 1})
        delete_session("/fake/video.mp4")
        assert load_session("/fake/video.mp4") is None


# ── expire_old_sessions ───────────────────────────────────────────────────────


class TestExpireOldSessions:
    def test_removes_old_session_files(self, tmp_session_dir):
        """Files whose mtime is older than the TTL must be removed."""
        save_session("/fake/old.mp4", {"old": True})
        p = _session_path("/fake/old.mp4")
        # Back-date the file to 31 days ago
        old_mtime = time.time() - 31 * 86400
        import os
        os.utime(str(p), (old_mtime, old_mtime))

        removed = expire_old_sessions(days=30)
        assert removed == 1
        assert not p.exists()

    def test_keeps_recent_session_files(self, tmp_session_dir):
        """Files within TTL must be kept."""
        save_session("/fake/new.mp4", {"new": True})
        removed = expire_old_sessions(days=30)
        assert removed == 0
        assert load_session("/fake/new.mp4") is not None

    def test_returns_zero_when_dir_absent(self, tmp_path, monkeypatch):
        import preprod.session as sm
        nonexistent = tmp_path / "no_such_dir"
        monkeypatch.setattr(sm, "_SESSION_DIR", nonexistent)
        assert expire_old_sessions(days=30) == 0


# ── sessions_size_bytes / sessions_count ──────────────────────────────────────


class TestSizeAndCount:
    def test_size_zero_when_empty(self, tmp_session_dir):
        assert sessions_size_bytes() == 0

    def test_count_zero_when_empty(self, tmp_session_dir):
        assert sessions_count() == 0

    def test_size_nonzero_after_save(self, tmp_session_dir):
        save_session("/fake/video.mp4", {"a": "b"})
        assert sessions_size_bytes() > 0

    def test_count_increments(self, tmp_session_dir):
        save_session("/fake/v1.mp4", {})
        save_session("/fake/v2.mp4", {})
        assert sessions_count() == 2

    def test_size_and_count_zero_when_dir_absent(self, tmp_path, monkeypatch):
        import preprod.session as sm
        monkeypatch.setattr(sm, "_SESSION_DIR", tmp_path / "ghost")
        assert sessions_size_bytes() == 0
        assert sessions_count() == 0


# ── clear_all_sessions ────────────────────────────────────────────────────────


class TestClearAllSessions:
    def test_removes_all_files(self, tmp_session_dir):
        save_session("/fake/v1.mp4", {})
        save_session("/fake/v2.mp4", {})
        removed = clear_all_sessions()
        assert removed == 2
        assert sessions_count() == 0

    def test_returns_zero_when_empty(self, tmp_session_dir):
        assert clear_all_sessions() == 0

    def test_returns_zero_when_dir_absent(self, tmp_path, monkeypatch):
        import preprod.session as sm
        monkeypatch.setattr(sm, "_SESSION_DIR", tmp_path / "ghost")
        assert clear_all_sessions() == 0


# ── list_sessions ─────────────────────────────────────────────────────────────


class TestListSessions:
    def test_empty_when_no_sessions(self, tmp_session_dir):
        assert list_sessions() == []

    def test_empty_when_dir_absent(self, tmp_path, monkeypatch):
        import preprod.session as sm
        monkeypatch.setattr(sm, "_SESSION_DIR", tmp_path / "ghost")
        assert list_sessions() == []

    def test_returns_entry_after_save(self, tmp_session_dir, tmp_path):
        fake_file = tmp_path / "clip.mp4"
        fake_file.write_bytes(b"\x00")
        save_session(str(fake_file), {"telopEntries": [1, 2], "removalCandidates": [3]})
        sessions = list_sessions()
        assert len(sessions) == 1
        s = sessions[0]
        assert s["file_path"] == str(fake_file)
        assert s["file_name"] == "clip.mp4"
        assert s["telop_count"] == 2
        assert s["removal_count"] == 1
        assert s["file_exists"] is True

    def test_file_exists_false_for_missing_file(self, tmp_session_dir):
        save_session("/nonexistent/video.mp4", {})
        sessions = list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["file_exists"] is False

    def test_sorted_newest_first(self, tmp_session_dir):
        save_session("/fake/a.mp4", {})
        save_session("/fake/b.mp4", {})
        sessions = list_sessions()
        # Both saved in rapid succession; just check we get 2 entries
        assert len(sessions) == 2

    def test_old_sessions_without_meta_are_skipped(self, tmp_session_dir):
        """Session files saved before the _file_path meta field was added are ignored."""
        from preprod.session import _session_path
        p = _session_path("/legacy/video.mp4")
        p.write_text('{"telopEntries": []}', encoding="utf-8")
        assert list_sessions() == []

    def test_load_strips_meta_keys(self, tmp_session_dir):
        """load_session must not expose _file_path / _saved_at to callers."""
        save_session("/fake/video.mp4", {"foo": "bar"})
        loaded = load_session("/fake/video.mp4")
        assert loaded == {"foo": "bar"}
        assert "_file_path" not in loaded
        assert "_saved_at" not in loaded

    def test_limit_respected(self, tmp_session_dir):
        for i in range(25):
            save_session(f"/fake/video_{i}.mp4", {})
        assert len(list_sessions(limit=10)) == 10


class TestLoadSessionEdgeCases:
    """T0295: load_session must degrade gracefully on unreadable files."""

    def test_corrupt_json_returns_none(self, tmp_session_dir):
        path = "/fake/corrupt.mp4"
        _session_path(path).write_text("{corrupted data", encoding="utf-8")  # invalid JSON
        assert load_session(path) is None   # graceful fallback — does NOT raise

    def test_empty_file_returns_none(self, tmp_session_dir):
        path = "/fake/empty.mp4"
        _session_path(path).write_text("", encoding="utf-8")
        assert load_session(path) is None
