#!/usr/bin/env python3
"""Launch VidTighten in a native macOS window using pywebview."""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import socket
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path


def _configure_logging() -> None:
    """Attach rotating-file + stderr handlers before any preprod code is imported."""
    log_dir = Path.home() / ".preprod" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.handlers.RotatingFileHandler(
        log_dir / "app.log", maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)


_configure_logging()

_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.isdir(_src):
    sys.path.insert(0, _src)

from preprod.web import app as flask_app

PORT = 9877


def _find_free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def _start_server(port: int) -> None:
    flask_app.run(host="127.0.0.1", port=port, use_reloader=False, threaded=True)


class _PywebviewApi:
    """Python methods exposed to JS as window.pywebview.api.*"""

    def get_dropped_paths(self) -> list[str]:
        """Return file paths from the most recent drag-and-drop operation.

        pywebview's cocoa backend intercepts drag operations in
        WebKitHost.performDragOperation_ and stores the dragged file paths in
        _dnd_state['paths'] (a list of (basename, full_path) tuples).  It only
        does this when _dnd_state['num_listeners'] > 0 — we set that to 1 at
        startup so every drop is captured.

        WKWebView loaded from http:// cannot expose file:// URIs through the JS
        dataTransfer API (WebKit security restriction), so JS calls this method
        after every drop event to get the real on-disk path.
        """
        from webview.dom import _dnd_state  # type: ignore

        paths = [item[1] for item in list(_dnd_state["paths"])]
        _dnd_state["paths"].clear()
        return paths

    # ── Export bridge methods ────────────────────────────────────────────────
    # JS calls these directly (window.pywebview.api.export_*) instead of using
    # fetch().  Any fetch() from WKWebView can trigger the navigation delegate
    # (decidePolicyForNavigationResponse) and blank the page, even for JSON
    # responses.  Going through the pywebview message-handler bridge bypasses
    # all WKWebView navigation handling entirely.

    def export_roughcut(
        self, path: str, removal_regions: list, padding_ms: int,
        threshold_db: float = -40.0,
    ) -> dict:
        """Generate rough-cut FCPXML and write to ~/Downloads without fetch()."""
        try:
            import preprod.web as _web
            from preprod.fcpxml_cut import generate_roughcut_fcpxml
            from preprod.probe import probe_media
            from preprod.segments import build_segments

            p = Path(path).expanduser().resolve()
            if not p.exists():
                return {"ok": False, "error": f"File not found: {p.name}"}

            with _web._media_cache_lock:
                media = _web._media_cache.get(str(p))
            if media is None:
                media = probe_media(p)
                _web._cache_media(str(p), media)

            segs = _web._build_segments_typed(
                removal_regions or [], p, media.duration,
                int(padding_ms) if padding_ms is not None else 200,
                threshold_db=float(threshold_db),
            )
            if not segs:
                return {"ok": False, "error": "No keep-segments after applying removals"}

            _web._EXPORT_DIR.mkdir(exist_ok=True)
            out = _web._EXPORT_DIR / f"{p.stem}_cut.fcpxml"
            generate_roughcut_fcpxml(segs, media, out)

            ok, err = _web._copy_to_downloads(out, f"{p.stem}_cut.fcpxml")
            if not ok:
                return {"ok": False, "error": err}
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def export_telop(
        self,
        path: str,
        duration: float,
        telop_entries: list,
        removal_regions: list,
        padding_ms: int,
        settings: dict,
        stem: str,
        use_source_timing: bool = False,
        threshold_db: float = -40.0,
    ) -> dict:
        """Generate telop FCPXML and write to ~/Downloads without fetch()."""
        try:
            import preprod.web as _web
            from preprod.fcpxml_telop import generate_telop_fcpxml

            _telop_path = Path(path).expanduser().resolve() if path else None
            dur = float(duration or 0)
            keep_segs = _web._build_segments_typed(
                removal_regions or [],
                _telop_path if (_telop_path and _telop_path.exists()) else None,
                dur, int(padding_ms) if padding_ms is not None else 200,
                threshold_db=float(threshold_db),
            ) if dur else []

            _web._EXPORT_DIR.mkdir(exist_ok=True)
            out = _web._EXPORT_DIR / f"{stem}_telop.fcpxml"
            generate_telop_fcpxml(
                telop_entries or [],
                keep_segs,
                total_source_duration=dur,
                settings=settings or {},
                stem=stem,
                output_path=out,
                use_source_timing=bool(use_source_timing),
            )

            ok, err = _web._copy_to_downloads(out, f"{stem}_telop.fcpxml")
            if not ok:
                return {"ok": False, "error": err}
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def export_subtitles(
        self,
        fmt: str,
        path: str,
        telop_entries: list,
        removal_regions: list,
        duration: float,
        padding_ms: int,
        stem: str,
        threshold_db: float = -40.0,
    ) -> dict:
        """Generate SRT/VTT subtitles and write to ~/Downloads without fetch()."""
        try:
            import preprod.web as _web
            from preprod.segments import map_span_to_output, filter_telop_entries

            fmt = (fmt or "srt").lower()
            if fmt not in ("srt", "vtt"):
                return {"ok": False, "error": "format must be 'srt' or 'vtt'"}

            _sub_path = Path(path).expanduser().resolve() if path else None
            dur = float(duration or 0)
            keep_segs = _web._build_segments_typed(
                removal_regions or [],
                _sub_path if (_sub_path and _sub_path.exists()) else None,
                dur, int(padding_ms) if padding_ms is not None else 200,
                threshold_db=float(threshold_db),
            ) if dur else []

            ts_fn = _web._ts_vtt if fmt == "vtt" else _web._ts_srt
            lines: list[str] = ["WEBVTT\n"] if fmt == "vtt" else []
            idx = 1
            for entry in filter_telop_entries(telop_entries or []):
                span = map_span_to_output(entry["start"], entry["end"], keep_segs)
                if span is None:
                    continue
                out_s, out_e = span
                text = entry.get("text", "").strip()
                if not text:
                    continue
                if fmt == "srt":
                    lines.append(str(idx))
                lines.append(f"{ts_fn(out_s)} --> {ts_fn(out_e)}")
                lines.append(text)
                lines.append("")
                idx += 1

            content = "\n".join(lines)
            _web._EXPORT_DIR.mkdir(exist_ok=True)
            tmp = _web._EXPORT_DIR / f"{stem}_subtitles.{fmt}"
            tmp.write_text(content, encoding="utf-8")

            ok, err = _web._copy_to_downloads(tmp, f"{stem}_subtitles.{fmt}")
            if not ok:
                return {"ok": False, "error": err}
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def save_file_content(self, content: str, filename: str) -> dict:
        """Write text content to ~/Downloads.

        dlBlob() in utils.js calls this via the pywebview bridge instead of the
        anchor.click() blob-URL trick (which blanks the page) or create_file_dialog
        (which also blanks the page — showing a modal panel from a JS-API background
        thread forces AppKit into an inconsistent WKWebView state on macOS).

        The file goes straight to ~/Downloads.

        NOTE: We intentionally do NOT call subprocess.run("open -R") here.
        Spawning a child process from a pywebview JS-API background thread causes
        side-effects that reset the WKWebView (confirmed from crash log).

        NOTE: We intentionally do NOT return the path in the success dict.
        pywebview serialises the Python return value via evaluate_js(); if the path
        contains multi-byte (e.g. Japanese) characters the resulting JS literal can
        be malformed, triggering a WKWebView navigation error that blanks the page.

        Returns:
            {"ok": True}             on success
            {"ok": False, "error": "..."}  on failure
        """
        downloads = Path.home() / "Downloads"
        try:
            downloads.mkdir(exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": f"Cannot create ~/Downloads: {exc}"}

        dest = downloads / filename

        # Avoid clobbering: append " (1)", " (2)", … if a file with that name exists.
        if dest.exists():
            stem, suffix = Path(filename).stem, Path(filename).suffix
            for n in range(1, 100):
                dest = downloads / f"{stem} ({n}){suffix}"
                if not dest.exists():
                    break

        try:
            dest.write_text(content, encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}

        return {"ok": True}


# ── Native menu bar ────────────────────────────────────────────────────────────
# File → Open File…  and  File → Open Recent  (last _MAX_RECENT sessions).
#
# Menu callbacks run on a background thread (cocoa.py handleMenuAction_ spawns
# a Thread per action), so create_file_dialog is safe (it bounces to the main
# AppKit thread internally via AppHelper.callAfter + Semaphore).  evaluate_js is
# also safe from background threads.
#
# Dynamic rebuild: after opening a file we update _recent_paths then call
# _rebuild_menu(), which mutates the BrowserView's menu reference and forces
# an immediate AppKit setMainMenu_ on the main thread via AppHelper.callAfter.

_MAX_RECENT = 5
_recent_paths: deque[str] = deque(maxlen=_MAX_RECENT)

_FILE_TYPES = (
    "All Media (*.mp4;*.mov;*.m4v;*.mxf;*.avi;*.mkv;*.mp3;*.aac;*.wav;*.m4a;*.flac)",
    "Video Files (*.mp4;*.mov;*.m4v;*.mxf;*.avi;*.mkv)",
    "Audio Files (*.mp3;*.aac;*.wav;*.m4a;*.flac)",
    "All Files (*.*)",
)

log = logging.getLogger(__name__)


def _evaluate_js(js: str) -> None:
    """Fire JS in the webview window from any thread (menu callbacks)."""
    import webview  # type: ignore
    wins = webview.windows
    if wins:
        try:
            wins[0].evaluate_js(js)
        except Exception:
            log.exception("evaluate_js failed: %s", js[:80])


def _build_file_menu() -> "object":
    from webview.menu import Menu, MenuAction, MenuSeparator  # type: ignore

    if _recent_paths:
        recent_items = [
            MenuAction(Path(p).name, lambda _p=p: _open_file_in_app(_p))
            for p in _recent_paths
        ]
    else:
        recent_items = [MenuAction("No Recent Files", lambda: None)]

    return Menu("File", [
        MenuAction("Open File\u2026", _menu_open_file),
        Menu("Open Recent", recent_items),
        MenuSeparator(),
        MenuAction("Export Rough-Cut\u2026",
                   lambda: _evaluate_js("window._vtExport('roughcut')")),
        MenuAction("Export Telop\u2026",
                   lambda: _evaluate_js("window._vtExport('telop')")),
        MenuSeparator(),
    ])


def _build_analyze_menu() -> "object":
    from webview.menu import Menu, MenuAction, MenuSeparator  # type: ignore

    return Menu("Analyze", [
        MenuAction("Re-analyze",
                   lambda: _evaluate_js("window._vtReanalyze && window._vtReanalyze()")),
        MenuSeparator(),
        MenuAction("Detect Silence Only",
                   lambda: _evaluate_js("window._vtRedetectSilence && window._vtRedetectSilence()")),
    ])


def _rebuild_menu() -> None:
    """Force-rebuild the native menu bar from the current _recent_paths state."""
    try:
        import webview  # type: ignore
        from PyObjCTools import AppHelper  # type: ignore
        from webview.platforms.cocoa import BrowserView  # type: ignore

        wins = webview.windows
        if not wins:
            return

        new_menu = [_build_file_menu(), _build_analyze_menu()]
        wins[0].menu = new_menu  # update Window reference for windowDidBecomeKey_

        def _apply() -> None:
            instance = next(iter(BrowserView.instances.values()), None)
            if instance:
                instance.menu = new_menu
                BrowserView.app.setMainMenu_(instance._recreate_menus(new_menu))

        AppHelper.callAfter(_apply)
    except Exception:
        log.exception("Menu rebuild failed")


def _open_file_in_app(path: str) -> None:
    """Load path into VidTighten: update recents → call JS loadFilePath → rebuild menu."""
    import webview  # type: ignore

    # Move to front of recents (deduplicates)
    try:
        _recent_paths.remove(path)
    except ValueError:
        pass
    _recent_paths.appendleft(path)

    wins = webview.windows
    if not wins:
        return

    try:
        wins[0].evaluate_js(f"window._vtLoadFile({json.dumps(path)})")
    except Exception:
        log.exception("evaluate_js _vtLoadFile failed for %s", path)

    _rebuild_menu()


def _menu_open_file() -> None:
    """MenuAction callback: show open-file dialog then load the chosen file."""
    import webview  # type: ignore

    wins = webview.windows
    if not wins:
        return

    result = wins[0].create_file_dialog(
        webview.OPEN_DIALOG,
        allow_multiple=False,
        file_types=_FILE_TYPES,
    )
    if result:
        _open_file_in_app(result[0])


def _load_recent_from_sessions() -> None:
    """Populate _recent_paths from saved sessions at startup (existing files only)."""
    try:
        from preprod.session import list_sessions  # type: ignore

        # list_sessions returns newest-first; add in reverse so appendleft keeps order
        sessions = list_sessions(limit=_MAX_RECENT * 2)  # fetch extra to filter non-existent
        valid = [
            s["file_path"] for s in sessions
            if s.get("file_exists") and s.get("file_path")
            # skip temp upload paths that won't survive restarts
            and "preprod_uploads" not in s["file_path"]
        ]
        for p in reversed(valid[:_MAX_RECENT]):
            _recent_paths.appendleft(p)
    except Exception:
        log.exception("Failed to load recent sessions for menu")


def _set_app_name(name: str) -> None:
    """Set the process display name so macOS menu bar shows 'name' instead of 'Python'.

    The VidTighten.app launcher uses `exec python3 …` which replaces the process,
    leaving the binary name ('Python3') as the process name.  Setting it here
    forces the correct name before AppKit builds the first menu.
    """
    try:
        from Foundation import NSProcessInfo  # type: ignore
        NSProcessInfo.processInfo().setProcessName_(name)
    except Exception:
        log.debug("setProcessName_ unavailable — menu bar app name may show as Python")


def _patch_menu_order() -> None:
    """Re-order the macOS menu bar so custom menus sit *before* Edit/View.

    pywebview's _recreate_menus appends custom menus (File) after the default
    Edit/View menus, giving: App | Edit | View | File.  Standard macOS order is
    App | File | Edit | View.

    We monkey-patch _recreate_menus to call the original, then move the
    custom-menu items from the end of the NSMenu to position 1 (right after the
    App menu), preserving their relative order.
    """
    try:
        from webview import settings as _wv_settings  # type: ignore
        from webview.menu import Menu  # type: ignore
        from webview.platforms.cocoa import BrowserView  # type: ignore

        _orig = BrowserView._recreate_menus

        def _recreate_menus_ordered(self, user_menu):
            main_menu = _orig(self, user_menu)

            if not user_menu or not _wv_settings.get("SHOW_DEFAULT_MENUS", True):
                return main_menu

            # Count menus that are NOT the special __app__ meta-menu.
            custom_count = sum(
                1 for m in user_menu
                if not (isinstance(m, Menu) and m.title == "__app__")
            )
            if custom_count <= 0:
                return main_menu

            # pywebview appended them at the end; pull them out and re-insert
            # right after the App menu (index 0) so the order becomes:
            #   App | File | Edit | View
            n = main_menu.numberOfItems()
            items = [
                main_menu.itemAtIndex_(n - custom_count + i)
                for i in range(custom_count)
            ]
            for item in items:
                main_menu.removeItem_(item)
            for i, item in enumerate(items):
                main_menu.insertItem_atIndex_(item, 1 + i)

            return main_menu

        BrowserView._recreate_menus = _recreate_menus_ordered
    except Exception:
        log.debug("Menu-order patch failed — File menu will appear after Edit/View")


def main() -> None:
    _set_app_name("VidTighten")
    _patch_menu_order()

    # Enable pywebview's native drag-and-drop path capture.
    # cocoa.py WebKitHost.performDragOperation_ only populates _dnd_state['paths']
    # from the drag pasteboard when num_listeners > 0.
    from webview.dom import _dnd_state  # type: ignore
    _dnd_state["num_listeners"] = 1

    _load_recent_from_sessions()

    port = _find_free_port(PORT)
    server_thread = threading.Thread(target=_start_server, args=(port,), daemon=True)
    server_thread.start()

    import webview  # type: ignore
    webview.create_window(
        "VidTighten",
        f"http://127.0.0.1:{port}",
        width=1200,
        height=820,
        min_size=(800, 560),
        js_api=_PywebviewApi(),
    )
    webview.start(menu=[_build_file_menu(), _build_analyze_menu()])


if __name__ == "__main__":
    main()
