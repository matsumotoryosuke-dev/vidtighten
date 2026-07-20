"""Tests for build_app.sh — regression tests for known build bugs.

Regression test for Bug #4:
  The launcher script inside the .app bundle didn't export Homebrew PATH,
  so ffmpeg (and other Homebrew tools) were not found when the app was
  launched from Finder (which doesn't inherit the shell's PATH).

  Fix: the launcher shell script must contain:
      export PATH="/opt/homebrew/bin:...
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

BUILD_SH_PATH = Path(__file__).parent.parent / "build_app.sh"


class TestBuildAppShLauncherPathExport:
    """Bug #4 regression: launcher must export Homebrew PATH."""

    def test_build_sh_exists(self):
        assert BUILD_SH_PATH.exists(), f"build_app.sh not found at {BUILD_SH_PATH}"

    def test_launcher_contains_path_export(self):
        """The launcher heredoc must contain an export PATH= line with /opt/homebrew/bin."""
        content = BUILD_SH_PATH.read_text(encoding="utf-8")
        # Look for the export PATH line within the launcher heredoc
        assert "export PATH=" in content, (
            "REGRESSION BUG #4: build_app.sh does not contain 'export PATH=' — "
            "Homebrew tools like ffmpeg will not be found when launched from Finder"
        )

    def test_launcher_includes_homebrew_bin_in_path(self):
        """The exported PATH must include /opt/homebrew/bin."""
        content = BUILD_SH_PATH.read_text(encoding="utf-8")
        # Find the export PATH line
        for line in content.splitlines():
            if "export PATH=" in line:
                if "/opt/homebrew/bin" in line:
                    return  # Found it — test passes
        pytest.fail(
            "REGRESSION BUG #4: No 'export PATH=...' line containing '/opt/homebrew/bin' "
            "found in build_app.sh. ffmpeg will not be found when the .app is opened from Finder."
        )

    def test_path_export_is_in_launcher_script_not_just_outer_script(self):
        """The PATH export must appear inside the launcher heredoc (the SHELL block),
        not only in the outer bash script logic."""
        content = BUILD_SH_PATH.read_text(encoding="utf-8")

        # The launcher heredoc starts after `cat > "${MACOS}/${APP_NAME}" << SHELL`
        # and ends at the matching `SHELL` terminator.
        # We identify this section by looking for the heredoc block.
        heredoc_match = re.search(
            r'cat\s+>.*?<<\s+SHELL\n(.*?)\nSHELL\b',
            content,
            re.DOTALL
        )
        assert heredoc_match is not None, (
            "Could not find the launcher heredoc (cat > ... << SHELL...SHELL) in build_app.sh"
        )

        launcher_body = heredoc_match.group(1)
        assert "export PATH=" in launcher_body, (
            "REGRESSION BUG #4: 'export PATH=' not found inside the launcher script heredoc. "
            "The PATH must be exported inside the .app launcher, not just in build_app.sh itself."
        )
        assert "/opt/homebrew/bin" in launcher_body, (
            "REGRESSION BUG #4: '/opt/homebrew/bin' not in the launcher heredoc PATH export."
        )


class TestBuildAppShSyntax:
    def test_build_sh_is_valid_bash_syntax(self):
        """bash -n performs a syntax check without executing the script."""
        result = subprocess.run(
            ["bash", "-n", str(BUILD_SH_PATH)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"build_app.sh has bash syntax errors:\n{result.stderr}"
        )


class TestBuildAppShPythonSelection:
    """Regression tests for the interpreter-selection preference chain.

    Bootstrap fix (2026-07-19): a fresh contributor clone has no
    ~/.preprod/venv, so build_app.sh must fall back to the repo-local .venv
    produced by `scripts/bootstrap.sh` (uv sync) BEFORE falling back to bare
    Homebrew python3 — Homebrew python3 tracks the latest CPython and is not
    guaranteed to satisfy whisperx's verified Python >=3.10,<3.14 range.
    """

    def test_preprod_venv_is_still_first_preference(self):
        """Must not regress Rio's existing hand-built daily-use venv path."""
        content = BUILD_SH_PATH.read_text(encoding="utf-8")
        assert '${HOME}/.preprod/venv/bin/python' in content, (
            "~/.preprod/venv must remain a candidate — it is the existing "
            "working daily-use runtime and must not be dropped."
        )

    def test_repo_local_venv_is_a_fallback_candidate(self):
        """The bootstrap-produced <repo>/.venv must be checked before bare python3."""
        content = BUILD_SH_PATH.read_text(encoding="utf-8")
        assert '${SCRIPT_DIR}/.venv/bin/python' in content, (
            "build_app.sh does not check <repo>/.venv (the scripts/bootstrap.sh "
            "output) — a fresh contributor clone with no ~/.preprod/venv will "
            "silently fall through to bare Homebrew python3, which is not "
            "guaranteed to support whisperx."
        )

    def test_preference_order_repo_venv_before_preprod_venv_before_homebrew(self):
        """<repo>/.venv (the reproducible bootstrap output) must be checked
        first, ~/.preprod/venv (legacy personal venv) second, Homebrew
        python3 last.

        <repo>/.venv is the canonical, documented, reproducible-from-lockfile
        path every contributor (including the maintainer, once they've run
        scripts/bootstrap.sh) gets. ~/.preprod/venv is a legacy fallback for
        continuity on machines that built it before scripts/bootstrap.sh
        existed — it must not outrank the reproducible path, or "run the
        bootstrap script" silently has no effect for whoever already has the
        old venv (see the OSS-readiness eng-director review, 2026-07-19).
        """
        content = BUILD_SH_PATH.read_text(encoding="utf-8")
        idx_repo_venv = content.find('${SCRIPT_DIR}/.venv/bin/python')
        idx_preprod = content.find('${HOME}/.preprod/venv/bin/python')
        idx_homebrew = content.find('/opt/homebrew/bin/python3"')
        assert -1 not in (idx_preprod, idx_repo_venv, idx_homebrew), (
            "One or more expected PYTHON candidates missing from build_app.sh"
        )
        assert idx_repo_venv < idx_preprod < idx_homebrew, (
            "Preference order regressed: expected <repo>/.venv, then "
            "~/.preprod/venv, then Homebrew python3, in that order."
        )
