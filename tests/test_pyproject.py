"""Contract tests for pyproject.toml's dependency declarations.

Regression coverage for the "declared install path is broken" blocker
(2026-07-19): whisper_worker.py imports and uses whisperx and faster_whisper
at runtime, but pyproject.toml declared neither, so `pip install -e ".[whisper]"`
produced an app that silently ran in degraded mode instead of working.

These tests parse the real pyproject.toml and assert the optional-dependency
groups actually cover what src/preprod/whisper_worker.py imports, so this
class of drift fails CI instead of shipping silently again.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
WHISPER_WORKER_PATH = REPO_ROOT / "src" / "preprod" / "whisper_worker.py"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with open(PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


def _dep_names(dep_specs: list[str]) -> set[str]:
    """Extract bare package names from PEP 508 dependency strings.

    Handles plain specs ("openai-whisper>=20231117") and self-referential
    extras ("preprod[whisper]") — the latter contributes no external
    package name of its own, so it is skipped rather than mis-parsed.
    """
    names = set()
    for spec in dep_specs:
        spec = spec.strip()
        if spec.startswith("preprod["):
            continue
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.\-]*)", spec)
        if m:
            names.add(m.group(1).lower())
    return names


class TestRequiresPython:
    """requires-python must reflect whisperx's verified, real support range."""

    def test_requires_python_has_a_floor_and_ceiling(self, pyproject):
        rp = pyproject["project"]["requires-python"]
        assert ">=3.10" in rp, (
            f"requires-python={rp!r} — expected a >=3.10 floor "
            "(whisperx's transitive deps require Python >=3.10)"
        )
        assert "<3.14" in rp, (
            f"requires-python={rp!r} — expected a <3.14 ceiling "
            "(whisperx's pinned torch stack publishes no cp314 wheels)"
        )


class TestWhisperExtraCoversRuntimeImports:
    """The `whisper` extra must declare every whisper-transcription backend
    whisper_worker.py actually tries to import."""

    def test_whisper_extra_declares_faster_whisper(self, pyproject):
        deps = pyproject["project"]["optional-dependencies"]["whisper"]
        names = _dep_names(deps)
        assert "faster-whisper" in names, (
            "whisper_worker.py imports `faster_whisper` as its PRIMARY "
            "transcription backend, but the `whisper` extra does not declare "
            "faster-whisper — a fresh `pip install .[whisper]` would silently "
            "fall back to the slower openai-whisper path every time."
        )

    def test_whisper_extra_declares_openai_whisper(self, pyproject):
        deps = pyproject["project"]["optional-dependencies"]["whisper"]
        names = _dep_names(deps)
        assert "openai-whisper" in names, (
            "whisper_worker.py imports `whisper` (openai-whisper) as its "
            "fallback transcription backend, but the `whisper` extra does "
            "not declare openai-whisper."
        )


class TestWhisperxExtraDeclared:
    """The `whisperx` extra must exist and declare the whisperx package
    that whisper_worker.py imports for forced alignment."""

    def test_whisperx_extra_exists(self, pyproject):
        extras = pyproject["project"]["optional-dependencies"]
        assert "whisperx" in extras, (
            "src/preprod/whisper_worker.py:_run_whisperx_alignment imports "
            "`whisperx` at runtime, but no `whisperx` optional-dependency "
            "group is declared in pyproject.toml — the declared install path "
            "cannot produce a working forced-alignment install."
        )

    def test_whisperx_extra_declares_whisperx_package(self, pyproject):
        deps = pyproject["project"]["optional-dependencies"]["whisperx"]
        names = _dep_names(deps)
        assert "whisperx" in names, (
            f"`whisperx` extra = {deps!r} does not declare the whisperx "
            "package itself."
        )

    def test_whisperx_extra_pulls_in_a_transcription_backend(self, pyproject):
        """whisperx alignment is only ever invoked on a transcription's
        output (see whisper_worker.py main()) — installing `whisperx` alone
        without a transcription backend would be a dead-end install."""
        deps = pyproject["project"]["optional-dependencies"]["whisperx"]
        assert any(d.strip().startswith("preprod[whisper") for d in deps), (
            f"`whisperx` extra = {deps!r} does not depend on the `whisper` "
            "extra — whisperx alignment is unreachable without a "
            "transcription backend already producing segments."
        )


class TestWhisperWorkerImportsAreDeclaredSomewhere:
    """Cross-check against the actual source: every `import whisperx` /
    `import faster_whisper` / `import whisper` in whisper_worker.py must
    correspond to a declared optional-dependency, not a hidden runtime
    dependency invisible to pyproject.toml."""

    def test_whisper_worker_imports_match_declared_extras(self, pyproject):
        source = WHISPER_WORKER_PATH.read_text(encoding="utf-8")
        extras = pyproject["project"]["optional-dependencies"]
        all_declared = _dep_names(extras["whisper"]) | _dep_names(extras["whisperx"])

        runtime_import_to_dist = {
            "whisperx": "whisperx",
            "faster_whisper": "faster-whisper",
            "whisper": "openai-whisper",
        }
        for import_name, dist_name in runtime_import_to_dist.items():
            # Accept both `import X` and `from X import ...` forms — e.g.
            # whisper_worker.py uses `from faster_whisper import WhisperModel`
            # but plain `import whisperx` / `import whisper`.
            pattern = rf"\bimport {import_name}\b|\bfrom {import_name} import\b"
            assert re.search(pattern, source), (
                f"expected whisper_worker.py to import {import_name!r} — "
                "test fixture is out of sync with the source, update it"
            )
            assert dist_name in all_declared, (
                f"whisper_worker.py imports {import_name!r} (dist "
                f"{dist_name!r}) but no optional-dependency group declares "
                f"{dist_name!r}"
            )
