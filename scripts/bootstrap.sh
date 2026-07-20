#!/usr/bin/env bash
# One-command dev environment setup for VidTighten (preprod).
#
# Usage:
#   scripts/bootstrap.sh
#
# What it does:
#   1. Installs `uv` (https://astral.sh/uv) if not already on PATH — uv pins
#      its own managed Python interpreters, so this sidesteps "which python3
#      do I even have" entirely.
#   2. Downloads/selects the exact Python version pinned in .python-version
#      (currently 3.13 — verified to work with every extra below; see the
#      requires-python comment in pyproject.toml for why the supported range
#      is >=3.10,<3.14).
#   3. Creates .venv/ at the repo root and installs preprod in editable mode
#      with every optional-dependency group (whisper transcription,
#      whisperx forced-alignment, Japanese morphological analysis, dev/test
#      tooling) pinned to the exact resolved versions in uv.lock.
#
# Result: a fully reproducible dev environment, without touching any
# system Python or any environment outside this repo.
#
# After this finishes:
#   source .venv/bin/activate         # or prefix commands with `uv run`
#   pytest tests/ -q                  # run the Python test suite
#   npm install && npm run test:js -- --run   # run the JS test suite
#   python run_web.py                 # run the app in a browser at :9877

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── 1. Ensure uv is available ────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "uv not found — installing (https://astral.sh/uv)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The official installer places uv in ~/.local/bin, which may not be on
    # PATH yet in this shell session.
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv &> /dev/null; then
        echo "uv installed but not found on PATH. Add ~/.local/bin to your" \
             "PATH and re-run this script." >&2
        exit 1
    fi
fi

echo "Using uv: $(command -v uv) ($(uv --version))"

# ── 2 + 3. Resolve Python + sync the full dev environment ───────────────
# --all-extras pulls in every group declared in pyproject.toml
# ([whisper], [whisperx], [japanese], [dev]) at the exact versions locked
# in uv.lock, so this is byte-for-byte reproducible across machines.
echo "Syncing environment (this downloads PyTorch + the whisperx alignment"
echo "stack — several GB on first run; subsequent runs are cached)..."
uv sync --all-extras

echo ""
echo "Done. Environment ready at ${REPO_ROOT}/.venv"
echo ""
echo "  Activate it:      source .venv/bin/activate"
echo "  Or run commands:  uv run pytest tests/ -q"
echo "  Run the app:      uv run python run_web.py   (http://127.0.0.1:9877)"
echo ""
