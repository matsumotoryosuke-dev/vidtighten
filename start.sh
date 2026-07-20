#!/usr/bin/env bash
# Preview launcher for VidTighten — sets PYTHONPATH so no install needed.
DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${DIR}/src:${PYTHONPATH:-}"
exec python3 -m preprod.web --port 9877
