#!/usr/bin/env bash
# run.sh — launch High Fantasy Word Explorer
# All arguments are forwarded to game.py (--seed, --steps, --guidance).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${PYTORCH_VENV:-/home/david/Code/venv/pytorch}"
PYTHON="$VENV/bin/python"
export OVIE_PATH="${OVIE_PATH:-$(dirname "$SCRIPT_DIR")/ovie}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at $PYTHON"
    echo "  Set PYTORCH_VENV=/path/to/venv or run install.sh first."
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/game.py" "$@"
