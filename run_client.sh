#!/usr/bin/env bash
set -euo pipefail


MODE="greedy"
CONFIG=""
PYTHON_BIN="python3"

usage () {
  echo "Usage: bash run_client.sh [--mode greedy] [--config /path/to/client_config.py] [--python python3]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1"; usage; exit 1;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [[ ! -f "requirements.txt" ]]; then
  echo "requirements.txt not found in repo root: $REPO_DIR"
  exit 1
fi

# If a venv exists, use it. Otherwise, run with system python.
if [[ -d ".venv" ]]; then
  source ".venv/bin/activate"
fi

if [[ -n "$CONFIG" ]]; then
  $PYTHON_BIN run_solver.py --mode "$MODE" --config "$CONFIG"
else
  $PYTHON_BIN run_solver.py --mode "$MODE"
fi
