#!/bin/sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "usage: ./start.sh STATE_ROOT [PORT]" >&2
  exit 2
fi
KIT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
STATE_ROOT=$1
PORT=${2:-8001}
STATE_ROOT=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$STATE_ROOT")
case "$STATE_ROOT/" in
  "$KIT_ROOT"/*) echo "STATE_ROOT must remain outside the delivery kit." >&2; exit 2 ;;
esac
mkdir -p -- "$STATE_ROOT"
python -m rag_evidence annotation serve --package "$KIT_ROOT/ann-pilot-b.json" --state "$STATE_ROOT" --host 127.0.0.1 --port "$PORT"
