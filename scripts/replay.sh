#!/usr/bin/env bash
# Host: replay a recorded race in RViz.
#   ./scripts/replay.sh results/<run> [rate]
set -e
cd "$(dirname "$0")/.."
RUN="${1:?usage: replay.sh results/<run> [rate]}"
RATE="${2:-1.0}"
REL="$(realpath --relative-to=results "$RUN")"
command -v xhost >/dev/null 2>&1 && xhost +local:docker >/dev/null 2>&1 || true
exec docker compose run --rm sim ros2 launch icuas27_bringup replay.launch.py \
  run_dir:="/home/dev/results/$REL" rate:="$RATE"
