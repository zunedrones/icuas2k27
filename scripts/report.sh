#!/usr/bin/env bash
# Host: (re)build the analytics report of a run, or compare several runs.
#   ./scripts/report.sh results/<run>
#   ./scripts/report.sh --compare results/<batch_or_runs...>
set -e
cd "$(dirname "$0")/.."
if [ "$1" = "--compare" ]; then
  shift
  ARGS=()
  for r in "$@"; do ARGS+=("/home/dev/results/$(realpath --relative-to=results "$r")"); done
  exec docker compose run --rm sim ros2 run icuas27_race race_compare "${ARGS[@]}"
fi
RUN="${1:?usage: report.sh results/<run>}"
exec docker compose run --rm sim ros2 run icuas27_race race_report \
  "/home/dev/results/$(realpath --relative-to=results "$RUN")"
