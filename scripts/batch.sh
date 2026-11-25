#!/usr/bin/env bash
# Host: run N headless races (the organisers run each solution at least 3 times and keep the
# lowest T_adj) and build the comparison report.
#   ./scripts/batch.sh [N] [launch_arg:=value ...]
set -e
cd "$(dirname "$0")/.."
N="${1:-3}"
shift || true
BATCH="batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "results/$BATCH"
export ICUAS_GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || true)"
docker compose run --rm sim bash -c "
  for i in \$(seq -w 1 $N); do
    /home/dev/scripts/run_race.sh --headless --dir /home/dev/results/$BATCH/run_\$i $* || true
  done
  ros2 run icuas27_race race_compare /home/dev/results/$BATCH --out /home/dev/results/$BATCH"
echo "comparison: results/$BATCH/compare.html"
