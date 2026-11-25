#!/usr/bin/env bash
# Inside the container: quick checks that do not need the simulator.
#   1. unit tests of the pure-Python core
#   2. launch files parse (arguments listed)
#   3. the race plan is predicted with the exact PMM planner and validated
set -e
if ! command -v ros2 >/dev/null 2>&1 || [ -z "${LUS_WS:-}" ]; then
  # shellcheck disable=SC1091
  source "$HOME/icuas_docker/entrypoint.sh" --source-only
fi
SRC="$HOME/ros2_ws/src"
echo "== unit tests"
(cd "$SRC/icuas27_race" && PYTHONPATH="$SRC/icuas27_race:$SRC/icuas27_race/test" \
  MPLBACKEND=Agg python3 -m pytest -q -p no:cacheprovider test/)
echo "== launch files"
for f in race_sim sim race replay; do
  ros2 launch icuas27_bringup "$f.launch.py" --show-args >/dev/null && echo "  $f.launch.py ok"
done
echo "== race plan (PMM prediction + validation)"
OUT="$(mktemp -d)"
ros2 run icuas27_race plan_course --start-predictor --out "$OUT"
python3 - "$OUT/plan_sweep.json" <<'EOF'
import json, sys
rows = json.load(open(sys.argv[1]))
bad = [r for r in rows if not r.get('ok')]
if bad:
    print('INVALID plan(s):', bad)
    sys.exit(1)
print('plan OK')
EOF
