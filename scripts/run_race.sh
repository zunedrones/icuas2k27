#!/usr/bin/env bash
# Run one autonomous race in the simulator (inside the container) and build its report.
#
# Usage: run_race.sh [--headless] [--name NAME] [--dir RUN_DIR] [launch_arg:=value ...]
#   --headless   no Gazebo GUI / RViz (evaluation mode)
#   --name NAME  suffix of the run directory (~/results/<timestamp>_<NAME>)
#   --dir DIR    explicit run directory
# Any other argument is passed to `ros2 launch icuas27_bringup race_sim.launch.py`.
set -o pipefail

HEADLESS=false
NAME=""
RUN_DIR=""
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --headless) HEADLESS=true ;;
    --name) NAME="$2"; shift ;;
    --dir) RUN_DIR="$2"; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) EXTRA+=("$1") ;;
  esac
  shift
done

if ! command -v ros2 >/dev/null 2>&1 || [ -z "${LUS_WS:-}" ]; then
  # shellcheck disable=SC1091
  source "$HOME/icuas_docker/entrypoint.sh" --source-only
fi

cleanup() {
  pkill -INT -f 'px4_sitl_default/bin/px4' 2>/dev/null
  pkill -INT -x gzserver 2>/dev/null
  pkill -INT -x gzclient 2>/dev/null
  pkill -INT -x MicroXRCEAgent 2>/dev/null
  sleep 2
  pkill -KILL -f 'px4_sitl_default/bin/px4' 2>/dev/null
  pkill -KILL -x gzserver 2>/dev/null
  pkill -KILL -x gzclient 2>/dev/null
  pkill -KILL -x MicroXRCEAgent 2>/dev/null
  return 0
}

cleanup
# Camera sensors need an OpenGL context: give a headless run a virtual display.
if [[ " ${EXTRA[*]} " == *" camera:=true "* ]] && [ -z "${DISPLAY:-}" -o "$HEADLESS" = true ]; then
  if ! pgrep -x Xvfb >/dev/null; then
    Xvfb :99 -screen 0 1280x800x24 >/dev/null 2>&1 &
    sleep 1
  fi
  export DISPLAY=:99
fi
if [ -z "$RUN_DIR" ]; then
  RUN_DIR="$HOME/results/$(date +%Y%m%d_%H%M%S)${NAME:+_$NAME}"
fi
mkdir -p "$RUN_DIR"
echo "[run_race] run directory: $RUN_DIR"

ros2 launch icuas27_bringup race_sim.launch.py run_dir:="$RUN_DIR" headless:="$HEADLESS" \
  shutdown_on_finish:=true "${EXTRA[@]}" 2>&1 | tee "$RUN_DIR/launch.log"
cleanup

if [ -f "$RUN_DIR/referee/summary.json" ]; then
  ros2 run icuas27_race race_report "$RUN_DIR" || echo "[run_race] report generation failed"
  python3 - "$RUN_DIR" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1] + '/referee/summary.json'))
adj = s.get('adjusted_time')
print('\n================ RACE RESULT ================')
print(f"state           : {s['state']}  {s.get('reason') or ''}")
print(f"gates / laps    : {s['gates_passed']}/{s['gates_total']}  {s['laps_completed']}/{s['laps_total']}")
print(f"T_elapsed       : {s['elapsed_time']:.3f} s")
print(f"penalties       : {s['penalty_time']:.1f} s  (hits {s['hits']}, missed {s['missed_entries']})")
print(f"T_adj           : {adj:.3f} s" if adj is not None else "T_adj           : - (run not valid)")
print(f"report          : {sys.argv[1]}/report/report.html")
print('=============================================')
EOF
else
  echo "[run_race] no referee summary: the race did not start (see $RUN_DIR/launch.log)"
  exit 1
fi
