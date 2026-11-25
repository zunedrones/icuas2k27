#!/usr/bin/env bash
# Host: run one race in the container with the Gazebo GUI and RViz (or --headless).
#   ./scripts/race.sh [--headless] [--name NAME] [launch_arg:=value ...]
set -e
cd "$(dirname "$0")/.."
mkdir -p results
command -v xhost >/dev/null 2>&1 && xhost +local:docker >/dev/null 2>&1 || true
export ICUAS_GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || true)"
exec docker compose run --rm sim /home/dev/scripts/run_race.sh "$@"
