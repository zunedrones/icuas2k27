#!/usr/bin/env bash
# Fetch the LASER UAV System and the ICUAS 2027 Crazyflie overlay at the pinned commits.
#
# Usage: docker/fetch_lus.sh [pins_file]
# The checkouts go to $GIT_ROOT (default: ~/git), the same layout the LUS installer uses,
# because LUS scripts such as spawn_drone.sh hard-code ~/git/laser_uav_system/ros_packages.
set -euo pipefail

PINS_FILE="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lus_pins.txt}"
GIT_ROOT="${GIT_ROOT:-$HOME/git}"

# PX4 submodules that the SITL build does not need (NuttX, other simulators, boards).
PX4_SKIP_SUBMODULES=(
  platforms/nuttx/NuttX/nuttx
  platforms/nuttx/NuttX/apps
  Tools/simulation/jmavsim/jMAVSim
  Tools/simulation/flightgear/flightgear_bridge
  Tools/simulation/jsbsim/jsbsim_bridge
  Tools/simulation/gz
  boards/modalai/voxl2/libfc-sensor-api
  src/drivers/cyphal/libcanard
  src/drivers/cyphal/public_regulated_data_types
  src/drivers/cyphal/legacy_data_types
)

clone_at() {
  local url=$1 sha=$2 dir=$3
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$sha" ]; then
    echo "[ ok ] $dir already at ${sha:0:7}"
    return
  fi
  mkdir -p "$dir"
  if [ ! -d "$dir/.git" ]; then
    git -C "$dir" init -q
    git -C "$dir" remote add origin "$url"
  fi
  # A shallow fetch of the exact commit keeps the checkout small (GitHub serves any commit).
  local attempt
  for attempt in 1 2 3; do
    if git -C "$dir" fetch -q --depth 1 origin "$sha"; then
      break
    fi
    echo "[warn] fetch of $url failed (attempt $attempt), retrying" >&2
    sleep $((attempt * 5))
    [ "$attempt" -eq 3 ] && { echo "[fail] cannot fetch $url@$sha" >&2; exit 1; }
  done
  git -C "$dir" -c advice.detachedHead=false checkout -q FETCH_HEAD
  echo "[ ok ] $dir @ ${sha:0:7}"
}

init_submodules() {
  local dir=$1
  shift
  local skip=("$@")
  [ -f "$dir/.gitmodules" ] || return 0
  local path
  local paths=()
  while read -r path; do
    local keep=1
    for s in "${skip[@]+"${skip[@]}"}"; do
      [ "$path" = "$s" ] && keep=0
    done
    [ "$keep" -eq 1 ] && paths+=("$path")
  done < <(git -C "$dir" config --file .gitmodules --get-regexp '\.path$' | awk '{print $2}')
  [ "${#paths[@]}" -eq 0 ] && return 0
  echo "[ .. ] submodules of $dir: ${paths[*]}"
  git -C "$dir" submodule sync -q
  if ! git -C "$dir" submodule update --init --recursive --depth 1 --jobs 8 -- "${paths[@]}"; then
    echo "[warn] shallow submodule update failed in $dir, retrying with full history" >&2
    git -C "$dir" submodule update --init --recursive --jobs 8 -- "${paths[@]}"
  fi
}

mkdir -p "$GIT_ROOT"
while read -r rel url sha; do
  [ -z "${rel:-}" ] && continue
  case "$rel" in \#*) continue ;; esac
  clone_at "$url" "$sha" "$GIT_ROOT/$rel"
done < "$PINS_FILE"

init_submodules "$GIT_ROOT/laser_uav_system/ros_packages/laser_uav_controllers"
init_submodules "$GIT_ROOT/laser_uav_system/ros_packages/px4_firmware" "${PX4_SKIP_SUBMODULES[@]}"

echo "LUS sources ready in $GIT_ROOT"
