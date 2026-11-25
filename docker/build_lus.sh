#!/usr/bin/env bash
# Build the pinned LASER UAV System for simulation: Crazyflie overlay, PX4 SITL with the
# Gazebo Classic plugins, the Micro XRCE-DDS agent, acados and the LUS ROS 2 workspace.
# Mirrors laser_uav_system/install.sh (simulation, px4) without its interactive prompts.
#
# Usage: build_lus.sh [all|overlay|px4|agent|acados|workspace]...   (default: all)
set -euo pipefail

GIT_ROOT="${GIT_ROOT:-$HOME/git}"
export LUS_ROOT="${LUS_ROOT:-$GIT_ROOT/laser_uav_system}"
export LUS_WS="${LUS_WS:-$HOME/laser_uav_system_ws}"
CRAZYFLIE_ROOT="${CRAZYFLIE_ROOT:-$GIT_ROOT/laser_crazyflie}"
JOBS="${JOBS:-$(nproc)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Release tag the pinned px4_firmware-ROS2- commit descends from (559b693 = v1.15.4-8-g559b693c).
PX4_TAG="${PX4_TAG:-v1.15.4}"

PKG="$LUS_ROOT/ros_packages"

step_overlay() {
  echo "=== Crazyflie overlay (laser_crazyflie/install_crazyflie.sh)"
  LASER_UAV_SYSTEM="$LUS_ROOT" bash "$CRAZYFLIE_ROOT/install_crazyflie.sh"
}

step_px4() {
  echo "=== PX4 SITL and Gazebo Classic plugins"
  cd "$PKG/px4_firmware"
  # PX4 derives its version from `git describe --tags`; the shallow checkout has no tags.
  # Deepen the pinned commit's history enough to contain the release tag it descends from.
  if ! git describe --tags --exclude 'ext/*' >/dev/null 2>&1; then
    local sha
    sha=$(git rev-parse HEAD)
    if ! git fetch -q --depth 64 origin "$sha" "+refs/tags/${PX4_TAG}:refs/tags/${PX4_TAG}"; then
      echo "[warn] could not fetch ${PX4_TAG}; tagging HEAD locally" >&2
      git tag -f "${PX4_TAG}" HEAD
    fi
  fi
  echo "PX4 version: $(git describe --tags --exclude 'ext/*')"
  make px4_sitl_default
  mkdir -p build/px4_sitl_default/build_gazebo-classic
  cd build/px4_sitl_default/build_gazebo-classic
  cmake ../../../Tools/simulation/gazebo-classic/sitl_gazebo-classic -DCMAKE_BUILD_TYPE=Release
  make -j"$JOBS"
  test -x "$PKG/px4_firmware/build/px4_sitl_default/bin/px4"
}

step_agent() {
  echo "=== Micro XRCE-DDS agent"
  cd "$PKG/micro_xrce_dds_agent"
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"$JOBS"
  sudo cmake --install build
  sudo ldconfig /usr/local/lib
  command -v MicroXRCEAgent
}

step_acados() {
  echo "=== acados (NMPC solver)"
  cd "$PKG/laser_uav_controllers/acados"
  mkdir -p build
  cd build
  cmake -DACADOS_WITH_QPOASES=ON ..
  make install -j"$JOBS"
}

step_workspace() {
  echo "=== LUS ROS 2 workspace"
  mkdir -p "$LUS_WS/src"
  local p
  for p in "$PKG"/*; do
    # laser_uav_system commits gitman links for every optional package; skip the ones whose
    # checkout was not fetched (ArduPilot, FAST-LIO, OpenVINS, micro-ROS) instead of
    # handing colcon dangling links.
    [ -e "$p" ] || continue
    case "$(basename "$p")" in
      px4_firmware | micro_xrce_dds_agent) ;;  # built separately, not colcon packages
      *) ln -sfn "$p" "$LUS_WS/src/$(basename "$p")" ;;
    esac
  done
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/lus_env.sh"
  set -u
  cd "$LUS_WS"
  colcon build --symlink-install --parallel-workers "$JOBS" \
    --packages-up-to laser_uav_managers laser_uav_hw_api laser_uav_simulation \
      laser_gazebo_resources laser_uav_tui laser_uav_imu_filter \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-Wno-error" \
      -DCMAKE_CXX_FLAGS="-Wno-error" -DBUILD_TESTING=OFF -Wno-dev \
    --event-handlers console_cohesion+
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(all)
for s in "${steps[@]}"; do
  case "$s" in
    all) (step_overlay) && (step_px4) && (step_agent) && (step_acados) && (step_workspace) ;;
    overlay|px4|agent|acados|workspace) ("step_$s") ;;
    *) echo "unknown step: $s" >&2; exit 2 ;;
  esac
done
echo "LUS build step(s) done: ${steps[*]}"
