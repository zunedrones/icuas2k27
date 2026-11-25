#!/usr/bin/env bash
# Container entrypoint: source ROS 2, the LASER UAV System and this workspace, then run CMD.
# With --source-only it just sets up the environment (used from ~/.bashrc).

_icuas_source() {
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  # shellcheck disable=SC1091
  source "$HOME/icuas_docker/lus_env.sh"
  if [ -f "$LUS_WS/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$LUS_WS/install/setup.bash"
  fi
  if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$HOME/ros2_ws/install/setup.bash"
  fi
}

if [ "${1:-}" = "--source-only" ]; then
  _icuas_source
  return 0 2>/dev/null || exit 0
fi

_icuas_source

# The repository is bind-mounted over ~/ros2_ws during development: build it on first use.
if [ "${ICUAS_AUTO_BUILD:-1}" = "1" ] && [ -d "$HOME/ros2_ws/src" ] && \
   [ ! -f "$HOME/ros2_ws/install/setup.bash" ]; then
  echo "[entrypoint] building ~/ros2_ws (first run)"
  (cd "$HOME/ros2_ws" && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release) || exit 1
  # shellcheck disable=SC1091
  source "$HOME/ros2_ws/install/setup.bash"
fi

exec "$@"
