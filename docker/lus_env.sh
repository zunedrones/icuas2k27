# Environment of the LASER UAV System simulation (sourced, not executed).
# Mirrors what laser_uav_system/install.sh appends to ~/.bashrc.

export REAL_UAV="${REAL_UAV:-false}"
export UAV_NAME="${UAV_NAME:-uav1}"
export UAV_TYPE="${UAV_TYPE:-crazyflie}"

export LUS_ROOT="${LUS_ROOT:-$HOME/git/laser_uav_system}"
export LUS_WS="${LUS_WS:-$HOME/laser_uav_system_ws}"

export ACADOS_SOURCE_DIR="$LUS_WS/src/laser_uav_controllers/acados"
_px4_gz_build="$LUS_ROOT/ros_packages/px4_firmware/build/px4_sitl_default/build_gazebo-classic"

if [ -f /usr/share/gazebo/setup.sh ]; then
  # shellcheck disable=SC1091
  . /usr/share/gazebo/setup.sh
fi

case ":${LD_LIBRARY_PATH:-}:" in
  *":$ACADOS_SOURCE_DIR/lib:"*) ;;
  *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ACADOS_SOURCE_DIR/lib:$_px4_gz_build" ;;
esac
case ":${GAZEBO_PLUGIN_PATH:-}:" in
  *":$_px4_gz_build:"*) ;;
  *) export GAZEBO_PLUGIN_PATH="${GAZEBO_PLUGIN_PATH:+$GAZEBO_PLUGIN_PATH:}$_px4_gz_build" ;;
esac
_lus_models="$LUS_ROOT/ros_packages/px4_firmware/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:$LUS_ROOT/ros_packages/laser_uav_simulation/models:$LUS_ROOT/ros_packages/laser_uav_simulation/core/models"
case ":${GAZEBO_MODEL_PATH:-}:" in
  *":$LUS_ROOT/ros_packages/laser_uav_simulation/models:"*) ;;
  *) export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH:+$GAZEBO_MODEL_PATH:}$_lus_models" ;;
esac

# Never reach for the online model database: the evaluation must run offline.
export GAZEBO_MODEL_DATABASE_URI=""

unset _px4_gz_build _lus_models
