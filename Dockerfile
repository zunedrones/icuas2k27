# ICUAS 2027 UAV Competition - autonomous drone racing (Phase 1, LASER UAV System simulation)
#
# Stages:
#   base  - ROS 2 Humble desktop, Gazebo Classic 11, PX4 SITL and LUS build dependencies
#   lus   - LASER UAV System + Crazyflie overlay at the pinned commits (docker/lus_pins.txt)
#   race  - this repository's ROS 2 workspace built on top of LUS (default target)
FROM osrf/ros:humble-desktop AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
      sudo git wget curl ca-certificates gnupg lsb-release bc \
      build-essential cmake ninja-build ccache pkg-config gdb rsync zip unzip astyle \
      python3-pip python3-dev python3-setuptools python3-wheel \
      python3-jinja2 python3-yaml python3-numpy python3-scipy python3-matplotlib \
      python3-opencv python3-pytest python3-colcon-common-extensions python3-vcstool \
      libxml2-dev libxml2-utils libimage-exiftool-perl libfuse2 \
      gazebo libgazebo-dev \
      ros-humble-gazebo-ros-pkgs ros-humble-gazebo-dev ros-humble-gazebo-plugins \
      ros-humble-mavlink ros-humble-pcl-conversions ros-humble-pcl-ros \
      ros-humble-tf2-sensor-msgs ros-humble-tf2-geometry-msgs ros-humble-tf2-eigen \
      ros-humble-eigen3-cmake-module ros-humble-image-transport ros-humble-cv-bridge \
      ros-humble-rqt-image-view ros-humble-plotjuggler-ros \
      libeigen3-dev libyaml-cpp-dev libboost-all-dev libunwind-dev libdw-dev libpcl-dev \
      libprotobuf-dev protobuf-compiler libopencv-dev \
      libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
      tmux tmuxinator xvfb mesa-utils libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# Python requirements of the PX4 1.15 build (Tools/setup/requirements.txt), minus the
# packages already provided by apt (numpy, matplotlib) and the ones SITL does not use.
# numpy stays below 2: the apt builds of scipy, matplotlib and cv_bridge need the 1.x ABI.
RUN pip3 install --no-cache-dir \
      argcomplete cerberus coverage empy==3.3.4 future "jinja2>=2.8" jsonschema kconfiglib \
      lxml packaging pkgconfig psutil pygments pymavlink pyros-genmsg pyserial \
      requests six "toml>=0.9" "numpy<2" \
    && python3 -c "import numpy, scipy, matplotlib; assert numpy.__version__.startswith('1.'), numpy.__version__"

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd -ms /bin/bash --uid ${USER_UID} --gid ${USER_GID} ${USERNAME} \
    && usermod -aG sudo,video ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# ---------------------------------------------------------------------------------------
FROM base AS lus

ARG USERNAME=dev
USER ${USERNAME}
WORKDIR /home/${USERNAME}

ENV LUS_ROOT=/home/${USERNAME}/git/laser_uav_system \
    LUS_WS=/home/${USERNAME}/laser_uav_system_ws \
    REAL_UAV=false \
    UAV_NAME=uav1 \
    UAV_TYPE=crazyflie

# Sources first, in their own layer: only a change of the pins re-downloads them.
COPY --chown=${USERNAME}:${USERNAME} docker/lus_pins.txt docker/fetch_lus.sh /home/${USERNAME}/icuas_docker/
RUN bash /home/${USERNAME}/icuas_docker/fetch_lus.sh /home/${USERNAME}/icuas_docker/lus_pins.txt

COPY --chown=${USERNAME}:${USERNAME} docker/build_lus.sh docker/lus_env.sh /home/${USERNAME}/icuas_docker/
ARG JOBS=8
ENV JOBS=${JOBS}
RUN bash /home/${USERNAME}/icuas_docker/build_lus.sh overlay
RUN bash /home/${USERNAME}/icuas_docker/build_lus.sh px4 \
    && find ${LUS_ROOT}/ros_packages/px4_firmware/build -name '*.o' -delete
RUN bash /home/${USERNAME}/icuas_docker/build_lus.sh agent
RUN bash /home/${USERNAME}/icuas_docker/build_lus.sh acados
# The LUS workspace is built with --symlink-install, so its build/ directory must stay.
RUN bash /home/${USERNAME}/icuas_docker/build_lus.sh workspace && rm -rf ${LUS_WS}/log

# ---------------------------------------------------------------------------------------
FROM lus AS race

ARG USERNAME=dev
COPY --chown=${USERNAME}:${USERNAME} docker/entrypoint.sh /home/${USERNAME}/icuas_docker/entrypoint.sh
COPY --chown=${USERNAME}:${USERNAME} scripts /home/${USERNAME}/scripts
COPY --chown=${USERNAME}:${USERNAME} ros2_ws/src /home/${USERNAME}/ros2_ws/src

RUN bash -c "source /opt/ros/humble/setup.bash \
      && source /home/${USERNAME}/icuas_docker/lus_env.sh \
      && source ${LUS_WS}/install/setup.bash \
      && cd /home/${USERNAME}/ros2_ws \
      && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release" \
    && echo 'source /home/'${USERNAME}'/icuas_docker/entrypoint.sh --source-only' >> /home/${USERNAME}/.bashrc \
    && chmod +x /home/${USERNAME}/scripts/*.sh /home/${USERNAME}/icuas_docker/*.sh

RUN mkdir -p /home/${USERNAME}/results

ENTRYPOINT ["/home/dev/icuas_docker/entrypoint.sh"]
CMD ["bash"]
