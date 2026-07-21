#!/usr/bin/env bash
set -eo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${1:-$HOME/ros_ws}"

source_ros() {
  if [[ -n "${ROS_DISTRO:-}" && -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
    # shellcheck disable=SC1090
    source "/opt/ros/${ROS_DISTRO}/setup.bash"
    return
  fi
  for distro in jazzy humble iron; do
    if [[ -f "/opt/ros/${distro}/setup.bash" ]]; then
      # shellcheck disable=SC1090
      source "/opt/ros/${distro}/setup.bash"
      return
    fi
  done
  echo "No ROS 2 installation found under /opt/ros." >&2
  exit 1
}

source_ros

echo "Using ROS_DISTRO=${ROS_DISTRO}"
echo "Workspace: ${WORKSPACE}"

if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  # Install only bootstrap tools explicitly. rosdep resolves the ROS/Gazebo
  # package names for the active distribution from package.xml.
  sudo apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-yaml \
    python3-pytest
else
  echo "sudo was not found; install the dependencies listed in INSTALL_VM.md." >&2
  exit 1
fi

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi
rosdep update

mkdir -p "${WORKSPACE}/src"
TARGET="${WORKSPACE}/src/trust_costmap"
PACKAGE_REAL="$(realpath "${PACKAGE_DIR}")"
TARGET_REAL="$(realpath -m "${TARGET}")"
if [[ "${PACKAGE_REAL}" != "${TARGET_REAL}" ]]; then
  if [[ -e "${TARGET}" || -L "${TARGET}" ]]; then
    echo "Removing existing workspace package path: ${TARGET}"
    rm -rf "${TARGET}"
  fi
  ln -s "${PACKAGE_REAL}" "${TARGET}"
fi

rosdep install \
  --from-paths "${TARGET}" \
  --ignore-src \
  --rosdistro "${ROS_DISTRO}" \
  -r -y

for required_package in \
  ros_gz_sim \
  ros_gz_bridge \
  turtlebot3_gazebo \
  turtlebot3_description \
  rviz2 \
  tf2_ros
do
  ros2 pkg prefix "${required_package}" >/dev/null || {
    echo "Required ROS package was not installed: ${required_package}" >&2
    exit 1
  }
done

MODEL_FILE="/opt/ros/${ROS_DISTRO}/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf"
[[ -f "${MODEL_FILE}" ]] || {
  echo "TurtleBot3 Burger Gazebo model was not found: ${MODEL_FILE}" >&2
  exit 1
}
command -v gz >/dev/null || {
  echo "Gazebo Sim command (gz) was not found." >&2
  exit 1
}
command -v rviz2 >/dev/null || {
  echo "RViz2 command was not found." >&2
  exit 1
}

bash "${PACKAGE_DIR}/scripts/verify_project.sh" --static-only

cd "${WORKSPACE}"
colcon build --symlink-install --packages-select trust_costmap
# shellcheck disable=SC1090
source "${WORKSPACE}/install/setup.bash"

EXECUTABLES="$(ros2 pkg executables trust_costmap)"
grep -q "trust_costmap experiment_manager" <<<"${EXECUTABLES}"
grep -q "trust_costmap lidar_mapper" <<<"${EXECUTABLES}"
ros2 launch trust_costmap experiment.launch.py --show-args >/dev/null

echo
echo "Setup complete. Run:"
echo "  source ${WORKSPACE}/install/setup.bash"
echo "  bash ${TARGET}/scripts/run_rviz_demo.sh"
