#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${ROS_WS:-$HOME/ros_ws}"
PACKAGE_DIR="$WORKSPACE/src/trust_costmap"

if [[ ! -d "$PACKAGE_DIR" ]]; then
  echo "ERROR: trust_costmap was not found at $PACKAGE_DIR" >&2
  exit 1
fi

set +u
source /opt/ros/jazzy/setup.bash
set -u

cd "$WORKSPACE"
colcon build --packages-select trust_costmap --symlink-install
set +u
source "$WORKSPACE/install/setup.bash"
set -u

exec ros2 launch trust_costmap rviz_shared_mapping.launch.py \
  map_name:="${MAP_NAME:-room-32-32-4}" \
  scenario_file:="${SCENARIO_FILE:-scenario.yaml}" \
  bridge_robot_topics:=true \
  enable_lidar_mapping:=true \
  enable_rviz:=true \
  mapping_resolution_m:="${MAPPING_RESOLUTION_M:-0.10}" \
  mapping_publish_rate_hz:="${MAPPING_PUBLISH_RATE_HZ:-2.0}" \
  max_mapping_range_m:="${MAX_MAPPING_RANGE_M:-3.5}" \
  gazebo_low_graphics:="${GAZEBO_LOW_GRAPHICS:-true}" \
  headless:="${HEADLESS:-true}" \
  experiment_duration_sec:="${EXPERIMENT_DURATION_SEC:--1}" \
  shutdown_on_manager_exit:="${SHUTDOWN_ON_MANAGER_EXIT:-false}" \
  simulate_lidar_from_map:="${SIMULATE_LIDAR_FROM_MAP:-true}" \
  "$@"
