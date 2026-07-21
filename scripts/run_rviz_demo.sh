#!/usr/bin/env bash
set -eo pipefail

WORKSPACE="${TRUST_COSTMAP_WORKSPACE:-$HOME/ros_ws}"
if [[ -f "${WORKSPACE}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${WORKSPACE}/install/setup.bash"
elif [[ -z "${ROS_DISTRO:-}" ]]; then
  echo "Source the ROS 2 and trust_costmap workspace setup files first." >&2
  exit 1
fi

exec ros2 launch trust_costmap experiment.launch.py \
  costmap_method:=hard_threshold \
  map_name:=room-32-32-4 \
  scenario_file:=scenario.yaml \
  planner:=astar \
  allow_diagonal:=false \
  bridge_robot_topics:=true \
  action_goal_count:=8 \
  action_goal_seed:=21 \
  run_label:=hard_threshold_rviz_test \
  enable_lidar_mapping:=true \
  enable_rviz:=true \
  rviz_robot_id:=auto \
  gazebo_low_graphics:=false \
  "$@"
