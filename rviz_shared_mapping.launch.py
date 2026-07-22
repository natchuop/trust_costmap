#!/usr/bin/env python3
"""Run the existing experiment plus the LiDAR-only shared RViz mapper."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node



def _declared_arguments(path: str):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return set()
    return set(
        re.findall(
            r"DeclareLaunchArgument\(\s*[\"']([^\"']+)[\"']",
            text,
            flags=re.MULTILINE,
        )
    )


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _setup(context):
    package_share = get_package_share_directory("trust_costmap")
    base_launch = os.path.join(package_share, "experiment.launch.py")
    map_name = LaunchConfiguration("map_name").perform(context)
    scenario_file = LaunchConfiguration("scenario_file").perform(context)
    scenario_path = os.path.join(package_share, scenario_file)
    map_path = os.path.join(package_share, "worlds", "movingai_mapf", f"{map_name}.map")

    # Capture parent enable flags BEFORE IncludeLaunchDescription overrides the
    # same names to false for the base experiment launch. Those overrides share
    # the launch context, so delayed IfCondition checks would otherwise stay false.
    enable_mapping = _truthy(
        LaunchConfiguration("enable_lidar_mapping").perform(context)
    )
    enable_rviz = _truthy(LaunchConfiguration("enable_rviz").perform(context))

    with open(scenario_path, "r", encoding="utf-8") as handle:
        scenario = yaml.safe_load(handle) or {}
    robots = [
        str(robot["id"])
        for robot in scenario.get("robots", [])
        if robot.get("enabled", True) and "id" in robot
    ]
    cell_size = float((scenario.get("visualization") or {}).get("cell_size_m", 0.5))

    supported = _declared_arguments(base_launch)
    possible_forward = (
        "map_name",
        "scenario_file",
        "bridge_robot_topics",
        "costmap_method",
        "planner",
        "allow_diagonal",
        "action_goal_count",
        "action_goal_seed",
        "run_label",
        "gazebo_low_graphics",
        "shutdown_on_manager_exit",
        "experiment_duration_sec",
        "headless",
    )
    launch_arguments = {
        name: LaunchConfiguration(name)
        for name in possible_forward
        if name in supported
    }
    # Disable any older integrated mapper/RViz implementation so there is only
    # one RViz window and no static-map GridCells display.
    if "enable_lidar_mapping" in supported:
        launch_arguments["enable_lidar_mapping"] = "false"
    if "enable_rviz" in supported:
        launch_arguments["enable_rviz"] = "false"

    actions = []

    # Start RViz/mapper BEFORE including experiment.launch.py. That include runs a
    # long attack-heatmap prepare step; if RViz is scheduled after it, no window
    # appears for ~1 minute and users think the launch failed.
    if enable_rviz:
        rviz_config = os.path.join(package_share, "config", "lidar_mapping.rviz")
        rviz = Node(
            package="rviz2",
            executable="rviz2",
            name="shared_lidar_rviz",
            output="screen",
            arguments=["-d", rviz_config],
            additional_env={"DISPLAY": os.environ.get("DISPLAY", ":0")},
        )
        actions.append(rviz)

    if enable_mapping:
        mapper = Node(
            package="trust_costmap",
            executable="lidar_mapper",
            name="shared_lidar_mapper",
            output="screen",
            parameters=[
                {
                    "robot_ids": robots,
                    "map_path": map_path,
                    "scenario_path": scenario_path,
                    "cell_size_m": cell_size,
                    "mapping_resolution_m": LaunchConfiguration("mapping_resolution_m"),
                    "publish_rate_hz": LaunchConfiguration("mapping_publish_rate_hz"),
                    "max_mapping_range_m": LaunchConfiguration("max_mapping_range_m"),
                    "scan_stride": LaunchConfiguration("scan_stride"),
                    "simulate_lidar_from_map": LaunchConfiguration(
                        "simulate_lidar_from_map"
                    ),
                }
            ],
        )
        actions.append(mapper)

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(base_launch),
            launch_arguments=launch_arguments.items(),
        )
    )
    return actions


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("map_name", default_value="room-32-32-4"),
        DeclareLaunchArgument("scenario_file", default_value="scenario.yaml"),
        DeclareLaunchArgument("bridge_robot_topics", default_value="true"),
        DeclareLaunchArgument("enable_lidar_mapping", default_value="true"),
        DeclareLaunchArgument("enable_rviz", default_value="true"),
        DeclareLaunchArgument("mapping_resolution_m", default_value="0.10"),
        DeclareLaunchArgument("mapping_publish_rate_hz", default_value="2.0"),
        DeclareLaunchArgument("max_mapping_range_m", default_value="3.5"),
        DeclareLaunchArgument("scan_stride", default_value="1"),
        DeclareLaunchArgument("costmap_method", default_value="hard_threshold"),
        DeclareLaunchArgument("planner", default_value="astar"),
        DeclareLaunchArgument("allow_diagonal", default_value="false"),
        DeclareLaunchArgument("action_goal_count", default_value="8"),
        DeclareLaunchArgument("action_goal_seed", default_value="21"),
        DeclareLaunchArgument("run_label", default_value="shared_lidar_rviz"),
        DeclareLaunchArgument("gazebo_low_graphics", default_value="true"),
        DeclareLaunchArgument(
            "shutdown_on_manager_exit",
            default_value="false",
            description="Keep RViz/mapping alive after the experiment manager exits.",
        ),
        DeclareLaunchArgument(
            "experiment_duration_sec",
            default_value="-1",
            description="Use <0 to disable the scenario timed shutdown.",
        ),
        DeclareLaunchArgument(
            "simulate_lidar_from_map",
            default_value="true",
            description=(
                "Raycast MovingAI walls immediately so RViz fills without waiting "
                "for slow Gazebo gpu_lidar on VMs."
            ),
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="true",
            description=(
                "Run Gazebo server-only. Recommended on VMs without GPU; "
                "this wrapper still launches its own RViz window."
            ),
        ),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_setup)])
