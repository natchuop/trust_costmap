#!/usr/bin/env python3
"""Static integration checks for launch helpers without requiring ROS 2.

The script supplies minimal stand-ins for launch classes, then exercises the
repository's pure launch-time logic: scenario parsing, benign RViz selection,
per-robot mapper creation, RViz rendering, and low-graphics SDF generation.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


PACKAGE_DIR = Path(__file__).resolve().parents[1]


class Dummy:
    """Capture launch-action constructor arguments for static inspection."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def install_import_stubs() -> None:
    ament = types.ModuleType("ament_index_python")
    ament_packages = types.ModuleType("ament_index_python.packages")
    ament_packages.get_package_prefix = lambda name: "/tmp/prefix"
    ament_packages.get_package_share_directory = lambda name: "/tmp/share"
    sys.modules["ament_index_python"] = ament
    sys.modules["ament_index_python.packages"] = ament_packages

    launch = types.ModuleType("launch")
    launch.LaunchDescription = Dummy
    sys.modules["launch"] = launch

    launch_actions = types.ModuleType("launch.actions")
    for name in (
        "DeclareLaunchArgument",
        "EmitEvent",
        "ExecuteProcess",
        "LogInfo",
        "OpaqueFunction",
        "RegisterEventHandler",
        "SetEnvironmentVariable",
        "TimerAction",
    ):
        setattr(launch_actions, name, Dummy)
    sys.modules["launch.actions"] = launch_actions

    launch_handlers = types.ModuleType("launch.event_handlers")
    launch_handlers.OnProcessExit = Dummy
    sys.modules["launch.event_handlers"] = launch_handlers

    launch_events = types.ModuleType("launch.events")
    launch_events.Shutdown = Dummy
    sys.modules["launch.events"] = launch_events

    launch_substitutions = types.ModuleType("launch.substitutions")
    launch_substitutions.LaunchConfiguration = Dummy
    sys.modules["launch.substitutions"] = launch_substitutions

    sys.modules["launch_ros"] = types.ModuleType("launch_ros")
    launch_ros_actions = types.ModuleType("launch_ros.actions")
    launch_ros_actions.Node = Dummy
    sys.modules["launch_ros.actions"] = launch_ros_actions

    launch_ros_parameters = types.ModuleType("launch_ros.parameter_descriptions")
    launch_ros_parameters.ParameterValue = Dummy
    sys.modules["launch_ros.parameter_descriptions"] = launch_ros_parameters


def load_launch_module():
    install_import_stubs()
    launch_path = PACKAGE_DIR / "experiment.launch.py"
    spec = importlib.util.spec_from_file_location(
        "trust_costmap_experiment_launch_static_check", launch_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load launch file: {launch_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_launch_module()
    scenario = yaml.safe_load(
        (PACKAGE_DIR / "scenario.yaml").read_text(encoding="utf-8")
    )
    map_data = module.load_movingai_map(
        PACKAGE_DIR / "worlds" / "movingai_mapf" / "room-32-32-4.map"
    )

    selected = module.select_rviz_robot_id(scenario, "auto")
    if selected != "benign_1":
        raise AssertionError(f"Expected benign_1, selected {selected!r}")

    try:
        module.select_rviz_robot_id(scenario, "does_not_exist")
    except RuntimeError as error:
        if "Choose one of" not in str(error):
            raise
    else:
        raise AssertionError("An invalid RViz robot ID did not fail")

    enabled = module.enabled_robots(scenario)
    actions = module.make_lidar_mapping_actions(
        scenario,
        map_data,
        publish_rate_hz=2.0,
        scan_stride=2,
        rviz_robot_id=selected,
        rviz_outputs_enabled=True,
    )
    if len(actions) != len(enabled):
        raise AssertionError("Mapper count does not match enabled robot count")

    rviz_output_count = 0
    for action, robot in zip(actions, enabled):
        parameters = action.kwargs["parameters"][0]
        robot_id = str(robot["id"])
        assert parameters["robot_id"] == robot_id
        assert parameters["scan_topic"] == f"/{robot_id}/scan"
        assert parameters["map_pose_topic"] == f"/{robot_id}/map_pose"
        assert parameters["scan_stride"] == 2
        if parameters["enable_rviz_outputs"]:
            rviz_output_count += 1
            assert robot_id == selected
    assert rviz_output_count == 1

    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        rendered = temporary / "benign_1.rviz"
        module.render_rviz_config(
            PACKAGE_DIR,
            rendered,
            selected,
            peer_robot_ids=["benign_2", "malicious_1"],
        )
        rendered_text = rendered.read_text(encoding="utf-8")
        assert "__ROBOT_ID__" not in rendered_text
        assert "__PEER_SCAN_DISPLAYS__" not in rendered_text
        assert "/benign_1/lidar_free_cells" in rendered_text
        assert "/benign_1/lidar_occupied_cells" in rendered_text
        assert "/rviz_base_free" in rendered_text
        assert "/rviz_base_occupied" in rendered_text
        assert "/benign_2/scan_rviz" in rendered_text
        assert "/malicious_1/scan_rviz" in rendered_text
        assert "rviz_default_plugins/Map" not in rendered_text
        yaml.safe_load(rendered_text)

        model_path = temporary / "model.sdf"
        model_path.write_text(
            """<sdf version="1.9">
  <model name="burger">
    <pose>0 0 0 0 0 0</pose>
    <link name="base_link">
      <visual name="textured">
        <geometry><box><size>1 1 1</size></box></geometry>
        <material><pbr><metal><albedo_map>old.png</albedo_map></metal></pbr></material>
      </visual>
      <collision name="collision">
        <geometry><box><size>1 1 1</size></box></geometry>
      </collision>
      <sensor name="lidar" type="gpu_lidar"/>
    </link>
    <plugin name="diff_drive" filename="gz-sim-diff-drive-system"/>
  </model>
</sdf>
""",
            encoding="utf-8",
        )
        world_path = temporary / "world.sdf"
        module.generate_sdf_world(
            "room-32-32-4",
            map_data,
            scenario,
            model_path,
            world_path,
            low_graphics=True,
        )
        root = ET.parse(world_path).getroot()
        assert root.find(".//scene/shadows").text == "false"
        models = {
            model.get("name"): model for model in root.findall(".//world/model")
        }
        for robot in enabled:
            robot_model = models[str(robot["id"])]
            assert robot_model.find(".//sensor") is not None
            assert robot_model.find(".//plugin") is not None
            visuals = robot_model.findall(".//visual")
            assert len(visuals) == 1
            assert visuals[0].get("name") == "low_graphics_body"
            assert robot_model.find(".//visual/material/pbr") is None
            assert robot_model.find(".//collision") is not None

    print("Launch helper integration checks passed")


if __name__ == "__main__":
    main()
