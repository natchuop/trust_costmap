#!/usr/bin/env python3
"""Launch a reproducible trust-costmap experiment.

This launch file prepares both malicious-attack experiments and clean/baseline
runs. It keeps experiment orchestration in one place while leaving navigation,
claim processing, trust updates, attack execution, and metric computation in
the experiment manager.

Prepared capabilities
---------------------
- MovingAI map and scenario validation.
- Per-run output directories and immutable launch manifests.
- Sweep-friendly run labels, trial indices, method overrides, and seed overrides.
- Optional bounded attack-reconnaissance heatmap generation and validation.
- Exact heatmap-path handoff to the experiment manager.
- Runtime Gazebo world generation outside the package share directory.
- Optional per-robot ROS-Gazebo bridges.
- Optional rosbag recording for later metric reconstruction and auditing.
- Scenario-duration or launch-duration timed shutdown.
- Shutdown of the complete launch if Gazebo or the experiment manager exits.
- Environment variables reserved for future metrics/logging components.

The launch file intentionally does not calculate final experiment metrics. A
future metrics collector can consume the run directory, launch manifest,
manager logs, claim/trust logs, and optional rosbag without requiring another
change to the experiment command line.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


PACKAGE_NAME = "trust_costmap"
FREE_SYMBOLS = {".", "G", "S"}
DEFAULT_RUNTIME_ROOT = Path.home() / ".ros" / PACKAGE_NAME
Cell = Tuple[int, int]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_auto_bool(value: object, automatic_value: bool) -> bool:
    clean = str(value).strip().lower()
    if clean in {"", "auto"}:
        return automatic_value
    if clean in {"1", "true", "yes", "y", "on"}:
        return True
    if clean in {"0", "false", "no", "n", "off"}:
        return False
    raise RuntimeError(f"Expected true, false, or auto; received {value!r}")


def sanitize_identifier(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "unnamed"
    result = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in text
    ).strip("._-")
    return result or "unnamed"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_path(package_share: Path, requested: str) -> Path:
    candidate = Path(os.path.expandvars(os.path.expanduser(requested)))
    return candidate if candidate.is_absolute() else package_share / candidate


def make_world_name(map_name: str) -> str:
    return f"{sanitize_identifier(map_name).replace('-', '_')}_world"


# ---------------------------------------------------------------------------
# MovingAI map handling
# ---------------------------------------------------------------------------


def load_movingai_map(map_path: Path) -> Dict[str, Any]:
    lines = map_path.read_text(encoding="utf-8").splitlines()
    map_type: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    map_start: Optional[int] = None

    for index, line in enumerate(lines):
        clean = line.strip()
        if clean.startswith("type "):
            map_type = clean.split(maxsplit=1)[1]
        elif clean.startswith("height "):
            height = int(clean.split(maxsplit=1)[1])
        elif clean.startswith("width "):
            width = int(clean.split(maxsplit=1)[1])
        elif clean == "map":
            map_start = index + 1
            break

    if map_type is None or height is None or width is None or map_start is None:
        raise RuntimeError(f"Invalid MovingAI map header: {map_path}")

    grid = lines[map_start : map_start + height]
    if len(grid) != height:
        raise RuntimeError(f"Expected {height} rows, found {len(grid)}: {map_path}")

    for row_index, row in enumerate(grid):
        if len(row) != width:
            raise RuntimeError(
                f"Map row {row_index} expected width {width}, found {len(row)}"
            )

    return {
        "name": map_path.stem,
        "map_type": map_type,
        "height": height,
        "width": width,
        "grid": grid,
    }


def is_blocked(character: str) -> bool:
    return character not in FREE_SYMBOLS


def is_free_cell(map_data: Dict[str, Any], cell: Cell) -> bool:
    row, col = cell
    return (
        0 <= row < int(map_data["height"])
        and 0 <= col < int(map_data["width"])
        and map_data["grid"][row][col] in FREE_SYMBOLS
    )


def cell_to_world(row: int, col: int, cell_size: float) -> Tuple[float, float]:
    return (col + 0.5) * cell_size, (row + 0.5) * cell_size


def extract_horizontal_wall_segments(grid: Sequence[str]) -> List[Dict[str, int]]:
    segments: List[Dict[str, int]] = []
    for row_index, row in enumerate(grid):
        col = 0
        while col < len(row):
            if not is_blocked(row[col]):
                col += 1
                continue
            start_col = col
            while col < len(row) and is_blocked(row[col]):
                col += 1
            end_col = col - 1
            segments.append(
                {
                    "row": row_index,
                    "start_col": start_col,
                    "end_col": end_col,
                    "length_cells": end_col - start_col + 1,
                }
            )
    return segments


# ---------------------------------------------------------------------------
# Scenario handling and validation
# ---------------------------------------------------------------------------


def load_scenario(path: Path) -> Dict[str, Any]:
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict):
        raise RuntimeError(f"Scenario must contain a YAML mapping: {path}")
    return scenario


def enabled_robots(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    robots = [
        robot
        for robot in scenario.get("robots", [])
        if bool(robot.get("enabled", True))
    ]
    if not robots:
        raise RuntimeError("Scenario must contain at least one enabled robot.")
    return robots


def validate_robot_configuration(
    scenario: Dict[str, Any], map_data: Dict[str, Any]
) -> None:
    seen_ids: set[str] = set()
    for robot in enabled_robots(scenario):
        robot_id = str(robot.get("id", "")).strip()
        if not robot_id:
            raise RuntimeError("Every enabled robot must define a non-empty id.")
        if robot_id in seen_ids:
            raise RuntimeError(f"Duplicate robot id: {robot_id}")
        seen_ids.add(robot_id)

        if "start_cell" not in robot:
            raise RuntimeError(f"Robot {robot_id} does not define start_cell.")
        start = tuple(int(value) for value in robot["start_cell"])
        if len(start) != 2 or not is_free_cell(map_data, start):
            raise RuntimeError(
                f"Robot {robot_id} start_cell is blocked or out of bounds: {start}"
            )

        goal = robot.get("goal_cell")
        if goal is not None:
            goal_cell = tuple(int(value) for value in goal)
            if len(goal_cell) != 2 or not is_free_cell(map_data, goal_cell):
                raise RuntimeError(
                    f"Robot {robot_id} goal_cell is blocked or out of bounds: "
                    f"{goal_cell}"
                )


def malicious_attack_config(scenario: Dict[str, Any]) -> Dict[str, Any]:
    config = scenario.get("malicious_attack", scenario.get("attack_runtime", {}))
    return config if isinstance(config, dict) else {}


def attack_enabled(scenario: Dict[str, Any]) -> bool:
    return bool(malicious_attack_config(scenario).get("enabled", False))


def malicious_robot_ids(scenario: Dict[str, Any]) -> List[str]:
    config = malicious_attack_config(scenario)
    explicit = [str(value) for value in config.get("robot_ids", [])]
    if explicit:
        return explicit
    return [
        str(robot["id"])
        for robot in enabled_robots(scenario)
        if "malicious" in str(robot.get("role", "")).lower()
        or "attacker" in str(robot.get("role", "")).lower()
    ]


def validate_attack_configuration(scenario: Dict[str, Any]) -> None:
    if not attack_enabled(scenario):
        return

    enabled_ids = {str(robot["id"]) for robot in enabled_robots(scenario)}
    attackers = malicious_robot_ids(scenario)
    if not attackers:
        raise RuntimeError(
            "malicious_attack.enabled is true, but no malicious robot could be "
            "resolved from malicious_attack.robot_ids or robot roles."
        )

    missing = sorted(set(attackers) - enabled_ids)
    if missing:
        raise RuntimeError(
            "malicious_attack references disabled or unknown robots: "
            + ", ".join(missing)
        )

    policy = str(
        malicious_attack_config(scenario).get("placement_policy", "topology_only")
    ).strip().lower()
    if policy not in {"topology_only", "route_informed", "route_only"}:
        raise RuntimeError(
            "malicious_attack.placement_policy must be topology_only, "
            "route_informed, or route_only."
        )


def effective_costmap_method(
    scenario: Dict[str, Any], launch_override: str
) -> str:
    override = launch_override.strip()
    if override:
        return override
    return str(scenario.get("experiment", {}).get("method", "static")).strip()


def effective_duration_sec(
    scenario: Dict[str, Any], duration_override: str
) -> float:
    override = float(duration_override)
    if override > 0.0:
        return override
    return max(0.0, float(scenario.get("experiment", {}).get("duration_sec", 0.0)))


def effective_experiment_seed(
    scenario: Dict[str, Any], seed_override: str
) -> int:
    override = int(seed_override)
    if override >= 0:
        return override
    return int(scenario.get("experiment", {}).get("random_seed", 0))


# ---------------------------------------------------------------------------
# Gazebo world generation
# ---------------------------------------------------------------------------


def append_box_model(
    parts: List[str],
    name: str,
    pose_xyz_rpy: Iterable[object],
    size_xyz: Iterable[object],
    ambient_rgba: str,
    diffuse_rgba: str,
    static: bool = True,
) -> None:
    parts.extend(
        [
            f'    <model name="{name}">',
            f"      <static>{str(static).lower()}</static>",
            f"      <pose>{' '.join(str(value) for value in pose_xyz_rpy)}</pose>",
            '      <link name="link">',
            '        <collision name="collision">',
            "          <geometry>",
            "            <box>",
            f"              <size>{' '.join(str(value) for value in size_xyz)}</size>",
            "            </box>",
            "          </geometry>",
            "        </collision>",
            '        <visual name="visual">',
            "          <cast_shadows>false</cast_shadows>",
            "          <geometry>",
            "            <box>",
            f"              <size>{' '.join(str(value) for value in size_xyz)}</size>",
            "            </box>",
            "          </geometry>",
            "          <material>",
            f"            <ambient>{ambient_rgba}</ambient>",
            f"            <diffuse>{diffuse_rgba}</diffuse>",
            "          </material>",
            "        </visual>",
            "      </link>",
            "    </model>",
        ]
    )


def set_xml_child_text(parent: ET.Element, tag: str, value: object) -> None:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    child.text = str(value)


def simplify_turtlebot_visuals(model: ET.Element) -> None:
    """Replace texture-heavy robot visuals with one primitive body.

    Collision geometry, joints, sensors, and plugins remain untouched.  Only
    render-only ``<visual>`` elements are removed, which avoids loading the
    TurtleBot mesh and PBR texture assets in slow virtual machines.
    """

    links = model.findall("./link")
    if not links:
        return

    for link in links:
        for visual in list(link.findall("visual")):
            link.remove(visual)

    body_link = next(
        (link for link in links if link.get("name", "") == "base_link"),
        links[0],
    )
    visual = ET.SubElement(body_link, "visual", {"name": "low_graphics_body"})
    set_xml_child_text(visual, "cast_shadows", "false")
    set_xml_child_text(visual, "pose", "0 0 0 0 0 0")
    geometry = ET.SubElement(visual, "geometry")
    box = ET.SubElement(geometry, "box")
    set_xml_child_text(box, "size", "0.18 0.14 0.10")
    material = ET.SubElement(visual, "material")
    set_xml_child_text(material, "ambient", "0.20 0.42 0.82 1")
    set_xml_child_text(material, "diffuse", "0.20 0.42 0.82 1")
    set_xml_child_text(material, "specular", "0 0 0 1")


def append_namespaced_turtlebot3_burger(
    parts: List[str],
    model_file: Path,
    name: str,
    x: float,
    y: float,
    yaw: float = 0.0,
    low_graphics: bool = True,
) -> None:
    tree = ET.parse(model_file)
    root = tree.getroot()
    model = root.find("model")
    if model is None:
        raise RuntimeError(f"Could not find <model> in {model_file}")

    model.set("name", name)
    pose = model.find("pose")
    if pose is None:
        pose = ET.Element("pose")
        model.insert(0, pose)
    pose.text = f"{x} {y} 0.01 0 0 {yaw}"

    diff_drive_plugins: List[ET.Element] = []
    for plugin in model.findall(".//plugin"):
        plugin_name = plugin.get("name", "").lower()
        plugin_filename = plugin.get("filename", "").lower()
        if any(
            token in plugin_name or token in plugin_filename
            for token in ("diffdrive", "diff_drive", "diff-drive")
        ):
            diff_drive_plugins.append(plugin)

    if not diff_drive_plugins:
        raise RuntimeError(f"Could not find TurtleBot3 DiffDrive plugin in {model_file}")

    for plugin in diff_drive_plugins:
        set_xml_child_text(plugin, "topic", f"/model/{name}/cmd_vel")
        set_xml_child_text(plugin, "odom_topic", f"/model/{name}/odometry")
        set_xml_child_text(plugin, "tf_topic", f"/model/{name}/tf")

    if low_graphics:
        simplify_turtlebot_visuals(model)

    ET.indent(model, space="  ", level=2)
    parts.extend(ET.tostring(model, encoding="unicode").splitlines())


def get_turtlebot3_model_paths() -> Tuple[Path, Path]:
    ros_distro = os.environ.get("ROS_DISTRO", "jazzy")
    models_dir = Path(f"/opt/ros/{ros_distro}/share/turtlebot3_gazebo/models")
    model_file = models_dir / "turtlebot3_burger" / "model.sdf"
    if not model_file.exists():
        raise RuntimeError(
            "Could not find TurtleBot3 Burger model.sdf.\n"
            f"Expected: {model_file}\n\n"
            "Install it with:\n"
            f"  sudo apt install ros-{ros_distro}-turtlebot3-gazebo "
            f"ros-{ros_distro}-turtlebot3-description"
        )
    return models_dir, model_file


def generate_sdf_world(
    map_name: str,
    map_data: Dict[str, Any],
    scenario: Dict[str, Any],
    turtlebot3_model_file: Path,
    output_path: Path,
    low_graphics: bool = True,
) -> None:
    visualization = scenario.get("visualization", {})
    cell_size = float(visualization.get("cell_size_m", 0.5))
    wall_height = float(visualization.get("wall_height_m", 0.6))
    wall_z = float(visualization.get("wall_z_m", wall_height / 2.0))

    width = int(map_data["width"])
    height = int(map_data["height"])
    grid = map_data["grid"]
    world_width = width * cell_size
    world_height = height * cell_size
    wall_segments = extract_horizontal_wall_segments(grid)

    parts: List[str] = [
        '<?xml version="1.0" ?>',
        '<sdf version="1.9">',
        f'  <world name="{make_world_name(map_name)}">',
        '    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>',
        '    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>',
        '    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>',
        "    <scene>",
        "      <ambient>0.65 0.65 0.65 1</ambient>",
        "      <background>0.16 0.17 0.19 1</background>",
        # Shadows stay off even with full TurtleBot meshes; they are expensive in VMs
        # and do not affect LiDAR, collisions, or navigation.
        "      <shadows>false</shadows>",
        "      <grid>false</grid>",
        "    </scene>",
        '    <light type="directional" name="sun">',
        "      <cast_shadows>false</cast_shadows>",
        f"      <pose>{world_width / 2.0} {world_height / 2.0} 10 0 0 0</pose>",
        "      <diffuse>0.72 0.72 0.72 1</diffuse>",
        "      <specular>0 0 0 1</specular>",
        "      <direction>-0.5 0.1 -0.9</direction>",
        "    </light>",
    ]

    append_box_model(
        parts,
        "ground_plane",
        [world_width / 2.0, world_height / 2.0, -0.025, 0, 0, 0],
        [world_width, world_height, 0.05],
        "0.82 0.82 0.82 1",
        "0.82 0.82 0.82 1",
    )

    for wall_index, segment in enumerate(wall_segments):
        center_col = segment["start_col"] + segment["length_cells"] / 2.0 - 0.5
        x = (center_col + 0.5) * cell_size
        y = (segment["row"] + 0.5) * cell_size
        append_box_model(
            parts,
            f"wall_segment_{wall_index}",
            [x, y, wall_z, 0, 0, 0],
            [segment["length_cells"] * cell_size, cell_size, wall_height],
            "0.12 0.12 0.12 1",
            "0.12 0.12 0.12 1",
        )

    for robot in enabled_robots(scenario):
        robot_id = str(robot["id"])
        row, col = (int(value) for value in robot["start_cell"])
        x, y = cell_to_world(row, col, cell_size)
        yaw = float(robot.get("start_yaw", robot.get("yaw", 0.0)))
        append_namespaced_turtlebot3_burger(
            parts,
            turtlebot3_model_file,
            robot_id,
            x,
            y,
            yaw,
            low_graphics=low_graphics,
        )

    parts.extend(["  </world>", "</sdf>"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Attack heatmap preparation
# ---------------------------------------------------------------------------


def find_heatmap_generator(package_share: Path, explicit_path: str) -> Path:
    candidates: List[Path] = []
    if explicit_path.strip():
        candidates.append(resolve_path(package_share, explicit_path))

    candidates.extend(
        [
            package_share / "scripts" / "generate_attack_heatmap.py",
            package_share / "lib" / PACKAGE_NAME / "generate_attack_heatmap.py",
            Path(get_package_prefix(PACKAGE_NAME))
            / "lib"
            / PACKAGE_NAME
            / "generate_attack_heatmap",
            Path.home()
            / "ros_ws"
            / "src"
            / PACKAGE_NAME
            / "scripts"
            / "generate_attack_heatmap.py",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise RuntimeError(
        "Attack heatmap generator was not found. Checked:\n" + formatted
    )


def default_heatmap_path(output_dir: Path, map_name: str) -> Path:
    return output_dir / f"{map_name}_latest.json"


def run_heatmap_generator(
    script: Path,
    map_path: Path,
    scenario_path: Path,
    output_dir: Path,
    action_goal_count: str,
    action_goal_seed: str,
    allow_diagonal: str,
    route_sample_count: str,
    reconnaissance_seed: str,
    detour_candidate_count: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    launcher = [str(script)] if os.access(script, os.X_OK) and script.suffix != ".py" else [sys.executable, str(script)]
    command = [
        *launcher,
        "--map-path",
        str(map_path),
        "--scenario-path",
        str(scenario_path),
        "--action-goal-count",
        action_goal_count,
        "--action-goal-seed",
        action_goal_seed,
        "--allow-diagonal",
        allow_diagonal,
        "--output-dir",
        str(output_dir),
        "--route-sample-count",
        route_sample_count,
        "--reconnaissance-seed",
        reconnaissance_seed,
        "--detour-candidate-count",
        detour_candidate_count,
    ]
    print("[trust_costmap] Generating attack reconnaissance heatmap:")
    print("  " + " ".join(command))
    subprocess.run(command, check=True)


def validate_heatmap(
    heatmap_path: Path,
    map_name: str,
    map_data: Dict[str, Any],
) -> Dict[str, Any]:
    if not heatmap_path.exists():
        raise RuntimeError(f"Attack heatmap not found: {heatmap_path}")

    payload = json.loads(heatmap_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Heatmap must contain a JSON object: {heatmap_path}")

    payload_map = str(payload.get("map_name", payload.get("map", ""))).strip()
    if payload_map and payload_map not in {map_name, Path(map_name).stem}:
        raise RuntimeError(
            f"Heatmap map mismatch: expected {map_name}, found {payload_map}"
        )

    payload_height = payload.get("height")
    payload_width = payload.get("width")
    if payload_height is not None and int(payload_height) != int(map_data["height"]):
        raise RuntimeError("Heatmap height does not match the selected map.")
    if payload_width is not None and int(payload_width) != int(map_data["width"]):
        raise RuntimeError("Heatmap width does not match the selected map.")

    cells = payload.get("cells", payload.get("ranked_cells", []))
    if not isinstance(cells, list) or not cells:
        raise RuntimeError(f"Heatmap contains no ranked cells: {heatmap_path}")

    usable = 0
    for item in cells:
        if not isinstance(item, dict):
            continue
        try:
            cell = (int(item["row"]), int(item["col"]))
            score = float(item.get("score", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if score > 0.0 and is_free_cell(map_data, cell):
            usable += 1

    if usable == 0:
        raise RuntimeError(f"Heatmap has no positive-score free cells: {heatmap_path}")

    return {"usable_ranked_cells": usable, "total_ranked_cells": len(cells)}


# ---------------------------------------------------------------------------
# ROS actions
# ---------------------------------------------------------------------------


def make_bridge_actions(
    map_name: str,
    scenario: Dict[str, Any],
    enabled: bool,
) -> List[Node]:
    if not enabled:
        return []

    world_name = make_world_name(map_name)
    actions: List[Node] = []
    for robot in enabled_robots(scenario):
        robot_id = str(robot["id"])
        scan_topic = (
            f"/world/{world_name}/model/{robot_id}/link/base_scan/"
            "sensor/hls_lfcd_lds/scan"
        )
        actions.append(
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name=f"{sanitize_identifier(robot_id)}_bridge",
                output="screen",
                arguments=[
                    f"/model/{robot_id}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                    f"/model/{robot_id}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                    f"{scan_topic}@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
                ],
                remappings=[
                    (f"/model/{robot_id}/cmd_vel", f"/{robot_id}/cmd_vel"),
                    (f"/model/{robot_id}/odometry", f"/{robot_id}/odom"),
                    (scan_topic, f"/{robot_id}/scan"),
                ],
            )
        )
    return actions


def select_rviz_robot_id(scenario: Dict[str, Any], requested: str) -> str:
    """Resolve ``auto`` to the first enabled benign robot."""

    robots = enabled_robots(scenario)
    requested_clean = str(requested).strip()
    valid_ids = [str(robot["id"]) for robot in robots]

    if requested_clean and requested_clean.lower() not in {"auto", "default"}:
        if requested_clean not in valid_ids:
            raise RuntimeError(
                f"rviz_robot_id={requested_clean!r} is not enabled. "
                f"Choose one of: {', '.join(valid_ids)}"
            )
        return requested_clean

    for robot in robots:
        robot_id = str(robot["id"])
        role = str(robot.get("role", ""))
        if "benign" in robot_id.lower() or "benign" in role.lower():
            return robot_id
    return valid_ids[0]


def make_peer_scan_display_yaml(peer_robot_ids: Sequence[str]) -> str:
    """Build extra LaserScan display blocks for peer robots (reference-style)."""

    # Distinct colors so overlapping shared LiDAR rays stay readable in one window.
    palette = (
        (0, 170, 255),
        (120, 255, 80),
        (255, 120, 255),
        (255, 200, 40),
        (80, 220, 200),
    )
    blocks: List[str] = []
    for index, peer_id in enumerate(peer_robot_ids):
        red, green, blue = palette[index % len(palette)]
        blocks.append(
            "\n".join(
                [
                    "        - Alpha: 1",
                    "          Autocompute Intensity Bounds: true",
                    "          Autocompute Value Bounds:",
                    "            Max Value: 10",
                    "            Min Value: -10",
                    "            Value: true",
                    "          Axis: Z",
                    "          Channel Name: intensity",
                    "          Class: rviz_default_plugins/LaserScan",
                    f"          Color: {red}; {green}; {blue}",
                    "          Color Transformer: FlatColor",
                    "          Decay Time: 0.25",
                    "          Enabled: true",
                    "          Invert Rainbow: false",
                    "          Max Color: 255; 255; 255",
                    "          Max Intensity: 4096",
                    "          Min Color: 0; 0; 0",
                    "          Min Intensity: 0",
                    f"          Name: Peer Scan {peer_id}",
                    "          Position Transformer: XYZ",
                    "          Selectable: true",
                    "          Size (Pixels): 4",
                    "          Size (m): 0.035",
                    "          Style: FlatSquares",
                    "          Topic:",
                    "            Depth: 5",
                    "            Durability Policy: Volatile",
                    "            History Policy: Keep Last",
                    "            Reliability Policy: Reliable",
                    f"            Value: /{peer_id}/scan_rviz",
                    "          Use Fixed Frame: true",
                    "          Use rainbow: false",
                    "          Value: true",
                ]
            )
        )
    if not blocks:
        return ""
    return "\n" + "\n".join(blocks) + "\n"


def render_rviz_config(
    package_share: Path,
    output_path: Path,
    robot_id: str,
    peer_robot_ids: Optional[Sequence[str]] = None,
) -> None:
    template_path = package_share / "config" / "lidar_mapping.rviz.in"
    if not template_path.exists():
        raise RuntimeError(f"RViz template not found: {template_path}")
    peers = [
        str(peer)
        for peer in (peer_robot_ids or [])
        if str(peer) and str(peer) != robot_id
    ]
    rendered = (
        template_path.read_text(encoding="utf-8")
        .replace("__ROBOT_ID__", robot_id)
        .replace("__PEER_SCAN_DISPLAYS__", make_peer_scan_display_yaml(peers))
    )
    if "__ROBOT_ID__" in rendered or "__PEER_SCAN_DISPLAYS__" in rendered:
        raise RuntimeError("RViz template still contains unresolved placeholders.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


def make_lidar_mapping_actions(
    scenario: Dict[str, Any],
    map_data: Dict[str, Any],
    publish_rate_hz: float,
    scan_stride: int,
    rviz_robot_id: str,
    rviz_outputs_enabled: bool,
) -> List[Node]:
    """Launch one isolated mapper for every enabled scenario robot."""

    robots = enabled_robots(scenario)
    robot_ids = [str(robot["id"]) for robot in robots]
    visualization = scenario.get("visualization", {})
    resolution = float(visualization.get("cell_size_m", 0.5))

    actions: List[Node] = []
    for robot in robots:
        robot_id = str(robot["id"])
        row, col = (int(value) for value in robot["start_cell"])
        spawn_x, spawn_y = cell_to_world(row, col, resolution)
        spawn_yaw = float(robot.get("start_yaw", robot.get("yaw", 0.0)))
        actions.append(
            Node(
                package=PACKAGE_NAME,
                executable="lidar_mapper",
                namespace=robot_id,
                name="lidar_mapper",
                output="screen",
                parameters=[
                    {
                        "robot_id": robot_id,
                        # An explicit empty-string sentinel keeps the ROS parameter
                        # type stable when a scenario contains only one enabled robot.
                        "peer_robot_ids": (
                            [peer_id for peer_id in robot_ids if peer_id != robot_id]
                            or [""]
                        ),
                        "map_width": int(map_data["width"]),
                        "map_height": int(map_data["height"]),
                        "resolution": resolution,
                        "spawn_x": spawn_x,
                        "spawn_y": spawn_y,
                        "spawn_yaw": spawn_yaw,
                        "scan_topic": f"/{robot_id}/scan",
                        "odom_topic": f"/{robot_id}/odom",
                        "map_pose_topic": f"/{robot_id}/map_pose",
                        "publish_rate_hz": float(publish_rate_hz),
                        "scan_stride": int(scan_stride),
                        "enable_rviz_outputs": bool(
                            rviz_outputs_enabled and robot_id == rviz_robot_id
                        ),
                    }
                ],
            )
        )
    return actions


def rosbag_topics(scenario: Dict[str, Any]) -> List[str]:
    topics = [
        "/base_map",
        "/planning_costmap",
        "/planned_path",
        "/plan_status",
        "/map_claims",
        "/trust_updates",
        "/action_goal_updates",
        "/robot_markers",
        "/start_goal_markers",
        "/action_goal_markers",
        "/agent_route_markers",
        "/clock",
    ]
    for robot in enabled_robots(scenario):
        robot_id = str(robot["id"])
        topics.extend(
            [
                f"/{robot_id}/odom",
                f"/{robot_id}/cmd_vel",
                f"/{robot_id}/scan",
                f"/{robot_id}/scan_rviz",
                f"/{robot_id}/map_pose",
                f"/{robot_id}/current_observation_map",
                f"/{robot_id}/local_map",
                f"/{robot_id}/shared_map",
                f"/{robot_id}/combined_map",
                f"/{robot_id}/combined_probability_markers",
                f"/{robot_id}/local_history_markers",
                f"/{robot_id}/shared_history_markers",
                f"/{robot_id}/current_observation_markers",
                f"/{robot_id}/footprint_markers",
                f"/{robot_id}/lidar_free_cells",
                f"/{robot_id}/lidar_occupied_cells",
                f"/agent_routes/{robot_id}",
            ]
        )
    topics.extend(["/rviz_base_free", "/rviz_base_occupied"])
    return list(dict.fromkeys(topics))


def build_gazebo_command(world_path: Path, headless: bool, verbosity: str) -> List[str]:
    command = ["gz", "sim", "-r", "-v", verbosity]
    if headless:
        command.append("-s")
    else:
        command.extend(["--render-engine", "ogre"])
    command.append(str(world_path))
    return command


# ---------------------------------------------------------------------------
# Main opaque setup
# ---------------------------------------------------------------------------


def prepare_experiment(context: Any) -> List[Any]:
    package_share = Path(get_package_share_directory(PACKAGE_NAME))

    def value(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    map_name = value("map_name")
    scenario_file = value("scenario_file")
    planner = value("planner")
    costmap_override = value("costmap_method")
    action_goal_count = value("action_goal_count")
    action_goal_seed = value("action_goal_seed")
    experiment_seed_override = value("experiment_seed")
    allow_diagonal = value("allow_diagonal")
    replan_period_sec = value("replan_period_sec")
    duration_override = value("experiment_duration_sec")
    gazebo_low_graphics = parse_bool(value("gazebo_low_graphics"))

    map_path = package_share / "worlds" / "movingai_mapf" / f"{map_name}.map"
    scenario_path = resolve_path(package_share, scenario_file)
    if not map_path.exists():
        raise RuntimeError(f"Map file not found: {map_path}")
    if not scenario_path.exists():
        raise RuntimeError(f"Scenario file not found: {scenario_path}")

    scenario = load_scenario(scenario_path)
    map_data = load_movingai_map(map_path)
    validate_robot_configuration(scenario, map_data)
    validate_attack_configuration(scenario)

    method = effective_costmap_method(scenario, costmap_override)
    experiment_seed = effective_experiment_seed(scenario, experiment_seed_override)
    duration_sec = effective_duration_sec(scenario, duration_override)
    scenario_name = sanitize_identifier(
        scenario.get("scenario_name", scenario_path.stem)
    )
    run_label = sanitize_identifier(value("run_label") or "run")
    trial_index = int(value("trial_index"))

    requested_run_id = value("run_id").strip()
    if requested_run_id:
        run_id = sanitize_identifier(requested_run_id)
    else:
        run_id = sanitize_identifier(
            f"{scenario_name}__{method}__seed-{experiment_seed}__trial-{trial_index}__"
            f"{run_label}__{utc_timestamp()}__{uuid.uuid4().hex[:8]}"
        )

    runtime_root = Path(os.path.expanduser(value("runtime_root"))).resolve()
    run_dir = runtime_root / "runs" / run_id
    world_dir = run_dir / "world"
    metadata_dir = run_dir / "metadata"
    logs_dir = run_dir / "logs"
    bags_dir = run_dir / "bags"
    for directory in (world_dir, metadata_dir, logs_dir, bags_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Preserve exact inputs for every baseline/attack trial.
    scenario_snapshot = metadata_dir / "scenario.resolved.yaml"
    map_snapshot = metadata_dir / map_path.name
    shutil.copy2(scenario_path, scenario_snapshot)
    shutil.copy2(map_path, map_snapshot)

    world_path = world_dir / f"generated_{sanitize_identifier(map_name)}.sdf"
    models_dir, model_file = get_turtlebot3_model_paths()
    generate_sdf_world(
        map_name,
        map_data,
        scenario,
        model_file,
        world_path,
        low_graphics=gazebo_low_graphics,
    )

    attack_is_enabled = attack_enabled(scenario)
    requested_generation = parse_auto_bool(
        value("generate_attack_heatmap"),
        automatic_value=attack_is_enabled
        or bool(scenario.get("attack_reconnaissance", {}).get("enabled", False)),
    )
    require_heatmap = parse_auto_bool(
        value("require_attack_heatmap"), automatic_value=attack_is_enabled
    )

    heatmap_output_dir = Path(
        os.path.expanduser(value("attack_heatmap_output_dir"))
    ).resolve()

    explicit_heatmap_path = value("attack_heatmap_path").strip()

    heatmap_path = (
        Path(os.path.expanduser(explicit_heatmap_path)).resolve()
        if explicit_heatmap_path
        else default_heatmap_path(heatmap_output_dir, map_name)
    )

    heatmap_validation: Dict[str, Any] = {}

    if require_heatmap and not requested_generation and not explicit_heatmap_path:
        raise RuntimeError(
            "An attack heatmap is required, but generation is disabled and "
            "attack_heatmap_path was not provided."
        )

    if requested_generation:
        existing_candidates = {
            path.resolve(): path.stat().st_mtime_ns
            for path in heatmap_output_dir.glob(f"{map_name}_*.json")
        }

        generator = find_heatmap_generator(
            package_share,
            value("attack_heatmap_generator"),
        )

        run_heatmap_generator(
            generator,
            map_path,
            scenario_path,
            heatmap_output_dir,
            action_goal_count,
            action_goal_seed,
            allow_diagonal,
            value("heatmap_route_sample_count"),
            value("heatmap_reconnaissance_seed"),
            value("heatmap_detour_candidate_count"),
        )

        if not explicit_heatmap_path:
            generated_candidates: List[Path] = []

            for path in heatmap_output_dir.glob(f"{map_name}_*.json"):
                resolved = path.resolve()
                previous_mtime = existing_candidates.get(resolved)
                current_mtime = path.stat().st_mtime_ns

                if previous_mtime is None or current_mtime > previous_mtime:
                    generated_candidates.append(path)

            generated_candidates.sort(
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )

            if not generated_candidates:
                raise RuntimeError(
                    "Heatmap generator completed but did not create or update "
                    f"a JSON artifact for map {map_name!r}."
                )

            heatmap_path = generated_candidates[0]

    if require_heatmap or heatmap_path.exists():
        heatmap_validation = validate_heatmap(
            heatmap_path,
            map_name,
            map_data,
        )
        shutil.copy2(
            heatmap_path,
            metadata_dir / "attack_heatmap.json",
        )

    bridge_enabled = parse_bool(value("bridge_robot_topics"))
    headless = parse_bool(value("headless"))
    record_bag = parse_bool(value("record_rosbag"))
    lidar_mapping_enabled = parse_bool(value("enable_lidar_mapping"))
    rviz_enabled = parse_auto_bool(
        value("enable_rviz"),
        automatic_value=lidar_mapping_enabled and not headless,
    )
    if headless and rviz_enabled:
        raise RuntimeError(
            "enable_rviz=true cannot be used with headless=true. "
            "Use enable_rviz=false or enable_rviz=auto."
        )
    rviz_robot_id = select_rviz_robot_id(scenario, value("rviz_robot_id"))
    rviz_config_path = metadata_dir / f"rviz_{sanitize_identifier(rviz_robot_id)}.rviz"
    if rviz_enabled:
        peer_ids = [
            str(robot["id"])
            for robot in enabled_robots(scenario)
            if str(robot["id"]) != rviz_robot_id
        ]
        render_rviz_config(
            package_share,
            rviz_config_path,
            rviz_robot_id,
            peer_robot_ids=peer_ids,
        )

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_label": run_label,
        "trial_index": trial_index,
        "scenario_name": scenario_name,
        "scenario_file": str(scenario_path),
        "scenario_sha256": sha256_file(scenario_path),
        "scenario_snapshot": str(scenario_snapshot),
        "map_name": map_name,
        "map_file": str(map_path),
        "map_sha256": sha256_file(map_path),
        "map_snapshot": str(map_snapshot),
        "map_dimensions": {
            "height": int(map_data["height"]),
            "width": int(map_data["width"]),
        },
        "world_file": str(world_path),
        "planner": planner,
        "costmap_method": method,
        "experiment_seed": experiment_seed,
        "action_goal_count_override": int(action_goal_count),
        "action_goal_seed_override": int(action_goal_seed),
        "allow_diagonal_override": parse_bool(allow_diagonal),
        "replan_period_sec_override": float(replan_period_sec),
        "duration_sec": duration_sec,
        "attack": {
            "enabled": attack_is_enabled,
            "malicious_robot_ids": malicious_robot_ids(scenario),
            "placement_policy": malicious_attack_config(scenario).get(
                "placement_policy", "topology_only"
            ),
            "heatmap_generation_requested": requested_generation,
            "heatmap_required": require_heatmap,
            "heatmap_path": str(heatmap_path),
            "heatmap_validation": heatmap_validation,
        },
        "recording": {
            "rosbag_enabled": record_bag,
            "rosbag_directory": str(bags_dir / "experiment"),
            "topics": rosbag_topics(scenario) if record_bag else [],
        },
        "runtime": {
            "run_directory": str(run_dir),
            "metadata_directory": str(metadata_dir),
            "logs_directory": str(logs_dir),
            "headless": headless,
            "bridges_enabled": bridge_enabled,
            "gazebo_low_graphics": gazebo_low_graphics,
            "lidar_mapping_enabled": lidar_mapping_enabled,
            "rviz_enabled": rviz_enabled,
            "rviz_robot_id": rviz_robot_id,
            "rviz_config": str(rviz_config_path) if rviz_enabled else "",
            "mapping_publish_rate_hz": float(value("mapping_publish_rate_hz")),
            "mapping_scan_stride": int(value("mapping_scan_stride")),
        },
        "future_metrics_contract": {
            "expected_summary_file": str(run_dir / "metrics" / "summary.json"),
            "expected_plan_log": str(run_dir / "metrics" / "plans.csv"),
            "expected_claim_log": str(run_dir / "metrics" / "claims.csv"),
            "expected_trust_log": str(run_dir / "metrics" / "trust.csv"),
            "expected_attack_log": str(run_dir / "metrics" / "attacks.csv"),
            "expected_trajectory_log": str(run_dir / "metrics" / "trajectories.csv"),
        },
    }
    manifest_path = metadata_dir / "launch_manifest.json"
    write_json_atomic(manifest_path, manifest)

    existing_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    gz_resource_path = (
        existing_resource_path + os.pathsep + str(models_dir)
        if existing_resource_path
        else str(models_dir)
    )

    environment_actions = [
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", gz_resource_path),
        SetEnvironmentVariable("TRUST_COSTMAP_RUN_ID", run_id),
        SetEnvironmentVariable("TRUST_COSTMAP_RUN_DIR", str(run_dir)),
        SetEnvironmentVariable("TRUST_COSTMAP_RESULTS_DIR", str(run_dir / "metrics")),
        SetEnvironmentVariable("TRUST_COSTMAP_MANIFEST", str(manifest_path)),
        SetEnvironmentVariable("TRUST_COSTMAP_SCENARIO_NAME", scenario_name),
        SetEnvironmentVariable("TRUST_COSTMAP_COSTMAP_METHOD", method),
        SetEnvironmentVariable("TRUST_COSTMAP_EXPERIMENT_SEED", str(experiment_seed)),
        SetEnvironmentVariable("TRUST_COSTMAP_TRIAL_INDEX", str(trial_index)),
    ]

    gazebo = ExecuteProcess(
        cmd=build_gazebo_command(world_path, headless, value("gz_verbosity")),
        output="screen",
        name="gazebo_sim",
    )

    manager = Node(
        package=PACKAGE_NAME,
        executable="experiment_manager",
        name="experiment_manager",
        output="screen",
        parameters=[
            {
                "map_name": map_name,
                "scenario_file": str(scenario_path),
                "planner": planner,
                "costmap_method": method,
                "replan_period_sec": float(replan_period_sec),
                "allow_diagonal": parse_bool(allow_diagonal),
                "action_goal_count": int(action_goal_count),
                "action_goal_seed": int(action_goal_seed),
                "heatmap_reconnaissance_seed": int(
                    value("heatmap_reconnaissance_seed")
                ),
                "attack_heatmap_path": (
                    str(heatmap_path) if heatmap_path.exists() else ""
                ),
                "require_attack_heatmap": require_heatmap,
                "enable_external_costmaps": parse_bool(
                    value("enable_external_costmaps")
                ),
                "external_costmap_dir": value("external_costmap_dir"),
            }
        ],
    )

    actions: List[Any] = [
        *environment_actions,
        LogInfo(msg=f"[trust_costmap] run_id={run_id}"),
        LogInfo(msg=f"[trust_costmap] run_dir={run_dir}"),
        LogInfo(msg=f"[trust_costmap] scenario={scenario_name}"),
        LogInfo(msg=f"[trust_costmap] method={method}, seed={experiment_seed}, trial={trial_index}"),
        LogInfo(msg=f"[trust_costmap] attack_enabled={attack_is_enabled}"),
        LogInfo(msg=f"[trust_costmap] manifest={manifest_path}"),
        LogInfo(msg=f"[trust_costmap] gazebo_low_graphics={gazebo_low_graphics}"),
        LogInfo(msg=f"[trust_costmap] lidar_mapping_enabled={lidar_mapping_enabled}"),
        LogInfo(msg=f"[trust_costmap] rviz_enabled={rviz_enabled}, rviz_robot_id={rviz_robot_id}"),
        gazebo,
    ]

    bridges = make_bridge_actions(map_name, scenario, bridge_enabled)
    if bridges:
        actions.append(
            TimerAction(period=float(value("bridge_start_delay_sec")), actions=bridges)
        )

    actions.append(
        TimerAction(
            period=float(value("manager_start_delay_sec")), actions=[manager]
        )
    )

    if lidar_mapping_enabled:
        mapping_actions = make_lidar_mapping_actions(
            scenario,
            map_data,
            publish_rate_hz=float(value("mapping_publish_rate_hz")),
            scan_stride=int(value("mapping_scan_stride")),
            rviz_robot_id=rviz_robot_id,
            rviz_outputs_enabled=rviz_enabled,
        )
        actions.append(
            TimerAction(
                period=float(value("mapping_start_delay_sec")),
                actions=mapping_actions,
            )
        )
        if not bridge_enabled:
            actions.append(
                LogInfo(
                    msg=(
                        "[trust_costmap] WARNING: LiDAR mapping is enabled while "
                        "bridge_robot_topics=false; external scan/odom publishers "
                        "must provide the robot topics."
                    )
                )
            )

    if rviz_enabled:
        rviz = Node(
            package="rviz2",
            executable="rviz2",
            name="trust_costmap_rviz",
            output="screen",
            arguments=["-d", str(rviz_config_path)],
        )
        actions.append(
            TimerAction(
                period=float(value("rviz_start_delay_sec")),
                actions=[rviz],
            )
        )

    if record_bag:
        bag_command = [
            "ros2",
            "bag",
            "record",
            "--output",
            str(bags_dir / "experiment"),
            "--storage",
            value("rosbag_storage"),
            *rosbag_topics(scenario),
        ]
        actions.append(
            TimerAction(
                period=float(value("rosbag_start_delay_sec")),
                actions=[
                    ExecuteProcess(
                        cmd=bag_command,
                        output="screen",
                        name="experiment_rosbag",
                    )
                ],
            )
        )

    if duration_sec > 0.0:
        shutdown_delay = duration_sec + float(value("shutdown_grace_sec"))
        actions.append(
            TimerAction(
                period=shutdown_delay,
                actions=[
                    LogInfo(
                        msg=(
                            "[trust_costmap] Experiment duration elapsed; "
                            "shutting down launch."
                        )
                    ),
                    EmitEvent(
                        event=Shutdown(
                            reason=f"Experiment duration {duration_sec:.3f}s elapsed"
                        )
                    ),
                ],
            )
        )

    if parse_bool(value("shutdown_on_gazebo_exit")):
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=gazebo,
                    on_exit=[
                        EmitEvent(event=Shutdown(reason="Gazebo exited"))
                    ],
                )
            )
        )

    if parse_bool(value("shutdown_on_manager_exit")):
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=manager,
                    on_exit=[
                        EmitEvent(event=Shutdown(reason="Experiment manager exited"))
                    ],
                )
            )
        )

    return actions


# ---------------------------------------------------------------------------
# Launch arguments
# ---------------------------------------------------------------------------


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument(
            "map_name",
            default_value="room-32-32-4",
            description="MovingAI map name without the .map extension.",
        ),
        DeclareLaunchArgument(
            "scenario_file",
            default_value="scenario.yaml",
            description="Scenario YAML path, absolute or relative to package share.",
        ),
        DeclareLaunchArgument(
            "planner",
            default_value="astar",
            description="Planner registered by the experiment manager.",
        ),
        DeclareLaunchArgument(
            "costmap_method",
            default_value="",
            description="Method override; empty uses experiment.method from YAML.",
        ),
        DeclareLaunchArgument(
            "experiment_seed",
            default_value="-1",
            description="Sweep-level experiment seed; -1 uses scenario random_seed.",
        ),
        DeclareLaunchArgument(
            "trial_index",
            default_value="0",
            description="Trial number recorded in the run manifest.",
        ),
        DeclareLaunchArgument(
            "run_label",
            default_value="manual",
            description="Human-readable label used in the generated run id.",
        ),
        DeclareLaunchArgument(
            "run_id",
            default_value="",
            description="Explicit run id; empty generates a unique reproducible label.",
        ),
        DeclareLaunchArgument(
            "runtime_root",
            default_value=str(DEFAULT_RUNTIME_ROOT),
            description="Writable root for generated worlds, runs, bags, and manifests.",
        ),
        DeclareLaunchArgument(
            "experiment_duration_sec",
            default_value="0.0",
            description="Duration override; <=0 uses experiment.duration_sec from YAML.",
        ),
        DeclareLaunchArgument(
            "shutdown_grace_sec",
            default_value="2.0",
            description="Extra time after nominal duration before launch shutdown.",
        ),
        DeclareLaunchArgument(
            "shutdown_on_gazebo_exit",
            default_value="true",
            description="Shut down all processes if Gazebo exits.",
        ),
        DeclareLaunchArgument(
            "shutdown_on_manager_exit",
            default_value="true",
            description="Shut down all processes if the manager exits.",
        ),
        DeclareLaunchArgument(
            "bridge_robot_topics",
            default_value="true",
            description="Start cmd_vel, odom, and scan bridges for each enabled robot.",
        ),
        DeclareLaunchArgument(
            "bridge_start_delay_sec",
            default_value="2.0",
            description="Delay before starting robot topic bridges.",
        ),
        DeclareLaunchArgument(
            "manager_start_delay_sec",
            default_value="3.0",
            description="Delay before starting the experiment manager.",
        ),
        DeclareLaunchArgument(
            "enable_lidar_mapping",
            default_value="true",
            description="Launch one visualization-only LiDAR mapper per enabled robot.",
        ),
        DeclareLaunchArgument(
            "mapping_start_delay_sec",
            default_value="3.25",
            description="Delay before starting per-robot LiDAR mapping nodes.",
        ),
        DeclareLaunchArgument(
            "mapping_publish_rate_hz",
            default_value="2.0",
            description="Persistent/shared occupancy publication rate per robot.",
        ),
        DeclareLaunchArgument(
            "mapping_scan_stride",
            default_value="1",
            description="Process every Nth LiDAR beam; increase to reduce CPU load.",
        ),
        DeclareLaunchArgument(
            "enable_rviz",
            default_value="auto",
            description="true, false, or auto; auto opens one window unless headless.",
        ),
        DeclareLaunchArgument(
            "rviz_robot_id",
            default_value="auto",
            description="Robot displayed by the single RViz window; auto selects a benign robot.",
        ),
        DeclareLaunchArgument(
            "rviz_start_delay_sec",
            default_value="4.0",
            description="Delay before opening the single RViz window.",
        ),
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Run Gazebo server-only for automated baseline sweeps.",
        ),
        DeclareLaunchArgument(
            "gz_verbosity",
            default_value="1",
            description="Gazebo verbosity passed to gz sim -v.",
        ),
        DeclareLaunchArgument(
            "gazebo_low_graphics",
            default_value="true",
            description="Disable shadows and PBR textures while retaining robots, routes, and goals.",
        ),
        DeclareLaunchArgument(
            "action_goal_count",
            default_value="-1",
            description="Action-goal count override; -1 uses scenario configuration.",
        ),
        DeclareLaunchArgument(
            "action_goal_seed",
            default_value="-1",
            description="Action-goal seed override; -1 uses scenario configuration.",
        ),
        DeclareLaunchArgument(
            "replan_period_sec",
            default_value="0.0",
            description="Replan period override; 0.0 lets the manager use scenario value.",
        ),
        DeclareLaunchArgument(
            "allow_diagonal",
            default_value="false",
            description="Allow diagonal planning motion.",
        ),
        DeclareLaunchArgument(
            "generate_attack_heatmap",
            default_value="auto",
            description="true, false, or auto based on attack/reconnaissance YAML.",
        ),
        DeclareLaunchArgument(
            "require_attack_heatmap",
            default_value="auto",
            description="true, false, or auto; auto requires it for enabled attacks.",
        ),
        DeclareLaunchArgument(
            "attack_heatmap_generator",
            default_value="",
            description="Optional explicit generator executable/script path.",
        ),
        DeclareLaunchArgument(
            "attack_heatmap_output_dir",
            default_value=str(DEFAULT_RUNTIME_ROOT / "attack_heatmaps"),
            description="Directory where reconnaissance artifacts are written.",
        ),
        DeclareLaunchArgument(
            "attack_heatmap_path",
            default_value="",
            description="Explicit JSON artifact; empty uses <map>_latest.json.",
        ),
        DeclareLaunchArgument(
            "heatmap_route_sample_count",
            default_value="-1",
            description="Reconnaissance route-pair sample count override.",
        ),
        DeclareLaunchArgument(
            "heatmap_reconnaissance_seed",
            default_value="-1",
            description="Reconnaissance sampling seed override.",
        ),
        DeclareLaunchArgument(
            "heatmap_detour_candidate_count",
            default_value="-1",
            description="Bounded block-and-replan candidate count override.",
        ),
        DeclareLaunchArgument(
            "enable_external_costmaps",
            default_value="true",
            description="Allow external baseline costmap plugins.",
        ),
        DeclareLaunchArgument(
            "external_costmap_dir",
            default_value="",
            description="Optional directory containing external baseline methods.",
        ),
        DeclareLaunchArgument(
            "record_rosbag",
            default_value="false",
            description="Record standard experiment topics for offline metrics.",
        ),
        DeclareLaunchArgument(
            "rosbag_start_delay_sec",
            default_value="2.5",
            description="Delay before rosbag recording starts.",
        ),
        DeclareLaunchArgument(
            "rosbag_storage",
            default_value="mcap",
            description="rosbag2 storage plugin, commonly mcap or sqlite3.",
        ),
    ]

    return LaunchDescription([*arguments, OpaqueFunction(function=prepare_experiment)])
