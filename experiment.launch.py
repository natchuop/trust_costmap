import os
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


FREE_SYMBOLS = {".", "G", "S"}


def load_movingai_map(map_path):
    with open(map_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    map_type = None
    height = None
    width = None
    map_start_index = None

    for i, line in enumerate(lines):
        clean = line.strip()

        if clean.startswith("type "):
            map_type = clean.split()[1]
        elif clean.startswith("height "):
            height = int(clean.split()[1])
        elif clean.startswith("width "):
            width = int(clean.split()[1])
        elif clean == "map":
            map_start_index = i + 1
            break

    if map_type is None or height is None or width is None or map_start_index is None:
        raise RuntimeError(f"Invalid MovingAI map file: {map_path}")

    grid = lines[map_start_index:map_start_index + height]

    if len(grid) != height:
        raise RuntimeError(f"Expected {height} rows, got {len(grid)}")

    for row in grid:
        if len(row) != width:
            raise RuntimeError(f"Expected width {width}, got {len(row)}")

    return {
        "map_type": map_type,
        "height": height,
        "width": width,
        "grid": grid,
    }


def is_blocked(ch):
    return ch not in FREE_SYMBOLS


def cell_to_world(row, col, cell_size):
    x = (col + 0.5) * cell_size
    y = (row + 0.5) * cell_size
    return x, y


def robot_material_for_role(role):
    if role == "malicious_reporter":
        return "0.8 0.05 0.05 1"
    if role == "benign_reporter":
        return "0.05 0.25 0.9 1"
    return "0.05 0.75 0.25 1"


def extract_horizontal_wall_segments(grid):
    """
    Converts blocked cells into horizontal wall runs.

    Example:
        @@@..@@
    becomes:
        segment length 3
        segment length 2

    This dramatically reduces Gazebo entity count compared with one box per cell.
    """
    segments = []

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
            length_cells = end_col - start_col + 1

            segments.append(
                {
                    "row": row_index,
                    "start_col": start_col,
                    "end_col": end_col,
                    "length_cells": length_cells,
                }
            )

    return segments


def append_box_model(
    parts,
    name,
    pose_xyz_rpy,
    size_xyz,
    ambient_rgba,
    diffuse_rgba,
    static=True,
):
    static_text = "true" if static else "false"
    pose = " ".join(str(value) for value in pose_xyz_rpy)
    size = " ".join(str(value) for value in size_xyz)

    parts.append(f'    <model name="{name}">')
    parts.append(f"      <static>{static_text}</static>")
    parts.append(f"      <pose>{pose}</pose>")
    parts.append('      <link name="link">')
    parts.append('        <collision name="collision">')
    parts.append('          <geometry>')
    parts.append('            <box>')
    parts.append(f"              <size>{size}</size>")
    parts.append('            </box>')
    parts.append('          </geometry>')
    parts.append('        </collision>')
    parts.append('        <visual name="visual">')
    parts.append('          <geometry>')
    parts.append('            <box>')
    parts.append(f"              <size>{size}</size>")
    parts.append('            </box>')
    parts.append('          </geometry>')
    parts.append('          <material>')
    parts.append(f"            <ambient>{ambient_rgba}</ambient>")
    parts.append(f"            <diffuse>{diffuse_rgba}</diffuse>")
    parts.append('          </material>')
    parts.append('        </visual>')
    parts.append('      </link>')
    parts.append('    </model>')


def generate_sdf_world(map_name, map_data, scenario, output_path):
    visualization = scenario.get("visualization", {})

    cell_size = float(visualization.get("cell_size_m", 0.5))
    wall_height = float(visualization.get("wall_height_m", 0.6))
    wall_z = float(visualization.get("wall_z_m", wall_height / 2.0))
    robot_height = float(visualization.get("robot_height_m", 0.25))
    robot_z = float(visualization.get("robot_z_m", robot_height / 2.0))

    width = int(map_data["width"])
    height = int(map_data["height"])
    grid = map_data["grid"]

    world_width = width * cell_size
    world_height = height * cell_size

    wall_segments = extract_horizontal_wall_segments(grid)

    parts = []

    parts.append('<?xml version="1.0" ?>')
    parts.append('<sdf version="1.9">')
    parts.append(f'  <world name="{map_name}_world">')

    parts.append(
        '    <plugin filename="gz-sim-scene-broadcaster-system" '
        'name="gz::sim::systems::SceneBroadcaster"/>'
    )

    parts.append('    <light type="directional" name="sun">')
    parts.append('      <cast_shadows>true</cast_shadows>')
    parts.append(f'      <pose>{world_width / 2.0} {world_height / 2.0} 10 0 0 0</pose>')
    parts.append('      <diffuse>0.8 0.8 0.8 1</diffuse>')
    parts.append('      <specular>0.2 0.2 0.2 1</specular>')
    parts.append('      <direction>-0.5 0.1 -0.9</direction>')
    parts.append('    </light>')

    append_box_model(
        parts=parts,
        name="ground_plane",
        pose_xyz_rpy=[world_width / 2.0, world_height / 2.0, -0.025, 0, 0, 0],
        size_xyz=[world_width, world_height, 0.05],
        ambient_rgba="0.82 0.82 0.82 1",
        diffuse_rgba="0.82 0.82 0.82 1",
        static=True,
    )

    for wall_index, segment in enumerate(wall_segments):
        row = segment["row"]
        start_col = segment["start_col"]
        length_cells = segment["length_cells"]

        center_col = start_col + (length_cells / 2.0) - 0.5

        x = (center_col + 0.5) * cell_size
        y = (row + 0.5) * cell_size

        wall_length_x = length_cells * cell_size
        wall_length_y = cell_size

        append_box_model(
            parts=parts,
            name=f"wall_segment_{wall_index}",
            pose_xyz_rpy=[x, y, wall_z, 0, 0, 0],
            size_xyz=[wall_length_x, wall_length_y, wall_height],
            ambient_rgba="0.12 0.12 0.12 1",
            diffuse_rgba="0.12 0.12 0.12 1",
            static=True,
        )

    for robot in scenario.get("robots", []):
        if not robot.get("enabled", True):
            continue

        robot_id = str(robot["id"])
        role = str(robot["role"])
        start_cell = robot["start_cell"]

        row = int(start_cell[0])
        col = int(start_cell[1])
        x, y = cell_to_world(row, col, cell_size)

        material = robot_material_for_role(role)

        append_box_model(
            parts=parts,
            name=robot_id,
            pose_xyz_rpy=[x, y, robot_z, 0, 0, 0],
            size_xyz=[cell_size * 0.7, cell_size * 0.7, robot_height],
            ambient_rgba=material,
            diffuse_rgba=material,
            static=True,
        )

    parts.append('  </world>')
    parts.append('</sdf>')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    print(
        f"[trust_costmap] Generated {output_path} "
        f"with {len(wall_segments)} merged wall segments "
        f"from {sum(1 for row in grid for ch in row if is_blocked(ch))} blocked cells."
    )


def prepare_generated_world(context):
    package_share = get_package_share_directory("trust_costmap")

    map_name = LaunchConfiguration("map_name").perform(context)
    scenario_file = LaunchConfiguration("scenario_file").perform(context)

    map_path = os.path.join(
        package_share,
        "worlds",
        "movingai_mapf",
        f"{map_name}.map",
    )

    scenario_path = os.path.join(package_share, scenario_file)

    generated_dir = os.path.join(package_share, "worlds")
    os.makedirs(generated_dir, exist_ok=True)

    generated_world_path = os.path.join(
        generated_dir,
        f"generated_{map_name}.sdf",
    )

    if not os.path.exists(map_path):
        raise RuntimeError(f"Map file not found: {map_path}")

    if not os.path.exists(scenario_path):
        raise RuntimeError(f"Scenario file not found: {scenario_path}")

    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)

    map_data = load_movingai_map(map_path)
    generate_sdf_world(map_name, map_data, scenario, generated_world_path)

    return [
        ExecuteProcess(
            cmd=["gz", "sim", "--render-engine", "ogre", "-r", generated_world_path],
            output="screen",
        )
    ]


def generate_launch_description():
    map_name_arg = DeclareLaunchArgument(
        "map_name",
        default_value="room-32-32-4",
        description="MovingAI map name without the .map extension",
    )

    scenario_file_arg = DeclareLaunchArgument(
        "scenario_file",
        default_value="scenario.yaml",
        description="Scenario YAML file in the trust_costmap package root",
    )

    return LaunchDescription(
        [
            map_name_arg,
            scenario_file_arg,

            OpaqueFunction(function=prepare_generated_world),

            Node(
                package="trust_costmap",
                executable="experiment_manager",
                name="experiment_manager",
                output="screen",
                parameters=[
                    {
                        "map_name": LaunchConfiguration("map_name"),
                        "scenario_file": LaunchConfiguration("scenario_file"),
                    }
                ],
            ),
        ]
    )
