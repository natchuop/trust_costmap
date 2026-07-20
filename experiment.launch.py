import os
from copy import deepcopy
from random import SystemRandom

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.actions import SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from trust_costmap.spawn_layout import FREE_SYMBOLS, build_spawn_layout


def load_movingai_map(map_path):
    with open(map_path, "r", encoding="utf-8") as file:
        lines = [line.rstrip("\n") for line in file]

    map_type = None
    height = None
    width = None
    map_start_index = None

    for index, line in enumerate(lines):
        clean = line.strip()
        if clean.startswith("type "):
            map_type = clean.split()[1]
        elif clean.startswith("height "):
            height = int(clean.split()[1])
        elif clean.startswith("width "):
            width = int(clean.split()[1])
        elif clean == "map":
            map_start_index = index + 1
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


def is_blocked(symbol):
    return symbol not in FREE_SYMBOLS


def cell_to_world(row, col, cell_size):
    return (col + 0.5) * cell_size, (row + 0.5) * cell_size


def extract_horizontal_wall_segments(grid):
    """Merge neighboring blocked cells into horizontal wall boxes."""
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

            segments.append(
                {
                    "row": row_index,
                    "start_col": start_col,
                    "length_cells": col - start_col,
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
    collision=True,
):
    static_text = "true" if static else "false"
    pose = " ".join(str(value) for value in pose_xyz_rpy)
    size = " ".join(str(value) for value in size_xyz)

    parts.append(f'    <model name="{name}">')
    parts.append(f"      <static>{static_text}</static>")
    parts.append(f"      <pose>{pose}</pose>")
    parts.append('      <link name="link">')

    if collision:
        parts.append('        <collision name="collision">')
        parts.append("          <geometry>")
        parts.append("            <box>")
        parts.append(f"              <size>{size}</size>")
        parts.append("            </box>")
        parts.append("          </geometry>")
        parts.append("        </collision>")

    parts.append('        <visual name="visual">')
    parts.append("          <geometry>")
    parts.append("            <box>")
    parts.append(f"              <size>{size}</size>")
    parts.append("            </box>")
    parts.append("          </geometry>")
    parts.append("          <material>")
    parts.append(f"            <ambient>{ambient_rgba}</ambient>")
    parts.append(f"            <diffuse>{diffuse_rgba}</diffuse>")
    parts.append("          </material>")
    parts.append("        </visual>")
    parts.append("      </link>")
    parts.append("    </model>")


def append_turtlebot3_burger(parts, name, x, y, yaw=0.0):
    parts.extend(
        [
            "    <include>",
            f"      <name>{name}</name>",
            f"      <pose>{x} {y} 0.01 0 0 {yaw}</pose>",
            "      <uri>model://turtlebot3_burger</uri>",
            "    </include>",
        ]
    )


def get_turtlebot3_model_paths():
    ros_distro = os.environ.get("ROS_DISTRO", "jazzy")

    try:
        turtlebot3_share = get_package_share_directory("turtlebot3_gazebo")
    except Exception as error:
        raise RuntimeError(
            "Could not locate the turtlebot3_gazebo package.\n"
            "Install the TurtleBot3 Gazebo packages with:\n"
            f"  sudo apt install ros-{ros_distro}-turtlebot3-gazebo "
            f"ros-{ros_distro}-turtlebot3-description\n"
        ) from error

    models_dir = os.path.join(turtlebot3_share, "models")
    model_file = os.path.join(models_dir, "turtlebot3_burger", "model.sdf")

    if not os.path.exists(model_file):
        raise RuntimeError(
            "Could not find TurtleBot3 Burger model.sdf.\n"
            f"Expected: {model_file}"
        )

    return models_dir, model_file


def resolve_scenario_layout(
    map_name,
    map_data,
    scenario,
    planner,
    allow_diagonal,
    action_goal_count,
    action_goal_seed,
):
    resolved = deepcopy(scenario)
    planning = resolved.setdefault("planning", {})

    if action_goal_count < 0:
        raise RuntimeError("action_goal_count cannot be negative.")

    planning["planner"] = planner
    planning["allow_diagonal"] = allow_diagonal
    planning["action_goal_count"] = action_goal_count
    planning["action_goal_seed"] = action_goal_seed

    enabled_robots = [
        robot for robot in resolved.get("robots", []) if robot.get("enabled", True)
    ]
    if not enabled_robots:
        raise RuntimeError("Scenario must contain at least one enabled robot.")

    robot_ids = [str(robot["id"]) for robot in enabled_robots]
    layout = build_spawn_layout(
        grid=map_data["grid"],
        robot_ids=robot_ids,
        action_goal_count=action_goal_count,
        action_goal_seed=action_goal_seed,
    )

    for robot in enabled_robots:
        robot_id = str(robot["id"])
        robot["start_cell"] = list(layout.robot_cells[robot_id])

    resolved["action_goals"] = [
        {
            "id": f"action_goal_{index + 1}",
            "cell": list(cell),
        }
        for index, cell in enumerate(layout.action_goal_cells)
    ]
    resolved["generated_layout"] = {
        "map_name": map_name,
        "action_goal_count": action_goal_count,
        "action_goal_seed": action_goal_seed,
        "robot_spawn_seed": action_goal_seed + 1,
        "connected_free_cell_count": layout.connected_free_cell_count,
    }

    return resolved


def generate_sdf_world(map_name, map_data, scenario, output_path):
    visualization = scenario.get("visualization", {})
    cell_size = float(visualization.get("cell_size_m", 0.5))
    wall_height = float(visualization.get("wall_height_m", 0.6))
    wall_z = float(visualization.get("wall_z_m", wall_height / 2.0))
    action_goal_size = float(
        visualization.get("action_goal_size_m", 0.18)
    )
    action_goal_height = float(
        visualization.get("action_goal_height_m", 0.04)
    )

    width = int(map_data["width"])
    height = int(map_data["height"])
    grid = map_data["grid"]
    world_width = width * cell_size
    world_height = height * cell_size
    wall_segments = extract_horizontal_wall_segments(grid)

    parts = [
        '<?xml version="1.0" ?>',
        '<sdf version="1.9">',
        f'  <world name="{map_name}_world">',
        '    <plugin filename="gz-sim-physics-system" '
        'name="gz::sim::systems::Physics"/>',
        '    <plugin filename="gz-sim-user-commands-system" '
        'name="gz::sim::systems::UserCommands"/>',
        '    <plugin filename="gz-sim-scene-broadcaster-system" '
        'name="gz::sim::systems::SceneBroadcaster"/>',
        '    <plugin filename="gz-sim-sensors-system" '
        'name="gz::sim::systems::Sensors">',
        '      <render_engine>ogre</render_engine>',
        '    </plugin>',
        '    <light type="directional" name="sun">',
        "      <cast_shadows>true</cast_shadows>",
        f"      <pose>{world_width / 2.0} {world_height / 2.0} 10 0 0 0</pose>",
        "      <diffuse>0.8 0.8 0.8 1</diffuse>",
        "      <specular>0.2 0.2 0.2 1</specular>",
        "      <direction>-0.5 0.1 -0.9</direction>",
        "    </light>",
    ]

    append_box_model(
        parts=parts,
        name="ground_plane",
        pose_xyz_rpy=[world_width / 2.0, world_height / 2.0, -0.025, 0, 0, 0],
        size_xyz=[world_width, world_height, 0.05],
        ambient_rgba="0.82 0.82 0.82 1",
        diffuse_rgba="0.82 0.82 0.82 1",
    )

    for wall_index, segment in enumerate(wall_segments):
        center_col = segment["start_col"] + segment["length_cells"] / 2.0
        x = center_col * cell_size
        y = (segment["row"] + 0.5) * cell_size

        append_box_model(
            parts=parts,
            name=f"wall_segment_{wall_index}",
            pose_xyz_rpy=[x, y, wall_z, 0, 0, 0],
            size_xyz=[
                segment["length_cells"] * cell_size,
                cell_size,
                wall_height,
            ],
            ambient_rgba="0.12 0.12 0.12 1",
            diffuse_rgba="0.12 0.12 0.12 1",
        )

    for action_goal in scenario.get("action_goals", []):
        row, col = (int(value) for value in action_goal["cell"])
        x, y = cell_to_world(row, col, cell_size)

        append_box_model(
            parts=parts,
            name=str(action_goal["id"]),
            pose_xyz_rpy=[x, y, action_goal_height / 2.0 + 0.001, 0, 0, 0],
            size_xyz=[action_goal_size, action_goal_size, action_goal_height],
            ambient_rgba="1.0 0.35 0.0 1",
            diffuse_rgba="1.0 0.45 0.0 1",
            static=True,
            collision=False,
        )

    for robot in scenario.get("robots", []):
        if not robot.get("enabled", True):
            continue

        model = str(robot.get("model", "turtlebot3_burger"))
        if model != "turtlebot3_burger":
            raise RuntimeError(
                f"Unsupported robot model '{model}' for robot {robot['id']}. "
                "This launch file currently supports turtlebot3_burger only."
            )

        row, col = (int(value) for value in robot["start_cell"])
        x, y = cell_to_world(row, col, cell_size)
        yaw = float(robot.get("start_yaw", robot.get("yaw", 0.0)))

        append_turtlebot3_burger(
            parts=parts,
            name=str(robot["id"]),
            x=x,
            y=y,
            yaw=yaw,
        )

    parts.extend(["  </world>", "</sdf>"])

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(parts))

    blocked_count = sum(
        1 for row in grid for symbol in row if is_blocked(symbol)
    )
    print(
        f"[trust_costmap] Generated {output_path} with "
        f"{len(wall_segments)} wall segments, {blocked_count} blocked cells, "
        f"{len(scenario.get('action_goals', []))} action goals, and "
        f"{sum(robot.get('enabled', True) for robot in scenario.get('robots', []))} robots."
    )


def make_optional_bridge_actions(map_name, scenario, bridge_robot_topics):
    if not bridge_robot_topics:
        return []

    world_name = f"{map_name}_world"
    bridge_actions = []

    for robot in scenario.get("robots", []):
        if not robot.get("enabled", True):
            continue

        robot_id = str(robot["id"])
        bridge_actions.append(
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name=f"{robot_id}_bridge",
                output="screen",
                arguments=[
                    f"/model/{robot_id}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
                    f"/model/{robot_id}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
                    (
                        f"/world/{world_name}/model/{robot_id}/link/base_scan/"
                        "sensor/hls_lfcd_lds/scan@sensor_msgs/msg/LaserScan"
                        "[gz.msgs.LaserScan"
                    ),
                ],
                remappings=[
                    (f"/model/{robot_id}/cmd_vel", f"/{robot_id}/cmd_vel"),
                    (f"/model/{robot_id}/odometry", f"/{robot_id}/odom"),
                    (
                        f"/world/{world_name}/model/{robot_id}/link/base_scan/"
                        "sensor/hls_lfcd_lds/scan",
                        f"/{robot_id}/scan",
                    ),
                ],
            )
        )

    return bridge_actions


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got: {value}")


def prepare_generated_world(context):
    package_share = get_package_share_directory("trust_costmap")
    map_name = LaunchConfiguration("map_name").perform(context)
    scenario_file = LaunchConfiguration("scenario_file").perform(context)
    planner = LaunchConfiguration("planner").perform(context).strip().lower()
    allow_diagonal = parse_bool(
        LaunchConfiguration("allow_diagonal").perform(context)
    )
    action_goal_count = int(
        LaunchConfiguration("action_goal_count").perform(context)
    )
    action_goal_seed = int(
        LaunchConfiguration("action_goal_seed").perform(context)
    )
    if action_goal_seed < 0:
        action_goal_seed = SystemRandom().randrange(0, 2**31)

    if not planner:
        raise RuntimeError("planner cannot be empty.")

    bridge_robot_topics = parse_bool(
        LaunchConfiguration("bridge_robot_topics").perform(context)
    )

    map_path = os.path.join(
        package_share,
        "worlds",
        "movingai_mapf",
        f"{map_name}.map",
    )
    scenario_path = (
        scenario_file
        if os.path.isabs(scenario_file)
        else os.path.join(package_share, scenario_file)
    )
    generated_dir = os.path.join(os.path.expanduser("~"), ".ros", "trust_costmap")
    os.makedirs(generated_dir, exist_ok=True)

    generated_world_path = os.path.join(generated_dir, f"generated_{map_name}.sdf")
    generated_scenario_path = os.path.join(
        generated_dir,
        f"generated_{map_name}_scenario.yaml",
    )

    if not os.path.exists(map_path):
        raise RuntimeError(f"Map file not found: {map_path}")
    if not os.path.exists(scenario_path):
        raise RuntimeError(f"Scenario file not found: {scenario_path}")

    with open(scenario_path, "r", encoding="utf-8") as file:
        scenario = yaml.safe_load(file)
    if not scenario:
        raise RuntimeError(f"Scenario file is empty: {scenario_path}")

    map_data = load_movingai_map(map_path)
    resolved_scenario = resolve_scenario_layout(
        map_name=map_name,
        map_data=map_data,
        scenario=scenario,
        planner=planner,
        allow_diagonal=allow_diagonal,
        action_goal_count=action_goal_count,
        action_goal_seed=action_goal_seed,
    )

    with open(generated_scenario_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(resolved_scenario, file, sort_keys=False)

    generate_sdf_world(
        map_name=map_name,
        map_data=map_data,
        scenario=resolved_scenario,
        output_path=generated_world_path,
    )

    print(
        "[trust_costmap] Planning: "
        f"planner={planner}, allow_diagonal={allow_diagonal}"
    )
    print(
        "[trust_costmap] Action goals: "
        f"count={action_goal_count}, seed={action_goal_seed}"
    )
    print(f"[trust_costmap] Resolved scenario: {generated_scenario_path}")
    for robot in resolved_scenario.get("robots", []):
        if robot.get("enabled", True):
            print(f"[trust_costmap] Robot {robot['id']}: cell {robot['start_cell']}")
    print(
        "[trust_costmap] Action goal cells: "
        + ", ".join(
            str(item["cell"]) for item in resolved_scenario["action_goals"]
        )
    )

    turtlebot3_models_dir, _ = get_turtlebot3_model_paths()
    existing_resource_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    gz_resource_path = turtlebot3_models_dir
    if existing_resource_path:
        gz_resource_path = existing_resource_path + os.pathsep + turtlebot3_models_dir

    actions = [
        SetEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=gz_resource_path,
        ),
        ExecuteProcess(
            cmd=[
                "gz",
                "sim",
                "--render-engine",
                "ogre",
                "-r",
                "-v",
                "1",
                generated_world_path,
            ],
            output="screen",
        ),
        Node(
            package="trust_costmap",
            executable="experiment_manager",
            name="experiment_manager",
            output="screen",
            parameters=[
                {
                    "map_name": map_name,
                    "scenario_file": generated_scenario_path,
                    "planner": planner,
                    "allow_diagonal": allow_diagonal,
                    "action_goal_count": action_goal_count,
                    "action_goal_seed": action_goal_seed,
                }
            ],
        ),
    ]

    bridge_actions = make_optional_bridge_actions(
        map_name=map_name,
        scenario=resolved_scenario,
        bridge_robot_topics=bridge_robot_topics,
    )
    if bridge_actions:
        actions.append(TimerAction(period=5.0, actions=bridge_actions))

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map_name",
                default_value="room-32-32-4",
                description="MovingAI map name without the .map extension",
            ),
            DeclareLaunchArgument(
                "scenario_file",
                default_value="scenario.yaml",
                description="Scenario YAML file in the trust_costmap package root",
            ),
            DeclareLaunchArgument(
                "planner",
                default_value="astar",
                description="Planner name passed to the experiment manager",
            ),
            DeclareLaunchArgument(
                "allow_diagonal",
                default_value="false",
                description="Allow diagonal grid moves in the planner",
            ),
            DeclareLaunchArgument(
                "action_goal_count",
                default_value="8",
                description=(
                    "Number of spread-out action goals to generate"
                ),
            ),
            DeclareLaunchArgument(
                "action_goal_seed",
                default_value="21",
                description=(
                    "Reproducible seed for action goals and robot spawn locations. "
                    "Use a negative value for a fresh random layout."
                ),
            ),
            DeclareLaunchArgument(
                "bridge_robot_topics",
                default_value="false",
                description=(
                    "If true, start ros_gz_bridge parameter bridges for each "
                    "robot's cmd_vel, odom, and scan topics."
                ),
            ),
            OpaqueFunction(function=prepare_generated_world),
        ]
    )
