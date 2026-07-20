import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

FREE_SYMBOLS = frozenset({".", "G", "S"})
Cell = Tuple[int, int]


@dataclass(frozen=True)
class MovingAIMap:
    name: str
    map_type: str
    height: int
    width: int
    grid: List[str]


@dataclass(frozen=True)
class RobotConfig:
    robot_id: str
    role: str
    model: str
    start_cell: Cell
    goal_cell: Optional[Cell]
    enabled: bool


class ExperimentManagerNode(Node):
    """Publish the static map and lightweight RViz debug markers."""

    def __init__(self) -> None:
        super().__init__("experiment_manager")

        self.declare_parameter("map_name", "room-32-32-4")
        self.declare_parameter("scenario_file", "scenario.yaml")
        self.declare_parameter("planner", "astar")
        self.declare_parameter("allow_diagonal", False)
        self.declare_parameter("action_goal_count", 8)
        self.declare_parameter("action_goal_seed", 21)

        self.map_name = self.get_parameter("map_name").value
        self.scenario_file = self.get_parameter("scenario_file").value
        self.planner = self.get_parameter("planner").value
        self.allow_diagonal = bool(self.get_parameter("allow_diagonal").value)
        self.action_goal_count = int(self.get_parameter("action_goal_count").value)
        self.action_goal_seed = int(self.get_parameter("action_goal_seed").value)

        self.package_share = get_package_share_directory("trust_costmap")
        self.scenario = self.load_scenario(self.scenario_file)
        self.map_data = self.load_map(self.map_name)

        visualization = self.scenario.get("visualization", {})
        self.cell_size_m = float(visualization.get("cell_size_m", 0.5))
        self.action_goal_size_m = float(
            visualization.get("action_goal_size_m", 0.18)
        )
        self.action_goal_height_m = float(
            visualization.get("action_goal_height_m", 0.04)
        )
        self.robots = self.load_robots(self.scenario.get("robots", []))
        self.action_goals = self.load_action_goals(
            self.scenario.get("action_goals", [])
        )
        self.validate_generated_settings()
        self.validate_layout()

        self.base_map_pub = self.create_publisher(OccupancyGrid, "/base_map", 10)
        self.robot_markers_pub = self.create_publisher(
            MarkerArray, "/robot_markers", 10
        )
        self.layout_markers_pub = self.create_publisher(
            MarkerArray, "/start_goal_markers", 10
        )

        self.print_summary()
        self.timer = self.create_timer(1.0, self.publish_debug_state)

    def load_scenario(self, scenario_file: str) -> Dict:
        scenario_path = (
            scenario_file
            if os.path.isabs(scenario_file)
            else os.path.join(self.package_share, scenario_file)
        )
        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

        with open(scenario_path, "r", encoding="utf-8") as file:
            scenario = yaml.safe_load(file)

        if not scenario:
            raise ValueError(f"Scenario file is empty: {scenario_path}")
        return scenario

    def load_map(self, map_name: str) -> MovingAIMap:
        map_path = os.path.join(
            self.package_share,
            "worlds",
            "movingai_mapf",
            f"{map_name}.map",
        )
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map file not found: {map_path}")

        with open(map_path, "r", encoding="utf-8") as file:
            lines = [line.rstrip("\n") for line in file]

        map_type = None
        height = None
        width = None
        map_start = None

        for index, line in enumerate(lines):
            clean = line.strip()
            if clean.startswith("type "):
                map_type = clean.split()[1]
            elif clean.startswith("height "):
                height = int(clean.split()[1])
            elif clean.startswith("width "):
                width = int(clean.split()[1])
            elif clean == "map":
                map_start = index + 1
                break

        if None in (map_type, height, width, map_start):
            raise ValueError(f"Invalid MovingAI map file: {map_path}")

        grid = lines[map_start : map_start + height]
        if len(grid) != height:
            raise ValueError(
                f"Expected {height} map rows, found {len(grid)} in {map_path}"
            )
        for row_index, row in enumerate(grid):
            if len(row) != width:
                raise ValueError(
                    f"Map row {row_index} has width {len(row)}; expected {width}"
                )

        return MovingAIMap(
            name=map_name,
            map_type=str(map_type),
            height=int(height),
            width=int(width),
            grid=grid,
        )

    @staticmethod
    def load_robots(items: Sequence[Dict]) -> List[RobotConfig]:
        robots: List[RobotConfig] = []
        for item in items:
            start = item.get("start_cell")
            if start is None:
                raise ValueError(
                    f"Robot {item.get('id', '<unknown>')} has no resolved start_cell"
                )

            goal = item.get("goal_cell")
            robots.append(
                RobotConfig(
                    robot_id=str(item["id"]),
                    role=str(item.get("role", "robot")),
                    model=str(item.get("model", "turtlebot3_burger")),
                    start_cell=(int(start[0]), int(start[1])),
                    goal_cell=(int(goal[0]), int(goal[1])) if goal else None,
                    enabled=bool(item.get("enabled", True)),
                )
            )

        if not any(robot.enabled for robot in robots):
            raise ValueError("Scenario must contain at least one enabled robot")
        return robots

    @staticmethod
    def load_action_goals(items: Sequence[Dict]) -> List[Cell]:
        goals: List[Cell] = []
        for item in items:
            cell = item.get("cell")
            if cell is None:
                raise ValueError(f"Action goal is missing a cell: {item}")
            goals.append((int(cell[0]), int(cell[1])))
        return goals

    def is_free(self, cell: Cell) -> bool:
        row, col = cell
        return self.map_data.grid[row][col] in FREE_SYMBOLS

    def validate_generated_settings(self) -> None:
        if len(self.action_goals) != self.action_goal_count:
            raise ValueError(
                "Generated action-goal count does not match the launch argument: "
                f"expected {self.action_goal_count}, found {len(self.action_goals)}"
            )

        generated = self.scenario.get("generated_layout", {})
        scenario_seed = generated.get("action_goal_seed")
        if scenario_seed is not None and int(scenario_seed) != self.action_goal_seed:
            raise ValueError(
                "Generated action-goal seed does not match the launch argument: "
                f"expected {self.action_goal_seed}, found {scenario_seed}"
            )

    def validate_cell(self, cell: Cell, label: str) -> None:
        row, col = cell
        if not (0 <= row < self.map_data.height and 0 <= col < self.map_data.width):
            raise ValueError(f"{label} is outside the map: {cell}")
        if not self.is_free(cell):
            raise ValueError(f"{label} is on an occupied map cell: {cell}")

    def validate_layout(self) -> None:
        occupied: Dict[Cell, str] = {}

        for robot in self.robots:
            if not robot.enabled:
                continue
            self.validate_cell(robot.start_cell, f"Robot {robot.robot_id} start")
            if robot.start_cell in occupied:
                raise ValueError(
                    f"Robot {robot.robot_id} overlaps {occupied[robot.start_cell]} "
                    f"at {robot.start_cell}"
                )
            occupied[robot.start_cell] = f"robot {robot.robot_id}"

            if robot.goal_cell is not None:
                self.validate_cell(robot.goal_cell, f"Robot {robot.robot_id} goal")

        for index, cell in enumerate(self.action_goals, start=1):
            self.validate_cell(cell, f"Action goal {index}")
            if cell in occupied:
                raise ValueError(
                    f"Action goal {index} overlaps {occupied[cell]} at {cell}"
                )
            occupied[cell] = f"action goal {index}"

    def cell_to_world(self, cell: Cell) -> Tuple[float, float]:
        row, col = cell
        return (col + 0.5) * self.cell_size_m, (row + 0.5) * self.cell_size_m

    def publish_debug_state(self) -> None:
        self.publish_base_map()
        self.publish_robot_markers()
        self.publish_layout_markers()

    def publish_base_map(self) -> None:
        message = OccupancyGrid()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.info.resolution = self.cell_size_m
        message.info.width = self.map_data.width
        message.info.height = self.map_data.height
        message.info.origin.orientation.w = 1.0
        message.data = [
            0 if symbol in FREE_SYMBOLS else 100
            for row in self.map_data.grid
            for symbol in row
        ]
        self.base_map_pub.publish(message)

    def publish_robot_markers(self) -> None:
        marker_array = MarkerArray()
        marker_id = 0

        for robot in self.robots:
            if not robot.enabled:
                continue

            x, y = self.cell_to_world(robot.start_cell)
            body = self.make_marker(
                marker_id=marker_id,
                namespace="robots",
                marker_type=Marker.CYLINDER,
                x=x,
                y=y,
                z=0.12,
                scale=(0.22, 0.22, 0.24),
                color=self.role_color(robot.role),
            )
            marker_array.markers.append(body)
            marker_id += 1

            label = self.make_marker(
                marker_id=marker_id,
                namespace="robot_labels",
                marker_type=Marker.TEXT_VIEW_FACING,
                x=x,
                y=y,
                z=0.55,
                scale=(0.0, 0.0, 0.22),
                color=(1.0, 1.0, 1.0, 1.0),
            )
            label.text = robot.robot_id
            marker_array.markers.append(label)
            marker_id += 1

        self.robot_markers_pub.publish(marker_array)

    def publish_layout_markers(self) -> None:
        marker_array = MarkerArray()
        marker_id = 0

        for robot in self.robots:
            if not robot.enabled:
                continue

            x, y = self.cell_to_world(robot.start_cell)
            marker_array.markers.append(
                self.make_marker(
                    marker_id=marker_id,
                    namespace="starts",
                    marker_type=Marker.SPHERE,
                    x=x,
                    y=y,
                    z=0.35,
                    scale=(0.16, 0.16, 0.16),
                    color=(0.0, 1.0, 0.0, 0.9),
                )
            )
            marker_id += 1

            if robot.goal_cell is not None:
                goal_x, goal_y = self.cell_to_world(robot.goal_cell)
                marker_array.markers.append(
                    self.make_marker(
                        marker_id=marker_id,
                        namespace="robot_goals",
                        marker_type=Marker.SPHERE,
                        x=goal_x,
                        y=goal_y,
                        z=0.35,
                        scale=(0.16, 0.16, 0.16),
                        color=(1.0, 0.0, 1.0, 0.9),
                    )
                )
                marker_id += 1

        for index, cell in enumerate(self.action_goals, start=1):
            x, y = self.cell_to_world(cell)
            marker = self.make_marker(
                marker_id=marker_id,
                namespace="action_goals",
                marker_type=Marker.CUBE,
                x=x,
                y=y,
                z=self.action_goal_height_m / 2.0 + 0.001,
                scale=(
                    self.action_goal_size_m,
                    self.action_goal_size_m,
                    self.action_goal_height_m,
                ),
                color=(1.0, 0.4, 0.0, 1.0),
            )
            marker_array.markers.append(marker)
            marker_id += 1

            label = self.make_marker(
                marker_id=marker_id,
                namespace="action_goal_labels",
                marker_type=Marker.TEXT_VIEW_FACING,
                x=x,
                y=y,
                z=0.28,
                scale=(0.0, 0.0, 0.18),
                color=(1.0, 0.7, 0.2, 1.0),
            )
            label.text = str(index)
            marker_array.markers.append(label)
            marker_id += 1

        self.layout_markers_pub.publish(marker_array)

    def make_marker(
        self,
        marker_id: int,
        namespace: str,
        marker_type: int,
        x: float,
        y: float,
        z: float,
        scale: Tuple[float, float, float],
        color: Tuple[float, float, float, float],
    ) -> Marker:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    @staticmethod
    def role_color(role: str) -> Tuple[float, float, float, float]:
        if "malicious" in role:
            return (0.9, 0.05, 0.05, 1.0)
        if "reporter" in role:
            return (0.05, 0.25, 0.9, 1.0)
        return (0.05, 0.75, 0.25, 1.0)

    def print_summary(self) -> None:
        enabled_robots = [robot for robot in self.robots if robot.enabled]
        free_count = sum(
            symbol in FREE_SYMBOLS
            for row in self.map_data.grid
            for symbol in row
        )
        blocked_count = self.map_data.height * self.map_data.width - free_count

        self.get_logger().info(
            f"Map {self.map_name}: {self.map_data.width}x{self.map_data.height}, "
            f"free={free_count}, blocked={blocked_count}"
        )
        self.get_logger().info(
            f"Planner={self.planner}, allow_diagonal={self.allow_diagonal}, "
            f"action_goal_count={len(self.action_goals)}, "
            f"action_goal_seed={self.action_goal_seed}"
        )
        for robot in enabled_robots:
            self.get_logger().info(
                f"Robot {robot.robot_id}: role={robot.role}, "
                f"start_cell={list(robot.start_cell)}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExperimentManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
