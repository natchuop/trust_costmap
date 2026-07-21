"""ROS 2 experiment manager for modular multi-agent grid navigation.

This single file intentionally keeps the rapid-development prototype compact while
preserving modular boundaries through registries and small, replaceable methods.

Implemented now:
- MovingAI map loading and validation
- scenario-driven robot selection
- interchangeable costmap builders and planners
- weighted A* over arbitrary positive cell costs
- source-linked claim storage and trust updates
- dynamic action-goal generation and multi-agent route planning
- Gazebo-native action-goal and route visualization
- odometry-based waypoint following with per-robot velocity commands
- ROS map, path, marker, status, and diagnostic publication
- lightweight planning metrics and periodic replanning

Not implemented here yet:
- sensor-derived claims
- physical dynamic obstacle scheduling
- full end-of-run experiment result aggregation

Those pieces can be added to this class without changing the planner or costmap
interfaces.
"""

from __future__ import annotations

import csv
import heapq
import importlib.util
import json
import math
import os
import random
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path as FilePath
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TextIO, Tuple

import yaml

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


Cell = Tuple[int, int]  # (row, column)
CostGrid = List[List[float]]
CostmapBuilder = Callable[[float], CostGrid]
Planner = Callable[[CostGrid, Cell, Cell], "PlannerResult"]

FREE_SYMBOLS = {".", "G", "S"}
BLOCKED_COST = math.inf


@dataclass
class RobotConfig:
    robot_id: str
    role: str
    model: str
    start_cell: List[int]
    goal_cell: Optional[List[int]]
    enabled: bool


@dataclass
class MovingAIMap:
    name: str
    map_type: str
    height: int
    width: int
    grid: List[str]

    def count_free(self) -> int:
        return sum(1 for row in self.grid for ch in row if ch in FREE_SYMBOLS)

    def count_blocked(self) -> int:
        return sum(1 for row in self.grid for ch in row if ch not in FREE_SYMBOLS)


@dataclass
class Claim:
    """Source-linked occupancy claim.

    timestamp_sec is expressed in the experiment node's ROS clock seconds.
    report_trust stores trust at report time for the trust-fused baseline.
    Current trust is intentionally looked up at planning time by the proposed
    source-linked method.
    """

    robot_id: str
    row: int
    col: int
    report_type: str
    timestamp_sec: float
    confidence: float
    report_trust: float
    metadata: Dict = field(default_factory=dict)

    @property
    def impact(self) -> float:
        if self.report_type == "occupied":
            return 1.0
        if self.report_type == "free":
            return -1.0
        raise ValueError(f"Unsupported report_type: {self.report_type}")


@dataclass
class PlannerResult:
    path: List[Cell]
    total_cost: float
    expanded_nodes: int
    planning_time_sec: float

    @property
    def found(self) -> bool:
        return bool(self.path)


class ExperimentManagerNode(Node):
    """Single-node research scaffold with swappable costmaps and planners."""

    def __init__(self) -> None:
        super().__init__("experiment_manager")

        self.declare_parameter("map_name", "room-32-32-4")
        self.declare_parameter("scenario_file", "scenario.yaml")
        self.declare_parameter("planner", "astar")
        self.declare_parameter("costmap_method", "")
        self.declare_parameter("replan_period_sec", 1.0)
        self.declare_parameter("allow_diagonal", False)
        self.declare_parameter("action_goal_count", -1)
        self.declare_parameter("action_goal_seed", -1)
        self.declare_parameter("heatmap_reconnaissance_seed", -1)
        self.declare_parameter("attack_heatmap_path", "")
        self.declare_parameter("require_attack_heatmap", False)
        self.declare_parameter("enable_external_costmaps", True)
        self.declare_parameter("external_costmap_dir", "")

        # Baseline / sweep orchestration parameters. The launch file normally
        # supplies these, but every value has a standalone-safe default.
        self.declare_parameter("run_id", "")
        self.declare_parameter("run_dir", "")
        self.declare_parameter("trial_index", 0)
        self.declare_parameter("experiment_seed", -1)
        self.declare_parameter("enable_metrics", True)
        self.declare_parameter("trajectory_log_rate_hz", 5.0)
        self.declare_parameter("metrics_flush_period_sec", 2.0)
        self.declare_parameter("baseline_profile", "")
        self.declare_parameter("baseline_suite_file", "")

        self.map_name = self.get_parameter("map_name").value
        self.scenario_file = self.get_parameter("scenario_file").value
        self.package_share = get_package_share_directory("trust_costmap")
        self.require_attack_heatmap = bool(
            self.get_parameter("require_attack_heatmap").value
        )
        self.run_id_override = str(self.get_parameter("run_id").value).strip()
        self.run_dir_override = str(self.get_parameter("run_dir").value).strip()
        self.trial_index = int(self.get_parameter("trial_index").value)
        parameter_seed = int(self.get_parameter("experiment_seed").value)
        self.experiment_seed = parameter_seed
        self.metrics_enabled = bool(self.get_parameter("enable_metrics").value)
        self.baseline_profile_name = str(
            self.get_parameter("baseline_profile").value
        ).strip()
        self.baseline_suite_file = str(
            self.get_parameter("baseline_suite_file").value
        ).strip()
        self.baseline_profile_payload: Dict[str, Any] = {}
        self.metrics_finalized = False
        self.metrics_files: Dict[str, TextIO] = {}
        self.metrics_writers: Dict[str, csv.DictWriter] = {}
        self.metrics_paths: Dict[str, FilePath] = {}
        self.metric_counters: Dict[str, int] = {
            "claims": 0,
            "trust_updates": 0,
            "attack_events": 0,
            "plans": 0,
            "trajectory_samples": 0,
            "goal_events": 0,
        }

        self.get_logger().info("Experiment manager starting.")
        self.get_logger().info(f"Selected map_name: {self.map_name}")
        self.get_logger().info(f"Selected scenario_file: {self.scenario_file}")

        self.scenario = self.load_scenario(self.scenario_file)
        self.apply_selected_baseline_profile()
        self.visualization_config = self.scenario.get("visualization", {})
        self.experiment_config = self.scenario.get("experiment", {})
        self.evaluation_config = self.scenario.get("evaluation", {})
        self.planning_config = self.scenario.get("planning", {})
        self.costmap_config = self.scenario.get("costmap", {})
        self.trust_config = self.scenario.get("trust", {})
        self.control_config = self.scenario.get("control", {})
        self.execution_config = self.scenario.get("execution", {})
        self.mission_config = self.scenario.get("missions", {})
        self.gazebo_visualization_config = self.scenario.get(
            "gazebo_visualization", {}
        )
        self.malicious_attack_config = self.scenario.get(
            "malicious_attack",
            self.scenario.get("attack_runtime", {}),
        )
        self.logging_config = self.scenario.get("logging", {})
        self.baseline_config = self.scenario.get(
            "baseline",
            self.scenario.get("baseline_experiments", {}),
        )
        self.termination_config = self.scenario.get(
            "termination",
            self.scenario.get("experiment_termination", {}),
        )

        self.cell_size_m = float(self.visualization_config.get("cell_size_m", 0.5))
        self.allow_diagonal = bool(
            self.get_parameter("allow_diagonal").value
            or self.planning_config.get("allow_diagonal", False)
        )

        requested_method = str(self.get_parameter("costmap_method").value).strip()
        scenario_method = str(self.experiment_config.get("method", "static"))
        self.costmap_method = self.normalize_method_name(requested_method or scenario_method)
        self.planner_name = str(
            self.get_parameter("planner").value
            or self.planning_config.get("planner", "astar")
        ).strip().lower()

        self.map_data = self.load_map_by_name(self.map_name)
        self.robots = self.load_robot_configs(self.scenario)
        self.primary_robot = self.select_primary_robot()

        self.action_goal_config = self.scenario.get("action_goals", {})
        self.action_goals: List[Cell] = []
        self.agent_routes: Dict[str, List[Cell]] = {}
        self.agent_route_results: Dict[str, List[PlannerResult]] = {}
        self.route_revision = 0

        self.claims: List[Claim] = []
        self.claims_by_cell: Dict[Cell, List[Claim]] = {}
        self.robot_trust = self.initialize_robot_trust()
        self.fused_log_odds = self.empty_numeric_grid(0.0)

        self.last_cost_grid: Optional[CostGrid] = None
        self.last_plan = PlannerResult([], math.inf, 0, 0.0)
        self.replan_count = 0
        self.last_claim_revision = 0
        self.last_planned_claim_revision = -1

        # Planning state is kept separate from mission and costmap state so
        # future navigation methods can replace either layer independently.
        self.last_planned_start_cells: Dict[str, Cell] = {}
        self.last_planned_active_goals: Dict[str, Optional[Cell]] = {}
        self.initial_live_pose_replan_done = False
        self.replan_in_progress = False

        # Resolve the shared run directory before any runtime artifact is
        # created. A sweep runner can therefore treat one directory as one
        # immutable trial, regardless of the selected baseline method.
        self.configure_experiment_output()
        self.initialize_metrics_pipeline()
        self.validate_baseline_contract()

        # Runtime state for Gazebo visualization and waypoint control.
        self.world_name = self.make_world_name(self.map_name)
        self.runtime_visual_dir = FilePath.home() / ".ros" / "trust_costmap" / "runtime_visuals"
        self.runtime_visual_dir.mkdir(parents=True, exist_ok=True)
        self.last_gazebo_visual_key: Optional[Tuple[int, int]] = None
        self.route_visual_revision = 0
        self.last_gazebo_goal_revision: Optional[int] = None
        self.gazebo_legacy_visual_cleanup_done = False
        self.last_gazebo_error = ""
        self.gazebo_visual_sync_in_progress = False

        # Kinematic motion is paused after every replan until the matching
        # action-goal and A* route models have been synchronized into Gazebo.
        # This prevents repeated pose-service calls from starving route
        # visualization creation.
        self.kinematic_motion_ready = False

        self.robot_pose: Dict[str, Tuple[float, float, float]] = {}
        self.robot_velocity: Dict[str, Tuple[float, float]] = {}
        self.robot_last_odom_time: Dict[str, float] = {}

        # Per-robot route-following state.
        self.robot_control_state: Dict[str, str] = {}
        self.robot_stationary_cycles: Dict[str, int] = {}
        self.robot_previous_command: Dict[str, Tuple[float, float]] = {}
        self.robot_last_control_time: Dict[str, float] = {}
        self.robot_last_pose_sync_time: Dict[str, float] = {}

        # Gazebo DiffDrive odometry is local to each robot's spawn pose.
        # These dictionaries convert local odometry into the shared map frame.
        self.robot_spawn_pose: Dict[str, Tuple[float, float, float]] = {}
        self.robot_initial_odom_pose: Dict[
            str,
            Tuple[float, float, float],
        ] = {}
        
        self.robot_waypoint_index: Dict[str, int] = {}
        self.robot_mission_complete: Dict[str, bool] = {}
        

        self.robot_goal_queues: Dict[str, List[Cell]] = {}
        self.robot_goal_indices: Dict[str, int] = {}
        self.robot_active_goals: Dict[str, Optional[Cell]] = {}

        # Independent deterministic random stream for each robot.
        self.robot_goal_rng: Dict[str, random.Random] = {}

        # Used to prevent a robot from immediately returning to the goal it
        # just completed.
        self.robot_previous_goals: Dict[str, Optional[Cell]] = {}

        self.robot_completed_goal_visits: Dict[str, int] = {}
        self.robot_mission_cycles: Dict[str, int] = {}
        self.mission_replan_in_progress = False

        # Runtime state for heatmap-driven fake-obstacle claims.
        self.experiment_start_time_sec = self.now_sec()
        self.attack_heatmap_path: Optional[FilePath] = None
        self.attack_heatmap_payload: Dict = {}
        self.attack_ranked_cells: List[Dict] = []
        self.attack_event_count = 0
        self.attack_recent_cells: Dict[Cell, float] = {}
        self.active_malicious_cells: List[Cell] = []
        self.malicious_visual_revision = 0
        self.last_malicious_visual_revision = -1
        self.malicious_visual_sync_in_progress = False

        self.costmap_builders: Dict[str, CostmapBuilder] = {
            "static": self.build_static_costmap,
            "hard_threshold": self.build_hard_threshold_costmap,
            "soft_probabilistic": self.build_soft_probabilistic_costmap,
            "time_decay": self.build_time_decay_costmap,
            "trust_fused": self.build_trust_fused_costmap,
            "source_linked": self.build_source_linked_costmap,
        }
        self.costmap_method_sources: Dict[str, str] = {
            name: "builtin" for name in self.costmap_builders
        }
        self.load_external_costmap_builders()

        self.planners: Dict[str, Planner] = {
            "astar": self.plan_astar,
            "dijkstra": self.plan_dijkstra,
        }

        self.base_map_pub = self.create_publisher(OccupancyGrid, "/base_map", 10)
        self.planning_costmap_pub = self.create_publisher(
            OccupancyGrid, "/planning_costmap", 10
        )
        self.robot_markers_pub = self.create_publisher(
            MarkerArray, "/robot_markers", 10
        )
        self.start_goal_markers_pub = self.create_publisher(
            MarkerArray, "/start_goal_markers", 10
        )
        self.planned_path_pub = self.create_publisher(Path, "/planned_path", 10)
        self.plan_status_pub = self.create_publisher(String, "/plan_status", 10)
        self.action_goal_markers_pub = self.create_publisher(
            MarkerArray, "/action_goal_markers", 10
        )
        self.agent_route_markers_pub = self.create_publisher(
            MarkerArray, "/agent_route_markers", 10
        )
        self.agent_route_publishers: Dict[str, object] = {
            robot.robot_id: self.create_publisher(
                Path, f"/agent_routes/{robot.robot_id}", 10
            )
            for robot in self.robots
            if robot.enabled
        }
        self.cmd_vel_publishers: Dict[str, object] = {
            robot.robot_id: self.create_publisher(
                Twist, f"/{robot.robot_id}/cmd_vel", 10
            )
            for robot in self.robots
            if robot.enabled
        }

        self.claim_sub = self.create_subscription(
            String, "/map_claims", self.claim_message_callback, 50
        )
        self.trust_update_sub = self.create_subscription(
            String, "/trust_updates", self.trust_update_callback, 20
        )
        self.action_goal_update_sub = self.create_subscription(
            String, "/action_goal_updates", self.action_goal_update_callback, 20
        )
        self.odom_subscriptions = [
            self.create_subscription(
                Odometry,
                f"/{robot.robot_id}/odom",
                lambda msg, robot_id=robot.robot_id: self.odom_callback(robot_id, msg),
                20,
            )
            for robot in self.robots
            if robot.enabled
        ]

        self.validate_robot_cells()
        self.initialize_robot_spawn_poses()
        self.initialize_action_goals()
        self.initialize_robot_goal_rngs()
        self.initialize_robot_missions()
        self.load_scripted_claims()
        self.initialize_malicious_attack()
        self.write_experiment_manifest()
        self.print_startup_summary()

        self.visual_timer = self.create_timer(1.0, self.publish_visual_debug)
        replan_period = float(
            self.get_parameter("replan_period_sec").value
            or self.planning_config.get("replan_period_sec", 1.0)
        )
        self.replan_timer = self.create_timer(max(0.05, replan_period), self.replan)

        control_rate_hz = max(1.0, float(self.control_config.get("control_rate_hz", 10.0)))
        self.control_timer = self.create_timer(1.0 / control_rate_hz, self.control_timer_callback)
        self.gazebo_visual_timer = self.create_timer(1.0, self.sync_gazebo_visuals)

        trajectory_rate_hz = max(0.1, float(
            self.get_parameter("trajectory_log_rate_hz").value
        ))
        self.trajectory_log_timer = self.create_timer(
            1.0 / trajectory_rate_hz,
            self.log_trajectory_samples,
        )
        flush_period_sec = max(0.25, float(
            self.get_parameter("metrics_flush_period_sec").value
        ))
        self.metrics_flush_timer = self.create_timer(
            flush_period_sec,
            self.flush_metric_files,
        )

        self.malicious_attack_timer = None
        if self.malicious_attack_enabled():
            attack_period = max(
                0.10,
                float(self.malicious_attack_config.get("report_period_sec", 5.0)),
            )
            self.malicious_attack_timer = self.create_timer(
                attack_period,
                self.malicious_attack_timer_callback,
            )

        self.replan(force=True)

    # ------------------------------------------------------------------
    # Optional baseline profile loading
    # ------------------------------------------------------------------

    @staticmethod
    def recursive_mapping_merge(
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recursively merge a profile into a scenario mapping.

        Lists and scalar values replace the base value. Nested dictionaries are
        merged. This keeps profiles small while ensuring every method receives
        the same untouched robot, control, map, and mission configuration unless
        the profile explicitly changes one of those fields.
        """

        merged: Dict[str, Any] = dict(base)
        for key, value in override.items():
            if (
                isinstance(value, dict)
                and isinstance(merged.get(key), dict)
            ):
                merged[key] = ExperimentManagerNode.recursive_mapping_merge(
                    dict(merged[key]), value
                )
            else:
                merged[key] = value
        return merged

    def resolve_auxiliary_config_path(self, requested: str) -> FilePath:
        candidate = FilePath(
            os.path.expandvars(os.path.expanduser(requested))
        )
        if candidate.is_absolute():
            return candidate
        return FilePath(self.package_share) / candidate

    def apply_selected_baseline_profile(self) -> None:
        """Apply an optional sweep profile before runtime config is extracted.

        Supported suite structure:

        profiles:
          clean:
            scenario_overrides:
              malicious_attack:
                enabled: false
          immediate_attack:
            scenario_overrides:
              malicious_attack:
                enabled: true
                start_delay_sec: 20.0

        A profile may also contain the scenario sections directly, without the
        scenario_overrides wrapper. Metadata keys beginning with an underscore
        are ignored when direct-section mode is used.
        """

        if not self.baseline_profile_name:
            return
        if not self.baseline_suite_file:
            raise ValueError(
                "baseline_profile was provided but baseline_suite_file is empty"
            )

        suite_path = self.resolve_auxiliary_config_path(
            self.baseline_suite_file
        )
        if not suite_path.exists():
            raise FileNotFoundError(
                f"Baseline suite file not found: {suite_path}"
            )

        suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        if not isinstance(suite, dict):
            raise ValueError(
                f"Baseline suite must contain a YAML mapping: {suite_path}"
            )

        profiles = suite.get("profiles", suite.get("baselines", {}))
        if not isinstance(profiles, dict):
            raise ValueError(
                "Baseline suite must define a profiles or baselines mapping"
            )

        profile = profiles.get(self.baseline_profile_name)
        if not isinstance(profile, dict):
            available = ", ".join(sorted(str(name) for name in profiles))
            raise ValueError(
                f"Unknown baseline profile '{self.baseline_profile_name}'. "
                f"Available profiles: {available or 'none'}"
            )

        raw_overrides = profile.get("scenario_overrides", profile)
        if not isinstance(raw_overrides, dict):
            raise ValueError(
                f"Baseline profile {self.baseline_profile_name} overrides "
                "must be a YAML mapping"
            )

        overrides = {
            key: value
            for key, value in raw_overrides.items()
            if not str(key).startswith("_")
            and key not in {"description", "tags", "expected_behavior"}
        }
        self.scenario = self.recursive_mapping_merge(
            self.scenario,
            overrides,
        )
        self.baseline_profile_payload = profile
        self.get_logger().info(
            f"Applied baseline profile '{self.baseline_profile_name}' "
            f"from {suite_path}"
        )

    # ------------------------------------------------------------------
    # Scenario and map loading
    # ------------------------------------------------------------------

    def load_scenario(self, scenario_file: str) -> Dict:
        scenario_path = os.path.join(self.package_share, scenario_file)
        if not os.path.exists(scenario_path):
            raise FileNotFoundError(
                f"Scenario file not found: {scenario_path}. "
                "Make sure it is installed by setup.py."
            )
        with open(scenario_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if not isinstance(data, dict):
            raise ValueError(f"Scenario must contain a YAML mapping: {scenario_path}")
        return data

    def load_map_by_name(self, map_name: str) -> MovingAIMap:
        map_path = os.path.join(
            self.package_share, "worlds", "movingai_mapf", f"{map_name}.map"
        )
        if not os.path.exists(map_path):
            raise FileNotFoundError(
                f"Map file not found: {map_path}. "
                f"Expected worlds/movingai_mapf/{map_name}.map"
            )
        return self.load_movingai_map(map_path, map_name)

    def load_movingai_map(self, path: str, map_name: str) -> MovingAIMap:
        with open(path, "r", encoding="utf-8") as file:
            lines = [line.rstrip("\n") for line in file]

        map_type: Optional[str] = None
        height: Optional[int] = None
        width: Optional[int] = None
        map_start_index: Optional[int] = None

        for index, line in enumerate(lines):
            clean = line.strip()
            if clean.startswith("type "):
                map_type = clean.split(maxsplit=1)[1]
            elif clean.startswith("height "):
                height = int(clean.split(maxsplit=1)[1])
            elif clean.startswith("width "):
                width = int(clean.split(maxsplit=1)[1])
            elif clean == "map":
                map_start_index = index + 1
                break

        if map_type is None or height is None or width is None or map_start_index is None:
            raise ValueError(f"Incomplete MovingAI header in map: {path}")

        grid = lines[map_start_index : map_start_index + height]
        if len(grid) != height:
            raise ValueError(f"Expected {height} grid rows, got {len(grid)} in {path}")
        for row_index, row in enumerate(grid):
            if len(row) != width:
                raise ValueError(
                    f"Map row {row_index} expected width {width}, got {len(row)}"
                )

        return MovingAIMap(map_name, map_type, height, width, grid)

    def load_robot_configs(self, scenario: Dict) -> List[RobotConfig]:
        robots_raw = scenario.get("robots", [])
        if not robots_raw:
            raise ValueError("Scenario must define at least one robot.")

        robots: List[RobotConfig] = []
        for item in robots_raw:
            goal_cell = item.get("goal_cell")
            robots.append(
                RobotConfig(
                    robot_id=str(item["id"]),
                    role=str(item["role"]),
                    model=str(item.get("model", "simple_box_robot")),
                    start_cell=list(item["start_cell"]),
                    goal_cell=list(goal_cell) if goal_cell is not None else None,
                    enabled=bool(item.get("enabled", True)),
                )
            )
        return robots

    def select_primary_robot(self) -> RobotConfig:
        requested = str(self.evaluation_config.get("primary_robot", "")).strip()
        if requested:
            for robot in self.robots:
                if robot.robot_id == requested and robot.enabled:
                    if robot.goal_cell is None:
                        raise ValueError(f"Primary robot {requested} has no goal_cell.")
                    return robot
            raise ValueError(f"Primary robot not found or disabled: {requested}")

        preferred_roles = {"navigation_robot", "navigator", "ego", "ego_robot"}
        for robot in self.robots:
            if robot.enabled and robot.goal_cell is not None and robot.role in preferred_roles:
                return robot
        for robot in self.robots:
            if robot.enabled and robot.goal_cell is not None:
                return robot
        raise ValueError("No enabled robot with a goal_cell is available for planning.")

    # ------------------------------------------------------------------
    # Geometry and validation
    # ------------------------------------------------------------------

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.map_data.height and 0 <= col < self.map_data.width

    def is_free_cell(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.map_data.grid[row][col] in FREE_SYMBOLS

    def is_blocked_cell(self, row: int, col: int) -> bool:
        return not self.is_free_cell(row, col)

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        x = (col + 0.5) * self.cell_size_m
        y = (row + 0.5) * self.cell_size_m
        return x, y

    def world_to_cell(self, x: float, y: float) -> Cell:
        return int(math.floor(y / self.cell_size_m)), int(math.floor(x / self.cell_size_m))

    @staticmethod
    def make_world_name(map_name: str) -> str:
        return f"{map_name.replace('-', '_')}_world"

    def validate_robot_cells(self) -> None:
        for robot in self.robots:
            if not robot.enabled:
                continue
            self.validate_named_cell(robot.robot_id, "start", robot.start_cell)
            if robot.goal_cell is not None:
                self.validate_named_cell(robot.robot_id, "goal", robot.goal_cell)

    def validate_named_cell(self, robot_id: str, label: str, cell: Sequence[int]) -> None:
        row, col = int(cell[0]), int(cell[1])
        if not self.in_bounds(row, col):
            raise ValueError(f"Robot {robot_id} {label}_cell out of bounds: {cell}")
        if self.is_blocked_cell(row, col):
            raise ValueError(f"Robot {robot_id} {label}_cell is blocked: {cell}")


    def initialize_robot_spawn_poses(self) -> None:
        """Store each robot's configured world pose at simulation startup."""

        for robot in self.robots:
            if not robot.enabled:
                continue

            row = int(robot.start_cell[0])
            col = int(robot.start_cell[1])
            spawn_x, spawn_y = self.cell_to_world(row, col)

            robot_config = next(
                (
                    item
                    for item in self.scenario.get("robots", [])
                    if str(item.get("id")) == robot.robot_id
                ),
                {},
            )

            spawn_yaw = float(
                robot_config.get(
                    "start_yaw",
                    robot_config.get("yaw", 0.0),
                )
            )

            self.robot_spawn_pose[robot.robot_id] = (
                spawn_x,
                spawn_y,
                spawn_yaw,
            )

            # In kinematic execution mode, the manager owns the map-frame pose.
            self.robot_pose[robot.robot_id] = (
                spawn_x,
                spawn_y,
                spawn_yaw,
            )
            self.robot_velocity[robot.robot_id] = (0.0, 0.0)
            self.robot_last_odom_time[robot.robot_id] = self.now_sec()

            self.get_logger().info(
                f"Robot spawn pose: {robot.robot_id} -> "
                f"x={spawn_x:.3f}, y={spawn_y:.3f}, "
                f"yaw={spawn_yaw:.3f}"
            )

    # ------------------------------------------------------------------
    # Dynamic action goals and multi-agent routes
    # ------------------------------------------------------------------

    def initialize_action_goals(self) -> None:
        configured_cells = self.action_goal_config.get("cells", [])
        if configured_cells:
            self.set_action_goals(configured_cells)
            return

        parameter_count = int(self.get_parameter("action_goal_count").value)
        count = (
            parameter_count
            if parameter_count >= 0
            else int(self.action_goal_config.get("count", 0))
        )
        if count > 0:
            self.regenerate_action_goals(count)

    def action_goal_seed(self) -> int:
        parameter_seed = int(self.get_parameter("action_goal_seed").value)
        if parameter_seed >= 0:
            return parameter_seed
        return int(self.action_goal_config.get("seed", 0))

    def reachable_free_cells(self, start: Cell) -> set[Cell]:
        if not self.is_free_cell(*start):
            return set()
        visited = {start}
        frontier = [start]
        while frontier:
            row, col = frontier.pop()
            for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (row + d_row, col + d_col)
                if neighbor not in visited and self.is_free_cell(*neighbor):
                    visited.add(neighbor)
                    frontier.append(neighbor)
        return visited

    def candidate_action_goal_cells(self) -> List[Cell]:
        enabled = [robot for robot in self.robots if robot.enabled]
        reserved = {
            (int(robot.start_cell[0]), int(robot.start_cell[1])) for robot in enabled
        }
        reserved.update(
            (int(robot.goal_cell[0]), int(robot.goal_cell[1]))
            for robot in enabled
            if robot.goal_cell is not None
        )

        reachable_sets = [
            self.reachable_free_cells(
                (int(robot.start_cell[0]), int(robot.start_cell[1]))
            )
            for robot in enabled
        ]
        common_reachable = (
            set.intersection(*reachable_sets) if reachable_sets else set()
        )
        return sorted(common_reachable - reserved)

    def regenerate_action_goals(self, count: int, seed: Optional[int] = None) -> None:
        if count < 0:
            raise ValueError("Action goal count cannot be negative.")
        candidates = self.candidate_action_goal_cells()
        if count > len(candidates):
            raise ValueError(
                f"Requested {count} action goals, but only {len(candidates)} free cells "
                "are available after reserving robot endpoints."
            )

        rng = random.Random(self.action_goal_seed() if seed is None else int(seed))
        min_spacing = max(0.0, float(self.action_goal_config.get("min_spacing_cells", 3.0)))
        rng.shuffle(candidates)
        selected: List[Cell] = []
        for candidate in candidates:
            if all(math.dist(candidate, existing) >= min_spacing for existing in selected):
                selected.append(candidate)
                if len(selected) == count:
                    break

        if len(selected) < count:
            self.get_logger().warn(
                f"Only {len(selected)} goals satisfied min_spacing_cells={min_spacing}; "
                "filling remaining goals from other free cells."
            )
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)
                    if len(selected) == count:
                        break

        self.action_goals = selected
        self.route_revision += 1
        self.get_logger().info(
            f"Generated {len(self.action_goals)} dynamic action goals with seed "
            f"{self.action_goal_seed() if seed is None else int(seed)}."
        )

    def set_action_goals(self, cells: Iterable[Sequence[int]]) -> None:
        parsed: List[Cell] = []
        for index, cell in enumerate(cells):
            row, col = int(cell[0]), int(cell[1])
            if not self.in_bounds(row, col):
                raise ValueError(f"Action goal #{index} is out of bounds: {(row, col)}")
            if self.is_blocked_cell(row, col):
                raise ValueError(f"Action goal #{index} is blocked: {(row, col)}")
            if (row, col) not in parsed:
                parsed.append((row, col))
        self.action_goals = parsed
        self.route_revision += 1

    def action_goal_update_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            operation = str(payload.get("operation", "add")).strip().lower()
            if operation == "add":
                row, col = self.extract_claim_cell(payload)
                self.set_action_goals([*self.action_goals, (row, col)])
            elif operation == "remove":
                row, col = self.extract_claim_cell(payload)
                self.action_goals = [cell for cell in self.action_goals if cell != (row, col)]
                self.route_revision += 1
            elif operation == "clear":
                self.action_goals = []
                self.route_revision += 1
            elif operation == "set":
                self.set_action_goals(payload.get("cells", []))
            elif operation == "regenerate":
                self.regenerate_action_goals(
                    int(payload.get("count", len(self.action_goals))),
                    payload.get("seed"),
                )
            else:
                raise ValueError(
                    "operation must be add, remove, clear, set, or regenerate"
                )
            self.initialize_robot_goal_rngs()
            self.initialize_robot_missions(reset_progress=True)
            self.last_claim_revision += 1
            self.replan(force=True)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"Rejected /action_goal_updates message: {exc}")

    def initialize_robot_goal_rngs(self) -> None:
        """Create a reproducible independent random stream for each robot."""

        base_seed = int(
            self.action_goal_config.get(
                "assignment_seed",
                self.action_goal_seed(),
            )
        )

        enabled_robots = [
            robot
            for robot in self.robots
            if robot.enabled
        ]
    
        for robot_index, robot in enumerate(enabled_robots):
            # Use a deterministic offset instead of hash(robot_id), because
            # Python hashes may differ between processes.
            robot_seed = base_seed + ((robot_index + 1) * 1009)

            self.robot_goal_rng[robot.robot_id] = random.Random(
                robot_seed
            )

            self.get_logger().info(
                f"Goal assignment RNG: {robot.robot_id}, "
                f"seed={robot_seed}"
            )

    def select_random_goal_for_robot(
        self,
        robot_id: str,
    ) -> Optional[Cell]:
        """
        Select a new random destination for one robot.

        When possible, avoid:
        - the robot's current goal;
        - the goal it just completed;
        - goals currently assigned to other robots.
        """

        if not self.action_goals:
            return None

        rng = self.robot_goal_rng.get(robot_id)

        if rng is None:
            raise RuntimeError(
                f"No goal-assignment RNG exists for {robot_id}"
            )

        current_goal = self.robot_active_goals.get(robot_id)
        previous_goal = self.robot_previous_goals.get(robot_id)

        excluded: set[Cell] = set()

        if bool(
            self.action_goal_config.get(
                "avoid_immediate_repeat",
                True,
            )
        ):
            if current_goal is not None:
                excluded.add(current_goal)

            if previous_goal is not None:
                excluded.add(previous_goal)

        if bool(
            self.action_goal_config.get(
                "avoid_shared_active_goals",
                True,
            )
        ):
            for other_robot_id, other_goal in self.robot_active_goals.items():
                if (
                    other_robot_id != robot_id
                    and other_goal is not None
                ):
                    excluded.add(other_goal)

        candidates = [
            goal
            for goal in self.action_goals
            if goal not in excluded
        ]

        # If all goals are currently assigned, allow sharing a destination,
        # but still avoid this robot's current and previous goals.
        if not candidates:
            candidates = [
                goal
                for goal in self.action_goals
                if (
                    goal != current_goal
                    and goal != previous_goal
                )
            ]

        # This only happens when there is one goal or an extremely small pool.
        if not candidates:
            candidates = list(self.action_goals)

        return rng.choice(candidates)

    def ordered_goals_for_robot(
        self,
        robot_index: int,
    ) -> List[Cell]:
        """Build a fixed goal queue for the non-random assignment modes."""

        if not self.action_goals:
            return []

        assignment = str(
            self.action_goal_config.get(
                "assignment",
                "random_dynamic",
            )
        ).strip().lower()

        # random_dynamic does not use a predetermined queue.
        if assignment == "random_dynamic":
            return []

        if assignment == "round_robin":
            enabled_count = max(
                1,
                len([
                    robot
                    for robot in self.robots
                    if robot.enabled
                ]),
            )

            return [
                goal
                for index, goal in enumerate(self.action_goals)
                if index % enabled_count == robot_index
            ]
    
        if assignment == "rotating_all":
            offset = robot_index % len(self.action_goals)

            return (
                self.action_goals[offset:]
                + self.action_goals[:offset]
            )

        raise ValueError(
            "action_goals.assignment must be one of: "
            "random_dynamic, rotating_all, round_robin"
        )

    def initialize_robot_missions(
        self,
        reset_progress: bool = True,
    ) -> None:
        """Initialize one active mission destination for every enabled robot."""

        enabled_robots = [
            robot
            for robot in self.robots
            if robot.enabled
        ]

        assignment = str(
            self.action_goal_config.get(
               "assignment",
                "random_dynamic",
            )
        ).strip().lower()

        if reset_progress:
            # Clear assignments first so initial random selection can avoid
            # giving two robots the same active destination.
            self.robot_active_goals.clear()

        for robot_index, robot in enumerate(enabled_robots):
            robot_id = robot.robot_id

            if reset_progress:
                self.robot_goal_indices[robot_id] = 0
                self.robot_completed_goal_visits[robot_id] = 0
                self.robot_mission_cycles[robot_id] = 0
                self.robot_previous_goals[robot_id] = None

            if assignment == "random_dynamic":
                selected_goal = self.select_random_goal_for_robot(
                    robot_id
                )

                # Preserve the configured goal_cell as a fallback when no
                # generated action goals exist.
                if selected_goal is None and robot.goal_cell is not None:
                    selected_goal = (
                        int(robot.goal_cell[0]),
                        int(robot.goal_cell[1]),
                    )

                self.robot_active_goals[robot_id] = selected_goal

                # For random mode, this list only represents the current
                # assignment. It is not a predetermined future queue.
                self.robot_goal_queues[robot_id] = (
                    [selected_goal]
                    if selected_goal is not None
                    else []
                )

                self.robot_mission_complete[robot_id] = (
                    selected_goal is None
                )

                self.get_logger().info(
                    f"Initial random goal: "
                    f"{robot_id} -> {selected_goal}"
                )

                continue

            # Existing fixed-queue modes.
            queue = self.ordered_goals_for_robot(robot_index)

            if not queue and robot.goal_cell is not None:
                queue = [
                    (
                        int(robot.goal_cell[0]),
                        int(robot.goal_cell[1]),
                    )
                ]

            if (
                bool(
                    self.action_goal_config.get(
                        "return_to_start",
                        False,
                    )
                )
                and queue
            ):
                queue = [
                    *queue,
                    (
                        int(robot.start_cell[0]),
                        int(robot.start_cell[1]),
                    ),
                ]

            self.robot_goal_queues[robot_id] = list(queue)

            goal_index = self.robot_goal_indices.get(
                robot_id,
                0,
            )

            self.robot_active_goals[robot_id] = (
                queue[goal_index % len(queue)]
                if queue
                else None
            )

            self.robot_mission_complete[robot_id] = not bool(queue)

    def current_robot_cell(self, robot: RobotConfig) -> Cell:
        """Return the robot's current free map cell.

        Live map-frame odometry is authoritative. The configured start cell is
        only a startup fallback before the first odometry sample is available.
        """

        pose = self.robot_pose.get(robot.robot_id)
        if pose is None:
            return int(robot.start_cell[0]), int(robot.start_cell[1])

        requested = self.world_to_cell(pose[0], pose[1])
        if self.is_free_cell(*requested):
            return requested

        nearest = self.nearest_free_cell(requested)
        if nearest is not None:
            self.get_logger().warn(
                f"{robot.robot_id} odometry mapped to blocked/out-of-bounds "
                f"cell {requested}; planning from nearest free cell {nearest}."
            )
            return nearest

        self.get_logger().warn(
            f"Could not map {robot.robot_id} odometry pose "
            f"({pose[0]:.3f}, {pose[1]:.3f}) to a nearby free cell; "
            "using the configured start cell."
        )
        return int(robot.start_cell[0]), int(robot.start_cell[1])

    def nearest_free_cell(
        self,
        requested: Cell,
        max_radius: int = 3,
    ) -> Optional[Cell]:
        """Return the closest free cell around a discretized odometry cell."""

        requested_row, requested_col = requested

        for radius in range(max_radius + 1):
            candidates: List[Tuple[float, Cell]] = []

            for d_row in range(-radius, radius + 1):
                for d_col in range(-radius, radius + 1):
                    if max(abs(d_row), abs(d_col)) != radius:
                        continue

                    candidate = (
                        requested_row + d_row,
                        requested_col + d_col,
                    )
                    if self.is_free_cell(*candidate):
                        candidates.append(
                            (math.hypot(d_row, d_col), candidate)
                        )

            if candidates:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]

        return None

    def active_goal_for_robot(self, robot_id: str) -> Optional[Cell]:
        return self.robot_active_goals.get(robot_id)

    def advance_robot_goal(self, robot_id: str) -> bool:
        """
        Finish the current visit and assign the robot's next destination.
    
        Returns True when another destination was assigned.
        """

        completed = (
            self.robot_completed_goal_visits.get(
                robot_id,
                0,
            )
            + 1
        )

        self.robot_completed_goal_visits[robot_id] = completed

        max_visits = int(
            self.mission_config.get(
                "max_goal_visits",
                0,
            )
        )

        if max_visits > 0 and completed >= max_visits:
            self.robot_active_goals[robot_id] = None
            self.robot_goal_queues[robot_id] = []
            self.robot_mission_complete[robot_id] = True
            return False

        assignment = str(
            self.action_goal_config.get(
                "assignment",
                "random_dynamic",
            )
        ).strip().lower()

        if assignment == "random_dynamic":
            if not bool(
                self.mission_config.get(
                    "repeat_goals",
                    True,
                )
            ):
                self.robot_active_goals[robot_id] = None
                self.robot_goal_queues[robot_id] = []
                self.robot_mission_complete[robot_id] = True
                return False

            old_goal = self.robot_active_goals.get(robot_id)
            self.robot_previous_goals[robot_id] = old_goal

            new_goal = self.select_random_goal_for_robot(    
                robot_id
            )

            if new_goal is None:
                self.robot_active_goals[robot_id] = None
                self.robot_goal_queues[robot_id] = []
                self.robot_mission_complete[robot_id] = True
                return False

            self.robot_active_goals[robot_id] = new_goal
            self.robot_goal_queues[robot_id] = [new_goal]
            self.robot_goal_indices[robot_id] = completed
            self.robot_mission_cycles[robot_id] = completed
            self.robot_mission_complete[robot_id] = False
    
            self.get_logger().info(
                f"Random goal assigned: "
                f"{robot_id}: {old_goal} -> {new_goal}"    
            )

            return True

        # Existing behavior for rotating_all and round_robin.
        queue = self.robot_goal_queues.get(robot_id, [])

        if not queue:
            self.robot_active_goals[robot_id] = None
            self.robot_mission_complete[robot_id] = True
            return False

        next_index = (
            self.robot_goal_indices.get(robot_id, 0)
            + 1
        )

        if next_index >= len(queue):
            if not bool(
                self.mission_config.get(
                    "repeat_goals",
                    True,
                )
            ):
                self.robot_active_goals[robot_id] = None
                self.robot_mission_complete[robot_id] = True
                return False

            next_index = 0

            self.robot_mission_cycles[robot_id] = (
                self.robot_mission_cycles.get(
                    robot_id,
                    0,
                )
                + 1
            )

        self.robot_goal_indices[robot_id] = next_index
        self.robot_active_goals[robot_id] = queue[next_index]
        self.robot_mission_complete[robot_id] = False
    
        return True

    def plan_agent_routes(self, cost_grid: CostGrid) -> None:
        """Plan one current-cell-to-active-goal route for every enabled robot."""

        self.agent_routes = {}
        self.agent_route_results = {}
        planner = self.planners[self.planner_name]

        for robot in (item for item in self.robots if item.enabled):
            robot_id = robot.robot_id
            current = self.current_robot_cell(robot)
            target = self.active_goal_for_robot(robot_id)

            if target is None:
                self.agent_routes[robot_id] = [current]
                self.agent_route_results[robot_id] = []
                self.publish_zero_velocity(robot_id)
                continue

            if current == target:
                result = PlannerResult(
                    path=[current],
                    total_cost=0.0,
                    expanded_nodes=0,
                    planning_time_sec=0.0,
                )
            else:
                try:
                    result = planner(cost_grid, current, target)
                except ValueError as exc:
                    self.get_logger().warning(
                        f"Could not plan {robot_id}: "
                        f"{current} -> {target}: {exc}"
                    )
                    self.agent_routes[robot_id] = [current]
                    self.agent_route_results[robot_id] = []
                    self.publish_zero_velocity(robot_id)
                    continue

            self.agent_route_results[robot_id] = [result]
            self.agent_routes[robot_id] = (
                result.path if result.found else [current]
            )

            if result.found:
                self.get_logger().info(
                    f"A* route ready: robot={robot_id}, "
                    f"start={current}, goal={target}, "
                    f"cells={len(result.path)}, cost={result.total_cost:.3f}"
                )
            else:
                self.get_logger().warning(
                    f"No route for {robot_id}: {current} -> {target}"
                )
                self.publish_zero_velocity(robot_id)

    def odom_callback(self, robot_id: str, msg: Odometry) -> None:
        """Convert robot-local DiffDrive odometry into shared map coordinates."""

        if self.kinematic_execution_enabled():
            # Direct model-pose execution is authoritative. Wheel odometry from
            # the retained TurtleBot visual model must not overwrite it.
            return

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        local_x = float(position.x)
        local_y = float(position.y)
        local_yaw = math.atan2(
            2.0 * (
                orientation.w * orientation.z
                + orientation.x * orientation.y
            ),
            1.0 - 2.0 * (
                orientation.y * orientation.y
                + orientation.z * orientation.z
            ),
        )

        if robot_id not in self.robot_initial_odom_pose:
            self.robot_initial_odom_pose[robot_id] = (
                local_x,
                local_y,
                local_yaw,
            )
            self.get_logger().info(
                f"Initial odometry for {robot_id}: "
                f"x={local_x:.3f}, y={local_y:.3f}, yaw={local_yaw:.3f}"
            )

        initial_x, initial_y, initial_yaw = (
            self.robot_initial_odom_pose[robot_id]
        )
        spawn_x, spawn_y, spawn_yaw = self.robot_spawn_pose.get(
            robot_id,
            (0.0, 0.0, 0.0),
        )

        delta_x_local = local_x - initial_x
        delta_y_local = local_y - initial_y

        cos_spawn = math.cos(spawn_yaw)
        sin_spawn = math.sin(spawn_yaw)

        world_x = spawn_x + (
            cos_spawn * delta_x_local
            - sin_spawn * delta_y_local
        )
        world_y = spawn_y + (
            sin_spawn * delta_x_local
            + cos_spawn * delta_y_local
        )
        world_yaw = self.normalize_angle(
            spawn_yaw
            + self.normalize_angle(local_yaw - initial_yaw)
        )

        self.robot_pose[robot_id] = (
            world_x,
            world_y,
            world_yaw,
        )
        
        self.robot_velocity[robot_id] = (
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.angular.z),
        )
        self.robot_last_odom_time[robot_id] = self.now_sec()

        # The constructor may create a temporary startup route from configured
        # start cells before Gazebo odometry arrives. Replace it once all enabled
        # robots have live poses so every baseline route starts from real state.
        if not self.initial_live_pose_replan_done:
            enabled_ids = {
                robot.robot_id
                for robot in self.robots
                if robot.enabled
            }
            if enabled_ids.issubset(self.robot_pose):
                self.initial_live_pose_replan_done = True
                self.get_logger().info(
                    "Live odometry received for all enabled robots; "
                    "replanning baseline routes from current poses."
                )
                self.replan(force=True)

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def kinematic_execution_enabled(self) -> bool:
        """Return True when routes are executed by direct planar motion."""

        mode = str(
            self.execution_config.get(
                "mode",
                "kinematic_planar",
            )
        ).strip().lower()
        return mode in {"kinematic", "kinematic_planar", "grid_kinematic"}

    @staticmethod
    def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
        half = yaw * 0.5
        return 0.0, 0.0, math.sin(half), math.cos(half)

    def set_gazebo_model_pose(
        self,
        robot_id: str,
        pose: Tuple[float, float, float],
    ) -> bool:
        """Set one Gazebo model pose through the UserCommands service."""

        x, y, yaw = pose
        qx, qy, qz, qw = self.quaternion_from_yaw(yaw)
        request = (
            f'name: "{robot_id}" '
            f'position {{ x: {x:.9f} y: {y:.9f} z: 0.01 }} '
            f'orientation {{ x: {qx:.9f} y: {qy:.9f} '
            f'z: {qz:.9f} w: {qw:.9f} }}'
        )
        return self.gazebo_entity_request(
            "set_pose",
            "gz.msgs.Pose",
            request,
        )

    def shortest_angular_step(
        self,
        current: float,
        target: float,
        maximum_step: float,
    ) -> Tuple[float, bool]:
        error = self.normalize_angle(target - current)
        if abs(error) <= maximum_step:
            return self.normalize_angle(target), True
        return (
            self.normalize_angle(
                current + math.copysign(maximum_step, error)
            ),
            False,
        )

    def reset_waypoint_progress(self) -> None:
        """Point each controller at the first movement cell of its new route."""

        now_sec = self.now_sec()

        for robot in (item for item in self.robots if item.enabled):
            robot_id = robot.robot_id
            route = self.agent_routes.get(robot_id, [])

            self.robot_control_state[robot_id] = "drive"
            self.robot_stationary_cycles[robot_id] = 0
            self.robot_previous_command[robot_id] = (0.0, 0.0)
            self.robot_last_control_time[robot_id] = now_sec

            if len(route) <= 1:
                self.robot_waypoint_index[robot_id] = len(route)
                self.robot_mission_complete[robot_id] = len(route) == 0
                continue

            # A newly planned route starts at the robot's current cell, so the
            # first commanded waypoint is route[1].
            self.robot_waypoint_index[robot_id] = 1
            self.robot_mission_complete[robot_id] = False

    def publish_zero_velocity(self, robot_id: str) -> None:
        """Stop one robot and clear acceleration-limiter history."""

        self.robot_previous_command[robot_id] = (0.0, 0.0)
        self.robot_last_control_time[robot_id] = self.now_sec()

        publisher = self.cmd_vel_publishers.get(robot_id)
        if publisher is not None:
            publisher.publish(Twist())

    def limit_velocity_command(
        self,
        *,
        robot_id: str,
        command: Twist,
        now_sec: float,
    ) -> Twist:
        """Limit command changes using configured acceleration bounds."""

        previous_linear, previous_angular = self.robot_previous_command.get(
            robot_id,
            (0.0, 0.0),
        )
        previous_time = self.robot_last_control_time.get(robot_id, now_sec)
        dt = max(0.001, min(0.10, now_sec - previous_time))

        robot_model_config = self.scenario.get("robot_model", {})
        max_linear_acceleration = max(
            0.0,
            float(
                robot_model_config.get(
                    "max_linear_acceleration_mps2",
                    2.0,
                )
            ),
        )
        max_angular_acceleration = max(
            0.0,
            float(
                robot_model_config.get(
                    "max_angular_acceleration_radps2",
                    4.5,
                )
            ),
        )

        maximum_linear_change = max_linear_acceleration * dt
        maximum_angular_change = max_angular_acceleration * dt

        limited = Twist()
        limited.linear.x = self.clamp(
            float(command.linear.x),
            previous_linear - maximum_linear_change,
            previous_linear + maximum_linear_change,
        )
        limited.angular.z = self.clamp(
            float(command.angular.z),
            previous_angular - maximum_angular_change,
            previous_angular + maximum_angular_change,
        )

        self.robot_previous_command[robot_id] = (
            float(limited.linear.x),
            float(limited.angular.z),
        )
        self.robot_last_control_time[robot_id] = now_sec
        return limited

    def compute_grid_locked_command(
        self,
        *,
        pose: Tuple[float, float, float],
        route: Sequence[Cell],
        waypoint_index: int,
        max_linear: float,
        max_angular: float,
        waypoint_tolerance: float,
        heading_tolerance: float,
        slowdown_distance: float,
        linear_gain: float,
        angular_gain: float,
    ) -> Tuple[Twist, int]:
        """Track one cardinal A* segment using heading and cross-track error."""

        command = Twist()
        if waypoint_index <= 0 or waypoint_index >= len(route):
            return command, waypoint_index

        x, y, yaw = pose
        previous_cell = route[waypoint_index - 1]
        target_cell = route[waypoint_index]
        previous_x, previous_y = self.cell_to_world(*previous_cell)
        target_x, target_y = self.cell_to_world(*target_cell)

        segment_dx = target_x - previous_x
        segment_dy = target_y - previous_y
        cardinal_epsilon = 1e-9
        horizontal = (
            abs(segment_dx) > cardinal_epsilon
            and abs(segment_dy) <= cardinal_epsilon
        )
        vertical = (
            abs(segment_dy) > cardinal_epsilon
            and abs(segment_dx) <= cardinal_epsilon
        )

        if not horizontal and not vertical:
            raise ValueError(
                "Grid-locked control requires cardinal route segments, got "
                f"{previous_cell} -> {target_cell}. Set allow_diagonal: false."
            )

        if horizontal:
            direction = 1.0 if segment_dx > 0.0 else -1.0
            segment_yaw = 0.0 if direction > 0.0 else math.pi
            along_track_remaining = direction * (target_x - x)
            lateral_error = direction * (y - previous_y)
        else:
            direction = 1.0 if segment_dy > 0.0 else -1.0
            segment_yaw = math.pi / 2.0 if direction > 0.0 else -math.pi / 2.0
            along_track_remaining = direction * (target_y - y)
            lateral_error = -direction * (x - previous_x)

        distance_to_target = math.hypot(target_x - x, target_y - y)
        if distance_to_target <= waypoint_tolerance:
            return command, waypoint_index + 1

        cross_track_gain = max(
            0.0,
            float(self.control_config.get("cross_track_gain", 1.5)),
        )
        cross_track_tolerance = max(
            0.001,
            float(
                self.control_config.get(
                    "cross_track_tolerance_m",
                    0.015,
                )
            ),
        )
        minimum_linear = max(
            0.0,
            float(
                self.control_config.get(
                    "minimum_linear_speed_mps",
                    0.03,
                )
            ),
        )
        minimum_angular = max(
            0.0,
            float(
                self.control_config.get(
                    "minimum_angular_speed_radps",
                    0.10,
                )
            ),
        )

        # Convert lateral displacement into a small corrected heading. This is
        # a line-following law; lateral metres are never added directly to yaw.
        lookahead_distance = max(
            0.12,
            min(slowdown_distance, max(distance_to_target, 0.12)),
        )
        corrected_yaw = self.normalize_angle(
            segment_yaw
            - math.atan2(
                cross_track_gain * lateral_error,
                lookahead_distance,
            )
        )
        heading_error = self.normalize_angle(corrected_yaw - yaw)

        angular_command = self.clamp(
            angular_gain * heading_error,
            -max_angular,
            max_angular,
        )
        if 0.0 < abs(angular_command) < minimum_angular:
            angular_command = math.copysign(minimum_angular, angular_command)

        # Rotate in place when substantially misaligned. Once close enough,
        # move while applying a bounded lateral correction.
        if abs(heading_error) > heading_tolerance:
            command.angular.z = angular_command
            return command, waypoint_index

        forward_distance = max(
            0.0,
            min(max(0.0, along_track_remaining), distance_to_target),
        )
        slowdown_scale = min(
            1.0,
            forward_distance / max(slowdown_distance, 1e-6),
        )
        lateral_scale = max(
            0.20,
            1.0
            - abs(lateral_error)
            / max(cross_track_tolerance * 4.0, 1e-6),
        )
        heading_scale = max(
            0.20,
            1.0
            - abs(heading_error)
            / max(heading_tolerance, 1e-6),
        )

        linear_command = min(
            max_linear,
            linear_gain * forward_distance,
        )
        linear_command *= min(slowdown_scale, lateral_scale, heading_scale)

        if (
            forward_distance > waypoint_tolerance
            and 0.0 < linear_command < minimum_linear
        ):
            linear_command = minimum_linear

        command.linear.x = linear_command
        command.angular.z = angular_command
        return command, waypoint_index

    def complete_robot_goal_if_reached(
        self,
        *,
        robot_id: str,
        pose: Tuple[float, float, float],
        waypoint_tolerance: float,
    ) -> None:
        """Advance the mission after the current route and goal are complete."""

        active_goal = self.active_goal_for_robot(robot_id)
        if active_goal is None:
            self.robot_mission_complete[robot_id] = True
            return

        x, y, _ = pose
        goal_x, goal_y = self.cell_to_world(*active_goal)
        goal_tolerance = max(
            waypoint_tolerance,
            float(self.mission_config.get("goal_tolerance_m", 0.12)),
        )

        if math.hypot(goal_x - x, goal_y - y) > goal_tolerance:
            return

        visit_number = self.robot_completed_goal_visits.get(robot_id, 0) + 1
        self.get_logger().info(
            f"Goal reached: {robot_id} -> {active_goal}; "
            f"visit={visit_number}"
        )
        self.write_metric_row(
            "goals",
            {
                "time_sec": self.now_sec(),
                "elapsed_sec": self.elapsed_experiment_sec(),
                "robot_id": robot_id,
                "event": "reached",
                "goal_row": active_goal[0],
                "goal_col": active_goal[1],
                "visit_number": visit_number,
                "next_goal_row": "",
                "next_goal_col": "",
            },
        )

        has_next = self.advance_robot_goal(robot_id)
        next_goal = self.active_goal_for_robot(robot_id)
        if has_next and next_goal is not None:
            self.write_metric_row(
                "goals",
                {
                    "time_sec": self.now_sec(),
                    "elapsed_sec": self.elapsed_experiment_sec(),
                    "robot_id": robot_id,
                    "event": "assigned",
                    "goal_row": active_goal[0],
                    "goal_col": active_goal[1],
                    "visit_number": visit_number,
                    "next_goal_row": next_goal[0],
                    "next_goal_col": next_goal[1],
                },
            )
        if has_next and not self.mission_replan_in_progress:
            self.mission_replan_in_progress = True
            try:
                self.replan(force=True)
            finally:
                self.mission_replan_in_progress = False
        else:
            self.robot_mission_complete[robot_id] = True
            self.get_logger().info(f"Mission complete: {robot_id}")

    def control_timer_callback(self) -> None:
        """Execute A* routes as deterministic planar kinematic motion.

        Translation is constrained to the active cardinal grid segment. The
        model rotates in place, advances by speed * dt, and snaps exactly to
        every cell center. No wheel dynamics or odometry feedback are used.
        """

        if not bool(self.control_config.get("enabled", False)):
            return

        if not self.kinematic_execution_enabled():
            self.get_logger().error(
                "This manager build expects execution.mode=kinematic_planar."
            )
            for robot in (item for item in self.robots if item.enabled):
                self.publish_zero_velocity(robot.robot_id)
            return

        # A new plan is not executed until its complete route visualization has
        # been created. This gives the comparatively expensive Gazebo model
        # creation calls priority over high-frequency pose synchronization.
        if not self.kinematic_motion_ready:
            return

        now_sec = self.now_sec()
        linear_speed = max(
            0.0,
            float(self.control_config.get("linear_speed_mps", 0.20)),
        )
        angular_speed = max(
            0.0,
            float(self.control_config.get("angular_speed_radps", 1.0)),
        )
        heading_tolerance = max(
            1e-4,
            float(self.control_config.get("heading_tolerance_rad", 0.02)),
        )
        maximum_dt = max(
            0.01,
            float(self.execution_config.get("maximum_step_sec", 0.10)),
        )
        pose_sync_period = 1.0 / max(
            1.0,
            float(self.execution_config.get("gazebo_pose_rate_hz", 15.0)),
        )

        for robot in (item for item in self.robots if item.enabled):
            robot_id = robot.robot_id
            route = self.agent_routes.get(robot_id, [])
            pose = self.robot_pose.get(robot_id)

            if pose is None or not route:
                continue

            previous_time = self.robot_last_control_time.get(robot_id, now_sec)
            dt = max(0.0, min(maximum_dt, now_sec - previous_time))
            self.robot_last_control_time[robot_id] = now_sec
            if dt <= 0.0:
                continue

            waypoint_index = self.robot_waypoint_index.get(
                robot_id,
                1 if len(route) > 1 else len(route),
            )

            if waypoint_index >= len(route):
                self.robot_velocity[robot_id] = (0.0, 0.0)
                self.complete_robot_goal_if_reached(
                    robot_id=robot_id,
                    pose=pose,
                    waypoint_tolerance=1e-6,
                )
                continue

            current_x, current_y, current_yaw = pose
            previous_cell = route[waypoint_index - 1]
            target_cell = route[waypoint_index]
            previous_x, previous_y = self.cell_to_world(*previous_cell)
            target_x, target_y = self.cell_to_world(*target_cell)

            delta_x = target_x - previous_x
            delta_y = target_y - previous_y
            epsilon = 1e-9
            horizontal = abs(delta_x) > epsilon and abs(delta_y) <= epsilon
            vertical = abs(delta_y) > epsilon and abs(delta_x) <= epsilon
            if not horizontal and not vertical:
                self.get_logger().error(
                    f"Kinematic execution requires cardinal segments: "
                    f"{previous_cell} -> {target_cell}"
                )
                continue

            if horizontal:
                desired_yaw = 0.0 if delta_x > 0.0 else math.pi
            else:
                desired_yaw = math.pi / 2.0 if delta_y > 0.0 else -math.pi / 2.0

            maximum_yaw_step = angular_speed * dt
            next_yaw, aligned = self.shortest_angular_step(
                current_yaw,
                desired_yaw,
                maximum_yaw_step,
            )

            next_x = current_x
            next_y = current_y
            linear_velocity = 0.0
            angular_velocity = (
                self.normalize_angle(next_yaw - current_yaw) / dt
                if dt > 0.0
                else 0.0
            )
            reached_waypoint = False

            if aligned or abs(self.normalize_angle(desired_yaw - next_yaw)) <= heading_tolerance:
                next_yaw = desired_yaw
                angular_velocity = 0.0
                distance_step = linear_speed * dt

                if horizontal:
                    direction = 1.0 if delta_x > 0.0 else -1.0
                    remaining = direction * (target_x - current_x)
                    if remaining <= distance_step + epsilon:
                        next_x = target_x
                        next_y = target_y
                        reached_waypoint = True
                    else:
                        next_x = current_x + direction * distance_step
                        next_y = previous_y
                        linear_velocity = linear_speed
                else:
                    direction = 1.0 if delta_y > 0.0 else -1.0
                    remaining = direction * (target_y - current_y)
                    if remaining <= distance_step + epsilon:
                        next_x = target_x
                        next_y = target_y
                        reached_waypoint = True
                    else:
                        next_x = previous_x
                        next_y = current_y + direction * distance_step
                        linear_velocity = linear_speed

            next_pose = (next_x, next_y, self.normalize_angle(next_yaw))
            self.robot_pose[robot_id] = next_pose
            self.robot_velocity[robot_id] = (linear_velocity, angular_velocity)
            self.robot_last_odom_time[robot_id] = now_sec

            if reached_waypoint:
                self.robot_pose[robot_id] = (target_x, target_y, desired_yaw)
                self.robot_velocity[robot_id] = (0.0, 0.0)
                self.robot_waypoint_index[robot_id] = waypoint_index + 1
                next_pose = self.robot_pose[robot_id]

            last_sync = self.robot_last_pose_sync_time.get(robot_id, -math.inf)
            if reached_waypoint or now_sec - last_sync >= pose_sync_period:
                if self.set_gazebo_model_pose(robot_id, next_pose):
                    self.robot_last_pose_sync_time[robot_id] = now_sec

            if reached_waypoint and waypoint_index + 1 >= len(route):
                self.complete_robot_goal_if_reached(
                    robot_id=robot_id,
                    pose=self.robot_pose[robot_id],
                    waypoint_tolerance=1e-6,
                )

    def gazebo_entity_request(
        self,
        service_suffix: str,
        request_type: str,
        request: str,
        *,
        operation: str = "gazebo request",
    ) -> bool:
        """Call a Gazebo service and emit enough detail to diagnose failures."""

        timeout_ms = max(
            250,
            int(self.gazebo_visualization_config.get("service_timeout_ms", 1500)),
        )
        service_name = f"/world/{self.world_name}/{service_suffix}"
        command = [
            "gz",
            "service",
            "-s",
            service_name,
            "--reqtype",
            request_type,
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            str(timeout_ms),
            "--req",
            request,
        ]

        debug_enabled = bool(
            self.gazebo_visualization_config.get("debug_services", True)
        )
        if debug_enabled:
            self.get_logger().info(
                f"[GZ-VIS] BEGIN operation={operation} "
                f"service={service_name} reqtype={request_type}"
            )

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=(timeout_ms / 1000.0) + 1.0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = (
                f"[GZ-VIS] EXCEPTION operation={operation} "
                f"service={service_name}: {exc}"
            )
            self.get_logger().error(error)
            self.last_gazebo_error = error
            return False

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        combined = (stdout + "\n" + stderr).strip()

        succeeded = (
            completed.returncode == 0
            and "data: true" in combined.lower()
        )

        if debug_enabled or not succeeded:
            log_message = (
                f"[GZ-VIS] END operation={operation} success={succeeded} "
                f"returncode={completed.returncode} stdout={stdout!r} "
                f"stderr={stderr!r}"
            )
            if succeeded:
                self.get_logger().info(log_message)
            else:
                self.get_logger().error(log_message)

        if not succeeded:
            self.last_gazebo_error = combined or (
                f"gz service returned {completed.returncode}"
            )
            return False

        self.last_gazebo_error = ""
        return True

    def remove_gazebo_model(self, name: str) -> bool:
        request = f'name: "{name}" type: MODEL'
        return self.gazebo_entity_request(
            "remove",
            "gz.msgs.Entity",
            request,
            operation=f"remove model {name}",
        )

    def spawn_gazebo_sdf(self, name: str, sdf_text: str) -> bool:
        path = self.runtime_visual_dir / f"{name}.sdf"
        path.write_text(sdf_text, encoding="utf-8")
        self.get_logger().info(
            f"[GZ-VIS] wrote model={name} path={path} "
            f"bytes={len(sdf_text.encode('utf-8'))}"
        )
        request = f'sdf_filename: "{path}"'
        return self.gazebo_entity_request(
            "create",
            "gz.msgs.EntityFactory",
            request,
            operation=f"create model {name}",
        )

    @staticmethod
    def material_xml(r: float, g: float, b: float, a: float = 1.0) -> str:
        rgba = f"{r:.3f} {g:.3f} {b:.3f} 0.300"
        return (
            "<material>"
            f"<ambient>{rgba}</ambient>"
            f"<diffuse>{rgba}</diffuse>"
            f"<emissive>{0.15*r:.3f} {0.15*g:.3f} {0.15*b:.3f} {a:.3f}</emissive>"
            "</material>"
        )

    def make_action_goals_sdf(self) -> str:
        radius = max(
            0.02,
            float(
                self.gazebo_visualization_config.get(
                    "goal_radius_m", self.cell_size_m * 0.28
                )
            ),
        )
        height = max(
            0.005,
            float(self.gazebo_visualization_config.get("goal_height_m", 0.035)),
        )
        goal_z = max(
            0.0,
            float(self.gazebo_visualization_config.get("goal_z_m", 0.022)),
        )

        visual_z = goal_z + (height / 2.0)
        visuals: List[str] = []
        for index, (row, col) in enumerate(self.action_goals):
            x, y = self.cell_to_world(row, col)
            visuals.append(
                f"""
        <visual name="goal_{index}">
          <pose>{x:.6f} {y:.6f} {visual_z:.6f} 0 0 0</pose>
          <geometry><cylinder><radius>{radius:.6f}</radius><length>{height:.6f}</length></cylinder></geometry>
          {self.material_xml(1.0, 0.78, 0.0)}
        </visual>"""
            )
        return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="trust_action_goals">
    <static>true</static>
    <link name="visuals">{''.join(visuals)}
    </link>
  </model>
</sdf>
"""

    def role_color(self, role: str) -> Tuple[float, float, float]:
        role_lower = role.lower()
        if "malicious" in role_lower:
            return 0.90, 0.05, 0.05
        if "reporter" in role_lower:
            return 0.05, 0.25, 0.90
        return 0.05, 0.75, 0.25

    def make_all_routes_sdf(self) -> Tuple[str, int, int]:
        """Build a low-overhead, fully opaque Gazebo route model.

        Consecutive collinear A* cells are merged into long bars. Each merged
        bar is a visual-only static link: no waypoint spheres, no collision
        bodies, no shadows. This keeps the route conspicuous while drastically
        reducing Gazebo entity and rendering overhead.
        """

        width = max(
            0.05,
            float(
                self.gazebo_visualization_config.get(
                    "route_width_m", self.cell_size_m * 0.20
                )
            ),
        )
        height = max(
            0.02,
            float(self.gazebo_visualization_config.get("route_height_m", 0.06)),
        )
        route_z = max(
            0.02,
            float(self.gazebo_visualization_config.get("route_z_m", 0.08)),
        )
        center_z = route_z + (height / 2.0)

        links: List[str] = []
        route_count = 0
        segment_count = 0
        raw_segment_count = 0

        for robot in (item for item in self.robots if item.enabled):
            route = self.agent_routes.get(robot.robot_id, [])
            if len(route) <= 1:
                continue

            route_count += 1
            r, g, b = self.role_color(robot.role)
            rgba = f"{r:.3f} {g:.3f} {b:.3f} 1.000"
            points = [self.cell_to_world(*cell) for cell in route]
            raw_segment_count += len(points) - 1

            # Merge adjacent steps that continue in the same direction.
            merged: List[Tuple[float, float, float, float]] = []
            run_x1, run_y1 = points[0]
            prev_x, prev_y = points[0]
            run_dx = run_dy = None

            for next_x, next_y in points[1:]:
                dx = next_x - prev_x
                dy = next_y - prev_y
                direction = (
                    0 if abs(dx) <= 1e-9 else (1 if dx > 0.0 else -1),
                    0 if abs(dy) <= 1e-9 else (1 if dy > 0.0 else -1),
                )
                if run_dx is None:
                    run_dx, run_dy = direction
                elif direction != (run_dx, run_dy):
                    merged.append((run_x1, run_y1, prev_x, prev_y))
                    run_x1, run_y1 = prev_x, prev_y
                    run_dx, run_dy = direction
                prev_x, prev_y = next_x, next_y

            merged.append((run_x1, run_y1, prev_x, prev_y))

            for index, (x1, y1, x2, y2) in enumerate(merged):
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length <= 1e-9:
                    continue

                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                yaw = math.atan2(dy, dx)
                link_name = f"{robot.robot_id}_route_{index}"
                links.append(
                    f"""
    <link name="{link_name}">
      <pose>{center_x:.6f} {center_y:.6f} {center_z:.6f} 0 0 {yaw:.6f}</pose>
      <visual name="visual">
        <cast_shadows>false</cast_shadows>
        <transparency>0.7</transparency>
        <geometry>
          <box><size>{length:.6f} {width:.6f} {height:.6f}</size></box>
        </geometry>
        <material>
          <ambient>{rgba}</ambient>
          <diffuse>{rgba}</diffuse>
          <specular>{rgba}</specular>
          <emissive>{rgba}</emissive>
        </material>
      </visual>
    </link>"""
                )
                segment_count += 1

        sdf_text = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="trust_all_routes">
    <static>true</static>
{''.join(links)}
  </model>
</sdf>
"""

        self.get_logger().info(
            f"[GZ-VIS] optimized route model: routes={route_count}, "
            f"merged_segments={segment_count}, raw_segments={raw_segment_count}, "
            f"width={width:.3f}, height={height:.3f}, base_z={route_z:.3f}"
        )
        return sdf_text, route_count, segment_count

    def sync_gazebo_visuals(self) -> None:
        """Synchronize goals and all A* routes before motion resumes."""

        enabled = bool(self.gazebo_visualization_config.get("enabled", True))
        show_goals = bool(
            self.gazebo_visualization_config.get("show_action_goals", True)
        )
        show_routes = bool(
            self.gazebo_visualization_config.get("show_routes", True)
        )
        visual_key = (self.route_revision, self.route_visual_revision)

        self.get_logger().info(
            f"[GZ-VIS] sync requested key={visual_key} enabled={enabled} "
            f"show_goals={show_goals} show_routes={show_routes} "
            f"last_key={self.last_gazebo_visual_key}"
        )

        if not enabled:
            self.kinematic_motion_ready = True
            return
        if self.gazebo_visual_sync_in_progress:
            self.get_logger().warn("[GZ-VIS] sync skipped: already in progress")
            return
        if visual_key == self.last_gazebo_visual_key:
            self.kinematic_motion_ready = True
            return

        self.gazebo_visual_sync_in_progress = True
        try:
            success = True

            if (
                show_goals
                and self.last_gazebo_goal_revision != self.route_revision
            ):
                self.remove_gazebo_model("trust_action_goals")
                goals_sdf = self.make_action_goals_sdf()
                goals_spawned = self.spawn_gazebo_sdf(
                    "trust_action_goals",
                    goals_sdf,
                )
                if goals_spawned:
                    self.last_gazebo_goal_revision = self.route_revision
                self.get_logger().info(
                    f"[GZ-VIS] goals result success={goals_spawned} "
                    f"count={len(self.action_goals)}"
                )
                success = goals_spawned and success

            if show_routes:
                self.remove_gazebo_model("trust_all_routes")
                # Old per-robot marker names only need cleanup once per process.
                if not self.gazebo_legacy_visual_cleanup_done:
                    for robot in (item for item in self.robots if item.enabled):
                        self.remove_gazebo_model(f"trust_route_{robot.robot_id}")
                    self.gazebo_legacy_visual_cleanup_done = True

                routes_sdf, route_count, segment_count = self.make_all_routes_sdf()
                routes_path = self.runtime_visual_dir / "trust_all_routes.sdf"
                routes_path.write_text(routes_sdf, encoding="utf-8")
                self.get_logger().info(
                    f"[GZ-VIS] combined route SDF path={routes_path} "
                    f"routes={route_count} segments={segment_count}"
                )

                if route_count == 0 or segment_count == 0:
                    self.get_logger().error(
                        "[GZ-VIS] no drawable route segments were generated"
                    )
                    routes_spawned = False
                else:
                    routes_spawned = self.spawn_gazebo_sdf(
                        "trust_all_routes",
                        routes_sdf,
                    )

                self.get_logger().info(
                    f"[GZ-VIS] routes result success={routes_spawned} "
                    f"routes={route_count} segments={segment_count}"
                )
                success = routes_spawned and success

            if success:
                self.last_gazebo_visual_key = visual_key
                self.kinematic_motion_ready = True
                self.get_logger().info(
                    f"[GZ-VIS] SYNC SUCCESS key={visual_key}; motion enabled"
                )
            else:
                self.kinematic_motion_ready = False
                self.get_logger().error(
                    f"[GZ-VIS] SYNC FAILED key={visual_key}; motion paused; "
                    f"last_error={self.last_gazebo_error!r}"
                )
        finally:
            self.gazebo_visual_sync_in_progress = False

    # ------------------------------------------------------------------
    # Heatmap-driven malicious fake-obstacle generation
    # ------------------------------------------------------------------

    def malicious_attack_enabled(self) -> bool:
        return bool(self.malicious_attack_config.get("enabled", False))

    def resolved_attack_goal_count(self) -> int:
        parameter_count = int(self.get_parameter("action_goal_count").value)
        if parameter_count >= 0:
            return parameter_count
        return int(self.action_goal_config.get("count", len(self.action_goals)))

    def resolved_attack_reconnaissance_seed(self) -> int:
        parameter_seed = int(self.get_parameter("heatmap_reconnaissance_seed").value)
        if parameter_seed >= 0:
            return parameter_seed
        config = self.scenario.get("attack_reconnaissance", {})
        return int(config.get("seed", self.action_goal_seed()))

    def default_attack_heatmap_path(self) -> FilePath:
        """Return the generator's stable per-map JSON alias.

        Versioned files remain available for provenance, while the launch file
        and manager use this deterministic alias as their runtime contract.
        """
        filename = f"{self.map_name}_latest.json"
        return (
            FilePath.home()
            / ".ros"
            / "trust_costmap"
            / "attack_heatmaps"
            / filename
        )

    def resolve_attack_heatmap_path(self) -> FilePath:
        parameter_path = str(self.get_parameter("attack_heatmap_path").value).strip()
        configured_path = str(self.malicious_attack_config.get("heatmap_path", "")).strip()
        raw_path = parameter_path or configured_path
        if not raw_path:
            return self.default_attack_heatmap_path()
        return FilePath(os.path.expandvars(os.path.expanduser(raw_path)))

    def initialize_malicious_attack(self) -> None:
        """Load and validate the generated reconnaissance heatmap.

        Loading is required when either the runtime malicious attack is enabled
        or the launch file explicitly sets require_attack_heatmap. This keeps
        clean experiments possible while making attack experiments fail fast
        instead of silently falling back to arbitrary fake-obstacle placement.
        """
        should_load = self.malicious_attack_enabled() or self.require_attack_heatmap
        if not should_load:
            self.get_logger().info("Malicious heatmap attack: disabled")
            return

        self.attack_heatmap_path = self.resolve_attack_heatmap_path()
        if not self.attack_heatmap_path.exists():
            raise FileNotFoundError(
                "Attack heatmap is required, but the JSON does not exist: "
                f"{self.attack_heatmap_path}"
            )

        try:
            payload = json.loads(
                self.attack_heatmap_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Attack heatmap is not valid JSON: {self.attack_heatmap_path}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError("Attack heatmap JSON must contain an object.")
        if str(payload.get("map_name", "")) != self.map_name:
            raise ValueError(
                "Attack heatmap map_name does not match the active map: "
                f"{payload.get('map_name')} != {self.map_name}"
            )
        if (
            int(payload.get("height", -1)) != self.map_data.height
            or int(payload.get("width", -1)) != self.map_data.width
        ):
            raise ValueError("Attack heatmap dimensions do not match the map.")

        contract = payload.get("consumer_contract", {})
        ranked_field = str(contract.get("ranked_cell_field", "cells"))
        score_field = str(contract.get("score_field", "score"))
        raw_cells = payload.get(ranked_field, [])
        if not isinstance(raw_cells, list):
            raise ValueError(
                f"Attack heatmap field {ranked_field!r} must contain a list."
            )

        ranked: List[Dict] = []
        for item in raw_cells:
            if not isinstance(item, dict):
                continue
            row = int(item["row"])
            col = int(item["col"])
            score = float(item.get(score_field, 0.0))
            if score > 0.0 and self.is_free_cell(row, col):
                ranked.append({**item, "row": row, "col": col, "score": score})

        ranked.sort(key=lambda item: (-item["score"], item["row"], item["col"]))
        if not ranked:
            raise ValueError(
                f"Attack heatmap contains no positive-score free cells: "
                f"{self.attack_heatmap_path}"
            )

        self.attack_heatmap_payload = payload
        self.attack_ranked_cells = ranked
        mode = "active" if self.malicious_attack_enabled() else "loaded-only"
        self.get_logger().info(
            f"Malicious heatmap loaded: mode={mode}, "
            f"path={self.attack_heatmap_path}, ranked_cells={len(ranked)}, "
            f"policy={self.malicious_attack_config.get('placement_policy', 'topology_only')}"
        )

    def malicious_robot_ids(self) -> List[str]:
        configured = self.malicious_attack_config.get("robot_ids")
        if configured is None and self.malicious_attack_config.get("robot_id") is not None:
            configured = [self.malicious_attack_config.get("robot_id")]

        if configured:
            ids = [str(value) for value in configured]
        else:
            ids = [
                robot.robot_id
                for robot in self.robots
                if robot.enabled and "malicious" in robot.role.lower()
            ]
        return [robot_id for robot_id in ids if robot_id in self.robot_trust]

    def attack_cell_is_eligible(
        self,
        cell: Cell,
        attacker_id: str,
        now_sec: float,
    ) -> bool:
        if not self.is_free_cell(*cell):
            return False

        # Keep fake objects far enough from every enabled robot that agents cannot
        # immediately verify the claimed obstacle using local observation.
        agent_exclusion_radius = max(
            0.0,
            float(
                self.malicious_attack_config.get(
                    "agent_exclusion_radius_cells",
                    5.0,
                )
            ),
        )

        for robot in self.robots:
            if not robot.enabled:
                continue

            robot_cell = self.current_robot_cell(robot)

            if math.dist(cell, robot_cell) <= agent_exclusion_radius:
                return False

        active_goal = self.active_goal_for_robot(
            self.primary_robot.robot_id
        )
        goal_radius = max(
            0.0,
            float(
                self.malicious_attack_config.get(
                    "goal_exclusion_radius_cells",
                    1.0,
                )
            ),
        )

        if (
            active_goal is not None
            and math.dist(cell, active_goal) <= goal_radius
        ):
            return False

        report_range = float(
            self.malicious_attack_config.get(
                "report_range_cells",
                0.0,
            )
        )

        if report_range > 0.0:
            attacker = next(
                (
                    robot
                    for robot in self.robots
                    if robot.robot_id == attacker_id
                ),
                None,
            )

            if attacker is None:
                return False

            if (
                math.dist(
                    cell,
                    self.current_robot_cell(attacker),
                )
                > report_range
            ):
                return False

        cooldown = max(
            0.0,
            float(
                self.malicious_attack_config.get(
                    "cell_cooldown_sec",
                    15.0,
                )
            ),
        )

        last_used = self.attack_recent_cells.get(cell)

        return (
            last_used is None
            or now_sec - last_used >= cooldown
        )

    def score_attack_candidate(self, item: Dict) -> float:
        base_score = float(item.get("score", 0.0))
        policy = str(
            self.malicious_attack_config.get("placement_policy", "topology_only")
        ).strip().lower()
        cell = (int(item["row"]), int(item["col"]))
        on_route = cell in self.agent_routes.get(self.primary_robot.robot_id, [])

        if policy == "topology_only":
            return base_score
        if policy == "route_only":
            return base_score if on_route else -math.inf
        if policy == "route_informed":
            route_multiplier = float(
                self.malicious_attack_config.get("route_score_multiplier", 2.0)
            )
            off_route_multiplier = float(
                self.malicious_attack_config.get("off_route_score_multiplier", 0.25)
            )
            return base_score * (route_multiplier if on_route else off_route_multiplier)
        raise ValueError(
            "placement_policy must be topology_only, route_informed, or route_only"
        )

    def expanded_fake_object_cells(self, center: Cell) -> List[Cell]:
        radius = max(
            0,
            int(self.malicious_attack_config.get("fake_object_radius_cells", 0)),
        )
        candidates: List[Tuple[float, Cell]] = []
        for d_row in range(-radius, radius + 1):
            for d_col in range(-radius, radius + 1):
                cell = (center[0] + d_row, center[1] + d_col)
                distance = math.hypot(d_row, d_col)
                if distance <= radius + 1e-9 and self.is_free_cell(*cell):
                    candidates.append((distance, cell))
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [cell for _, cell in candidates]

    def select_malicious_attack_cells(
        self,
        attacker_id: str,
        now_sec: float,
    ) -> List[Tuple[Cell, Dict]]:
        candidates: List[Tuple[float, Dict]] = []
        for item in self.attack_ranked_cells:
            cell = (int(item["row"]), int(item["col"]))
            if not self.attack_cell_is_eligible(cell, attacker_id, now_sec):
                continue
            score = self.score_attack_candidate(item)
            if math.isfinite(score) and score > 0.0:
                candidates.append((score, item))

        candidates.sort(
            key=lambda value: (-value[0], int(value[1]["row"]), int(value[1]["col"]))
        )
        if not candidates:
            return []

        centers_per_event = max(
            1,
            int(self.malicious_attack_config.get("centers_per_event", 1)),
        )
        max_claims = max(
            1,
            int(self.malicious_attack_config.get("max_claims_per_event", 1)),
        )

        selected: List[Tuple[Cell, Dict]] = []
        used: set[Cell] = set()
        for _, center_item in candidates[:centers_per_event]:
            center = (int(center_item["row"]), int(center_item["col"]))
            for cell in self.expanded_fake_object_cells(center):
                if cell in used:
                    continue
                selected.append((cell, center_item))
                used.add(cell)
                if len(selected) >= max_claims:
                    return selected
        return selected

    def malicious_attack_timer_callback(self) -> None:
        if not self.malicious_attack_enabled() or not self.attack_ranked_cells:
            return

        now_sec = self.now_sec()
        elapsed = now_sec - self.experiment_start_time_sec
        start_delay = max(
            0.0,
            float(self.malicious_attack_config.get("start_delay_sec", 20.0)),
        )
        if elapsed < start_delay:
            return

        max_events = int(self.malicious_attack_config.get("max_attack_events", 0))
        if max_events > 0 and self.attack_event_count >= max_events:
            return

        attackers = self.malicious_robot_ids()
        if not attackers:
            self.get_logger().warn(
                "Malicious attack enabled, but no enabled malicious robot was found."
            )
            return

        confidence = self.clamp01(
            float(self.malicious_attack_config.get("confidence", 0.95))
        )
        event_cells: List[Cell] = []

        for attacker_id in attackers:
            for cell, source_item in self.select_malicious_attack_cells(attacker_id, now_sec):
                payload = {
                    "robot_id": attacker_id,
                    "row": cell[0],
                    "col": cell[1],
                    "report_type": "occupied",
                    "timestamp": now_sec,
                    "confidence": confidence,
                    "metadata": {
                        "attack": True,
                        "attack_type": "heatmap_fake_object",
                        "attack_event": self.attack_event_count,
                        "placement_policy": str(
                            self.malicious_attack_config.get(
                                "placement_policy", "topology_only"
                            )
                        ),
                        "heatmap_score": float(source_item.get("score", 0.0)),
                        "heatmap_center": [
                            int(source_item["row"]),
                            int(source_item["col"]),
                        ],
                    },
                }
                self.add_claim_from_mapping(payload)
                self.attack_recent_cells[cell] = now_sec
                event_cells.append(cell)

        if not event_cells:
            self.get_logger().warn(
                f"Attack event {self.attack_event_count} found no eligible heatmap cells."
            )
            return

        self.active_malicious_cells = list(
            dict.fromkeys([*self.active_malicious_cells, *event_cells])
        )
        completed_event_index = self.attack_event_count
        self.attack_event_count += 1
        self.metric_counters["attack_events"] += 1
        
        
        
        for cell in self.active_malicious_cells:
            ranked = next(
                (item for item in self.attack_ranked_cells
                 if int(item.get("row", -1)) == cell[0]
                 and int(item.get("col", -1)) == cell[1]),
                {},
            )
            self.write_metric_row(
                "attacks",
                {
                    "time_sec": now_sec,
                    "elapsed_sec": elapsed,
                    "event_index": completed_event_index,
                    "attackers_json": json.dumps(attackers),
                    "row": cell[0],
                    "col": cell[1],
                    "confidence": confidence,
                    "placement_policy": self.malicious_attack_config.get(
                        "placement_policy", "topology_only"
                    ),
                    "heatmap_score": ranked.get("score", ""),
                    "active_route_contains_cell": cell in self.agent_routes.get(
                        self.primary_robot.robot_id, []
                    ),
                },
            )
        self.malicious_visual_revision += 1
        self.sync_malicious_object_visuals()

        self.get_logger().info(
            f"Malicious claims added at {event_cells}; forcing immediate replan."
        )
        self.replan(force=True)

        self.get_logger().warn(
            f"Malicious fake-object event={self.attack_event_count - 1}, "
            f"cells={self.active_malicious_cells}, attackers={attackers}"
        )

    def make_malicious_objects_sdf(self) -> str:
        radius = max(
            0.03,
            float(
                self.malicious_attack_config.get(
                    "visual_radius_m", self.cell_size_m * 0.30
                )
            ),
        )
        height = max(
            0.01,
            float(self.malicious_attack_config.get("visual_height_m", 0.12)),
        )
        center_z = max(
            0.0,
            float(self.malicious_attack_config.get("visual_z_m", 0.03)),
        ) + height / 2.0

        visuals: List[str] = []
        for index, (row, col) in enumerate(self.active_malicious_cells):
            x, y = self.cell_to_world(row, col)
            visuals.append(
                "\n".join(
                    [
                        f'        <visual name="fake_obstacle_{index}">',
                        f"          <pose>{x:.6f} {y:.6f} {center_z:.6f} 0 0 0</pose>",
                        "          <cast_shadows>false</cast_shadows>",
                        "          <transparency>0.30</transparency>",
                        "          <geometry>",
                        "            <cylinder>",
                        f"              <radius>{radius:.6f}</radius>",
                        f"              <length>{height:.6f}</length>",
                        "            </cylinder>",
                        "          </geometry>",
                        "          <material>",
                        "            <ambient>0.95 0.02 0.02 0.70</ambient>",
                        "            <diffuse>0.95 0.02 0.02 0.70</diffuse>",
                        "            <emissive>0.40 0.00 0.00 0.70</emissive>",
                        "          </material>",
                        "        </visual>",
                    ]
                )
            )

        return "\n".join(
            [
                '<?xml version="1.0"?>',
                '<sdf version="1.9">',
                '  <model name="trust_malicious_objects">',
                '    <static>true</static>',
                '    <link name="visuals">',
                *visuals,
                '    </link>',
                '  </model>',
                '</sdf>',
                '',
            ]
        )

    def sync_malicious_object_visuals(self) -> None:
        if not bool(self.gazebo_visualization_config.get("enabled", True)):
            return
        if not bool(self.gazebo_visualization_config.get("show_malicious_objects", True)):
            return
        show_malicious = self.malicious_attack_config.get(
            "show_gazebo_visuals",
            self.malicious_attack_config.get("show_visuals", True),
        )
        if not bool(show_malicious):
            return
        if self.malicious_visual_sync_in_progress:
            return
        if self.last_malicious_visual_revision == self.malicious_visual_revision:
            return

        self.malicious_visual_sync_in_progress = True
        try:
            self.remove_gazebo_model("trust_malicious_objects")
            if not self.active_malicious_cells:
                self.last_malicious_visual_revision = self.malicious_visual_revision
                return
            if self.spawn_gazebo_sdf(
                "trust_malicious_objects",
                self.make_malicious_objects_sdf(),
            ):
                self.last_malicious_visual_revision = self.malicious_visual_revision
        finally:
            self.malicious_visual_sync_in_progress = False

    # ------------------------------------------------------------------
    # Claim and trust handling
    # ------------------------------------------------------------------

    def initialize_robot_trust(self) -> Dict[str, float]:
        default_trust = self.clamp01(float(self.trust_config.get("default", 0.5)))
        initial = self.trust_config.get("initial", {})
        if not isinstance(initial, dict):
            initial = {}

        trust: Dict[str, float] = {}
        raw_robots = {str(item.get("id")): item for item in self.scenario.get("robots", [])}
        for robot in self.robots:
            robot_value = raw_robots.get(robot.robot_id, {}).get("initial_trust")
            value = initial.get(robot.robot_id, robot_value)
            trust[robot.robot_id] = self.clamp01(
                default_trust if value is None else float(value)
            )
        return trust

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1_000_000_000.0

    def claim_message_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            self.add_claim_from_mapping(payload)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"Rejected /map_claims message: {exc}")

    def trust_update_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            robot_id = str(payload["robot_id"])
            trust = self.clamp01(float(payload["trust"]))
            if robot_id not in self.robot_trust:
                self.get_logger().warn(f"Trust update introduced unknown robot: {robot_id}")
            old = self.robot_trust.get(robot_id, 0.5)
            self.robot_trust[robot_id] = trust
            self.last_claim_revision += 1
            self.get_logger().info(
                f"Trust update: {robot_id} {old:.3f} -> {trust:.3f}"
            )
            self.log_trust_update(
                robot_id=robot_id,
                old_trust=old,
                new_trust=trust,
                reason=str(payload.get("reason", "external")),
                source_robot_id=str(payload.get("source_robot_id", "")),
                row=payload.get("row", ""),
                col=payload.get("col", ""),
                delta_t_sec=payload.get("delta_t_sec", ""),
            )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"Rejected /trust_updates message: {exc}")

    def add_claim_from_mapping(self, payload: Dict) -> Claim:
        robot_id = str(payload["robot_id"])
        row, col = self.extract_claim_cell(payload)
        report_type = str(payload["report_type"]).strip().lower()
        if report_type not in {"occupied", "free"}:
            raise ValueError("report_type must be 'occupied' or 'free'")
        if not self.in_bounds(row, col):
            raise ValueError(f"Claim cell out of bounds: row={row}, col={col}")

        confidence = self.clamp01(float(payload.get("confidence", 1.0)))
        timestamp_sec = float(payload.get("timestamp", self.now_sec()))
        report_trust = self.clamp01(
            float(payload.get("report_trust", self.robot_trust.get(robot_id, 0.5)))
        )

        claim = Claim(
            robot_id=robot_id,
            row=row,
            col=col,
            report_type=report_type,
            timestamp_sec=timestamp_sec,
            confidence=confidence,
            report_trust=report_trust,
            metadata=dict(payload.get("metadata", {})),
        )
        self.add_claim(claim)
        return claim

    def extract_claim_cell(self, payload: Dict) -> Cell:
        if "row" in payload and "col" in payload:
            return int(payload["row"]), int(payload["col"])
        if "cell_y" in payload and "cell_x" in payload:
            return int(payload["cell_y"]), int(payload["cell_x"])
        if "cell" in payload:
            cell = payload["cell"]
            return int(cell[0]), int(cell[1])
        raise KeyError("Claim requires row/col, cell_y/cell_x, or cell")

    def add_claim(self, claim: Claim) -> None:
        self.claims.append(claim)
        self.claims_by_cell.setdefault((claim.row, claim.col), []).append(claim)
        self.update_fused_log_odds(claim)
        self.maybe_update_trust_from_contradiction(claim)
        self.last_claim_revision += 1
        self.get_logger().info(
            f"Claim: source={claim.robot_id}, cell=({claim.row},{claim.col}), "
            f"type={claim.report_type}, confidence={claim.confidence:.3f}, "
            f"report_trust={claim.report_trust:.3f}"
        )
        self.write_metric_row(
            "claims",
            {
                "time_sec": claim.timestamp_sec,
                "elapsed_sec": claim.timestamp_sec - self.experiment_start_time_sec,
                "robot_id": claim.robot_id,
                "row": claim.row,
                "col": claim.col,
                "report_type": claim.report_type,
                "confidence": claim.confidence,
                "report_trust": claim.report_trust,
                "current_trust": self.robot_trust.get(claim.robot_id, 0.5),
                "is_attack": bool(claim.metadata.get("attack", False)),
                "attack_type": claim.metadata.get("attack_type", ""),
                "attack_event": claim.metadata.get("attack_event", ""),
                "metadata_json": json.dumps(claim.metadata, sort_keys=True),
            },
        )

    def load_scripted_claims(self) -> None:
        raw_claims = self.scenario.get("scripted_claims", [])

        if not raw_claims:
            return

        if not isinstance(raw_claims, list):
            raise ValueError("scripted_claims must be a YAML list.")

        for index, payload in enumerate(raw_claims):
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Scripted claim #{index} must be a YAML mapping."
                )

            try:
                self.add_claim_from_mapping(payload)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid scripted claim #{index}: {exc}"
                ) from exc

    def update_fused_log_odds(self, claim: Claim) -> None:
        occupied_increment = float(self.costmap_config.get("l_occ", 1.0))
        free_increment = float(self.costmap_config.get("l_free", -1.0))
        increment = occupied_increment if claim.report_type == "occupied" else free_increment
        self.fused_log_odds[claim.row][claim.col] += (
            claim.report_trust * claim.confidence * increment
        )

    def maybe_update_trust_from_contradiction(self, new_claim: Claim) -> None:
        if not bool(self.trust_config.get("auto_update_on_contradiction", False)):
            return

        previous = self.claims_by_cell.get((new_claim.row, new_claim.col), [])[:-1]
        opposing = [
            claim
            for claim in previous
            if claim.robot_id != new_claim.robot_id
            and claim.report_type != new_claim.report_type
            and claim.timestamp_sec <= new_claim.timestamp_sec
        ]
        if not opposing:
            return

        old_claim = max(opposing, key=lambda claim: claim.timestamp_sec)
        delta_t = max(0.0, new_claim.timestamp_sec - old_claim.timestamp_sec)
        beta = float(self.trust_config.get("contradiction_beta", 0.25))
        eta = float(self.trust_config.get("contradiction_eta", 0.05))
        time_factor = math.exp(-eta * delta_t)
        current = self.robot_trust.get(old_claim.robot_id, 0.5)
        updated = self.clamp01(current - beta * time_factor * current)
        self.robot_trust[old_claim.robot_id] = updated
        self.get_logger().info(
            f"Contradiction trust update: {old_claim.robot_id} "
            f"{current:.3f} -> {updated:.3f}, delta_t={delta_t:.2f}s"
        )
        self.log_trust_update(
            robot_id=old_claim.robot_id,
            old_trust=current,
            new_trust=updated,
            reason="contradiction",
            source_robot_id=new_claim.robot_id,
            row=new_claim.row,
            col=new_claim.col,
            delta_t_sec=delta_t,
        )

    # ------------------------------------------------------------------
    # External costmap method plugins
    # ------------------------------------------------------------------

    def external_costmap_directories(self) -> List[FilePath]:
        """Return ordered directories searched for experimental costmaps.

        A plugin is an ordinary Python file. It may expose either:

        1. METHOD_NAME and build_costmap(manager, now_sec), or
        2. register_costmap_methods(manager) returning {name: callable}.

        The callable receives the manager and current ROS time and must return
        a map-shaped positive CostGrid with math.inf for blocked cells.
        """
        configured = str(
            self.get_parameter("external_costmap_dir").value
        ).strip()
        scenario_dir = str(
            self.experiment_config.get("external_costmap_dir", "")
        ).strip()

        candidates: List[FilePath] = []
        for raw in (configured, scenario_dir):
            if raw:
                candidates.append(
                    FilePath(os.path.expandvars(os.path.expanduser(raw)))
                )

        candidates.extend(
            [
                FilePath.home()
                / "ros_ws"
                / "src"
                / "trust_costmap"
                / "scripts"
                / "costmap_methods",
                FilePath(self.package_share) / "scripts" / "costmap_methods",
            ]
        )

        unique: List[FilePath] = []
        seen = set()
        for path in candidates:
            resolved = path.resolve() if path.exists() else path
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def validate_external_cost_grid(
        self,
        method_name: str,
        grid: CostGrid,
    ) -> CostGrid:
        if len(grid) != self.map_data.height:
            raise ValueError(
                f"External costmap {method_name!r} returned {len(grid)} rows; "
                f"expected {self.map_data.height}."
            )
        for row_index, row in enumerate(grid):
            if len(row) != self.map_data.width:
                raise ValueError(
                    f"External costmap {method_name!r} row {row_index} has "
                    f"{len(row)} columns; expected {self.map_data.width}."
                )
            for col_index, value in enumerate(row):
                numeric = float(value)
                if math.isnan(numeric) or numeric <= 0.0:
                    raise ValueError(
                        f"External costmap {method_name!r} returned invalid cost "
                        f"{value!r} at ({row_index}, {col_index})."
                    )
        return [[float(value) for value in row] for row in grid]

    def wrap_external_costmap_builder(
        self,
        method_name: str,
        builder: Callable,
    ) -> CostmapBuilder:
        def wrapped(now_sec: float) -> CostGrid:
            grid = builder(self, now_sec)
            return self.validate_external_cost_grid(method_name, grid)

        return wrapped

    def load_external_costmap_builders(self) -> None:
        if not bool(self.get_parameter("enable_external_costmaps").value):
            return

        loaded_count = 0
        for directory in self.external_costmap_directories():
            if not directory.is_dir():
                continue

            for path in sorted(directory.glob("*.py")):
                if path.name.startswith("_"):
                    continue

                module_name = (
                    "trust_costmap_external_"
                    + path.stem.replace("-", "_")
                    + "_"
                    + str(abs(hash(str(path))))
                )
                try:
                    spec = importlib.util.spec_from_file_location(
                        module_name,
                        path,
                    )
                    if spec is None or spec.loader is None:
                        raise ImportError("could not create module specification")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    registrations: Dict[str, Callable] = {}
                    register = getattr(module, "register_costmap_methods", None)
                    if callable(register):
                        result = register(self)
                        if not isinstance(result, dict):
                            raise TypeError(
                                "register_costmap_methods must return a dictionary"
                            )
                        registrations.update(result)
                    else:
                        method_name = str(
                            getattr(module, "METHOD_NAME", path.stem)
                        )
                        builder = getattr(module, "build_costmap", None)
                        if not callable(builder):
                            raise AttributeError(
                                "plugin must define build_costmap or "
                                "register_costmap_methods"
                            )
                        registrations[method_name] = builder

                    allow_override = bool(
                        getattr(module, "ALLOW_OVERRIDE", False)
                    )
                    for raw_name, builder in registrations.items():
                        normalized = self.normalize_method_name(str(raw_name))
                        if normalized in self.costmap_builders and not allow_override:
                            raise ValueError(
                                f"method {normalized!r} already exists; set "
                                "ALLOW_OVERRIDE=True to replace it"
                            )
                        if not callable(builder):
                            raise TypeError(
                                f"builder for {normalized!r} is not callable"
                            )
                        self.costmap_builders[normalized] = (
                            self.wrap_external_costmap_builder(normalized, builder)
                        )
                        self.costmap_method_sources[normalized] = str(path)
                        loaded_count += 1
                        self.get_logger().info(
                            f"Loaded external costmap method {normalized!r} "
                            f"from {path}"
                        )
                except Exception as exc:
                    self.get_logger().error(
                        f"Failed to load external costmap plugin {path}: {exc}\n"
                        f"{traceback.format_exc()}"
                    )

        if loaded_count == 0:
            self.get_logger().info(
                "No external costmap plugins loaded. Built-in baselines remain "
                "available."
            )

    # ------------------------------------------------------------------
    # Costmap strategies
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_method_name(name: str) -> str:
        normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "hard": "hard_threshold",
            "hard_threshold_occupancy_grid": "hard_threshold",
            "soft": "soft_probabilistic",
            "soft_probabilistic_costmap": "soft_probabilistic",
            "decay": "time_decay",
            "time_decay_dynamic_costmap": "time_decay",
            "trust_weighted": "trust_fused",
            "trust_weighted_fused_occupancy_grid": "trust_fused",
            "proposed": "source_linked",
            "source_linked_trust_revisable_cost_surface": "source_linked",
        }
        return aliases.get(normalized, normalized or "static")

    def empty_numeric_grid(self, value: float) -> CostGrid:
        return [
            [float(value) for _ in range(self.map_data.width)]
            for _ in range(self.map_data.height)
        ]

    def base_traversal_grid(self) -> CostGrid:
        free_cost = float(self.costmap_config.get("free_cost", 1.0))
        grid = self.empty_numeric_grid(free_cost)
        for row in range(self.map_data.height):
            for col in range(self.map_data.width):
                if self.is_blocked_cell(row, col):
                    grid[row][col] = BLOCKED_COST
        return grid

    def build_static_costmap(self, now_sec: float) -> CostGrid:
        del now_sec
        return self.base_traversal_grid()

    def build_hard_threshold_costmap(self, now_sec: float) -> CostGrid:
        del now_sec
        threshold = float(self.costmap_config.get("occupancy_threshold", 0.65))
        grid = self.base_traversal_grid()
        for cell, claims in self.claims_by_cell.items():
            row, col = cell
            if self.is_blocked_cell(row, col):
                continue
            probability = self.unweighted_claim_probability(claims)
            grid[row][col] = BLOCKED_COST if probability > threshold else 1.0
        return grid

    def build_soft_probabilistic_costmap(self, now_sec: float) -> CostGrid:
        del now_sec
        grid = self.base_traversal_grid()
        for cell, claims in self.claims_by_cell.items():
            row, col = cell
            if self.is_blocked_cell(row, col):
                continue
            probability = self.unweighted_claim_probability(claims)
            grid[row][col] = self.probability_to_cost(probability)
        return grid

    def build_time_decay_costmap(self, now_sec: float) -> CostGrid:
        gamma = float(self.costmap_config.get("gamma", 0.03))
        grid = self.base_traversal_grid()
        for (row, col), claims in self.claims_by_cell.items():
            if self.is_blocked_cell(row, col):
                continue
            evidence = 0.0
            for claim in claims:
                age = max(0.0, now_sec - claim.timestamp_sec)
                weight = math.exp(-gamma * age) * claim.confidence
                evidence += weight * claim.impact
            grid[row][col] = self.probability_to_cost(self.evidence_probability(evidence))
        return grid

    def build_trust_fused_costmap(self, now_sec: float) -> CostGrid:
        del now_sec
        grid = self.base_traversal_grid()
        for row in range(self.map_data.height):
            for col in range(self.map_data.width):
                if self.is_blocked_cell(row, col):
                    continue
                if (row, col) not in self.claims_by_cell:
                    continue
                probability = self.sigmoid(self.fused_log_odds[row][col])
                grid[row][col] = self.probability_to_cost(probability)
        return grid

    def build_source_linked_costmap(self, now_sec: float) -> CostGrid:
        gamma = float(self.costmap_config.get("gamma", 0.03))
        use_time_decay = bool(self.costmap_config.get("use_time_decay", True))
        use_current_trust = bool(self.costmap_config.get("use_current_trust", True))
        hard_threshold = bool(self.costmap_config.get("proposed_hard_threshold", False))
        threshold = float(self.costmap_config.get("occupancy_threshold", 0.65))

        grid = self.base_traversal_grid()
        for (row, col), claims in self.claims_by_cell.items():
            if self.is_blocked_cell(row, col):
                continue
            evidence = 0.0
            for claim in claims:
                trust = (
                    self.robot_trust.get(claim.robot_id, 0.5)
                    if use_current_trust
                    else claim.report_trust
                )
                age = max(0.0, now_sec - claim.timestamp_sec)
                decay = math.exp(-gamma * age) if use_time_decay else 1.0
                evidence += trust * decay * claim.confidence * claim.impact
            probability = self.evidence_probability(evidence)
            if hard_threshold:
                grid[row][col] = BLOCKED_COST if probability > threshold else 1.0
            else:
                grid[row][col] = self.probability_to_cost(probability)
        return grid

    def unweighted_claim_probability(self, claims: Iterable[Claim]) -> float:
        evidence = sum(claim.confidence * claim.impact for claim in claims)
        return self.evidence_probability(evidence)

    @staticmethod
    def evidence_probability(evidence: float) -> float:
        # No reports means no added occupancy evidence, rather than an arbitrary 0.5.
        if abs(evidence) < 1e-12:
            return 0.0
        return ExperimentManagerNode.sigmoid(evidence)

    def probability_to_cost(self, probability: float) -> float:
        lam = float(self.costmap_config.get("lambda", 10.0))
        exponent = float(self.costmap_config.get("k", 2.0))
        free_cost = float(self.costmap_config.get("free_cost", 1.0))
        return free_cost + lam * (self.clamp01(probability) ** exponent)

    @staticmethod
    def sigmoid(value: float) -> float:
        if value >= 0.0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    @staticmethod
    def clamp01(value: float) -> float:
        return min(1.0, max(0.0, value))

    # ------------------------------------------------------------------
    # Planner strategies
    # ------------------------------------------------------------------

    def plan_astar(self, cost_grid: CostGrid, start: Cell, goal: Cell) -> PlannerResult:
        return self.search_grid(cost_grid, start, goal, use_heuristic=True)

    def plan_dijkstra(self, cost_grid: CostGrid, start: Cell, goal: Cell) -> PlannerResult:
        return self.search_grid(cost_grid, start, goal, use_heuristic=False)

    def search_grid(
        self, cost_grid: CostGrid, start: Cell, goal: Cell, use_heuristic: bool
    ) -> PlannerResult:
        started = time.perf_counter()
        self.validate_planning_endpoint(cost_grid, "start", start)
        self.validate_planning_endpoint(cost_grid, "goal", goal)

        frontier: List[Tuple[float, int, Cell]] = []
        tie_breaker = 0
        heapq.heappush(frontier, (0.0, tie_breaker, start))

        came_from: Dict[Cell, Optional[Cell]] = {start: None}
        cost_so_far: Dict[Cell, float] = {start: 0.0}
        expanded = 0

        while frontier:
            _, _, current = heapq.heappop(frontier)
            expanded += 1
            if current == goal:
                break

            for neighbor, movement_multiplier in self.neighbors(current, cost_grid):
                row, col = neighbor
                cell_cost = cost_grid[row][col]
                new_cost = cost_so_far[current] + movement_multiplier * cell_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heuristic = self.heuristic(neighbor, goal) if use_heuristic else 0.0
                    priority = new_cost + heuristic
                    tie_breaker += 1
                    heapq.heappush(frontier, (priority, tie_breaker, neighbor))
                    came_from[neighbor] = current

        elapsed = time.perf_counter() - started
        if goal not in came_from:
            return PlannerResult([], math.inf, expanded, elapsed)

        path = self.reconstruct_path(came_from, goal)
        return PlannerResult(path, cost_so_far[goal], expanded, elapsed)

    def validate_planning_endpoint(
        self, cost_grid: CostGrid, label: str, cell: Cell
    ) -> None:
        row, col = cell
        if not self.in_bounds(row, col):
            raise ValueError(f"Planning {label} is out of bounds: {cell}")
        if not math.isfinite(cost_grid[row][col]):
            raise ValueError(f"Planning {label} is blocked in the selected costmap: {cell}")

    def neighbors(self, cell: Cell, cost_grid: CostGrid) -> Iterable[Tuple[Cell, float]]:
        row, col = cell
        cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for d_row, d_col in cardinal:
            next_row, next_col = row + d_row, col + d_col
            if self.in_bounds(next_row, next_col) and math.isfinite(
                cost_grid[next_row][next_col]
            ):
                yield (next_row, next_col), 1.0

        if not self.allow_diagonal:
            return

        for d_row, d_col in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            next_row, next_col = row + d_row, col + d_col
            if not self.in_bounds(next_row, next_col):
                continue
            if not math.isfinite(cost_grid[next_row][next_col]):
                continue
            # Prevent diagonal corner cutting through two orthogonal walls.
            if not math.isfinite(cost_grid[row + d_row][col]):
                continue
            if not math.isfinite(cost_grid[row][col + d_col]):
                continue
            yield (next_row, next_col), math.sqrt(2.0)

    def heuristic(self, cell: Cell, goal: Cell) -> float:
        d_row = abs(cell[0] - goal[0])
        d_col = abs(cell[1] - goal[1])
        minimum_cost = float(self.costmap_config.get("free_cost", 1.0))
        if self.allow_diagonal:
            diagonal = min(d_row, d_col)
            straight = max(d_row, d_col) - diagonal
            return minimum_cost * (math.sqrt(2.0) * diagonal + straight)
        return minimum_cost * (d_row + d_col)

    @staticmethod
    def reconstruct_path(
        came_from: Dict[Cell, Optional[Cell]], goal: Cell
    ) -> List[Cell]:
        path = [goal]
        current = goal
        while came_from[current] is not None:
            current = came_from[current]  # type: ignore[assignment]
            path.append(current)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Planning lifecycle and publications
    # ------------------------------------------------------------------

    def replan(self, force: bool = False) -> None:
        """Rebuild routes when claims, goals, or an explicit event requires it.

        The clean baseline is intentionally event-driven. Routes remain stable
        and visible while robots follow them. A new route is created at startup,
        after a costmap revision, and immediately after a new random goal is
        assigned. Future costmap methods continue to use this same lifecycle.
        """

        if self.replan_in_progress:
            return

        claims_changed = (
            self.last_planned_claim_revision
            != self.last_claim_revision
        )
        goals_changed = any(
            self.last_planned_active_goals.get(robot.robot_id)
            != self.active_goal_for_robot(robot.robot_id)
            for robot in self.robots
            if robot.enabled
        )
        always_replan = bool(
            self.planning_config.get("always_replan", False)
        )

        if not (force or claims_changed or goals_changed or always_replan):
            return

        builder = self.costmap_builders.get(self.costmap_method)
        if builder is None:
            available = ", ".join(sorted(self.costmap_builders))
            self.get_logger().error(
                f"Unknown costmap method '{self.costmap_method}'. "
                f"Available: {available}"
            )
            return

        if self.planner_name not in self.planners:
            available = ", ".join(sorted(self.planners))
            self.get_logger().error(
                f"Unknown planner '{self.planner_name}'. "
                f"Available: {available}"
            )
            return

        self.replan_in_progress = True
        try:
            now_sec = self.now_sec()
            cost_grid = builder(now_sec)
            self.plan_agent_routes(cost_grid)

            primary_results = self.agent_route_results.get(
                self.primary_robot.robot_id,
                [],
            )
            result = (
                primary_results[0]
                if primary_results
                else PlannerResult(
                    self.agent_routes.get(
                        self.primary_robot.robot_id,
                        [],
                    ),
                    0.0,
                    0,
                    0.0,
                )
            )

            self.last_cost_grid = cost_grid
            self.last_plan = result
            self.reset_waypoint_progress()
            self.replan_count += 1

            # A* may run frequently, but Gazebo visualization is expensive.
            # Only pause motion and rebuild route models for meaningful events
            # such as startup, a new claim, or a changed goal. Periodic replans
            # update the controller's route without making models flash.
            visualize_periodic = bool(
                self.planning_config.get(
                    "visualize_periodic_replans",
                    False,
                )
            )
            visual_refresh_required = (
                force
                or claims_changed
                or goals_changed
                or visualize_periodic
            )

            if visual_refresh_required:
                self.route_visual_revision += 1
                self.last_gazebo_visual_key = None
                self.kinematic_motion_ready = False
            else:
                self.kinematic_motion_ready = True

            self.last_planned_claim_revision = self.last_claim_revision
            self.last_planned_start_cells = {
                robot.robot_id: self.current_robot_cell(robot)
                for robot in self.robots
                if robot.enabled
            }
            self.last_planned_active_goals = {
                robot.robot_id: self.active_goal_for_robot(robot.robot_id)
                for robot in self.robots
                if robot.enabled
            }

            self.publish_planning_costmap(cost_grid)
            self.publish_planned_path(result.path)
            self.publish_agent_routes()
            self.publish_plan_status(result)
            self.log_plan_metrics(result, force, claims_changed, goals_changed)

            active_route_count = sum(
                1
                for route in self.agent_routes.values()
                if len(route) > 1
            )
            self.get_logger().info(
                f"Plan #{self.replan_count}: planner={self.planner_name}, "
                f"costmap={self.costmap_method}, "
                f"active_routes={active_route_count}"
            )

            if result.found:
                geometric_length = self.path_length_m(result.path)
                self.get_logger().info(
                    f"Primary route: robot={self.primary_robot.robot_id}, "
                    f"cells={len(result.path)}, "
                    f"length={geometric_length:.3f}m, "
                    f"objective={result.total_cost:.3f}, "
                    f"expanded={result.expanded_nodes}, "
                    f"time={result.planning_time_sec * 1000.0:.3f}ms"
                )
            else:
                self.get_logger().warn(
                    f"No primary route found using "
                    f"planner={self.planner_name}, "
                    f"costmap={self.costmap_method}"
                )
        except (ValueError, OverflowError) as exc:
            self.get_logger().error(f"Planning failed: {exc}")
        finally:
            self.replan_in_progress = False

    def path_length_m(self, path: Sequence[Cell]) -> float:
        length_cells = 0.0
        for previous, current in zip(path, path[1:]):
            d_row = current[0] - previous[0]
            d_col = current[1] - previous[1]
            length_cells += math.hypot(d_row, d_col)
        return length_cells * self.cell_size_m

    def publish_visual_debug(self) -> None:
        self.publish_base_map()
        self.publish_robot_markers()
        self.publish_start_goal_markers()
        self.publish_action_goal_markers()
        self.publish_agent_routes()
        if self.last_cost_grid is not None:
            self.publish_planning_costmap(self.last_cost_grid)
        self.publish_planned_path(self.last_plan.path)

    def publish_base_map(self) -> None:
        message = self.make_grid_message()
        message.data = [
            100 if self.is_blocked_cell(row, col) else 0
            for row in range(self.map_data.height)
            for col in range(self.map_data.width)
        ]
        self.base_map_pub.publish(message)

    def publish_planning_costmap(self, cost_grid: CostGrid) -> None:
        message = self.make_grid_message()
        finite_costs = [
            value
            for row in cost_grid
            for value in row
            if math.isfinite(value)
        ]
        base_cost = min(finite_costs, default=1.0)
        max_cost = max(finite_costs, default=base_cost)
        span = max(max_cost - base_cost, 1e-12)

        data: List[int] = []
        for row in cost_grid:
            for value in row:
                if not math.isfinite(value):
                    data.append(100)
                else:
                    scaled = int(round(99.0 * (value - base_cost) / span))
                    data.append(max(0, min(99, scaled)))
        message.data = data
        self.planning_costmap_pub.publish(message)

    def make_grid_message(self) -> OccupancyGrid:
        message = OccupancyGrid()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.info.resolution = self.cell_size_m
        message.info.width = self.map_data.width
        message.info.height = self.map_data.height
        message.info.origin.position.x = 0.0
        message.info.origin.position.y = 0.0
        message.info.origin.position.z = 0.0
        message.info.origin.orientation.w = 1.0
        return message

    def publish_planned_path(self, path: Sequence[Cell]) -> None:
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        for row, col in path:
            x, y = self.cell_to_world(row, col)
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.05
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        self.planned_path_pub.publish(message)

    def publish_plan_status(self, result: PlannerResult) -> None:
        payload = {
            "found": result.found,
            "planner": self.planner_name,
            "costmap_method": self.costmap_method,
            "primary_robot": self.primary_robot.robot_id,
            "path_cells": len(result.path),
            "path_length_m": self.path_length_m(result.path),
            "objective_cost": result.total_cost if math.isfinite(result.total_cost) else None,
            "expanded_nodes": result.expanded_nodes,
            "planning_time_sec": result.planning_time_sec,
            "replan_count": self.replan_count,
            "active_claim_count": len(self.claims),
            "action_goal_count": len(self.action_goals),
            "action_goals": [list(cell) for cell in self.action_goals],
            "agent_route_cells": {
                robot_id: len(route) for robot_id, route in self.agent_routes.items()
            },
            "trust": self.robot_trust,
            "control_enabled": bool(self.control_config.get("enabled", False)),
            "robot_waypoint_index": self.robot_waypoint_index,
            "robot_mission_complete": self.robot_mission_complete,
            "robot_goal_queues": {
                robot_id: [list(cell) for cell in queue]
                for robot_id, queue in self.robot_goal_queues.items()
            },
            "robot_goal_index": self.robot_goal_indices,
            "robot_active_goal": {
                robot_id: list(goal) if goal is not None else None
                for robot_id, goal in self.robot_active_goals.items()
            },
            "robot_completed_goal_visits": self.robot_completed_goal_visits,
            "robot_mission_cycles": self.robot_mission_cycles,
        }
        message = String()
        message.data = json.dumps(payload, sort_keys=True)
        self.plan_status_pub.publish(message)

    # ------------------------------------------------------------------
    # RViz markers
    # ------------------------------------------------------------------

    def marker_color_for_role(self, marker: Marker, role: str) -> None:
        marker.color.a = 1.0
        if "malicious" in role:
            marker.color.r, marker.color.g, marker.color.b = 0.9, 0.05, 0.05
        elif "reporter" in role:
            marker.color.r, marker.color.g, marker.color.b = 0.05, 0.25, 0.9
        else:
            marker.color.r, marker.color.g, marker.color.b = 0.05, 0.75, 0.25

    def publish_robot_markers(self) -> None:
        markers = MarkerArray()
        marker_id = 0
        for robot in self.robots:
            if not robot.enabled:
                continue
            row, col = int(robot.start_cell[0]), int(robot.start_cell[1])
            x, y = self.cell_to_world(row, col)

            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "map"
            marker.ns = "robots"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.15
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.cell_size_m * 0.7
            marker.scale.y = self.cell_size_m * 0.7
            marker.scale.z = 0.3
            self.marker_color_for_role(marker, robot.role)
            markers.markers.append(marker)
            marker_id += 1

            label = Marker()
            label.header = marker.header
            label.ns = "robot_labels"
            label.id = marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = 0.7
            label.pose.orientation.w = 1.0
            label.scale.z = 0.25
            label.color.a = 1.0
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.text = robot.robot_id
            markers.markers.append(label)
            marker_id += 1

        self.robot_markers_pub.publish(markers)

    def publish_action_goal_markers(self) -> None:
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        for marker_id, cell in enumerate(self.action_goals):
            row, col = cell
            x, y = self.cell_to_world(row, col)
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "map"
            marker.ns = "action_goals"
            marker.id = marker_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.12
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.cell_size_m * 0.65
            marker.scale.y = self.cell_size_m * 0.65
            marker.scale.z = 0.24
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.85
            marker.color.b = 0.0
            markers.markers.append(marker)

            label = Marker()
            label.header = marker.header
            label.ns = "action_goal_labels"
            label.id = marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = 0.55
            label.pose.orientation.w = 1.0
            label.scale.z = 0.22
            label.color.a = 1.0
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 0.2
            label.text = f"G{marker_id + 1}"
            markers.markers.append(label)
        self.action_goal_markers_pub.publish(markers)

    def publish_agent_routes(self) -> None:
        route_markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        route_markers.markers.append(clear)
        for marker_id, robot in enumerate(r for r in self.robots if r.enabled):
            route = self.agent_routes.get(robot.robot_id, [])
            message = Path()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "map"
            for row, col in route:
                x, y = self.cell_to_world(row, col)
                pose = PoseStamped()
                pose.header = message.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.08
                pose.pose.orientation.w = 1.0
                message.poses.append(pose)
            publisher = self.agent_route_publishers.get(robot.robot_id)
            if publisher is not None:
                publisher.publish(message)

            marker = Marker()
            marker.header = message.header
            marker.ns = "agent_routes"
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = max(0.025, self.cell_size_m * 0.10)
            self.marker_color_for_role(marker, robot.role)
            for pose in message.poses:
                marker.points.append(pose.pose.position)
            route_markers.markers.append(marker)
        self.agent_route_markers_pub.publish(route_markers)

    def publish_start_goal_markers(self) -> None:
        markers = MarkerArray()
        marker_id = 0
        for robot in self.robots:
            if not robot.enabled:
                continue
            marker_id = self.append_cell_marker(
                markers,
                marker_id,
                "starts",
                robot.start_cell,
                (0.0, 1.0, 0.0),
                0.45,
            )
            if robot.goal_cell is not None:
                marker_id = self.append_cell_marker(
                    markers,
                    marker_id,
                    "goals",
                    robot.goal_cell,
                    (1.0, 0.8, 0.0),
                    0.50,
                )
        self.start_goal_markers_pub.publish(markers)

    def append_cell_marker(
        self,
        markers: MarkerArray,
        marker_id: int,
        namespace: str,
        cell: Sequence[int],
        color: Tuple[float, float, float],
        scale_factor: float,
    ) -> int:
        row, col = int(cell[0]), int(cell[1])
        x, y = self.cell_to_world(row, col)
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.45
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.cell_size_m * scale_factor
        marker.scale.y = self.cell_size_m * scale_factor
        marker.scale.z = self.cell_size_m * scale_factor
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = color
        markers.markers.append(marker)
        return marker_id + 1

    def destroy_node(self) -> bool:
        for robot in (item for item in self.robots if item.enabled):
            self.publish_zero_velocity(robot.robot_id)
        self.finalize_metrics(reason="node_destroy")
        return super().destroy_node()

    # ------------------------------------------------------------------
    # Baseline experiment artifacts and metrics
    # ------------------------------------------------------------------

    @staticmethod
    def safe_identifier(value: object) -> str:
        text = str(value).strip()
        clean = "".join(
            character if character.isalnum() or character in {"-", "_", "."}
            else "_"
            for character in text
        ).strip("._-")
        return clean or "unnamed"

    def effective_experiment_seed(self) -> int:
        if self.experiment_seed >= 0:
            return self.experiment_seed
        return int(self.experiment_config.get("random_seed", 0))

    def elapsed_experiment_sec(self) -> float:
        return max(0.0, self.now_sec() - self.experiment_start_time_sec)

    def configure_experiment_output(self) -> None:
        self.experiment_output_dir = self.experiment_run_directory()
        self.metadata_dir = self.experiment_output_dir / "metadata"
        self.metrics_dir = self.experiment_output_dir / "metrics"
        self.logs_dir = self.experiment_output_dir / "logs"
        for directory in (
            self.experiment_output_dir,
            self.metadata_dir,
            self.metrics_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        environment_run_id = os.environ.get("TRUST_COSTMAP_RUN_ID", "").strip()
        self.run_id = (
            self.run_id_override
            or environment_run_id
            or self.experiment_output_dir.name
        )
        self.run_id = self.safe_identifier(self.run_id)
        self.experiment_manifest_path = self.metadata_dir / "manager_manifest.json"
        self.summary_path = self.metrics_dir / "summary.json"

    def validate_baseline_contract(self) -> None:
        known_methods = {
            "static",
            "hard_threshold",
            "soft_probabilistic",
            "time_decay",
            "trust_fused",
            "source_linked",
        }
        configured_methods = self.baseline_config.get("methods", [])
        if configured_methods:
            unknown = [
                str(method) for method in configured_methods
                if self.normalize_method_name(str(method)) not in known_methods
            ]
            if unknown:
                self.get_logger().warn(
                    "Baseline configuration contains methods not known to the "
                    f"built-in registry: {unknown}. External plugins may still provide them."
                )

        if self.costmap_method == "static" and self.malicious_attack_enabled():
            self.get_logger().warn(
                "Static costmap selected while the malicious attack is enabled. "
                "Claims will be generated and logged but cannot influence routing. "
                "This is valid as an attack-injection control, not as an attack defense."
            )

        if self.malicious_attack_enabled() and not self.attack_reconnaissance_config_enabled():
            self.get_logger().warn(
                "Runtime attack is enabled while attack_reconnaissance.enabled is false. "
                "An explicitly supplied heatmap can still make this valid."
            )

    def attack_reconnaissance_config_enabled(self) -> bool:
        config = self.scenario.get("attack_reconnaissance", {})
        return bool(config.get("enabled", False)) if isinstance(config, dict) else False

    def initialize_metrics_pipeline(self) -> None:
        if not self.metrics_enabled:
            self.get_logger().warn("Metrics pipeline disabled by parameter.")
            return

        schemas: Dict[str, List[str]] = {
            "plans": [
                "run_id", "time_sec", "elapsed_sec", "plan_index",
                "robot_id", "planner", "costmap_method", "found",
                "path_cells", "path_length_m", "objective_cost",
                "expanded_nodes", "planning_time_sec", "active_claims",
                "claim_revision", "route_revision", "forced",
                "claims_changed", "goals_changed", "start_row", "start_col",
                "goal_row", "goal_col",
            ],
            "claims": [
                "run_id", "time_sec", "elapsed_sec", "robot_id", "row", "col",
                "report_type", "confidence", "report_trust", "current_trust",
                "is_attack", "attack_type", "attack_event", "metadata_json",
            ],
            "trust": [
                "run_id", "time_sec", "elapsed_sec", "robot_id", "old_trust",
                "new_trust", "reason", "source_robot_id", "row", "col",
                "delta_t_sec",
            ],
            "attacks": [
                "run_id", "time_sec", "elapsed_sec", "event_index",
                "attackers_json", "row", "col", "confidence",
                "placement_policy", "heatmap_score",
                "active_route_contains_cell",
            ],
            "trajectories": [
                "run_id", "time_sec", "elapsed_sec", "robot_id", "x", "y",
                "yaw", "row", "col", "linear_velocity", "angular_velocity",
                "active_goal_row", "active_goal_col", "waypoint_index",
                "mission_complete",
            ],
            "goals": [
                "run_id", "time_sec", "elapsed_sec", "robot_id", "event",
                "goal_row", "goal_col", "visit_number", "next_goal_row",
                "next_goal_col",
            ],
        }

        for name, fieldnames in schemas.items():
            path = self.metrics_dir / f"{name}.csv"
            file = path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            self.metrics_paths[name] = path
            self.metrics_files[name] = file
            self.metrics_writers[name] = writer

    def write_metric_row(self, stream: str, row: Dict[str, Any]) -> None:
        if not self.metrics_enabled or self.metrics_finalized:
            return
        writer = self.metrics_writers.get(stream)
        if writer is None:
            return
        payload = {"run_id": self.run_id, **row}
        writer.writerow(payload)
        counter_name = {
            "plans": "plans",
            "claims": "claims",
            "trust": "trust_updates",
            "trajectories": "trajectory_samples",
            "goals": "goal_events",
        }.get(stream)
        if counter_name:
            self.metric_counters[counter_name] += 1

    def flush_metric_files(self) -> None:
        if not self.metrics_enabled or self.metrics_finalized:
            return
        for file in self.metrics_files.values():
            file.flush()

    def log_trust_update(
        self,
        *,
        robot_id: str,
        old_trust: float,
        new_trust: float,
        reason: str,
        source_robot_id: str = "",
        row: object = "",
        col: object = "",
        delta_t_sec: object = "",
    ) -> None:
        self.write_metric_row(
            "trust",
            {
                "time_sec": self.now_sec(),
                "elapsed_sec": self.elapsed_experiment_sec(),
                "robot_id": robot_id,
                "old_trust": old_trust,
                "new_trust": new_trust,
                "reason": reason,
                "source_robot_id": source_robot_id,
                "row": row,
                "col": col,
                "delta_t_sec": delta_t_sec,
            },
        )

    def log_plan_metrics(
        self,
        result: PlannerResult,
        forced: bool,
        claims_changed: bool,
        goals_changed: bool,
    ) -> None:
        robot_id = self.primary_robot.robot_id
        start = self.current_robot_cell(self.primary_robot)
        goal = self.active_goal_for_robot(robot_id)
        self.write_metric_row(
            "plans",
            {
                "time_sec": self.now_sec(),
                "elapsed_sec": self.elapsed_experiment_sec(),
                "plan_index": self.replan_count,
                "robot_id": robot_id,
                "planner": self.planner_name,
                "costmap_method": self.costmap_method,
                "found": result.found,
                "path_cells": len(result.path),
                "path_length_m": self.path_length_m(result.path),
                "objective_cost": result.total_cost,
                "expanded_nodes": result.expanded_nodes,
                "planning_time_sec": result.planning_time_sec,
                "active_claims": len(self.claims),
                "claim_revision": self.last_claim_revision,
                "route_revision": self.route_revision,
                "forced": forced,
                "claims_changed": claims_changed,
                "goals_changed": goals_changed,
                "start_row": start[0],
                "start_col": start[1],
                "goal_row": goal[0] if goal is not None else "",
                "goal_col": goal[1] if goal is not None else "",
            },
        )

    def log_trajectory_samples(self) -> None:
        if not self.metrics_enabled or self.metrics_finalized:
            return
        now_sec = self.now_sec()
        elapsed = max(0.0, now_sec - self.experiment_start_time_sec)
        for robot in (item for item in self.robots if item.enabled):
            pose = self.robot_pose.get(robot.robot_id)
            if pose is None:
                continue
            velocity = self.robot_velocity.get(robot.robot_id, (0.0, 0.0))
            cell = self.world_to_cell(pose[0], pose[1])
            goal = self.active_goal_for_robot(robot.robot_id)
            self.write_metric_row(
                "trajectories",
                {
                    "time_sec": now_sec,
                    "elapsed_sec": elapsed,
                    "robot_id": robot.robot_id,
                    "x": pose[0],
                    "y": pose[1],
                    "yaw": pose[2],
                    "row": cell[0],
                    "col": cell[1],
                    "linear_velocity": velocity[0],
                    "angular_velocity": velocity[1],
                    "active_goal_row": goal[0] if goal is not None else "",
                    "active_goal_col": goal[1] if goal is not None else "",
                    "waypoint_index": self.robot_waypoint_index.get(robot.robot_id, ""),
                    "mission_complete": self.robot_mission_complete.get(
                        robot.robot_id, False
                    ),
                },
            )

    def build_final_summary(self, reason: str) -> Dict[str, Any]:
        elapsed = self.elapsed_experiment_sec()
        primary_id = self.primary_robot.robot_id
        successful_plans = sum(
            1 for results in self.agent_route_results.values()
            for result in results if result.found
        )
        return {
            "schema_version": 2,
            "run_id": self.run_id,
            "finalized_utc": datetime.now(timezone.utc).isoformat(),
            "finalization_reason": reason,
            "scenario_name": self.scenario.get("scenario_name", "unnamed_scenario"),
            "map_name": self.map_name,
            "costmap_method": self.costmap_method,
            "planner": self.planner_name,
            "experiment_seed": self.effective_experiment_seed(),
            "trial_index": self.trial_index,
            "baseline_profile": self.baseline_profile_name,
            "baseline_suite_file": self.baseline_suite_file,
            "elapsed_sec": elapsed,
            "attack_enabled": self.malicious_attack_enabled(),
            "attack_event_count": self.attack_event_count,
            "claim_count": len(self.claims),
            "replan_count": self.replan_count,
            "successful_current_routes": successful_plans,
            "primary_robot": primary_id,
            "primary_completed_goal_visits": self.robot_completed_goal_visits.get(
                primary_id, 0
            ),
            "completed_goal_visits": dict(self.robot_completed_goal_visits),
            "mission_cycles": dict(self.robot_mission_cycles),
            "final_trust": dict(self.robot_trust),
            "metric_counters": dict(self.metric_counters),
            "metric_files": {
                name: str(path) for name, path in self.metrics_paths.items()
            },
            "limitations": [
                "Collision metrics require a collision/contact source not yet integrated.",
                "Ground-truth costmap error requires a physical obstacle schedule or oracle.",
                "Recovery-time aggregation should be computed by the offline baseline analyzer.",
            ],
        }

    def finalize_metrics(self, reason: str) -> None:
        if self.metrics_finalized:
            return
        self.metrics_finalized = True
        for file in self.metrics_files.values():
            try:
                file.flush()
                file.close()
            except OSError:
                pass
        summary = self.build_final_summary(reason)
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.summary_path)
        self.get_logger().info(f"Final experiment summary: {self.summary_path}")

    # ------------------------------------------------------------------
    # Experiment provenance
    # ------------------------------------------------------------------

    def experiment_run_directory(self) -> FilePath:
        explicit = self.run_dir_override or os.environ.get(
            "TRUST_COSTMAP_RUN_DIR", ""
        ).strip()
        if explicit:
            return FilePath(os.path.expandvars(os.path.expanduser(explicit)))

        configured = str(
            self.evaluation_config.get(
                "output_dir",
                self.logging_config.get("results_dir", ""),
            )
        ).strip()
        root = (
            FilePath(os.path.expandvars(os.path.expanduser(configured)))
            if configured
            else FilePath.home() / ".ros" / "trust_costmap" / "runs"
        )
        scenario_name = self.safe_identifier(
            self.scenario.get("scenario_name", "unnamed_scenario")
        )
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return root / (
            f"{stamp}_{scenario_name}_{self.costmap_method}"
            f"_seed-{self.effective_experiment_seed()}"
            f"_trial-{self.trial_index}"
        )

    def write_experiment_manifest(self) -> None:
        """Persist enough provenance to reproduce this run later.

        This is intentionally a manifest, not the final metrics pipeline. It
        records the selected baseline, planner, seeds, map, attack heatmap, and
        configuration before motion starts. Future result writers can append
        route, trust, recovery, and collision metrics into the same directory.
        """
        heatmap_path = (
            str(self.attack_heatmap_path)
            if self.attack_heatmap_path is not None
            else ""
        )
        payload = {
            "schema_version": 2,
            "run_id": self.run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_directory": str(self.experiment_output_dir),
            "metrics_directory": str(self.metrics_dir),
            "trial_index": self.trial_index,
            "experiment_seed": self.effective_experiment_seed(),
            "baseline_profile": self.baseline_profile_name,
            "baseline_suite_file": self.baseline_suite_file,
            "baseline_profile_payload": self.baseline_profile_payload,
            "scenario_name": self.scenario.get(
                "scenario_name", "unnamed_scenario"
            ),
            "map_name": self.map_name,
            "map_width": self.map_data.width,
            "map_height": self.map_data.height,
            "costmap_method": self.costmap_method,
            "costmap_method_source": self.costmap_method_sources.get(
                self.costmap_method, "unknown"
            ),
            "planner": self.planner_name,
            "allow_diagonal": self.allow_diagonal,
            "action_goal_count": len(self.action_goals),
            "action_goal_seed": self.action_goal_seed(),
            "action_goals": [list(cell) for cell in self.action_goals],
            "attack_enabled": self.malicious_attack_enabled(),
            "attack_heatmap_required": self.require_attack_heatmap,
            "attack_heatmap_path": heatmap_path,
            "attack_ranked_cell_count": len(self.attack_ranked_cells),
            "robots": [
                {
                    "id": robot.robot_id,
                    "role": robot.role,
                    "enabled": robot.enabled,
                    "start_cell": robot.start_cell,
                    "goal_cell": robot.goal_cell,
                    "initial_trust": self.robot_trust.get(robot.robot_id, 0.5),
                }
                for robot in self.robots
            ],
            "available_costmap_methods": {
                name: self.costmap_method_sources.get(name, "unknown")
                for name in sorted(self.costmap_builders)
            },
            "costmap_config": self.costmap_config,
            "trust_config": self.trust_config,
            "planning_config": self.planning_config,
            "malicious_attack_config": self.malicious_attack_config,
            "baseline_config": self.baseline_config,
            "logging_config": self.logging_config,
            "termination_config": self.termination_config,
            "metric_files": {
                name: str(path) for name, path in self.metrics_paths.items()
            },
        }
        self.experiment_manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.get_logger().info(
            f"Experiment manifest: {self.experiment_manifest_path}"
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_startup_summary(self) -> None:
        scenario_name = self.scenario.get("scenario_name", "unnamed_scenario")
        total_count = self.map_data.width * self.map_data.height

        self.get_logger().info("")
        self.get_logger().info("========== Experiment Summary ==========")
        self.get_logger().info(f"Scenario: {scenario_name}")
        self.get_logger().info(f"Run ID: {self.run_id}")
        self.get_logger().info(f"Trial index: {self.trial_index}")
        self.get_logger().info(
            f"Baseline profile: {self.baseline_profile_name or '(scenario only)'}"
        )
        self.get_logger().info(
            f"Experiment seed: {self.effective_experiment_seed()}"
        )
        self.get_logger().info(f"Costmap method: {self.costmap_method}")
        self.get_logger().info(
            "Costmap source: "
            + self.costmap_method_sources.get(self.costmap_method, "unknown")
        )
        self.get_logger().info(f"Planner: {self.planner_name}")
        self.get_logger().info(f"Diagonal motion: {self.allow_diagonal}")
        self.get_logger().info(f"Primary robot: {self.primary_robot.robot_id}")
        self.get_logger().info(f"Dynamic action goals: {len(self.action_goals)}")
        self.get_logger().info(
            f"Malicious attack enabled: {self.malicious_attack_enabled()}"
        )
        self.get_logger().info(
            f"Attack heatmap required: {self.require_attack_heatmap}"
        )
        if self.attack_heatmap_path is not None:
            self.get_logger().info(f"Attack heatmap: {self.attack_heatmap_path}")
            self.get_logger().info(
                "Malicious attackers: " + str(self.malicious_robot_ids())
            )
        self.get_logger().info(
            "Action goal cells: " + str([list(cell) for cell in self.action_goals])
        )
        self.get_logger().info("")
        self.get_logger().info("Map:")
        self.get_logger().info(f"  name: {self.map_data.name}")
        self.get_logger().info(f"  type: {self.map_data.map_type}")
        self.get_logger().info(
            f"  size: {self.map_data.width} x {self.map_data.height}"
        )
        self.get_logger().info(f"  total cells: {total_count}")
        self.get_logger().info(f"  free cells: {self.map_data.count_free()}")
        self.get_logger().info(f"  blocked cells: {self.map_data.count_blocked()}")
        self.get_logger().info(f"  cell size: {self.cell_size_m} m")
        self.get_logger().info("")
        self.get_logger().info("Robots:")
        for robot in self.robots:
            self.get_logger().info(
                f"  {robot.robot_id}: role={robot.role}, model={robot.model}, "
                f"start_cell={robot.start_cell}, goal_cell={robot.goal_cell}, "
                f"enabled={robot.enabled}, trust={self.robot_trust.get(robot.robot_id, 0.5):.3f}"
            )
        self.get_logger().info("")
        self.get_logger().info(
            "Available costmaps: " + ", ".join(sorted(self.costmap_builders))
        )
        self.get_logger().info(
            f"Experiment output: {self.experiment_output_dir}"
        )
        self.get_logger().info(
            f"Metrics enabled: {self.metrics_enabled}; directory={self.metrics_dir}"
        )
        self.get_logger().info(
            "Available planners: " + ", ".join(sorted(self.planners))
        )
        self.get_logger().info("========================================")
        self.get_logger().info("")


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
