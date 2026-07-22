#!/usr/bin/env python3
"""Shared multi-robot LiDAR occupancy mapping and RViz visualization.

Displayed cells come only from raycast observations (Gazebo LaserScan and/or an
immediate MovingAI map raycast used when GPU LiDAR is too slow). The static map
is never drawn wholesale into RViz.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose2D, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from .shared_mapping_utils import (
    bresenham_cells,
    grid_to_world,
    heatmap_rgba,
    load_movingai_occupancy,
    normalize_angle,
    probability_from_log_odds,
    quaternion_to_yaw,
    raycast_lidar_ranges,
    world_to_grid,
)


ROBOT_COLORS: Sequence[Tuple[float, float, float]] = (
    (1.00, 0.18, 0.12),
    (0.10, 0.95, 0.25),
    (1.00, 0.75, 0.05),
    (0.10, 0.90, 1.00),
    (1.00, 0.20, 0.85),
    (0.95, 0.95, 0.95),
    (0.55, 1.00, 0.10),
    (1.00, 0.45, 0.05),
)


@dataclass
class PoseEstimate:
    x: float
    y: float
    yaw: float
    stamp_ns: int = 0
    source: str = "scenario"


@dataclass
class RobotState:
    robot_id: str
    initial_pose: PoseEstimate
    local_log_odds: np.ndarray
    local_observed: np.ndarray
    current_log_odds: np.ndarray
    current_observed: np.ndarray
    pose: PoseEstimate
    first_odom: Optional[PoseEstimate] = None
    map_pose_subscription_created: bool = False
    map_pose_type: str = ""
    latest_hit_points: List[Tuple[float, float]] = field(default_factory=list)
    scan_count: int = 0
    last_real_scan_ns: int = 0
    last_scan_source: str = ""


class SharedLidarMapper(Node):
    def __init__(self) -> None:
        super().__init__("shared_lidar_mapper")

        self.declare_parameter("robot_ids", ["benign_1", "benign_2", "malicious_1"])
        self.declare_parameter("map_path", "")
        self.declare_parameter("scenario_path", "")
        self.declare_parameter("cell_size_m", 0.5)
        self.declare_parameter("mapping_resolution_m", 0.10)
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("max_mapping_range_m", 3.5)
        self.declare_parameter("free_log_odds_delta", -0.40)
        self.declare_parameter("occupied_log_odds_delta", 0.85)
        self.declare_parameter("min_log_odds", -4.0)
        self.declare_parameter("max_log_odds", 4.0)
        self.declare_parameter("hit_epsilon_m", 0.04)
        self.declare_parameter("scan_stride", 1)
        self.declare_parameter("footprint_size_m", 0.34)
        self.declare_parameter("map_frame", "map")
        # Gazebo gpu_lidar on software-rendered VMs can take minutes; raycast the
        # MovingAI walls immediately so RViz fills as soon as poses are known.
        self.declare_parameter("simulate_lidar_from_map", True)
        self.declare_parameter("simulated_lidar_beams", 72)
        self.declare_parameter("simulated_lidar_hz", 5.0)
        self.declare_parameter("prefer_real_scan_for_sec", 1.0)

        self.robot_ids = [str(value) for value in self.get_parameter("robot_ids").value]
        self.map_path = str(self.get_parameter("map_path").value)
        self.scenario_path = str(self.get_parameter("scenario_path").value)
        self.cell_size = float(self.get_parameter("cell_size_m").value)
        self.resolution = float(self.get_parameter("mapping_resolution_m").value)
        self.max_mapping_range = float(self.get_parameter("max_mapping_range_m").value)
        self.free_delta = float(self.get_parameter("free_log_odds_delta").value)
        self.occupied_delta = float(self.get_parameter("occupied_log_odds_delta").value)
        self.min_log_odds = float(self.get_parameter("min_log_odds").value)
        self.max_log_odds = float(self.get_parameter("max_log_odds").value)
        self.hit_epsilon = float(self.get_parameter("hit_epsilon_m").value)
        self.scan_stride = max(1, int(self.get_parameter("scan_stride").value))
        self.footprint_size = float(self.get_parameter("footprint_size_m").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.simulate_lidar_from_map = bool(
            self.get_parameter("simulate_lidar_from_map").value
        )
        self.simulated_lidar_beams = max(
            8, int(self.get_parameter("simulated_lidar_beams").value)
        )
        self.simulated_lidar_hz = max(
            0.5, float(self.get_parameter("simulated_lidar_hz").value)
        )
        self.prefer_real_scan_for_sec = max(
            0.0, float(self.get_parameter("prefer_real_scan_for_sec").value)
        )

        if self.resolution <= 0.0:
            raise ValueError("mapping_resolution_m must be positive")

        map_width_cells, map_height_cells = self._load_map_dimensions(self.map_path)
        self.world_width = map_width_cells * self.cell_size
        self.world_height = map_height_cells * self.cell_size
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.grid_width = int(math.ceil(self.world_width / self.resolution))
        self.grid_height = int(math.ceil(self.world_height / self.resolution))
        self.grid_shape = (self.grid_height, self.grid_width)
        self.map_occupied: Optional[np.ndarray] = None
        if self.simulate_lidar_from_map:
            self.map_occupied, _, _ = load_movingai_occupancy(self.map_path)

        starts = self._load_scenario_starts(self.scenario_path)
        self.states: Dict[str, RobotState] = {}
        for robot_id in self.robot_ids:
            initial = starts.get(robot_id, PoseEstimate(0.0, 0.0, 0.0))
            self.states[robot_id] = RobotState(
                robot_id=robot_id,
                initial_pose=initial,
                local_log_odds=np.zeros(self.grid_shape, dtype=np.float32),
                local_observed=np.zeros(self.grid_shape, dtype=np.uint16),
                current_log_odds=np.zeros(self.grid_shape, dtype=np.float32),
                current_observed=np.zeros(self.grid_shape, dtype=np.uint16),
                pose=PoseEstimate(initial.x, initial.y, initial.yaw, source="scenario"),
            )

        # Match ros_gz_bridge LaserScan/Odometry publishers (RELIABLE/VOLATILE).
        sensor_qos = QoSProfile(depth=8)
        sensor_qos.reliability = ReliabilityPolicy.RELIABLE
        sensor_qos.durability = DurabilityPolicy.VOLATILE

        # Use VOLATILE for RViz displays. Transient-local MarkerArray publishers
        # often fail to match RViz subscriptions on Jazzy, leaving only /planned_path.
        viz_qos = QoSProfile(depth=5)
        viz_qos.reliability = ReliabilityPolicy.RELIABLE
        viz_qos.durability = DurabilityPolicy.VOLATILE

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.heatmap_pub = self.create_publisher(
            MarkerArray, "/shared_lidar/probability_markers", viz_qos
        )
        self.endpoint_pub = self.create_publisher(
            MarkerArray, "/shared_lidar/scan_endpoints", viz_qos
        )
        self.footprint_pub = self.create_publisher(
            MarkerArray, "/shared_lidar/robot_footprints", viz_qos
        )
        self.heatmap_cloud_pub = self.create_publisher(
            PointCloud2, "/shared_lidar/probability_cloud", viz_qos
        )
        self._occupancy_publish_counter = 0

        self.map_publishers: Dict[str, Dict[str, object]] = {}
        self.subscriptions_kept_alive: List[object] = []
        for robot_id in self.robot_ids:
            self.map_publishers[robot_id] = {
                "current": self.create_publisher(
                    OccupancyGrid, f"/{robot_id}/current_observation_map", map_qos
                ),
                "local": self.create_publisher(
                    OccupancyGrid, f"/{robot_id}/local_map", map_qos
                ),
                "shared": self.create_publisher(
                    OccupancyGrid, f"/{robot_id}/shared_map", map_qos
                ),
                "combined": self.create_publisher(
                    OccupancyGrid, f"/{robot_id}/combined_map", map_qos
                ),
            }
            self.subscriptions_kept_alive.append(
                self.create_subscription(
                    LaserScan,
                    f"/{robot_id}/scan",
                    lambda msg, rid=robot_id: self._scan_callback(rid, msg),
                    sensor_qos,
                )
            )
            self.subscriptions_kept_alive.append(
                self.create_subscription(
                    Odometry,
                    f"/{robot_id}/odom",
                    lambda msg, rid=robot_id: self._odom_callback(rid, msg),
                    sensor_qos,
                )
            )

        publish_rate = max(0.2, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / publish_rate, self._publish_all)
        self.create_timer(1.0, self._discover_map_pose_topics)
        if self.simulate_lidar_from_map and self.map_occupied is not None:
            self.create_timer(1.0 / self.simulated_lidar_hz, self._simulate_lidar_tick)
            self.get_logger().info(
                "Immediate map-raycast LiDAR enabled "
                f"({self.simulated_lidar_beams} beams @ {self.simulated_lidar_hz:.1f} Hz). "
                "Gazebo scans are still used when they arrive."
            )

        self.get_logger().info(
            "Shared LiDAR mapper ready: robots=%s grid=%dx%d resolution=%.3fm "
            "world=%.2fx%.2fm. Only raycast-observed cells are visualized."
            % (
                ",".join(self.robot_ids),
                self.grid_width,
                self.grid_height,
                self.resolution,
                self.world_width,
                self.world_height,
            )
        )

    @staticmethod
    def _load_map_dimensions(map_path: str) -> Tuple[int, int]:
        if not map_path or not os.path.exists(map_path):
            raise FileNotFoundError(f"MovingAI map not found: {map_path}")
        width = None
        height = None
        with open(map_path, "r", encoding="utf-8") as handle:
            for line in handle:
                clean = line.strip()
                if clean.startswith("width "):
                    width = int(clean.split()[1])
                elif clean.startswith("height "):
                    height = int(clean.split()[1])
                elif clean == "map":
                    break
        if width is None or height is None:
            raise RuntimeError(f"Invalid MovingAI map metadata: {map_path}")
        return width, height

    def _load_scenario_starts(self, scenario_path: str) -> Dict[str, PoseEstimate]:
        result: Dict[str, PoseEstimate] = {}
        if not scenario_path or not os.path.exists(scenario_path):
            self.get_logger().warning(
                f"Scenario file not found; pose anchoring will rely on odometry: {scenario_path}"
            )
            return result
        with open(scenario_path, "r", encoding="utf-8") as handle:
            scenario = yaml.safe_load(handle) or {}
        scenario_cell_size = float(
            (scenario.get("visualization") or {}).get("cell_size_m", self.cell_size)
        )
        for robot in scenario.get("robots", []):
            if not robot.get("enabled", True):
                continue
            robot_id = str(robot.get("id", ""))
            start = robot.get("start_cell")
            if not robot_id or not isinstance(start, (list, tuple)) or len(start) < 2:
                continue
            row = int(start[0])
            col = int(start[1])
            yaw = float(robot.get("start_yaw", robot.get("yaw", 0.0)))
            result[robot_id] = PoseEstimate(
                x=(col + 0.5) * scenario_cell_size,
                y=(row + 0.5) * scenario_cell_size,
                yaw=yaw,
                source="scenario",
            )
        return result

    def _discover_map_pose_topics(self) -> None:
        """Subscribe to /<robot>/map_pose using whichever supported type is present."""
        topic_types = dict(self.get_topic_names_and_types())
        for robot_id, state in self.states.items():
            if state.map_pose_subscription_created:
                continue
            topic = f"/{robot_id}/map_pose"
            types = topic_types.get(topic, [])
            if not types:
                continue
            chosen = types[0]
            if "geometry_msgs/msg/PoseStamped" in types:
                chosen = "geometry_msgs/msg/PoseStamped"
                msg_type = PoseStamped
                callback = lambda msg, rid=robot_id: self._pose_stamped_callback(rid, msg)
            elif "geometry_msgs/msg/Pose2D" in types:
                chosen = "geometry_msgs/msg/Pose2D"
                msg_type = Pose2D
                callback = lambda msg, rid=robot_id: self._pose2d_callback(rid, msg)
            elif "nav_msgs/msg/Odometry" in types:
                chosen = "nav_msgs/msg/Odometry"
                msg_type = Odometry
                callback = lambda msg, rid=robot_id: self._map_odom_callback(rid, msg)
            else:
                self.get_logger().warning(
                    f"Unsupported map_pose type on {topic}: {', '.join(types)}; using odom fallback"
                )
                state.map_pose_subscription_created = True
                state.map_pose_type = "unsupported"
                continue
            qos = QoSProfile(depth=10)
            qos.reliability = ReliabilityPolicy.RELIABLE
            self.subscriptions_kept_alive.append(
                self.create_subscription(msg_type, topic, callback, qos)
            )
            state.map_pose_subscription_created = True
            state.map_pose_type = chosen
            self.get_logger().info(f"Using {topic} ({chosen}) for global LiDAR placement")

    def _pose_stamped_callback(self, robot_id: str, msg: PoseStamped) -> None:
        yaw = quaternion_to_yaw(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        self.states[robot_id].pose = PoseEstimate(
            msg.pose.position.x,
            msg.pose.position.y,
            yaw,
            self.get_clock().now().nanoseconds,
            "map_pose",
        )

    def _pose2d_callback(self, robot_id: str, msg: Pose2D) -> None:
        self.states[robot_id].pose = PoseEstimate(
            msg.x,
            msg.y,
            msg.theta,
            self.get_clock().now().nanoseconds,
            "map_pose",
        )

    def _map_odom_callback(self, robot_id: str, msg: Odometry) -> None:
        pose = msg.pose.pose
        yaw = quaternion_to_yaw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        self.states[robot_id].pose = PoseEstimate(
            pose.position.x,
            pose.position.y,
            yaw,
            self.get_clock().now().nanoseconds,
            "map_pose",
        )

    def _odom_callback(self, robot_id: str, msg: Odometry) -> None:
        state = self.states[robot_id]
        # Do not overwrite a functioning map_pose stream with odometry.
        if state.pose.source == "map_pose":
            return
        pose = msg.pose.pose
        raw = PoseEstimate(
            pose.position.x,
            pose.position.y,
            quaternion_to_yaw(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ),
            self.get_clock().now().nanoseconds,
            "odom",
        )
        if state.first_odom is None:
            state.first_odom = raw

        first = state.first_odom
        dx = raw.x - first.x
        dy = raw.y - first.y
        rotation = state.initial_pose.yaw - first.yaw
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        map_dx = cos_r * dx - sin_r * dy
        map_dy = sin_r * dx + cos_r * dy
        state.pose = PoseEstimate(
            state.initial_pose.x + map_dx,
            state.initial_pose.y + map_dy,
            normalize_angle(state.initial_pose.yaw + raw.yaw - first.yaw),
            raw.stamp_ns,
            "odom",
        )

    def _inside(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.grid_width and 0 <= gy < self.grid_height

    def _apply_delta(
        self,
        log_odds: np.ndarray,
        observed: np.ndarray,
        gx: int,
        gy: int,
        delta: float,
    ) -> None:
        if not self._inside(gx, gy):
            return
        value = float(log_odds[gy, gx]) + delta
        log_odds[gy, gx] = min(self.max_log_odds, max(self.min_log_odds, value))
        if observed[gy, gx] < np.iinfo(np.uint16).max:
            observed[gy, gx] += 1

    def _scan_callback(self, robot_id: str, msg: LaserScan) -> None:
        state = self.states[robot_id]
        pose = state.pose
        if not (math.isfinite(pose.x) and math.isfinite(pose.y) and math.isfinite(pose.yaw)):
            return

        usable_max = self.max_mapping_range
        if math.isfinite(msg.range_max) and msg.range_max > 0.0:
            usable_max = min(usable_max, float(msg.range_max))
        usable_min = max(0.0, float(msg.range_min))
        angle_min = float(msg.angle_min)
        angle_increment = float(msg.angle_increment)
        ranges = [float(value) for value in msg.ranges]
        self._integrate_scan_ranges(
            robot_id,
            ranges,
            angle_min=angle_min,
            angle_increment=angle_increment,
            usable_min=usable_min,
            usable_max=usable_max,
            sensor_range_max=float(msg.range_max) if math.isfinite(msg.range_max) else usable_max,
            source="gazebo",
        )
        state.last_real_scan_ns = self.get_clock().now().nanoseconds

    def _simulate_lidar_tick(self) -> None:
        if self.map_occupied is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        for robot_id, state in self.states.items():
            if state.last_real_scan_ns > 0:
                age_sec = (now_ns - state.last_real_scan_ns) * 1e-9
                if age_sec <= self.prefer_real_scan_for_sec:
                    continue
            pose = state.pose
            if not (
                math.isfinite(pose.x) and math.isfinite(pose.y) and math.isfinite(pose.yaw)
            ):
                continue
            ranges, angle_increment = raycast_lidar_ranges(
                self.map_occupied,
                self.cell_size,
                self.origin_x,
                self.origin_y,
                pose.x,
                pose.y,
                pose.yaw,
                self.max_mapping_range,
                beam_count=self.simulated_lidar_beams,
            )
            self._integrate_scan_ranges(
                robot_id,
                ranges,
                angle_min=0.0,
                angle_increment=angle_increment,
                usable_min=0.05,
                usable_max=self.max_mapping_range,
                sensor_range_max=self.max_mapping_range,
                source="map_raycast",
            )

    def _integrate_scan_ranges(
        self,
        robot_id: str,
        ranges: Sequence[float],
        *,
        angle_min: float,
        angle_increment: float,
        usable_min: float,
        usable_max: float,
        sensor_range_max: float,
        source: str,
    ) -> None:
        state = self.states[robot_id]
        pose = state.pose
        if not (math.isfinite(pose.x) and math.isfinite(pose.y) and math.isfinite(pose.yaw)):
            return

        state.current_log_odds.fill(0.0)
        state.current_observed.fill(0)
        hit_points: List[Tuple[float, float]] = []

        start_gx, start_gy = world_to_grid(
            pose.x, pose.y, self.origin_x, self.origin_y, self.resolution
        )
        if not self._inside(start_gx, start_gy):
            self.get_logger().warning(
                f"Ignoring {robot_id} scan because robot pose is outside map bounds: "
                f"({pose.x:.2f}, {pose.y:.2f})",
                throttle_duration_sec=5.0,
            )
            return

        for index in range(0, len(ranges), self.scan_stride):
            raw_range = float(ranges[index])
            if math.isnan(raw_range) or raw_range < usable_min:
                continue

            is_finite_hit = math.isfinite(raw_range)
            if is_finite_hit:
                measured = min(raw_range, usable_max)
                hit = raw_range < usable_max - self.hit_epsilon
                if math.isfinite(sensor_range_max):
                    hit = hit and raw_range < float(sensor_range_max) - self.hit_epsilon
            else:
                measured = usable_max
                hit = False

            if measured <= 0.0:
                continue
            beam_angle = pose.yaw + angle_min + index * angle_increment
            end_x = pose.x + measured * math.cos(beam_angle)
            end_y = pose.y + measured * math.sin(beam_angle)
            end_gx, end_gy = world_to_grid(
                end_x, end_y, self.origin_x, self.origin_y, self.resolution
            )
            cells = list(bresenham_cells(start_gx, start_gy, end_gx, end_gy))
            if not cells:
                continue

            free_cells = cells[:-1] if hit else cells
            # Skip the sensor's own cell to avoid over-weighting it every beam.
            for gx, gy in free_cells[1:]:
                self._apply_delta(
                    state.local_log_odds, state.local_observed, gx, gy, self.free_delta
                )
                self._apply_delta(
                    state.current_log_odds, state.current_observed, gx, gy, self.free_delta
                )

            if hit:
                gx, gy = cells[-1]
                self._apply_delta(
                    state.local_log_odds,
                    state.local_observed,
                    gx,
                    gy,
                    self.occupied_delta,
                )
                self._apply_delta(
                    state.current_log_odds,
                    state.current_observed,
                    gx,
                    gy,
                    self.occupied_delta,
                )
                if self._inside(gx, gy):
                    hit_points.append((end_x, end_y))

        state.latest_hit_points = hit_points
        state.scan_count += 1
        state.last_scan_source = source
        if state.scan_count == 1 or state.scan_count % 25 == 0:
            self.get_logger().info(
                f"Processed {state.scan_count} scans from {robot_id} "
                f"via {source} (hits={len(hit_points)}, pose=({pose.x:.2f},{pose.y:.2f}))"
            )

    def _header(self) -> Header:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.map_frame
        return header

    def _occupancy_grid(self, log_odds: np.ndarray, observed: np.ndarray) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header = self._header()
        msg.info.map_load_time = msg.header.stamp
        msg.info.resolution = float(self.resolution)
        msg.info.width = int(self.grid_width)
        msg.info.height = int(self.grid_height)
        msg.info.origin.position.x = float(self.origin_x)
        msg.info.origin.position.y = float(self.origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        flat_log = log_odds.reshape(-1)
        flat_seen = observed.reshape(-1)
        data = np.full(flat_log.shape, -1, dtype=np.int8)
        indices = np.nonzero(flat_seen > 0)[0]
        if indices.size:
            values = flat_log[indices].astype(np.float64)
            probabilities = np.where(
                values >= 0.0,
                1.0 / (1.0 + np.exp(-values)),
                np.exp(values) / (1.0 + np.exp(values)),
            )
            data[indices] = np.rint(probabilities * 100.0).astype(np.int8)
        msg.data = [int(value) for value in data]
        return msg

    def _combined_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        logs = np.zeros(self.grid_shape, dtype=np.float32)
        seen = np.zeros(self.grid_shape, dtype=np.uint32)
        for state in self.states.values():
            logs += state.local_log_odds
            seen += state.local_observed.astype(np.uint32)
        np.clip(logs, self.min_log_odds, self.max_log_odds, out=logs)
        return logs, seen

    def _publish_all(self) -> None:
        combined_log, combined_seen = self._combined_arrays()
        self._publish_heatmap(combined_log, combined_seen)
        self._publish_endpoints()
        self._publish_footprints()

        # OccupancyGrid list conversion is expensive; publish less often than markers.
        self._occupancy_publish_counter += 1
        if self._occupancy_publish_counter % 2 != 0:
            return

        for robot_id, state in self.states.items():
            shared_log = combined_log - state.local_log_odds
            shared_seen = combined_seen.astype(np.int64) - state.local_observed.astype(np.int64)
            np.clip(shared_log, self.min_log_odds, self.max_log_odds, out=shared_log)
            shared_seen = np.maximum(shared_seen, 0).astype(np.uint32)
            publishers = self.map_publishers[robot_id]
            publishers["current"].publish(
                self._occupancy_grid(state.current_log_odds, state.current_observed)
            )
            publishers["local"].publish(
                self._occupancy_grid(state.local_log_odds, state.local_observed)
            )
            publishers["shared"].publish(
                self._occupancy_grid(shared_log, shared_seen)
            )
            publishers["combined"].publish(
                self._occupancy_grid(combined_log, combined_seen)
            )

    def _publish_heatmap(self, log_odds: np.ndarray, observed: np.ndarray) -> None:
        marker = Marker()
        marker.header = self._header()
        marker.ns = "shared_lidar_probability"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.resolution * 0.95
        marker.scale.y = self.resolution * 0.95
        marker.scale.z = max(0.08, self.resolution * 0.70)
        # RViz can ignore per-point colors when marker.color.a is left at 0.
        marker.color = ColorRGBA(r=0.2, g=0.4, b=1.0, a=1.0)
        marker.lifetime = Duration(sec=0, nanosec=0)

        cloud_points: List[Tuple[float, float, float, float]] = []
        ys, xs = np.nonzero(observed > 0)
        for gy, gx in zip(ys.tolist(), xs.tolist()):
            world_x, world_y = grid_to_world(
                gx, gy, self.origin_x, self.origin_y, self.resolution
            )
            probability = probability_from_log_odds(float(log_odds[gy, gx]))
            r, g, b, a = heatmap_rgba(probability)
            point = Point(x=world_x, y=world_y, z=marker.scale.z * 0.5)
            marker.points.append(point)
            marker.colors.append(ColorRGBA(r=r, g=g, b=b, a=max(a, 0.75)))
            # Pack RGB into a float for PointCloud2 rgb field (RViz RGB8).
            rgb_uint = (
                (int(r * 255.0) << 16) | (int(g * 255.0) << 8) | int(b * 255.0)
            )
            rgb_float = struct.unpack("<f", struct.pack("<I", rgb_uint))[0]
            cloud_points.append((world_x, world_y, 0.05, rgb_float))

        array = MarkerArray()
        array.markers.append(marker)
        self.heatmap_pub.publish(array)
        self.heatmap_cloud_pub.publish(self._probability_cloud(cloud_points))

    def _probability_cloud(
        self, points: Sequence[Tuple[float, float, float, float]]
    ) -> PointCloud2:
        message = PointCloud2()
        message.header = self._header()
        message.height = 1
        message.width = len(points)
        message.is_dense = True
        message.is_bigendian = False
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        message.point_step = 16
        message.row_step = message.point_step * message.width
        if points:
            packed = np.asarray(points, dtype=np.float32)
            message.data = packed.tobytes()
        else:
            message.data = b""
        return message

    def _publish_endpoints(self) -> None:
        array = MarkerArray()
        for index, robot_id in enumerate(self.robot_ids):
            state = self.states[robot_id]
            r, g, b = ROBOT_COLORS[index % len(ROBOT_COLORS)]
            marker = Marker()
            marker.header = self._header()
            marker.ns = "live_lidar_endpoints"
            marker.id = index
            marker.type = Marker.SPHERE_LIST
            marker.pose.orientation.w = 1.0
            marker.scale.x = max(0.06, self.resolution * 0.70)
            marker.scale.y = marker.scale.x
            marker.scale.z = marker.scale.x
            marker.action = Marker.ADD
            marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            marker.points = [
                Point(x=x, y=y, z=0.12) for x, y in state.latest_hit_points
            ]
            if not marker.points:
                # Keep the namespace alive with an empty ADD instead of DELETE so
                # RViz does not disable the display namespace after startup.
                marker.points = []
            array.markers.append(marker)
        self.endpoint_pub.publish(array)

    def _publish_footprints(self) -> None:
        array = MarkerArray()
        for index, robot_id in enumerate(self.robot_ids):
            state = self.states[robot_id]
            r, g, b = ROBOT_COLORS[index % len(ROBOT_COLORS)]
            marker = Marker()
            marker.header = self._header()
            marker.ns = "robot_footprints"
            marker.id = index
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = state.pose.x
            marker.pose.position.y = state.pose.y
            marker.pose.position.z = 0.12
            marker.pose.orientation.z = math.sin(state.pose.yaw / 2.0)
            marker.pose.orientation.w = math.cos(state.pose.yaw / 2.0)
            marker.scale.x = max(0.45, self.footprint_size)
            marker.scale.y = max(0.45, self.footprint_size)
            marker.scale.z = 0.10
            marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            array.markers.append(marker)

            label = Marker()
            label.header = self._header()
            label.ns = "robot_footprints"
            label.id = 100 + index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = state.pose.x
            label.pose.position.y = state.pose.y
            label.pose.position.z = 0.35
            label.pose.orientation.w = 1.0
            label.scale.z = 0.28
            label.color = ColorRGBA(r=0.05, g=0.05, b=0.05, a=1.0)
            label.text = robot_id
            array.markers.append(label)
        self.footprint_pub.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SharedLidarMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
