"""Per-robot LiDAR mapping, map sharing, and RViz visualization node.

This node is deliberately isolated from the experiment planner.  It reads a
robot's scan and map-frame pose, builds local/current/shared/combined occupancy
layers, and publishes RViz-friendly markers.  It never writes to
``/planning_costmap`` or ``/<robot>/cmd_vel``.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import GridCells, OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .mapping_utils import (
    aggregate_log_odds,
    bresenham_cells,
    cell_index,
    cell_to_world,
    clamp,
    in_bounds,
    logistic,
    occupancy_to_log_odds,
    probability_to_occupancy,
    quaternion_from_yaw,
    transform_local_odometry_to_map,
    world_to_cell,
    yaw_from_quaternion,
)


Pose2D = Tuple[float, float, float]


class LidarMappingNode(Node):
    """Build and share occupancy evidence for one scenario robot."""

    def __init__(self) -> None:
        super().__init__("lidar_mapper")

        self.declare_parameter("robot_id", "robot")
        self.declare_parameter("peer_robot_ids", [""])
        self.declare_parameter("map_width", 1)
        self.declare_parameter("map_height", 1)
        self.declare_parameter("resolution", 0.5)
        self.declare_parameter("spawn_x", 0.0)
        self.declare_parameter("spawn_y", 0.0)
        self.declare_parameter("spawn_yaw", 0.0)
        self.declare_parameter("scan_topic", "")
        self.declare_parameter("odom_topic", "")
        self.declare_parameter("map_pose_topic", "")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("scan_stride", 1)
        self.declare_parameter("free_log_odds_delta", -0.40)
        self.declare_parameter("occupied_log_odds_delta", 0.85)
        self.declare_parameter("maximum_absolute_log_odds", 8.0)
        self.declare_parameter("distance_falloff_m", 5.0)
        self.declare_parameter("minimum_observation_weight", 0.25)
        self.declare_parameter("peer_evidence_scale", 1.0)
        self.declare_parameter("current_priority_scale", 0.50)
        self.declare_parameter("lidar_height_m", 0.18)
        self.declare_parameter("footprint_length_m", 0.18)
        self.declare_parameter("footprint_width_m", 0.14)
        self.declare_parameter("maximum_scan_range_m", 3.5)
        self.declare_parameter("enable_rviz_outputs", True)

        self.robot_id = str(self.get_parameter("robot_id").value).strip()
        if not self.robot_id:
            raise ValueError("robot_id must not be empty")

        raw_peers = self.get_parameter("peer_robot_ids").value
        self.peer_robot_ids = [
            str(value).strip()
            for value in raw_peers
            if str(value).strip() and str(value).strip() != self.robot_id
        ]
        self.width = int(self.get_parameter("map_width").value)
        self.height = int(self.get_parameter("map_height").value)
        self.resolution = float(self.get_parameter("resolution").value)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("map_width and map_height must be positive")
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")

        self.size = self.width * self.height
        self.spawn_pose: Pose2D = (
            float(self.get_parameter("spawn_x").value),
            float(self.get_parameter("spawn_y").value),
            float(self.get_parameter("spawn_yaw").value),
        )
        self.map_frame = str(self.get_parameter("map_frame").value).strip() or "map"
        self.scan_frame = f"{self.robot_id}/base_scan"
        self.base_frame = f"{self.robot_id}/base_link"

        self.free_delta = float(self.get_parameter("free_log_odds_delta").value)
        self.occupied_delta = float(
            self.get_parameter("occupied_log_odds_delta").value
        )
        if self.free_delta >= 0.0:
            raise ValueError("free_log_odds_delta must be negative")
        if self.occupied_delta <= 0.0:
            raise ValueError("occupied_log_odds_delta must be positive")
        self.log_odds_limit = max(
            0.1,
            abs(float(self.get_parameter("maximum_absolute_log_odds").value)),
        )
        self.distance_falloff_m = max(
            0.01, float(self.get_parameter("distance_falloff_m").value)
        )
        self.minimum_observation_weight = clamp(
            float(self.get_parameter("minimum_observation_weight").value),
            0.0,
            1.0,
        )
        self.peer_evidence_scale = max(
            0.0, float(self.get_parameter("peer_evidence_scale").value)
        )
        self.current_priority_scale = max(
            0.0, float(self.get_parameter("current_priority_scale").value)
        )
        self.scan_stride = max(1, int(self.get_parameter("scan_stride").value))
        self.lidar_height_m = max(
            0.0, float(self.get_parameter("lidar_height_m").value)
        )
        self.footprint_length_m = max(
            0.05, float(self.get_parameter("footprint_length_m").value)
        )
        self.footprint_width_m = max(
            0.05, float(self.get_parameter("footprint_width_m").value)
        )
        self.maximum_scan_range_m = max(
            0.1, float(self.get_parameter("maximum_scan_range_m").value)
        )
        self.enable_rviz_outputs = bool(
            self.get_parameter("enable_rviz_outputs").value
        )

        self.local_log_odds = [0.0] * self.size
        self.local_observed = bytearray(self.size)
        self.current_log_odds = [0.0] * self.size
        self.current_observed = bytearray(self.size)
        self.peer_layers: Dict[str, Tuple[List[float], bytearray]] = {}

        self.map_pose: Optional[Pose2D] = None
        self.last_map_pose_monotonic = 0.0
        self.initial_odom_pose: Optional[Pose2D] = None
        self.received_scan_count = 0
        self.last_missing_pose_warning = 0.0
        self.last_peer_warning: Dict[str, float] = {}

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.local_map_pub = self.create_publisher(
            OccupancyGrid, "local_map", map_qos
        )
        self.current_map_pub = self.create_publisher(
            OccupancyGrid, "current_observation_map", map_qos
        )
        self.shared_map_pub = self.create_publisher(
            OccupancyGrid, "shared_map", map_qos
        )
        self.combined_map_pub = self.create_publisher(
            OccupancyGrid, "combined_map", map_qos
        )
        self.scan_rviz_pub = None
        self.combined_markers_pub = None
        self.local_markers_pub = None
        self.shared_markers_pub = None
        self.current_markers_pub = None
        self.footprint_markers_pub = None
        self.lidar_free_cells_pub = None
        self.lidar_occupied_cells_pub = None
        # Always republish scans into a Reliable topic so RViz can show every
        # robot's LiDAR (shared visualization). Heavy markers stay optional.
        scan_rviz_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.scan_rviz_pub = self.create_publisher(
            LaserScan, "scan_rviz", scan_rviz_qos
        )
        if self.enable_rviz_outputs:
            self.combined_markers_pub = self.create_publisher(
                MarkerArray, "combined_probability_markers", marker_qos
            )
            self.local_markers_pub = self.create_publisher(
                MarkerArray, "local_history_markers", marker_qos
            )
            self.shared_markers_pub = self.create_publisher(
                MarkerArray, "shared_history_markers", marker_qos
            )
            self.current_markers_pub = self.create_publisher(
                MarkerArray, "current_observation_markers", marker_qos
            )
            self.footprint_markers_pub = self.create_publisher(
                MarkerArray, "footprint_markers", marker_qos
            )
            # GridCells avoid the broken OccupancyGrid Map shader on many VMs.
            self.lidar_free_cells_pub = self.create_publisher(
                GridCells, "lidar_free_cells", marker_qos
            )
            self.lidar_occupied_cells_pub = self.create_publisher(
                GridCells, "lidar_occupied_cells", marker_qos
            )

        scan_topic = str(self.get_parameter("scan_topic").value).strip()
        odom_topic = str(self.get_parameter("odom_topic").value).strip()
        map_pose_topic = str(self.get_parameter("map_pose_topic").value).strip()
        if not scan_topic:
            scan_topic = f"/{self.robot_id}/scan"
        if not odom_topic:
            odom_topic = f"/{self.robot_id}/odom"
        if not map_pose_topic:
            map_pose_topic = f"/{self.robot_id}/map_pose"

        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 20
        )
        self.map_pose_sub = self.create_subscription(
            PoseStamped, map_pose_topic, self.map_pose_callback, 20
        )

        self.peer_subscriptions = []
        for peer_id in self.peer_robot_ids:
            subscription = self.create_subscription(
                OccupancyGrid,
                f"/{peer_id}/local_map",
                lambda message, source=peer_id: self.peer_map_callback(
                    source, message
                ),
                map_qos,
            )
            self.peer_subscriptions.append(subscription)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.publish_static_lidar_transform()

        publish_rate_hz = max(
            0.2, float(self.get_parameter("publish_rate_hz").value)
        )
        self.publish_timer = self.create_timer(
            1.0 / publish_rate_hz, self.publish_all_layers
        )

        self.get_logger().info(
            "LiDAR mapping ready: "
            f"robot={self.robot_id}, peers={self.peer_robot_ids}, "
            f"map={self.width}x{self.height}@{self.resolution:.3f}m, "
            f"scan={scan_topic}, pose={map_pose_topic}, "
            f"rviz_outputs={self.enable_rviz_outputs}"
        )

    # ------------------------------------------------------------------
    # Pose and TF handling
    # ------------------------------------------------------------------

    def map_pose_callback(self, message: PoseStamped) -> None:
        orientation = message.pose.orientation
        self.map_pose = (
            float(message.pose.position.x),
            float(message.pose.position.y),
            yaw_from_quaternion(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ),
        )
        self.last_map_pose_monotonic = time.monotonic()
        self.publish_dynamic_transform()

    def odom_callback(self, message: Odometry) -> None:
        """Use odometry only as a fallback when manager map-pose is unavailable."""

        # The experiment's default kinematic mode publishes map_pose directly.
        # Prefer it for two seconds before considering wheel-odometry fallback.
        if time.monotonic() - self.last_map_pose_monotonic < 2.0:
            return

        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        local_pose = (
            float(position.x),
            float(position.y),
            yaw_from_quaternion(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ),
        )
        if self.initial_odom_pose is None:
            self.initial_odom_pose = local_pose

        self.map_pose = transform_local_odometry_to_map(
            local_pose,
            self.initial_odom_pose,
            self.spawn_pose,
        )
        self.publish_dynamic_transform()

    def publish_static_lidar_transform(self) -> None:
        if self.static_tf_broadcaster is None:
            return
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.base_frame
        transform.child_frame_id = self.scan_frame
        transform.transform.translation.z = self.lidar_height_m
        transform.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(transform)

    def publish_dynamic_transform(self, stamp=None) -> None:
        if self.map_pose is None or self.tf_broadcaster is None:
            return
        x, y, yaw = self.map_pose
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        transform = TransformStamped()
        transform.header.stamp = stamp or self.get_clock().now().to_msg()
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)

    # ------------------------------------------------------------------
    # LiDAR processing
    # ------------------------------------------------------------------

    def observation_weight(self, distance_m: float) -> float:
        return max(
            self.minimum_observation_weight,
            math.exp(-max(0.0, distance_m) / self.distance_falloff_m),
        )

    def add_evidence(
        self,
        values: List[float],
        observed: bytearray,
        index: int,
        delta: float,
    ) -> None:
        observed[index] = 1
        values[index] = clamp(
            values[index] + delta,
            -self.log_odds_limit,
            self.log_odds_limit,
        )

    def scan_callback(self, message: LaserScan) -> None:
        if self.map_pose is None:
            now = time.monotonic()
            if now - self.last_missing_pose_warning > 5.0:
                self.get_logger().warning(
                    f"Waiting for /{self.robot_id}/map_pose or odometry before mapping scans."
                )
                self.last_missing_pose_warning = now
            return

        self.current_log_odds = [0.0] * self.size
        self.current_observed = bytearray(self.size)

        robot_x, robot_y, robot_yaw = self.map_pose
        start_cell = world_to_cell(robot_x, robot_y, self.resolution)
        if not in_bounds(start_cell, self.height, self.width):
            self.get_logger().warning(
                f"Robot pose is outside the mapping grid: pose={self.map_pose}"
            )
            return

        reported_range_max = float(message.range_max)
        range_max = (
            reported_range_max
            if math.isfinite(reported_range_max) and reported_range_max > 0.0
            else self.maximum_scan_range_m
        )
        range_max = min(range_max, self.maximum_scan_range_m)
        range_min = max(0.0, float(message.range_min))
        hit_epsilon = max(0.01, range_max * 0.005)

        for beam_index in range(0, len(message.ranges), self.scan_stride):
            raw_range = float(message.ranges[beam_index])
            if math.isnan(raw_range) or raw_range < range_min:
                continue

            finite_hit = math.isfinite(raw_range) and raw_range < range_max - hit_epsilon
            distance = raw_range if math.isfinite(raw_range) else range_max
            distance = clamp(distance, range_min, range_max)
            if distance <= 0.0:
                continue

            beam_angle = (
                robot_yaw
                + float(message.angle_min)
                + beam_index * float(message.angle_increment)
            )
            endpoint_x = robot_x + distance * math.cos(beam_angle)
            endpoint_y = robot_y + distance * math.sin(beam_angle)
            endpoint_cell = world_to_cell(endpoint_x, endpoint_y, self.resolution)
            ray = bresenham_cells(start_cell, endpoint_cell)

            endpoint_is_inside = in_bounds(
                endpoint_cell, self.height, self.width
            )
            if finite_hit and endpoint_is_inside:
                free_cells = ray[:-1]
            else:
                free_cells = ray

            weight = self.observation_weight(distance)
            free_delta = self.free_delta * weight
            occupied_delta = self.occupied_delta * weight

            for cell in free_cells:
                if not in_bounds(cell, self.height, self.width):
                    continue
                index = cell_index(cell, self.width)
                self.add_evidence(
                    self.local_log_odds,
                    self.local_observed,
                    index,
                    free_delta,
                )
                self.add_evidence(
                    self.current_log_odds,
                    self.current_observed,
                    index,
                    free_delta,
                )

            if finite_hit and endpoint_is_inside:
                index = cell_index(endpoint_cell, self.width)
                self.add_evidence(
                    self.local_log_odds,
                    self.local_observed,
                    index,
                    occupied_delta,
                )
                self.add_evidence(
                    self.current_log_odds,
                    self.current_observed,
                    index,
                    occupied_delta,
                )

        self.received_scan_count += 1
        # Use one timestamp for the TF and republished scan so RViz never has
        # to extrapolate between mismatched "now" values.
        stamp = self.get_clock().now().to_msg()
        self.publish_dynamic_transform(stamp)
        self.publish_rviz_scan(message, stamp)
        # Publish the current scan immediately; persistent/shared layers remain
        # rate limited by publish_timer.
        self.current_map_pub.publish(
            self.make_occupancy_grid(
                self.current_log_odds, self.current_observed, stamp
            )
        )
        if self.current_markers_pub is not None:
            self.current_markers_pub.publish(
                self.make_probability_markers(
                    namespace="current_observation",
                    values=self.current_log_odds,
                    observed=self.current_observed,
                    z=0.075,
                    color_mode="current",
                    exclude_mask=None,
                )
            )

    def publish_rviz_scan(self, message: LaserScan, stamp=None) -> None:
        rviz_scan = LaserScan()
        rviz_scan.header.stamp = stamp or self.get_clock().now().to_msg()
        rviz_scan.header.frame_id = self.scan_frame
        rviz_scan.angle_min = message.angle_min
        rviz_scan.angle_max = message.angle_max
        rviz_scan.angle_increment = message.angle_increment
        rviz_scan.time_increment = message.time_increment
        rviz_scan.scan_time = message.scan_time
        rviz_scan.range_min = message.range_min
        rviz_scan.range_max = message.range_max
        rviz_scan.ranges = list(message.ranges)
        rviz_scan.intensities = list(message.intensities)
        if self.scan_rviz_pub is not None:
            self.scan_rviz_pub.publish(rviz_scan)

    # ------------------------------------------------------------------
    # Peer sharing and map publication
    # ------------------------------------------------------------------

    def peer_map_callback(self, peer_id: str, message: OccupancyGrid) -> None:
        valid = (
            int(message.info.width) == self.width
            and int(message.info.height) == self.height
            and math.isclose(
                float(message.info.resolution),
                self.resolution,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            and len(message.data) == self.size
        )
        if not valid:
            now = time.monotonic()
            if now - self.last_peer_warning.get(peer_id, 0.0) > 10.0:
                self.get_logger().warning(
                    f"Ignoring incompatible local map from {peer_id}: "
                    f"{message.info.width}x{message.info.height}@"
                    f"{message.info.resolution}"
                )
                self.last_peer_warning[peer_id] = now
            return

        values = [0.0] * self.size
        observed = bytearray(self.size)
        for index, occupancy in enumerate(message.data):
            numeric = int(occupancy)
            if numeric < 0:
                continue
            observed[index] = 1
            values[index] = clamp(
                occupancy_to_log_odds(numeric),
                -self.log_odds_limit,
                self.log_odds_limit,
            )
        self.peer_layers[peer_id] = (values, observed)

    def aggregate_shared_layer(self) -> Tuple[List[float], bytearray]:
        layers = [
            (values, observed, self.peer_evidence_scale)
            for values, observed in self.peer_layers.values()
        ]
        return aggregate_log_odds(layers, self.size, self.log_odds_limit)

    def aggregate_combined_layer(
        self, shared_values: Sequence[float], shared_observed: Sequence[int]
    ) -> Tuple[List[float], bytearray]:
        return aggregate_log_odds(
            [
                (self.local_log_odds, self.local_observed, 1.0),
                (shared_values, shared_observed, 1.0),
                (
                    self.current_log_odds,
                    self.current_observed,
                    self.current_priority_scale,
                ),
            ],
            self.size,
            self.log_odds_limit,
        )

    def publish_all_layers(self) -> None:
        stamp = self.get_clock().now().to_msg()
        shared_values, shared_observed = self.aggregate_shared_layer()
        combined_values, combined_observed = self.aggregate_combined_layer(
            shared_values, shared_observed
        )

        self.local_map_pub.publish(
            self.make_occupancy_grid(
                self.local_log_odds, self.local_observed, stamp
            )
        )
        self.current_map_pub.publish(
            self.make_occupancy_grid(
                self.current_log_odds, self.current_observed, stamp
            )
        )
        self.shared_map_pub.publish(
            self.make_occupancy_grid(shared_values, shared_observed, stamp)
        )
        self.combined_map_pub.publish(
            self.make_occupancy_grid(combined_values, combined_observed, stamp)
        )

        if self.enable_rviz_outputs:
            self.combined_markers_pub.publish(
                self.make_probability_markers(
                    namespace="combined_probability",
                    values=combined_values,
                    observed=combined_observed,
                    z=0.012,
                    color_mode="heatmap",
                    exclude_mask=None,
                )
            )
            # "Previously mapped" excludes cells in the newest local scan so the
            # current observation overlay remains visually distinct.
            self.local_markers_pub.publish(
                self.make_probability_markers(
                    namespace="local_history",
                    values=self.local_log_odds,
                    observed=self.local_observed,
                    z=0.032,
                    color_mode="local",
                    exclude_mask=self.current_observed,
                )
            )
            self.shared_markers_pub.publish(
                self.make_probability_markers(
                    namespace="shared_history",
                    values=shared_values,
                    observed=shared_observed,
                    z=0.052,
                    color_mode="shared",
                    exclude_mask=None,
                )
            )
            self.current_markers_pub.publish(
                self.make_probability_markers(
                    namespace="current_observation",
                    values=self.current_log_odds,
                    observed=self.current_observed,
                    z=0.075,
                    color_mode="current",
                    exclude_mask=None,
                )
            )
            self.footprint_markers_pub.publish(self.make_footprint_markers())
            self.publish_lidar_grid_cells(
                combined_values, combined_observed, stamp
            )
            self.publish_dynamic_transform()

        # Keep TF fresh even when heavy RViz markers are disabled.
        if not self.enable_rviz_outputs:
            self.publish_dynamic_transform()

    def publish_lidar_grid_cells(
        self,
        values: Sequence[float],
        observed: Sequence[int],
        stamp: object,
    ) -> None:
        if (
            self.lidar_free_cells_pub is None
            or self.lidar_occupied_cells_pub is None
        ):
            return

        free = GridCells()
        free.header.stamp = stamp
        free.header.frame_id = self.map_frame
        free.cell_width = self.resolution * 0.92
        free.cell_height = self.resolution * 0.92
        occupied = GridCells()
        occupied.header = free.header
        occupied.cell_width = free.cell_width
        occupied.cell_height = free.cell_height

        for index in range(self.size):
            if not observed[index]:
                continue
            row, col = divmod(index, self.width)
            x, y = cell_to_world(row, col, self.resolution)
            probability = logistic(values[index])
            if probability >= 0.55:
                occupied.cells.append(Point(x=x, y=y, z=0.05))
            else:
                free.cells.append(Point(x=x, y=y, z=0.02))

        self.lidar_free_cells_pub.publish(free)
        self.lidar_occupied_cells_pub.publish(occupied)

    def make_occupancy_grid(
        self,
        values: Sequence[float],
        observed: Sequence[int],
        stamp: object,
    ) -> OccupancyGrid:
        message = OccupancyGrid()
        message.header.stamp = stamp
        message.header.frame_id = self.map_frame
        message.info.map_load_time = stamp
        message.info.resolution = self.resolution
        message.info.width = self.width
        message.info.height = self.height
        message.info.origin.orientation.w = 1.0
        message.data = [
            probability_to_occupancy(logistic(values[index]))
            if observed[index]
            else -1
            for index in range(self.size)
        ]
        return message

    # ------------------------------------------------------------------
    # RViz markers
    # ------------------------------------------------------------------

    @staticmethod
    def heatmap_color(probability: float, alpha: float = 0.55) -> ColorRGBA:
        """Blue free-space / amber occupied overlay (reference-repo style)."""

        p = clamp(probability, 0.0, 1.0)
        if p <= 0.5:
            # Free / low occupancy: bright cyan-blue floor.
            fraction = p / 0.5
            red = 0.10 + 0.05 * fraction
            green = 0.55 + 0.20 * fraction
            blue = 0.95
            alpha = 0.35 + 0.20 * (1.0 - fraction)
        else:
            # Occupied: purple/magenta wall evidence (like reference).
            fraction = (p - 0.5) / 0.5
            red = 0.45 + 0.45 * fraction
            green = 0.15 * (1.0 - fraction)
            blue = 0.75 - 0.25 * fraction
            alpha = 0.45 + 0.35 * fraction
        return ColorRGBA(r=red, g=green, b=blue, a=alpha)

    @staticmethod
    def layer_color(probability: float, mode: str) -> ColorRGBA:
        certainty = abs(clamp(probability, 0.0, 1.0) - 0.5) * 2.0
        alpha = 0.18 + 0.42 * certainty
        if mode == "local":
            # Blue local history; occupied cells are darker.
            return ColorRGBA(
                r=0.05,
                g=0.25 + 0.30 * (1.0 - probability),
                b=0.75 + 0.25 * probability,
                a=alpha,
            )
        if mode == "shared":
            # Green shared history; occupied cells trend toward magenta.
            return ColorRGBA(
                r=0.15 + 0.60 * probability,
                g=0.75 - 0.30 * probability,
                b=0.20 + 0.35 * probability,
                a=alpha,
            )
        # Bright current scan: cyan free rays, orange/red obstacle endpoints.
        return ColorRGBA(
            r=0.10 + 0.90 * probability,
            g=0.90 - 0.40 * probability,
            b=0.95 * (1.0 - probability),
            a=0.72,
        )

    def make_probability_markers(
        self,
        *,
        namespace: str,
        values: Sequence[float],
        observed: Sequence[int],
        z: float,
        color_mode: str,
        exclude_mask: Optional[Sequence[int]],
    ) -> MarkerArray:
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = namespace
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.resolution * 0.94
        marker.scale.y = self.resolution * 0.94
        marker.scale.z = max(0.008, self.resolution * 0.025)

        for index in range(self.size):
            if not observed[index]:
                continue
            if exclude_mask is not None and exclude_mask[index]:
                continue
            row, col = divmod(index, self.width)
            x, y = cell_to_world(row, col, self.resolution)
            marker.points.append(Point(x=x, y=y, z=float(z)))
            probability = logistic(values[index])
            if color_mode == "heatmap":
                marker.colors.append(self.heatmap_color(probability))
            else:
                marker.colors.append(self.layer_color(probability, color_mode))

        if marker.points:
            array.markers.append(marker)
        return array

    def make_footprint_markers(self) -> MarkerArray:
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)
        if self.map_pose is None:
            return array

        x, y, yaw = self.map_pose
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        stamp = self.get_clock().now().to_msg()

        body = Marker()
        body.header.stamp = stamp
        body.header.frame_id = self.map_frame
        body.ns = "turtlebot_footprint"
        body.id = 0
        body.type = Marker.CUBE
        body.action = Marker.ADD
        body.pose.position.x = x
        body.pose.position.y = y
        body.pose.position.z = 0.055
        body.pose.orientation.x = qx
        body.pose.orientation.y = qy
        body.pose.orientation.z = qz
        body.pose.orientation.w = qw
        body.scale.x = self.footprint_length_m
        body.scale.y = self.footprint_width_m
        body.scale.z = 0.10
        # White/light square in RViz only (Gazebo keeps the TurtleBot mesh).
        body.color = ColorRGBA(r=0.95, g=0.95, b=0.98, a=0.95)
        array.markers.append(body)

        heading = Marker()
        heading.header = body.header
        heading.ns = "turtlebot_heading"
        heading.id = 1
        heading.type = Marker.ARROW
        heading.action = Marker.ADD
        heading.pose.position.x = body.pose.position.x
        heading.pose.position.y = body.pose.position.y
        heading.pose.position.z = 0.13
        heading.pose.orientation.x = body.pose.orientation.x
        heading.pose.orientation.y = body.pose.orientation.y
        heading.pose.orientation.z = body.pose.orientation.z
        heading.pose.orientation.w = body.pose.orientation.w
        heading.scale.x = self.footprint_length_m * 0.90
        heading.scale.y = max(0.025, self.footprint_width_m * 0.20)
        heading.scale.z = max(0.04, self.footprint_width_m * 0.32)
        heading.color = ColorRGBA(r=1.0, g=0.85, b=0.05, a=1.0)
        array.markers.append(heading)

        label = Marker()
        label.header = body.header
        label.ns = "turtlebot_label"
        label.id = 2
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = x
        label.pose.position.y = y
        label.pose.position.z = 0.32
        label.pose.orientation.w = 1.0
        label.scale.z = max(0.14, self.resolution * 0.32)
        label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        label.text = self.robot_id
        array.markers.append(label)

        return array


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[LidarMappingNode] = None
    try:
        node = LidarMappingNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
