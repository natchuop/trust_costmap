#!/usr/bin/env python3
"""Runtime smoke checks for the shared LiDAR mapping topics."""

from __future__ import annotations

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray


class Checker(Node):
    def __init__(self):
        super().__init__("shared_mapping_checker")
        self.received = {}
        # Match lidar_mapper VOLATILE visualization publishers (RViz-compatible).
        qos = QoSProfile(depth=5, history=HistoryPolicy.KEEP_LAST)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE
        topics = {
            "/shared_lidar/probability_markers": MarkerArray,
            "/shared_lidar/scan_endpoints": MarkerArray,
            "/shared_lidar/robot_footprints": MarkerArray,
            "/shared_lidar/probability_cloud": PointCloud2,
        }
        for topic, msg_type in topics.items():
            self.create_subscription(
                msg_type,
                topic,
                lambda msg, name=topic: self._record(name, msg),
                qos,
            )

    def _record(self, topic, msg):
        self.received[topic] = msg


def main():
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    rclpy.init()
    node = Checker()
    deadline = time.monotonic() + timeout
    required = (
        "/shared_lidar/probability_markers",
        "/shared_lidar/scan_endpoints",
        "/shared_lidar/robot_footprints",
    )
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.25)
        if not all(topic in node.received for topic in required):
            continue
        heatmap = node.received["/shared_lidar/probability_markers"]
        heatmap_points = sum(len(marker.points) for marker in heatmap.markers)
        cloud = node.received.get("/shared_lidar/probability_cloud")
        cloud_width = getattr(cloud, "width", 0) if cloud is not None else 0
        if heatmap_points > 0 or cloud_width > 0:
            break
    missing = [topic for topic in required if topic not in node.received]
    if missing:
        print("FAIL: missing topics/messages:", ", ".join(missing))
        code = 1
    else:
        heatmap = node.received["/shared_lidar/probability_markers"]
        footprints = node.received["/shared_lidar/robot_footprints"]
        heatmap_points = sum(len(marker.points) for marker in heatmap.markers)
        cloud = node.received.get("/shared_lidar/probability_cloud")
        cloud_width = getattr(cloud, "width", 0) if cloud is not None else 0
        print(
            f"PASS: heatmap points={heatmap_points}, "
            f"cloud width={cloud_width}, footprints={len(footprints.markers)}"
        )
        if heatmap_points <= 0 and cloud_width <= 0:
            print("WARN: visualization topics are alive but still empty.")
        code = 0
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
