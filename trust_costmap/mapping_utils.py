"""Pure helpers for LiDAR occupancy mapping and map-frame pose conversion.

The functions in this module have no ROS dependencies so they can be unit tested
on machines that do not have ROS 2 installed.  The runtime node lives in
``lidar_mapping_node.py``.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


Cell = Tuple[int, int]


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp ``value`` to the inclusive ``[lower, upper]`` interval."""

    return max(lower, min(upper, value))


def normalize_angle(angle: float) -> float:
    """Normalize an angle to ``[-pi, pi]``."""

    return math.atan2(math.sin(angle), math.cos(angle))


def logistic(log_odds: float) -> float:
    """Convert log-odds evidence into an occupancy probability."""

    value = clamp(float(log_odds), -60.0, 60.0)
    return 1.0 / (1.0 + math.exp(-value))


def logit(probability: float, epsilon: float = 1e-4) -> float:
    """Convert an occupancy probability into finite log odds."""

    p = clamp(float(probability), epsilon, 1.0 - epsilon)
    return math.log(p / (1.0 - p))


def probability_to_occupancy(probability: float) -> int:
    """Convert a probability to the ROS OccupancyGrid integer range."""

    return int(round(clamp(float(probability), 0.0, 1.0) * 100.0))


def occupancy_to_log_odds(value: int) -> float:
    """Convert a known OccupancyGrid value in ``[0, 100]`` to log odds."""

    return logit(clamp(int(value), 0, 100) / 100.0)


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw from a quaternion."""

    return math.atan2(
        2.0 * (float(w) * float(z) + float(x) * float(y)),
        1.0 - 2.0 * (float(y) * float(y) + float(z) * float(z)),
    )


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """Return an ``(x, y, z, w)`` quaternion for planar yaw."""

    half = float(yaw) * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def transform_local_odometry_to_map(
    local_pose: Tuple[float, float, float],
    initial_local_pose: Tuple[float, float, float],
    spawn_pose: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Apply the same spawn-offset conversion as the experiment manager.

    Gazebo DiffDrive odometry is local to each TurtleBot's initial pose.  The
    experiment manager rotates the local displacement by the configured spawn
    yaw and then translates it into the common MovingAI/Gazebo map frame.
    """

    local_x, local_y, local_yaw = (float(value) for value in local_pose)
    initial_x, initial_y, initial_yaw = (
        float(value) for value in initial_local_pose
    )
    spawn_x, spawn_y, spawn_yaw = (float(value) for value in spawn_pose)

    delta_x_local = local_x - initial_x
    delta_y_local = local_y - initial_y

    cos_spawn = math.cos(spawn_yaw)
    sin_spawn = math.sin(spawn_yaw)

    world_x = spawn_x + cos_spawn * delta_x_local - sin_spawn * delta_y_local
    world_y = spawn_y + sin_spawn * delta_x_local + cos_spawn * delta_y_local
    world_yaw = normalize_angle(
        spawn_yaw + normalize_angle(local_yaw - initial_yaw)
    )

    return world_x, world_y, world_yaw


def world_to_cell(
    x: float,
    y: float,
    resolution: float,
) -> Cell:
    """Convert map-frame coordinates to a ``(row, column)`` grid cell."""

    if resolution <= 0.0:
        raise ValueError("resolution must be greater than zero")
    return int(math.floor(float(y) / resolution)), int(
        math.floor(float(x) / resolution)
    )


def cell_to_world(row: int, col: int, resolution: float) -> Tuple[float, float]:
    """Return the center of a ``(row, column)`` cell in map coordinates."""

    if resolution <= 0.0:
        raise ValueError("resolution must be greater than zero")
    return (
        (int(col) + 0.5) * resolution,
        (int(row) + 0.5) * resolution,
    )


def in_bounds(cell: Cell, height: int, width: int) -> bool:
    """Return whether a cell lies inside the map."""

    row, col = cell
    return 0 <= row < int(height) and 0 <= col < int(width)


def cell_index(cell: Cell, width: int) -> int:
    """Convert ``(row, column)`` into row-major storage index."""

    row, col = cell
    return int(row) * int(width) + int(col)


def bresenham_cells(start: Cell, end: Cell) -> List[Cell]:
    """Return every integer cell crossed by a line, including both endpoints."""

    row0, col0 = (int(value) for value in start)
    row1, col1 = (int(value) for value in end)

    x0, y0 = col0, row0
    x1, y1 = col1, row1
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    cells: List[Cell] = []
    while True:
        cells.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy

    return cells


def aggregate_log_odds(
    layers: Iterable[Tuple[Sequence[float], Sequence[int], float]],
    size: int,
    maximum_absolute_log_odds: float,
) -> Tuple[List[float], bytearray]:
    """Combine observed log-odds layers without treating unknown as 0.5.

    Each item is ``(log_odds, observed_mask, scale)``.  Unknown cells do not
    contribute evidence and remain unknown when no layer has observed them.
    """

    result = [0.0] * int(size)
    observed = bytearray(int(size))
    limit = abs(float(maximum_absolute_log_odds))

    for values, mask, scale in layers:
        if len(values) != size or len(mask) != size:
            raise ValueError("all map layers must have the requested size")
        factor = float(scale)
        for index in range(size):
            if mask[index]:
                observed[index] = 1
                result[index] = clamp(
                    result[index] + float(values[index]) * factor,
                    -limit,
                    limit,
                )

    return result, observed
