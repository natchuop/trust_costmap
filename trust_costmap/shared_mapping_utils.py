"""Pure mapping helpers for the shared LiDAR occupancy mapper."""

from __future__ import annotations

import math
from typing import Iterator, List, Tuple

import numpy as np


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def probability_from_log_odds(log_odds: float) -> float:
    """Convert log odds to an occupancy probability without overflow."""
    if log_odds >= 0.0:
        z = math.exp(-log_odds)
        return 1.0 / (1.0 + z)
    z = math.exp(log_odds)
    return z / (1.0 + z)


def occupancy_value(log_odds: float) -> int:
    return int(round(100.0 * probability_from_log_odds(log_odds)))


def world_to_grid(
    x: float,
    y: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> Tuple[int, int]:
    return (
        int(math.floor((x - origin_x) / resolution)),
        int(math.floor((y - origin_y) / resolution)),
    )


def grid_to_world(
    gx: int,
    gy: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> Tuple[float, float]:
    return (
        origin_x + (gx + 0.5) * resolution,
        origin_y + (gy + 0.5) * resolution,
    )


def bresenham_cells(x0: int, y0: int, x1: int, y1: int) -> Iterator[Tuple[int, int]]:
    """Yield all integer grid cells on a line, including both endpoints."""
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    x = x0
    y = y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x += sx
        if doubled <= dx:
            error += dx
            y += sy


def heatmap_rgba(probability: float) -> Tuple[float, float, float, float]:
    """Blue for free, violet/dark purple for occupied; keep cells readable in RViz."""
    p = clamp(probability, 0.0, 1.0)

    if p <= 0.5:
        # Strong free evidence -> saturated blue.
        t = (0.5 - p) / 0.5
        r = 0.35 * (1.0 - t) + 0.05 * t
        g = 0.45 * (1.0 - t) + 0.25 * t
        b = 0.95 * (1.0 - t) + 1.00 * t
    else:
        # Strong occupied evidence -> violet / dark purple.
        t = (p - 0.5) / 0.5
        r = 0.55 * (1.0 - t) + 0.35 * t
        g = 0.20 * (1.0 - t) + 0.00 * t
        b = 0.90 * (1.0 - t) + 0.45 * t

    # Keep even early/weak evidence visible on the light RViz background.
    alpha = 0.55 + 0.45 * abs(p - 0.5) * 2.0
    return r, g, b, alpha


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def load_movingai_occupancy(map_path: str) -> Tuple[np.ndarray, int, int]:
    """Load MovingAI map as a boolean occupancy grid (True = blocked)."""
    width = None
    height = None
    rows: List[str] = []
    with open(map_path, "r", encoding="utf-8") as handle:
        for line in handle:
            clean = line.rstrip("\n")
            if clean.startswith("width "):
                width = int(clean.split()[1])
            elif clean.startswith("height "):
                height = int(clean.split()[1])
            elif clean == "map":
                break
        if width is None or height is None:
            raise RuntimeError(f"Invalid MovingAI map metadata: {map_path}")
        for _ in range(height):
            row = handle.readline()
            if not row:
                break
            rows.append(row.rstrip("\n"))
    if len(rows) != height:
        raise RuntimeError(f"MovingAI map row count mismatch: {map_path}")
    occupied = np.zeros((height, width), dtype=bool)
    free_symbols = set(".G")
    for row_idx, row in enumerate(rows):
        for col_idx, symbol in enumerate(row[:width]):
            occupied[row_idx, col_idx] = symbol not in free_symbols
    return occupied, width, height


def raycast_lidar_ranges(
    occupied: np.ndarray,
    cell_size_m: float,
    origin_x: float,
    origin_y: float,
    x: float,
    y: float,
    yaw: float,
    max_range_m: float,
    beam_count: int = 72,
    angle_min: float = 0.0,
    angle_max: float = 2.0 * math.pi,
) -> Tuple[List[float], float]:
    """Cast beams on a MovingAI occupancy grid and return ranges + angle increment."""
    height, width = occupied.shape
    if beam_count < 2:
        raise ValueError("beam_count must be >= 2")
    span = angle_max - angle_min
    angle_increment = span / float(beam_count)
    ranges: List[float] = []
    start_col = int(math.floor((x - origin_x) / cell_size_m))
    start_row = int(math.floor((y - origin_y) / cell_size_m))

    for index in range(beam_count):
        angle = yaw + angle_min + index * angle_increment
        end_x = x + max_range_m * math.cos(angle)
        end_y = y + max_range_m * math.sin(angle)
        end_col = int(math.floor((end_x - origin_x) / cell_size_m))
        end_row = int(math.floor((end_y - origin_y) / cell_size_m))
        hit_range = max_range_m
        for col, row in bresenham_cells(start_col, start_row, end_col, end_row):
            if col == start_col and row == start_row:
                continue
            if row < 0 or col < 0 or row >= height or col >= width:
                # Treat map edge as a hit at the boundary cell center.
                cx = origin_x + (col + 0.5) * cell_size_m
                cy = origin_y + (row + 0.5) * cell_size_m
                hit_range = min(max_range_m, math.hypot(cx - x, cy - y))
                break
            if occupied[row, col]:
                cx = origin_x + (col + 0.5) * cell_size_m
                cy = origin_y + (row + 0.5) * cell_size_m
                hit_range = min(max_range_m, math.hypot(cx - x, cy - y))
                break
        ranges.append(hit_range)
    return ranges, angle_increment
