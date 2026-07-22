import math
from pathlib import Path

import numpy as np

from trust_costmap.shared_mapping_utils import (
    bresenham_cells,
    heatmap_rgba,
    load_movingai_occupancy,
    probability_from_log_odds,
    raycast_lidar_ranges,
    world_to_grid,
)


def test_bresenham_includes_both_endpoints():
    cells = list(bresenham_cells(1, 2, 4, 2))
    assert cells == [(1, 2), (2, 2), (3, 2), (4, 2)]


def test_log_odds_probability():
    assert math.isclose(probability_from_log_odds(0.0), 0.5)
    assert probability_from_log_odds(-3.0) < 0.1
    assert probability_from_log_odds(3.0) > 0.9


def test_heatmap_free_is_bluer_and_occupied_is_darker_purple():
    free = heatmap_rgba(0.05)
    occupied = heatmap_rgba(0.95)
    assert free[2] > free[0]
    assert occupied[0] > occupied[1]
    assert free[3] > 0.8
    assert occupied[3] > 0.8


def test_world_to_grid_uses_floor():
    assert world_to_grid(0.19, 0.21, 0.0, 0.0, 0.1) == (1, 2)


def test_raycast_hits_nearby_wall():
    occupied = np.zeros((8, 8), dtype=bool)
    occupied[4, 6] = True
    ranges, increment = raycast_lidar_ranges(
        occupied,
        cell_size_m=0.5,
        origin_x=0.0,
        origin_y=0.0,
        x=2.25,
        y=2.25,
        yaw=0.0,
        max_range_m=3.5,
        beam_count=8,
        angle_min=-0.1,
        angle_max=0.1,
    )
    assert len(ranges) == 8
    assert increment > 0.0
    assert min(ranges) < 3.0


def test_load_movingai_occupancy_reads_room_map():
    map_path = Path(__file__).resolve().parents[1] / "worlds" / "movingai_mapf" / "room-32-32-4.map"
    occupied, width, height = load_movingai_occupancy(str(map_path))
    assert width == 32
    assert height == 32
    assert occupied.shape == (32, 32)
    assert occupied.any()
    assert (~occupied).any()
