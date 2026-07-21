import math

from trust_costmap.mapping_utils import (
    aggregate_log_odds,
    bresenham_cells,
    cell_to_world,
    logistic,
    logit,
    transform_local_odometry_to_map,
    world_to_cell,
)


def test_bresenham_cardinal_and_diagonal():
    assert bresenham_cells((1, 1), (1, 4)) == [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ]
    assert bresenham_cells((0, 0), (3, 3)) == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]


def test_probability_round_trip():
    for probability in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert math.isclose(logistic(logit(probability)), probability, abs_tol=1e-9)


def test_spawn_offset_pose_conversion():
    world = transform_local_odometry_to_map(
        local_pose=(1.0, 0.0, math.pi / 4.0),
        initial_local_pose=(0.0, 0.0, 0.0),
        spawn_pose=(2.0, 3.0, math.pi / 2.0),
    )
    assert math.isclose(world[0], 2.0, abs_tol=1e-9)
    assert math.isclose(world[1], 4.0, abs_tol=1e-9)
    assert math.isclose(world[2], 3.0 * math.pi / 4.0, abs_tol=1e-9)


def test_cell_coordinate_round_trip():
    x, y = cell_to_world(4, 7, 0.5)
    assert world_to_cell(x, y, 0.5) == (4, 7)


def test_aggregate_preserves_unknown_cells():
    values, observed = aggregate_log_odds(
        [([1.0, 2.0, 0.0], bytearray([1, 0, 0]), 1.0)],
        size=3,
        maximum_absolute_log_odds=8.0,
    )
    assert values == [1.0, 0.0, 0.0]
    assert list(observed) == [1, 0, 0]
