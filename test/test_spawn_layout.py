from random import Random

import pytest

from trust_costmap.spawn_layout import (
    FREE_SYMBOLS,
    build_spawn_layout,
    choose_spread_cells,
    largest_free_component,
)


def test_layout_is_reproducible_unique_and_free():
    grid = [
        "........",
        "..@@....",
        "..@@....",
        "........",
        "....@@..",
        "....@@..",
        "........",
        "........",
    ]

    first = build_spawn_layout(grid, ["r1", "r2", "r3"], 6, action_goal_seed=21)
    second = build_spawn_layout(grid, ["r1", "r2", "r3"], 6, action_goal_seed=21)

    assert first == second

    all_cells = list(first.robot_cells.values()) + first.action_goal_cells
    assert len(all_cells) == len(set(all_cells))
    assert all(grid[row][col] in FREE_SYMBOLS for row, col in all_cells)


def test_goal_cells_do_not_change_when_robot_count_changes():
    grid = [".........." for _ in range(10)]

    two_robots = build_spawn_layout(grid, ["r1", "r2"], 8, action_goal_seed=21)
    four_robots = build_spawn_layout(
        grid,
        ["r1", "r2", "r3", "r4"],
        8,
        action_goal_seed=21,
    )

    assert two_robots.action_goal_cells == four_robots.action_goal_cells


def test_largest_connected_component_is_used():
    grid = [
        "..@@....",
        "..@@....",
        "@@@@@@@@",
        ".@@@@@@@",
    ]

    component = largest_free_component(grid)

    assert len(component) == 8
    assert all(col >= 4 for _, col in component)


def test_spread_selection_rejects_too_many_cells():
    with pytest.raises(ValueError, match="only 2 candidates"):
        choose_spread_cells([(0, 0), (0, 1)], 3, Random(1))


def test_layout_rejects_map_without_enough_connected_space():
    grid = [".@."]

    with pytest.raises(ValueError, match="need 3, found 1"):
        build_spawn_layout(grid, ["r1", "r2"], 1, action_goal_seed=1)


def test_action_goal_count_controls_exact_number_of_goals():
    grid = ["............" for _ in range(12)]

    layout = build_spawn_layout(
        grid,
        ["r1", "r2", "r3"],
        action_goal_count=8,
        action_goal_seed=21,
    )

    assert len(layout.action_goal_cells) == 8


def test_action_goal_seed_changes_goal_and_robot_layout():
    grid = ["............" for _ in range(12)]

    seed_21 = build_spawn_layout(
        grid,
        ["r1", "r2", "r3"],
        action_goal_count=8,
        action_goal_seed=21,
    )
    seed_22 = build_spawn_layout(
        grid,
        ["r1", "r2", "r3"],
        action_goal_count=8,
        action_goal_seed=22,
    )

    assert seed_21.action_goal_cells != seed_22.action_goal_cells
    assert seed_21.robot_cells != seed_22.robot_cells
