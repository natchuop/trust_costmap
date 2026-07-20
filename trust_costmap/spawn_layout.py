from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Dict, Iterable, List, Sequence, Tuple

FREE_SYMBOLS = frozenset({".", "G", "S"})
Cell = Tuple[int, int]


@dataclass(frozen=True)
class SpawnLayout:
    robot_cells: Dict[str, Cell]
    action_goal_cells: List[Cell]
    connected_free_cell_count: int


def largest_free_component(grid: Sequence[str]) -> List[Cell]:
    """Return the largest 4-connected group of free cells in row-major order."""
    free = {
        (row, col)
        for row, line in enumerate(grid)
        for col, symbol in enumerate(line)
        if symbol in FREE_SYMBOLS
    }
    visited = set()
    largest: List[Cell] = []

    for start in sorted(free):
        if start in visited:
            continue

        component: List[Cell] = []
        queue = deque([start])
        visited.add(start)

        while queue:
            row, col = queue.popleft()
            component.append((row, col))

            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbor in free and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(component) > len(largest):
            largest = component

    return largest


def choose_spread_cells(
    candidates: Iterable[Cell],
    count: int,
    rng: Random,
) -> List[Cell]:
    """Choose seeded-random cells using simple farthest-point sampling."""
    pool = list(candidates)

    if count < 0:
        raise ValueError("count cannot be negative")
    if count > len(pool):
        raise ValueError(
            f"Requested {count} cells, but only {len(pool)} candidates are available."
        )
    if count == 0:
        return []

    rng.shuffle(pool)
    selected = [pool.pop()]

    while len(selected) < count:
        best_index = max(
            range(len(pool)),
            key=lambda index: min(
                _distance_squared(pool[index], chosen) for chosen in selected
            ),
        )
        selected.append(pool.pop(best_index))

    return selected


def build_spawn_layout(
    grid: Sequence[str],
    robot_ids: Sequence[str],
    action_goal_count: int,
    action_goal_seed: int,
) -> SpawnLayout:
    """Create one reproducible layout for robots and visual action goals."""
    component = largest_free_component(grid)
    required_cells = len(robot_ids) + action_goal_count

    if not component:
        raise ValueError("The selected map contains no free cells.")
    if required_cells > len(component):
        raise ValueError(
            "Not enough connected free cells for the requested layout: "
            f"need {required_cells}, found {len(component)}."
        )

    goal_rng = Random(action_goal_seed)
    action_goal_cells = choose_spread_cells(
        component,
        action_goal_count,
        goal_rng,
    )
    action_goal_set = set(action_goal_cells)

    robot_candidates = [cell for cell in component if cell not in action_goal_set]
    robot_rng = Random(action_goal_seed + 1)
    robot_cells_list = robot_rng.sample(robot_candidates, k=len(robot_ids))
    robot_cells = dict(zip(robot_ids, robot_cells_list))

    return SpawnLayout(
        robot_cells=robot_cells,
        action_goal_cells=action_goal_cells,
        connected_free_cell_count=len(component),
    )


def _distance_squared(first: Cell, second: Cell) -> int:
    row_delta = first[0] - second[0]
    col_delta = first[1] - second[1]
    return row_delta * row_delta + col_delta * col_delta
