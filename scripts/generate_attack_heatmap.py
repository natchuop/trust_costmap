#!/usr/bin/env python3
"""Generate a bounded topology reconnaissance heatmap for fake-obstacle attacks.

The generator intentionally mirrors the experiment manager's action-goal
selection rules so a launch using the same map, goal count, and goal seed
produces the same goal cells. It uses only static topology and scenario robot
endpoints; no live route, trust, claim, or defense state is inspected.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

Cell = Tuple[int, int]
FREE_SYMBOLS = {".", "G", "S"}


@dataclass(frozen=True)
class MapData:
    name: str
    map_type: str
    height: int
    width: int
    grid: List[str]

    def is_free(self, cell: Cell) -> bool:
        row, col = cell
        return (
            0 <= row < self.height
            and 0 <= col < self.width
            and self.grid[row][col] in FREE_SYMBOLS
        )

    def free_cells(self) -> List[Cell]:
        return [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if self.is_free((row, col))
        ]


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_movingai_map(path: Path) -> MapData:
    lines = path.read_text(encoding="utf-8").splitlines()
    map_type: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    map_start: Optional[int] = None

    for index, line in enumerate(lines):
        clean = line.strip()
        if clean.startswith("type "):
            map_type = clean.split(maxsplit=1)[1]
        elif clean.startswith("height "):
            height = int(clean.split(maxsplit=1)[1])
        elif clean.startswith("width "):
            width = int(clean.split(maxsplit=1)[1])
        elif clean == "map":
            map_start = index + 1
            break

    if map_type is None or height is None or width is None or map_start is None:
        raise ValueError(f"Incomplete MovingAI header: {path}")

    grid = lines[map_start : map_start + height]
    if len(grid) != height:
        raise ValueError(f"Expected {height} map rows, found {len(grid)}")
    for row_index, row in enumerate(grid):
        if len(row) != width:
            raise ValueError(
                f"Map row {row_index} expected width {width}, found {len(row)}"
            )

    return MapData(path.stem, map_type, height, width, grid)


def load_scenario(path: Path) -> Dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Scenario must contain a YAML mapping: {path}")
    return data


def enabled_robot_endpoints(scenario: Dict) -> Tuple[List[Cell], Set[Cell]]:
    starts: List[Cell] = []
    reserved: Set[Cell] = set()
    for robot in scenario.get("robots", []):
        if not robot.get("enabled", True):
            continue
        start = tuple(int(value) for value in robot["start_cell"])
        starts.append(start)  # type: ignore[arg-type]
        reserved.add(start)  # type: ignore[arg-type]
        goal = robot.get("goal_cell")
        if goal is not None:
            reserved.add((int(goal[0]), int(goal[1])))
    if not starts:
        raise ValueError("Scenario must contain at least one enabled robot.")
    return starts, reserved


def reachable_free_cells(map_data: MapData, start: Cell) -> Set[Cell]:
    if not map_data.is_free(start):
        return set()
    visited = {start}
    frontier = [start]
    while frontier:
        row, col = frontier.pop()
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + d_row, col + d_col)
            if neighbor not in visited and map_data.is_free(neighbor):
                visited.add(neighbor)
                frontier.append(neighbor)
    return visited


def generate_action_goals(
    map_data: MapData,
    scenario: Dict,
    count_override: int,
    seed_override: int,
) -> Tuple[List[Cell], int, int]:
    config = scenario.get("action_goals", {})
    configured = config.get("cells", [])
    if configured:
        goals = [(int(cell[0]), int(cell[1])) for cell in configured]
        return goals, len(goals), seed_override if seed_override >= 0 else int(config.get("seed", 0))

    count = count_override if count_override >= 0 else int(config.get("count", 0))
    seed = seed_override if seed_override >= 0 else int(config.get("seed", 0))
    min_spacing = max(0.0, float(config.get("min_spacing_cells", 3.0)))

    starts, reserved = enabled_robot_endpoints(scenario)
    reachable_sets = [reachable_free_cells(map_data, start) for start in starts]
    common_reachable = set.intersection(*reachable_sets) if reachable_sets else set()
    candidates = sorted(common_reachable - reserved)

    if count > len(candidates):
        raise ValueError(
            f"Requested {count} action goals, but only {len(candidates)} candidates exist."
        )

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected: List[Cell] = []
    for candidate in candidates:
        if all(math.dist(candidate, existing) >= min_spacing for existing in selected):
            selected.append(candidate)
            if len(selected) == count:
                break

    if len(selected) < count:
        for candidate in candidates:
            if candidate not in selected:
                selected.append(candidate)
                if len(selected) == count:
                    break

    return selected, count, seed


def neighbors(
    map_data: MapData,
    cell: Cell,
    allow_diagonal: bool,
    blocked: Optional[Cell] = None,
) -> Iterable[Tuple[Cell, float]]:
    row, col = cell
    for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        candidate = (row + d_row, col + d_col)
        if candidate != blocked and map_data.is_free(candidate):
            yield candidate, 1.0

    if not allow_diagonal:
        return

    for d_row, d_col in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        candidate = (row + d_row, col + d_col)
        if candidate == blocked or not map_data.is_free(candidate):
            continue
        side_a = (row + d_row, col)
        side_b = (row, col + d_col)
        if side_a == blocked or side_b == blocked:
            continue
        if not map_data.is_free(side_a) or not map_data.is_free(side_b):
            continue
        yield candidate, math.sqrt(2.0)


def heuristic(a: Cell, b: Cell, allow_diagonal: bool) -> float:
    d_row = abs(a[0] - b[0])
    d_col = abs(a[1] - b[1])
    if allow_diagonal:
        diagonal = min(d_row, d_col)
        straight = max(d_row, d_col) - diagonal
        return math.sqrt(2.0) * diagonal + straight
    return float(d_row + d_col)


def astar(
    map_data: MapData,
    start: Cell,
    goal: Cell,
    allow_diagonal: bool,
    blocked: Optional[Cell] = None,
) -> Tuple[List[Cell], float]:
    if start == blocked or goal == blocked or not map_data.is_free(start) or not map_data.is_free(goal):
        return [], math.inf

    frontier: List[Tuple[float, int, Cell]] = [(0.0, 0, start)]
    came_from: Dict[Cell, Optional[Cell]] = {start: None}
    cost_so_far: Dict[Cell, float] = {start: 0.0}
    tie = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor, multiplier in neighbors(map_data, current, allow_diagonal, blocked):
            new_cost = cost_so_far[current] + multiplier
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                tie += 1
                priority = new_cost + heuristic(neighbor, goal, allow_diagonal)
                heapq.heappush(frontier, (priority, tie, neighbor))
                came_from[neighbor] = current

    if goal not in came_from:
        return [], math.inf

    path = [goal]
    current = goal
    while came_from[current] is not None:
        current = came_from[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path, cost_so_far[goal]


def normalize(values: Dict[Cell, float]) -> Dict[Cell, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high):
        return {cell: 0.0 for cell in values}
    return {cell: (value - low) / (high - low) for cell, value in values.items()}


def build_route_pairs(
    starts: Sequence[Cell],
    goals: Sequence[Cell],
    sample_count: int,
    seed: int,
) -> List[Tuple[Cell, Cell]]:
    origins = list(dict.fromkeys([*starts, *goals]))
    pairs = [(origin, goal) for origin in origins for goal in goals if origin != goal]
    if sample_count <= 0 or sample_count >= len(pairs):
        return pairs
    rng = random.Random(seed)
    return rng.sample(pairs, sample_count)


def compute_heatmap(
    map_data: MapData,
    starts: Sequence[Cell],
    goals: Sequence[Cell],
    allow_diagonal: bool,
    route_sample_count: int,
    reconnaissance_seed: int,
    detour_candidate_count: int,
    endpoint_exclusion_radius: float,
    route_frequency_weight: float,
    detour_weight: float,
    chokepoint_weight: float,
    unreachable_penalty: float,
) -> Dict:
    pairs = build_route_pairs(starts, goals, route_sample_count, reconnaissance_seed)
    route_counts: Dict[Cell, float] = {cell: 0.0 for cell in map_data.free_cells()}
    route_records: List[Tuple[Cell, Cell, List[Cell], float]] = []

    for start, goal in pairs:
        path, cost = astar(map_data, start, goal, allow_diagonal)
        if not path:
            continue
        route_records.append((start, goal, path, cost))
        for cell in path[1:-1]:
            route_counts[cell] += 1.0

    successful_routes = len(route_records)
    frequency = {
        cell: count / max(1, successful_routes)
        for cell, count in route_counts.items()
    }

    excluded_endpoints = set(starts) | set(goals)
    eligible = [
        cell
        for cell, count in route_counts.items()
        if count > 0.0
        and all(math.dist(cell, endpoint) > endpoint_exclusion_radius for endpoint in excluded_endpoints)
    ]
    eligible.sort(key=lambda cell: (-route_counts[cell], cell))
    detour_candidates = eligible[: max(0, detour_candidate_count)]

    detour_raw: Dict[Cell, float] = {cell: 0.0 for cell in route_counts}
    for candidate in detour_candidates:
        impacts: List[float] = []
        for start, goal, path, original_cost in route_records:
            if candidate not in path[1:-1]:
                continue
            _, blocked_cost = astar(
                map_data, start, goal, allow_diagonal, blocked=candidate
            )
            if not math.isfinite(blocked_cost):
                impacts.append(unreachable_penalty)
            else:
                impacts.append(max(0.0, (blocked_cost - original_cost) / max(original_cost, 1e-9)))
        if impacts:
            detour_raw[candidate] = sum(impacts) / len(impacts)

    chokepoint_raw: Dict[Cell, float] = {}
    max_degree = 8.0 if allow_diagonal else 4.0
    for cell in route_counts:
        degree = sum(1 for _ in neighbors(map_data, cell, allow_diagonal))
        chokepoint_raw[cell] = 1.0 - min(1.0, degree / max_degree)

    frequency_norm = normalize(frequency)
    detour_norm = normalize(detour_raw)
    chokepoint_norm = normalize(chokepoint_raw)

    weight_sum = route_frequency_weight + detour_weight + chokepoint_weight
    if weight_sum <= 0.0:
        raise ValueError("At least one heatmap component weight must be positive.")

    scores: Dict[Cell, float] = {}
    for cell in route_counts:
        score = (
            route_frequency_weight * frequency_norm[cell]
            + detour_weight * detour_norm[cell]
            + chokepoint_weight * chokepoint_norm[cell]
        ) / weight_sum
        if cell not in eligible:
            score = 0.0
        scores[cell] = score

    ranked = sorted(scores, key=lambda cell: (-scores[cell], cell))
    cells = [
        {
            "row": cell[0],
            "col": cell[1],
            "score": round(scores[cell], 9),
            "route_frequency": round(frequency[cell], 9),
            "route_frequency_normalized": round(frequency_norm[cell], 9),
            "detour_impact": round(detour_raw[cell], 9),
            "detour_normalized": round(detour_norm[cell], 9),
            "chokepoint": round(chokepoint_raw[cell], 9),
            "chokepoint_normalized": round(chokepoint_norm[cell], 9),
            "evaluated_for_detour": cell in detour_candidates,
        }
        for cell in ranked
    ]

    return {
        "schema_version": 1,
        "knowledge_model": "topology_only",
        "map_name": map_data.name,
        "map_type": map_data.map_type,
        "height": map_data.height,
        "width": map_data.width,
        "allow_diagonal": allow_diagonal,
        "action_goals": [list(cell) for cell in goals],
        "robot_start_cells": [list(cell) for cell in starts],
        "route_pair_count_requested": len(pairs),
        "successful_route_count": successful_routes,
        "detour_candidate_count": len(detour_candidates),
        "weights": {
            "route_frequency": route_frequency_weight,
            "detour": detour_weight,
            "chokepoint": chokepoint_weight,
        },
        "cells": cells,
    }


def write_pgm(path: Path, map_data: MapData, payload: Dict) -> None:
    score_by_cell = {
        (int(item["row"]), int(item["col"])): float(item["score"])
        for item in payload["cells"]
    }
    rows: List[str] = []
    for row in range(map_data.height - 1, -1, -1):
        pixels = []
        for col in range(map_data.width):
            cell = (row, col)
            if not map_data.is_free(cell):
                pixels.append("0")
            else:
                pixels.append(str(int(round(32 + 223 * score_by_cell.get(cell, 0.0)))))
        rows.append(" ".join(pixels))
    path.write_text(
        "P2\n" + f"# attack heatmap for {map_data.name}\n" + f"{map_data.width} {map_data.height}\n255\n" + "\n".join(rows) + "\n",
        encoding="ascii",
    )


def write_csv(path: Path, payload: Dict) -> None:
    fields = [
        "row", "col", "score", "route_frequency", "route_frequency_normalized",
        "detour_impact", "detour_normalized", "chokepoint",
        "chokepoint_normalized", "evaluated_for_detour",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["cells"])



def payload_matrix(
    map_data: MapData,
    payload: Dict,
    field: str,
    default: float = 0.0,
):
    """Convert a per-cell payload field into a map-shaped NumPy array."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "PNG output requires NumPy. Install it with: "
            "sudo apt install python3-numpy"
        ) from exc

    matrix = np.full(
        (map_data.height, map_data.width),
        float(default),
        dtype=float,
    )
    for item in payload.get("cells", []):
        row = int(item["row"])
        col = int(item["col"])
        if 0 <= row < map_data.height and 0 <= col < map_data.width:
            matrix[row, col] = float(item.get(field, default))
    return matrix


def obstacle_mask(map_data: MapData):
    """Return a map-shaped mask where blocked cells are 1 and free cells are 0."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "PNG output requires NumPy. Install it with: "
            "sudo apt install python3-numpy"
        ) from exc

    mask = np.zeros((map_data.height, map_data.width), dtype=float)
    for row in range(map_data.height):
        for col in range(map_data.width):
            if not map_data.is_free((row, col)):
                mask[row, col] = 1.0
    return mask


def ranked_positive_cells(payload: Dict, limit: int) -> List[Dict]:
    """Return the highest-scoring positive cells in payload order."""
    return [
        item
        for item in payload.get("cells", [])
        if float(item.get("score", 0.0)) > 0.0
    ][: max(0, int(limit))]


def add_map_annotations(
    ax,
    payload: Dict,
    top_cells: Sequence[Dict],
    show_labels: bool = True,
) -> None:
    """Overlay starts, goals, and ranked attack cells on a Matplotlib axis."""
    starts = [tuple(cell) for cell in payload.get("robot_start_cells", [])]
    goals = [tuple(cell) for cell in payload.get("action_goals", [])]

    if starts:
        ax.scatter(
            [cell[1] for cell in starts],
            [cell[0] for cell in starts],
            marker="o",
            s=70,
            facecolors="none",
            edgecolors="deepskyblue",
            linewidths=1.8,
            label="Robot starts",
            zorder=5,
        )

    if goals:
        ax.scatter(
            [cell[1] for cell in goals],
            [cell[0] for cell in goals],
            marker="*",
            s=120,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.6,
            label="Action goals",
            zorder=6,
        )

    if top_cells:
        rows = [int(item["row"]) for item in top_cells]
        cols = [int(item["col"]) for item in top_cells]
        ax.scatter(
            cols,
            rows,
            marker="x",
            s=65,
            linewidths=1.8,
            color="lime",
            label="Top attack cells",
            zorder=7,
        )

        if show_labels:
            for rank, item in enumerate(top_cells, start=1):
                row = int(item["row"])
                col = int(item["col"])
                ax.text(
                    col + 0.25,
                    row - 0.25,
                    str(rank),
                    fontsize=7,
                    color="lime",
                    weight="bold",
                    zorder=8,
                )


def configure_grid_axis(ax, map_data: MapData) -> None:
    """Apply grid-cell coordinates and a light minor grid."""
    import numpy as np

    ax.set_xlim(-0.5, map_data.width - 0.5)
    ax.set_ylim(map_data.height - 0.5, -0.5)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(np.arange(-0.5, map_data.width, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, map_data.height, 1.0), minor=True)
    ax.grid(which="minor", linewidth=0.18, alpha=0.22)
    ax.tick_params(which="minor", bottom=False, left=False)


def write_png(
    path: Path,
    map_data: MapData,
    payload: Dict,
    top_cell_count: int = 10,
    dpi: int = 200,
) -> None:
    """Write the combined reconnaissance score as a publication-friendly PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "PNG output requires Matplotlib and NumPy. Install them with: "
            "sudo apt install python3-matplotlib python3-numpy"
        ) from exc

    scores = payload_matrix(map_data, payload, "score")
    blocked = obstacle_mask(map_data)
    visible_scores = np.ma.masked_where(scores <= 0.0, scores)
    top_cells = ranked_positive_cells(payload, top_cell_count)

    figure_width = max(8.0, min(16.0, map_data.width / 3.2))
    figure_height = max(7.0, min(14.0, map_data.height / 3.2))
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    ax.imshow(
        blocked,
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        origin="upper",
        alpha=0.92,
    )
    heat_image = ax.imshow(
        visible_scores,
        cmap="hot",
        vmin=0.0,
        vmax=max(1.0, float(scores.max())),
        interpolation="nearest",
        origin="upper",
        alpha=0.82,
    )

    add_map_annotations(ax, payload, top_cells, show_labels=True)
    configure_grid_axis(ax, map_data)
    ax.set_title(
        "Attack Reconnaissance Heatmap\n"
        "Combined route frequency, detour impact, and chokepoint score"
    )

    if np.any(scores > 0.0):
        colorbar = fig.colorbar(heat_image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Combined attack score")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=max(72, int(dpi)), bbox_inches="tight")
    plt.close(fig)


def write_component_png(
    path: Path,
    map_data: MapData,
    payload: Dict,
    top_cell_count: int = 10,
    dpi: int = 200,
) -> None:
    """Write a four-panel PNG exposing each heatmap component and final score."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "PNG output requires Matplotlib and NumPy. Install them with: "
            "sudo apt install python3-matplotlib python3-numpy"
        ) from exc

    blocked = obstacle_mask(map_data)
    top_cells = ranked_positive_cells(payload, top_cell_count)
    panels = [
        ("route_frequency_normalized", "Normalized route frequency"),
        ("detour_normalized", "Normalized detour impact"),
        ("chokepoint_normalized", "Normalized chokepoint score"),
        ("score", "Combined attack score"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for ax, (field, title) in zip(axes.flat, panels):
        values = payload_matrix(map_data, payload, field)
        visible = np.ma.masked_where(values <= 0.0, values)
        ax.imshow(
            blocked,
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            origin="upper",
            alpha=0.92,
        )
        image = ax.imshow(
            visible,
            cmap="hot",
            vmin=0.0,
            vmax=max(1.0, float(values.max())),
            interpolation="nearest",
            origin="upper",
            alpha=0.82,
        )
        add_map_annotations(
            ax,
            payload,
            top_cells if field == "score" else [],
            show_labels=False,
        )
        configure_grid_axis(ax, map_data)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Attack Reconnaissance Heatmap Components",
        fontsize=15,
    )
    fig.savefig(path, dpi=max(72, int(dpi)), bbox_inches="tight")
    plt.close(fig)


def heatmap_statistics(payload: Dict) -> Dict[str, float]:
    """Return compact diagnostics used to detect empty or uniform heatmaps."""
    values = [float(item.get("score", 0.0)) for item in payload.get("cells", [])]
    positive = [value for value in values if value > 0.0]
    return {
        "minimum": min(values) if values else 0.0,
        "maximum": max(values) if values else 0.0,
        "sum": sum(values),
        "positive_cell_count": float(len(positive)),
        "mean_positive": (sum(positive) / len(positive)) if positive else 0.0,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-path", required=True, type=Path)
    parser.add_argument("--scenario-path", required=True, type=Path)
    parser.add_argument("--action-goal-count", type=int, default=-1)
    parser.add_argument("--action-goal-seed", type=int, default=-1)
    parser.add_argument("--allow-diagonal", default="false")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--route-sample-count", type=int, default=-1)
    parser.add_argument("--reconnaissance-seed", type=int, default=-1)
    parser.add_argument("--detour-candidate-count", type=int, default=-1)
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Disable PNG visualization output.",
    )
    parser.add_argument(
        "--png-top-cell-count",
        type=int,
        default=10,
        help="Number of highest-scoring cells to mark and rank on the PNG.",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=200,
        help="Resolution of generated PNG files.",
    )
    args = parser.parse_args()

    map_data = load_movingai_map(args.map_path)
    scenario = load_scenario(args.scenario_path)
    config = scenario.get("attack_reconnaissance", {})
    goals, resolved_count, resolved_goal_seed = generate_action_goals(
        map_data, scenario, args.action_goal_count, args.action_goal_seed
    )
    starts, _ = enabled_robot_endpoints(scenario)

    route_sample_count = (
        args.route_sample_count
        if args.route_sample_count >= 0
        else int(config.get("route_sample_count", 100))
    )
    reconnaissance_seed = (
        args.reconnaissance_seed
        if args.reconnaissance_seed >= 0
        else int(config.get("seed", resolved_goal_seed))
    )
    detour_candidate_count = (
        args.detour_candidate_count
        if args.detour_candidate_count >= 0
        else int(config.get("detour_candidate_count", 40))
    )

    payload = compute_heatmap(
        map_data=map_data,
        starts=starts,
        goals=goals,
        allow_diagonal=parse_bool(args.allow_diagonal),
        route_sample_count=route_sample_count,
        reconnaissance_seed=reconnaissance_seed,
        detour_candidate_count=detour_candidate_count,
        endpoint_exclusion_radius=float(config.get("endpoint_exclusion_radius_cells", 2.0)),
        route_frequency_weight=float(config.get("route_frequency_weight", 0.40)),
        detour_weight=float(config.get("detour_weight", 0.45)),
        chokepoint_weight=float(config.get("chokepoint_weight", 0.15)),
        unreachable_penalty=float(config.get("unreachable_penalty", 2.0)),
    )
    payload["action_goal_count"] = resolved_count
    payload["action_goal_seed"] = resolved_goal_seed
    payload["reconnaissance_seed"] = reconnaissance_seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{map_data.name}_goals-{resolved_count}_goal-seed-{resolved_goal_seed}"
        f"_recon-seed-{reconnaissance_seed}"
    )
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    pgm_path = args.output_dir / f"{stem}.pgm"
    png_path = args.output_dir / f"{stem}.png"
    component_png_path = args.output_dir / f"{stem}_components.png"

    statistics = heatmap_statistics(payload)
    payload["statistics"] = {
        "minimum_score": round(statistics["minimum"], 9),
        "maximum_score": round(statistics["maximum"], 9),
        "score_sum": round(statistics["sum"], 9),
        "positive_cell_count": int(statistics["positive_cell_count"]),
        "mean_positive_score": round(statistics["mean_positive"], 9),
    }

    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, payload)
    write_pgm(pgm_path, map_data, payload)

    if not args.no_png:
        write_png(
            png_path,
            map_data,
            payload,
            top_cell_count=args.png_top_cell_count,
            dpi=args.png_dpi,
        )
        write_component_png(
            component_png_path,
            map_data,
            payload,
            top_cell_count=args.png_top_cell_count,
            dpi=args.png_dpi,
        )

    top = ranked_positive_cells(payload, 10)
    print(f"[attack_heatmap] map={map_data.name} goals={goals}")
    print(
        f"[attack_heatmap] routes={payload['successful_route_count']} "
        f"detour_candidates={payload['detour_candidate_count']}"
    )
    print(f"[attack_heatmap] JSON: {json_path}")
    print(f"[attack_heatmap] CSV:  {csv_path}")
    print(f"[attack_heatmap] PGM:  {pgm_path}")
    if not args.no_png:
        print(f"[attack_heatmap] PNG:  {png_path}")
        print(f"[attack_heatmap] component PNG: {component_png_path}")
    print(
        "[attack_heatmap] score stats: "
        f"min={statistics['minimum']:.3f} "
        f"max={statistics['maximum']:.3f} "
        f"positive={int(statistics['positive_cell_count'])} "
        f"sum={statistics['sum']:.3f} "
        f"mean_positive={statistics['mean_positive']:.3f}"
    )
    if statistics["positive_cell_count"] == 0:
        print(
            "[attack_heatmap] WARNING: heatmap contains no positive cells. "
            "Check route sampling, connectivity, and endpoint exclusion settings."
        )
    print("[attack_heatmap] top cells: " + ", ".join(
        f"({item['row']},{item['col']})={item['score']:.3f}" for item in top
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
