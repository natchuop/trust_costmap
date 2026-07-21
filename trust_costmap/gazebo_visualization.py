"""Gazebo-native visualization utilities for trust_costmap experiments.

This module intentionally contains no ROS Node subclass and no planning logic.

Responsibilities:
- Generate SDF for action goals, routes, claims, and dynamic obstacles.
- Write generated SDF files atomically.
- Spawn, replace, and remove Gazebo entities through Gazebo Transport services.
- Centralize colors, dimensions, naming rules, and service interaction.
- Provide reusable building blocks for future experiment visualizations.

Non-responsibilities:
- Selecting action goals.
- Planning routes.
- Updating trust.
- Deciding whether a claim is truthful or malicious.
- Publishing cmd_vel.
- Following waypoints.

Keeping those responsibilities separate allows the experiment manager to remain
focused on experiment state and methodology rather than XML and subprocess calls.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Cell = Tuple[int, int]
Point2D = Tuple[float, float]
Rgba = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GazeboVisualizationError(RuntimeError):
    """Base error for Gazebo visualization failures."""


class GazeboServiceError(GazeboVisualizationError):
    """Raised when a Gazebo Transport service call fails."""


class SdfValidationError(GazeboVisualizationError):
    """Raised when invalid visualization geometry is requested."""


# ---------------------------------------------------------------------------
# Configuration data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GazeboServiceConfig:
    """Settings used when communicating with Gazebo Transport services."""

    world_name: str
    timeout_ms: int = 1500
    executable: str = "gz"

    @property
    def create_service(self) -> str:
        return f"/world/{self.world_name}/create"

    @property
    def remove_service(self) -> str:
        return f"/world/{self.world_name}/remove"

    @property
    def pose_service(self) -> str:
        return f"/world/{self.world_name}/set_pose"


@dataclass(frozen=True)
class GoalVisualConfig:
    """Geometry and appearance of action-goal discs."""

    radius_m: float = 0.14
    height_m: float = 0.035
    z_m: float = 0.022
    color: Rgba = (1.0, 0.80, 0.0, 1.0)


@dataclass(frozen=True)
class RouteVisualConfig:
    """Geometry and appearance of planned route strips."""

    width_m: float = 0.05
    height_m: float = 0.012
    z_m: float = 0.012
    color: Rgba = (0.05, 0.75, 0.25, 0.85)
    draw_waypoint_discs: bool = False
    waypoint_radius_m: float = 0.035


@dataclass(frozen=True)
class ClaimVisualConfig:
    """Geometry and appearance of reported occupancy claims."""

    radius_m: float = 0.12
    height_m: float = 0.08
    z_m: float = 0.045

    occupied_color: Rgba = (0.95, 0.10, 0.10, 0.82)
    free_color: Rgba = (0.10, 0.55, 1.0, 0.70)
    disputed_color: Rgba = (0.75, 0.10, 0.90, 0.82)


@dataclass(frozen=True)
class ObstacleVisualConfig:
    """Geometry and appearance of experiment-controlled obstacles."""

    height_m: float = 0.25
    z_m: float = 0.125
    color: Rgba = (0.85, 0.20, 0.08, 1.0)
    collision_enabled: bool = True


@dataclass(frozen=True)
class RouteStyle:
    """Named route appearance for a robot or role."""

    color: Rgba
    width_scale: float = 1.0
    z_offset_m: float = 0.0


@dataclass(frozen=True)
class SpawnResult:
    """Outcome of one entity operation."""

    entity_name: str
    success: bool
    return_code: int
    stdout: str
    stderr: str


# ---------------------------------------------------------------------------
# General validation and formatting
# ---------------------------------------------------------------------------


def sanitize_entity_name(value: str, fallback: str = "entity") -> str:
    """Convert arbitrary text into a Gazebo-safe entity name."""

    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip())
    cleaned = cleaned.strip("_")

    if not cleaned:
        cleaned = fallback

    if cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}"

    return cleaned


def make_world_name(map_name: str) -> str:
    """Return the same sanitized world name used by the launch file."""

    return f"{sanitize_entity_name(map_name, fallback='map')}_world"


def rgba_text(color: Rgba) -> str:
    """Convert an RGBA tuple into SDF-compatible text."""

    if len(color) != 4:
        raise SdfValidationError(f"RGBA color must contain four values: {color}")

    values = tuple(float(value) for value in color)

    if any(not math.isfinite(value) for value in values):
        raise SdfValidationError(f"RGBA color must be finite: {color}")

    if any(value < 0.0 or value > 1.0 for value in values):
        raise SdfValidationError(
            f"RGBA values must be between 0 and 1: {color}"
        )

    return " ".join(f"{value:.6f}" for value in values)


def validate_positive(name: str, value: float) -> float:
    """Validate and return a strictly positive finite number."""

    converted = float(value)

    if not math.isfinite(converted) or converted <= 0.0:
        raise SdfValidationError(
            f"{name} must be finite and greater than zero, got {value}"
        )

    return converted


def validate_nonnegative(name: str, value: float) -> float:
    """Validate and return a finite nonnegative number."""

    converted = float(value)

    if not math.isfinite(converted) or converted < 0.0:
        raise SdfValidationError(
            f"{name} must be finite and nonnegative, got {value}"
        )

    return converted


def cell_to_world(row: int, col: int, cell_size_m: float) -> Point2D:
    """Convert a MovingAI grid cell into the center of a Gazebo cell."""

    cell_size = validate_positive("cell_size_m", cell_size_m)

    x = (int(col) + 0.5) * cell_size
    y = (int(row) + 0.5) * cell_size

    return x, y


def stable_content_hash(text: str, length: int = 16) -> str:
    """Generate a deterministic digest for visualization revision tracking."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------


def make_material_xml(color: Rgba, indent: str = "") -> str:
    """Create a simple Gazebo material block."""

    text = rgba_text(color)

    return "\n".join(
        [
            f"{indent}<material>",
            f"{indent}  <ambient>{text}</ambient>",
            f"{indent}  <diffuse>{text}</diffuse>",
            f"{indent}  <specular>0.1 0.1 0.1 1</specular>",
            f"{indent}</material>",
        ]
    )


def make_cylinder_visual_xml(
    *,
    visual_name: str,
    x: float,
    y: float,
    z: float,
    radius_m: float,
    height_m: float,
    color: Rgba,
    indent: str = "      ",
) -> str:
    """Create one visual-only cylinder."""

    radius = validate_positive("radius_m", radius_m)
    height = validate_positive("height_m", height_m)
    z_value = validate_nonnegative("z", z)

    name = sanitize_entity_name(visual_name, fallback="cylinder")

    return "\n".join(
        [
            f'{indent}<visual name="{name}">',
            f"{indent}  <pose>{float(x):.6f} {float(y):.6f} "
            f"{z_value:.6f} 0 0 0</pose>",
            f"{indent}  <geometry>",
            f"{indent}    <cylinder>",
            f"{indent}      <radius>{radius:.6f}</radius>",
            f"{indent}      <length>{height:.6f}</length>",
            f"{indent}    </cylinder>",
            f"{indent}  </geometry>",
            make_material_xml(color, indent=f"{indent}  "),
            f"{indent}</visual>",
        ]
    )


def make_box_visual_xml(
    *,
    visual_name: str,
    x: float,
    y: float,
    z: float,
    size_x_m: float,
    size_y_m: float,
    size_z_m: float,
    yaw_rad: float,
    color: Rgba,
    indent: str = "      ",
) -> str:
    """Create one visual-only box."""

    size_x = validate_positive("size_x_m", size_x_m)
    size_y = validate_positive("size_y_m", size_y_m)
    size_z = validate_positive("size_z_m", size_z_m)
    z_value = validate_nonnegative("z", z)

    name = sanitize_entity_name(visual_name, fallback="box")

    return "\n".join(
        [
            f'{indent}<visual name="{name}">',
            f"{indent}  <pose>{float(x):.6f} {float(y):.6f} "
            f"{z_value:.6f} 0 0 {float(yaw_rad):.6f}</pose>",
            f"{indent}  <geometry>",
            f"{indent}    <box>",
            f"{indent}      <size>{size_x:.6f} {size_y:.6f} "
            f"{size_z:.6f}</size>",
            f"{indent}    </box>",
            f"{indent}  </geometry>",
            make_material_xml(color, indent=f"{indent}  "),
            f"{indent}</visual>",
        ]
    )


def make_box_collision_xml(
    *,
    collision_name: str,
    size_x_m: float,
    size_y_m: float,
    size_z_m: float,
    indent: str = "      ",
) -> str:
    """Create collision geometry for a physical experiment obstacle."""

    size_x = validate_positive("size_x_m", size_x_m)
    size_y = validate_positive("size_y_m", size_y_m)
    size_z = validate_positive("size_z_m", size_z_m)

    name = sanitize_entity_name(collision_name, fallback="collision")

    return "\n".join(
        [
            f'{indent}<collision name="{name}">',
            f"{indent}  <geometry>",
            f"{indent}    <box>",
            f"{indent}      <size>{size_x:.6f} {size_y:.6f} "
            f"{size_z:.6f}</size>",
            f"{indent}    </box>",
            f"{indent}  </geometry>",
            f"{indent}</collision>",
        ]
    )


def wrap_visuals_in_static_model(
    *,
    model_name: str,
    visual_fragments: Sequence[str],
    collision_fragments: Sequence[str] = (),
    pose: Tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
) -> str:
    """Wrap visual and optional collision fragments in a complete SDF model."""

    name = sanitize_entity_name(model_name, fallback="visualization")

    if not visual_fragments and not collision_fragments:
        raise SdfValidationError(
            f"Model {name} must contain at least one visual or collision"
        )

    pose_text = " ".join(f"{float(value):.6f}" for value in pose)

    content = [
        '<?xml version="1.0"?>',
        '<sdf version="1.9">',
        f'  <model name="{name}">',
        "    <static>true</static>",
        f"    <pose>{pose_text}</pose>",
        '    <link name="visualization_link">',
    ]

    content.extend(visual_fragments)
    content.extend(collision_fragments)

    content.extend(
        [
            "    </link>",
            "  </model>",
            "</sdf>",
            "",
        ]
    )

    return "\n".join(content)


# ---------------------------------------------------------------------------
# Action-goal visualization
# ---------------------------------------------------------------------------


def make_action_goals_sdf(
    *,
    model_name: str,
    goals: Sequence[Cell],
    cell_size_m: float,
    config: GoalVisualConfig = GoalVisualConfig(),
) -> str:
    """Create one model containing all action-goal disc visuals."""

    cell_size = validate_positive("cell_size_m", cell_size_m)
    visuals: List[str] = []

    for index, cell in enumerate(goals):
        row, col = int(cell[0]), int(cell[1])
        x, y = cell_to_world(row, col, cell_size)

        visuals.append(
            make_cylinder_visual_xml(
                visual_name=f"goal_{index + 1}",
                x=x,
                y=y,
                z=config.z_m,
                radius_m=config.radius_m,
                height_m=config.height_m,
                color=config.color,
            )
        )

    if not visuals:
        # Gazebo cannot spawn an empty model. The caller should remove the
        # existing goal entity instead of attempting to spawn this.
        raise SdfValidationError("Cannot create action-goal SDF with no goals")

    return wrap_visuals_in_static_model(
        model_name=model_name,
        visual_fragments=visuals,
    )


# ---------------------------------------------------------------------------
# Route visualization
# ---------------------------------------------------------------------------


def route_role_style(role: str) -> RouteStyle:
    """Return a default route style based on experiment role."""

    normalized = str(role).strip().lower()

    if "malicious" in normalized or "attacker" in normalized:
        return RouteStyle(color=(0.92, 0.08, 0.08, 0.88))

    if "reporter" in normalized:
        return RouteStyle(color=(0.08, 0.32, 0.95, 0.86))

    return RouteStyle(color=(0.05, 0.78, 0.25, 0.88))


def make_route_segment_xml(
    *,
    name: str,
    start: Point2D,
    end: Point2D,
    config: RouteVisualConfig,
    index: int,
) -> str:
    """Create one box visual connecting two route points."""

    start_x, start_y = start
    end_x, end_y = end

    delta_x = end_x - start_x
    delta_y = end_y - start_y
    length = math.hypot(delta_x, delta_y)

    if length <= 1e-9:
        raise SdfValidationError(
            f"Route segment {index} has zero length: {start} -> {end}"
        )

    center_x = (start_x + end_x) / 2.0
    center_y = (start_y + end_y) / 2.0
    yaw = math.atan2(delta_y, delta_x)

    return make_box_visual_xml(
        visual_name=f"{name}_segment_{index}",
        x=center_x,
        y=center_y,
        z=config.z_m,
        size_x_m=length,
        size_y_m=config.width_m,
        size_z_m=config.height_m,
        yaw_rad=yaw,
        color=config.color,
    )


def make_route_sdf(
    *,
    model_name: str,
    route: Sequence[Cell],
    cell_size_m: float,
    config: RouteVisualConfig = RouteVisualConfig(),
) -> str:
    """Create one model containing all segments of a planned grid route."""

    cell_size = validate_positive("cell_size_m", cell_size_m)

    if not route:
        raise SdfValidationError("Cannot create route SDF from an empty route")

    points = [
        cell_to_world(int(row), int(col), cell_size)
        for row, col in route
    ]

    visuals: List[str] = []
    safe_name = sanitize_entity_name(model_name, fallback="route")

    for index, (start, end) in enumerate(zip(points, points[1:])):
        if start == end:
            continue

        visuals.append(
            make_route_segment_xml(
                name=safe_name,
                start=start,
                end=end,
                config=config,
                index=index,
            )
        )

    if config.draw_waypoint_discs:
        for index, (x, y) in enumerate(points):
            visuals.append(
                make_cylinder_visual_xml(
                    visual_name=f"{safe_name}_waypoint_{index}",
                    x=x,
                    y=y,
                    z=config.z_m + config.height_m,
                    radius_m=config.waypoint_radius_m,
                    height_m=config.height_m,
                    color=config.color,
                )
            )

    if not visuals:
        # A one-cell route still deserves a visible point.
        x, y = points[0]

        visuals.append(
            make_cylinder_visual_xml(
                visual_name=f"{safe_name}_single_point",
                x=x,
                y=y,
                z=config.z_m,
                radius_m=max(config.width_m, 0.025),
                height_m=config.height_m,
                color=config.color,
            )
        )

    return wrap_visuals_in_static_model(
        model_name=model_name,
        visual_fragments=visuals,
    )


# ---------------------------------------------------------------------------
# Claim visualization
# ---------------------------------------------------------------------------


def claim_color(
    report_type: str,
    *,
    disputed: bool,
    config: ClaimVisualConfig,
) -> Rgba:
    """Resolve the display color for a map claim."""

    if disputed:
        return config.disputed_color

    normalized = str(report_type).strip().lower()

    if normalized == "occupied":
        return config.occupied_color

    if normalized == "free":
        return config.free_color

    raise SdfValidationError(f"Unsupported claim type: {report_type}")


def make_claims_sdf(
    *,
    model_name: str,
    claims: Iterable[Mapping[str, object]],
    cell_size_m: float,
    config: ClaimVisualConfig = ClaimVisualConfig(),
) -> str:
    """Create visual markers for occupancy claims.

    Expected keys per claim:
    - row
    - col
    - report_type
    - disputed, optional
    - confidence, optional
    """

    cell_size = validate_positive("cell_size_m", cell_size_m)
    visuals: List[str] = []

    for index, claim in enumerate(claims):
        row = int(claim["row"])
        col = int(claim["col"])
        report_type = str(claim["report_type"])
        disputed = bool(claim.get("disputed", False))

        confidence = float(claim.get("confidence", 1.0))
        confidence = min(1.0, max(0.0, confidence))

        x, y = cell_to_world(row, col, cell_size)

        base_color = claim_color(
            report_type,
            disputed=disputed,
            config=config,
        )

        color = (
            base_color[0],
            base_color[1],
            base_color[2],
            base_color[3] * max(0.20, confidence),
        )

        visuals.append(
            make_cylinder_visual_xml(
                visual_name=f"claim_{index}",
                x=x,
                y=y,
                z=config.z_m,
                radius_m=config.radius_m,
                height_m=config.height_m,
                color=color,
            )
        )

    if not visuals:
        raise SdfValidationError("Cannot create claims SDF with no claims")

    return wrap_visuals_in_static_model(
        model_name=model_name,
        visual_fragments=visuals,
    )


# ---------------------------------------------------------------------------
# Physical dynamic-obstacle visualization
# ---------------------------------------------------------------------------


def make_grid_obstacle_sdf(
    *,
    model_name: str,
    cell: Cell,
    cell_size_m: float,
    footprint_scale: float = 0.72,
    config: ObstacleVisualConfig = ObstacleVisualConfig(),
) -> str:
    """Create a physical or visual-only obstacle centered in one map cell."""

    cell_size = validate_positive("cell_size_m", cell_size_m)
    scale = validate_positive("footprint_scale", footprint_scale)

    if scale > 1.0:
        raise SdfValidationError(
            "footprint_scale should not exceed 1.0 for a grid-cell obstacle"
        )

    row, col = int(cell[0]), int(cell[1])
    x, y = cell_to_world(row, col, cell_size)

    width = cell_size * scale
    depth = cell_size * scale

    visual = make_box_visual_xml(
        visual_name=f"{model_name}_visual",
        x=0.0,
        y=0.0,
        z=0.0,
        size_x_m=width,
        size_y_m=depth,
        size_z_m=config.height_m,
        yaw_rad=0.0,
        color=config.color,
    )

    collisions: List[str] = []

    if config.collision_enabled:
        collisions.append(
            make_box_collision_xml(
                collision_name=f"{model_name}_collision",
                size_x_m=width,
                size_y_m=depth,
                size_z_m=config.height_m,
            )
        )

    return wrap_visuals_in_static_model(
        model_name=model_name,
        visual_fragments=[visual],
        collision_fragments=collisions,
        pose=(x, y, config.z_m, 0.0, 0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Generated-file storage
# ---------------------------------------------------------------------------


class SdfFileStore:
    """Stores generated SDF files with atomic replacement.

    Generated files are intentionally kept outside the package installation
    directory so simulation runs do not modify installed artifacts.
    """

    def __init__(self, base_directory: str | Path) -> None:
        self.base_directory = Path(base_directory).expanduser().resolve()
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, entity_name: str) -> Path:
        safe_name = sanitize_entity_name(entity_name)
        return self.base_directory / f"{safe_name}.sdf"

    def write(self, entity_name: str, sdf_text: str) -> Path:
        """Atomically write an SDF file and return its path."""

        if not sdf_text.strip():
            raise SdfValidationError("Cannot write an empty SDF document")

        destination = self.path_for(entity_name)

        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{destination.stem}_",
            suffix=".tmp",
            dir=str(self.base_directory),
            text=True,
        )

        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
                file.write(sdf_text)
                file.flush()
                os.fsync(file.fileno())

            os.replace(temporary_path, destination)
        except Exception:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise

        return destination

    def remove(self, entity_name: str) -> None:
        """Remove one generated SDF file if it exists."""

        try:
            self.path_for(entity_name).unlink()
        except FileNotFoundError:
            return


# ---------------------------------------------------------------------------
# Gazebo Transport service client
# ---------------------------------------------------------------------------


class GazeboEntityClient:
    """Thin synchronous wrapper around Gazebo Transport entity services.

    This class deliberately avoids depending on rclpy. The experiment manager
    may therefore use it from a timer, callback group, worker, or test harness.
    """

    def __init__(
        self,
        config: GazeboServiceConfig,
        *,
        environment: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.config = config
        self.environment = dict(environment) if environment else None

    def service_available(self, service_name: str) -> bool:
        """Return true when the requested Gazebo service is advertised."""

        result = self._run(
            [
                self.config.executable,
                "service",
                "-l",
            ],
            check=False,
        )

        if result.returncode != 0:
            return False

        services = {
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip()
        }

        return service_name in services

    def world_available(self) -> bool:
        """Return true when create and remove services are available."""

        return (
            self.service_available(self.config.create_service)
            and self.service_available(self.config.remove_service)
        )

    def spawn_from_file(
        self,
        *,
        entity_name: str,
        sdf_path: str | Path,
        allow_renaming: bool = False,
    ) -> SpawnResult:
        """Spawn an entity from an SDF file."""

        safe_name = sanitize_entity_name(entity_name)
        path = Path(sdf_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"SDF file does not exist: {path}")

        request = (
            f'sdf_filename: "{escape_gz_string(str(path))}", '
            f'name: "{escape_gz_string(safe_name)}", '
            f'allow_renaming: {"true" if allow_renaming else "false"}'
        )

        completed = self._call_service(
            service_name=self.config.create_service,
            request_type="gz.msgs.EntityFactory",
            response_type="gz.msgs.Boolean",
            request=request,
        )

        success = completed.returncode == 0 and response_indicates_success(
            completed.stdout
        )

        return SpawnResult(
            entity_name=safe_name,
            success=success,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def spawn_from_string(
        self,
        *,
        entity_name: str,
        sdf_text: str,
        allow_renaming: bool = False,
    ) -> SpawnResult:
        """Spawn an entity by sending SDF text directly to Gazebo."""

        safe_name = sanitize_entity_name(entity_name)

        if not sdf_text.strip():
            raise SdfValidationError("Cannot spawn an empty SDF document")

        request = (
            f'sdf: "{escape_gz_string(sdf_text)}", '
            f'name: "{escape_gz_string(safe_name)}", '
            f'allow_renaming: {"true" if allow_renaming else "false"}'
        )

        completed = self._call_service(
            service_name=self.config.create_service,
            request_type="gz.msgs.EntityFactory",
            response_type="gz.msgs.Boolean",
            request=request,
        )

        success = completed.returncode == 0 and response_indicates_success(
            completed.stdout
        )

        return SpawnResult(
            entity_name=safe_name,
            success=success,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def remove(self, entity_name: str) -> SpawnResult:
        """Remove an entity by name.

        Removing an entity that does not exist may still return false from
        Gazebo. Callers performing replace operations can safely ignore that
        specific failure and continue with creation.
        """

        safe_name = sanitize_entity_name(entity_name)

        request = (
            f'name: "{escape_gz_string(safe_name)}", '
            "type: MODEL"
        )

        completed = self._call_service(
            service_name=self.config.remove_service,
            request_type="gz.msgs.Entity",
            response_type="gz.msgs.Boolean",
            request=request,
        )

        success = completed.returncode == 0 and response_indicates_success(
            completed.stdout
        )

        return SpawnResult(
            entity_name=safe_name,
            success=success,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def replace_from_file(
        self,
        *,
        entity_name: str,
        sdf_path: str | Path,
    ) -> SpawnResult:
        """Remove an existing entity and create its replacement."""

        self.remove(entity_name)

        return self.spawn_from_file(
            entity_name=entity_name,
            sdf_path=sdf_path,
            allow_renaming=False,
        )

    def replace_from_string(
        self,
        *,
        entity_name: str,
        sdf_text: str,
    ) -> SpawnResult:
        """Remove an existing entity and create its replacement from text."""

        self.remove(entity_name)

        return self.spawn_from_string(
            entity_name=entity_name,
            sdf_text=sdf_text,
            allow_renaming=False,
        )

    def _call_service(
        self,
        *,
        service_name: str,
        request_type: str,
        response_type: str,
        request: str,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            self.config.executable,
            "service",
            "-s",
            service_name,
            "--reqtype",
            request_type,
            "--reptype",
            response_type,
            "--timeout",
            str(int(self.config.timeout_ms)),
            "--req",
            request,
        ]

        return self._run(command, check=False)

    def _run(
        self,
        command: Sequence[str],
        *,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                timeout=max(1.0, self.config.timeout_ms / 1000.0 + 1.0),
                env=self.environment,
                check=False,
            )
        except FileNotFoundError as exc:
            raise GazeboServiceError(
                f"Gazebo executable not found: {self.config.executable}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise GazeboServiceError(
                f"Gazebo command timed out: {' '.join(command)}"
            ) from exc
        except OSError as exc:
            raise GazeboServiceError(
                f"Could not execute Gazebo command: {' '.join(command)}"
            ) from exc

        if check and completed.returncode != 0:
            raise GazeboServiceError(
                "Gazebo command failed.\n"
                f"Command: {' '.join(command)}\n"
                f"stdout: {completed.stdout.strip()}\n"
                f"stderr: {completed.stderr.strip()}"
            )

        return completed


def response_indicates_success(stdout: str) -> bool:
    """Interpret the textual output from a gz service Boolean response."""

    normalized = stdout.strip().lower()

    if not normalized:
        return False

    return (
        "data: true" in normalized
        or normalized == "true"
        or normalized.endswith("\ntrue")
    )


def escape_gz_string(value: str) -> str:
    """Escape a Python string for use inside a Gazebo protobuf request."""

    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


# ---------------------------------------------------------------------------
# Higher-level visualization coordinator
# ---------------------------------------------------------------------------


class GazeboVisualizationManager:
    """Coordinates generated SDF files and Gazebo entity replacement.

    The experiment manager can keep one instance of this class and call its
    methods whenever goals, routes, claims, or physical obstacles change.
    """

    def __init__(
        self,
        *,
        world_name: str,
        storage_directory: str | Path,
        timeout_ms: int = 1500,
    ) -> None:
        self.file_store = SdfFileStore(storage_directory)
        self.entity_client = GazeboEntityClient(
            GazeboServiceConfig(
                world_name=world_name,
                timeout_ms=timeout_ms,
            )
        )

        self._entity_hashes: Dict[str, str] = {}

    def is_ready(self) -> bool:
        """Return true when Gazebo entity services are available."""

        return self.entity_client.world_available()

    def remove(self, entity_name: str) -> SpawnResult:
        """Remove an entity and forget its cached content hash."""

        safe_name = sanitize_entity_name(entity_name)
        self._entity_hashes.pop(safe_name, None)

        return self.entity_client.remove(safe_name)

    def synchronize_sdf(
        self,
        *,
        entity_name: str,
        sdf_text: str,
        force: bool = False,
    ) -> Optional[SpawnResult]:
        """Replace an entity only when its SDF content changed.

        Returns:
            SpawnResult when a Gazebo operation was performed.
            None when the existing entity already matches the requested SDF.
        """

        safe_name = sanitize_entity_name(entity_name)
        content_hash = stable_content_hash(sdf_text)

        if not force and self._entity_hashes.get(safe_name) == content_hash:
            return None

        sdf_path = self.file_store.write(safe_name, sdf_text)

        result = self.entity_client.replace_from_file(
            entity_name=safe_name,
            sdf_path=sdf_path,
        )

        if result.success:
            self._entity_hashes[safe_name] = content_hash

        return result

    def synchronize_action_goals(
        self,
        *,
        goals: Sequence[Cell],
        cell_size_m: float,
        config: GoalVisualConfig = GoalVisualConfig(),
        entity_name: str = "trust_action_goals",
        force: bool = False,
    ) -> Optional[SpawnResult]:
        """Synchronize the complete action-goal visualization."""

        if not goals:
            self.remove(entity_name)
            return None

        sdf_text = make_action_goals_sdf(
            model_name=entity_name,
            goals=goals,
            cell_size_m=cell_size_m,
            config=config,
        )

        return self.synchronize_sdf(
            entity_name=entity_name,
            sdf_text=sdf_text,
            force=force,
        )

    def synchronize_route(
        self,
        *,
        robot_id: str,
        route: Sequence[Cell],
        cell_size_m: float,
        config: RouteVisualConfig,
        force: bool = False,
    ) -> Optional[SpawnResult]:
        """Synchronize one robot's route visualization."""

        entity_name = f"trust_route_{sanitize_entity_name(robot_id)}"

        if not route:
            self.remove(entity_name)
            return None

        sdf_text = make_route_sdf(
            model_name=entity_name,
            route=route,
            cell_size_m=cell_size_m,
            config=config,
        )

        return self.synchronize_sdf(
            entity_name=entity_name,
            sdf_text=sdf_text,
            force=force,
        )

    def synchronize_claims(
        self,
        *,
        claims: Sequence[Mapping[str, object]],
        cell_size_m: float,
        config: ClaimVisualConfig = ClaimVisualConfig(),
        entity_name: str = "trust_claims",
        force: bool = False,
    ) -> Optional[SpawnResult]:
        """Synchronize all current claim markers."""

        if not claims:
            self.remove(entity_name)
            return None

        sdf_text = make_claims_sdf(
            model_name=entity_name,
            claims=claims,
            cell_size_m=cell_size_m,
            config=config,
        )

        return self.synchronize_sdf(
            entity_name=entity_name,
            sdf_text=sdf_text,
            force=force,
        )

    def synchronize_obstacle(
        self,
        *,
        obstacle_id: str,
        cell: Cell,
        cell_size_m: float,
        footprint_scale: float = 0.72,
        config: ObstacleVisualConfig = ObstacleVisualConfig(),
        force: bool = False,
    ) -> Optional[SpawnResult]:
        """Synchronize one physical or visual-only dynamic obstacle."""

        entity_name = f"trust_obstacle_{sanitize_entity_name(obstacle_id)}"

        sdf_text = make_grid_obstacle_sdf(
            model_name=entity_name,
            cell=cell,
            cell_size_m=cell_size_m,
            footprint_scale=footprint_scale,
            config=config,
        )

        return self.synchronize_sdf(
            entity_name=entity_name,
            sdf_text=sdf_text,
            force=force,
        )

    def remove_obstacle(self, obstacle_id: str) -> SpawnResult:
        """Remove one experiment-controlled obstacle."""

        entity_name = f"trust_obstacle_{sanitize_entity_name(obstacle_id)}"
        return self.remove(entity_name)
