#!/usr/bin/env python3
"""Generate real-dimension MuJoCo wall templates.

This generator is intentionally fixed to the physical real workspace.
The workspace dimensions are named constants below and are never exposed
through CLI or YAML.

Any CLI option other than ``--config`` may also be provided as a top-level
YAML key. CLI values override YAML values.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REAL_CONFIG_YAML = REPO_ROOT / "tiny_robot_control" / "config" / "real.yaml"
DEFAULT_WAVEFRONT_INFLATION_YAML = (
    REPO_ROOT / "tiny_robot_control" / "config" / "wavefront_inflation.yaml"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated_templates_real"


# Fixed physical workspace. These are not configurable.
REAL_WORKSPACE_WIDTH_M = 0.49
REAL_WORKSPACE_HEIGHT_M = 0.775

# Boundary wall geometry from the unscaled real XML.
BOUNDARY_WALL_HALF_THICKNESS_M = 0.01
WALL_HALF_HEIGHT_M = 0.05

# User-requested inner-wall thickness.
INNER_WALL_FULL_WIDTH_M = 0.057
INNER_WALL_HALF_WIDTH_M = INNER_WALL_FULL_WIDTH_M / 2.0


ROBOT_MASS_KG = 5.0
ROBOT_FRICTION = "1.0 0.005 0.001"
FLOOR_FRICTION = "0.5 0.005 0.001"
WALL_FRICTION = "1.0 0.005 0.0001"
WALL_RGBA = "0.8 0.8 0.8 1"
GOAL_RGBA = "0 1 0 0.5"
GOAL_RADIUS_M = 0.05
FLOAT_EPS = 1e-9

INFLATION_CONTEXTS = {
    "navigation",
    "push_approach",
    "xml_collision_resolution",
}


@dataclass(frozen=True)
class InflationSettings:
    tier1_base_inflation_margin_m: float = 0.005
    navigation_additional_margin_m: float = 0.0
    push_approach_additional_margin_m: float = 0.003
    xml_min_separation_m: float = 0.005
    xml_collision_additional_margin_m: float = 0.08


@dataclass(frozen=True)
class AnchorPoint:
    side: str
    x: float
    y: float


@dataclass(frozen=True)
class WallGeom:
    name: str
    x: float
    y: float
    half_length: float
    half_width: float
    rotation_deg: float


@dataclass(frozen=True)
class ResolvedConfig:
    output_dir: Path
    num_templates: int
    points_per_side: int
    start_seed: int
    robot_config_yaml: Path
    wavefront_inflation_yaml: Path
    robot_width_cm: float
    robot_height_cm: float
    inflation_context: str
    required_gap_length_m: float
    min_gap_length_m: float
    min_segment_length_m: float
    corner_margin_m: float
    max_layout_attempts: int


class LayoutGenerationError(RuntimeError):
    """Raised when a valid wall layout cannot be generated."""


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_yaml_dict(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}, got {type(data).__name__}")
    return data


def _resolve_path(raw_path: str, config_dir: Optional[Path]) -> Path:
    path = Path(raw_path)
    if not path.is_absolute() and config_dir is not None:
        path = config_dir / path
    return path.resolve()


def load_robot_dimensions_from_yaml(path: Path) -> Tuple[float, float]:
    data = _load_yaml_dict(path)
    robot_cfg = data.get("robot", {})
    if not isinstance(robot_cfg, dict):
        raise ValueError(f"'robot' must be a mapping in {path}")

    width_cm = _safe_float(robot_cfg.get("width_cm"), 7.0)
    height_cm = _safe_float(robot_cfg.get("height_cm"), 7.0)
    return width_cm, height_cm


def load_inflation_settings(path: Path) -> InflationSettings:
    if not path.exists():
        return InflationSettings()

    data = _load_yaml_dict(path)
    return InflationSettings(
        tier1_base_inflation_margin_m=_safe_float(
            (data.get("tier1") or {}).get("base_inflation_margin_m"), 0.005
        ),
        navigation_additional_margin_m=_safe_float(
            (data.get("navigation") or {}).get("additional_margin_m"), 0.0
        ),
        push_approach_additional_margin_m=_safe_float(
            (data.get("push_approach") or {}).get("additional_margin_m"), 0.003
        ),
        xml_min_separation_m=_safe_float(
            (data.get("xml_collision_resolution") or {}).get("min_separation_m"),
            0.005,
        ),
        xml_collision_additional_margin_m=_safe_float(
            (data.get("xml_collision_resolution") or {}).get("additional_margin_m"),
            0.08,
        ),
    )


def effective_robot_radius_m(robot_width_cm: float, robot_height_cm: float) -> float:
    return max(abs(float(robot_width_cm)), abs(float(robot_height_cm))) / 200.0


def inflated_robot_radius_m(
    robot_width_cm: float,
    robot_height_cm: float,
    inflation_settings: InflationSettings,
    context: str,
) -> float:
    robot_radius_m = effective_robot_radius_m(robot_width_cm, robot_height_cm)

    if context == "navigation":
        margin_m = (
            inflation_settings.tier1_base_inflation_margin_m
            + inflation_settings.navigation_additional_margin_m
        )
    elif context == "push_approach":
        margin_m = (
            inflation_settings.tier1_base_inflation_margin_m
            + inflation_settings.push_approach_additional_margin_m
        )
    elif context == "xml_collision_resolution":
        margin_m = (
            inflation_settings.xml_min_separation_m
            + inflation_settings.xml_collision_additional_margin_m
        )
    else:
        raise ValueError(f"Unsupported inflation context: {context}")

    return robot_radius_m + margin_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MuJoCo templates for the fixed real workspace."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config. Any CLI value overrides the YAML value.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--num_templates",
        type=int,
        default=None,
        help="Number of template XML files to generate.",
    )
    parser.add_argument(
        "--points_per_side",
        type=int,
        default=None,
        help="Number of sampled points on each workspace side.",
    )
    parser.add_argument(
        "--start_seed",
        type=int,
        default=None,
        help="Seed for the first template. Later templates use consecutive seeds.",
    )
    parser.add_argument(
        "--robot_config_yaml",
        type=str,
        default=None,
        help=f"YAML with robot width/height defaults. Default: {DEFAULT_REAL_CONFIG_YAML}",
    )
    parser.add_argument(
        "--wavefront_inflation_yaml",
        type=str,
        default=None,
        help=(
            "Wavefront inflation YAML used to derive the minimum wall gap. "
            f"Default: {DEFAULT_WAVEFRONT_INFLATION_YAML}"
        ),
    )
    parser.add_argument(
        "--robot_width_cm",
        type=float,
        default=None,
        help="Robot width in cm for gap sizing. Defaults to robot_config_yaml.",
    )
    parser.add_argument(
        "--robot_height_cm",
        type=float,
        default=None,
        help="Robot height in cm for gap sizing. Defaults to robot_config_yaml.",
    )
    parser.add_argument(
        "--inflation_context",
        type=str,
        choices=sorted(INFLATION_CONTEXTS),
        default=None,
        help="Which shared wavefront-inflation context to use for required gap sizing.",
    )
    parser.add_argument(
        "--min_gap_length_m",
        type=float,
        default=None,
        help=(
            "Requested minimum doorway gap in meters. The effective gap is "
            "max(requested, required_from_inflation)."
        ),
    )
    parser.add_argument(
        "--min_segment_length_m",
        type=float,
        default=None,
        help="Minimum remaining wall length on each side of the carved gap.",
    )
    parser.add_argument(
        "--corner_margin_m",
        type=float,
        default=None,
        help="Corner exclusion margin when sampling boundary points.",
    )
    parser.add_argument(
        "--max_layout_attempts",
        type=int,
        default=None,
        help="Maximum layout retries per template.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> ResolvedConfig:
    defaults: Dict[str, Any] = {
        "output_dir": str(DEFAULT_OUTPUT_DIR),
        "num_templates": 1,
        "points_per_side": 2,
        "start_seed": 0,
        "robot_config_yaml": str(DEFAULT_REAL_CONFIG_YAML),
        "wavefront_inflation_yaml": str(DEFAULT_WAVEFRONT_INFLATION_YAML),
        "robot_width_cm": None,
        "robot_height_cm": None,
        "inflation_context": "navigation",
        "min_gap_length_m": None,
        "min_segment_length_m": None,
        "corner_margin_m": INNER_WALL_HALF_WIDTH_M,
        "max_layout_attempts": 200,
    }

    config_dir: Optional[Path] = None
    merged = dict(defaults)
    cli_overrides = {
        key
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }

    if args.config is not None:
        config_path = Path(args.config).resolve()
        config_dir = config_path.parent
        yaml_values = _load_yaml_dict(config_path)
        unknown_keys = set(yaml_values) - set(defaults)
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unknown config key(s) in {config_path}: {unknown}")
        merged.update(yaml_values)

    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        merged[key] = value

    def resolve_merged_path(key: str) -> Path:
        base_dir = None if key in cli_overrides else config_dir
        return _resolve_path(str(merged[key]), base_dir)

    output_dir = resolve_merged_path("output_dir")
    robot_config_yaml = resolve_merged_path("robot_config_yaml")
    wavefront_inflation_yaml = resolve_merged_path("wavefront_inflation_yaml")

    robot_width_cm = merged["robot_width_cm"]
    robot_height_cm = merged["robot_height_cm"]
    if robot_width_cm is None or robot_height_cm is None:
        default_width_cm, default_height_cm = load_robot_dimensions_from_yaml(
            robot_config_yaml
        )
        if robot_width_cm is None:
            robot_width_cm = default_width_cm
        if robot_height_cm is None:
            robot_height_cm = default_height_cm

    inflation_context = str(merged["inflation_context"])
    if inflation_context not in INFLATION_CONTEXTS:
        choices = ", ".join(sorted(INFLATION_CONTEXTS))
        raise ValueError(
            f"Unsupported inflation_context '{inflation_context}'. Choices: {choices}"
        )

    if int(merged["num_templates"]) < 1:
        raise ValueError("num_templates must be at least 1")
    if int(merged["points_per_side"]) < 1:
        raise ValueError("points_per_side must be at least 1")
    if int(merged["max_layout_attempts"]) < 1:
        raise ValueError("max_layout_attempts must be at least 1")

    corner_margin_m = float(merged["corner_margin_m"])
    if corner_margin_m < INNER_WALL_HALF_WIDTH_M:
        raise ValueError(
            "corner_margin_m must be at least the inner-wall half-width "
            f"({INNER_WALL_HALF_WIDTH_M:.4f} m)"
        )
    if corner_margin_m * 2.0 >= min(REAL_WORKSPACE_WIDTH_M, REAL_WORKSPACE_HEIGHT_M):
        raise ValueError("corner_margin_m is too large for the fixed workspace")

    inflation_settings = load_inflation_settings(wavefront_inflation_yaml)
    required_gap_length_m = 2.0 * inflated_robot_radius_m(
        float(robot_width_cm),
        float(robot_height_cm),
        inflation_settings,
        inflation_context,
    )

    requested_gap_length_m = merged["min_gap_length_m"]
    if requested_gap_length_m is None:
        min_gap_length_m = required_gap_length_m
    else:
        min_gap_length_m = max(required_gap_length_m, float(requested_gap_length_m))

    requested_min_segment_length_m = merged["min_segment_length_m"]
    if requested_min_segment_length_m is None:
        min_segment_length_m = max(
            INNER_WALL_FULL_WIDTH_M,
            min_gap_length_m / 2.0,
        )
    else:
        min_segment_length_m = float(requested_min_segment_length_m)

    if min_gap_length_m <= 0.0:
        raise ValueError("min_gap_length_m must be positive")
    if min_segment_length_m <= 0.0:
        raise ValueError("min_segment_length_m must be positive")

    return ResolvedConfig(
        output_dir=output_dir,
        num_templates=int(merged["num_templates"]),
        points_per_side=int(merged["points_per_side"]),
        start_seed=int(merged["start_seed"]),
        robot_config_yaml=robot_config_yaml,
        wavefront_inflation_yaml=wavefront_inflation_yaml,
        robot_width_cm=float(robot_width_cm),
        robot_height_cm=float(robot_height_cm),
        inflation_context=inflation_context,
        required_gap_length_m=required_gap_length_m,
        min_gap_length_m=min_gap_length_m,
        min_segment_length_m=min_segment_length_m,
        corner_margin_m=corner_margin_m,
        max_layout_attempts=int(merged["max_layout_attempts"]),
    )


def sample_side_points(
    rng: random.Random,
    side: str,
    count: int,
    corner_margin_m: float,
) -> List[AnchorPoint]:
    points: List[AnchorPoint] = []
    if side in {"left", "right"}:
        y_min = corner_margin_m
        y_max = REAL_WORKSPACE_HEIGHT_M - corner_margin_m
        x_value = 0.0 if side == "left" else REAL_WORKSPACE_WIDTH_M
        for _ in range(count):
            points.append(AnchorPoint(side=side, x=x_value, y=rng.uniform(y_min, y_max)))
    else:
        x_min = corner_margin_m
        x_max = REAL_WORKSPACE_WIDTH_M - corner_margin_m
        y_value = 0.0 if side == "bottom" else REAL_WORKSPACE_HEIGHT_M
        for _ in range(count):
            points.append(AnchorPoint(side=side, x=rng.uniform(x_min, x_max), y=y_value))
    return points


def random_pairing_counts(points_per_side: int, rng: random.Random) -> Tuple[int, int, int]:
    compositions = [
        (a, b, points_per_side - a - b)
        for a in range(points_per_side + 1)
        for b in range(points_per_side - a + 1)
    ]
    return rng.choice(compositions)


def pair_side_points(
    points_by_side: Dict[str, List[AnchorPoint]],
    rng: random.Random,
) -> List[Tuple[AnchorPoint, AnchorPoint]]:
    count = len(points_by_side["left"])
    a_count, b_count, c_count = random_pairing_counts(count, rng)

    shuffled: Dict[str, List[AnchorPoint]] = {
        side: list(points)
        for side, points in points_by_side.items()
    }
    for points in shuffled.values():
        rng.shuffle(points)

    offsets = {side: 0 for side in shuffled}

    def take(side: str) -> AnchorPoint:
        index = offsets[side]
        offsets[side] = index + 1
        return shuffled[side][index]

    pairs: List[Tuple[AnchorPoint, AnchorPoint]] = []
    pair_counts = [
        ("left", "right", a_count),
        ("left", "bottom", b_count),
        ("left", "top", c_count),
        ("right", "bottom", c_count),
        ("right", "top", b_count),
        ("bottom", "top", a_count),
    ]

    for side_a, side_b, pair_count in pair_counts:
        for _ in range(pair_count):
            pairs.append((take(side_a), take(side_b)))

    rng.shuffle(pairs)
    return pairs


def build_segments_for_pair(
    pair: Tuple[AnchorPoint, AnchorPoint],
    wall_index: int,
    rng: random.Random,
    min_gap_length_m: float,
    min_segment_length_m: float,
) -> List[WallGeom]:
    point_a, point_b = pair
    dx = point_b.x - point_a.x
    dy = point_b.y - point_a.y
    wall_length_m = math.hypot(dx, dy)

    required_length_m = 2.0 * min_segment_length_m + min_gap_length_m
    if wall_length_m + FLOAT_EPS < required_length_m:
        raise LayoutGenerationError(
            f"Wall {wall_index} too short for requested gap and segment lengths: "
            f"{wall_length_m:.4f} m < {required_length_m:.4f} m"
        )

    available_gap_length_m = wall_length_m - 2.0 * min_segment_length_m
    if available_gap_length_m <= min_gap_length_m + FLOAT_EPS:
        gap_length_m = min_gap_length_m
    else:
        gap_length_m = rng.uniform(min_gap_length_m, available_gap_length_m)

    gap_start_min_m = min_segment_length_m
    gap_start_max_m = wall_length_m - min_segment_length_m - gap_length_m
    if gap_start_max_m + FLOAT_EPS < gap_start_min_m:
        raise LayoutGenerationError(
            f"Wall {wall_index} could not place a valid gap under current constraints"
        )

    if gap_start_max_m <= gap_start_min_m + FLOAT_EPS:
        gap_start_m = gap_start_min_m
    else:
        gap_start_m = rng.uniform(gap_start_min_m, gap_start_max_m)
    gap_end_m = gap_start_m + gap_length_m

    direction_x = dx / wall_length_m
    direction_y = dy / wall_length_m
    rotation_deg = math.degrees(math.atan2(dy, dx))

    remaining_segments = [
        (0.0, gap_start_m),
        (gap_end_m, wall_length_m),
    ]

    geoms: List[WallGeom] = []
    suffix = 0
    for segment_start_m, segment_end_m in remaining_segments:
        segment_length_m = segment_end_m - segment_start_m
        if segment_length_m <= FLOAT_EPS:
            continue

        segment_midpoint_m = segment_start_m + segment_length_m / 2.0
        center_x = point_a.x + direction_x * segment_midpoint_m
        center_y = point_a.y + direction_y * segment_midpoint_m
        suffix += 1
        geoms.append(
            WallGeom(
                name=f"wall_inner_{wall_index}_{suffix}",
                x=center_x,
                y=center_y,
                half_length=segment_length_m / 2.0,
                half_width=INNER_WALL_HALF_WIDTH_M,
                rotation_deg=rotation_deg,
            )
        )

    return geoms


def point_collides_with_rotated_wall(
    x: float,
    y: float,
    radius_m: float,
    wall: WallGeom,
) -> bool:
    angle_rad = math.radians(wall.rotation_deg)
    dx = x - wall.x
    dy = y - wall.y

    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    local_x = dx * cos_angle + dy * sin_angle
    local_y = -dx * sin_angle + dy * cos_angle

    expanded_x = abs(local_x) - wall.half_length
    expanded_y = abs(local_y) - wall.half_width
    outside_x = max(expanded_x, 0.0)
    outside_y = max(expanded_y, 0.0)
    return (outside_x * outside_x + outside_y * outside_y) <= radius_m * radius_m


def sample_free_point(
    rng: random.Random,
    walls: Sequence[WallGeom],
    clearance_radius_m: float,
    max_attempts: int,
) -> Optional[Tuple[float, float]]:
    x_min = clearance_radius_m
    x_max = REAL_WORKSPACE_WIDTH_M - clearance_radius_m
    y_min = clearance_radius_m
    y_max = REAL_WORKSPACE_HEIGHT_M - clearance_radius_m
    if x_min >= x_max or y_min >= y_max:
        return None

    for _ in range(max_attempts):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        if any(
            point_collides_with_rotated_wall(x, y, clearance_radius_m, wall)
            for wall in walls
        ):
            continue
        return (x, y)
    return None


def sample_robot_and_goal(
    rng: random.Random,
    walls: Sequence[WallGeom],
    robot_clearance_radius_m: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    robot_point = sample_free_point(
        rng,
        walls,
        robot_clearance_radius_m,
        max_attempts=500,
    )
    if robot_point is None:
        raise LayoutGenerationError("Could not place placeholder robot in free space")

    min_robot_goal_distance_m = max(2.0 * robot_clearance_radius_m, 0.15)
    for _ in range(500):
        goal_point = sample_free_point(
            rng,
            walls,
            GOAL_RADIUS_M,
            max_attempts=50,
        )
        if goal_point is None:
            break
        dx = goal_point[0] - robot_point[0]
        dy = goal_point[1] - robot_point[1]
        if math.hypot(dx, dy) >= min_robot_goal_distance_m:
            return robot_point, goal_point

    raise LayoutGenerationError("Could not place placeholder goal in free space")


def add_common_assets(root: Element) -> None:
    SubElement(
        root,
        "option",
        timestep="0.002",
        integrator="implicitfast",
        iterations="100",
        cone="elliptic",
    )

    default = SubElement(root, "default")
    SubElement(default, "geom", density="1")

    asset = SubElement(root, "asset")
    SubElement(
        asset,
        "texture",
        builtin="gradient",
        height="3072",
        rgb1="0.3 0.5 0.7",
        rgb2="0 0 0",
        type="skybox",
        width="512",
    )
    SubElement(
        asset,
        "texture",
        builtin="checker",
        height="300",
        mark="edge",
        markrgb="0.8 0.8 0.8",
        name="groundplane",
        rgb1="0.2 0.3 0.4",
        rgb2="0.1 0.2 0.3",
        type="2d",
        width="300",
    )
    SubElement(
        asset,
        "material",
        name="groundplane",
        reflectance="0.2",
        texrepeat="5 5",
        texture="groundplane",
        texuniform="true",
    )
    SubElement(asset, "material", name="robot", rgba="1.0 1.0 0.0 1.0")


def add_boundary_walls(walls_body: Element) -> None:
    x_mid = REAL_WORKSPACE_WIDTH_M / 2.0
    y_mid = REAL_WORKSPACE_HEIGHT_M / 2.0
    half_width = REAL_WORKSPACE_WIDTH_M / 2.0
    half_height = REAL_WORKSPACE_HEIGHT_M / 2.0
    t = BOUNDARY_WALL_HALF_THICKNESS_M

    boundary_specs = [
        (
            "wall_boundary_left",
            -t,
            y_mid,
            t,
            half_height + 2.0 * t,
            0.0,
        ),
        (
            "wall_boundary_right",
            REAL_WORKSPACE_WIDTH_M + t,
            y_mid,
            t,
            half_height + 2.0 * t,
            0.0,
        ),
        (
            "wall_boundary_bottom",
            x_mid,
            -t,
            half_width,
            t,
            0.0,
        ),
        (
            "wall_boundary_top",
            x_mid,
            REAL_WORKSPACE_HEIGHT_M + t,
            half_width,
            t,
            0.0,
        ),
    ]

    for name, x, y, half_x, half_y, rotation_deg in boundary_specs:
        SubElement(
            walls_body,
            "geom",
            name=name,
            type="box",
            condim="4",
            friction=WALL_FRICTION,
            rgba=WALL_RGBA,
            pos=f"{x:.6f} {y:.6f} {WALL_HALF_HEIGHT_M:.6f}",
            euler=f"0 0 {rotation_deg:.6f}",
            size=f"{half_x:.6f} {half_y:.6f} {WALL_HALF_HEIGHT_M:.6f}",
        )


def add_inner_walls(walls_body: Element, walls: Sequence[WallGeom]) -> None:
    for wall in walls:
        SubElement(
            walls_body,
            "geom",
            name=wall.name,
            type="box",
            condim="4",
            friction=WALL_FRICTION,
            rgba=WALL_RGBA,
            pos=f"{wall.x:.6f} {wall.y:.6f} {WALL_HALF_HEIGHT_M:.6f}",
            euler=f"0 0 {wall.rotation_deg:.6f}",
            size=(
                f"{wall.half_length:.6f} "
                f"{wall.half_width:.6f} "
                f"{WALL_HALF_HEIGHT_M:.6f}"
            ),
        )


def build_xml(
    wall_geoms: Sequence[WallGeom],
    robot_position: Tuple[float, float],
    goal_position: Tuple[float, float],
    robot_radius_m: float,
) -> str:
    root = Element("mujoco", model="generated_real_template")
    SubElement(root, "compiler", angle="degree")
    add_common_assets(root)

    worldbody = SubElement(root, "worldbody")
    SubElement(worldbody, "light", dir="0 0 -1", directional="true", pos="0 0 1.5")
    SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        condim="4",
        friction=FLOOR_FRICTION,
        material="groundplane",
        size="0 0 0.05",
    )

    walls_body = SubElement(worldbody, "body", name="walls")
    add_boundary_walls(walls_body)
    add_inner_walls(walls_body, wall_geoms)

    robot_body = SubElement(worldbody, "body", name="robot")
    SubElement(robot_body, "joint", name="joint_x", type="slide", pos="0 0 0", axis="1 0 0")
    SubElement(robot_body, "joint", name="joint_y", type="slide", pos="0 0 0", axis="0 1 0")
    SubElement(
        robot_body,
        "geom",
        name="robot",
        type="sphere",
        pos=f"{robot_position[0]:.6f} {robot_position[1]:.6f} {robot_radius_m:.6f}",
        size=f"{robot_radius_m:.6f}",
        mass=f"{ROBOT_MASS_KG:.6f}",
        friction=ROBOT_FRICTION,
        condim="4",
    )
    SubElement(robot_body, "site", name="sensor_ball")

    SubElement(
        worldbody,
        "site",
        name="goal",
        type="sphere",
        pos=f"{goal_position[0]:.6f} {goal_position[1]:.6f} 0.0",
        size=f"{GOAL_RADIUS_M:.6f}",
        rgba=GOAL_RGBA,
    )

    actuator = SubElement(root, "actuator")
    SubElement(
        actuator,
        "motor",
        name="actuator_x",
        joint="joint_x",
        gear="1",
        ctrlrange="-1 1",
    )
    SubElement(
        actuator,
        "motor",
        name="actuator_y",
        joint="joint_y",
        gear="1",
        ctrlrange="-1 1",
    )

    rough_xml = tostring(root, encoding="unicode")
    return minidom.parseString(rough_xml).toprettyxml(indent="  ")


def generate_template(config: ResolvedConfig, seed: int) -> str:
    rng = random.Random(seed)
    placeholder_clearance_radius_m = config.required_gap_length_m / 2.0
    robot_radius_m = effective_robot_radius_m(
        config.robot_width_cm,
        config.robot_height_cm,
    )

    for _ in range(config.max_layout_attempts):
        points_by_side = {
            "left": sample_side_points(
                rng, "left", config.points_per_side, config.corner_margin_m
            ),
            "right": sample_side_points(
                rng, "right", config.points_per_side, config.corner_margin_m
            ),
            "bottom": sample_side_points(
                rng, "bottom", config.points_per_side, config.corner_margin_m
            ),
            "top": sample_side_points(
                rng, "top", config.points_per_side, config.corner_margin_m
            ),
        }
        pairs = pair_side_points(points_by_side, rng)

        wall_geoms: List[WallGeom] = []
        try:
            for wall_index, pair in enumerate(pairs, start=1):
                wall_geoms.extend(
                    build_segments_for_pair(
                        pair,
                        wall_index,
                        rng,
                        min_gap_length_m=config.min_gap_length_m,
                        min_segment_length_m=config.min_segment_length_m,
                    )
                )
            robot_position, goal_position = sample_robot_and_goal(
                rng,
                wall_geoms,
                placeholder_clearance_radius_m,
            )
        except LayoutGenerationError:
            continue

        return build_xml(wall_geoms, robot_position, goal_position, robot_radius_m)

    raise LayoutGenerationError(
        "Failed to generate a valid layout after "
        f"{config.max_layout_attempts} attempts. Try fewer points_per_side or "
        "smaller gap/segment constraints."
    )


def save_templates(config: ResolvedConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Generating templates with fixed workspace "
        f"{REAL_WORKSPACE_WIDTH_M:.3f}m x {REAL_WORKSPACE_HEIGHT_M:.3f}m"
    )
    print(
        "Required min gap from inflation: "
        f"{config.required_gap_length_m:.4f} m "
        f"(context={config.inflation_context}, robot={config.robot_width_cm:.2f}x"
        f"{config.robot_height_cm:.2f} cm)"
    )
    print(
        "Effective min gap / min segment: "
        f"{config.min_gap_length_m:.4f} m / {config.min_segment_length_m:.4f} m"
    )

    for index in range(config.num_templates):
        seed = config.start_seed + index
        xml_text = generate_template(config, seed)
        output_path = (
            config.output_dir
            / f"real_template_pts{config.points_per_side}_seed{seed:05d}.xml"
        )
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(xml_text)
        print(f"Generated {output_path}")


def main() -> None:
    args = parse_args()
    config = resolve_config(args)
    save_templates(config)


if __name__ == "__main__":
    main()
