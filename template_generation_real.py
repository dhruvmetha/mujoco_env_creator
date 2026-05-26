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
from typing import Dict, List, Optional, Sequence, Tuple
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
BOUNDARY_WALL_FULL_THICKNESS_M = BOUNDARY_WALL_HALF_THICKNESS_M * 2.0
WALL_HALF_HEIGHT_M = 0.05

# User-requested inner-wall thickness.
INNER_WALL_FULL_WIDTH_M = 0.057
INNER_WALL_HALF_WIDTH_M = INNER_WALL_FULL_WIDTH_M / 2.0

FLOOR_FRICTION = "0.5 0.005 0.001"
WALL_FRICTION = "1.0 0.005 0.0001"
WALL_RGBA = "0.8 0.8 0.8 1"
FLOAT_EPS = 1e-9
GEOMETRY_SAFETY_MARGIN_M = 1e-6

INFLATION_CONTEXTS = {
    "navigation",
    "push_approach",
    "xml_collision_resolution",
}
STRAIGHT_WALL_MODES = {
    "mixed",
}
SIDE_ORDER = ("left", "right", "bottom", "top")


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
    straight_wall_fraction: float
    straight_wall_mode: str
    straight_wall_count: int
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
        "--straight_wall_fraction",
        type=float,
        default=None,
        help=(
            "Fraction of logical inner walls to force straight. "
            "Straight walls are axis-aligned, using left-right and/or "
            "bottom-top pairings."
        ),
    )
    parser.add_argument(
        "--straight_wall_mode",
        type=str,
        choices=sorted(STRAIGHT_WALL_MODES),
        default=None,
        help=(
            "Straight-wall mode. Only 'mixed' is supported, meaning forced "
            "straight walls may be horizontal and/or vertical."
        ),
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
        "straight_wall_fraction": 0.0,
        "straight_wall_mode": "mixed",
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

    straight_wall_fraction = float(merged["straight_wall_fraction"])
    if not 0.0 <= straight_wall_fraction <= 1.0:
        raise ValueError("straight_wall_fraction must be between 0.0 and 1.0")

    straight_wall_mode = str(merged["straight_wall_mode"])
    if straight_wall_mode not in STRAIGHT_WALL_MODES:
        choices = ", ".join(sorted(STRAIGHT_WALL_MODES))
        raise ValueError(
            f"Unsupported straight_wall_mode '{straight_wall_mode}'. Choices: {choices}"
        )

    total_logical_walls = 2 * int(merged["points_per_side"])
    straight_wall_count = int(
        round(straight_wall_fraction * total_logical_walls)
    )
    max_straight_wall_count = max_straight_walls_for_mode(
        int(merged["points_per_side"]),
        straight_wall_mode,
    )
    if straight_wall_count > max_straight_wall_count:
        raise ValueError(
            "Requested straight_wall_fraction is not feasible for the chosen mode: "
            f"{straight_wall_count} straight walls requested, but mode="
            f"{straight_wall_mode} allows at most {max_straight_wall_count} with "
            f"points_per_side={int(merged['points_per_side'])}."
        )

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
        straight_wall_fraction=straight_wall_fraction,
        straight_wall_mode=straight_wall_mode,
        straight_wall_count=straight_wall_count,
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


def max_straight_walls_for_mode(points_per_side: int, straight_wall_mode: str) -> int:
    if straight_wall_mode != "mixed":
        raise ValueError(f"Unsupported straight_wall_mode: {straight_wall_mode}")
    return 2 * points_per_side


def random_straight_wall_counts(
    total_straight_walls: int,
    points_per_side: int,
    straight_wall_mode: str,
    rng: random.Random,
) -> Tuple[int, int]:
    if total_straight_walls < 0:
        raise ValueError("total_straight_walls must be non-negative")
    if straight_wall_mode != "mixed":
        raise ValueError(f"Unsupported straight_wall_mode: {straight_wall_mode}")

    min_horizontal = max(0, total_straight_walls - points_per_side)
    max_horizontal = min(total_straight_walls, points_per_side)
    horizontal_count = rng.randint(min_horizontal, max_horizontal)
    vertical_count = total_straight_walls - horizontal_count
    return horizontal_count, vertical_count


def sample_straight_pairs(
    horizontal_count: int,
    vertical_count: int,
    rng: random.Random,
    corner_margin_m: float,
) -> List[Tuple[AnchorPoint, AnchorPoint]]:
    pairs: List[Tuple[AnchorPoint, AnchorPoint]] = []

    for _ in range(horizontal_count):
        y_value = rng.uniform(
            corner_margin_m,
            REAL_WORKSPACE_HEIGHT_M - corner_margin_m,
        )
        pairs.append(
            (
                AnchorPoint(side="left", x=0.0, y=y_value),
                AnchorPoint(side="right", x=REAL_WORKSPACE_WIDTH_M, y=y_value),
            )
        )

    for _ in range(vertical_count):
        x_value = rng.uniform(
            corner_margin_m,
            REAL_WORKSPACE_WIDTH_M - corner_margin_m,
        )
        pairs.append(
            (
                AnchorPoint(side="bottom", x=x_value, y=0.0),
                AnchorPoint(side="top", x=x_value, y=REAL_WORKSPACE_HEIGHT_M),
            )
        )

    rng.shuffle(pairs)
    return pairs


def can_complete_pairing(counts: Dict[str, int]) -> bool:
    total_points = sum(counts.values())
    if total_points % 2 != 0:
        return False
    if total_points == 0:
        return True
    return max(counts.values()) * 2 <= total_points


def sample_pair_side_sequence(
    counts: Dict[str, int],
    rng: random.Random,
) -> List[Tuple[str, str]]:
    if not can_complete_pairing(counts):
        raise LayoutGenerationError(f"Infeasible side counts for pairing: {counts}")

    dead_states = set()

    def backtrack(state: Tuple[int, int, int, int]) -> Optional[List[Tuple[str, str]]]:
        if sum(state) == 0:
            return []
        if state in dead_states:
            return None

        max_count = max(state)
        candidate_a_indices = [
            index for index, value in enumerate(state) if value == max_count
        ]
        rng.shuffle(candidate_a_indices)

        for side_a_index in candidate_a_indices:
            side_b_indices = [
                index
                for index, value in enumerate(state)
                if index != side_a_index and value > 0
            ]
            rng.shuffle(side_b_indices)

            for side_b_index in side_b_indices:
                next_state = list(state)
                next_state[side_a_index] -= 1
                next_state[side_b_index] -= 1
                next_counts = {
                    side: next_state[index]
                    for index, side in enumerate(SIDE_ORDER)
                }
                if not can_complete_pairing(next_counts):
                    continue

                remainder = backtrack(tuple(next_state))
                if remainder is not None:
                    return [
                        (SIDE_ORDER[side_a_index], SIDE_ORDER[side_b_index]),
                        *remainder,
                    ]

        dead_states.add(state)
        return None

    initial_state = tuple(counts[side] for side in SIDE_ORDER)
    result = backtrack(initial_state)
    if result is None:
        raise LayoutGenerationError(
            f"Failed to find a valid cross-side pairing for counts {counts}"
        )
    return result


def pair_side_points(
    points_by_side: Dict[str, List[AnchorPoint]],
    rng: random.Random,
) -> List[Tuple[AnchorPoint, AnchorPoint]]:
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

    side_counts = {
        side: len(points)
        for side, points in shuffled.items()
    }
    pair_sequence = sample_pair_side_sequence(side_counts, rng)
    return [(take(side_a), take(side_b)) for side_a, side_b in pair_sequence]


def clip_segment_to_axis_aligned_box(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Clip a segment to an axis-aligned box using Liang-Barsky."""
    dx = end_x - start_x
    dy = end_y - start_y
    t_min = 0.0
    t_max = 1.0

    for p, q in (
        (-dx, start_x - min_x),
        (dx, max_x - start_x),
        (-dy, start_y - min_y),
        (dy, max_y - start_y),
    ):
        if abs(p) <= FLOAT_EPS:
            if q < 0.0:
                return None
            continue

        t = q / p
        if p < 0.0:
            t_min = max(t_min, t)
        else:
            t_max = min(t_max, t)

        if t_min > t_max + FLOAT_EPS:
            return None

    clipped_start = (start_x + t_min * dx, start_y + t_min * dy)
    clipped_end = (start_x + t_max * dx, start_y + t_max * dy)
    return clipped_start, clipped_end


def clip_wall_centerline_to_enclosure(
    point_a: AnchorPoint,
    point_b: AnchorPoint,
    half_width_m: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    dx = point_b.x - point_a.x
    dy = point_b.y - point_a.y
    wall_length_m = math.hypot(dx, dy)
    if wall_length_m <= FLOAT_EPS:
        raise LayoutGenerationError("Wall endpoints collapsed to zero length")

    # Allow overlap with the boundary-wall volume, but not past the outer face.
    # This keeps slanted walls visually connected to the boundary while
    # preventing their corners from poking outside the enclosure.
    x_margin = half_width_m * abs(dy) / wall_length_m
    y_margin = half_width_m * abs(dx) / wall_length_m

    min_x = -BOUNDARY_WALL_FULL_THICKNESS_M + x_margin + GEOMETRY_SAFETY_MARGIN_M
    max_x = (
        REAL_WORKSPACE_WIDTH_M
        + BOUNDARY_WALL_FULL_THICKNESS_M
        - x_margin
        - GEOMETRY_SAFETY_MARGIN_M
    )
    min_y = -BOUNDARY_WALL_FULL_THICKNESS_M + y_margin + GEOMETRY_SAFETY_MARGIN_M
    max_y = (
        REAL_WORKSPACE_HEIGHT_M
        + BOUNDARY_WALL_FULL_THICKNESS_M
        - y_margin
        - GEOMETRY_SAFETY_MARGIN_M
    )

    clipped = clip_segment_to_axis_aligned_box(
        point_a.x,
        point_a.y,
        point_b.x,
        point_b.y,
        min_x,
        max_x,
        min_y,
        max_y,
    )
    if clipped is None:
        raise LayoutGenerationError("Could not clip wall to the physical enclosure")
    return clipped


def build_segments_for_pair(
    pair: Tuple[AnchorPoint, AnchorPoint],
    wall_index: int,
    rng: random.Random,
    min_gap_length_m: float,
    min_segment_length_m: float,
) -> List[WallGeom]:
    raw_point_a, raw_point_b = pair
    (start_x, start_y), (end_x, end_y) = clip_wall_centerline_to_enclosure(
        raw_point_a,
        raw_point_b,
        INNER_WALL_HALF_WIDTH_M,
    )
    dx = end_x - start_x
    dy = end_y - start_y
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
        center_x = start_x + direction_x * segment_midpoint_m
        center_y = start_y + direction_y * segment_midpoint_m
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


def build_xml(wall_geoms: Sequence[WallGeom]) -> str:
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

    rough_xml = tostring(root, encoding="unicode")
    return minidom.parseString(rough_xml).toprettyxml(indent="  ")


def generate_template(config: ResolvedConfig, seed: int) -> str:
    rng = random.Random(seed)

    for _ in range(config.max_layout_attempts):
        horizontal_straight_count, vertical_straight_count = random_straight_wall_counts(
            config.straight_wall_count,
            config.points_per_side,
            config.straight_wall_mode,
            rng,
        )
        straight_pairs = sample_straight_pairs(
            horizontal_straight_count,
            vertical_straight_count,
            rng,
            config.corner_margin_m,
        )

        points_by_side = {
            "left": sample_side_points(
                rng,
                "left",
                config.points_per_side - horizontal_straight_count,
                config.corner_margin_m,
            ),
            "right": sample_side_points(
                rng,
                "right",
                config.points_per_side - horizontal_straight_count,
                config.corner_margin_m,
            ),
            "bottom": sample_side_points(
                rng,
                "bottom",
                config.points_per_side - vertical_straight_count,
                config.corner_margin_m,
            ),
            "top": sample_side_points(
                rng,
                "top",
                config.points_per_side - vertical_straight_count,
                config.corner_margin_m,
            ),
        }
        pairs = [*straight_pairs, *pair_side_points(points_by_side, rng)]
        rng.shuffle(pairs)

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
            return build_xml(wall_geoms)
        except LayoutGenerationError:
            continue

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
    print(
        "Forced straight walls: "
        f"{config.straight_wall_count} of {2 * config.points_per_side} logical walls "
        f"(fraction={config.straight_wall_fraction:.3f}, mode={config.straight_wall_mode})"
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
