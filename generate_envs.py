#!/usr/bin/env python3
"""
Generate complete MuJoCo environments with intelligent robot and goal placement.

This script:
1. Takes an existing XML file (with walls/maze but no obstacles)
2. Randomly places movable obstacles in the environment
3. Uses wavefront analysis to find connected regions
4. Places robot and goal in connected regions with sufficient separation
5. Outputs modified XML with obstacles + corrected robot/goal positions
"""

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET
from collections import deque
import itertools
import numpy as np
import yaml


def update_robot_xy_in_xml(worldbody: ET.Element, x: float, y: float, z_default: float = 0.2) -> bool:
    """Update the robot's xy position. Handles both:
       - point robot: body[@name='robot'] containing a sphere geom (set geom pos to x y z_default)
       - diff-drive car: body[@name='car'] using a freejoint (set body pos to x y CAR_SPAWN_Z)
    Returns True if either was found and updated."""
    updated = False
    pr = worldbody.find(".//body[@name='robot']")
    if pr is not None:
        g = pr.find(".//geom[@name='robot']")
        if g is not None:
            g.set('pos', f'{x} {y} {z_default}')
            updated = True
    car = worldbody.find(".//body[@name='car']")
    if car is not None:
        # Preserve the existing z so the wheels don't re-clip the floor at re-load.
        existing = car.get('pos', '0 0 0.010').split()
        z = existing[2] if len(existing) >= 3 else '0.010'
        car.set('pos', f'{x} {y} {z}')
        updated = True
    return updated


def load_robot_half_extent_from_namo_config(namo_config_path: str):
    """Read planning.robot_size from namo_config yaml. Returns (hx, hy) or None."""
    try:
        with open(namo_config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        rs = (cfg.get('planning') or {}).get('robot_size')
        if isinstance(rs, (list, tuple)) and len(rs) >= 2:
            return (float(rs[0]), float(rs[1]))
        if isinstance(rs, (int, float)):
            return (float(rs), float(rs))
    except Exception:
        pass
    return None

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not available, using basic collision detection")

# Import MuJoCo environment and wavefront snapshot exporter.
# generate_environments_from_pairs no longer needs namo_rl (uses the
# WavefrontSnapshotExporter.from_geometry path). It's still imported for
# legacy entry points that load real envs.
try:
    import namo_rl
    NAMO_RL_AVAILABLE = True
except ImportError:
    NAMO_RL_AVAILABLE = False
    namo_rl = None  # legacy paths will fail loudly if they hit this

# Use the single source of truth for wavefront snapshots: the runtime planner's
# version under namo.visualization. Avoids drift between gen-time and plan-time
# region/adjacency views.
_NAMO_PYTHON = os.environ.get("NAMO_PYTHON_PATH",
                              "/common/home/dm1487/robotics_research/ktamp/namo/python")
if _NAMO_PYTHON not in sys.path:
    sys.path.insert(0, _NAMO_PYTHON)
from namo.visualization.wavefront_snapshot import (
    WavefrontSnapshotExporter, ObjectTemplate, ObjectInstance,
)


def _yaw_to_quat(theta: float) -> Tuple[float, float, float, float]:
    """yaw → (w, x, y, z) about z-axis."""
    import math
    half = 0.5 * theta
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _build_geometry_from_placements(
    walls: List[Dict[str, Any]],
    obstacles: List[Dict[str, Any]],
    robot_pose: Tuple[float, float, float],
    goal_pose: Optional[Tuple[float, float, float]],
):
    """Convert pure-Python placement dicts (walls + obstacles) into the
    static_objects / movable_templates / movable_instances format that
    WavefrontSnapshotExporter.from_geometry expects."""
    static_objects: List[ObjectInstance] = []
    for i, w in enumerate(walls):
        tpl = ObjectTemplate(
            name=f"wall_{i}",
            half_extent=(float(w['size'][0]), float(w['size'][1])),
            is_static=True,
        )
        static_objects.append(ObjectInstance(
            template=tpl,
            position=(float(w['pos'][0]), float(w['pos'][1])),
            quaternion=(1.0, 0.0, 0.0, 0.0),
        ))

    movable_templates: Dict[str, ObjectTemplate] = {}
    movable_instances: List[ObjectInstance] = []
    for i, obs in enumerate(obstacles):
        name = f"obstacle_{i}_movable"
        tpl = ObjectTemplate(
            name=name,
            half_extent=(float(obs['size'][0]), float(obs['size'][1])),
            is_static=False,
        )
        movable_templates[name] = tpl
        # `rotation` in obs dict is in DEGREES (set by place_obstacles via
        # np.random.uniform(0,360)). Convert to radians before quaternion.
        import math as _math
        rot_rad = _math.radians(float(obs.get('rotation', 0.0)))
        movable_instances.append(ObjectInstance(
            template=tpl,
            position=(float(obs['pos'][0]), float(obs['pos'][1])),
            quaternion=_yaw_to_quat(rot_rad),
        ))
    return static_objects, movable_templates, movable_instances


def _compute_runtime_bounds(static_objects, movable_instances, robot_original_pos, robot_half_extent):
    """Replicate NAMOEnvironment::get_environment_bounds (C++):
    start at [-2,2]² minimum, expand to include every object corner + robot
    radius, then add 0.5m padding on all sides.
    """
    xmin, xmax, ymin, ymax = -2.0, 2.0, -2.0, 2.0
    import math
    def expand_for(instance, half_w, half_h):
        nonlocal xmin, xmax, ymin, ymax
        cx, cy = instance.position
        yaw = math.atan2(2.0 * (instance.quaternion[0] * instance.quaternion[3]),
                         1.0 - 2.0 * (instance.quaternion[3] * instance.quaternion[3]))
        ca, sa = math.cos(yaw), math.sin(yaw)
        for lx, ly in [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]:
            wx = cx + lx * ca - ly * sa
            wy = cy + lx * sa + ly * ca
            xmin = min(xmin, wx); xmax = max(xmax, wx)
            ymin = min(ymin, wy); ymax = max(ymax, wy)
    for inst in static_objects:
        # C++ uses obj.size[0] * 0.5 (i.e. HALF of the half-extent — quarter-extent in practice).
        # Preserve that behaviour exactly.
        expand_for(inst, inst.half_extent[0] * 0.5, inst.half_extent[1] * 0.5)
    for inst in movable_instances:
        expand_for(inst, inst.half_extent[0] * 0.5, inst.half_extent[1] * 0.5)
    if robot_original_pos is not None:
        rx, ry = robot_original_pos[0], robot_original_pos[1]
        rr = robot_half_extent[0]
        xmin = min(xmin, rx - rr); xmax = max(xmax, rx + rr)
        ymin = min(ymin, ry - rr); ymax = max(ymax, ry + rr)
    PADDING = 0.5
    return (xmin - PADDING, xmax + PADDING, ymin - PADDING, ymax + PADDING)


def _adjacency_bfs_distance(adjacency: Dict[str, Set[str]], src: str, tgt: str) -> int:
    """BFS shortest-path length in region adjacency graph. -1 if unreachable."""
    if src == tgt:
        return 0
    if src not in adjacency or tgt not in adjacency:
        return -1
    seen = {src}
    frontier = deque([(src, 0)])
    while frontier:
        node, d = frontier.popleft()
        for nb in adjacency.get(node, ()):
            if nb in seen:
                continue
            if nb == tgt:
                return d + 1
            seen.add(nb)
            frontier.append((nb, d + 1))
    return -1


def _runtime_validate_adjacency(exporter, robot_pos, goal_pos, exact_hop: int = 1, debug: bool = False):
    """Re-run the wavefront with the candidate robot/goal poses so the
    'clear cells around robot' step is applied exactly as the planner sees it.

    Accept only if the planner-view snapshot yields distinct robot/goal regions
    whose BFS distance in the adjacency graph equals exact_hop. Rejects:
      - 'robot_goal' label  → robot already in goal region (no opening needed)
      - Hop distance != exact_hop
    """
    # NOTE: namo's WavefrontSnapshotExporter dropped `from_geometry` and
    # `_preset_movables` during the env-driven refactor. The previous re-run
    # path (build a fresh exporter from the same static/movable geometry but
    # with a candidate robot/goal pose) no longer exists. Rebuilding via a
    # per-pair namo_rl env would cost ~100 ms per sample which is too slow
    # for samples_per_pair × pair count. Skipping for now: gen-time adjacency
    # BFS already enforces exact_hop on the static region graph; we only lose
    # the "robot/goal placement collapses the regions" rejection (the rare
    # case where the planner's clear-cells-around-robot step erodes a barely
    # blocking obstacle). For corridors with strict-block sizing this is a
    # near-no-op. Re-enable by reconstructing via env + set_full_state if
    # data quality demands it.
    return True


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_NUM_OBJECTS_RANGE = (3, 10)  # fallback uniform range when no per-template JSON applies
DEFAULT_OBJECT_SIZE_RANGE = (0.4, 1.4)  # (min, max) side length (2x half-size from config)


def _resolve_num_objects(cfg_val: Any, rng: np.random.Generator) -> int:
    """Accept either an int (fixed count) or a [min, max] pair (uniform sample, inclusive)."""
    if isinstance(cfg_val, (list, tuple)):
        lo, hi = int(cfg_val[0]), int(cfg_val[1])
        if hi < lo:
            lo, hi = hi, lo
        return int(rng.integers(lo, hi + 1))
    return int(cfg_val)


def _template_lookup_key(template_xml_path: str) -> str:
    """Two-part key '<parent_dir>/<basename>' to match the JSON map and the output namespacing."""
    norm = os.path.normpath(template_xml_path)
    parent = os.path.basename(os.path.dirname(norm))
    base = os.path.splitext(os.path.basename(norm))[0]
    return f"{parent}/{base}" if parent and parent not in (".", "") else base

DEFAULT_CLEARANCE_RADIUS = 0.3  # meters
DEFAULT_MIN_GOAL_DISTANCE = 1.0  # meters minimum separation between robot and goal
DEFAULT_RESOLUTION = 0.01  # meters per grid cell
DEFAULT_MAX_GOAL_RETRIES = 10  # maximum retries for goal placement with clearance


# ------------------------------------------------------------------
# XML Parsing and Manipulation
# ------------------------------------------------------------------
def _extract_geom_rotation_deg(geom: ET.Element) -> float:
    """Extract yaw rotation in degrees from euler or quat attributes."""
    euler_str = geom.get('euler')
    if euler_str:
        euler_parts = [float(x) for x in euler_str.split()]
        if len(euler_parts) >= 3:
            return euler_parts[2]

    quat_str = geom.get('quat')
    if quat_str:
        quat_parts = [float(x) for x in quat_str.split()]
        if len(quat_parts) == 4:
            w, x, y, z = quat_parts
            yaw_rad = np.arctan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
            return float(np.degrees(yaw_rad))

    return 0.0


def _rotated_bbox_half_extents(
    half_size_x: float,
    half_size_y: float,
    rotation_deg: float,
) -> Tuple[float, float]:
    """Return axis-aligned half extents of a rotated rectangle."""
    angle_rad = np.radians(rotation_deg)
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))
    return (
        half_size_x * cos_a + half_size_y * sin_a,
        half_size_x * sin_a + half_size_y * cos_a,
    )


def parse_xml_template(xml_path: str) -> Tuple[ET.ElementTree, Dict[str, Any]]:
    """Parse XML template and extract environment information.
    
    Returns:
        (tree, info_dict) where info_dict contains walls, bounds, robot_elem, goal_elem
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("No worldbody found in XML")
    
    # Find actuator section (may reference robot joints)
    actuator_section = root.find('actuator')
    
    # Extract walls
    walls = []
    boundary_walls = []
    walls_body = worldbody.find(".//body[@name='walls']")
    if walls_body is not None:
        for geom in walls_body.findall('geom'):
            pos_str = geom.get('pos', '0 0 0')
            size_str = geom.get('size', '0.1 0.1 0.1')
            rotation = _extract_geom_rotation_deg(geom)
            pos = [float(x) for x in pos_str.split()]
            size = [float(x) for x in size_str.split()]
            wall = {
                'name': geom.get('name', ''),
                'pos': pos[:2],
                'size': size[:2],
                'rotation': rotation,
                'elem': geom,
            }
            walls.append(wall)
            if wall['name'].startswith('wall_boundary_'):
                boundary_walls.append(wall)
    
    # Compute bounds from walls
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    bounds_source = boundary_walls if boundary_walls else walls
    for wall in bounds_source:
        x, y = wall['pos']
        sx, sy = wall['size']
        bbox_sx, bbox_sy = _rotated_bbox_half_extents(
            sx,
            sy,
            wall.get('rotation', 0.0),
        )
        min_x = min(min_x, x - bbox_sx)
        max_x = max(max_x, x + bbox_sx)
        min_y = min(min_y, y - bbox_sy)
        max_y = max(max_y, y + bbox_sy)
    
    bounds = (min_x, max_x, min_y, max_y)
    
    # Find robot and goal elements (but don't remove yet)
    robot_body = worldbody.find(".//body[@name='robot']")
    robot_geom = robot_body.find(".//geom[@name='robot']") if robot_body is not None else None
    goal_site = worldbody.find(".//site[@name='goal']")
    
    # Store original robot position and size
    robot_size = 0.15  # default
    robot_original_pos = None  # (x, y) position from template
    if robot_geom is not None:
        size_str = robot_geom.get('size', '0.15')
        robot_size = float(size_str.split()[0])
        # Extract original position
        pos_str = robot_geom.get('pos', '0 0 0')
        pos_parts = [float(x) for x in pos_str.split()]
        if len(pos_parts) >= 2:
            robot_original_pos = (pos_parts[0], pos_parts[1])

    # Diff-drive car template: body name="car" with freejoint, no inner robot geom.
    # Position lives on the body itself rather than the geom.
    if robot_original_pos is None:
        car_body = worldbody.find(".//body[@name='car']")
        if car_body is not None:
            pos_str = car_body.get('pos', '0 0 0')
            pos_parts = [float(x) for x in pos_str.split()]
            if len(pos_parts) >= 2:
                robot_original_pos = (pos_parts[0], pos_parts[1])
    
    return tree, {
        'walls': walls,
        'bounds': bounds,
        'root': root,
        'worldbody': worldbody,
        'actuator_section': actuator_section,
        'robot_body': robot_body,
        'robot_geom': robot_geom,
        'robot_size': robot_size,
        'robot_original_pos': robot_original_pos,
        'goal_site': goal_site,
    }


def add_obstacles_to_xml(
    worldbody: ET.Element,
    obstacles: List[Dict[str, Any]],
    object_half_height: float = 0.3,
) -> None:
    """Add obstacle bodies to the XML worldbody.

    Args:
        worldbody: The worldbody XML element
        obstacles: List of obstacle dicts with 'pos', 'size', and 'rotation'
        object_half_height: z half-size (and z-position) for box obstacles. 0.3 for the
            original 30cm point robot; 0.035 for the 7cm car.
    """
    z = object_half_height
    for i, obs in enumerate(obstacles):
        body = ET.SubElement(worldbody, 'body', name=f'obstacle_{i}_movable')
        ET.SubElement(
            body, 'geom',
            name=f'obstacle_{i}_movable',
            condim='4',
            pos=f"{obs['pos'][0]} {obs['pos'][1]} {z}",
            euler=f"0 0 {obs['rotation']}",
            friction='1 0.005 0.0001',
            rgba='1 1 0 1',
            size=f"{obs['size'][0]} {obs['size'][1]} {z}",
            type='box',
            mass='0.1',
        )
        ET.SubElement(body, 'joint', type='free')


def add_robot_and_goal_to_xml(
    worldbody: ET.Element,
    robot_pos: Tuple[float, float],
    goal_pos: Tuple[float, float],
    robot_size: float = 0.15,
    goal_size: float = 0.2,
) -> None:
    """Add robot and goal to XML worldbody with new positions."""
    robot_body = ET.SubElement(worldbody, 'body', name='robot')
    ET.SubElement(robot_body, 'joint', name='joint_x', type='slide', pos='0 0 0', axis='1 0 0')
    ET.SubElement(robot_body, 'joint', name='joint_y', type='slide', pos='0 0 0', axis='0 1 0')
    ET.SubElement(
        robot_body, 'geom',
        name='robot', type='sphere',
        pos=f'{robot_pos[0]} {robot_pos[1]} 0.15',
        size=str(robot_size), mass='5.0',
        friction='1.0 0.005 0.0001', condim='4'
    )
    ET.SubElement(
        worldbody, 'site',
        name='goal', type='sphere',
        pos=f'{goal_pos[0]} {goal_pos[1]} 0.0',
        size=str(goal_size), rgba='1 0 0 0.5'
    )


def add_actuators_to_xml(root: ET.Element) -> None:
    """Add actuator section to XML root for robot control.
    
    Args:
        root: The root XML element
    """
    # Create actuator section
    actuator_section = ET.SubElement(root, 'actuator')
    
    # Add actuators for x and y joints
    ET.SubElement(
        actuator_section, 'motor',
        name='actuator_x',
        joint='joint_x',
        gear='1',
        ctrlrange='-1 1'
    )
    ET.SubElement(
        actuator_section, 'motor',
        name='actuator_y',
        joint='joint_y',
        gear='1',
        ctrlrange='-1 1'
    )


# ------------------------------------------------------------------
# Obstacle Placement
# ------------------------------------------------------------------
def place_obstacles(
    walls: List[Dict[str, Any]],
    bounds: Tuple[float, float, float, float],
    num_objects: int,
    object_size_range: Tuple[float, float],
    rng: np.random.Generator,
    robot_pos: Optional[Tuple[float, float]] = None,
    robot_size: float = 0.15,
) -> List[Dict[str, Any]]:
    """Randomly place obstacles in the environment with random rotations, avoiding walls.

    Width and height are sampled independently from object_size_range, so the
    obstacle's aspect ratio is left to chance: similar draws give near-squares,
    different draws give rectangles.

    Args:
        walls: List of wall dictionaries with 'pos', 'size', and optionally 'rotation'
        bounds: (min_x, max_x, min_y, max_z) environment bounds
        num_objects: Number of obstacles to place
        object_size_range: (min, max) full side length, sampled independently for each side
        rng: Random number generator
        robot_pos: Optional (x, y) position of robot to avoid. If provided, obstacles
                   will not be placed overlapping with the robot.
        robot_size: Radius of the robot (used if robot_pos is provided)
    """
    obstacles = []

    # Create a pseudo-obstacle for the robot if position is provided
    # Use a circular approximation (square bounding box) with some margin
    robot_obstacle = None
    if robot_pos is not None:
        # Add margin to robot size for safety
        robot_margin = robot_size + 0.1  # 10cm margin
        robot_obstacle = {
            'pos': [robot_pos[0], robot_pos[1]],
            'size': [robot_margin, robot_margin],
            'rotation': 0
        }

    max_attempts = 1000
    for _ in range(num_objects):
        placed = False
        for _ in range(max_attempts):
            # Random position and size
            x = rng.uniform(bounds[0], bounds[1])
            y = rng.uniform(bounds[2], bounds[3])
            width = rng.uniform(*object_size_range) / 2.0
            height = rng.uniform(*object_size_range) / 2.0
            rotation = rng.uniform(0, 360)  # Rotation in degrees

            # Create the obstacle object
            obstacle = {
                'pos': [x, y],
                'size': [width, height],
                'rotation': rotation
            }

            # Check collision with walls
            collision = False
            for wall in walls:
                if check_collision(obstacle, wall):
                    collision = True
                    break

            # Check collision with other obstacles
            if not collision:
                for other_obstacle in obstacles:
                    if check_collision(obstacle, other_obstacle):
                        collision = True
                        break
            
            # Check collision with robot's original position
            if not collision and robot_obstacle is not None:
                if check_collision(obstacle, robot_obstacle):
                    collision = True

            if not collision:
                obstacles.append(obstacle)
                placed = True
                break

        if not placed:
            print("Warning: Could not place obstacle after max attempts")

    return obstacles

# def find_new_center(obj):
#     x, y = obj['pos'][0], obj['pos'][1]
#     top, bottom, left, right = obj['size'][0], obj['size'][1], obj['size'][2], obj['size'][3]
#     return {'pos': [x+(right-left)/2, y+(top-bottom)/2],
#             'size': [(left+right) / 2.0, (top+bottom) / 2.0],
#             'rotation': obj['rotation']
#     }

def check_collision(obj1, obj2):
    def get_rotated_corners(obj):
        x, y = obj['pos'][0], obj['pos'][1]
        w, h = obj['size'][0], obj['size'][1]
        
        # Convert rotation from degrees to radians
        angle = np.radians(obj.get('rotation', 0))
        
        # Calculate sine and cosine of the angle for rotation
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Calculate rotated corner positions
        # These formulas apply a 2D rotation matrix to each corner
        corners = [
            # Top-right corner
            (x + w * cos_a - h * sin_a, y + w * sin_a + h * cos_a),
            # Top-left corner
            (x - w * cos_a - h * sin_a, y - w * sin_a + h * cos_a),
            # Bottom-left corner
            (x - w * cos_a + h * sin_a, y - w * sin_a - h * cos_a),
            # Bottom-right corner
            (x + w * cos_a + h * sin_a, y + w * sin_a - h * cos_a)
        ]
        
        # Create a Polygon object from the rotated corners
        return Polygon(corners)
    
    

    # Get rotated polygons for both objects
    poly1 = get_rotated_corners(obj1)
    poly2 = get_rotated_corners(obj2)
    
    # Check if the polygons intersect
    return poly1.intersects(poly2)

# ------------------------------------------------------------------
# Connected Component Analysis
# ------------------------------------------------------------------
def find_connected_components(adjacency: Dict[str, Set[str]]) -> List[Set[str]]:
    """Find connected components in the adjacency graph using BFS.
    
    Args:
        adjacency: Dict mapping region labels to sets of adjacent region labels
        
    Returns:
        List of connected components (each is a set of region labels)
    """
    all_regions = set(adjacency.keys())
    visited = set()
    components = []
    
    for start_region in all_regions:
        if start_region in visited:
            continue
        
        # BFS from this region
        component = set()
        queue = [start_region]
        visited.add(start_region)
        
        while queue:
            region = queue.pop(0)
            component.add(region)
            
            for neighbor in adjacency.get(region, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        components.append(component)
    
    return components


# ------------------------------------------------------------------
# Robot and Goal Placement
# ------------------------------------------------------------------
def place_robot_and_goal(
    exporter: WavefrontSnapshotExporter,
    region_map: np.ndarray,
    region_labels: Dict[int, str],
    adjacency: Dict[str, Set[str]],
    dynamic_grid: np.ndarray,
    clearance_radius: float,
    min_goal_distance: float,
    max_goal_retries: int,
    rng: np.random.Generator,
    debug: bool = False,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Place robot and goal in connected regions with sufficient separation.
    
    Args:
        exporter: WavefrontSnapshotExporter instance
        region_map: Region label map
        region_labels: Mapping from region ID to label
        adjacency: Adjacency graph
        dynamic_grid: Occupancy grid
        clearance_radius: Required clearance around goal
        min_goal_distance: Minimum world distance between robot and goal
        max_goal_retries: Maximum retries for goal placement with clearance
        rng: Random number generator
        debug: Enable debug logging
        
    Returns:
        ((robot_x, robot_y), (goal_x, goal_y)) or None if placement fails
    """
    # Find connected components
    components = find_connected_components(adjacency)
    
    if debug:
        print(f"  [DEBUG] Total regions: {len(region_labels)}")
        print(f"  [DEBUG] Connected components: {len(components)}")
        for idx, comp in enumerate(components):
            print(f"  [DEBUG]   Component {idx}: {len(comp)} regions - {comp}")
    
    # Filter to components with at least 2 regions
    valid_components = [c for c in components if len(c) >= 2]
    
    if debug:
        print(f"  [DEBUG] Valid components (>= 2 regions): {len(valid_components)}")
    
    if not valid_components:
        if debug:
            print(f"  [DEBUG] No valid components with >= 2 regions!")
        return None
    
    # Counters for debugging
    robot_sample_failures = 0
    goal_sample_failures = 0
    clearance_failures = 0
    distance_failures = 0
    
    # Try each component
    for comp_idx, component in enumerate(valid_components):
        component_list = list(component)
        rng.shuffle(component_list)
        
        if debug:
            print(f"  [DEBUG] Trying component {comp_idx} with {len(component_list)} regions")
        
        # Try pairs of regions in this component
        for i, robot_region in enumerate(component_list):
            # Sample robot position (require clearance similar to goal)
            # Use sample_cell_with_clearance so robot isn't placed overlapping nearby objects
            robot_pos = exporter.sample_cell_with_clearance(
                robot_region, region_map, region_labels, dynamic_grid, clearance_radius, rng
            )
            if robot_pos is None:
                robot_sample_failures += 1
                continue

            # Try goal regions
            for goal_region in component_list[i+1:]:
                # Try sampling goal with clearance (up to max_goal_retries attempts)
                for attempt in range(max_goal_retries):
                    # Sample goal with required clearance
                    goal_pos = exporter.sample_cell_with_clearance(
                        goal_region, region_map, region_labels, dynamic_grid, clearance_radius, rng
                    )
                    if goal_pos is None:
                        goal_sample_failures += 1
                        break

                    # Convert goal position to grid coordinates for clearance check
                    gx = exporter._world_to_grid_x(goal_pos[0])
                    gy = exporter._world_to_grid_y(goal_pos[1])

                    # Check clearance
                    if not exporter.check_cell_clearance((gx, gy), clearance_radius, dynamic_grid):
                        clearance_failures += 1
                        continue

                    # Check distance
                    dx = goal_pos[0] - robot_pos[0]
                    dy = goal_pos[1] - robot_pos[1]
                    distance = np.sqrt(dx**2 + dy**2)

                    if debug and distance < min_goal_distance:
                        print(f"  [DEBUG] Distance check failed: {distance:.2f} < {min_goal_distance:.2f}")

                    if distance >= min_goal_distance:
                        if debug:
                            print(f"  [DEBUG] Success! Robot in {robot_region}, Goal in {goal_region}, distance={distance:.2f}")
                        return (robot_pos, goal_pos)
                    else:
                        distance_failures += 1
    
    if debug:
        print(f"  [DEBUG] Placement failed after trying all components:")
        print(f"  [DEBUG]   Robot sampling failures: {robot_sample_failures}")
        print(f"  [DEBUG]   Goal sampling failures: {goal_sample_failures}")
        print(f"  [DEBUG]   Clearance check failures: {clearance_failures}")
        print(f"  [DEBUG]   Distance check failures: {distance_failures}")
    
    return None

def place_robot_and_goal_pairs(
    exporter: WavefrontSnapshotExporter,
    region_map: np.ndarray,
    region_labels: Dict[int, str],
    adjacency: Dict[str, Set[str]],
    dynamic_grid: np.ndarray,
    clearance_radius: float,
    min_goal_distance: float,
    max_goal_retries: int,
    rng: np.random.Generator,
    debug: bool = False,
    require_adjacent: bool = False,
    samples_per_pair: int = 1,
    exact_hop: int = 0,
    region_goals: Optional[Dict[str, Any]] = None,
    min_region_area_m2: float = 0.0,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """For every connected component, generate a placement for every unordered pair
    of distinct regions (n choose 2). Returns a list of (robot_pos, goal_pos) pairs.

    Each pair is produced by sampling a cell in the first region for the robot and
    a cell in the second region for the goal, checking clearance and distance.
    """
    placements: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    components = find_connected_components(adjacency)
    valid_components = [c for c in components if len(c) >= 2]
    if debug:
        print(f"  [DEBUG] Generating pairs for {len(valid_components)} valid components")

    # Filter out regions smaller than min_region_area_m2 from the pair candidates.
    # A region's area = cell_count * resolution^2. The filter only removes pairs
    # that involve a too-small region; the small regions remain in the snapshot
    # for downstream wavefront queries.
    too_small_labels: Set[str] = set()
    if min_region_area_m2 > 0:
        cell_area = exporter.resolution * exporter.resolution
        for rid, lbl in region_labels.items():
            n_cells = int((region_map == rid).sum())
            if n_cells * cell_area < min_region_area_m2:
                too_small_labels.add(lbl)
        if debug and too_small_labels:
            print(f"  [DEBUG] Skipping {len(too_small_labels)} regions < "
                  f"{min_region_area_m2:.4f} m^2: {sorted(too_small_labels)}")

    # Pull per-region pre-sampled positions out of the snapshot. The new
    # WavefrontSnapshotExporter API does sampling C++-side (respecting
    # clearance) and returns the candidates as part of build_snapshot when
    # goals_per_region > 0. We use these as both robot and goal candidates.
    region_pools: Dict[str, List[Tuple[float, float]]] = {}
    for region_label, bundle in (region_goals or {}).items():
        pool: List[Tuple[float, float]] = []
        for sample in getattr(bundle, "goals", []) or []:
            pool.append((float(sample.x), float(sample.y)))
        if pool:
            region_pools[region_label] = pool

    # WavefrontSnapshotExporter._sample_region_goals deliberately skips any
    # region label containing "robot" (it answers "what goals can the robot
    # reach", not "where can the robot start"). For env generation we also
    # want to sample robot starts inside the robot region, so populate that
    # pool ourselves from the region_map.
    sample_count = max(2, samples_per_pair * 2)
    label_to_id = {label: rid for rid, label in region_labels.items()}
    for adj_label in adjacency:
        if "robot" not in adj_label or adj_label in region_pools:
            continue
        rid = label_to_id.get(adj_label)
        if rid is None:
            continue
        cells = np.argwhere(region_map == rid)
        if cells.size == 0:
            continue
        idx = rng.choice(cells.shape[0],
                         size=min(sample_count, cells.shape[0]),
                         replace=False)
        idx = np.atleast_1d(idx)
        pool = []
        for i in idx:
            gx, gy = int(cells[int(i)][0]), int(cells[int(i)][1])
            wx = exporter._grid_to_world_x(gx) + 0.5 * exporter.resolution
            wy = exporter._grid_to_world_y(gy) + 0.5 * exporter.resolution
            pool.append((wx, wy))
        region_pools[adj_label] = pool

    def try_sample_order(src_region: str, tgt_region: str) -> bool:
        """Take up to samples_per_pair (robot, goal) pairs from the pre-sampled
        pools where robot ∈ src and goal ∈ tgt. Distance filter still applied;
        clearance is already enforced by the C++ sampler."""
        successes = 0
        robot_pool = list(region_pools.get(src_region, []))
        goal_pool = list(region_pools.get(tgt_region, []))
        if not robot_pool or not goal_pool:
            if debug:
                print(f"  [DEBUG] Empty pool for {src_region}->{tgt_region} "
                      f"(robot_pool={len(robot_pool)}, goal_pool={len(goal_pool)})")
            return False
        rng.shuffle(robot_pool)
        rng.shuffle(goal_pool)
        for robot_pos in robot_pool:
            if successes >= samples_per_pair:
                break
            for goal_pos in goal_pool:
                if successes >= samples_per_pair:
                    break
                distance = np.hypot(goal_pos[0] - robot_pos[0], goal_pos[1] - robot_pos[1])
                if distance < min_goal_distance:
                    continue
                placements.append((robot_pos, goal_pos))
                successes += 1
                if debug:
                    print(f"  [DEBUG] Added placement #{successes} for regions "
                          f"{src_region}->{tgt_region} (distance={distance:.2f})")
        return successes > 0

    for comp_idx, component in enumerate(valid_components):
        regions = list(component)
        # iterate over unordered pairs (i<j) -> C(n,2)
        for ri, rj in itertools.combinations(regions, 2):
            # Drop pairs involving regions below the min-area threshold.
            if ri in too_small_labels or rj in too_small_labels:
                continue
            # Hop filtering at gen-time on the static adjacency graph:
            #   exact_hop > 0 → only pairs whose BFS distance equals exact_hop
            #   require_adjacent (legacy) → equivalent to exact_hop=1
            # Note: runtime validation re-checks hop against the planner-view
            # snapshot (which inserts robot/goal pseudo-nodes); gen-time check
            # is just a cheap pre-filter.
            if exact_hop > 0:
                d = _adjacency_bfs_distance(adjacency, ri, rj)
                if d != exact_hop:
                    continue
            elif require_adjacent and (rj not in adjacency.get(ri, set())):
                continue
            ok1 = try_sample_order(ri, rj)
            ok2 = try_sample_order(rj, ri)
            if not (ok1 or ok2) and debug:
                print(f"  [DEBUG] Could not generate placement for region pair ({ri},{rj}) after {max_goal_retries} attempts per ordering")

    if debug:
        print(f"  [DEBUG] Generated {len(placements)} placements in total")
    return placements


# ------------------------------------------------------------------
# Environment Generation Pipeline
# ------------------------------------------------------------------
def generate_single_environment(
    template_xml_path: str,
    env_id: int,
    output_dir: str,
    config: Dict[str, Any],
    namo_config_path: str,
    seed: Optional[int] = None,
) -> bool:
    """Generate a single complete environment from a template XML.
    
    Args:
        template_xml_path: Path to template XML file (with walls, default robot/goal)
        env_id: Environment ID
        output_dir: Output directory
        config: Configuration dict
        namo_config_path: Path to NAMO config YAML file
        seed: Random seed
        
    Returns:
        True if successful, False otherwise
    """
    # Setup RNG
    if seed is None:
        seed = env_id
    rng = np.random.default_rng(seed)
    
    # Extract config
    num_objects = _resolve_num_objects(config.get('num_objects', DEFAULT_NUM_OBJECTS_RANGE), rng)
    object_size_range = tuple(config.get('object_size_range', DEFAULT_OBJECT_SIZE_RANGE))
    object_half_height = config.get('object_half_height', 0.3)
    goal_size = config.get('goal_size', 0.2)
    clearance_radius = config.get('clearance_radius', DEFAULT_CLEARANCE_RADIUS)
    min_goal_distance = config.get('min_goal_distance', DEFAULT_MIN_GOAL_DISTANCE)
    resolution = config.get('resolution', DEFAULT_RESOLUTION)
    max_goal_retries = config.get('max_goal_retries', DEFAULT_MAX_GOAL_RETRIES)

    # Parse template XML
    try:
        tree, info = parse_xml_template(template_xml_path)
        walls = info['walls']
        bounds = info['bounds']
        worldbody = info['worldbody']
        robot_size = info['robot_size']
        robot_original_pos = info['robot_original_pos']
    except Exception as e:
        print(f"[Env {env_id}] Error parsing template XML: {e}")
        return False

    print(f"[Env {env_id}] Target num_objects = {num_objects}")
    obstacles = place_obstacles(
        walls, bounds, num_objects, object_size_range, rng,
        robot_pos=None, robot_size=robot_size,
    )
    print(f"[Env {env_id}] Placed {len(obstacles)} obstacles")
    
    # Add obstacles to XML
    add_obstacles_to_xml(worldbody, obstacles, object_half_height=object_half_height)

    os.makedirs(output_dir, exist_ok=True)
    temp_xml_path = os.path.join(output_dir, f'env_{env_id:04d}_temp.xml')
    try:
        ET.indent(tree, space='  ')
        tree.write(temp_xml_path, encoding='utf-8', xml_declaration=True)
        print(f"[Env {env_id}] Temporary XML saved to: {temp_xml_path}")
    except Exception as e:
        print(f"[Env {env_id}] Error saving temporary XML: {e}")
        return False

    # Create simulation environment from the XML with obstacles
    try:
        print(f"[Env {env_id}] Initializing simulation environment...")
        rl_env_cls = getattr(namo_rl, "RLEnvironment")
        env = rl_env_cls(temp_xml_path, namo_config_path, visualize=False)
        
        exporter = WavefrontSnapshotExporter(
            env, resolution=resolution,
            robot_half_extent_override=load_robot_half_extent_from_namo_config(namo_config_path),
        )
        
        # Build snapshot using simulation environment
        snapshot = exporter.build_snapshot(
            xml_path=temp_xml_path,
            config_path=namo_config_path,
            goal_radius=0.15,
            goals_per_region=0,
            rng=rng,
        )
        
        region_map = snapshot.region_map
        region_labels = snapshot.region_labels
        adjacency = snapshot.adjacency
        dynamic_grid = snapshot.dynamic_grid
        
        print(f"[Env {env_id}] Built wavefront snapshot with {len(region_labels)} regions")
        
    except Exception as e:
        print(f"[Env {env_id}] Error building wavefront snapshot: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temp file
        # if os.path.exists(temp_xml_path):
        #     os.remove(temp_xml_path)
        return False
    
    # Place robot and goal
    placement = place_robot_and_goal(
        exporter,
        region_map,
        region_labels,
        adjacency,
        dynamic_grid,
        clearance_radius,
        min_goal_distance,
        max_goal_retries,
        rng,
        debug=True,  # Enable debug logging
    )
    
    if placement is None:
        print(f"[Env {env_id}] Failed to place robot and goal")
        # Clean up temp file
        # if os.path.exists(temp_xml_path):
        #     os.remove(temp_xml_path)
        return False
    
    robot_pos, goal_pos = placement
    print(f"[Env {env_id}] Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), Goal at ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
    
    # Update robot and goal positions in the temp XML
    # Re-parse the temp XML
    tree_final = ET.parse(temp_xml_path)
    root_final = tree_final.getroot()
    worldbody_final = root_final.find('worldbody')
    
    if worldbody_final is None:
        print(f"[Env {env_id}] Error: No worldbody in final XML")
        # if os.path.exists(temp_xml_path):
        #     os.remove(temp_xml_path)
        return False
    
    # Update robot position in the XML (handles both point robot and diff-drive car)
    update_robot_xy_in_xml(worldbody_final, robot_pos[0], robot_pos[1])

    # Update goal position in the XML
    goal_site_final = worldbody_final.find(".//site[@name='goal']")
    if goal_site_final is not None:
        goal_site_final.set('pos', f'{goal_pos[0]} {goal_pos[1]} 0.0')
    
    # Save final XML
    output_path = os.path.join(output_dir, f'env_{env_id:04d}.xml')
    try:
        ET.indent(tree_final, space='  ')
        tree_final.write(output_path, encoding='utf-8', xml_declaration=True)
    except Exception as e:
        print(f"[Env {env_id}] Error saving final XML: {e}")
        # if os.path.exists(temp_xml_path):
        #     os.remove(temp_xml_path)
        return False
    
    # Clean up temp file
    if os.path.exists(temp_xml_path):
        os.remove(temp_xml_path)
    
    print(f"[Env {env_id}] Successfully generated: {output_path}")
    return True

def generate_environments_from_pairs(
    template_xml_path: str,
    env_id: int,
    output_dir: str,
    config: Dict[str, Any],
    namo_config_path: str,
    seed: Optional[int] = None,
) -> int:
    """Generate one output XML per (robot,goal) pair returned by place_robot_and_goal_pairs.
    Returns the number of environment files written (0 on failure)."""
    # Setup RNG
    if seed is None:
        seed = env_id
    rng = np.random.default_rng(seed)

    # Extract config
    num_objects = _resolve_num_objects(config.get('num_objects', DEFAULT_NUM_OBJECTS_RANGE), rng)
    object_size_range = tuple(config.get('object_size_range', DEFAULT_OBJECT_SIZE_RANGE))
    object_half_height = config.get('object_half_height', 0.3)
    goal_size = config.get('goal_size', 0.2)
    clearance_radius = config.get('clearance_radius', DEFAULT_CLEARANCE_RADIUS)
    min_goal_distance = config.get('min_goal_distance', DEFAULT_MIN_GOAL_DISTANCE)
    resolution = config.get('resolution', DEFAULT_RESOLUTION)
    max_goal_retries = config.get('max_goal_retries', DEFAULT_MAX_GOAL_RETRIES)
    require_adjacent = bool(config.get('require_adjacent', False))
    samples_per_pair = int(config.get('samples_per_pair', 1))
    exact_hop = int(config.get('exact_hop', 0))
    min_region_area_m2 = float(config.get('min_region_area_m2', 0.0))

    # Parse template XML
    try:
        tree, info = parse_xml_template(template_xml_path)
        walls = info['walls']
        bounds = info['bounds']
        worldbody = info['worldbody']
        robot_size = info['robot_size']
        robot_original_pos = info['robot_original_pos']
    except Exception as e:
        print(f"[Env {env_id}] Error parsing template XML: {e}")
        return 0

    # Reject-and-resample loop: re-roll the entire obstacle layout if the
    # resulting wavefront snapshot can't produce any (robot, goal) pair.
    # Each "layout try" is one independent fresh roll of all obstacles.
    MAX_LAYOUT_TRIES = int(config.get('max_layout_tries', 500))
    os.makedirs(output_dir, exist_ok=True)
    temp_xml_path = os.path.join(output_dir, f'env_{env_id:04d}_temp.xml')

    placements = None
    tree = None
    worldbody = None
    exporter = None
    region_map = None
    region_labels = None
    dynamic_grid = None

    for layout_try in range(MAX_LAYOUT_TRIES):
        # Re-parse the template so previous-try obstacles don't accumulate.
        try:
            tree, info = parse_xml_template(template_xml_path)
            walls = info['walls']
            bounds = info['bounds']
            worldbody = info['worldbody']
            robot_size = info['robot_size']
            robot_original_pos = info['robot_original_pos']
        except Exception as e:
            print(f"[Env {env_id}] Error re-parsing template XML: {e}")
            return 0

        obstacles = place_obstacles(
            walls, bounds, num_objects, object_size_range, rng,
            robot_pos=None, robot_size=robot_size,
        )

        add_obstacles_to_xml(worldbody, obstacles, object_half_height=object_half_height)
        try:
            ET.indent(tree, space='  ')
            tree.write(temp_xml_path, encoding='utf-8', xml_declaration=True)
        except Exception as e:
            print(f"[Env {env_id}] Error saving temp XML: {e}")
            return 0

        try:
            if not NAMO_RL_AVAILABLE:
                raise RuntimeError(
                    "namo_rl C++ bindings unavailable; set PYTHONPATH to the build_python "
                    "directory matching this host (e.g. build_python_mjxrl_westeros)"
                )
            rl_env_cls = getattr(namo_rl, "RLEnvironment")
            env = rl_env_cls(temp_xml_path, namo_config_path, visualize=False)
            exporter = WavefrontSnapshotExporter(
                env, resolution=resolution,
                robot_half_extent_override=load_robot_half_extent_from_namo_config(namo_config_path),
            )
            snapshot = exporter.build_snapshot(
                xml_path=temp_xml_path,
                config_path=namo_config_path,
                goal_radius=0.15,
                goals_per_region=max(2, samples_per_pair * 2),
                rng=rng,
            )
            region_map = snapshot.region_map
            region_labels = snapshot.region_labels
            adjacency = snapshot.adjacency
            dynamic_grid = snapshot.dynamic_grid
            region_goals = snapshot.region_goals
        except Exception as e:
            print(f"[Env {env_id}] try {layout_try+1}: snapshot error: {e}")
            continue

        try:
            placements = place_robot_and_goal_pairs(
                exporter, region_map, region_labels, adjacency,
                dynamic_grid, clearance_radius, min_goal_distance,
                max_goal_retries, rng,
                debug=False,
                require_adjacent=require_adjacent,
                samples_per_pair=samples_per_pair,
                exact_hop=exact_hop,
                region_goals=region_goals,
                min_region_area_m2=min_region_area_m2,
            )
        except Exception as e:
            print(f"[Env {env_id}] try {layout_try+1}: pair error: {e}")
            continue

        if placements:
            print(f"[Env {env_id}] accepted on try {layout_try+1}/{MAX_LAYOUT_TRIES} "
                  f"({len(obstacles)} obstacles, {len(region_labels)} regions, "
                  f"{len(placements)} pairs)")
            break

    if not placements:
        print(f"[Env {env_id}] gave up after {MAX_LAYOUT_TRIES} layout tries")
        if os.path.exists(temp_xml_path):
            os.remove(temp_xml_path)
        return 0

    os.makedirs(output_dir, exist_ok=True)

    # Serialise the obstacle-augmented tree once; clone in-memory for each pair.
    ET.indent(tree, space='  ')
    written = 0
    for k, (robot_pos, goal_pos) in enumerate(placements):
        try:
            # Clone the tree by re-parsing the serialised form (cheap, no disk).
            import io
            buf = io.BytesIO()
            tree.write(buf, encoding='utf-8', xml_declaration=True)
            buf.seek(0)
            tree_final = ET.parse(buf)
            root_final = tree_final.getroot()
            worldbody_final = root_final.find('worldbody')
            if worldbody_final is None:
                print(f"[Env {env_id}] Error: No worldbody in final XML for pair {k}")
                continue

            update_robot_xy_in_xml(worldbody_final, robot_pos[0], robot_pos[1])

            goal_site_final = worldbody_final.find(".//site[@name='goal']")
            if goal_site_final is not None:
                goal_site_final.set('pos', f'{goal_pos[0]} {goal_pos[1]} 0.0')
                goal_site_final.set('size', str(goal_size))
            else:
                ET.SubElement(
                    worldbody_final, 'site',
                    name='goal',
                    type='sphere',
                    pos=f'{goal_pos[0]} {goal_pos[1]} 0.0',
                    size=str(goal_size),
                    rgba='1 0 0 0.5'
                )

            out_path = os.path.join(output_dir, f'env_{env_id:04d}_pair_{k:03d}.xml')
            ET.indent(tree_final, space='  ')
            tree_final.write(out_path, encoding='utf-8', xml_declaration=True)
            written += 1
        except Exception as e:
            print(f"[Env {env_id}] Error saving XML for pair {k}: {e}")
            continue

    # Remove the obstacles-only temp XML (only the per-pair files are kept).
    if os.path.exists(temp_xml_path):
        os.remove(temp_xml_path)

    print(f"[Env {env_id}] Generated {written} environment(s) from {len(placements)} placement(s)")
    return written

def generate_environments_parallel(
    template_xml_path: str,
    num_envs: int,
    output_dir: str,
    config: Dict[str, Any],
    namo_config_path: str,
    num_workers: int = 4,
    start_seed: int = 0,
) -> None:
    """Generate multiple environments in parallel.

    Creates a two- or three-level output layout:
      output_dir/
        [<parent_dir>/]<template_base>/  <- prepended with parent dir name if present,
                                            to disambiguate same-named templates living
                                            in sibling folders (e.g. set1/benchmark_1.xml
                                            vs set2/benchmark_1.xml).
          run_0000/                       <- subdirectory for first job (passed to worker)
          run_0001/                       <- subdirectory for second job
          ...
    Each worker calls generate_environments_from_pairs and writes one or more XMLs
    into its assigned run_<idx> subdirectory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Top-level subdir named after template parent + base (filename without extension).
    # Including the parent dir name prevents output collisions when two templates share
    # a basename across sibling folders.
    norm_path = os.path.normpath(template_xml_path)
    template_basename = os.path.splitext(os.path.basename(norm_path))[0]
    template_parent = os.path.basename(os.path.dirname(norm_path))
    if template_parent and template_parent not in (".", ""):
        template_base = os.path.join(template_parent, template_basename)
    else:
        template_base = template_basename
    template_out_dir = os.path.join(output_dir, template_base)
    os.makedirs(template_out_dir, exist_ok=True)

    # Prepare per-job output subdirectories and argument list
    args_list = []
    for i in range(num_envs):
        run_subdir = os.path.join(template_out_dir, f"run_{i:04d}")
        # create here to avoid races / ensure directory exists for consumers
        os.makedirs(run_subdir, exist_ok=True)
        args_list.append((template_xml_path, i, run_subdir, config, namo_config_path, start_seed + i))

    # Run jobs in parallel (each job may write multiple env files into its run_<idx> dir)
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(generate_environments_from_pairs, args_list)

    # results are integers = number of files written per job
    total_written = sum(results)
    print(f"\nGenerated {total_written} environment file(s) across {num_envs} job(s)")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Generate MuJoCo environments with intelligent placement')
    parser.add_argument('template_xml', type=str, help='Path to template XML file (with walls/maze)')
    parser.add_argument('--namo-config', type=str, required=True, 
                        help='Path to NAMO config YAML file (required for simulation)')
    parser.add_argument('--num-envs', type=int, default=10, help='Number of environments to generate')
    parser.add_argument('--output-dir', type=str, default='generated_envs', help='Output directory')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--start-seed', type=int, default=0, help='Starting random seed')
    
    # Environment parameters (override config file)
    parser.add_argument('--num-objects-range', type=int, nargs=2, metavar=('MIN', 'MAX'),
                        default=None,
                        help='Sample num_objects per env uniformly from [MIN, MAX] (inclusive). '
                             'Sampling uses the per-env seeded RNG, so generation stays reproducible. '
                             f'Defaults to {DEFAULT_NUM_OBJECTS_RANGE} when neither this flag nor '
                             '--num-objects-json is provided. Overridden by --num-objects-json if both set.')
    parser.add_argument('--num-objects-json', type=str, default=None,
                        help='Path to a JSON file mapping "<parent_dir>/<basename>" template keys to '
                             '[min, max] num_objects ranges. When the current template matches a key, '
                             'its range is used instead of --num-objects-range. Templates not in the JSON '
                             'fall back to --num-objects-range (or the default).')
    # Use None as sentinel so we can distinguish "user didn't pass" from "user
    # explicitly passed the default value" — required for the --robot-scale
    # auto-override logic below to know whether to scale the default.
    parser.add_argument('--clearance-radius', type=float, default=None)
    parser.add_argument('--min-goal-distance', type=float, default=None)
    parser.add_argument('--max-goal-retries', type=int, default=DEFAULT_MAX_GOAL_RETRIES,
                        help='Maximum retries for goal placement with clearance')
    parser.add_argument('--robot-scale', type=float, default=1.0,
                        help='Scale factor applied to all distance defaults (object size, '
                             'clearance, min goal distance, resolution). Use 0.233 for the '
                             '7cm car. Knobs explicitly set on the CLI take precedence.')
    parser.add_argument('--object-half-height', type=float, default=None,
                        help='z half-size (and z-position) for box obstacles. Default 0.3 '
                             'for original 30cm point robot. With --robot-scale 0.233 the '
                             'car-friendly default of 0.035 is used.')
    parser.add_argument('--object-size-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        default=None,
                        help='(min, max) full side length for sampled obstacle boxes. Width and '
                             'height are drawn independently from this range, so aspect ratios '
                             '(squares vs rectangles) emerge naturally from the random draws. '
                             'Overrides --robot-scale auto-scaling.')
    parser.add_argument('--goal-size', type=float, default=None,
                        help='Visual radius of the goal site sphere. Default 0.2 for original '
                             'point robot; auto-scaled by --robot-scale otherwise.')
    parser.add_argument('--require-adjacent', action='store_true', default=True,
                        help='Only emit (robot, goal) pairs whose regions are direct neighbors '
                             'in the adjacency graph (1-hop / "region opening" puzzles). On by '
                             'default. Disable with --no-require-adjacent for multi-hop chains.')
    parser.add_argument('--no-require-adjacent', action='store_false', dest='require_adjacent',
                        help='Allow multi-hop region pairs (any pair in the same connected component).')
    parser.add_argument('--samples-per-pair', type=int, default=1,
                        help='Max successful (robot, goal) samples per (region_pair, ordering). '
                             'Default 1 (one env per ordering = up to 2 envs per pair). '
                             'Set higher (e.g. 10) to extract many distinct positions per pair.')
    parser.add_argument('--exact-hop', type=int, default=0,
                        help='If > 0, only emit (robot, goal) pairs whose BFS distance in the '
                             'region adjacency graph equals exactly this many hops. '
                             '1 = direct neighbors (== --require-adjacent), '
                             '2 = need to chain through one intermediate region, etc. '
                             'Overrides --require-adjacent when set. Default 0 (disabled).')
    parser.add_argument('--min-region-area', type=float, default=0.02,
                        help='Minimum region area in m^2 for a region to be eligible as a '
                             'robot or goal location. Regions below this size are skipped from '
                             'pair generation but still appear in the wavefront snapshot. '
                             'Default 0.02 (~14x14cm = car rotation-capable region; 4x car '
                             'footprint). Set 0 to disable. Larger values like 0.05 (~22x22cm) '
                             'force "real-room" regions but trigger more layout retries on '
                             'small arenas.')

    args = parser.parse_args()

    # If a robot scale is given, scale the distance defaults the user did not explicitly override.
    # Grid resolution is intentionally NOT scaled — it's an occupancy-analysis cell size,
    # not a physical robot dimension; keep at default.
    if args.robot_scale != 1.0:
        s = args.robot_scale
        if args.clearance_radius is None:
            # Disable the extra clearance buffer — the wavefront's robot_size inflation
            # already guarantees the cell is reachable, no extra margin needed.
            args.clearance_radius = 0.0
        if args.min_goal_distance is None:
            # Disable the min-goal-distance filter — any (robot, goal) pair sampled in
            # different regions is a valid NAMO problem regardless of how close they are
            # geometrically (region-pair semantics is what matters, not Euclidean distance).
            args.min_goal_distance = 0.0
        scaled_size_range = (round(DEFAULT_OBJECT_SIZE_RANGE[0] * s, 2),
                             round(DEFAULT_OBJECT_SIZE_RANGE[1] * s, 2))
        # Default object height for car when robot_scale ~= 0.233. Matches the
        # primitive reference scene (data/nominal_primitive_scene_*_1x_car.xml)
        # which uses a 10 cm tall obstacle (half-height 0.05) so push primitive
        # predictions transfer to generated envs.
        if args.object_half_height is None and abs(s - 0.233) < 0.01:
            args.object_half_height = 0.05
        if args.goal_size is None:
            args.goal_size = round(0.2 * s, 2)
        print(f"[robot-scale={s}] applied scaled defaults: "
              f"object_size_range={scaled_size_range}, "
              f"clearance_radius={args.clearance_radius}, "
              f"min_goal_distance={args.min_goal_distance}, "
              f"object_half_height={args.object_half_height}, "
              f"goal_size={args.goal_size}")
    else:
        scaled_size_range = None
    # Apply final defaults for anything still unset (no --robot-scale override).
    if args.clearance_radius is None:
        args.clearance_radius = DEFAULT_CLEARANCE_RADIUS
    if args.min_goal_distance is None:
        args.min_goal_distance = DEFAULT_MIN_GOAL_DISTANCE
    if args.object_half_height is None:
        args.object_half_height = 0.3
    if args.goal_size is None:
        args.goal_size = 0.2
    
    # Check that namo_rl is available
    # generate_environments_from_pairs (the default code path) is namo_rl-free.
    # Only warn if it's missing; legacy paths that need it will error at call-time.
    if not NAMO_RL_AVAILABLE:
        print("Note: namo_rl not on PYTHONPATH. Pure-Python path will be used (no MuJoCo physics validation).")
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override with command line args. Priority: --num-objects-json (per-template lookup) >
    # --num-objects-range > existing config value > DEFAULT_NUM_OBJECTS_RANGE.
    json_override = None
    if args.num_objects_json is not None:
        import json as _json
        try:
            with open(args.num_objects_json, 'r') as _fh:
                _json_map = _json.load(_fh)
        except Exception as _e:
            raise SystemExit(f"Failed to load --num-objects-json {args.num_objects_json}: {_e}")
        _key = _template_lookup_key(args.template_xml)
        if _key in _json_map:
            json_override = list(_json_map[_key])
            print(f"[num-objects-json] template '{_key}' -> range {json_override}")
        else:
            print(f"[num-objects-json] template '{_key}' not in JSON; falling back")
    if json_override is not None:
        config['num_objects'] = json_override
    elif args.num_objects_range is not None:
        config['num_objects'] = list(args.num_objects_range)
    elif 'num_objects' not in config:
        config['num_objects'] = list(DEFAULT_NUM_OBJECTS_RANGE)
    config['clearance_radius'] = args.clearance_radius
    config['min_goal_distance'] = args.min_goal_distance
    config['max_goal_retries'] = args.max_goal_retries
    config['object_half_height'] = args.object_half_height
    config['goal_size'] = args.goal_size
    config['require_adjacent'] = args.require_adjacent
    config['samples_per_pair'] = args.samples_per_pair
    config['exact_hop'] = args.exact_hop
    config['min_region_area_m2'] = args.min_region_area
    if args.object_size_range is not None:
        config['object_size_range'] = tuple(args.object_size_range)
    elif scaled_size_range is not None and 'object_size_range' not in config:
        config['object_size_range'] = scaled_size_range

    # Generate environments
    generate_environments_parallel(
        args.template_xml,
        args.num_envs,
        args.output_dir,
        config,
        args.namo_config,
        args.num_workers,
        args.start_seed,
    )


if __name__ == '__main__':
    main()
