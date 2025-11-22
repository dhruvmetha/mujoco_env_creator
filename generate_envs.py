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

try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not available, using basic collision detection")

# Import MuJoCo environment and wavefront snapshot exporter
try:
    import namo_rl
    NAMO_RL_AVAILABLE = True
except ImportError:
    NAMO_RL_AVAILABLE = False
    print("Warning: namo_rl not available. Make sure PYTHONPATH includes the namo_cpp build directory.")
    sys.exit(1)

from wavefront_snapshot import WavefrontSnapshotExporter


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_NUM_OBJECTS = 10
DEFAULT_OBJECT_SIZE_RANGE = (0.4, 1.4)  # (min, max) side length (2x half-size from config)
DEFAULT_CLEARANCE_RADIUS = 0.3  # meters
DEFAULT_MIN_GOAL_DISTANCE = 1.0  # meters minimum separation between robot and goal
DEFAULT_RESOLUTION = 0.01  # meters per grid cell
DEFAULT_MAX_GOAL_RETRIES = 10  # maximum retries for goal placement with clearance


# ------------------------------------------------------------------
# XML Parsing and Manipulation
# ------------------------------------------------------------------
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
    walls_body = worldbody.find(".//body[@name='walls']")
    if walls_body is not None:
        for geom in walls_body.findall('geom'):
            pos_str = geom.get('pos', '0 0 0')
            size_str = geom.get('size', '0.1 0.1 0.1')
            pos = [float(x) for x in pos_str.split()]
            size = [float(x) for x in size_str.split()]
            walls.append({'pos': pos[:2], 'size': size[:2], 'elem': geom})
    
    # Compute bounds from walls
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    
    for wall in walls:
        x, y = wall['pos']
        sx, sy = wall['size']
        min_x = min(min_x, x - sx)
        max_x = max(max_x, x + sx)
        min_y = min(min_y, y - sy)
        max_y = max(max_y, y + sy)
    
    bounds = (min_x, max_x, min_y, max_y)
    
    # Find robot and goal elements (but don't remove yet)
    robot_body = worldbody.find(".//body[@name='robot']")
    robot_geom = robot_body.find(".//geom[@name='robot']") if robot_body is not None else None
    goal_site = worldbody.find(".//site[@name='goal']")
    
    # Store original robot position if available (for robot size)
    robot_size = 0.15  # default
    if robot_geom is not None:
        size_str = robot_geom.get('size', '0.15')
        robot_size = float(size_str.split()[0])
    
    return tree, {
        'walls': walls,
        'bounds': bounds,
        'root': root,
        'worldbody': worldbody,
        'actuator_section': actuator_section,
        'robot_body': robot_body,
        'robot_geom': robot_geom,
        'robot_size': robot_size,
        'goal_site': goal_site,
    }


def add_obstacles_to_xml(
    worldbody: ET.Element,
    obstacles: List[Dict[str, Any]],
) -> None:
    """Add obstacle bodies to the XML worldbody.

    Args:
        worldbody: The worldbody XML element
        obstacles: List of obstacle dicts with 'pos', 'size', and 'rotation'
    """
    for i, obs in enumerate(obstacles):
        # Create body with free joint (movable)
        body = ET.SubElement(worldbody, 'body', name=f'obstacle_{i}_movable')

        # Add geom (rectangular box with rotation)
        ET.SubElement(
            body, 'geom',
            name=f'obstacle_{i}_movable',
            condim='4',
            pos=f"{obs['pos'][0]} {obs['pos'][1]} 0.2",
            euler=f"0 0 {obs['rotation']}",
            friction='1.0 0.005 0.001',
            rgba='1 1 0 1',
            size=f"{obs['size'][0]} {obs['size'][1]} 0.2",
            type='box',
            mass='0.2',
        )

        # Add free joint
        ET.SubElement(body, 'joint', type='free')


def add_robot_and_goal_to_xml(
    worldbody: ET.Element,
    robot_pos: Tuple[float, float],
    goal_pos: Tuple[float, float],
    robot_size: float = 0.15,
) -> None:
    """Add robot and goal to XML worldbody with new positions.
    
    Args:
        worldbody: The worldbody XML element
        robot_pos: (x, y) position for robot
        goal_pos: (x, y) position for goal
        robot_size: Radius of the robot sphere
    """
    # Add robot body
    robot_body = ET.SubElement(worldbody, 'body', name='robot')
    ET.SubElement(robot_body, 'joint', name='joint_x', type='slide', pos='0 0 0', axis='1 0 0')
    ET.SubElement(robot_body, 'joint', name='joint_y', type='slide', pos='0 0 0', axis='0 1 0')
    ET.SubElement(
        robot_body, 'geom',
        name='robot',
        type='sphere',
        pos=f'{robot_pos[0]} {robot_pos[1]} 0.2',
        size=str(robot_size),
        mass='5.0',
        friction='1.0 0.005 0.001',
        condim='4'
    )
    
    # Add goal site
    ET.SubElement(
        worldbody, 'site',
        name='goal',
        type='sphere',
        pos=f'{goal_pos[0]} {goal_pos[1]} 0.0',
        size='0.2',
        rgba='1 0 0 0.5'
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
) -> List[Dict[str, Any]]:
    """Randomly place obstacles in the environment with random rotations, avoiding walls."""
    obstacles = []

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

    def try_sample_order(src_region: str, tgt_region: str) -> bool:
        """Try up to max_goal_retries to sample (robot in src_region, goal in tgt_region).
        On success appends to placements and returns True."""
        for attempt in range(max_goal_retries):
            # sample robot with clearance
            robot_pos = exporter.sample_cell_with_clearance(src_region, region_map, region_labels, dynamic_grid, clearance_radius, rng)
            if robot_pos is None:
                if debug:
                    print(f"  [DEBUG] Failed to sample robot in region {src_region} (attempt {attempt})")
                continue

            # sample goal with clearance
            goal_pos = exporter.sample_cell_with_clearance(tgt_region, region_map, region_labels, dynamic_grid, clearance_radius, rng)
            if goal_pos is None:
                if debug:
                    print(f"  [DEBUG] Failed to sample goal in region {tgt_region} (attempt {attempt})")
                continue

            # clearance check for goal
            gx = exporter._world_to_grid_x(goal_pos[0])
            gy = exporter._world_to_grid_y(goal_pos[1])
            if not exporter.check_cell_clearance((gx, gy), clearance_radius, dynamic_grid):
                if debug:
                    print(f"  [DEBUG] Goal clearance failed at attempt {attempt} for regions {src_region}->{tgt_region}")
                continue

            # distance check
            distance = np.hypot(goal_pos[0] - robot_pos[0], goal_pos[1] - robot_pos[1])
            if distance < min_goal_distance:
                if debug:
                    print(f"  [DEBUG] Distance {distance:.2f} < min {min_goal_distance:.2f} for regions {src_region}->{tgt_region}")
                continue

            # passed all checks
            placements.append((robot_pos, goal_pos))
            if debug:
                print(f"  [DEBUG] Added placement for regions {src_region}->{tgt_region} (distance={distance:.2f})")
            return True

        # failed all attempts
        return False

    for comp_idx, component in enumerate(valid_components):
        regions = list(component)
        # iterate over unordered pairs (i<j) -> C(n,2)
        for ri, rj in itertools.combinations(regions, 2):
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
    num_objects = config.get('num_objects', DEFAULT_NUM_OBJECTS)
    object_size_range = tuple(config.get('object_size_range', DEFAULT_OBJECT_SIZE_RANGE))
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
    except Exception as e:
        print(f"[Env {env_id}] Error parsing template XML: {e}")
        return False
    
    # Place obstacles
    obstacles = place_obstacles(walls, bounds, num_objects, object_size_range, rng)
    print(f"[Env {env_id}] Placed {len(obstacles)} obstacles")
    
    # Add obstacles to XML
    add_obstacles_to_xml(worldbody, obstacles)

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
        
        exporter = WavefrontSnapshotExporter(env, resolution=resolution)
        
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
    
    # Update robot position in the XML
    robot_body_final = worldbody_final.find(".//body[@name='robot']")
    if robot_body_final is not None:
        robot_geom_final = robot_body_final.find(".//geom[@name='robot']")
        if robot_geom_final is not None:
            robot_geom_final.set('pos', f'{robot_pos[0]} {robot_pos[1]} 0.2')
    
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
    num_objects = config.get('num_objects', DEFAULT_NUM_OBJECTS)
    object_size_range = tuple(config.get('object_size_range', DEFAULT_OBJECT_SIZE_RANGE))
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
    except Exception as e:
        print(f"[Env {env_id}] Error parsing template XML: {e}")
        return 0

    # Place obstacles
    obstacles = place_obstacles(walls, bounds, num_objects, object_size_range, rng)
    print(f"[Env {env_id}] Placed {len(obstacles)} obstacles")

    # Add obstacles to XML
    add_obstacles_to_xml(worldbody, obstacles)

    # Save intermediate XML with obstacles but no robot/goal (for simulation).
    # Use the run-specific output directory so concurrent templates cannot overwrite
    # each other's temporary files before final XMLs are written.
    os.makedirs(output_dir, exist_ok=True)
    temp_xml_path = os.path.join(output_dir, f'env_{env_id:04d}_temp.xml')
    try:
        ET.indent(tree, space='  ')
        tree.write(temp_xml_path, encoding='utf-8', xml_declaration=True)
        print(f"[Env {env_id}] Temporary XML saved to: {temp_xml_path}")
    except Exception as e:
        print(f"[Env {env_id}] Error saving temporary XML: {e}")
        return 0

    # Create simulation environment from the XML with obstacles and build snapshot
    try:
        print(f"[Env {env_id}] Initializing simulation environment...")
        rl_env_cls = getattr(namo_rl, "RLEnvironment")
        env = rl_env_cls(temp_xml_path, namo_config_path, visualize=False)

        exporter = WavefrontSnapshotExporter(env, resolution=resolution)

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
        return 0

    # Generate all (robot, goal) pairs
    placements = place_robot_and_goal_pairs(
        exporter,
        region_map,
        region_labels,
        adjacency,
        dynamic_grid,
        clearance_radius,
        min_goal_distance,
        max_goal_retries,
        rng,
        debug=True,
    )

    if not placements:
        print(f"[Env {env_id}] No valid (robot,goal) placements found")
        return 0

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    written = 0
    for k, (robot_pos, goal_pos) in enumerate(placements):
        try:
            # Re-parse the temp XML for each pair to keep obstacles identical
            tree_final = ET.parse(temp_xml_path)
            root_final = tree_final.getroot()
            worldbody_final = root_final.find('worldbody')
            if worldbody_final is None:
                print(f"[Env {env_id}] Error: No worldbody in final XML for pair {k}")
                continue

            robot_body_final = worldbody_final.find(".//body[@name='robot']")
            if robot_body_final is not None:
                robot_geom_final = robot_body_final.find(".//geom[@name='robot']")
                if robot_geom_final is not None:
                    robot_geom_final.set('pos', f'{robot_pos[0]} {robot_pos[1]} 0.2')

            goal_site_final = worldbody_final.find(".//site[@name='goal']")
            if goal_site_final is not None:
                goal_site_final.set('pos', f'{goal_pos[0]} {goal_pos[1]} 0.0')
            else:
                # if robot was present but goal missing, add site
                ET.SubElement(
                    worldbody_final, 'site',
                    name='goal',
                    type='sphere',
                    pos=f'{goal_pos[0]} {goal_pos[1]} 0.0',
                    size='0.2',
                    rgba='1 0 0 0.5'
                )

            # Save final XML for this pair
            out_path = os.path.join(output_dir, f'env_{env_id:04d}_pair_{k:03d}.xml')
            ET.indent(tree_final, space='  ')
            tree_final.write(out_path, encoding='utf-8', xml_declaration=True)
            written += 1
        except Exception as e:
            print(f"[Env {env_id}] Error saving XML for pair {k}: {e}")
            continue

    print(f"[Env {env_id}] Generated {written} environment(s) from {len(placements)} placement(s)")
    if os.path.exists(temp_xml_path):
        os.remove(temp_xml_path)
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

    Creates a two-level output layout:
      output_dir/
        <template_base>/               <- directory named after template (no .xml)
          run_0000/                    <- subdirectory for first job (passed to worker)
          run_0001/                    <- subdirectory for second job
          ...
    Each worker calls generate_environments_from_pairs and writes one or more XMLs
    into its assigned run_<idx> subdirectory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Top-level subdir named after template base (filename without extension)
    template_base = os.path.splitext(os.path.basename(template_xml_path))[0]
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
    parser.add_argument('--num-objects', type=int, default=DEFAULT_NUM_OBJECTS)
    parser.add_argument('--clearance-radius', type=float, default=DEFAULT_CLEARANCE_RADIUS)
    parser.add_argument('--min-goal-distance', type=float, default=DEFAULT_MIN_GOAL_DISTANCE)
    parser.add_argument('--max-goal-retries', type=int, default=DEFAULT_MAX_GOAL_RETRIES, 
                        help='Maximum retries for goal placement with clearance')
    
    args = parser.parse_args()
    
    # Check that namo_rl is available
    if not NAMO_RL_AVAILABLE:
        print("Error: namo_rl not available. Please ensure PYTHONPATH includes the namo_cpp build directory.")
        sys.exit(1)
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override with command line args
    config['num_objects'] = args.num_objects
    config['clearance_radius'] = args.clearance_radius
    config['min_goal_distance'] = args.min_goal_distance
    config['max_goal_retries'] = args.max_goal_retries
    
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
