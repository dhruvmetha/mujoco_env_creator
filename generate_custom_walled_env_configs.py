# Generate config files for procedurally generating environments.
# These configs are used to generate MuJoCo XML files.
# The floor is infinite in extent.

# An environment is defined as the following:
# - A bounded 3D space, with env_size as a parameter.
# - Walls bounding the space. These walls are rigid bodies and are gray in color.
# - A box-shaped robot placed at the origin (0, 0, 0) within the environment bounds.
# - The walls surround the robot, with the robot at the center of the environment.

# MAKE SURE NO RIGID BODY IS COLLIDING WITH EACH OTHER (use shapely if needed).

import yaml
import numpy as np
from shapely.geometry import Polygon
import os
from tqdm import tqdm
import argparse
from scipy.spatial.transform import Rotation as R
import random
from collections import deque
from shapely.geometry import Polygon, Point
from shapely.affinity import scale
import multiprocessing as mp
from functools import partial

from matplotlib import pyplot as plt


def get_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def load_generator_config(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

# Add this function to convert NumPy types to Python types
def convert_to_python_type(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_to_python_type(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_python_type(value) for key, value in obj.items()}
    else:
        return obj

def generate_edge_points(pos, size, rotation):
    """Generate 3 evenly spaced points on each edge of the box (top-down view)."""
    points = []
    x, y = pos[0], pos[1]
    w, d = size[0], size[1]
    angle = np.radians(rotation)
    offset = 0.1
    
    # Helper function to rotate points
    def rotate_point(px, py, center_x, center_y):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = px - center_x, py - center_y
        rx = center_x + dx * cos_a - dy * sin_a
        ry = center_y + dx * sin_a + dy * cos_a
        return [rx, ry]

    # Generate 3 points on each edge
    edges = [
        # Top edge
        [(x + w * (i - 1), y + d + offset) for i in range(3)],
        # Bottom edge
        [(x + w * (i - 1), y - d - offset) for i in range(3)],
        # Right edge
        [(x + w + offset, y + d * (i - 1)) for i in range(3)],
        # Left edge
        [(x - w - offset, y + d * (i - 1)) for i in range(3)]
    ]

    # Rotate points if needed
    for edge in edges:
        for point in edge:
            if rotation != 0:
                rotated = rotate_point(point[0], point[1], x, y)
                points.append(rotated)
            else:
                points.append([point[0], point[1]])

    return points

def generate_robot_config(robot_config, env_size):
    if robot_config['placement_strategy'] == 'fixed':
        pos = robot_config['pos']
        
    elif robot_config['placement_strategy'] == 'random':
        pos = [np.random.uniform(-env_size[0]/2 + robot_config['size'][0], env_size[0]/2 - robot_config['size'][0]),
               np.random.uniform(-env_size[1]/2 + robot_config['size'][1], env_size[1]/2 - robot_config['size'][1]),
               robot_config['size'][2]]
    
    robot = {
        'name': 'robot',
        'type': robot_config['type'],
        'pos': pos,
        'size': robot_config['size'],
        'material': robot_config['color'],
        'friction': robot_config['friction'],
        'mass': robot_config['mass']
    }
    return robot

def generate_env_config(env_size, gen_config, walls, pre_existing_obstacles, obstacles):
    is_one_env = gen_config['one_env']
    config = {
        'env_size': env_size,
        'worldbody': {
            'light': {
                'pos': gen_config['light']['pos'],
                'dir': gen_config['light']['dir'],
                'directional': 'true'
            },
            'geom': {
                'name': 'floor',
                'type': 'plane',
                'size': gen_config['floor']['size'],
                'material': 'groundplane',
                'friction': gen_config['floor']['friction'],
                'condim': 4
            },
            'asset': {
                'texture': [
                    {
                        'type': 'skybox',
                        'builtin': 'gradient',
                        'rgb1': gen_config['skybox']['rgb1'],
                        'rgb2': gen_config['skybox']['rgb2'],
                        'width': gen_config['skybox']['width'],
                        'height': gen_config['skybox']['height']
                    },
                    {
                        'type': '2d',
                        'name': 'groundplane',
                        'builtin': 'checker',
                        'mark': 'edge',
                        'rgb1': gen_config['groundplane']['rgb1'],
                        'rgb2': gen_config['groundplane']['rgb2'],
                        'markrgb': gen_config['groundplane']['markrgb'],
                        'width': gen_config['groundplane']['width'],
                        'height': gen_config['groundplane']['height']
                    }
                ],
                'material': [
                    {
                        'name': 'groundplane',
                        'texture': 'groundplane',
                        'texuniform': 'true',
                        'texrepeat': gen_config['groundplane']['texrepeat'],
                        'reflectance': gen_config['groundplane']['reflectance']
                    },
                    {
                        'name': 'robot',
                        'rgba': gen_config['robot']['color']
                    }
                ]
            },
            'walls': walls,
            'obstacles': [],
            'goal': {
                'name': 'goal',
                'type': 'sphere',
                'pos': gen_config['goal']['pos'],
                'size': gen_config['goal']['size'],
                'rgba': gen_config['goal']['color']
            }
        }
    }
    
    if pre_existing_obstacles is not None:
        config['worldbody']['obstacles'] = pre_existing_obstacles
    
    if not is_one_env:
        robot = generate_robot_config(gen_config['robot'], env_size)
        while True:
            if all(not check_collision(robot, obj) for obj in config['worldbody']['walls'] + config['worldbody']['obstacles']):
                config['worldbody']['robot'] = robot
                break
            else:
                if gen_config['robot']['placement_strategy'] == 'random':
                    robot = generate_robot_config(gen_config['robot'], env_size)
                else:
                    raise ValueError("Fixed robot pos is causing robot collision with walls or obstacles")
    
    if pre_existing_obstacles is None:
        if obstacles is None:
            num_obstacles = np.random.randint(gen_config['obstacles']['count']['min'], gen_config['obstacles']['count']['max'] + 1)
            max_attempts = 50
            movable_done = False
            existing_obstacles_count = len(config['worldbody']['obstacles'])

            for i in range(num_obstacles):
                for attempt in range(max_attempts):
                    # Generate a random size for the obstacle within the specified range
                    size = [
                        np.random.uniform(gen_config['obstacles']['size']['width']['min'], gen_config['obstacles']['size']['width']['max']), # width
                        np.random.uniform(gen_config['obstacles']['size']['depth']['min'], gen_config['obstacles']['size']['depth']['max']), # depth
                        np.random.uniform(0.3, 0.3)  # Height is fixed at 0.6 for simplicity
                    ]
                    # Generate a random position for the obstacle within the environment bounds
                    pos = [np.random.uniform(-env_size[0]/2 + size[0], env_size[0]/2 - size[0]),
                        np.random.uniform(-env_size[1]/2 + size[1], env_size[1]/2 - size[1]),
                        size[2]]  # Adjust z-position to half the height
                    # Generate a random rotation for the obstacle
                    rotation = np.random.uniform(0, 360)
                    # Randomly decide if the obstacle is movable or not

                    is_movable = np.random.choice([True, False], p=[gen_config['obstacles']['movable_probability'], 1 - gen_config['obstacles']['movable_probability']])


                    obstacle = {
                        'name': f'obstacle_{existing_obstacles_count+i+1}{"_movable" if is_movable else "_static"}',
                        'type': 'box',
                        'pos': pos,
                        'size': size,
                        'rgba': gen_config['obstacles']['color']['movable'] if is_movable else gen_config['obstacles']['color']['static'],
                        'rotation': rotation,
                        'movable': is_movable,
                        'friction': gen_config['obstacles']['friction'],
                        'condim': 4
                    }
                    
                    if is_movable:
                        obstacle['mass'] = gen_config['obstacles']['movable_mass']
                        # Add edge points for movable obstacles
                        obstacle['edge_points'] = generate_edge_points(pos, size, rotation)
                    
                    # Check if the obstacle collides with any existing objects in the environment
                    all_objects = config['worldbody']['walls'] + config['worldbody']['obstacles'] + [config['worldbody']['goal']] + [config['worldbody']['robot']]
                    
                    if all(not check_collision(obstacle, obj) for obj in all_objects):
                        # If no collision is detected, add the obstacle to the environment
                        # if not check_collision(obstacle, config['worldbody']['robot']):
                        config['worldbody']['obstacles'].append(obstacle)
                        movable_done = is_movable
                        break
        else:
            # print('using last generated obstacles')
            config['worldbody']['obstacles'] = obstacles
        
    if is_one_env:
        # Generate a random robot position that doesn't collide with any walls or obstacles
        robot = generate_robot_config(gen_config['robot'], env_size)
        while True:
            if all(not check_collision(robot, obj) for obj in config['worldbody']['walls'] + config['worldbody']['obstacles']):
                config['worldbody']['robot'] = robot
                break
            else:
                if gen_config['robot']['placement_strategy'] == 'random':
                    robot = generate_robot_config(gen_config['robot'], env_size)
                else:
                    raise ValueError("Fixed robot pos is causing robot collision with walls or obstacles and one_env is true")
                
    if gen_config['goal']['placement_strategy'] == 'fixed':
        goal_pos = gen_config['goal']['pos']
    else:
        for max_attempts in range(50):
            goal_pos = [np.random.uniform(-env_size[0]/2 + gen_config['goal']['size'][0], env_size[0]/2 - gen_config['goal']['size'][0]),
                        np.random.uniform(-env_size[1]/2 + gen_config['goal']['size'][1], env_size[1]/2 - gen_config['goal']['size'][1]),
                        gen_config['goal']['pos'][2]]

            robot_pos = config['worldbody']['robot']['pos']

            config['worldbody']['goal'] = {
                'name': 'goal',
                'type': 'sphere',
                'pos': goal_pos,
                'size': gen_config['goal']['size'],
                'rgba': gen_config['goal']['color']
            }
            if get_distance(goal_pos, robot_pos) > (env_size[0] / 2) and all(not check_collision(config['worldbody']['goal'], obj) for obj in config['worldbody']['walls'] + config['worldbody']['obstacles']):
                break
    
    distance = get_distance(config['worldbody']['robot']['pos'], config['worldbody']['goal']['pos']) > 0.5
    # print(f"Distance between robot and goal: {distance}")
    
    if is_one_env:
        return convert_to_python_type(config), config['worldbody']['obstacles'].copy(), distance
    else:
        return convert_to_python_type(config), None, distance

def check_collision(obj1, obj2):
    def get_rotated_corners(obj):
        # Extract position and size
        if obj['name'] == 'robot' or obj['name'] == 'goal':
            x, y = obj['pos'][0], obj['pos'][1]
            w, h = obj['size'][0], obj['size'][0]
            return Polygon([(x - w, y - h), (x + w, y - h), (x + w, y + h), (x - w, y + h)])
    
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

def save_config(config, filename):
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)



# Add these new functions after the existing utility functions

def world_to_grid(world_pos, env_size, grid_size):
    """Convert world coordinates to grid coordinates."""
    x, y = world_pos[0], world_pos[1]
    grid_x = int((x + env_size[0]/2) / env_size[0] * grid_size[0])
    grid_y = int((y + env_size[1]/2) / env_size[1] * grid_size[1])
    
    # Clamp to grid bounds
    grid_x = max(0, min(grid_size[0] - 1, grid_x))
    grid_y = max(0, min(grid_size[1] - 1, grid_y))
    
    # Return (row, col) for NumPy indexing instead of (x, y)
    return (grid_y, grid_x)  # FIXED: Return (row, col) not (x, y)

def grid_to_world(grid_pos, env_size, grid_size):
    """Convert grid coordinates to world coordinates."""
    grid_r, grid_c = grid_pos  # FIXED: Now expecting (row, col)
    x = (grid_c / grid_size[0]) * env_size[0] - env_size[0]/2
    y = (grid_r / grid_size[1]) * env_size[1] - env_size[1]/2
    return (x, y)

def get_inflated_obstacle_polygon(obstacle, robot_radius):
    """Get the polygon representation of an obstacle inflated by robot radius."""
    x, y = obstacle['pos'][0], obstacle['pos'][1]
    w, h = obstacle['size'][0], obstacle['size'][1]
    
    # Get rotation angle
    angle = np.radians(obstacle.get('rotation', 0))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    # Create original corners
    corners = [
        (x + w * cos_a - h * sin_a, y + w * sin_a + h * cos_a),  # Top-right
        (x - w * cos_a - h * sin_a, y - w * sin_a + h * cos_a),  # Top-left
        (x - w * cos_a + h * sin_a, y - w * sin_a - h * cos_a),  # Bottom-left
        (x + w * cos_a + h * sin_a, y + w * sin_a - h * cos_a)   # Bottom-right
    ]
    
    # Create polygon and inflate it
    poly = Polygon(corners)
    inflated_poly = poly.buffer(robot_radius, join_style=2)  # join_style=2 for mitered joins
    return inflated_poly

def create_occupancy_grid(env_size, robot_radius, walls, obstacles, resolution=0.05):
    """Create an occupancy grid with inflated obstacles.

    Optimisation: instead of checking every grid cell against every obstacle,
    we first restrict the search area to the grid‐cells whose centres fall
    within the *axis-aligned bounding box* of the inflated obstacle.  This
    reduces the per-obstacle checks from (H×W) to roughly the number of cells
    covered by the obstacle footprint, which yields a large speed-up when the
    map is spacious.
    """

    # Calculate grid dimensions
    grid_width = int(env_size[0] / resolution)
    grid_height = int(env_size[1] / resolution)
    grid_size = (grid_width, grid_height)

    # Initialise grid (False = free, True = occupied)
    grid = np.zeros((grid_height, grid_width), dtype=bool)

    all_obstacles = walls + obstacles  # iterate once over all geometry

    cell_size = resolution / 2  # half-cell used repeatedly below

    for obstacle in all_obstacles:
        inflated_poly = get_inflated_obstacle_polygon(obstacle, robot_radius)

        # Axis-aligned bounding box in world coordinates
        minx, miny, maxx, maxy = inflated_poly.bounds

        # Convert bbox corners → grid indices (row, col)
        r_min, c_min = world_to_grid((minx, miny), env_size, grid_size)
        r_max, c_max = world_to_grid((maxx, maxy), env_size, grid_size)

        # Ensure increasing order and clamp to grid extents
        r0, r1 = sorted((r_min, r_max))
        c0, c1 = sorted((c_min, c_max))
        r0, r1 = max(0, r0), min(grid_height - 1, r1)
        c0, c1 = max(0, c0), min(grid_width - 1, c1)

        # Iterate only over the cells that lie inside the bounding box
        for i in range(r0, r1 + 1):  # rows (Y)
            for j in range(c0, c1 + 1):  # cols (X)
                world_pos = grid_to_world((i, j), env_size, grid_size)

                # Build a small square around the cell centre
                cell_square = Polygon([
                    (world_pos[0] - cell_size, world_pos[1] - cell_size),
                    (world_pos[0] + cell_size, world_pos[1] - cell_size),
                    (world_pos[0] + cell_size, world_pos[1] + cell_size),
                    (world_pos[0] - cell_size, world_pos[1] + cell_size)
                ])

                # Mark cell occupied if it intersects the inflated obstacle
                if inflated_poly.intersects(cell_square):
                    grid[i, j] = True

    return grid, grid_size

def bfs_connectivity_check(grid, start_pos, goal_pos):
    """Check if start and goal positions are connected using BFS."""
    rows, cols = grid.shape
    start_r, start_c = start_pos
    goal_r, goal_c = goal_pos
    
    # Check if start or goal are in occupied cells
    
    # print(start_r, start_c, goal_r, goal_c)
    if grid[start_r, start_c] or grid[goal_r, goal_c]:
        return False
    
    # BFS setup
    visited = np.zeros_like(grid, dtype=bool)
    queue = deque([(start_r, start_c)])
    visited[start_r, start_c] = True
    
    # 8-connectivity (including diagonals)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while queue:
        r, c = queue.popleft()
        
        # Check if we reached the goal
        if (r, c) == (goal_r, goal_c):
            return True
        
        # Explore neighbors
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check bounds
            if 0 <= nr < rows and 0 <= nc < cols:
                # Check if not visited and not occupied
                if not visited[nr, nc] and not grid[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
    
    return False

def check_robot_goal_connectivity(env_size, robot, goal, walls, obstacles, resolution=0.05, idx=0):
    robot_radius = robot['size'][0]
    
    # print(robot)
    # print(robot_radius, env_size, resolution)
    # Create occupancy grid
    grid, grid_size = create_occupancy_grid(env_size, robot_radius, walls, obstacles, resolution)
    
    
    # print(grid.shape)
    
    # plt.figure(figsize=(10, 10))
    # plt.imshow(grid)
    # plt.savefig(f'grid_{idx}.png')
    # plt.colorbar()
    
    
    # Convert robot and goal positions to grid coordinates
    robot_grid_pos = world_to_grid(robot['pos'], env_size, grid_size)
    goal_grid_pos = world_to_grid(goal['pos'], env_size, grid_size)
    
    # Check connectivity
    return bfs_connectivity_check(grid, robot_grid_pos, goal_grid_pos)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='executables/mujoco_env_creator/generator_config.yaml')
    parser.add_argument('--model_path', type=str, default='resources/models/custom/benchmark_room_2.xml')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of parallel processes (default: CPU count)')
    args = parser.parse_args()
    return args

# Add new function for parallel processing
def generate_single_env_config(env_index, shared_data):
    """
    Generate a single environment configuration.
    This function will be called in parallel for each environment.
    """
    env_size, generator_config, walls, pre_existing_obstacles, config_dir, wavefront_resolution, enable_connectivity_check = shared_data
    
    # Set random seed for reproducibility
    np.random.seed(env_index + 1)
    
    new_obstacles = None
    
    robot_size = generator_config['robot']['size']
    min_distance = max(robot_size[0], robot_size[1]) / 2 + 0.1
    
    robot_pos = [
        np.random.uniform(-env_size[0]/2 + min_distance, env_size[0]/2 - min_distance),
        np.random.uniform(-env_size[1]/2 + min_distance, env_size[1]/2 - min_distance),
        robot_size[2]
    ]
    
    # Handle pre-existing obstacles
    processed_pre_existing_obstacles = pre_existing_obstacles
    if processed_pre_existing_obstacles is not None and len(processed_pre_existing_obstacles) == 0:
        processed_pre_existing_obstacles = None
    
    config, new_obstacles, distance_bool = generate_env_config(
        env_size, generator_config, walls, processed_pre_existing_obstacles, new_obstacles
    )
    
    if not distance_bool:
        return env_index, False, "Distance constraint not satisfied"
    
    # Check robot-goal connectivity if enabled
    is_robot_goal_connected = True
    if enable_connectivity_check:
        robot_obj = config['worldbody']['robot']
        goal_obj = config['worldbody']['goal']
        all_obstacles = config['worldbody']['obstacles']
        walls_obj = config['worldbody']['walls']
        is_robot_goal_connected = check_robot_goal_connectivity(
            env_size, robot_obj, goal_obj, walls_obj, all_obstacles, wavefront_resolution, env_index
        )
    
    if is_robot_goal_connected:
        return env_index, False, "Robot-goal connectivity check failed"
    
    # Save the configuration
    output_path = os.path.join(config_dir, f'env_config_{env_index+1}.yaml')
    save_config(config, output_path)
    
    return env_index, True, "Success"

if __name__ == "__main__":
    args = parse_args()
    generator_config = load_generator_config(args.config)
    model_path = args.model_path
    num_configs = generator_config['num_configs']
    config_dir = generator_config.get('config_dir', 'env_configs')
    os.makedirs(config_dir, exist_ok=True)
    
    # Determine number of processes
    num_processes = args.num_processes
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Add wavefront parameters to generator config if not present
    wavefront_resolution = generator_config.get('wavefront_resolution', 0.03)
    enable_connectivity_check = generator_config.get('enable_connectivity_check', True)

    import mujoco
    model = mujoco.MjModel.from_xml_path(model_path)
    walls = []
    robot = None
    obstacles = []
    pre_existing_obstacles = []
    x_limits = [-1, 1]
    y_limits = [-1, 1]
    for i in range(model.ngeom):
        if model.geom(i).name.startswith('wall_'):
            wall_props = {
                'name': model.geom(i).name,
                'type': 'box',
                'pos': model.geom(i).pos.tolist(),
                'size': model.geom(i).size.tolist(),
                'rgba': model.geom(i).rgba.tolist(),
                'condim': model.geom(i).condim.item()
            }

            walls.append(wall_props)
            x_limits.append(wall_props['pos'][0] - wall_props['size'][0] / 2)
            x_limits.append(wall_props['pos'][0] + wall_props['size'][0] / 2)
            y_limits.append(wall_props['pos'][1] - wall_props['size'][1] / 2)
            y_limits.append(wall_props['pos'][1] + wall_props['size'][1] / 2)

        if model.geom(i).name == 'robot':
            robot = {
                'name': model.geom(i).name,
                'type': 'sphere',
                'pos': model.geom(i).pos.tolist(),
                'size': model.geom(i).size.tolist(),
                'rgba': model.geom(i).rgba.tolist(),
                'condim': model.geom(i).condim.item(),
                'friction': model.geom(i).friction.tolist(),
                'mass': generator_config['robot']['mass']
            }

        if model.geom(i).name.startswith('obstacle_'):
            is_movable = any(joint_type in model.geom(i).name for joint_type in ['movable'])
            
            obstacle = {
                'name': model.geom(i).name,
                'type':['plane', 'hfield', 'sphere', 'capsule', 'ellipsoid', 'cylinder', 'box'][int(model.geom(i).type)],
                'pos': model.geom(i).pos.tolist(),
                'rotation': R.from_quat(model.geom(i).quat.tolist(), scalar_first=True).as_euler('xyz', degrees=True)[2],
                'size': model.geom(i).size.tolist(),
                'rgba': model.geom(i).rgba.tolist(),
                'condim': model.geom(i).condim.item(),
                'movable': is_movable,
                'friction': generator_config['obstacles']['friction'],
                'condim': 4
            }

            if is_movable:
                obstacle['mass'] = generator_config['obstacles']['movable_mass']
            
            pre_existing_obstacles.append(obstacle)
    
    env_size = [max(x_limits) - min(x_limits), max(y_limits) - min(y_limits), 0.3]
    
    # Prepare shared data for parallel processing
    shared_data = (
        env_size, 
        generator_config, 
        walls, 
        pre_existing_obstacles, 
        config_dir, 
        wavefront_resolution, 
        enable_connectivity_check
    )
    
    print(f"Generating {num_configs} environment configurations using {num_processes} processes...")
    
    # Use multiprocessing to generate environments in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Create partial function with shared data
        worker_func = partial(generate_single_env_config, shared_data=shared_data)
        
        # Use imap for progress tracking
        results = []
        successful_configs = 0
        
        with tqdm(total=num_configs, desc="Generating environment configs") as pbar:
            for result in pool.imap(worker_func, range(num_configs)):
                env_index, success, message = result
                results.append(result)
                if success:
                    successful_configs += 1
                pbar.update(1)
                pbar.set_postfix({
                    'successful': successful_configs,
                    'failed': len(results) - successful_configs
                })
    
    print(f"\nGeneration complete!")
    print(f"Successfully generated: {successful_configs}/{num_configs} configurations")
    print(f"Configurations saved in the '{config_dir}' directory")
    
    # Print failure summary if needed
    failed_results = [(idx, msg) for idx, success, msg in results if not success]
    if failed_results:
        print(f"\nFailed configurations:")
        failure_reasons = {}
        for idx, msg in failed_results:
            failure_reasons[msg] = failure_reasons.get(msg, 0) + 1
        
        for reason, count in failure_reasons.items():
            print(f"  {reason}: {count} configs")