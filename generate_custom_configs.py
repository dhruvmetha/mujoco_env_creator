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
import mujoco
from placement_strategies import (
    check_collision,
    MultiMovableArcPlacement,
    QuadrantMixedPlacement,
    SinglePairArcPlacement
)

# np.random.seed(2)

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

def create_obstacle(pos, size, rotation, is_movable, index, gen_config):
    """Create an obstacle with given properties."""
    obstacle = {
        'name': f'obstacle_{index}_{"movable" if is_movable else "static"}',
        'type': 'box',
        'pos': pos,
        'size': size,
        'rgba': gen_config['obstacles']['color']['movable' if is_movable else 'static'],
        'rotation': rotation,
        'movable': is_movable,
        'friction': gen_config['obstacles']['friction'],
        'condim': 4
    }
    
    if is_movable:
        obstacle['mass'] = gen_config['obstacles']['movable_mass']
        obstacle['edge_points'] = generate_edge_points(pos, size, rotation)
    
    return obstacle

def generate_random_size(gen_config):
    """Generate random size within configured bounds."""
    return [
        np.random.uniform(gen_config['obstacles']['size']['width']['min'], 
                         gen_config['obstacles']['size']['width']['max']),
        np.random.uniform(gen_config['obstacles']['size']['depth']['min'], 
                         gen_config['obstacles']['size']['depth']['max']),
        0.3  # Fixed height
    ]

def is_valid_placement(obstacle, existing_objects):
    """Check if obstacle placement is valid."""
    return all(not check_collision(obstacle, obj) for obj in existing_objects)

def generate_env_config(env_size, robot_pos, gen_config, walls, robot, existing_obstacles):
    """Generate environment configuration with obstacles."""
    # Convert numpy types in robot_pos
    robot_pos = [float(p) for p in robot_pos]
    env_size = [float(s) for s in env_size]
    
    # Get selected strategy and generate obstacles
    strategy_name = gen_config['obstacles']['placement_strategy']
    strategies = {
        "single_pair_arc": SinglePairArcPlacement,
        # "multi_movable_arc": MultiMovableArcPlacement,
        # "quadrant_mixed": QuadrantMixedPlacement
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Invalid placement strategy: {strategy_name}. "
                       f"Must be one of {list(strategies.keys())}")
    
    strategy = strategies[strategy_name](env_size, robot['pos'], gen_config)
    for _ in range(10):
        obstacles = strategy.place_obstacles(existing_obstacles + [robot] + walls)
        if len(obstacles) > 0:
            break
    if len(obstacles) == 0:
        return None
    
    # Create the properly structured config
    config = {
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
                        'name': 'skybox',
                        'type': 'skybox',
                        'builtin': 'gradient',
                        'rgb1': gen_config['skybox']['rgb1'],
                        'rgb2': gen_config['skybox']['rgb2'],
                        'width': gen_config['skybox']['width'],
                        'height': gen_config['skybox']['height']
                    },
                    {
                        'name': 'groundplane',
                        'type': '2d',
                        'builtin': 'checker',
                        'rgb1': gen_config['groundplane']['rgb1'],
                        'rgb2': gen_config['groundplane']['rgb2'],
                        'width': gen_config['groundplane']['width'],
                        'height': gen_config['groundplane']['height']
                    }
                ],
                'material': [
                    {
                        'name': 'groundplane',
                        'texture': 'groundplane',
                        'texrepeat': gen_config['groundplane']['texrepeat'],
                        'reflectance': gen_config['groundplane']['reflectance']
                    }
                ]
            },
            'geom': {
                'name': 'floor',
                'type': 'plane',
                'pos': [0, 0, 0],
                'size': gen_config['floor']['size'],
                'material': 'groundplane',
                'friction': gen_config['floor']['friction'],
                'condim': 3
            },
            'walls': walls,
            'robot': robot,
            'obstacles': obstacles
        }
    }
    
    return config

def convert_numpy_types(obj):
    """Convert numpy types to native Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):  # Use np.integer instead of specific types
        return int(obj)
    elif isinstance(obj, np.floating):  # Use np.floating instead of specific types
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def save_config(config, filename):
    """Save config to YAML file with numpy type conversion."""
    # Convert numpy types to native Python types
    config = convert_numpy_types(config)
    
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='executables/mujoco_env_creator/generator_config.yaml')
    parser.add_argument('--model_path', type=str, default='resources/models/simple_envs/cylinder_empty.xml')
    parser.add_argument('--random_start', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    generator_config = load_generator_config(args.config)
    model_path = args.model_path
    num_configs = generator_config['num_configs']
    config_dir = generator_config.get('config_dir', 'env_configs')
    os.makedirs(config_dir, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(model_path)
    walls = []
    robot = None

    # Load walls and robot from model
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

        if model.geom(i).name == 'robot':
            robot = {
                'name': model.geom(i).name,
                'type': ['plane', 'hfield', 'sphere', 'capsule', 'ellipsoid', 'cylinder', 'box'][model.geom(i).type.item()],
                'pos': model.geom(i).pos.tolist(),
                'size': model.geom(i).size.tolist(),
                'rgba': model.geom(i).rgba.tolist(),
                'condim': model.geom(i).condim.item(),
                'friction': generator_config['robot']['friction'],
                'mass': generator_config['robot']['mass']
            }

    # Generate configurations
    for i in tqdm(range(num_configs), desc="Generating environment configs"):
        env_size = [4, 4, 2]  # Fixed size as per your comment
        robot_size = generator_config['robot']['size']
        min_distance = max(robot_size[0], robot_size[1]) / 2 + 0.1
        
        robot_pos = [
            np.random.uniform(-env_size[0]/2 + min_distance, env_size[0]/2 - min_distance),
            np.random.uniform(-env_size[1]/2 + min_distance, env_size[1]/2 - min_distance),
            robot_size[2]
        ]
        
        config = generate_env_config(env_size, robot_pos, generator_config, walls, robot, [])
        if config is not None:
            save_config(config, os.path.join(config_dir, f'env_config_{i+1}.yaml'))

    print(f"{num_configs} environment configurations generated and saved in the '{config_dir}' directory")