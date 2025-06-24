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

np.random.seed(1)

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

def generate_env_config(env_size, robot_pos, gen_config):
    wall_thickness = gen_config['wall']['thickness']
    wall_height = 0.3 # env_size[2] / 2
    half_width = env_size[0] / 2
    half_depth = env_size[1] / 2

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
            'walls': [
                {
                    'name': 'wall_1',
                    'type': 'box',
                    'pos': [-half_width, 0, wall_height],
                    'size': [wall_thickness, half_depth, wall_height],
                    'rgba': gen_config['wall']['color'],
                    'condim': 4
                },
                {
                    'name': 'wall_2',
                    'type': 'box',
                    'pos': [half_width, 0, wall_height],
                    'size': [wall_thickness, half_depth, wall_height],
                    'rgba': gen_config['wall']['color'],
                    'condim': 4
                },
                {
                    'name': 'wall_3',
                    'type': 'box',
                    'pos': [0, -half_depth, wall_height],
                    'size': [half_width, wall_thickness, wall_height],
                    'rgba': gen_config['wall']['color'],
                    'condim': 4
                },
                {
                    'name': 'wall_4',
                    'type': 'box',
                    'pos': [0, half_depth, wall_height],
                    'size': [half_width, wall_thickness, wall_height],
                    'rgba': gen_config['wall']['color'],
                    'condim': 4
                }
            ],
            'robot': {
                'name': 'robot',
                'type': 'box',
                'pos': robot_pos,
                'size': gen_config['robot']['size'],
                'material': 'robot',
                'friction': gen_config['robot']['friction'],
                'mass': gen_config['robot']['mass']
            },
            'obstacles': []
        }
    }

    num_obstacles = np.random.randint(gen_config['obstacles']['count']['min'], gen_config['obstacles']['count']['max'] + 1)
    max_attempts = 100
    for i in range(num_obstacles):
        for attempt in range(max_attempts):
            # Generate a random size for the obstacle within the specified range
            size = [
                np.random.uniform(gen_config['obstacles']['size']['width']['min'], gen_config['obstacles']['size']['width']['max']), # width
                np.random.uniform(gen_config['obstacles']['size']['depth']['min'], gen_config['obstacles']['size']['depth']['max']), # depth
                np.random.uniform(0.6, 0.6)  # Height is fixed at 0.6 for simplicity
            ]
            # Generate a random position for the obstacle within the environment bounds
            pos = [np.random.uniform(-env_size[0]/2 + size[0], env_size[0]/2 - size[0]),
                   np.random.uniform(-env_size[1]/2 + size[1], env_size[1]/2 - size[1]),
                   size[2]]  # Adjust z-position to half the height
            # Generate a random rotation for the obstacle
            rotation = np.random.uniform(0, 360)
            # Randomly decide if the obstacle is movable or not
            is_movable = np.random.choice([True, False], p=[gen_config['obstacles']['movable_probability'], 1 - gen_config['obstacles']['movable_probability']])
            # Create the obstacle configuration

            obstacle = {
                'name': f'obstacle_{i+1}{"_movable" if is_movable else "_static"}',
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
            if all(not check_collision(obstacle, obj) for obj in config['worldbody']['walls'] + config['worldbody']['obstacles'] + [config['worldbody']['robot']]):
                # If no collision is detected, add the obstacle to the environment
                config['worldbody']['obstacles'].append(obstacle)
                break  # Exit the loop for this obstacle
        else:
            # If the loop completes without finding a valid placement, print a message and exit
            print(f"Could not find a valid placement for obstacle {i+1} after {max_attempts} attempts. Proceeding with {len(config['worldbody']['obstacles'])} obstacles.")
            break  # Exit the loop for all obstacles

    # Convert all NumPy types to Python types before returning
    return convert_to_python_type(config)

def check_collision(obj1, obj2):
    def get_rotated_corners(obj):
        # Extract position and size
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='executables/mujoco_env_creator/generator_config.yaml')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    generator_config = load_generator_config(args.config)
    num_configs = generator_config['num_configs']
    config_dir = generator_config.get('config_dir', 'env_configs')
    os.makedirs(config_dir, exist_ok=True)

    for i in tqdm(range(num_configs), desc="Generating environment configs"):
        env_size = [
            np.random.uniform(generator_config['env_size']['width']['min'], generator_config['env_size']['width']['max']),
            np.random.uniform(generator_config['env_size']['depth']['min'], generator_config['env_size']['depth']['max']),
            np.random.uniform(generator_config['env_size']['height']['min'], generator_config['env_size']['height']['max'])
        ]
        robot_size = generator_config['robot']['size']
        
        # Calculate the minimum distance from the wall
        min_distance = max(robot_size[0], robot_size[1]) / 2 + 0.1
        
        robot_pos = [
            np.random.uniform(-env_size[0]/2 + min_distance, env_size[0]/2 - min_distance),
            np.random.uniform(-env_size[1]/2 + min_distance, env_size[1]/2 - min_distance),
            robot_size[2] # + 0.05
        ]
        config = generate_env_config(env_size, robot_pos, generator_config)
        save_config(config, os.path.join(config_dir, f'env_config_{i+1}.yaml'))
    
    print(f"{num_configs} environment configurations generated and saved in the '{config_dir}' directory")
