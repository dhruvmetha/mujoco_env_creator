import yaml
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import os
import argparse
from tqdm import tqdm
from typing import Dict, Any, List
import numpy as np

np.random.seed(42)

# Constants
NAME = 'name'
POS = 'pos'
ROTATION = 'rotation'
MOVABLE = 'movable'
MASS = 'mass'

def load_config(filename: str) -> Dict[str, Any]:
    try:
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except IOError as e:
        raise IOError(f"Error reading file: {e}")

def load_generator_config(filename: str) -> Dict[str, Any]:
    try:
        with open(filename, 'r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except IOError as e:
        raise IOError(f"Error reading file: {e}")

def create_mujoco_xml(config):
    mujoco = Element('mujoco')
    mujoco.set('model', 'generated_environment')

    option = SubElement(mujoco, 'option')
    option.set('timestep', '0.01')
    option.set('integrator', 'RK4')
    option.set('cone', 'elliptic')

    # Add default section
    default = SubElement(mujoco, 'default')
    geom = SubElement(default, 'geom')
    geom.set('density', '1')

    
    asset = SubElement(mujoco, 'asset')
    create_textures(asset, config['worldbody']['asset']['texture'])
    create_materials(asset, config['worldbody']['asset']['material'])

    worldbody = SubElement(mujoco, 'worldbody')
    create_light(worldbody, config['worldbody']['light'])
    create_floor(worldbody, config['worldbody']['geom'])
    create_walls(worldbody, config['worldbody']['walls'])
    create_robot(worldbody, config['worldbody']['robot'])
    create_obstacles(worldbody, config['worldbody']['obstacles'])
    create_goal(worldbody, config['worldbody']['goal'])
    # Add actuators to the Mujoco XML
    create_actuators(mujoco)

    xml_str = minidom.parseString(tostring(mujoco)).toprettyxml(indent="  ")
    return xml_str

def create_textures(asset: Element, textures_config: List[Dict[str, Any]]) -> None:
    for texture_config in textures_config:
        texture = SubElement(asset, 'texture')
        for key, value in texture_config.items():
            if isinstance(value, list):
                texture.set(key, ' '.join(map(str, value)))
            else:
                texture.set(key, str(value))

def create_materials(asset: Element, materials_config: List[Dict[str, Any]]) -> None:
    for material_config in materials_config:
        material = SubElement(asset, 'material')
        for key, value in material_config.items():
            if isinstance(value, list):
                material.set(key, ' '.join(map(str, value)))
            else:
                material.set(key, str(value))

def create_light(worldbody: Element, light_config: Dict[str, Any]) -> None:
    light = SubElement(worldbody, 'light')
    for key, value in light_config.items():
        if isinstance(value, list):
            light.set(key, ' '.join(map(str, value)))
        else:
            light.set(key, str(value))

def create_floor(worldbody: Element, floor_config: Dict[str, Any]) -> None:
    floor = SubElement(worldbody, 'geom')
    for key, value in floor_config.items():
        if isinstance(value, list):
            floor.set(key, ' '.join(map(str, value)))
        else:
            floor.set(key, str(value))

def create_walls(worldbody: Element, walls_config: List[Dict[str, Any]]) -> None:
    # Create a single body element for all walls
    walls_body = SubElement(worldbody, 'body')
    walls_body.set(NAME, 'walls')
    
    for i, wall in enumerate(walls_config):
        # Create geom inside the walls body
        wall_name = wall.get(NAME, f'wall_{i}')  # Use provided name or generate one
        wall_geom = SubElement(walls_body, 'geom')
        wall_geom.set('name', wall_name)
        wall_geom.set('condim', str(wall.get('condim', 4)))  # Default to 6 if not specified
        
        # Set all other properties on the geom
        for key, value in wall.items():
            if key not in ['condim', NAME]:  # Skip condim as we've already set it
                if isinstance(value, list):
                    wall_geom.set(key, ' '.join(map(str, value)))
                else:
                    wall_geom.set(key, str(value))
                    
def create_goal(worldbody: Element, goal_config: Dict[str, Any]) -> None:
    goal_name = goal_config[NAME]
    site = SubElement(worldbody, 'site')
    site.set('name', goal_name)
    site.set('type', 'sphere')
    site.set('size', ' '.join(map(str, goal_config['size'])))
    site.set('rgba', ' '.join(map(str, goal_config['rgba'])))
    site.set('pos', ' '.join(map(str, goal_config['pos'])))
    
    # for key, value in goal_config.items():
    #     if key not in [NAME, POS, 'type', 'size', 'rgba']:
    #         if isinstance(value, list):
    #             site.set(key, ' '.join(map(str, value)))
    #         else:
    #             site.set(key, str(value))
                
def create_robot(worldbody: Element, robot_config: Dict[str, Any]) -> None:
    robot_name = robot_config[NAME]
    robot = SubElement(worldbody, 'body')
    robot.set(NAME, robot_name)
    # robot.set(POS, ' '.join(map(str, robot_config[POS])))

    # Add separate joints for x and y axes
    for axis in ['x', 'y']:
        joint = SubElement(robot, 'joint')
        joint.set('name', f'joint_{axis}')
        joint.set('type', 'slide')
        joint.set('pos', '0 0 0')
        joint.set('axis', f'{"1" if axis == "x" else "0"} {"1" if axis == "y" else "0"} 0')

    # Define geometry for the ball robot
    robot_size = robot_config['size']
    # robot_size[0] = 0.1
    # robot_size[1] = 0.0
    # robot_size[2] = 0.0
    geom = SubElement(robot, 'geom')
    geom.set('name', robot_name)
    geom.set('type', robot_config['type'])
    geom.set('pos', ' '.join(map(str, robot_config['pos'])))
    geom.set('size', ' '.join(map(str, robot_size)))
    geom.set('mass', str(robot_config['mass']))
    # geom.set('material', robot_config['material'])
    geom.set('friction', ' '.join(map(str, robot_config['friction'])))
    geom.set('condim', '4')

    # Add a site for the sensor
    # site = SubElement(robot, 'site')
    # site.set('name', 'sensor_ball')

    # Set other properties from robot_config if provided
    for key, value in robot_config.items():
        
        if key not in [NAME, POS, 'type', 'size', 'mass', 'material', 'friction', 'condim']:
            if isinstance(value, list):
                geom.set(key, ' '.join(map(str, value)))
            else:
                geom.set(key, str(value))

def create_obstacles(worldbody: Element, obstacles_config: List[Dict[str, Any]]) -> None:
    for obstacle in obstacles_config:
        obstacle_name = obstacle[NAME]
        obstacle_body = SubElement(worldbody, 'body')
        obstacle_body.set(NAME, obstacle_name)
        
        pos = obstacle[POS]
        # obstacle_body.set(POS, ' '.join(map(str, pos)))

        # Set rotation on body
        # if ROTATION in obstacle:
        #     obstacle_body.set('euler', f"0 0 {obstacle[ROTATION]}")

        geom = SubElement(obstacle_body, 'geom')
        geom.set('name', obstacle_name)
        geom.set('condim', str(obstacle.get('condim', 6)))
        geom.set('pos', ' '.join(map(str, pos)))
        if ROTATION in obstacle:
            geom.set('euler', f"0 0 {obstacle[ROTATION]}")

        for key, value in obstacle.items():
            if key not in [NAME, POS, ROTATION, MOVABLE, MASS, 'condim', 'edge_points']:
                if isinstance(value, list):
                    geom.set(key, ' '.join(map(str, value)))
                else:
                    geom.set(key, str(value))

        if obstacle[MOVABLE]:
            joint = SubElement(obstacle_body, 'joint')
            joint.set('type', 'free')
            if MASS in obstacle and obstacle[MASS] is not None:
                geom.set(MASS, str(obstacle[MASS]))
            
            # Add visual markers for edge points
            # if 'edge_points' in obstacle:
            #     for i, point in enumerate(obstacle['edge_points']):
            #         site = SubElement(worldbody, 'site')  # Attach to worldbody instead
            #         site.set('name', f'{obstacle[NAME]}_edge_point_{i}')
            #         site.set('type', 'sphere')
            #         site.set('size', '0.02')
            #         site.set('rgba', '0 0 1 1')
            #         # Use absolute coordinates instead of relative
            #         site.set('pos', f'{point[0]} {point[1]} 0')

def create_actuators(mujoco: Element) -> None:
    actuator = SubElement(mujoco, 'actuator')
    
    # Add torque actuators for x and y joints
    for axis in ['x', 'y']:
        motor = SubElement(actuator, 'motor')
        motor.set('name', f'actuator_{axis}')
        motor.set('joint', f'joint_{axis}')
        motor.set('gear', '1')  # Adjust this value as needed
        motor.set('ctrlrange', '-1 1')  # Set control range for the torque

def save_mujoco_xml(xml_str: str, filename: str) -> None:
    with open(filename, 'w') as f:
        f.write(xml_str)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='executables/mujoco_env_creator/generator_config.yaml')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    generator_config = load_generator_config(args.config)
    config_dir = generator_config.get('config_dir', 'env_configs')
    output_dir = generator_config.get('output_dir', 'mujoco_envs')
    
    # Clear out existing XML files
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.xml'):
                os.remove(os.path.join(output_dir, file))
    
    os.makedirs(output_dir, exist_ok=True)

    config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    
    for config_file in tqdm(config_files, desc="Generating MuJoCo XML files"):
        config_path = os.path.join(config_dir, config_file)
        config = load_config(config_path)
        mujoco_xml = create_mujoco_xml(config)
        output_file = os.path.join(output_dir, f"{os.path.splitext(config_file)[0]}.xml")
        save_mujoco_xml(mujoco_xml, output_file)
        # Remove the config file after successful creation of XML
        os.remove(config_path)
        # except Exception as e:
        #     print(f"Error processing {config_file}: {e}")

    print(f"MuJoCo XML files generated and saved in the '{output_dir}' directory")
