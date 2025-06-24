# plot the environment config in 2D using matplotlib

import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
import os
from tqdm import tqdm
import numpy as np


def load_goal_regions(filename):
    with open(filename, 'r') as f:
        regions = np.loadtxt(f, delimiter=',')
    
    print(regions.ndim)
    if regions.ndim == 1:
        regions = regions.reshape(1, -1)
    
    return regions

def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

def plot_env_config(config):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot floor
    env_size = config['env_size']
    ax.add_patch(Rectangle((-env_size[0]/2, -env_size[1]/2), 
                           env_size[0], env_size[1], 
                           fill=False, edgecolor='black'))
    
    # Plot walls
    for wall in config['worldbody']['walls']:
        x, y = wall['pos'][0], wall['pos'][1]
        w, h = 2 * wall['size'][0], 2 * wall['size'][1]
        ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, fill=True, facecolor='gray', alpha=0.5))
    
    # Plot robot
    robot = config['worldbody']['robot']
    x, y = robot['pos'][0], robot['pos'][1]
    w, h = 2 * robot['size'][0], 2 * robot['size'][1]
    # ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, fill=True, facecolor='green'))
    ax.add_patch(plt.Circle((x, y), radius=max(w, h)/8, fill=True, facecolor='gray'))
    
    # Plot obstacles
    for obstacle in config['worldbody']['obstacles']:
        x, y = obstacle['pos'][0], obstacle['pos'][1]
        w, h = 2 * obstacle['size'][0], 2 * obstacle['size'][1]
        rotation = obstacle.get('rotation', 0)
        color = 'yellow' if obstacle['movable'] else 'red'
        rect = Rectangle((x-w/2, y-h/2), w, h, fill=True, facecolor=color, alpha=0.5)
        t = transforms.Affine2D().rotate_deg_around(x, y, rotation) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    # plot goal regions from success and failure files
    wd = '/media/dhruv/a7519aee-b272-44ae-a117-1f1ea1796db6/2024/NAMO/random/data'
    success_file = os.path.join(wd, 'success_goals.txt')
    failure_file = os.path.join(wd, 'failed_goals.txt')
    success_regions = load_goal_regions(success_file)
    print(success_regions)
    failure_regions = load_goal_regions(failure_file)
    print(failure_regions)

    REGION_SIZE = 0.2

    for region in success_regions:
        print(region)
        ax.add_patch(Rectangle((region[0] - REGION_SIZE/2, region[1] - REGION_SIZE/2), 
                               REGION_SIZE, REGION_SIZE, 
                               fill=True, facecolor='green', alpha=0.5))
    for region in failure_regions:
        ax.add_patch(Rectangle((region[0] - REGION_SIZE/2, region[1] - REGION_SIZE/2), 
                               REGION_SIZE, REGION_SIZE, 
                               fill=True, facecolor='red', alpha=0.5))


    # Set axis limits and labels
    ax.set_xlim(-env_size[0]/2, env_size[0]/2)
    ax.set_ylim(-env_size[1]/2, env_size[1]/2)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Environment Configuration')

    
    
    plt.grid(True)

if __name__ == "__main__":

    config_dir = '/home/dhruv/2024/projects/ml4kp_ktamp/resources/input_files/mujoco_envs/env_configs'
    plots_dir = 'env_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
    
    for config_file in tqdm(config_files, desc="Plotting environment configs"):
        if config_file == 'env_config_44.yaml':
            # print(config_file)
            config_path = os.path.join(config_dir, config_file)
            config = load_config(config_path)
            plot_env_config(config)
            plt.savefig(os.path.join(plots_dir, f"{os.path.splitext(config_file)[0]}.png"))
            plt.close()
    
    print(f"All environment configurations plotted and saved in the '{plots_dir}' directory")
