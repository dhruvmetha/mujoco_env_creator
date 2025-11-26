"""
MuJoCo Template Generation System

This module adapts maze generation algorithms to create MuJoCo XML environment files.
It integrates the maze generation logic from Assignment1/lib/maze.py with MuJoCo XML
structure to create navigable 3D environments.
"""

import math
import os
import random
import argparse
from enum import Enum
from typing import List, Tuple, Optional
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

# ============================================================================
# Core Types and Constants (adapted from Assignment1/lib/types.py)
# ============================================================================

# Direction constants: [dy, dx] for up, right, down, left
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# Type aliases
Coordinate = Tuple[int, int]
Grid = List[List[bool]]
IntGrid = List[List[int]]

class Annotations(Enum):
    NONE = 0
    VISITED = 1
    PATH = 2

class Environment:
    """Environment configuration for maze generation"""
    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        self.width = width
        self.height = height
        # Start and goal in guaranteed open areas for generated mazes
        self.start = (1, 1)  # Top-left open area
        self.goal = (height - 2, width - 2)  # Bottom-right open area
        self.debug = False
        if seed is not None:
            self.rand = random.Random(seed)
        else:
            self.rand = random.Random()
    
    def find_valid_start_goal(self, maze: Grid) -> Tuple[Coordinate, Coordinate]:
        """Find valid start and goal positions in open areas"""
        height = len(maze)
        width = len(maze[0])
        
        # Find all open cells
        open_cells = []
        for y in range(height):
            for x in range(width):
                if not maze[y][x]:  # False means open
                    open_cells.append((y, x))
        
        if len(open_cells) < 2:
            raise ValueError("Maze has fewer than 2 open cells - cannot place start and goal")
        
        # Pick start from top-left area, goal from bottom-right area
        start_candidates = [(y, x) for y, x in open_cells if y < height//2 and x < width//2]
        goal_candidates = [(y, x) for y, x in open_cells if y >= height//2 and x >= width//2]
        
        if not start_candidates:
            start_candidates = open_cells[:len(open_cells)//2]
        if not goal_candidates:
            goal_candidates = open_cells[len(open_cells)//2:]
        
        start = self.rand.choice(start_candidates)
        goal = self.rand.choice(goal_candidates)
        
        return start, goal

# ============================================================================
# Utility Functions (adapted from Assignment1/lib/types.py)
# ============================================================================

def check(pos: Coordinate, height: int, width: int) -> bool:
    """Check if position is within grid bounds"""
    y, x = pos
    return 0 <= y < height and 0 <= x < width

def empty_grid(height: int, width: int) -> Grid:
    """Create an empty boolean grid filled with False"""
    return [[False for _ in range(width)] for _ in range(height)]

def filled_grid_bool(height: int, width: int) -> Grid:
    """Create a boolean grid filled with True"""
    return [[True for _ in range(width)] for _ in range(height)]

def filled_grid_int(height: int, width: int, fill_value: int) -> IntGrid:
    """Create an integer grid filled with specified value"""
    return [[fill_value for _ in range(width)] for _ in range(height)]

def shift(pos: Coordinate, direction: int) -> Coordinate:
    """Shift position by one step in given direction"""
    dy, dx = DIRECTIONS[direction]
    return (pos[0] + dy, pos[1] + dx)

def binomial(rand: random.Random, p: float, n: int) -> int:
    """Generate binomial random number"""
    count = 0
    for _ in range(n):
        if rand.random() < p:
            count += 1
    return count

# ============================================================================
# Maze Generation Algorithms (adapted from Assignment1/lib/maze.py)
# ============================================================================

def random_maze(env: Environment) -> Grid:
    """Generate a random maze with 30% wall fill ratio"""
    fill_ratio: float = 0.3
    maze = empty_grid(env.height, env.width)
    for i in range(env.height):
        for j in range(env.width):
            maze[i][j] = env.rand.random() < fill_ratio
    return maze

def random_coordinate(visited: Grid, env: Environment, unvisited: int = -1) -> Optional[Coordinate]:
    """Find a random unvisited coordinate"""
    if unvisited == -1:
        unvisited = sum(row.count(False) for row in visited)
    if unvisited == 0:
        return None
    index = env.rand.randint(1, unvisited)
    for i in range(env.height):
        for j in range(env.width):
            if not visited[i][j]:
                index -= 1
            if index == 0:
                return (i, j)
    return None

def dfs_maze(env: Environment) -> Grid:
    """Generate maze using depth-first search"""
    fill_ratio: float = 0.3
    maze = empty_grid(env.height, env.width)
    visited = empty_grid(env.height, env.width)
    unvisited = env.width * env.height

    stack: List[Coordinate] = []
    counter = 0

    while unvisited > 0:
        counter += 1
        start = random_coordinate(visited, env, unvisited)
        assert start is not None, "Could not find start coordinate"
        visited[start[0]][start[1]] = True
        maze[start[0]][start[1]] = False
        stack.append(start)
        unvisited -= 1

        while stack:
            y, x = stack[-1]
            if not visited[y][x]:
                maze[y][x] = env.rand.random() < fill_ratio
                visited[y][x] = True
                unvisited -= 1

            if maze[y][x]:
                stack.pop()
                continue

            valid: List[int] = []
            for i in range(4):
                dy, dx = DIRECTIONS[i]
                if (
                    check((y + dy, x + dx), env.height, env.width)
                    and not visited[y + dy][x + dx]
                ):
                    valid.append(i)
            if not valid:
                stack.pop()
                continue

            dir_choice = env.rand.choice(valid)
            dy, dx = DIRECTIONS[dir_choice]
            y += dy
            x += dx
            stack.append((y, x))

    return maze

def convert_nodes(nodes: IntGrid) -> Grid:
    """Convert node-based maze to grid representation"""
    h = len(nodes)
    w = len(nodes[0])
    height = h * 2 - 1
    width = w * 2 - 1

    grid = filled_grid_bool(height, width)
    for i in range(h):
        for j in range(w):
            grid[i * 2][j * 2] = False
            for d, [dy, dx] in enumerate(DIRECTIONS):
                y = i * 2 + dy
                x = j * 2 + dx
                if nodes[i][j] & (1 << d) and check((y, x), height, width):
                    grid[y][x] = False

    return grid

def recursive_backtrack(env: Environment) -> Grid:
    """Generate maze using recursive backtracking"""
    if env.width % 2 == 0 or env.height % 2 == 0:
        raise Exception("node based maze requires odd dimensions")

    w = 1 + env.width // 2
    h = 1 + env.height // 2
    nodes = filled_grid_int(h, w, 0)
    stack: List[Coordinate] = [(0, 0)]

    while stack:
        y, x = stack[-1]
        valid: List[int] = []
        for i in range(4):
            dy, dx = DIRECTIONS[i]
            if check((y + dy, x + dx), h, w) and nodes[y + dy][x + dx] == 0:
                valid.append(i)
        if not valid:
            stack.pop()
            continue

        dir_choice = env.rand.choice(valid)
        dy, dx = DIRECTIONS[dir_choice]
        y += dy
        x += dx
        dir_choice = (dir_choice + 2) % 4
        nodes[y][x] |= 1 << dir_choice
        stack.append((y, x))

    return convert_nodes(nodes)

# ============================================================================
# MuJoCo XML Generation System
# ============================================================================

class MuJoCoMazeGenerator:
    """Generates MuJoCo XML environments from maze grids"""
    
    def __init__(self, wall_deletion_prob: float = 0.0):
        # Physical constants from benchmark file
        self.wall_thickness = 0.05
        self.wall_height = 0.3
        self.wall_z_pos = 0.3
        self.robot_radius = 0.15
        self.robot_mass = 5.0
        self.robot_z_pos = 0.2
        self.goal_radius = 0.3
        self.goal_z_pos = 0.0
        self.goal_alpha = 0.2
        
        # Make each grid cell larger to create wider pathways
        self.cell_size = 1.0  # Each cell is 1.0x1.0 meters - good corridor size
        
        # Wall deletion probability (0.0 = no deletion, 1.0 = delete all walls)
        self.wall_deletion_prob = wall_deletion_prob
        
    def create_base_xml(self) -> Element:
        """Create the base MuJoCo XML structure"""
        root = Element("mujoco", model="generated_maze_environment")
        
        # Options
        option = SubElement(root, "option", 
                          timestep="0.01", 
                          integrator="RK4", 
                          cone="elliptic")
        
        # Default settings
        default = SubElement(root, "default")
        SubElement(default, "geom", density="1")
        
        # Assets
        asset = SubElement(root, "asset")
        SubElement(asset, "texture", 
                  builtin="gradient", 
                  height="3072", 
                  rgb1="0.3 0.5 0.7", 
                  rgb2="0 0 0", 
                  type="skybox", 
                  width="512")
        SubElement(asset, "texture", 
                  builtin="checker", 
                  height="300", 
                  mark="edge", 
                  markrgb="0.8 0.8 0.8", 
                  name="groundplane", 
                  rgb1="0.2 0.3 0.4", 
                  rgb2="0.1 0.2 0.3", 
                  type="2d", 
                  width="300")
        SubElement(asset, "material", 
                  name="groundplane", 
                  reflectance="0.2", 
                  texrepeat="5 5", 
                  texture="groundplane", 
                  texuniform="true")
        SubElement(asset, "material", 
                  name="robot", 
                  rgba="0.0 1.0 0.0 1")
        
        return root
    
    def add_worldbody(self, root: Element, maze: Grid, env: Environment) -> None:
        """Add worldbody with maze walls, robot, and goal"""
        worldbody = SubElement(root, "worldbody")
        
        # Find valid start and goal positions
        start_pos, goal_pos = env.find_valid_start_goal(maze)
        
        # Lighting
        SubElement(worldbody, "light", 
                  dir="0 0 -1", 
                  directional="true", 
                  pos="0 0 1.5")
        
        # Ground plane
        SubElement(worldbody, "geom", 
                  condim="4", 
                  friction="0.5 0.005 0.001", 
                  material="groundplane", 
                  name="floor", 
                  size="0 0 0.05", 
                  type="plane")
        
        # Origin markers
        SubElement(worldbody, "site", 
                  name="origin_x", 
                  pos="0.1 0 0.0", 
                  rgba="1 0 0 1", 
                  size="0.05", 
                  type="sphere")
        SubElement(worldbody, "site", 
                  name="origin_y", 
                  pos="0 0.1 0", 
                  rgba="0 1 0 1", 
                  size="0.05", 
                  type="sphere")
        SubElement(worldbody, "site", 
                  name="origin_z", 
                  pos="0 0 0.1", 
                  rgba="0 0 1 1", 
                  size="0.05", 
                  type="sphere")
        
        # Add walls
        self.add_walls(worldbody, maze, env)
        
        # Add robot with valid start position
        self.add_robot(worldbody, env, start_pos)
        
        # Add goal with valid goal position
        self.add_goal(worldbody, env, goal_pos)
    
    def add_walls(self, worldbody: Element, maze: Grid, env: Environment) -> None:
        """Add wall geometries based on maze grid"""
        walls_body = SubElement(worldbody, "body", name="walls")
        
        # Add boundary walls
        self.add_boundary_walls(walls_body, maze)
        
        # Add internal walls based on maze grid
        self.add_internal_walls(walls_body, maze, env)
    
    def add_boundary_walls(self, walls_body: Element, maze: Grid) -> None:
        """Add perimeter walls around the maze"""
        height = len(maze)
        width = len(maze[0])
        
        # Convert grid dimensions to world coordinates
        world_width = width * self.cell_size
        world_height = height * self.cell_size
        
        # Calculate wall positions (center the maze at origin)
        half_width = world_width / 2
        half_height = world_height / 2
        
        # Top wall
        SubElement(walls_body, "geom", 
                  name="wall_boundary_top",
                  condim="4",
                  pos=f"0 {half_height + self.wall_thickness/2} {self.wall_z_pos}",
                  rgba="0.8 0.8 0.8 1",
                  size=f"{half_width + self.wall_thickness} {self.wall_thickness/2} {self.wall_height}",
                  type="box")
        
        # Bottom wall
        SubElement(walls_body, "geom", 
                  name="wall_boundary_bottom",
                  condim="4",
                  pos=f"0 {-half_height - self.wall_thickness/2} {self.wall_z_pos}",
                  rgba="0.8 0.8 0.8 1",
                  size=f"{half_width + self.wall_thickness} {self.wall_thickness/2} {self.wall_height}",
                  type="box")
        
        # Left wall
        SubElement(walls_body, "geom", 
                  name="wall_boundary_left",
                  condim="4",
                  pos=f"{-half_width - self.wall_thickness/2} 0 {self.wall_z_pos}",
                  rgba="0.8 0.8 0.8 1",
                  size=f"{self.wall_thickness/2} {half_height} {self.wall_height}",
                  type="box")
        
        # Right wall
        SubElement(walls_body, "geom", 
                  name="wall_boundary_right",
                  condim="4",
                  pos=f"{half_width + self.wall_thickness/2} 0 {self.wall_z_pos}",
                  rgba="0.8 0.8 0.8 1",
                  size=f"{self.wall_thickness/2} {half_height} {self.wall_height}",
                  type="box")
    
    def add_internal_walls(self, walls_body: Element, maze: Grid, env: Environment) -> None:
        """Add internal walls as thin lines - only where needed to separate open spaces"""
        height = len(maze)
        width = len(maze[0])
        wall_count = 0
        
        # Add horizontal walls (between vertically adjacent cells)
        for y in range(height - 1):
            for x in range(width):
                # Add wall if one cell is open and the other is blocked, or if both are blocked
                cell_above = maze[y][x]     # True = blocked, False = open
                cell_below = maze[y+1][x]   # True = blocked, False = open
                
                # Only add wall if we need to separate different cell types or block passage
                if cell_above != cell_below or (cell_above and cell_below):
                    # Randomly delete wall based on deletion probability
                    if env.rand.random() >= self.wall_deletion_prob:
                        wall_count += 1
                        # Position wall between the cells
                        world_x = (x - width/2 + 0.5) * self.cell_size
                        world_y = (height/2 - y - 1) * self.cell_size  # Between y and y+1
                        
                        SubElement(walls_body, "geom",
                                  name=f"wall_{wall_count}",
                                  condim="4",
                                  pos=f"{world_x} {world_y} {self.wall_z_pos}",
                                  rgba="0.8 0.8 0.8 1",
                                  size=f"{self.cell_size/2} {self.wall_thickness/2} {self.wall_height}",
                                  type="box")
        
        # Add vertical walls (between horizontally adjacent cells)
        for y in range(height):
            for x in range(width - 1):
                # Add wall if one cell is open and the other is blocked, or if both are blocked
                cell_left = maze[y][x]      # True = blocked, False = open
                cell_right = maze[y][x+1]   # True = blocked, False = open
                
                # Only add wall if we need to separate different cell types or block passage
                if cell_left != cell_right or (cell_left and cell_right):
                    # Randomly delete wall based on deletion probability
                    if env.rand.random() >= self.wall_deletion_prob:
                        wall_count += 1
                        # Position wall between the cells
                        world_x = (x - width/2 + 1) * self.cell_size  # Between x and x+1
                        world_y = (height/2 - y - 0.5) * self.cell_size
                        
                        SubElement(walls_body, "geom",
                                  name=f"wall_{wall_count}",
                                  condim="4",
                                  pos=f"{world_x} {world_y} {self.wall_z_pos}",
                                  rgba="0.8 0.8 0.8 1",
                                  size=f"{self.wall_thickness/2} {self.cell_size/2} {self.wall_height}",
                                  type="box")
    
    def add_robot(self, worldbody: Element, env: Environment, start_pos: Coordinate) -> None:
        """Add robot body with proper joints and geometry"""
        robot_body = SubElement(worldbody, "body", name="robot")
        
        # Add sliding joints
        SubElement(robot_body, "joint", 
                  name="joint_x", 
                  type="slide", 
                  pos="0 0 0", 
                  axis="1 0 0")
        SubElement(robot_body, "joint", 
                  name="joint_y", 
                  type="slide", 
                  pos="0 0 0", 
                  axis="0 1 0")
        
        # Convert start position to world coordinates
        start_y, start_x = start_pos
        world_x = (start_x - env.width/2 + 0.5) * self.cell_size
        world_y = (env.height/2 - start_y - 0.5) * self.cell_size
        
        # Add robot geometry
        SubElement(robot_body, "geom", 
                  name="robot",
                  type="sphere",
                  pos=f"{world_x} {world_y} {self.robot_z_pos}",
                  size=f"{self.robot_radius}",
                  mass=f"{self.robot_mass}",
                  friction="1.0 0.005 0.001",
                  condim="4")
        
        # Add sensor site
        SubElement(robot_body, "site", name="sensor_ball")
    
    def add_goal(self, worldbody: Element, env: Environment, goal_pos: Coordinate) -> None:
        """Add goal site at maze end position"""
        # Convert goal position to world coordinates
        goal_y, goal_x = goal_pos
        world_x = (goal_x - env.width/2 + 0.5) * self.cell_size
        world_y = (env.height/2 - goal_y - 0.5) * self.cell_size
        
        SubElement(worldbody, "site", 
                  name="goal",
                  pos=f"{world_x} {world_y} {self.goal_z_pos}",
                  rgba=f"0 1 0 {self.goal_alpha}",
                  size=f"{self.goal_radius}",
                  type="sphere")
    
    def add_actuators(self, root: Element) -> None:
        """Add motor actuators for robot control"""
        actuator = SubElement(root, "actuator")
        SubElement(actuator, "motor", 
                  name="actuator_x", 
                  joint="joint_x", 
                  gear="1", 
                  ctrlrange="-1 1")
        SubElement(actuator, "motor", 
                  name="actuator_y", 
                  joint="joint_y", 
                  gear="1", 
                  ctrlrange="-1 1")
    
    def generate_xml(self, maze: Grid, env: Environment) -> str:
        """Generate complete MuJoCo XML from maze grid"""
        root = self.create_base_xml()
        self.add_worldbody(root, maze, env)
        self.add_actuators(root)
        
        # Convert to pretty-printed string
        rough_string = tostring(root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def save_xml(self, maze: Grid, env: Environment, filename: str, output_dir: str = "../generated_templates") -> None:
        """Save maze as MuJoCo XML file"""
        os.makedirs(output_dir, exist_ok=True)
        xml_content = self.generate_xml(maze, env)
        
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            f.write(xml_content)
        print(f"Generated maze environment: {filepath}")
    
    def generate_multiple_mazes(self, height: int, width: int, deletion_rate: float, num_mazes: int, output_dir: str = "../generated_templates") -> None:
        """Generate multiple maze environments with specified parameters.
        
        Args:
            height: Height of the maze
            width: Width of the maze  
            deletion_rate: Wall deletion probability. If negative, generates with incremental rates (10%, 20%, etc.)
            num_mazes: Number of mazes to generate
        """
        if deletion_rate < 0:
            # Generate with incremental deletion rates: 10%, 20%, 30%, etc.
            mazes_per_rate = math.ceil(num_mazes / 10)
            
            generated_count = 0
            for rate_level in range(10):  # 10%, 20%, ..., 100%
                if generated_count >= num_mazes:
                    break
                    
                current_deletion_rate = (rate_level + 1) * 0.1  # 10%, 20%, 30%, etc.
                
                # Generate up to mazes_per_rate mazes at this deletion rate
                mazes_to_generate = min(mazes_per_rate, num_mazes - generated_count)
                
                for j in range(mazes_to_generate):
                    seed = random.randint(1, 100000)
                    
                    # Create new environment and generator with this deletion rate
                    env = Environment(width=width, height=height, seed=seed)
                    maze = recursive_backtrack(env)
                    generator = MuJoCoMazeGenerator(wall_deletion_prob=current_deletion_rate)
                    
                    filename = f"maze_{height}x{width}_del{int(current_deletion_rate*100)}p_seed{seed}.xml"
                    generator.save_xml(maze, env, filename, output_dir=output_dir)
                    generated_count += 1
                    print(f"Generated maze {generated_count}/{num_mazes}: {filename}")
        else:
            # Generate all mazes with the same deletion probability
            for i in range(num_mazes):
                seed = random.randint(1, 100000)
                
                # Create new environment and generator with specified deletion rate
                env = Environment(width=width, height=height, seed=seed)
                maze = recursive_backtrack(env)
                generator = MuJoCoMazeGenerator(wall_deletion_prob=deletion_rate)
                
                filename = f"maze_{height}x{width}_del{int(deletion_rate*100)}p_seed{seed}.xml"
                generator.save_xml(maze, env, filename, output_dir=output_dir)
                print(f"Generated maze {i + 1}/{num_mazes}: {filename}")
        
        print(f"\nCompleted generation of {num_mazes} mazes ({height}x{width})")

def main():
    """Command line interface for maze generation."""
    parser = argparse.ArgumentParser(description='Generate MuJoCo maze environments')
    parser.add_argument('--height', type=int, default=11, help='Height of the maze (default: 11)')
    parser.add_argument('--width', type=int, default=11, help='Width of the maze (default: 11)')  
    parser.add_argument('--deletion_rate', type=float, default=-1, 
                       help='Wall deletion probability 0.0-1.0. If negative, generates incremental rates (default: -1)')
    parser.add_argument('--num_mazes', type=int, default=1, help='Number of mazes to generate (default: 1)')
    parser.add_argument('--output_dir', type=str, default='../generated_templates', help='Output directory for generated XML files')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.height < 3 or args.width < 3:
        print("Error: Height and width must be at least 3")
        return
        
    if args.deletion_rate >= 0 and (args.deletion_rate < 0.0 or args.deletion_rate > 1.0):
        print("Error: Deletion rate must be between 0.0 and 1.0, or negative for incremental rates")
        return
        
    if args.num_mazes < 1:
        print("Error: Number of mazes must be at least 1")  
        return
    
    print(f"Generating {args.num_mazes} maze(s) with dimensions {args.height}x{args.width}")
    if args.deletion_rate < 0:
        print("Using incremental deletion rates (10%, 20%, 30%, etc.)")
        print(f"Generating {math.ceil(args.num_mazes/10)} mazes per deletion rate level")
    else:
        print(f"Using {int(args.deletion_rate*100)}% wall deletion probability")
    print()
    
    # Create generator and generate the requested mazes
    generator = MuJoCoMazeGenerator()  # We'll create specific generators inside the method
    generator.generate_multiple_mazes(args.height, args.width, args.deletion_rate, args.num_mazes, args.output_dir)


if __name__ == "__main__":
    main()
