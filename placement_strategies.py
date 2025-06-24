from abc import ABC, abstractmethod
import numpy as np

from shapely.geometry import Polygon

def check_collision(obj1, obj2):
    """Check collision between two objects using their polygons."""
    def get_rotated_corners(obj):
        if obj['name'] == 'robot':
            x, y = obj['pos'][0], obj['pos'][1]
            w, h = obj['size'][0], obj['size'][0]
            return Polygon([(x - w, y - h), (x + w, y - h), 
                          (x + w, y + h), (x - w, y + h)])
    
        x, y = obj['pos'][0], obj['pos'][1]
        w, h = obj['size'][0], obj['size'][1]
        angle = np.radians(obj.get('rotation', 0))
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        corners = [
            (x + w * cos_a - h * sin_a, y + w * sin_a + h * cos_a),
            (x - w * cos_a - h * sin_a, y - w * sin_a + h * cos_a),
            (x - w * cos_a + h * sin_a, y - w * sin_a - h * cos_a),
            (x + w * cos_a + h * sin_a, y + w * sin_a - h * cos_a)
        ]
        return Polygon(corners)

    poly1 = get_rotated_corners(obj1)
    poly2 = get_rotated_corners(obj2)
    return poly1.intersects(poly2)

def generate_edge_points(pos, size, rotation):
    """Generate edge points for movable objects."""
    points = []
    x, y = pos[0], pos[1]
    w, d = size[0], size[1]
    angle = np.radians(rotation)
    offset = 0.1
    
    def rotate_point(px, py, center_x, center_y):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx, dy = px - center_x, py - center_y
        rx = center_x + dx * cos_a - dy * sin_a
        ry = center_y + dx * sin_a + dy * cos_a
        return [rx, ry]

    edges = [
        [(x + w * (i - 1), y + d + offset) for i in range(3)],
        [(x + w * (i - 1), y - d - offset) for i in range(3)],
        [(x + w + offset, y + d * (i - 1)) for i in range(3)],
        [(x - w - offset, y + d * (i - 1)) for i in range(3)]
    ]

    for edge in edges:
        for point in edge:
            if rotation != 0:
                rotated = rotate_point(point[0], point[1], x, y)
                points.append(rotated)
            else:
                points.append([point[0], point[1]])

    return points

def generate_random_size(gen_config):
    """Generate random size within configured bounds."""
    size = [
        np.random.uniform(gen_config['obstacles']['size']['width']['min'], 
                         gen_config['obstacles']['size']['width']['max']),
        np.random.uniform(gen_config['obstacles']['size']['depth']['min'], 
                         gen_config['obstacles']['size']['depth']['max']),
        0.3  # Fixed height
    ]

    # check area is not too small
    if size[0] * size[1] < 0.06:
        return generate_random_size(gen_config)
    
    return size

def create_obstacle(pos, size, rotation, is_movable, index, gen_config):
    """Create an obstacle with given properties."""
    # Convert numpy types to native Python types
    pos = [float(p) for p in pos]
    size = [float(s) for s in size]
    rotation = float(rotation)
    
    obstacle = {
        'name': f'obstacle_{index}_{"movable" if is_movable else "static"}',
        'type': 'box',
        'pos': pos,
        'size': size,
        'rgba': [float(x) for x in gen_config['obstacles']['color']['movable' if is_movable else 'static']],
        'rotation': rotation,
        'movable': bool(is_movable),
        'friction': [float(f) for f in gen_config['obstacles']['friction']],
        'condim': 4
    }
    
    if is_movable:
        obstacle['mass'] = float(gen_config['obstacles']['movable_mass'])
        obstacle['edge_points'] = [[float(x) for x in point] for point in generate_edge_points(pos, size, rotation)]
    
    return obstacle

def is_valid_placement(obstacle, existing_objects):
    """Check if obstacle placement is valid."""
    return all(not check_collision(obstacle, obj) for obj in existing_objects)

class PlacementStrategy(ABC):
    """Base class for all placement strategies."""
    def __init__(self, env_size, robot_pos, gen_config):
        self.env_size = env_size
        self.robot_pos = robot_pos
        self.gen_config = gen_config
        self.max_attempts = 100
        self.min_robot_distance = 0.1

    def is_valid_distance_from_robot(self, pos):
        """Check if position is far enough from robot."""
        dx = pos[0] - self.robot_pos[0]
        dy = pos[1] - self.robot_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)
        return distance >= self.min_robot_distance

    def get_random_position(self, size):
        """Get random position within environment bounds."""
        for _ in range(self.max_attempts):
            pos = [
                np.random.uniform(-self.env_size[0]/2 + size[0], self.env_size[0]/2 - size[0]),
                np.random.uniform(-self.env_size[1]/2 + size[1], self.env_size[1]/2 - size[1]),
                size[2]
            ]
            if self.is_valid_distance_from_robot(pos):
                return self.clamp_position(pos, size)
        
        # If no valid position found, return clamped position and let collision check handle it
        return self.clamp_position(pos, size)

    def clamp_position(self, pos, size):
        """Clamp position to environment bounds."""
        return [
            np.clip(pos[0], -self.env_size[0]/2 + size[0], self.env_size[0]/2 - size[0]),
            np.clip(pos[1], -self.env_size[1]/2 + size[1], self.env_size[1]/2 - size[1]),
            pos[2]
        ]

    @abstractmethod
    def place_obstacles(self, existing_objects):
        """Generate and place obstacles according to strategy."""
        pass

class ArcBasedStrategy(PlacementStrategy):
    """Base class for arc-based placement strategies."""
    def __init__(self, env_size, robot_pos, gen_config, num_arcs):
        super().__init__(env_size, robot_pos, gen_config)
        self.num_arcs = num_arcs
        self.arc_size = 2 * np.pi / self.num_arcs
    
    def get_corner_points(self, pos, size, rotation):
        """Get the four corners of a rotated rectangle."""
        x, y = pos[0], pos[1]
        w, d = size[0], size[1]
        angle = np.radians(rotation)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        return [
            (x + w * cos_a - d * sin_a, y + w * sin_a + d * cos_a),  # top-right
            (x - w * cos_a - d * sin_a, y - w * sin_a + d * cos_a),  # top-left
            (x - w * cos_a + d * sin_a, y - w * sin_a - d * cos_a),  # bottom-left
            (x + w * cos_a + d * sin_a, y + w * sin_a - d * cos_a)   # bottom-right
        ]
    
    def distance_to_robot(self, point):
        """Calculate distance from a point to robot center."""
        return np.sqrt(
            (point[0] - self.robot_pos[0])**2 + 
            (point[1] - self.robot_pos[1])**2
        )

class SinglePairArcPlacement(ArcBasedStrategy):
    """Place one movable and one static object in 16 arcs, with static behind movable."""
    def __init__(self, env_size, robot_pos, gen_config):
        super().__init__(env_size, robot_pos, gen_config, num_arcs=16)
        
    def is_within_bounds(self, pos):
        """Check if position is within environment bounds."""
        return (-self.env_size[0]/2 <= pos[0] <= self.env_size[0]/2 and 
                -self.env_size[1]/2 <= pos[1] <= self.env_size[1]/2)
    
    def place_obstacles(self, existing_objects):
        for attempt in range(self.max_attempts):
            # 1. Randomly select an arc
            arc_index = np.random.randint(0, self.num_arcs)
            angle = np.random.uniform(arc_index * self.arc_size, (arc_index + 1) * self.arc_size)
            
            # 2. Place movable in the arc
            movable_size = generate_random_size(self.gen_config)
            movable = None
            
            # Keep sampling until we get a valid movable position
            for _ in range(self.max_attempts):
                distance = np.random.uniform(self.min_robot_distance, min(self.env_size[0]/2, self.env_size[1]/2))
                
                movable_pos = [
                    self.robot_pos[0] + distance * np.cos(angle),
                    self.robot_pos[1] + distance * np.sin(angle),
                    movable_size[2]
                ]
                
                if not self.is_within_bounds(movable_pos):
                    continue
                
                
                movable = create_obstacle(
                    pos=movable_pos,
                    size=movable_size,
                    rotation=np.random.uniform(0, 360),
                    is_movable=True,
                    index=1,
                    gen_config=self.gen_config
                )
                
                if is_valid_placement(movable, existing_objects):
                    break
                # print("not valid placement")
                movable = None
            
            if not movable:
                continue
                
            # 3. Place static in the same arc
            static_size = generate_random_size(self.gen_config)
            min_distance = self.distance_to_robot(movable['pos'])
            
            for _ in range(self.max_attempts):
                distance = np.random.uniform(min_distance, min(self.env_size[0]/2, self.env_size[1]/2))
                angle = np.random.uniform(arc_index * self.arc_size, (arc_index + 1) * self.arc_size)
                
                static_pos = [
                    self.robot_pos[0] + distance * np.cos(angle),
                    self.robot_pos[1] + distance * np.sin(angle),
                    static_size[2]
                ]

                # if static_pos[0] < movable_pos[0] and static_pos[1] < movable_pos[1]:
                #     continue
                
                if not self.is_within_bounds(static_pos):
                    continue
                
                static = create_obstacle(
                    pos=static_pos,
                    size=static_size,
                    rotation=np.random.uniform(0, 360),
                    is_movable=False,
                    index=2,
                    gen_config=self.gen_config
                )

                if is_valid_placement(static, existing_objects + [movable]):
                    return [movable, static]
                
                # Check if static is behind movable
                # movable_corners = self.get_corner_points(movable['pos'], movable['size'], movable['rotation'])
                # max_movable_distance = max(self.distance_to_robot(corner) for corner in movable_corners)
                # static_corners = self.get_corner_points(static['pos'], static['size'], static['rotation'])
                
                # if any(self.distance_to_robot(corner) <= max_movable_distance for corner in static_corners):
                #     print("not behind movable")
                #     continue
                
                # print("not valid placement")
                    
        return []

class MultiMovableArcPlacement(ArcBasedStrategy):
    """Place multiple movable objects in 16 arcs with increasing distances."""
    def __init__(self, env_size, robot_pos, gen_config):
        super().__init__(env_size, robot_pos, gen_config, num_arcs=16)
        self.num_movable = gen_config['obstacles']['num_movable']

    def place_obstacles(self, existing_objects):
        for attempt in range(self.max_attempts):
            obstacles = []
            start_arc_index = np.random.randint(0, self.num_arcs)
            
            for i in range(self.num_movable):
                arc_variation = i + 1
                arc_index = (start_arc_index + i * arc_variation) % self.num_arcs
                base_angle = arc_index * self.arc_size
                
                min_distance = 0.8 + i * 0.3
                max_distance = min_distance + 0.5
                
                movable = self._place_movable(i+1, base_angle, min_distance, max_distance, 
                                            existing_objects + obstacles)
                if not movable:
                    break
                obstacles.append(movable)
            
            if len(obstacles) == self.num_movable:
                return obstacles
                
        return []

    def _place_movable(self, index, base_angle, min_distance, max_distance, existing_objects):
        """Place single movable object with specified parameters."""
        for _ in range(self.max_attempts):
            size = generate_random_size(self.gen_config)
            distance = np.random.uniform(min_distance, max_distance)
            angle_variation = self.arc_size * (1 + (index-1) * 0.5)
            angle = base_angle + np.random.uniform(-angle_variation/2, angle_variation/2)
            
            pos = [
                self.robot_pos[0] + distance * np.cos(angle),
                self.robot_pos[1] + distance * np.sin(angle),
                0.3
            ]
            pos = self.clamp_position(pos, size)
            
            movable = create_obstacle(
                pos=pos,
                size=size,
                rotation=np.random.uniform(0, 360),
                is_movable=True,
                index=index,
                gen_config=self.gen_config
            )
            
            if is_valid_placement(movable, existing_objects):
                return movable
        return None

class QuadrantMixedPlacement(ArcBasedStrategy):
    """Place random number of mixed objects in 4 quadrants."""
    def __init__(self, env_size, robot_pos, gen_config):
        super().__init__(env_size, robot_pos, gen_config, num_arcs=4)
        self.min_count = gen_config['obstacles']['count']['min']
        self.max_count = gen_config['obstacles']['count']['max']
        self.movable_prob = gen_config['obstacles']['movable_probability']

    def place_obstacles(self, existing_objects):
        total_obstacles = np.random.randint(self.min_count, self.max_count + 1)
        
        for attempt in range(self.max_attempts):
            obstacles = []
            start_quadrant = np.random.randint(0, self.num_arcs)
            
            for i in range(total_obstacles):
                is_movable = np.random.random() < self.movable_prob
                quadrant_index = (start_quadrant + i) % self.num_arcs
                base_angle = quadrant_index * self.arc_size
                
                min_distance = 0.8 + (i // self.num_arcs) * 0.4
                max_distance = min_distance + 0.5
                
                obstacle = self._place_obstacle(i+1, base_angle, min_distance, max_distance,
                                             is_movable, existing_objects + obstacles)
                if not obstacle:
                    break
                obstacles.append(obstacle)
            
            if len(obstacles) == total_obstacles:
                return obstacles
                
        return []

    def _place_obstacle(self, index, base_angle, min_distance, max_distance, 
                       is_movable, existing_objects):
        """Place single obstacle with specified parameters."""
        for _ in range(self.max_attempts):
            size = generate_random_size(self.gen_config)
            distance = np.random.uniform(min_distance, max_distance)
            angle = base_angle + np.random.uniform(0, self.arc_size)
            
            pos = [
                self.robot_pos[0] + distance * np.cos(angle),
                self.robot_pos[1] + distance * np.sin(angle),
                0.3
            ]
            pos = self.clamp_position(pos, size)
            
            obstacle = create_obstacle(
                pos=pos,
                size=size,
                rotation=np.random.uniform(0, 360),
                is_movable=is_movable,
                index=index,
                gen_config=self.gen_config
            )
            
            if is_valid_placement(obstacle, existing_objects):
                return obstacle
        return None