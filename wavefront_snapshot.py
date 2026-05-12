from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, cast
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray


GridArray = NDArray[np.int_]


@dataclass
class MovableObjectRecord:
    name: str
    x: float
    y: float
    theta: float
    half_extent_x: float
    half_extent_y: float

    def to_json(self) -> Dict[str, Union[str, float]]:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "half_extent_x": self.half_extent_x,
            "half_extent_y": self.half_extent_y,
        }


@dataclass(frozen=True)
class ObjectTemplate:
    """Immutable geometric information for an object."""

    name: str
    half_extent: Tuple[float, float]
    is_static: bool


@dataclass
class ObjectInstance:
    """Concrete pose for an object at export time."""

    template: ObjectTemplate
    position: Tuple[float, float]
    quaternion: Tuple[float, float, float, float]

    @property
    def half_extent(self) -> Tuple[float, float]:
        return self.template.half_extent

    @property
    def name(self) -> str:
        return self.template.name


@dataclass
class WavefrontSnapshot:
    resolution: float
    bounds: Tuple[float, float, float, float]
    uninflated_grid: GridArray
    static_grid: GridArray
    dynamic_grid: GridArray
    region_map: GridArray
    region_labels: Dict[int, str]
    adjacency: Dict[str, Set[str]]
    edge_objects: Dict[str, Dict[str, Set[str]]]
    robot_pose: Tuple[float, float, float]
    goal_pose: Optional[Tuple[float, float, float]]
    xml_path: str
    config_path: str
    robot_half_extent: Tuple[float, float]
    movable_objects: List[MovableObjectRecord]

    def metadata(self) -> Dict[str, object]:
        """Return a JSON-serialisable metadata dictionary."""

        return {
            "resolution": self.resolution,
            "bounds": self.bounds,
            "grid_shape": list(self.dynamic_grid.shape),
            "robot_pose": self.robot_pose,
            "goal_pose": self.goal_pose,
            "robot_half_extent": self.robot_half_extent,
            "region_labels": {str(idx): label for idx, label in self.region_labels.items()},
            "adjacency": {region: sorted(neighbors) for region, neighbors in self.adjacency.items()},
            "adjacency_objects": {
                region: {
                    neighbor: sorted(objs)
                    for neighbor, objs in neighbor_map.items()
                }
                for region, neighbor_map in self.edge_objects.items()
            },
            "xml_path": self.xml_path,
            "config_path": self.config_path,
            "movable_objects": [record.to_json() for record in self.movable_objects],
        }

    def save(self, output_dir: Path, prefix: str = "snapshot") -> Dict[str, Path]:
        """Persist snapshot arrays and metadata to disk."""

        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "uninflated_grid": output_dir / f"{prefix}_uninflated_grid.npy",
            "static_grid": output_dir / f"{prefix}_static_grid.npy",
            "dynamic_grid": output_dir / f"{prefix}_dynamic_grid.npy",
            "region_map": output_dir / f"{prefix}_region_map.npy",
            "metadata": output_dir / f"{prefix}_metadata.json",
        }

        np.save(paths["uninflated_grid"], self.uninflated_grid)
        np.save(paths["static_grid"], self.static_grid)
        np.save(paths["dynamic_grid"], self.dynamic_grid)
        np.save(paths["region_map"], self.region_map)

        with paths["metadata"].open("w", encoding="utf-8") as handle:
            json.dump(self.metadata(), handle, indent=2)

        return paths


class WavefrontSnapshotExporter:
    """Recreates wavefront grid data within Python for visualisation."""

    # Match the fixed resolution used by the C++ implementation
    DEFAULT_RESOLUTION: float = 0.01
    INFLATION_EPSILON: float = 0.005
    # Skip "regions" smaller than this many cells — they're sub-robot-footprint and
    # can't host the car anyway, just clutter the graph.
    MIN_REGION_CELLS: int = 8
    NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    )

    def __init__(self, env: Any, resolution: Optional[float] = None,
                 robot_half_extent_override: Optional[Tuple[float, float]] = None) -> None:
        self._env = env
        self.resolution = resolution or self.DEFAULT_RESOLUTION
        bounds_sequence = cast(Sequence[float], self._env.get_world_bounds())
        if len(bounds_sequence) != 4:
            raise ValueError("World bounds must contain [xmin, xmax, ymin, ymax]")
        self.bounds: Tuple[float, float, float, float] = (
            float(bounds_sequence[0]),
            float(bounds_sequence[1]),
            float(bounds_sequence[2]),
            float(bounds_sequence[3]),
        )

        object_info = cast(Dict[str, Dict[str, float]], self._env.get_object_info())
        if "robot" not in object_info:
            raise ValueError("Environment did not provide robot geometry via get_object_info()")

        if robot_half_extent_override is not None:
            # Caller supplied authoritative robot footprint (e.g. from namo_config.planning.robot_size).
            # Use it instead of the env's first-geom guess, which under-estimates multi-geom robots.
            self.robot_half_extent = (float(robot_half_extent_override[0]),
                                      float(robot_half_extent_override[1]))
        else:
            robot_info = object_info["robot"]
            size_x = float(robot_info.get("size_x", 0.25))
            size_y_raw = robot_info.get("size_y")
            if size_y_raw is None:
                size_y = size_x
            else:
                size_y = float(size_y_raw)
                if size_y <= 1e-6:
                    size_y = size_x  # Some environments report a zero Y half-extent for spheres

            self.robot_half_extent = (size_x, size_y)

        self.static_objects = self._build_static_objects(object_info)
        self.movable_templates = self._build_movable_templates(object_info)

        self.grid_width = max(1, int((self.bounds[1] - self.bounds[0]) / self.resolution))
        self.grid_height = max(1, int((self.bounds[3] - self.bounds[2]) / self.resolution))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_snapshot(
        self,
        xml_path: str,
        config_path: str,
        goal_radius: float = 0.15,
        goals_per_region: int = 0,
        rng: Optional[np.random.Generator] = None,
        use_current_state: bool = False,
    ) -> WavefrontSnapshot:
        """Construct grids, regions, and adjacency information.

        Args:
            use_current_state: If True, use current env state instead of resetting.
                              Useful for multi-level exploration where state was set via set_full_state().
        """

        if not use_current_state:
            self._env.reset()
        observation = self._env.get_observation()
        robot_pose = tuple(observation.get("robot_pose", [0.0, 0.0, 0.0]))  # type: ignore

        # Always use XML goal for consistency across snapshots
        goal_pose_xml = self._extract_goal_pose_from_xml(xml_path)
        goal_pose = goal_pose_xml

        if goal_pose is None:
            # Fallback to env goal only if XML goal not found
            goal_pose_env = tuple(self._env.get_robot_goal()) if hasattr(self._env, "get_robot_goal") else None
            goal_pose = goal_pose_env

        movable_instances = self._instantiate_movable_objects(observation)

        grids = self._build_grids(self.static_objects, movable_instances)

        goal_cells = self._goal_cells(goal_pose, goal_radius)
        region_map, region_labels = self._compute_regions(grids["dynamic"], robot_pose, goal_cells)
        adjacency, edge_objects = self._build_connectivity(
            grids["dynamic"].copy(),
            region_map,
            region_labels,
            movable_instances,
        )

        movable_metadata: List[MovableObjectRecord] = [
            MovableObjectRecord(
                name=inst.name,
                x=inst.position[0],
                y=inst.position[1],
                theta=self._quaternion_to_yaw(inst.quaternion),
                half_extent_x=inst.half_extent[0],
                half_extent_y=inst.half_extent[1],
            )
            for inst in movable_instances
        ]

        return WavefrontSnapshot(
            resolution=self.resolution,
            bounds=self.bounds,
            uninflated_grid=grids["uninflated"],
            static_grid=grids["static"],
            dynamic_grid=grids["dynamic"],
            region_map=region_map,
            region_labels=region_labels,
            adjacency=adjacency,
            edge_objects=edge_objects,
            robot_pose=robot_pose,
            goal_pose=goal_pose,
            xml_path=xml_path,
            config_path=config_path,
            robot_half_extent=self.robot_half_extent,
            movable_objects=movable_metadata,
        )

    # ------------------------------------------------------------------
    # Grid construction helpers
    # ------------------------------------------------------------------
    def _build_grids(
        self,
        static_objects: Sequence[ObjectInstance],
        movable_objects: Sequence[ObjectInstance],
    ) -> Dict[str, GridArray]:
        shape = (self.grid_width, self.grid_height)
        uninflated = np.full(shape, -1, dtype=np.int16)
        static_grid = np.full(shape, -1, dtype=np.int16)
        dynamic_grid = np.full(shape, -1, dtype=np.int16)

        inflate_x = self.robot_half_extent[0] + self.INFLATION_EPSILON
        inflate_y = self.robot_half_extent[1] + self.INFLATION_EPSILON

        print(f"Robot half-extent: {self.robot_half_extent}")
        print(f"Inflation values: inflate_x={inflate_x}, inflate_y={inflate_y}")

        # Rasterise static objects
        for instance in static_objects:
            self._rasterise_object(instance, instance.half_extent, uninflated)
            inflated_extent = (instance.half_extent[0] + inflate_x, instance.half_extent[1] + inflate_y)
            self._rasterise_object(instance, inflated_extent, static_grid)
            self._rasterise_object(instance, inflated_extent, dynamic_grid)

        # Rasterise movable objects
        for instance in movable_objects:
            self._rasterise_object(instance, instance.half_extent, uninflated)
            inflated_extent = (instance.half_extent[0] + inflate_x, instance.half_extent[1] + inflate_y)
            self._rasterise_object(instance, inflated_extent, static_grid)
            self._rasterise_object(instance, inflated_extent, dynamic_grid)

        return {
            "uninflated": uninflated,
            "static": static_grid,
            "dynamic": dynamic_grid,
        }

    # ------------------------------------------------------------------
    # Goal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _should_override_env_goal(goal_pose: Optional[Tuple[float, float, float]]) -> bool:
        if goal_pose is None:
            return True
        return math.isclose(goal_pose[0], 0.0, abs_tol=1e-6) and math.isclose(goal_pose[1], 0.0, abs_tol=1e-6)

    @staticmethod
    def _extract_goal_pose_from_xml(xml_path: Union[str, Path]) -> Optional[Tuple[float, float, float]]:
        try:
            tree = ET.parse(str(xml_path))
        except (ET.ParseError, FileNotFoundError, OSError):
            return None

        root = tree.getroot()
        candidates: List[Tuple[float, float, float]] = []

        for site in root.iter("site"):
            name = site.get("name", "")
            if not name:
                continue
            if "goal" not in name.lower():
                continue
            pos_str = site.get("pos", "0 0 0")
            try:
                pos = [float(component) for component in pos_str.split()]
            except ValueError:
                continue
            if len(pos) < 2:
                continue
            x, y = pos[0], pos[1]
            theta = 0.0
            if len(pos) >= 3:
                theta = pos[2]
            candidates.append((x, y, theta))

        if not candidates:
            return None

        # Prefer exact "goal" name if present
        for site in root.iter("site"):
            if site.get("name", "").lower() == "goal":
                pos_str = site.get("pos", "0 0 0")
                try:
                    pos = [float(component) for component in pos_str.split()]
                except ValueError:
                    break
                if len(pos) >= 2:
                    return (pos[0], pos[1], 0.0 if len(pos) < 3 else pos[2])
        # Fallback to first candidate
        return candidates[0] if candidates else None

    def _rasterise_object(
        self,
        instance: ObjectInstance,
        half_extent: Tuple[float, float],
        grid: GridArray,
    ) -> None:
        half_w, half_h = half_extent
        if half_w <= 0 or half_h <= 0:
            return

        yaw = self._quaternion_to_yaw(instance.quaternion)
        cos_a = math.cos(yaw)
        sin_a = math.sin(yaw)

        center_x, center_y = instance.position

        # Determine conservative bounding box in grid coordinates
        corners = self._rotated_corners(center_x, center_y, half_w, half_h, cos_a, sin_a)
        min_x = max(0, self._world_to_grid_x(min(pt[0] for pt in corners)))
        max_x = min(self.grid_width - 1, self._world_to_grid_x(max(pt[0] for pt in corners)))
        min_y = max(0, self._world_to_grid_y(min(pt[1] for pt in corners)))
        max_y = min(self.grid_height - 1, self._world_to_grid_y(max(pt[1] for pt in corners)))

        for gx in range(min_x, max_x + 1):
            world_x = self._grid_to_world_x(gx) + 0.5 * self.resolution
            dx = world_x - center_x
            for gy in range(min_y, max_y + 1):
                world_y = self._grid_to_world_y(gy) + 0.5 * self.resolution
                dy = world_y - center_y

                local_x = dx * cos_a + dy * sin_a
                local_y = -dx * sin_a + dy * cos_a

                if abs(local_x) <= half_w and abs(local_y) <= half_h:
                    grid[gx, gy] = -2

    # ------------------------------------------------------------------
    # Region computation
    # ------------------------------------------------------------------
    def _compute_regions(
        self,
        dynamic_grid: GridArray,
        robot_pose: Tuple[float, float, float],
        goal_cells: Set[Tuple[int, int]],
    ) -> Tuple[GridArray, Dict[int, str]]:
        region_map = np.zeros_like(dynamic_grid, dtype=np.int32)
        visited = np.zeros_like(dynamic_grid, dtype=bool)
        width, height = dynamic_grid.shape
        neighbor_offsets = self.NEIGHBOR_OFFSETS

        def bfs(seed: Tuple[int, int]) -> List[Tuple[int, int]]:
            sx, sy = seed
            if visited[sx, sy] or dynamic_grid[sx, sy] == -2:
                return []

            queue: deque[Tuple[int, int]] = deque([seed])
            visited[sx, sy] = True
            cells: List[Tuple[int, int]] = []

            while queue:
                x, y = queue.popleft()
                cells.append((x, y))

                for dx, dy in neighbor_offsets:
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if visited[nx, ny] or dynamic_grid[nx, ny] == -2:
                        continue
                    visited[nx, ny] = True
                    queue.append((nx, ny))

            return cells

        def touches_border(cells: Sequence[Tuple[int, int]]) -> bool:
            if not cells:
                return False
            max_x = self.grid_width - 1
            max_y = self.grid_height - 1
            for gx, gy in cells:
                if gx == 0 or gy == 0 or gx == max_x or gy == max_y:
                    return True
            return False

        region_labels: Dict[int, str] = {}
        region_id = 1

        for gx in range(width):
            for gy in range(height):
                if visited[gx, gy] or dynamic_grid[gx, gy] == -2:
                    continue
                region_cells = bfs((gx, gy))
                if not region_cells:
                    continue
                # Skip tiny regions — they can't host the robot footprint.
                if len(region_cells) < self.MIN_REGION_CELLS:
                    continue
                # Skip the "outside the walls" region: it wraps around the maze and
                # touches the grid boundary (the wavefront's grid is slightly larger
                # than the maze interior). Inner regions are blocked from the grid edge
                # by walls, so they don't trigger this.
                if touches_border(region_cells):
                    continue
                for cell in region_cells:
                    region_map[cell] = region_id
                region_labels[region_id] = f"region_{region_id}"
                region_id += 1

        return region_map, region_labels

    # ------------------------------------------------------------------
    # Connectivity graph construction
    # ------------------------------------------------------------------
    def _build_connectivity(
        self,
        dynamic_grid: GridArray,
        region_map: GridArray,
        region_labels: Dict[int, str],
        movable_objects: Sequence[ObjectInstance],
    ) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, Set[str]]]]:
        adjacency: Dict[str, Set[str]] = {label: set() for label in region_labels.values()}
        edge_objects: Dict[str, Dict[str, Set[str]]] = {
            label: {} for label in region_labels.values()
        }
        inflate_x = self.robot_half_extent[0] + self.INFLATION_EPSILON
        inflate_y = self.robot_half_extent[1] + self.INFLATION_EPSILON
        neighbor_offsets = self.NEIGHBOR_OFFSETS

        for instance in movable_objects:
            inflated_extent = (
                instance.half_extent[0] + inflate_x,
                instance.half_extent[1] + inflate_y,
            )
            footprint = self._collect_footprint_cells(instance, inflated_extent)
            if not footprint:
                continue

            removed_cells: List[Tuple[int, int]] = []
            for cell in footprint:
                if dynamic_grid[cell] == -2:
                    dynamic_grid[cell] = -1
                    removed_cells.append(cell)

            if not removed_cells:
                continue

            removed_set: Set[Tuple[int, int]] = set(removed_cells)
            seed = removed_cells[0]
            queue: deque[Tuple[int, int]] = deque([seed])
            visited: Set[Tuple[int, int]] = {seed}
            connected_regions: Set[int] = set()

            while queue:
                x, y = queue.popleft()
                for dx, dy in neighbor_offsets:
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or nx >= self.grid_width or ny < 0 or ny >= self.grid_height:
                        continue
                    if dynamic_grid[nx, ny] == -2:
                        continue
                    neighbor = (nx, ny)
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)

                    if neighbor in removed_set:
                        queue.append(neighbor)
                    else:
                        region_id = region_map[neighbor]
                        if region_id > 0:
                            connected_regions.add(int(region_id))

            if len(connected_regions) >= 2:
                labels = [region_labels[rid] for rid in connected_regions if rid in region_labels]
                for i in range(len(labels)):
                    for j in range(i + 1, len(labels)):
                        label_a = labels[i]
                        label_b = labels[j]
                        adjacency[label_a].add(label_b)
                        adjacency[label_b].add(label_a)
                        edge_objects.setdefault(label_a, {}).setdefault(label_b, set()).add(instance.name)
                        edge_objects.setdefault(label_b, {}).setdefault(label_a, set()).add(instance.name)

            for cell in removed_cells:
                dynamic_grid[cell] = -2

        return adjacency, edge_objects

    # ------------------------------------------------------------------
    # Object collection
    # ------------------------------------------------------------------
    def _build_static_objects(self, object_info: Dict[str, Dict[str, float]]) -> List[ObjectInstance]:
        static_objects: List[ObjectInstance] = []
        for name, attrs in object_info.items():
            if name == "robot":
                continue
            if {"pos_x", "pos_y", "quat_w", "quat_x", "quat_y", "quat_z"}.issubset(attrs.keys()):
                template = ObjectTemplate(
                    name=name,
                    half_extent=(float(attrs["size_x"]), float(attrs["size_y"])),
                    is_static=True,
                )
                instance = ObjectInstance(
                    template=template,
                    position=(float(attrs["pos_x"]), float(attrs["pos_y"])),
                    quaternion=(
                        float(attrs["quat_w"]),
                        float(attrs["quat_x"]),
                        float(attrs["quat_y"]),
                        float(attrs["quat_z"]),
                    ),
                )
                static_objects.append(instance)
        return static_objects

    def _build_movable_templates(self, object_info: Dict[str, Dict[str, float]]) -> Dict[str, ObjectTemplate]:
        templates: Dict[str, ObjectTemplate] = {}
        for name, attrs in object_info.items():
            if name == "robot":
                continue
            if "size_x" in attrs and "pos_x" not in attrs:
                templates[name] = ObjectTemplate(
                    name=name,
                    half_extent=(float(attrs["size_x"]), float(attrs["size_y"])),
                    is_static=False,
                )
        return templates

    def _instantiate_movable_objects(self, observation: Dict[str, Sequence[float]]) -> List[ObjectInstance]:
        instances: List[ObjectInstance] = []
        for name, template in self.movable_templates.items():
            key = f"{name}_pose"
            if key not in observation:
                continue
            x, y, theta = observation[key]
            quaternion = self._yaw_to_quaternion(theta)
            instances.append(
                ObjectInstance(
                    template=template,
                    position=(float(x), float(y)),
                    quaternion=quaternion,
                )
            )
        return instances

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _collect_footprint_cells(
        self,
        instance: ObjectInstance,
        half_extent: Tuple[float, float],
    ) -> List[Tuple[int, int]]:
        half_w, half_h = half_extent
        if half_w <= 0 or half_h <= 0:
            return []

        yaw = self._quaternion_to_yaw(instance.quaternion)
        cos_a = math.cos(yaw)
        sin_a = math.sin(yaw)
        center_x, center_y = instance.position
        corners = self._rotated_corners(center_x, center_y, half_w, half_h, cos_a, sin_a)
        min_x = max(0, self._world_to_grid_x(min(pt[0] for pt in corners)))
        max_x = min(self.grid_width - 1, self._world_to_grid_x(max(pt[0] for pt in corners)))
        min_y = max(0, self._world_to_grid_y(min(pt[1] for pt in corners)))
        max_y = min(self.grid_height - 1, self._world_to_grid_y(max(pt[1] for pt in corners)))

        cells: List[Tuple[int, int]] = []
        for gx in range(min_x, max_x + 1):
            world_x = self._grid_to_world_x(gx) + 0.5 * self.resolution
            dx = world_x - center_x
            for gy in range(min_y, max_y + 1):
                world_y = self._grid_to_world_y(gy) + 0.5 * self.resolution
                dy = world_y - center_y
                local_x = dx * cos_a + dy * sin_a
                local_y = -dx * sin_a + dy * cos_a
                if abs(local_x) <= half_w and abs(local_y) <= half_h:
                    cells.append((gx, gy))
        return cells

    def _rotated_corners(
        self,
        cx: float,
        cy: float,
        half_w: float,
        half_h: float,
        cos_a: float,
        sin_a: float,
    ) -> List[Tuple[float, float]]:
        corners_local = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]
        corners_world: List[Tuple[float, float]] = []
        for lx, ly in corners_local:
            wx = cx + lx * cos_a - ly * sin_a
            wy = cy + lx * sin_a + ly * cos_a
            corners_world.append((wx, wy))
        return corners_world

    # ------------------------------------------------------------------
    # Grid coordinate utilities
    # ------------------------------------------------------------------
    def _world_to_grid_x(self, world_x: float) -> int:
        return int(math.floor((world_x - self.bounds[0]) / self.resolution))

    def _world_to_grid_y(self, world_y: float) -> int:
        return int(math.floor((world_y - self.bounds[2]) / self.resolution))

    def _grid_to_world_x(self, grid_x: int) -> float:
        return self.bounds[0] + grid_x * self.resolution

    def _grid_to_world_y(self, grid_y: int) -> float:
        return self.bounds[2] + grid_y * self.resolution

    def _valid_coord(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    def _clamp_grid_x(self, grid_x: int) -> int:
        return max(0, min(self.grid_width - 1, grid_x))

    def _clamp_grid_y(self, grid_y: int) -> int:
        return max(0, min(self.grid_height - 1, grid_y))

    def _neighbors_including_center(self, x: int, y: int) -> Iterable[Tuple[int, int]]:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield x + dx, y + dy

    # ------------------------------------------------------------------
    # Goal helpers
    # ------------------------------------------------------------------
    def _goal_cells(
        self,
        goal_pose: Optional[Tuple[float, float, float]],
        radius: float,
    ) -> Set[Tuple[int, int]]:
        if goal_pose is None:
            return set()

        cx = self._clamp_grid_x(self._world_to_grid_x(goal_pose[0]))
        cy = self._clamp_grid_y(self._world_to_grid_y(goal_pose[1]))
        radius_cells = max(1, int(math.ceil(radius / self.resolution)))

        cells: Set[Tuple[int, int]] = set()
        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                gx, gy = cx + dx, cy + dy
                if self._valid_coord(gx, gy):
                    cells.add((gx, gy))
        return cells

    # ------------------------------------------------------------------
    # Quaternion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _quaternion_to_yaw(quaternion: Tuple[float, float, float, float]) -> float:
        w, x, y, z = quaternion
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _yaw_to_quaternion(theta: float) -> Tuple[float, float, float, float]:
        half_theta = 0.5 * theta
        return (math.cos(half_theta), 0.0, 0.0, math.sin(half_theta))

    # ------------------------------------------------------------------
    # Public helper methods for environment generation
    # ------------------------------------------------------------------
    def sample_cell_in_region(
        self,
        region_label: str,
        region_map: GridArray,
        region_labels: Dict[int, str],
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[Tuple[float, float]]:
        """Sample a random free cell in the given region and return world coordinates.
        
        Args:
            region_label: The label of the region to sample from
            region_map: The region map array
            region_labels: Mapping from region ID to label
            rng: Optional random number generator
            
        Returns:
            (x, y) world coordinates, or None if region is empty
        """
        generator = rng or np.random.default_rng()
        
        # Find region ID for this label
        region_id = None
        for rid, label in region_labels.items():
            if label == region_label:
                region_id = rid
                break
        
        if region_id is None:
            return None
        
        # Get all cells in this region
        cells = np.argwhere(region_map == region_id)
        if cells.size == 0:
            return None
        
        # Sample one cell
        idx = generator.choice(cells.shape[0])
        gx = int(cells[idx][0])
        gy = int(cells[idx][1])
        
        # Convert to world coordinates (cell center)
        wx = self._grid_to_world_x(gx) + 0.5 * self.resolution
        wy = self._grid_to_world_y(gy) + 0.5 * self.resolution
        
        return (wx, wy)
    
    def check_cell_clearance(
        self,
        grid_pos: Tuple[int, int],
        clearance_radius: float,
        dynamic_grid: GridArray,
    ) -> bool:
        """Check if a cell has sufficient clearance in all directions.
        
        Args:
            grid_pos: (gx, gy) grid coordinates
            clearance_radius: Required clearance in world units
            dynamic_grid: The occupancy grid
            
        Returns:
            True if cell has sufficient clearance, False otherwise
        """
        gx, gy = grid_pos
        clearance_cells = int(math.ceil(clearance_radius / self.resolution))
        
        # Check all cells within clearance radius
        for dx in range(-clearance_cells, clearance_cells + 1):
            for dy in range(-clearance_cells, clearance_cells + 1):
                nx = gx + dx
                ny = gy + dy
                
                # Check bounds
                if nx < 0 or nx >= self.grid_width or ny < 0 or ny >= self.grid_height:
                    return False
                
                # Check if occupied
                if dynamic_grid[nx, ny] == -2:
                    return False
        
        return True
    
    def sample_cell_with_clearance(
        self,
        region_label: str,
        region_map: GridArray,
        region_labels: Dict[int, str],
        dynamic_grid: GridArray,
        clearance_radius: float,
        rng: Optional[np.random.Generator] = None,
        max_attempts: int = 100,
    ) -> Optional[Tuple[float, float]]:
        """Sample a random cell in a region that has sufficient clearance.
        
        Args:
            region_label: The label of the region to sample from
            region_map: The region map array
            region_labels: Mapping from region ID to label
            dynamic_grid: The occupancy grid
            clearance_radius: Required clearance in world units
            rng: Optional random number generator
            max_attempts: Maximum number of sampling attempts
            
        Returns:
            (x, y) world coordinates, or None if no valid cell found
        """
        generator = rng or np.random.default_rng()
        
        # Find region ID for this label
        region_id = None
        for rid, label in region_labels.items():
            if label == region_label:
                region_id = rid
                break
        
        if region_id is None:
            return None
        
        # Get all cells in this region
        cells = np.argwhere(region_map == region_id)
        if cells.size == 0:
            return None
        
        # Try sampling cells until we find one with clearance
        for _ in range(max_attempts):
            idx = generator.choice(cells.shape[0])
            gx = int(cells[idx][0])
            gy = int(cells[idx][1])
            
            # Check clearance
            if self.check_cell_clearance((gx, gy), clearance_radius, dynamic_grid):
                # Convert to world coordinates (cell center)
                wx = self._grid_to_world_x(gx) + 0.5 * self.resolution
                wy = self._grid_to_world_y(gy) + 0.5 * self.resolution
                return (wx, wy)
        
        return None
