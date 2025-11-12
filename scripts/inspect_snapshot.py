#!/usr/bin/env python3
"""Inspect a Wavefront snapshot folder produced by export_wavefront_snapshot.

This script reads the metadata JSON and NumPy grids and reports:
- robot and goal world positions
- their mapped grid indices
- dynamic/static/uninflated grid values at those cells
- nearby cell values
- which movable objects (from metadata) would contain the sampled world points

Read-only diagnostics only; does not modify any files.
"""
import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np


def world_to_grid(world_x: float, world_y: float, bounds: List[float], resolution: float) -> Tuple[int, int]:
    gx = int(math.floor((world_x - bounds[0]) / resolution))
    gy = int(math.floor((world_y - bounds[2]) / resolution))
    return gx, gy


def point_in_rotated_rect(px: float, py: float, cx: float, cy: float, half_w: float, half_h: float, theta: float) -> bool:
    # rotate point into rectangle local frame
    dx = px - cx
    dy = py - cy
    cos_a = math.cos(-theta)
    sin_a = math.sin(-theta)
    local_x = dx * cos_a - dy * sin_a
    local_y = dx * sin_a + dy * cos_a
    return abs(local_x) <= half_w + 1e-9 and abs(local_y) <= half_h + 1e-9


def main():
    p = argparse.ArgumentParser(description="Inspect a wavefront snapshot folder")
    p.add_argument("snapshot_dir", help="Path to snapshot folder (metadata.json + *_grid.npy files)")
    args = p.parse_args()

    d = Path(args.snapshot_dir)
    meta_path = d / (d.name + "_metadata.json")
    if not meta_path.exists():
        print(f"Metadata not found: {meta_path}")
        return

    meta = json.load(open(meta_path))
    resolution = float(meta["resolution"])
    bounds = meta["bounds"]

    def load(name: str):
        p = d / (d.name + f"_{name}.npy")
        if not p.exists():
            return None
        return np.load(p)

    dynamic = load("dynamic_grid")
    static = load("static_grid")
    uninflated = load("uninflated_grid")
    region_map = load("region_map")

    robot = meta.get("robot_pose")
    goal = meta.get("goal_pose")

    print(f"Snapshot: {d}")
    print(f" resolution={resolution} bounds={bounds} grid_shape={meta.get('grid_shape')}\n")

    def report_point(label: str, pose):
        if pose is None:
            print(f"{label}: <none>")
            return
        wx, wy = pose[0], pose[1]
        gx, gy = world_to_grid(wx, wy, bounds, resolution)
        print(f"{label}: world=({wx:.6f},{wy:.6f}) -> grid=({gx},{gy})")
        def val(arr):
            if arr is None:
                return None
            if gx < 0 or gy < 0 or gx >= arr.shape[0] or gy >= arr.shape[1]:
                return 'OOB'
            return int(arr[gx, gy])

        print(f"  dynamic={val(dynamic)} static={val(static)} uninflated={val(uninflated)} region_map={val(region_map)}")
        # nearby cells
        if dynamic is not None:
            neigh = []
            for dx in range(-2, 3):
                row = []
                for dy in range(-2, 3):
                    nx, ny = gx + dx, gy + dy
                    if nx < 0 or ny < 0 or nx >= dynamic.shape[0] or ny >= dynamic.shape[1]:
                        row.append('X')
                    else:
                        row.append(int(dynamic[nx, ny]))
                neigh.append(row)
            print("  nearby dynamic grid (5x5):")
            for r in neigh:
                print("   ", " ".join(str(x) for x in r))

    report_point("robot", robot)
    report_point("goal", goal)

    # Check movable objects overlap
    mov = meta.get("movable_objects", [])
    if mov:
        print("\nMovable objects and overlap with robot/goal:")
        for obj in mov:
            name = obj.get("name")
            cx = obj.get("x")
            cy = obj.get("y")
            hx = obj.get("half_extent_x")
            hy = obj.get("half_extent_y")
            theta = obj.get("theta")
            r_in = point_in_rotated_rect(robot[0], robot[1], cx, cy, hx, hy, theta) if robot else False
            g_in = point_in_rotated_rect(goal[0], goal[1], cx, cy, hx, hy, theta) if goal else False
            print(f"  {name}: center=({cx:.3f},{cy:.3f}) half_ext=({hx:.3f},{hy:.3f}) theta={theta:.3f} robot_in={r_in} goal_in={g_in}")


if __name__ == '__main__':
    main()
