#!/usr/bin/env python3
"""Top-down 2D rendering of a NAMO env XML using matplotlib.

Parses walls, obstacles, robot, and goal from the XML and draws them as
simple 2D shapes. No camera, no shadows, no perspective distortion.

Usage:
  python3 render_env_topdown.py <xml_path> [out.png] [--inches W H] [--dpi 100]
"""
import argparse
import math
import os
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def _parse_floats(s, n=None):
    vals = [float(v) for v in s.split()]
    return vals if n is None else vals[:n]


def parse_env(xml_path):
    """Pull walls, obstacles, robot, goal out of the env XML."""
    root = ET.parse(xml_path).getroot()

    walls = []
    for g in root.findall(".//body[@name='walls']/geom"):
        pos = _parse_floats(g.get('pos', '0 0 0'), 3)
        size = _parse_floats(g.get('size', '0 0 0'), 3)
        walls.append({'cx': pos[0], 'cy': pos[1],
                      'w': 2 * size[0], 'h': 2 * size[1]})

    obstacles = []
    for body in root.findall(".//body"):
        name = body.get('name', '')
        if not name.startswith('obstacle_'):
            continue
        g = body.find('geom')
        if g is None:
            continue
        pos = _parse_floats(g.get('pos', '0 0 0'), 3)
        size = _parse_floats(g.get('size', '0 0 0'), 3)
        # rotation around Z (degrees); template uses "0 0 yaw"
        euler = _parse_floats(g.get('euler', '0 0 0'), 3)
        yaw_deg = euler[2] if len(euler) >= 3 else 0.0
        obstacles.append({'cx': pos[0], 'cy': pos[1],
                          'w': 2 * size[0], 'h': 2 * size[1],
                          'yaw': yaw_deg})

    # Robot: point robot uses body[@name='robot']/geom; car uses body[@name='car']
    robot = None
    point_geom = root.find(".//body[@name='robot']/geom[@name='robot']")
    if point_geom is not None:
        pos = _parse_floats(point_geom.get('pos', '0 0 0'), 3)
        size = _parse_floats(point_geom.get('size', '0.15'), 1)
        robot = {'cx': pos[0], 'cy': pos[1], 'r': size[0], 'kind': 'point'}
    else:
        car_body = root.find(".//body[@name='car']")
        if car_body is not None:
            pos = _parse_floats(car_body.get('pos', '0 0 0'), 3)
            # Approximate car footprint from chassis collision geoms: total
            # length is two 0.035m chassis halves; width is 0.07m.
            robot = {'cx': pos[0], 'cy': pos[1],
                     'w': 0.07, 'h': 0.07, 'kind': 'car'}

    goal = None
    goal_site = root.find(".//site[@name='goal']")
    if goal_site is not None:
        pos = _parse_floats(goal_site.get('pos', '0 0 0'), 3)
        size = _parse_floats(goal_site.get('size', '0.05'), 1)
        goal = {'cx': pos[0], 'cy': pos[1], 'r': size[0]}

    # Arena bounds from walls (fall back to obstacle extents if no walls)
    if walls:
        xs = [w['cx'] - w['w']/2 for w in walls] + [w['cx'] + w['w']/2 for w in walls]
        ys = [w['cy'] - w['h']/2 for w in walls] + [w['cy'] + w['h']/2 for w in walls]
    else:
        xs = [-1, 1]; ys = [-1, 1]
    bounds = (min(xs), max(xs), min(ys), max(ys))

    return walls, obstacles, robot, goal, bounds


def render(xml_path, out_path, inches=None, dpi=120, pad_frac=0.02):
    walls, obstacles, robot, goal, bounds = parse_env(xml_path)
    xmin, xmax, ymin, ymax = bounds
    arena_w, arena_h = xmax - xmin, ymax - ymin

    # Default figure size: match arena aspect ratio, longer side = 5 inches.
    if inches is None:
        longest = max(arena_w, arena_h)
        in_w = 5.0 * arena_w / longest
        in_h = 5.0 * arena_h / longest
    else:
        in_w, in_h = inches

    fig, ax = plt.subplots(figsize=(in_w, in_h), dpi=dpi)
    ax.set_facecolor('#e8e8e8')

    # Walls
    for w in walls:
        ax.add_patch(Rectangle(
            (w['cx'] - w['w']/2, w['cy'] - w['h']/2),
            w['w'], w['h'],
            facecolor='#444444', edgecolor='none', zorder=2))

    # Obstacles (yellow, rotated)
    for o in obstacles:
        rect = Rectangle(
            (-o['w']/2, -o['h']/2), o['w'], o['h'],
            facecolor='#ffd84a', edgecolor='#aa8800', linewidth=0.8, zorder=3)
        t = matplotlib.transforms.Affine2D() \
            .rotate_deg(o['yaw']) \
            .translate(o['cx'], o['cy']) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    # Goal (green dot)
    if goal is not None:
        ax.add_patch(Circle(
            (goal['cx'], goal['cy']), goal['r'],
            facecolor='#1bbf3a', edgecolor='#0a6b1d', linewidth=1.2,
            alpha=0.85, zorder=4))

    # Robot (red dot for point, red square for car)
    if robot is not None:
        if robot['kind'] == 'point':
            ax.add_patch(Circle(
                (robot['cx'], robot['cy']), robot['r'],
                facecolor='#e63b3b', edgecolor='#7a1010', linewidth=1.2, zorder=5))
        else:  # car
            ax.add_patch(Rectangle(
                (robot['cx'] - robot['w']/2, robot['cy'] - robot['h']/2),
                robot['w'], robot['h'],
                facecolor='#e63b3b', edgecolor='#7a1010', linewidth=1.2, zorder=5))

    # Frame
    pad_x = max(arena_w, arena_h) * pad_frac
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_x, ymax + pad_x)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    plt.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"wrote {out_path}  ({in_w:.2f}x{in_h:.2f}in @ {dpi}dpi, "
          f"arena {arena_w:.2f}x{arena_h:.2f}m, "
          f"{len(obstacles)} obstacles)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('xml')
    p.add_argument('out', nargs='?', default=None)
    p.add_argument('--inches', type=float, nargs=2, default=None,
                   metavar=('W', 'H'),
                   help='Figure size in inches. Default: aspect-matched to arena.')
    p.add_argument('--dpi', type=int, default=120)
    args = p.parse_args()
    out = args.out or os.path.splitext(os.path.basename(args.xml))[0] + '.png'
    render(args.xml, out, inches=args.inches, dpi=args.dpi)


if __name__ == '__main__':
    main()
