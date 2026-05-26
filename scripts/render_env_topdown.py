#!/usr/bin/env python3
"""Top-down render of a generated NAMO env XML.

Usage: python3 render_env_topdown.py <xml_path> [out.png] [--size W H]
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')

import mujoco
import numpy as np
from PIL import Image


def compute_bounds(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    walls_body = root.find(".//body[@name='walls']")
    xs, ys = [], []
    if walls_body is not None:
        for g in walls_body.findall('geom'):
            pos = [float(v) for v in g.get('pos', '0 0 0').split()]
            size = [float(v) for v in g.get('size', '0.1 0.1 0.1').split()]
            xs += [pos[0] - size[0], pos[0] + size[0]]
            ys += [pos[1] - size[1], pos[1] + size[1]]
    if not xs:
        xs = [-1, 1]; ys = [-1, 1]
    return min(xs), max(xs), min(ys), max(ys)


def render(xml_path, out_path, width, height):
    with open(xml_path) as f:
        xml = f.read()
    visual = f'<visual><global offwidth="{width}" offheight="{height}"/></visual>\n'
    xml = xml.replace('<mujoco model="generated_environment">',
                      f'<mujoco model="generated_environment">\n{visual}')

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    xmin, xmax, ymin, ymax = compute_bounds(xml_path)
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    span = max(xmax - xmin, ymax - ymin) * 1.05

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [cx, cy, 0.0]
    cam.distance = span  # orthographic-ish from above
    cam.elevation = -90.0
    cam.azimuth = 0.0

    opt = mujoco.MjvOption()
    renderer.update_scene(data, camera=cam, scene_option=opt)
    img = renderer.render()
    Image.fromarray(img).save(out_path)
    print(f"wrote {out_path}  ({width}x{height}, bounds=[{xmin:.2f},{xmax:.2f}]x[{ymin:.2f},{ymax:.2f}])")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('xml')
    p.add_argument('out', nargs='?', default=None)
    p.add_argument('--size', type=int, nargs=2, default=[800, 800], metavar=('W', 'H'))
    args = p.parse_args()
    out = args.out or os.path.splitext(os.path.basename(args.xml))[0] + '.png'
    render(args.xml, out, args.size[0], args.size[1])


if __name__ == '__main__':
    main()
