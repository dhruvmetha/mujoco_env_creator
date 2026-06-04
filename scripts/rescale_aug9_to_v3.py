#!/usr/bin/env python3
"""Rescale aug9_car arena templates from 2.33 × 2.33 m to v3 scale.

aug9_car templates have walls at ±1.165 m (full arena = 2.33 m). We multiply
all wall positions/sizes AND the car's initial body position by FACTOR to map
to a smaller arena. Car geometry itself stays unchanged — same 7 cm physical
robot, same primitives, just operating in a smaller workspace.

Default FACTOR = 0.21 → arena becomes ~0.49 × 0.49 m (matches v3 feb_car's
short axis; square aspect — different from feb's rectangular 0.49 × 0.775).

Usage:
    python rescale_aug9_to_v3.py \\
        --src-dir templates/aug9_car \\
        --dst-dir templates/aug9_car_v3 \\
        --factor 0.21
"""
import argparse
import re
import shutil
import sys
from pathlib import Path


# Scale ONLY:
#  - <geom name="wall_*"> inside <body name="walls"> — wall pos + size
#  - <body name="car" pos="..."> — car spawn position (XY only; Z stays 0.01)
#  - <site name="goal" pos="..."> — goal location if present
#
# DO NOT scale:
#  - car body geoms (chassis, wheels, marker)
#  - floor geom (plane)
#  - num_objects.json (controls obstacle count, not geometry)


def scale_xyz(text: str, factor: float, *, scale_z: bool = True) -> str:
    """Multiply each whitespace-separated coord in `text` by factor."""
    parts = text.split()
    if len(parts) != 3:
        return text
    x, y, z = (float(p) for p in parts)
    nx = x * factor
    ny = y * factor
    nz = z * factor if scale_z else z
    return f"{nx:.6f} {ny:.6f} {nz:.6f}"


def rescale_one_xml(src: Path, dst: Path, factor: float) -> None:
    raw = src.read_text()
    out = raw

    # Walk through wall geoms inside <body name="walls">. They look like:
    #   <geom name="wall_N" ... pos="X Y Z" ... size="SX SY SZ" type="box" />
    # Scale pos (all 3 axes) and size (all 3 axes incl. wall height for physics
    # consistency — keep Z untouched for size so wall height stays 8cm in real-world).
    # Actually wall height (Z extent) is intentional 0.08m for visual + the robot is
    # always at Z<0.1, so keep height untouched.
    # Wall thickness floor — match v3 feb_car's 1 cm wall thickness.
    # Walls are boxes with one long axis and one short ("thickness") axis.
    # We scale the LONG axis proportionally with the arena, but clamp the
    # thickness axis to MIN_WALL_HALF (1 cm half-extent = 2 cm full thickness)
    # so walls don't dwindle to sub-millimeter at the new scale.
    MIN_WALL_HALF = 0.010   # 1 cm half-extent (matches v3 feb_car boundary walls)

    def scale_wall_size(s: str) -> str:
        sx, sy, sz = (float(p) for p in s.split())
        # Identify thickness axis = the smaller of sx, sy
        if sx <= sy:
            # X is thickness — keep at MIN_WALL_HALF, scale Y
            nsx = MIN_WALL_HALF
            nsy = sy * factor
        else:
            # Y is thickness — keep at MIN_WALL_HALF, scale X
            nsx = sx * factor
            nsy = MIN_WALL_HALF
        return f"{nsx:.6f} {nsy:.6f} {sz:.6f}"

    def repl_wall(m: re.Match) -> str:
        attrs = m.group(0)
        # pos: scale X, Y only — keep Z (the wall's height-above-floor)
        attrs = re.sub(
            r'pos="([^"]+)"',
            lambda pm: f'pos="{scale_xyz(pm.group(1), factor, scale_z=False)}"',
            attrs, count=1,
        )
        # size: scale long axis proportionally; clamp thickness axis to MIN_WALL_HALF
        attrs = re.sub(
            r'size="([^"]+)"',
            lambda sm: f'size="{scale_wall_size(sm.group(1))}"',
            attrs, count=1,
        )
        return attrs

    out = re.sub(r'<geom name="wall_\d+"[^/]+/>', repl_wall, out)

    # Car body initial position: scale X, Y; keep Z (0.01 m = 1cm clearance)
    out = re.sub(
        r'(<body name="car"\s+pos=)"([^"]+)"',
        lambda m: f'{m.group(1)}"{scale_xyz(m.group(2), factor, scale_z=False)}"',
        out, count=1,
    )

    # Goal site (if present): scale X, Y, keep Z
    out = re.sub(
        r'(<site name="goal"[^/]*pos=)"([^"]+)"',
        lambda m: f'{m.group(1)}"{scale_xyz(m.group(2), factor, scale_z=False)}"',
        out,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, required=True,
                    help="aug9_car templates (e.g. mujoco_env_creator/templates/aug9_car)")
    ap.add_argument("--dst-dir", type=Path, required=True,
                    help="output dir (e.g. templates/aug9_car_v3)")
    ap.add_argument("--factor", type=float, default=0.23,
                    help="uniform scale factor (default 0.23 → ~0.54m arena from 2.35m)")
    args = ap.parse_args()

    if args.dst_dir.exists():
        print(f"ERROR: destination {args.dst_dir} already exists; remove first.", file=sys.stderr)
        sys.exit(1)

    src_xmls = sorted(args.src_dir.rglob("*.xml"))
    print(f"Found {len(src_xmls)} XMLs in {args.src_dir}")

    for src in src_xmls:
        rel = src.relative_to(args.src_dir)
        dst = args.dst_dir / rel
        rescale_one_xml(src, dst, args.factor)
        print(f"  {rel}")

    # Copy num_objects.json untouched (it controls obstacle count, not geometry)
    nobj = args.src_dir / "num_objects.json"
    if nobj.exists():
        shutil.copy(nobj, args.dst_dir / "num_objects.json")
        print(f"  num_objects.json (copied)")

    print(f"\nDone. Outputs at {args.dst_dir}")
    print(f"Scale factor: {args.factor} → arena {2.33 * args.factor:.3f} m × {2.33 * args.factor:.3f} m")


if __name__ == "__main__":
    main()
