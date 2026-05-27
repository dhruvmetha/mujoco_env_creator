#!/usr/bin/env python3
"""Emit the v2 joblist (one shell command per chunk) for xargs.

Layout:
  aug9_car  (10 templates):  6 chunks x 5000 envs    = 300,000 envs total
  feb_car  (422 templates):  1 chunk  x 1660 envs    = 700,520 envs total
  Grand total                                       ~1,000,520 logical envs

Seed allocation (all >= 200000, disjoint per template/chunk):
  aug9_car:  200000 + t * 1_000_000 + j * 10_000   (chunks span 5000 seeds each, 5000 slack)
  feb_car: 100000000 + t * 10_000                  (each template gets 10000 seeds; uses 1660)
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = "/common/users/dm1487/envs/mjxrl/bin/python3"
SCRIPT = str(REPO / "generate_envs.py")
NAMO_CONFIG = "/common/home/dm1487/robotics_research/ktamp/namo/config/namo_config_complete_skill15_car_1x.yaml"
OUTPUT_ROOT = "/common/users/dm1487/corl2026/namo/envs/v2"

AUG9_TEMPLATES_DIR = REPO / "templates" / "aug9_car"
FEB_TEMPLATES_DIR = REPO / "templates" / "feb_car"

# Per-template env budgets
AUG9_ENVS_PER_TEMPLATE = 30000
AUG9_ENVS_PER_CHUNK = 5000   # => 6 chunks per template
FEB_ENVS_PER_TEMPLATE = 1660

AUG9_SEED_BASE = 200_000
AUG9_TEMPLATE_STRIDE = 1_000_000
AUG9_CHUNK_STRIDE = 10_000

FEB_SEED_BASE = 100_000_000
FEB_TEMPLATE_STRIDE = 10_000


def aug9_templates():
    # ordered like the existing num_objects.json: set1/benchmark_1..5, set2/benchmark_1..5
    out = []
    for set_name in ("set1", "set2"):
        for xml in sorted((AUG9_TEMPLATES_DIR / set_name).glob("*.xml")):
            out.append(xml)
    return out


def feb_templates():
    out = []
    for bucket in ("straight0_pts2", "straight50_pts2", "straight100_pts2"):
        for xml in sorted((FEB_TEMPLATES_DIR / bucket).glob("*.xml")):
            out.append(xml)
    return out


# Per-type sizing args, reverse-engineered from v1 output XMLs.
# aug9_car: arena ~2.3m wide. --robot-scale 0.233 triggers the code's car
#   special case (line 1512 of generate_envs.py): object_size 0.09-0.33m,
#   object_half_height 0.05m, goal_size 0.047m, clearance_radius 0, min_goal_distance 0.
# feb_car: arena 0.49 x 0.775m. Explicit overrides match v1's 4-10cm obstacles.
AUG9_SIZING = "--robot-scale 0.233"
FEB_SIZING = (
    "--object-size-range 0.04 0.10 "
    "--object-half-height 0.05 "
    "--goal-size 0.02 "
    "--clearance-radius 0.0 "
    "--min-goal-distance 0.0"
)


def emit_command(template_xml, num_envs, start_seed, run_id_offset, output_dir, num_objects_json, sizing):
    return (
        f"{PYTHON} {SCRIPT} "
        f"{template_xml} "
        f"--namo-config {NAMO_CONFIG} "
        f"--num-envs {num_envs} "
        f"--output-dir {output_dir} "
        f"--num-workers 1 "
        f"--start-seed {start_seed} "
        f"--run-id-offset {run_id_offset} "
        f"--num-objects-json {num_objects_json} "
        f"{sizing}"
    )


def main():
    lines = []

    # aug9_car
    aug9_out = f"{OUTPUT_ROOT}/aug9_car"
    aug9_json = str(AUG9_TEMPLATES_DIR / "num_objects.json")
    aug9_chunks = AUG9_ENVS_PER_TEMPLATE // AUG9_ENVS_PER_CHUNK
    for t_idx, xml in enumerate(aug9_templates()):
        for chunk in range(aug9_chunks):
            seed = AUG9_SEED_BASE + t_idx * AUG9_TEMPLATE_STRIDE + chunk * AUG9_CHUNK_STRIDE
            offset = chunk * AUG9_ENVS_PER_CHUNK
            lines.append(emit_command(
                str(xml), AUG9_ENVS_PER_CHUNK, seed, offset, aug9_out, aug9_json,
                AUG9_SIZING,
            ))

    # feb_car
    feb_out = f"{OUTPUT_ROOT}/feb_car"
    feb_json = str(FEB_TEMPLATES_DIR / "num_objects.json")
    for t_idx, xml in enumerate(feb_templates()):
        seed = FEB_SEED_BASE + t_idx * FEB_TEMPLATE_STRIDE
        lines.append(emit_command(
            str(xml), FEB_ENVS_PER_TEMPLATE, seed, 0, feb_out, feb_json,
            FEB_SIZING,
        ))

    out_path = REPO / "scripts" / "v2_joblist.txt"
    out_path.write_text("\n".join(lines) + "\n")
    aug_count = len(aug9_templates()) * aug9_chunks
    feb_count = len(feb_templates())
    total_envs = (
        len(aug9_templates()) * AUG9_ENVS_PER_TEMPLATE
        + len(feb_templates()) * FEB_ENVS_PER_TEMPLATE
    )
    print(f"wrote {out_path}")
    print(f"  aug9 jobs: {aug_count} ({len(aug9_templates())} templates x {aug9_chunks} chunks x {AUG9_ENVS_PER_CHUNK} envs)")
    print(f"  feb  jobs: {feb_count} ({len(feb_templates())} templates x 1 chunk x {FEB_ENVS_PER_TEMPLATE} envs)")
    print(f"  total jobs: {len(lines)}")
    print(f"  total envs: {total_envs}")


if __name__ == "__main__":
    main()
