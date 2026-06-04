#!/usr/bin/env python3
"""Emit the v3 feb_car TEST joblist.

5K test envs across 422 feb_car templates -> ~12 envs/template.
Disjoint from the training set (which uses SEED_BASE 200M and writes to
.../v3/feb_car). Test seeds start at 400M and write to .../v3/test/feb_car.
Object size range: 6-16 cm (full side length), matches the training set.
Output dir: /scratch/dm1487/datasets/car_envs/v3/test/feb_car/

Run:
    python scripts/generate_v3_feb_test_joblist.py > scripts/v3_feb_test_joblist.txt
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = "/scratch/dm1487/envs/namo/bin/python"
SCRIPT = str(REPO / "generate_envs.py")
NAMO_CONFIG = "/cache/home/dm1487/projects/namo/namo_cpp/config/namo_config_complete_skill15_car_1x.yaml"
OUTPUT_ROOT = "/scratch/dm1487/datasets/car_envs/v3"

FEB_TEMPLATES_DIR = REPO / "templates" / "feb_car"
NUM_OBJECTS_JSON = str(FEB_TEMPLATES_DIR / "num_objects.json")

# Budget
TOTAL_ENVS = 5_000
SEED_BASE = 400_000_000   # disjoint from v2 (100M), v3_feb train (200M), v3_aug9 train (300M)
SEED_STRIDE_PER_TEMPLATE = 10_000   # plenty of headroom per template
CHUNKS_PER_TEMPLATE = 1             # ~12 envs/template — one command per template (422 commands)

OBJ_SIZE_MIN = 0.06
OBJ_SIZE_MAX = 0.16


def feb_templates():
    out = []
    for sub in sorted(FEB_TEMPLATES_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for xml in sorted(sub.glob("*.xml")):
            out.append(xml)
    return out


def main():
    templates = feb_templates()
    n_templates = len(templates)
    envs_per_template = TOTAL_ENVS // n_templates
    remainder = TOTAL_ENVS - envs_per_template * n_templates
    for t_idx, xml in enumerate(templates):
        n_template = envs_per_template + (1 if t_idx < remainder else 0)
        base_chunk = n_template // CHUNKS_PER_TEMPLATE
        chunk_remainder = n_template - base_chunk * CHUNKS_PER_TEMPLATE
        cum_offset = 0
        for c_idx in range(CHUNKS_PER_TEMPLATE):
            n_envs = base_chunk + (1 if c_idx < chunk_remainder else 0)
            if n_envs <= 0:
                continue
            seed = SEED_BASE + t_idx * SEED_STRIDE_PER_TEMPLATE + c_idx
            cmd = (
                f"{PYTHON} {SCRIPT} {xml} "
                f"--namo-config {NAMO_CONFIG} "
                f"--num-envs {n_envs} "
                f"--output-dir {OUTPUT_ROOT}/test/feb_car "
                f"--num-workers 1 "
                f"--start-seed {seed} "
                f"--run-id-offset {cum_offset} "
                f"--num-objects-json {NUM_OBJECTS_JSON} "
                f"--object-size-range {OBJ_SIZE_MIN} {OBJ_SIZE_MAX} "
                f"--object-half-height 0.05 "
                f"--goal-size 0.02 "
                f"--clearance-radius 0.0 "
                f"--min-goal-distance 0.0"
            )
            print(cmd)
            cum_offset += n_envs


if __name__ == "__main__":
    main()
