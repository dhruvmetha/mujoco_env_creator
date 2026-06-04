#!/usr/bin/env python3
"""Emit the v3 aug9_car joblist.

100K envs across 10 aug9_car_v3 templates → 10,000 envs/template.
Templates are rescaled aug9 layouts (factor 0.21 → 0.49×0.49 m arena).
Object size range: 6-16 cm (full side length), matches v3 feb_car.
Output dir: /scratch/dm1487/datasets/car_envs/v3/aug9_car/

Run:
    python scripts/generate_v3_aug9_joblist.py > scripts/v3_aug9_joblist.txt
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = "/scratch/dm1487/envs/namo/bin/python"
SCRIPT = str(REPO / "generate_envs.py")
NAMO_CONFIG = "/cache/home/dm1487/projects/namo/namo_cpp/config/namo_config_complete_skill15_car_1x.yaml"
OUTPUT_ROOT = "/scratch/dm1487/datasets/car_envs/v3"

AUG9_V3_TEMPLATES_DIR = REPO / "templates" / "aug9_car_v3"
NUM_OBJECTS_JSON = str(AUG9_V3_TEMPLATES_DIR / "num_objects.json")

# Budget
TOTAL_ENVS = 100_000
SEED_BASE = 300_000_000   # disjoint from v2 (100M) and v3_feb (200M)
SEED_STRIDE_PER_TEMPLATE = 100_000   # plenty of headroom — 10K envs/template
SEED_STRIDE_PER_CHUNK = 2_000        # > envs/chunk so seeds never collide
CHUNKS_PER_TEMPLATE = 10             # 10 templates × 10 chunks = 100 commands

OBJ_SIZE_MIN = 0.06
OBJ_SIZE_MAX = 0.16


def aug9_v3_templates():
    out = []
    for sub in sorted(AUG9_V3_TEMPLATES_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for xml in sorted(sub.glob("*.xml")):
            out.append(xml)
    return out


def main():
    templates = aug9_v3_templates()
    n_templates = len(templates)
    if n_templates == 0:
        raise SystemExit(f"No templates found under {AUG9_V3_TEMPLATES_DIR}")
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
            seed = SEED_BASE + t_idx * SEED_STRIDE_PER_TEMPLATE + c_idx * SEED_STRIDE_PER_CHUNK
            cmd = (
                f"{PYTHON} {SCRIPT} {xml} "
                f"--namo-config {NAMO_CONFIG} "
                f"--num-envs {n_envs} "
                f"--output-dir {OUTPUT_ROOT}/aug9_car "
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
