## mujoco_env_creator — README

This directory contains tools to generate complete MuJoCo environments from template XMLs by
randomly placing movable obstacles and then using a wavefront-based occupancy analysis to
place the robot and goal in connected, collision-free regions.

This README documents the new/updated usage patterns and verification steps for the
wavefront exporter and the environment generation pipeline.

## What this does
- Reads a template XML (walls/maze present, obstacles absent).
- Randomly places movable obstacles (boxes) while avoiding walls and other obstacles.
- Runs a MuJoCo simulation of the scene and builds a wavefront snapshot (occupancy grids,
  connected regions, adjacency graph).
- Samples robot and goal positions that satisfy clearance and minimum separation constraints.
- Writes final XML(s) with obstacles and placed robot/goal into the requested output directory.

## Important files
- `generate_envs.py` — main driver. Supports single generation, pair generation, and parallel/batch generation.
- `wavefront_snapshot.py` — exporter used to compute occupancy grids, region maps, sampling helpers, and connectivity.
  - Note: rasterisation now tests the cell *center* to align with sampling, avoiding cases where a sampled point
    (cell center) would lie inside an object despite the raster test marking the cell free.
- `scripts/` — helper scripts for running batches and SLURM submissions (examples included in the repo).

## Quick single-run (debug)
This runs the generator for a single environment using a template XML and the NAMO config YAML. It saves a temporary
XML (with obstacles only) into a debug folder so you can inspect simulation inputs, and writes the final XML to the
output directory.

Replace the paths below with your template and config.

```bash
python3 generate_envs.py \
  path/to/template.xml \
  --namo-config path/to/namo_config.yaml \
  --num-envs 1 \
  --output-dir /tmp/generated_envs \
  --num-workers 1 \
  --start-seed 0
```

After the run, check the debug XML (intermediate obstacles-only XML):

And the final generated file (example):

  /tmp/generated_envs/<template_base>/run_0000/env_0000.xml

## Where outputs go
- Final outputs are written to the `--output-dir` you pass to `generate_envs.py`.
  The layout is:

  output_dir/<template_base>/run_<NNNN>/env_XXXX.xml  (or _pair_YYY.xml for pair outputs)

Use that temporary XML if you want to re-run the exact simulated scene or to inspect the obstacle placements.

## Quick verification / debugging steps
1. Re-run the generator for the same template/seed (use `--start-seed` or pass a known seed) so outputs are reproducible.
2. Use the inspector script in `mujoco_env_creator/scripts/inspect_snapshot.py` (if present) to examine the saved
   wavefront snapshot produced during `build_snapshot()` — it prints the robot/world coordinates, grid indices, and
   per-grid values (uninflated/static/dynamic) for a cell of interest.
3. Things to check if you see a robot/goal inside geometry:
   - Confirm `dynamic_grid`, `static_grid`, and `uninflated_grid` agree at the sampled cell. They should not disagree
     for the cell center after the recent rasterisation alignment change.
   - If `dynamic_grid` is free but `static_grid` is occupied, that can indicate a persistent mutation or that you used
     an alternate exporter (the repo contains two exporter copies; ensure the intended one is on PYTHONPATH).

## Notes about the wavefront exporter
- Grids use the following conventions:
  - -2 => occupied
  - -1 => free
- `region_map` labels connected free regions, `region_labels` maps region ids to labels, and `adjacency` indicates
  reachable neighboring regions.
- Sampling helpers:
  - `sample_cell_in_region()` returns a world point at the cell *center*.
  - `sample_cell_with_clearance()` additionally verifies clearance against the `dynamic_grid`.

## Common troubleshooting
- Missing NAMO Python bindings: Set PYTHONPATH so `namo_rl` import succeeds (the generation script will exit with a message
  if namo_rl is unavailable).
- If your placements still overlap geometry after these checks, re-run the generator with `debug=True` or inspect the
  temporary XML in `test_debug` and run the snapshot inspector on the saved snapshot to see grid values and sampling details.

## Next steps / validation checklist
1. Run a single controlled generation with a known seed and template.
2. Inspect `test_debug/env_XXXX_temp.xml` and run the snapshot inspector to confirm `dynamic_grid`/`static_grid` agree on the
   sampled cell center.
3. When satisfied, run the batch script or SLURM submission to produce many environments.

## Contact / provenance
If you need more help with a failing example, provide:
- The template XML path used
- The seed (start seed) and the generated temp XML path (from `test_debug`)
- The inspector output (if available)

This README was generated to document the new/updated generation and snapshot behavior after rasterisation sampling fixes.
