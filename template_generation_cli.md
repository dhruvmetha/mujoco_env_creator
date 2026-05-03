# MuJoCo Maze Template Generation CLI

This script generates MuJoCo XML maze environments using customizable parameters. It adapts maze generation algorithms for robotics and RL research.

## Usage

```bash
python template_generation.py [--height HEIGHT] [--width WIDTH] [--deletion_rate RATE] [--num_mazes N] [--output_dir PATH]
```

## Arguments

- `--height` (int): Height of the maze (default: 11, must be ≥ 3)
- `--width` (int): Width of the maze (default: 11, must be ≥ 3)
- `--deletion_rate` (float): Wall deletion probability (0.0–1.0). If negative, generates incremental rates (default: -1)
- `--num_mazes` (int): Number of mazes to generate (default: 1, must be ≥ 1)
- `--output_dir` (str): Output directory for generated XML files (default: ../generated_templates)

## Examples

- Generate a single 11x11 maze:
  ```bash
  python template_generation.py
  ```
- Generate 5 mazes of size 15x15 with 20% wall deletion:
  ```bash
  python template_generation.py --height 15 --width 15 --deletion_rate 0.2 --num_mazes 5
  ```
- Generate mazes with incremental deletion rates:
  ```bash
  python template_generation.py --num_mazes 30 --deletion_rate -1
  ```

## Notes

- If `deletion_rate` is negative, mazes are generated with increasing wall deletion rates (e.g., 10%, 20%, ...).
- All arguments are optional and have sensible defaults.
- Output files are saved in the working directory or as specified in the script.

---
For further details, see the script docstrings or contact the maintainers.