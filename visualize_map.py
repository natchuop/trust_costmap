import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_movingai_map(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    height = None
    width = None
    map_start = None

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("height "):
            height = int(line.split()[1])
        elif line.startswith("width "):
            width = int(line.split()[1])
        elif line == "map":
            map_start = i + 1
            break

    if height is None or width is None or map_start is None:
        raise ValueError(f"Invalid MovingAI map: {path}")

    grid = lines[map_start:map_start + height]

    image = np.zeros((height, width), dtype=float)

    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            # Free symbols in MovingAI maps are usually '.', 'G', or 'S'.
            # Everything else is treated as blocked.
            image[r, c] = 1.0 if ch in {".", "G", "S"} else 0.0

    return image


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 visualize_map.py <map_name_without_extension>")
        print("Example: python3 visualize_map.py room-32-32-4")
        sys.exit(1)

    map_name = sys.argv[1]
    map_path = Path("worlds") / "movingai_mapf" / f"{map_name}.map"

    if not map_path.exists():
        matches = list(Path("worlds/movingai_mapf").rglob(f"{map_name}.map"))
        if not matches:
            raise FileNotFoundError(f"Could not find {map_name}.map")
        map_path = matches[0]

    image = load_movingai_map(map_path)

    output_path = Path(f"{map_name}.png")

    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.title(map_name)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()

    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()
