import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FREE_SYMBOLS = frozenset({".", "G", "S"})


def load_movingai_map(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as file:
        lines = [line.rstrip("\n") for line in file]

    height = None
    width = None
    map_start = None

    for index, line in enumerate(lines):
        clean = line.strip()
        if clean.startswith("height "):
            height = int(clean.split()[1])
        elif clean.startswith("width "):
            width = int(clean.split()[1])
        elif clean == "map":
            map_start = index + 1
            break

    if None in (height, width, map_start):
        raise ValueError(f"Invalid MovingAI map: {path}")

    grid = lines[map_start : map_start + height]
    image = np.zeros((height, width), dtype=float)
    for row_index, row in enumerate(grid):
        for col_index, symbol in enumerate(row):
            image[row_index, col_index] = 1.0 if symbol in FREE_SYMBOLS else 0.0
    return image


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 visualize_map.py <map_name>")

    map_name = sys.argv[1]
    map_path = Path("worlds/movingai_mapf") / f"{map_name}.map"
    if not map_path.exists():
        raise FileNotFoundError(f"Could not find {map_path}")

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
