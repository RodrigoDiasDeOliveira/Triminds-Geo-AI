#!/usr/bin/env python3
"""Create a tiny deterministic RGB dataset for the local Geo-AI demo.

This dataset is intentionally synthetic. It validates the software path only;
it is not intended to measure model quality or represent Earth Observation data.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def create_image(class_index: int, seed: int, size: int = 224) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("RGB", (size, size), (230, 230, 230))
    draw = ImageDraw.Draw(image)

    margin = 20 + (class_index * 3) % 25
    accent = (
        30 + class_index * 17,
        50 + class_index * 13,
        80 + class_index * 11,
    )

    if class_index % 3 == 0:
        for x in range(margin, size - margin, 16):
            draw.line((x, margin, x, size - margin), fill=accent, width=8)
    elif class_index % 3 == 1:
        for y in range(margin, size - margin, 16):
            draw.line((margin, y, size - margin, y), fill=accent, width=8)
    else:
        draw.rectangle(
            (margin, margin, size - margin, size - margin),
            outline=accent,
            width=12,
        )
        for _ in range(10):
            x = rng.randint(margin, size - margin)
            y = rng.randint(margin, size - margin)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=accent)

    return image


def create_dataset(root: Path, images_per_split: int) -> None:
    for split in ("train", "val"):
        for class_index, class_name in enumerate(CLASSES):
            output_dir = root / split / class_name
            output_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(images_per_split):
                seed = class_index * 1000 + image_index + (0 if split == "train" else 100)
                image = create_image(class_index, seed)
                image.save(output_dir / f"sample_{image_index:02d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo", help="Output dataset directory")
    parser.add_argument("--images-per-split", type=int, default=2)
    args = parser.parse_args()

    if args.images_per_split < 1:
        parser.error("--images-per-split must be at least 1")

    create_dataset(Path(args.output), args.images_per_split)
    print(f"Demo dataset created at {args.output}")


if __name__ == "__main__":
    main()
