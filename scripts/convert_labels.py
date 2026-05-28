#!/usr/bin/env python3
"""Convert ISPRS RGB label images to grayscale class-index label images.

MMSegmentation expects segmentation labels as single-channel images where
pixel values are class indices (e.g. 0-5 for ISPRS 6-class). This script
reads RGB TIF labels and converts them using the ISPRS color map.

Usage:
    python scripts/convert_labels.py --label_dir data/potsdam/labels --output_dir data/potsdam/ann_dir
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ISPRS_COLOR_MAP = {
    (255, 255, 255): 0,
    (0, 0, 255): 1,
    (0, 255, 255): 2,
    (0, 255, 0): 3,
    (255, 255, 0): 4,
    (255, 0, 0): 5,
}


def rgb_to_class_index(label_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = label_rgb.shape
    class_index = np.zeros((h, w), dtype=np.uint8)
    for rgb, idx in ISPRS_COLOR_MAP.items():
        mask = np.all(label_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        class_index[mask] = idx
    return class_index


def main():
    parser = argparse.ArgumentParser(description="Convert RGB labels to class-index labels")
    parser.add_argument("--label_dir", type=str, required=True, help="Directory with RGB label TIFs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for grayscale label PNGs")
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted([f for f in label_dir.iterdir() if f.suffix in (".tif", ".png", ".jpg") and not f.name.endswith("tfw")])
    print(f"Found {len(label_files)} label files in {label_dir}")

    for label_path in tqdm(label_files, desc="Converting labels"):
        label_rgb = np.array(Image.open(label_path).convert("RGB"), dtype=np.uint8)
        class_index = rgb_to_class_index(label_rgb)
        out_name = label_path.stem + ".png"
        Image.fromarray(class_index, mode="L").save(output_dir / out_name)

    print(f"Saved {len(label_files)} converted labels to {output_dir}")


if __name__ == "__main__":
    main()
