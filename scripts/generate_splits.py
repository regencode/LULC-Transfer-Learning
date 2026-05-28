#!/usr/bin/env python3
"""Generate train/val/test split files for ISPRS datasets.

Uses the same split logic as the original ISPRSBaseDataset to ensure
reproducibility. Outputs train.txt, val.txt, test.txt with image filename
stems (without extension).

Usage:
    python scripts/generate_splits.py --image_dir data/potsdam/images --output_dir data/potsdam/splits --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def build_image_label_pairs(image_dir: Path):
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.name.endswith("tfw"):
            continue
        stem = image_path.stem
        if stem.endswith("_RGB"):
            label_stem = stem.replace("_RGB", "_label")
        else:
            label_stem = stem + "_label"
        pairs.append(image_path.stem)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate train/val/test split files")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with image files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for split files")
    parser.add_argument("--train_size", type=float, default=0.8, help="Training set proportion")
    parser.add_argument("--test_size", type=float, default=0.5, help="Proportion of remaining for test (rest is val)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stems = build_image_label_pairs(image_dir)
    print(f"Found {len(stems)} images in {image_dir}")

    train_stems, val_stems = train_test_split(stems, train_size=args.train_size, random_state=args.seed)
    test_stems, val_stems = train_test_split(val_stems, test_size=args.test_size, random_state=args.seed)

    for split_name, split_stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        out_path = output_dir / f"{split_name}.txt"
        with open(out_path, "w") as f:
            for stem in sorted(split_stems):
                f.write(stem + "\n")
        print(f"  {split_name}: {len(split_stems)} images -> {out_path}")


if __name__ == "__main__":
    main()
