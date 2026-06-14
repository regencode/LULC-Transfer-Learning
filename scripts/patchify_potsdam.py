#!/usr/bin/env python3
"""Patchify Potsdam images and labels into patches with configurable stride.

Reads full-resolution images and RGB labels, slices them into patches,
converts RGB label patches to class-index maps using ISPRS color mapping,
and inherits train/val/test splits from the source dataset.

Requires prepare_potsdam.py to have been run first (generates the source
splits at <src>/splits/).

Usage:
    python scripts/patchify_potsdam.py --src data/potsdam --dst data/potsdam_patch256 --patch-size 256 --stride 128
    python scripts/patchify_potsdam.py --src data/potsdam --dst data/potsdam_patch512 --patch-size 512 --stride 256
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


def parse_args():
    parser = argparse.ArgumentParser(description="Patchify Potsdam dataset")
    parser.add_argument("--src", type=str, default="data/potsdam")
    parser.add_argument("--dst", type=str, default="data/potsdam_patch256")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    return parser.parse_args()


def rgb_to_class_index(label_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = label_rgb.shape
    class_index = np.zeros((h, w), dtype=np.uint8)
    for rgb, idx in ISPRS_COLOR_MAP.items():
        mask = np.all(label_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        class_index[mask] = idx
    return class_index


def patchify(image: np.ndarray, patch_size: int, stride: int):
    h, w = image.shape[:2]
    patches = []
    positions = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            positions.append((y, x))

    rem_h = h - patch_size
    rem_w = w - patch_size
    last_y = (rem_h // stride) * stride
    last_x = (rem_w // stride) * stride

    if last_y + patch_size < h:
        y = h - patch_size
        for x_start in range(0, w - patch_size + 1, stride):
            if (y, x_start) not in positions:
                patches.append(image[y:y + patch_size, x_start:x_start + patch_size])
                positions.append((y, x_start))
        x = w - patch_size
        if (y, x) not in positions:
            patches.append(image[y:y + patch_size, x:x + patch_size])
            positions.append((y, x))

    if last_x + patch_size < w:
        x = w - patch_size
        for y_start in range(0, h - patch_size + 1, stride):
            if (y_start, x) not in positions:
                patches.append(image[y_start:y_start + patch_size, x:x + patch_size])
                positions.append((y_start, x))

    return patches


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    dst_img = dst / "images"
    dst_ann = dst / "ann_dir"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_ann.mkdir(parents=True, exist_ok=True)

    splits_src = src / "splits"
    if not splits_src.is_dir():
        raise FileNotFoundError(
            f"Splits directory not found at {splits_src}. "
            f"Run prepare_potsdam.py first to generate the source dataset with splits."
        )

    image_files = sorted([f for f in (src / "images").iterdir() if f.suffix == ".tif"])
    print(f"Found {len(image_files)} images to patchify (patch_size={args.patch_size}, stride={args.stride})")

    all_stems = []
    for img_path in tqdm(image_files, desc="Patchifying"):
        stem = img_path.stem
        label_name = stem.replace("_RGB", "_label") + ".tif"
        label_path = src / "labels" / label_name

        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        label_rgb = np.array(Image.open(label_path).convert("RGB"))

        img_patches = patchify(img_rgb, args.patch_size, args.stride)
        lbl_patches = patchify(label_rgb, args.patch_size, args.stride)

        assert len(img_patches) == len(lbl_patches), \
            f"Patch count mismatch for {stem}: {len(img_patches)} vs {len(lbl_patches)}"

        for i, (img_p, lbl_p) in enumerate(zip(img_patches, lbl_patches)):
            patch_stem = f"{stem}_{i:04d}"
            Image.fromarray(img_p).save(dst_img / f"{patch_stem}.tif")

            class_idx = rgb_to_class_index(lbl_p)
            Image.fromarray(class_idx, mode="L").save(dst_ann / f"{patch_stem}.png")
            all_stems.append(patch_stem)

    print(f"Created {len(all_stems)} patches from {len(image_files)} images")

    img_to_split = {}
    for split_name in ["train", "val", "test"]:
        split_file = splits_src / f"{split_name}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                f"Run prepare_potsdam.py first to generate the source dataset with splits."
            )
        with open(split_file) as f:
            for line in f:
                img_to_split[line.strip()] = split_name

    splits_dir = dst / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    split_patches = {"train": [], "val": [], "test": []}
    for stem in all_stems:
        parent_img = stem.rsplit("_", 1)[0]
        if parent_img not in img_to_split:
            raise ValueError(
                f"Parent image '{parent_img}' for patch '{stem}' not found in source splits. "
                f"Ensure prepare_potsdam.py was run with the same source data."
            )
        split_patches[img_to_split[parent_img]].append(stem)

    for split_name, patches in split_patches.items():
        out_path = splits_dir / f"{split_name}.txt"
        with open(out_path, "w") as f:
            for s in sorted(patches):
                f.write(s + "\n")
        print(f"  {split_name}: {len(patches)} patches -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
