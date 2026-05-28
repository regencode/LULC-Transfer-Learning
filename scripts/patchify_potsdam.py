#!/usr/bin/env python3
"""Patchify Potsdam images and labels into 256x256 patches with configurable stride.

Reads full-resolution images and RGB labels, slices them into patches,
converts RGB label patches to class-index maps using ISPRS color mapping,
and generates train/val/test split files.

Usage:
    python scripts/patchify_potsdam.py --src data/potsdam --dst data/potsdam_patch256 --patch-size 256 --stride 128
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
    parser.add_argument("--seed", type=int, default=42)
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

    splits_dir = dst / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    image_stems = sorted(set(s.rsplit("_", 1)[0] for s in all_stems))
    train_imgs, val_imgs = train_test_split(image_stems, train_size=0.8, random_state=args.seed)
    test_imgs, val_imgs = train_test_split(val_imgs, test_size=0.5, random_state=args.seed)

    img_to_splits = {}
    for img in train_imgs:
        img_to_splits[img] = "train"
    for img in val_imgs:
        img_to_splits[img] = "val"
    for img in test_imgs:
        img_to_splits[img] = "test"

    split_patches = {"train": [], "val": [], "test": []}
    for stem in all_stems:
        parent_img = stem.rsplit("_", 1)[0]
        split_patches[img_to_splits[parent_img]].append(stem)

    for split_name, patches in split_patches.items():
        out_path = splits_dir / f"{split_name}.txt"
        with open(out_path, "w") as f:
            for s in sorted(patches):
                f.write(s + "\n")
        print(f"  {split_name}: {len(patches)} patches -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
