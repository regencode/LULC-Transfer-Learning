#!/usr/bin/env python3
"""Extract and prepare Potsdam data for MMSegmentation.

Extracts RGB images and labels from the nested Potsdam.zip, converts
RGB labels to grayscale class-index PNGs, and generates train/val/test splits.

Does NOT patchify - the mmseg data pipeline handles resizing to 256x256.

Usage:
    python scripts/prepare_potsdam.py --source Potsdam.zip --dest data/potsdam
"""

import argparse
import io
import zipfile
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
    parser = argparse.ArgumentParser(description="Prepare Potsdam dataset for mmseg")
    parser.add_argument("--source", type=str, required=True, help="Path to Potsdam.zip")
    parser.add_argument("--dest", type=str, default="data/potsdam", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    return parser.parse_args()


def rgb_to_class_index(label_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = label_rgb.shape
    class_index = np.zeros((h, w), dtype=np.uint8)
    for rgb, idx in ISPRS_COLOR_MAP.items():
        mask = np.all(label_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        class_index[mask] = idx
    return class_index


def extract_images(outer_zip, dest: Path):
    print("Extracting RGB images...")
    img_dir = dest / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    inner_data = outer_zip.read("Potsdam/2_Ortho_RGB.zip")
    inner_zip = zipfile.ZipFile(io.BytesIO(inner_data))
    tif_files = [n for n in inner_zip.namelist() if n.endswith(".tif") and not n.endswith(".tfw")]
    for name in tqdm(tif_files, desc="Extracting images"):
        filename = Path(name).name
        data = inner_zip.read(name)
        (img_dir / filename).write_bytes(data)
    inner_zip.close()
    return tif_files


def extract_and_convert_labels(outer_zip, dest: Path):
    print("Extracting and converting labels...")
    label_dir = dest / "labels"
    ann_dir = dest / "ann_dir"
    label_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    inner_data = outer_zip.read("Potsdam/5_Labels_all.zip")
    inner_zip = zipfile.ZipFile(io.BytesIO(inner_data))
    tif_files = [n for n in inner_zip.namelist() if n.endswith(".tif")]
    for name in tqdm(tif_files, desc="Converting labels"):
        filename = Path(name).name
        data = inner_zip.read(name)
        label_rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)

        (label_dir / filename).write_bytes(data)

        class_index = rgb_to_class_index(label_rgb)
        ann_stem = filename.replace("_label.tif", "_RGB")
        out_name = ann_stem + ".png"
        Image.fromarray(class_index, mode="L").save(ann_dir / out_name)
    inner_zip.close()
    return tif_files


def generate_splits(img_dir: Path, output_dir: Path, seed: int):
    print("Generating train/val/test splits...")
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted([f.stem for f in img_dir.iterdir() if f.suffix == ".tif"])
    print(f"Found {len(stems)} images")

    train_stems, val_stems = train_test_split(stems, train_size=0.8, random_state=seed)
    test_stems, val_stems = train_test_split(val_stems, test_size=0.5, random_state=seed)

    for split_name, split_stems in [("train", train_stems), ("val", val_stems), ("test", test_stems)]:
        out_path = splits_dir / f"{split_name}.txt"
        with open(out_path, "w") as f:
            for stem in sorted(split_stems):
                f.write(stem + "\n")
        print(f"  {split_name}: {len(split_stems)} -> {out_path}")


def main():
    args = parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Opening {args.source}...")
    outer_zip = zipfile.ZipFile(args.source)

    extract_images(outer_zip, dest)
    extract_and_convert_labels(outer_zip, dest)
    outer_zip.close()

    generate_splits(dest / "images", dest, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
