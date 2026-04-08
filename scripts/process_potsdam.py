import argparse
import zipfile
from pathlib import Path
import shutil
import numpy as np
import einops as ein
from PIL import Image 

def parse_args():
    parser = argparse.ArgumentParser(description="Process Potsdam dataset")
    parser.add_argument("--source", type=str, required=True, help="Path to Potsdam .zip file")
    parser.add_argument("--dest", type=str, required=True, help="Place to unzip and process Potsdam files")
    parser.add_argument("--skip-patchify", action="store_true", help="Skip patchify")
    parser.add_argument("--patch-width", type=int, default=256, help="Width of each patch")
    parser.add_argument("--patch-height", type=int, default=256, help="Height of each patch")
    parser.add_argument("--patch-stride", type=int, default=128, help="Stride between each patch")
    return parser.parse_args()

def slice_image_with_stride(image_path: Path, save_folder_path: Path, 
                            H:       int  =  256,
                            W:       int  =  256,
                            stride:  int  =  128):
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w, c = image.shape

    # Calculate padded dimensions to ensure complete coverage
    num_patches_h = max(1, ((h - H) // stride) + 1)
    num_patches_w = max(1, ((w - W) // stride) + 1)

    padded_h = (num_patches_h * stride) + H
    padded_w = (num_patches_w * stride) + W

    # Pad image (using edge replication for better results)
    padded_image = np.pad(image, ((0, padded_h - h), (0, padded_w - w), (0, 0)), 
                          mode='edge')

    # Extract patches using sliding window
    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            start_h = i * stride
            start_w = j * stride
            patch = padded_image[start_h:start_h + H, start_w:start_w + W, :]
            patches.append(patch)

    patches = np.array(patches)

    # Save patches
    for patch_idx in range(patches.shape[0]):
        filename = f'{image_path.name.removesuffix(".tif")}_{patch_idx}.tif'
        Image.fromarray(patches[patch_idx]).save(save_folder_path / filename)
    return


def patchify_images_in_folder(image_root: Path,  save_folder_path: Path,
                              H:       int,                                                             
                              W:       int,                                                             
                              stride:  int):                                                            
    save_folder_path.mkdir(parents=True, exist_ok=True)
    for image_path in image_root.iterdir():
        slice_image_with_stride(image_path, save_folder_path,
                            H=H, 
                            W=W, 
                            stride=stride)
    return

def main():
    args = parse_args()
    source = Path(args.source)
    data_root = Path(args.dest)
    for sub in ["images", "labels", "temp"]:
        (data_root / sub).mkdir(parents=True, exist_ok=True)
    temp = data_root / "temp"
    print(f"Copying Potsdam from {source} to {temp}...")
    shutil.copy(source, temp)
    print(f"Unpacking Potsdam...")
    shutil.unpack_archive(temp / source.name, temp)
    original_potsdam_root = temp / "Potsdam"
    print(f"Unpacking Potsdam images...")
    with zipfile.ZipFile(original_potsdam_root / "2_Ortho_RGB.zip") as z:
        for member in z.namelist():
            filename = Path(member).name # strip the top-level folder
            if filename.endswith(".tif"):  # read only .tif files
                (data_root / "images" / filename).write_bytes(z.read(member))
    print(f"Unpacking Potsdam labels...")
    shutil.unpack_archive(original_potsdam_root / "5_Labels_all.zip", data_root / "labels")
    print(f"Potsdam dataset unpacking complete.")
    if args.skip_patchify: return
    # patchify
    shutil.move(data_root / "images", temp / "images")
    shutil.move(data_root / "labels", temp / "labels")
    print(f"Patchifying images...")
    patchify_images_in_folder(temp / "images", data_root / "images", 
                              args.patch_height,
                              args.patch_width,
                              args.patch_stride)
    print(f"Patchifying labels...")
    patchify_images_in_folder(temp / "labels", data_root / "labels", 
                              args.patch_height,
                              args.patch_width,
                              args.patch_stride)

if __name__ == "__main__":
    main()
