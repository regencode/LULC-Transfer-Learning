import argparse
import subprocess
import zipfile
from pathlib import Path
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Process Potsdam dataset")
    parser.add_argument("--source", type=str, required=True, help="Path to Potsdam .zip file")
    parser.add_argument("--dest", type=str, required=True, help="Place to unzip and process Potsdam files")
    return parser.parse_args()


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
            filename = Path(member).name # Strip the top-level folder
            if filename:  # skip directories
                (data_root / "images" / filename).write_bytes(z.read(member))
    print(f"Unpacking Potsdam labels...")
    shutil.unpack_archive(original_potsdam_root / "5_Labels_all.zip", data_root / "labels")
    print(f"Potsdam dataset unpacking complete.")
if __name__ == "__main__":
    main()
