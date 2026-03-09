from pathlib import Path
from typing import Optional, Callable, Tuple, List

from .base_dataset import ISPRSBaseDataset
from .registry import register_dataset


@register_dataset("vaihingen")
class VaihingenDataset(ISPRSBaseDataset):
    """ISPRS Vaihingen 2D semantic segmentation dataset.

    Expects the following directory layout under root:
        root/
            images/       (RGB .tif or .png files)
            labels/       (RGB label .tif or .png files)
            splits/
                train.txt
                val.txt
                test.txt
    Each .txt file contains one filename stem per line (e.g. "top_mosaic_09cm_area1").
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(root, split, transform, target_transform)

    def _load_file_list(self) -> Tuple[List[Path], List[Path]]:
        split_file = self.root / "splits" / f"{self.split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        stems = split_file.read_text().strip().splitlines()
        image_dir = self.root / "images"
        label_dir = self.root / "labels"

        images, labels = [], []
        for stem in stems:
            stem = stem.strip()
            img = self._find_file(image_dir, stem)
            lbl = self._find_file(label_dir, stem)
            images.append(img)
            labels.append(lbl)

        return images, labels

    @staticmethod
    def _find_file(directory: Path, stem: str) -> Path:
        for ext in [".tif", ".png", ".jpg"]:
            path = directory / f"{stem}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"No image found for stem '{stem}' in {directory}")
