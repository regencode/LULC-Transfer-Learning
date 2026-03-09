from pathlib import Path
from typing import Optional, Callable, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


ISPRS_CLASSES = [
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter",
]

ISPRS_COLOR_MAP = {
    (255, 255, 255): 0,  # impervious surface
    (0, 0, 255): 1,      # building
    (0, 255, 255): 2,    # low vegetation
    (0, 255, 0): 3,      # tree
    (255, 255, 0): 4,    # car
    (255, 0, 0): 5,      # clutter
}

NUM_CLASSES = len(ISPRS_CLASSES)


class ISPRSBaseDataset(Dataset):
    """Base dataset for ISPRS Potsdam and Vaihingen semantic segmentation.

    Subclasses must define image_dir, label_dir, and the file listing logic.
    Labels are RGB images that get converted to class index tensors via ISPRS_COLOR_MAP.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.images, self.labels = self._load_file_list()

    def _load_file_list(self):
        raise NotImplementedError

    @staticmethod
    def rgb_to_class_index(label_rgb: np.ndarray) -> np.ndarray:
        h, w, _ = label_rgb.shape
        class_index = np.zeros((h, w), dtype=np.int64)
        for rgb, idx in ISPRS_COLOR_MAP.items():
            mask = np.all(label_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
            class_index[mask] = idx
        return class_index

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert("RGB")
        label = Image.open(self.labels[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        label_np = np.array(label, dtype=np.uint8)
        label_idx = self.rgb_to_class_index(label_np)

        if self.target_transform:
            label_idx = self.target_transform(label_idx)

        label_tensor = torch.from_numpy(label_idx).long()
        return image, label_tensor
