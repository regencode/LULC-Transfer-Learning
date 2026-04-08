from pathlib import Path
from typing import Optional, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split


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
        image_folder: Path,
        label_folder: Path,
        split: str,
        train_size: Optional[float] = 0.8,
        test_size: Optional[float] = 0.5,
        seed: Optional[int] = 42,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.images = image_folder
        self.labels = label_folder
        self.split = split
        self.seed = seed
        self.train_size = train_size
        self.test_size = test_size
        self.file_list = self._load_file_list()

    def _load_file_list(self) -> list[tuple[str, str]]:
        image_label_pairs = []
        for image_path in Path(self.images).iterdir():
            if image_path.name.endswith("tfw"):
                continue # skip .tfw files
            label_path = f"{image_path.name[:-7]}label.tif"
            image_label_pairs.append((image_path, label_path))
        train_pairs, val_pairs = train_test_split(image_label_pairs, 
                                                  train_size=self.train_size, random_state=self.seed)
        test_pairs, val_pairs = train_test_split(val_pairs, 
                                                 train_size=self.test_size,
                                                 random_state=self.seed)
        match self.split:
            case "train": return train_pairs
            case "val": return val_pairs 
            case "test": return test_pairs
        raise Exception("Please specify [ train | val | test ] for self.split")

    @staticmethod
    def rgb_to_class_index(label_rgb: np.ndarray) -> np.ndarray:
        h, w, _ = label_rgb.shape
        class_index = np.zeros((h, w), dtype=np.int64)
        for rgb, idx in ISPRS_COLOR_MAP.items():
            mask = np.all(label_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
            class_index[mask] = idx
        return class_index

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.file_list[idx][0]).convert("RGB")
        label = Image.open(self.file_list[idx][1]).convert("RGB")
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
