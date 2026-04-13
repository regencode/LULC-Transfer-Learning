from pathlib import Path
from typing import Optional, Callable, Tuple, List

from .base_dataset import ISPRSBaseDataset
from .registry import register_dataset


@register_dataset("potsdam")
class PotsdamDataset(ISPRSBaseDataset):
    """ISPRS Potsdam 2D semantic segmentation dataset.

    Expects the following directory layout under root:
        root/
            images/       (RGB .tif or .png files)
            labels/       (RGB label .tif or .png files)
            splits/
                train.txt
                val.txt
                test.txt
    Each .txt file contains one filename stem per line (e.g. "top_potsdam_2_10").
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        train_size: float = 0.8,
        test_size: float = 0.5,
        seed: int = 42,
        pair_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.image_folder = Path(root) / "images"
        self.label_folder = Path(root) / "labels"
        super().__init__(self.image_folder, self.label_folder, 
                         split=split, 
                         train_size=train_size,
                         test_size=test_size,
                         seed=seed,
                         pair_transform=pair_transform,
                         transform=transform,
                         target_transform=target_transform
        )
