"""ISPRS Potsdam semantic segmentation dataset for MMSegmentation."""

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class ISPRSPotsdamDataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            "impervious_surface",
            "building",
            "low_vegetation",
            "tree",
            "car",
            "clutter",
        ),
        palette=(
            (255, 255, 255),
            (0, 0, 255),
            (0, 255, 255),
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
        ),
    )

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix=".tif",
            seg_map_suffix=".png",
            reduce_zero_label=False,
            **kwargs,
        )
