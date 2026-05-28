"""Custom data transforms for MMSegmentation pipeline."""

from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SyncOriShape(BaseTransform):
    def transform(self, results):
        if "img_shape" in results:
            results["ori_shape"] = results["img_shape"]
        return results
