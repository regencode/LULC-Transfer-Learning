from typing import Optional, List
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ..datasets.base_dataset import ISPRS_CLASSES, ISPRS_COLOR_MAP


ISPRS_CMAP_ARRAY = np.zeros((len(ISPRS_CLASSES), 3), dtype=np.float32)
for rgb, idx in ISPRS_COLOR_MAP.items():
    ISPRS_CMAP_ARRAY[idx] = np.array(rgb) / 255.0

ISPRS_CMAP = mcolors.ListedColormap(ISPRS_CMAP_ARRAY)


def visualize_prediction(
    image: torch.Tensor,
    label: torch.Tensor,
    prediction: torch.Tensor,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """Display [original image | ground truth | prediction] in a single row.

    Args:
        image: (C, H, W) tensor, float [0, 1] or ImageNet-normalized.
        label: (H, W) tensor of class indices.
        prediction: (num_classes, H, W) logits/probs or (H, W) class indices.
        class_names: Optional list for colorbar ticks.
        save_path: If provided, saves the figure instead of showing it.
    """
    if class_names is None:
        class_names = ISPRS_CLASSES

    image_np = _tensor_to_display_image(image)
    label_np = label.cpu().numpy() if torch.is_tensor(label) else label
    pred_np = _to_class_map(prediction)

    num_classes = len(class_names)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(label_np, cmap=ISPRS_CMAP, vmin=0, vmax=num_classes - 1, interpolation="nearest")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    im = axes[2].imshow(pred_np, cmap=ISPRS_CMAP, vmin=0, vmax=num_classes - 1, interpolation="nearest")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, ticks=range(num_classes))
    cbar.ax.set_yticklabels(class_names)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _tensor_to_display_image(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.detach().cpu().numpy()
    if img.shape[0] <= 4:
        img = img[:3].transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def _to_class_map(prediction) -> np.ndarray:
    if torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    if prediction.ndim == 3 and prediction.shape[0] > 1:
        prediction = prediction.argmax(axis=0)
    return prediction
