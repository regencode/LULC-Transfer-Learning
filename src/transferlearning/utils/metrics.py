import torch
import numpy as np


def compute_confusion_matrix(
    label: torch.Tensor,
    prediction: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix from label and prediction tensors.

    Args:
        label: Ground truth tensor of shape (H, W) with class indices.
        prediction: Prediction tensor of shape (num_classes, H, W) with logits/probs,
                    or (H, W) with class indices.
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes) where
        rows are true labels and columns are predicted labels.
    """
    if prediction.dim() == 3 and prediction.shape[0] == num_classes:
        prediction = prediction.argmax(dim=0)

    label_flat = label.view(-1).cpu().numpy()
    pred_flat = prediction.view(-1).cpu().numpy()

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_cls in range(num_classes):
        mask = label_flat == true_cls
        for pred_cls in range(num_classes):
            cm[true_cls, pred_cls] = int(np.sum(pred_flat[mask] == pred_cls))

    return cm


def iou_from_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def f1_from_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    precision = precision_from_confusion_matrix(cm)
    recall = recall_from_confusion_matrix(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    return f1


def precision_from_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    col_sums = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(col_sums > 0, np.diag(cm) / col_sums, 0.0)


def recall_from_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    row_sums = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(row_sums > 0, np.diag(cm) / row_sums, 0.0)


def overall_accuracy_from_confusion_matrix(cm: np.ndarray) -> float:
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.diag(cm).sum() / total)
