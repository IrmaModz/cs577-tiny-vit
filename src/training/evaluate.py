"""
Evaluate model on a data loader: average loss and accuracy.

Paper reference: none
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    per_class: bool = False,
) -> dict:
    """
    Put model in eval mode, go over all batches, compute loss and accuracy.
    After finish, put model back in mode it was before (train or eval).

    Args:
        model:     model that gives logits (N, 10)
        loader:    val or test DataLoader
        device:    where to run
        criterion: loss function, if None use CrossEntropyLoss
        per_class: also return accuracy for each class

    Returns:
        {"loss": float, "acc": float, "n": int}
        plus "per_class_acc": list of 10 floats if per_class=True
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    was_training = model.training
    model.eval()
    try:
        loss_sum = 0.0  # sum of loss for each image (not average of batch averages)
        correct = 0
        total = 0
        class_correct: torch.Tensor | None = None
        class_total: torch.Tensor | None = None

        for images, labels in loader:
            images = images.to(device)  # (N, 3, 32, 32)
            labels = labels.to(device)  # (N,)

            logits = model(images)  # (N, 10)
            n = labels.size(0)

            # criterion gives average of batch, so we multiply by n to get sum.
            # like this, small last batch is counted right
            loss_sum += criterion(logits, labels).item() * n

            preds = logits.argmax(dim=1)  # (N,)
            hits = preds == labels  # (N,) True where prediction is right
            correct += hits.sum().item()
            total += n

            if per_class:
                num_classes = logits.size(1)
                if class_correct is None:
                    class_correct = torch.zeros(num_classes, dtype=torch.long)
                    class_total = torch.zeros(num_classes, dtype=torch.long)
                labels_cpu = labels.cpu()
                class_total += torch.bincount(labels_cpu, minlength=num_classes)  # (10,)
                class_correct += torch.bincount(
                    labels_cpu[hits.cpu()], minlength=num_classes
                )  # (10,)
    finally:
        model.train(was_training)  # put model back in mode like before

    if total == 0:
        raise ValueError("evaluate() got empty loader, there are no images")

    result = {"loss": loss_sum / total, "acc": correct / total, "n": total}
    if per_class:
        result["per_class_acc"] = [
            (c / t) if t > 0 else float("nan")
            for c, t in zip(class_correct.tolist(), class_total.tolist())
        ]
    return result