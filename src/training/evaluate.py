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
    # TODO
    raise NotImplementedError
