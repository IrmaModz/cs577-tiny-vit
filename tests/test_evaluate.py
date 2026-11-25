"""
Small tests for src/training/evaluate.py, 
answers checked by hand. 
Run: pytest tests/test_evaluate.py
"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.evaluate import evaluate


class FixedLogits(nn.Module):
    """Fake model: gives logits what we prepared before. First pixel of image says which row to take."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("table", logits)  # (num_images, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # images (N, 3, 32, 32)
        idx = images[:, 0, 0, 0].long()  # (N,)
        return self.table[idx]  # (N, 10)


def make_case():
    # 4 images, labels 0,1,2,3. Model is right for 0,1,2 and says 9 for last one -> 3/4
    labels = torch.tensor([0, 1, 2, 3])
    preds = torch.tensor([0, 1, 2, 9])
    logits = torch.full((4, 10), -5.0)
    logits[torch.arange(4), preds] = 5.0
    images = torch.zeros(4, 3, 32, 32)
    images[:, 0, 0, 0] = torch.arange(4).float()  # so fake model knows which row to take
    return FixedLogits(logits), images, labels, logits


def test_accuracy_three_of_four_and_exact_loss():
    model, images, labels, logits = make_case()
    # batch_size=3 -> last batch has only 1 image, so we check loss is counted right
    loader = DataLoader(TensorDataset(images, labels), batch_size=3)
    out = evaluate(model, loader, torch.device("cpu"))
    expected_loss = nn.functional.cross_entropy(logits, labels).item()
    assert out["n"] == 4
    assert out["acc"] == 0.75
    assert math.isclose(out["loss"], expected_loss, rel_tol=1e-6)


def test_mode_is_restored():
    model, images, labels, _ = make_case()
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model.train()
    evaluate(model, loader, torch.device("cpu"))
    assert model.training is True
    model.eval()
    evaluate(model, loader, torch.device("cpu"))
    assert model.training is False


def test_per_class():
    model, images, labels, _ = make_case()
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    out = evaluate(model, loader, torch.device("cpu"), per_class=True)
    pc = out["per_class_acc"]
    assert len(pc) == 10
    assert pc[:4] == [1.0, 1.0, 1.0, 0.0]
    assert all(math.isnan(x) for x in pc[4:])  # these classes have no images


def test_weights_unchanged():
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    before = [p.clone() for p in model.parameters()]
    loader = DataLoader(TensorDataset(torch.randn(5, 3, 32, 32), torch.randint(0, 10, (5,))), batch_size=2)
    evaluate(model, loader, torch.device("cpu"))
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))