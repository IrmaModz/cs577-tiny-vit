"""
Small tests for src/reproducibility.py, they check all points from "Done when". 
Run: pytest tests/test_reproducibility.py
"""

import json
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.reproducibility import (
    configure_determinism,
    get_device,
    get_env_info,
    make_generator,
    seed_worker,
    set_seed,
)


def draw_numbers():
    """Take one random number from each library."""
    return random.random(), np.random.rand(), torch.rand(1).item()


def test_set_seed_two_times_gives_same_numbers():
    set_seed(0)
    first = draw_numbers()
    set_seed(0)
    second = draw_numbers()
    assert first == second


def test_different_seed_gives_different_numbers():
    set_seed(0)
    a = draw_numbers()
    set_seed(1)
    b = draw_numbers()
    assert a != b


def shuffle_order(seed):
    """Order of items after DataLoader shuffle, with generator from this seed."""
    data = TensorDataset(torch.arange(20))
    loader = DataLoader(data, batch_size=5, shuffle=True, generator=make_generator(seed))
    return [x.item() for (batch,) in loader for x in batch]


def test_same_seed_same_shuffle_order():
    assert shuffle_order(0) == shuffle_order(0)


def test_different_seed_different_shuffle_order():
    assert shuffle_order(0) != shuffle_order(1)


def test_seed_worker_is_repeatable():
    torch.manual_seed(5)
    seed_worker(0)
    a = (random.random(), np.random.rand())
    torch.manual_seed(5)
    seed_worker(0)
    b = (random.random(), np.random.rand())
    assert a == b


def test_get_device():
    assert get_device("cpu") == torch.device("cpu")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert get_device("auto").type == expected  # cpu on laptop, cuda on Colab
    assert get_device().type == expected  # "auto" is default


def test_get_device_bad_name():
    with pytest.raises(ValueError):
        get_device("gpu")


def test_env_info_is_json():
    info = get_env_info()
    json.dumps(info)  # must not give error
    for key in ("python", "torch", "torchvision", "numpy", "cuda", "device_name"):
        assert key in info


def test_configure_determinism():
    # remember settings before, and put them back at end, so other tests are not changed
    old = (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.are_deterministic_algorithms_enabled(),
    )
    try:
        configure_determinism(True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.are_deterministic_algorithms_enabled()

        configure_determinism(False)
        assert torch.backends.cudnn.benchmark is True
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old[0], old[1]
        torch.use_deterministic_algorithms(old[2])