"""
Small helpers so our runs can be repeated: seeds, deterministic mode, device choice,
and info about environment (versions, GPU) that we save in every result file.

Paper reference: none
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import torch


def set_seed(seed: int) -> None:
    """
    Set same seed for python `random`, numpy, torch and torch.cuda.
    Call this at start of every run, before creating model or data loaders.
    """
    # TODO
    raise NotImplementedError


def make_generator(seed: int) -> torch.Generator:
    """
    Make torch.Generator with this seed. We give it to DataLoader so shuffle order
    is same every time for same seed.
    """
    # TODO
    raise NotImplementedError


def seed_worker(worker_id: int) -> None:
    """
    Give this to DataLoader(worker_init_fn=seed_worker). Each worker gets its own
    seed made from torch initial seed, so augmentation in workers is repeatable too.
    """
    # TODO
    raise NotImplementedError


def configure_determinism(strict: bool) -> None:
    """
    If strict is True: cudnn.deterministic = True, cudnn.benchmark = False and
    torch.use_deterministic_algorithms(True, warn_only=True). Slower but more repeatable.
    If False: cudnn.benchmark = True (faster).
    """
    # TODO
    raise NotImplementedError


def get_device(name: str = "auto") -> torch.device:
    """
    "auto" -> cuda if it is available, otherwise cpu.
    "cpu" / "cuda" -> just that device.
    """
    # TODO
    raise NotImplementedError


def get_env_info() -> dict:
    """
    Info about where code was run, we save it inside every result JSON.

    Returns:
        dict with python, torch, torchvision, numpy versions, cuda version,
        and GPU name (or CPU name if no GPU)
    """
    # TODO
    raise NotImplementedError
