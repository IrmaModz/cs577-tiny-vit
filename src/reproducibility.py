"""
Small helpers so our runs can be repeated: seeds, deterministic mode, device choice,
and info about environment (versions, GPU) that we save in every result file.

Paper reference: none
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import torch
import os
import platform
import random

import numpy as np

VALID_DEVICES = ("auto", "cpu", "cuda")


def set_seed(seed: int) -> None:
    """
    Set same seed for python `random`, numpy, torch and torch.cuda.
    Call this at start of every run, before creating model or data loaders.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_generator(seed: int) -> torch.Generator:
    """
    Make torch.Generator with this seed. We give it to DataLoader so shuffle order
    is same every time for same seed.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id: int) -> None:
    """
    Give this to DataLoader(worker_init_fn=seed_worker). Each worker gets its own
    seed made from torch initial seed, so augmentation in workers is repeatable too.
    """
    worker_seed = torch.initial_seed()% 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_determinism(strict: bool) -> None:
    """
    If strict is True: cudnn.deterministic = True, cudnn.benchmark = False and
    torch.use_deterministic_algorithms(True, warn_only=True). Slower but more repeatable.
    If False: cudnn.benchmark = True (faster).
    Note: warn_only=True, so some operations may still not be fully deterministic
    """
    if strict:
        # some CUDA matrix operations need this setting to be deterministic.
        # setdefault: if somebody already set it, we dont change it
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only=True: if some operation can not be deterministic, we get warning, not crash
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # also turn off strict settings, in case strict=True was called before
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def get_device(name: str = "auto") -> torch.device:
    """
    "auto" -> cuda if it is available, otherwise cpu.
    "cpu" / "cuda" -> just that device.
    """
    if name not in VALID_DEVICES:
        raise ValueError(f"device must be 'auto', 'cpu' or 'cuda', but it is {name!r}")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        # better clear error here than strange error later in training
        raise ValueError("device 'cuda' was asked, but no GPU is available here")
    return torch.device(name)

def get_env_info(device: torch.device | str | None = None) -> dict:
    """
    Info about where code was run, we save it inside every result JSON.

    Args:
        device: device the run really used. If None, we only report what is available.

    Returns:
        dict with python, torch, torchvision, numpy versions, cuda version,
        the device used, and its name (GPU name or CPU name)
    """
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except ImportError:
        torchvision_version = None  # not installed

    if device is not None:
        used = torch.device(device)
    else:
        used = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if used.type == "cuda":
        device_name = torch.cuda.get_device_name(used)
    else:
        device_name = platform.processor() or platform.machine()

    # only strings and None here, so json.dumps always works
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,  # None when torch has no CUDA
        "device": used.type,
        "device_name": device_name,
        "platform": platform.platform(),
    }