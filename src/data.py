"""
CIFAR-10 data: download, 45k/5k train/val split, augmentation and DataLoaders.
Also has fake=True mode with random tensors, so tests and smoke run dont need download.

Paper reference: none (dataset: Krizhevsky, 2009)
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from typing import Callable, Sequence

from torch.utils.data import DataLoader

from configs.config import TrainConfig

# mean and std of CIFAR-10 train images, per channel (R, G, B)
# TODO: double check these numbers and write source in README
CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


def build_transforms(train: bool) -> Callable:
    """
    Image transforms.

    train=True:  RandomCrop(32, padding=4) -> RandomHorizontalFlip -> ToTensor -> Normalize
    train=False: ToTensor -> Normalize   (no augmentation for val and test!)

    Output of transform for one image: tensor (3, 32, 32)
    """
    # TODO
    raise NotImplementedError


def stratified_split(
    targets: Sequence[int], val_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """
    Split indices of CIFAR train set into train and val, with same number of
    images from each class in val (val_size / 10 per class, so 500).

    Args:
        targets:  label of each image in CIFAR train set (50000 labels)
        val_size: how many images go to val (5000)
        seed:     split_seed from config. ALWAYS same, so val set never changes

    Returns:
        (train_idx, val_idx) -> lengths 45000 and 5000, no overlap
    """
    # TODO
    raise NotImplementedError


def get_dataloaders(
    train_cfg: TrainConfig, seed: int, fake: bool = False
) -> dict[str, DataLoader]:
    """
    Make train, val and test DataLoaders.

    Important:
        - make TWO CIFAR10(train=True) datasets: one with train transforms (for train_idx),
          one with eval transforms (for val_idx). otherwise val gets augmentation by mistake
        - test set is CIFAR10(train=False) with eval transforms, used only for final evaluation
        - `seed` changes only shuffle order (generator + seed_worker), not the split
        - fake=True: random tensors instead of CIFAR (256 train, 64 val, 64 test),
          same shapes and labels in [0, 10), no download

    Args:
        train_cfg: training settings (batch_size, val_size, split_seed, data_dir, num_workers)
        seed:      training seed for this run
        fake:      use fake data

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
        each batch: images (N, 3, 32, 32), labels (N,)
    """
    # TODO
    raise NotImplementedError
