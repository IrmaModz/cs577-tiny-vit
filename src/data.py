"""
CIFAR-10 data: download, 45k/5k train/val split, augmentation and DataLoaders.
Also has fake=True mode with random tensors, so tests and smoke run dont need download.

Paper reference: none (dataset: Krizhevsky, 2009)
Author: <name>, Chatgpt
Adapted from: none
"""

from __future__ import annotations

from typing import Callable, Sequence

from configs.config import TrainConfig

from torchvision import transforms
import torch
from src.reproducibility import make_generator, seed_worker
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset, TensorDataset

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
    if train:
        return transforms.Compose([ 
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])


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
    targets = torch.as_tensor(targets)

    # Count required val samples per class
    num_classes = 10
    val_per_class = val_size // num_classes

    generator = make_generator(seed)

    train_idx = []
    val_idx = []

    for class_id in range(num_classes):
        indices = torch.where(targets == class_id)[0]

        # take val set indices randomly
        perm = torch.randperm(len(indices), generator=generator)
        indices = indices[perm]

        # add to val and add rest of the image to train
        val_idx.extend(indices[:val_per_class].tolist())
        train_idx.extend(indices[val_per_class:].tolist())

    return train_idx, val_idx


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
    if fake:
        # Fake datasets for testing without downloading CIFAR-10
        generator = torch.Generator().manual_seed(train_cfg.split_seed)

        def make_fake_dataset(size):
            images = torch.randn(size, 3, 32, 32, generator=generator)
            labels = torch.randint(0, 10, (size,), generator=generator)

            return TensorDataset(images, labels)

        train_dataset = make_fake_dataset(256)
        val_dataset = make_fake_dataset(64)
        test_dataset = make_fake_dataset(64)

    else:
        # Training dataset with augmentation
        train_full = CIFAR10(
            root=train_cfg.data_dir,
            train=True,
            download=True,
            transform=build_transforms(train=True)
        )

        # Validation dataset without augmentation
        val_full = CIFAR10(
            root=train_cfg.data_dir,
            train=True,
            download=True,
            transform=build_transforms(train=False)
        )

        # Stratified split with fixed split_seed
        # Instead of full dataset use subset 45000/5000 with stratified split
        train_idx, val_idx = stratified_split(
            targets=train_full.targets,
            val_size=train_cfg.val_size,
            seed=train_cfg.split_seed
        )

        train_dataset = Subset(train_full, train_idx)
        val_dataset = Subset(val_full, val_idx)

        # Official test dataset
        test_dataset = CIFAR10(
            root=train_cfg.data_dir,
            train=False,
            download=True,
            transform=build_transforms(train=False)
        )

    # Batch images: (N, 3, 32, 32)
    # Batch labels: (N,)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        generator=make_generator(seed),
        worker_init_fn=seed_worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
