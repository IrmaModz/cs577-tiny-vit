"""
All settings for our project live here. Model settings (ViTConfig) and training
settings (TrainConfig), and also small function to load experiment from experiments.json.

Please dont hard-code numbers in other files, always take them from here!

Paper reference: ViT paper (Dosovitskiy et al., ICLR 2021), Table 1, but we make it much smaller for CIFAR-10
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# experiments.json is in same folder as this file
DEFAULT_EXPERIMENTS_PATH = Path(__file__).with_name("experiments.json")

VALID_DEVICES = ("auto", "cpu", "cuda")

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
@dataclass
class ViTConfig:
    """
    Settings for the shape of our Vision Transformer.

    Letters we use everywhere for shapes (also in report):
        N batch, C channels, H/W image size, p patch size, P number of patches,
        T number of tokens, K kept patches, D embedding size, h heads,
        d_h size of one head, L depth, M hidden size in MLP

    Fields:
        image_size:         H = W, CIFAR images are 32x32
        in_channels:        C, RGB so it is 3
        num_classes:        CIFAR-10 has 10 classes
        patch_size:         p, size of one square patch. baseline is 4, ablation is 8
        embed_dim:          D, size of each token vector
        depth:              L, how many transformer blocks
        num_heads:          h, how many attention heads
        mlp_ratio:          hidden size of MLP is mlp_ratio * D
        dropout:            normal dropout (after pos embedding, in MLP, after attention proj)
        attn_dropout:       dropout on attention weights
        drop_path_rate:     max rate for stochastic depth, it grows linear over blocks
        patch_dropout_rate: how much patch tokens we drop in training (0 means off)
    """

    image_size: int = 32
    in_channels: int = 3
    num_classes: int = 10
    patch_size: int = 4
    embed_dim: int = 128
    depth: int = 6
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    drop_path_rate: float = 0.1
    patch_dropout_rate: float = 0.0

    def __post_init__(self) -> None:
        """
        Check that settings make sense, right after config is created.

        Should give ValueError (with clear message) when:
            - image_size can not be divided by patch_size
            - embed_dim can not be divided by num_heads
            - some rate (dropout, attn_dropout, drop_path_rate, patch_dropout_rate) is not in [0, 1)
        """
        # sizes must be bigger than 0, otherwise below we divide by 0 and get crash
        for name in ("image_size", "in_channels", "num_classes", "patch_size",
                     "embed_dim", "depth", "num_heads"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"ViTConfig.{name} must be bigger than 0, but it is {value}")
        if self.mlp_ratio <= 0:
            raise ValueError(f"ViTConfig.mlp_ratio must be bigger than 0, but it is {self.mlp_ratio}")

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) can not be divided by "
                f"patch_size ({self.patch_size})"
            )
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) can not be divided by "
                f"num_heads ({self.num_heads})"
            )

        for name in ("dropout", "attn_dropout", "drop_path_rate", "patch_dropout_rate"):
            rate = getattr(self, name)
            if not 0.0 <= rate < 1.0:
                raise ValueError(f"ViTConfig.{name} must be in [0, 1), but it is {rate}")

    @property
    def num_patches(self) -> int:
        """P = (image_size / patch_size) ** 2. for baseline 64, for p=8 it is 16."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_tokens(self) -> int:
        """T = P + 1, because we add CLS token. baseline gives 65."""
        return self.num_patches + 1

    @property
    def head_dim(self) -> int:
        """d_h = D / h. baseline gives 32."""
        return self.embed_dim // self.num_heads

    @property
    def mlp_dim(self) -> int:
        """M = int(mlp_ratio * D). baseline gives 512."""
        return int(self.mlp_ratio * self.embed_dim)

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """
    Settings for how we train and evaluate the model.

    Fields:
        epochs:          how many epochs we train (no early stopping). we fix it after 1 epoch timing run
        batch_size:      batch size N
        lr:              max learning rate for AdamW
        weight_decay:    weight decay for AdamW (not for bias, LayerNorm, cls_token, pos_embed)
        warmup_epochs:   first few epochs lr goes up linear, after that cosine goes down
        label_smoothing: label smoothing for cross entropy, 0 means off
        grad_clip:       max norm of gradient, None if we dont want clipping
        amp:             mixed precision on GPU (faster)
        val_size:        how many images we take from CIFAR train for validation (500 per class)
        split_seed:      seed for train/val split. ALWAYS same, so every run has same val set!
        seeds:           seeds for repeated runs
        data_dir:        where CIFAR-10 is downloaded (its in .gitignore)
        out_dir:         where we save results
        num_workers:     workers for DataLoader
        device:          "auto" or "cpu" or "cuda"
        deterministic:   make cuDNN deterministic (but slower)
    """

    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    label_smoothing: float = 0.0
    grad_clip: float | None = 1.0
    amp: bool = True
    val_size: int = 5000
    split_seed: int = 0
    seeds: tuple[int, ...] = (0, 1, 2)
    data_dir: str = "data"
    out_dir: str = "results"
    num_workers: int = 2
    device: str = "auto"
    deterministic: bool = False

    def __post_init__(self) -> None:
        """
        Check training settings after config is created.

        Should give ValueError when:
            - epochs, batch_size or val_size is not positive
            - warmup_epochs is bigger than epochs
            - device is not "auto", "cpu" or "cuda"
        Also seeds should become tuple (from JSON it comes as list).
        """
        for name in ("epochs", "batch_size", "val_size"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"TrainConfig.{name} must be bigger than 0, but it is {value}")

        if self.warmup_epochs < 0:
            raise ValueError(f"TrainConfig.warmup_epochs can not be negative, but it is {self.warmup_epochs}")
        if self.warmup_epochs > self.epochs:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) can not be bigger than "
                f"epochs ({self.epochs})"
            )

        if self.device not in VALID_DEVICES:
            raise ValueError(
                f"TrainConfig.device must be 'auto', 'cpu' or 'cuda', but it is {self.device!r}"
            )

        # from JSON seeds come as list, we make it tuple
        self.seeds = tuple(int(s) for s in self.seeds)


# ---------------------------------------------------------------------------
# Loading experiments
# ---------------------------------------------------------------------------
def load_experiment(
    name: str, path: str | Path = DEFAULT_EXPERIMENTS_PATH
) -> tuple[ViTConfig, TrainConfig]:
    """
    Make configs for one experiment by its name.

    First we take default ViTConfig and TrainConfig, then we change values that are
    written under `name` in experiments.json. Each key goes to the class that has
    field with this name. Keys that start with "_" (like "_description") are just
    comments, so skip them.

    Args:
        name: name of experiment, for example "baseline_p4", "ablation_p8", "smoke"
        path: path to experiments.json

    Returns:
        (model_cfg, train_cfg)

    Raises:
        KeyError:   if there is no experiment with this name
        ValueError: if some key (not starting with "_") is not field in ViTConfig or
                    TrainConfig. this is to catch typos!
    """
    with open(path, encoding="utf-8") as f:
        experiments = json.load(f)

    if name not in experiments:
        known = sorted(k for k in experiments if not k.startswith("_"))
        raise KeyError(f"No experiment with name {name!r} in {path}. We have: {known}")

    overrides = experiments[name]
    if not isinstance(overrides, dict):
        raise ValueError(f"Experiment {name!r} must be {{...}} in JSON, but it is {type(overrides).__name__}")

    model_fields = {f.name for f in fields(ViTConfig)}
    train_fields = {f.name for f in fields(TrainConfig)}

    model_kwargs: dict = {}
    train_kwargs: dict = {}
    unknown: list[str] = []
    for key, value in overrides.items():
        if key.startswith("_"):
            continue  # comment, like "_description"
        if key in model_fields:
            model_kwargs[key] = value
        elif key in train_fields:
            train_kwargs[key] = value
        else:
            unknown.append(key)

    if unknown:
        raise ValueError(
            f"Unknown key(s) in experiment {name!r}: {unknown}. "
            f"Maybe typo? Key must be field of ViTConfig or TrainConfig."
        )

    # here __post_init__ runs, so wrong values give error already here
    model_cfg = ViTConfig(**model_kwargs)
    train_cfg = TrainConfig(**train_kwargs)
    return model_cfg, train_cfg


def config_to_dict(model_cfg: ViTConfig, train_cfg: TrainConfig) -> dict:
    """
    Put both configs in one dict, so we can save it inside every result JSON.

    Returns:
        {"model": {...ViTConfig fields...}, "train": {...TrainConfig fields...}}
    """
    train = asdict(train_cfg)
    train["seeds"] = list(train["seeds"])  # tuple -> list, same like in JSON
    return {"model": asdict(model_cfg), "train": train}