"""
All settings for our project live here. Model settings (ViTConfig) and training
settings (TrainConfig), and also small function to load experiment from experiments.json.

Please dont hard-code numbers in other files, always take them from here!

Paper reference: ViT paper (Dosovitskiy et al., ICLR 2021), Table 1, but we make it much smaller for CIFAR-10
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# experiments.json is in same folder as this file
DEFAULT_EXPERIMENTS_PATH = Path(__file__).with_name("experiments.json")


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
        # TODO: write these checks. for now we leave it empty (pass),
        # otherwise nobody can create config and other people get blocked
        pass

    @property
    def num_patches(self) -> int:
        """P = (image_size / patch_size) ** 2. for baseline 64, for p=8 it is 16."""
        raise NotImplementedError

    @property
    def num_tokens(self) -> int:
        """T = P + 1, because we add CLS token. baseline gives 65."""
        raise NotImplementedError

    @property
    def head_dim(self) -> int:
        """d_h = D / h. baseline gives 32."""
        raise NotImplementedError

    @property
    def mlp_dim(self) -> int:
        """M = int(mlp_ratio * D). baseline gives 512."""
        raise NotImplementedError


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
        # TODO: write these checks. empty for now, same reason like in ViTConfig
        pass


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
    # TODO: read json, skip "_" keys, put other keys to correct class,
    # give error for unknown keys
    raise NotImplementedError


def config_to_dict(model_cfg: ViTConfig, train_cfg: TrainConfig) -> dict:
    """
    Put both configs in one dict, so we can save it inside every result JSON.

    Returns:
        {"model": {...ViTConfig fields...}, "train": {...TrainConfig fields...}}
    """
    # TODO: dataclasses.asdict can help here
    raise NotImplementedError