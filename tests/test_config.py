"""Small tests for configs/config.py, they check all points from "Done when". Run: pytest tests/test_config.py"""

import json

import pytest

from configs.config import TrainConfig, ViTConfig, config_to_dict, load_experiment


# --- uses our real configs/experiments.json ---------------------------------

def test_baseline_p4():
    model_cfg, _ = load_experiment("baseline_p4")
    assert model_cfg.patch_size == 4
    assert model_cfg.num_patches == 64
    assert model_cfg.num_tokens == 65


def test_ablation_p8():
    model_cfg, _ = load_experiment("ablation_p8")
    assert model_cfg.patch_size == 8
    assert model_cfg.num_patches == 16
    assert model_cfg.num_tokens == 17


def test_smoke_works():
    model_cfg, train_cfg = load_experiment("smoke")
    assert isinstance(train_cfg.seeds, tuple)  # from JSON it was list


def test_unknown_experiment_name():
    with pytest.raises(KeyError):
        load_experiment("this_name_does_not_exist")


# --- uses small fake JSON file, so we dont touch real one -------------------

def test_typo_key_gives_error(tmp_path):
    path = tmp_path / "experiments.json"
    path.write_text(json.dumps({"bad": {"embed_dimm": 64}}))
    with pytest.raises(ValueError):
        load_experiment("bad", path)


def test_underscore_keys_are_skipped(tmp_path):
    path = tmp_path / "experiments.json"
    path.write_text(json.dumps({"ok": {"_description": "just comment", "depth": 2, "epochs": 5}}))
    model_cfg, train_cfg = load_experiment("ok", path)
    assert model_cfg.depth == 2  # this key went to ViTConfig
    assert train_cfg.epochs == 5  # this key went to TrainConfig


# --- properties with default values ----------------------------------------

def test_properties_baseline():
    cfg = ViTConfig()
    assert cfg.head_dim == 32
    assert cfg.mlp_dim == 512


# --- wrong values must give ValueError -------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"patch_size": 5},           # 32 can not be divided by 5
    {"embed_dim": 130},          # 130 can not be divided by 4 heads
    {"dropout": 1.0},            # rate must be < 1
    {"attn_dropout": -0.1},      # rate must be >= 0
    {"drop_path_rate": 1.5},
    {"patch_dropout_rate": 1.0},
])
def test_bad_vit_config(kwargs):
    with pytest.raises(ValueError):
        ViTConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0},
    {"batch_size": -1},
    {"val_size": 0},
    {"epochs": 5, "warmup_epochs": 6},  # warmup bigger than epochs
    {"device": "gpu"},                  # only auto / cpu / cuda
])
def test_bad_train_config(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_seeds_list_becomes_tuple():
    assert TrainConfig(seeds=[3, 4]).seeds == (3, 4)


def test_config_to_dict():
    d = config_to_dict(ViTConfig(), TrainConfig())
    assert set(d) == {"model", "train"}
    assert d["model"]["patch_size"] == 4
    assert d["train"]["epochs"] == 30
    json.dumps(d)  # must be possible to save as JSON