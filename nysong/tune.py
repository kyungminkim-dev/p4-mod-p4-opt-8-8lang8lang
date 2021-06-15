"""Tune Model.

- Author: Junghoon Kim, Jongsun Shin
- Contact: placidus36@gmail.com, shinn1897@makinarocks.ai
"""
import argparse
import copy
import optuna
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info
from src.utils.common import read_yaml
from src.utils.macs import calc_macs
from src.trainer import TorchTrainer
from typing import Any, Dict, List, Tuple, Union
from train import train
import wandb
from optuna import TrialPruned

MODEL_CONFIG = read_yaml(cfg="configs/model/mobilenetv3.yaml")
DATA_CONFIG = read_yaml(cfg="configs/data/taco.yaml")

base_module_categories = ["Conv",
        "DWConv",
        "InvertedResidualv2",
        "InvertedResidualv3",
        "Fire",
        "MBConv",
        "Pass",
        ]


def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    epochs = trial.suggest_int("epochs", low=50, high=200, step=50)
    img_size = trial.suggest_categorical("img_size", [96, 112, 168, 224])
    n_select = trial.suggest_int("n_select", low=0, high=6, step=2)
    batch_size = trial.suggest_int("batch_size", low=16, high=64, step=16)
    return {
        "EPOCHS": epochs,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": batch_size
    }

def search_model(trial: optuna.trial.Trial) -> List[Any]:
    """Search model structure from user-specified search space."""
    model = []
    # 1, 2,3, 4,5, 6,7, 8,9
    # TODO: remove hard-coded stride
    n_stride = 0
    MAX_NUM_STRIDE = 5
    UPPER_STRIDE = 2 # 5(224 example): 224, 112, 56, 28, 14, 7

    # Module 1
    m1 = trial.suggest_categorical("m1", ["Conv", "DWConv"])
    m1_args = []
    m1_repeat = trial.suggest_int("m1/repeat", 1, 2)
    m1_out_channel = trial.suggest_int("m1/out_channels", low=16, high=32, step=4)
    m1_stride = trial.suggest_int("m1/stride", low=1, high=UPPER_STRIDE)
    if m1_stride == 2:
        n_stride += 1
    m1_activation = trial.suggest_categorical(
        "m1/activation", ["ReLU", "Hardswish"]
        )
    if m1 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m1_args = [m1_out_channel, 3, m1_stride, None, 1, m1_activation]
    elif m1 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m1_args = [m1_out_channel, 3, m1_stride, None, m1_activation]
    model.append([m1_repeat, m1, m1_args])

    # Module 2    
    m2 = trial.suggest_categorical(
        "m2",
        [
        "InvertedResidualv2",
        "InvertedResidualv3",
        "MBConv",
        "Pass"
        ]
        )
    m2_args = []
    m2_repeat = trial.suggest_int("m2/repeat", 1, 3)
    m2_out_channel = trial.suggest_int("m2/out_channels", low=16, high=64, step=8)
    m2_stride = trial.suggest_int("m2/stride", low=1, high=UPPER_STRIDE)
    # force stride m2
    if n_stride == 0:
        m2_stride = 2
    if m2 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m2_kernel = trial.suggest_int("m2/kernel_size", low=1, high=5, step=2)
        m2_activation = trial.suggest_categorical("m2/activation", ["ReLU", "Hardswish"])
        m2_args = [m2_out_channel, m2_kernel, m2_stride, None, 1, m2_activation]
    elif m2 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m2_kernel = trial.suggest_int("m2/kernel_size", low=1, high=5, step=2)
        m2_activation = trial.suggest_categorical("m2/activation", ["ReLU", "Hardswish"])
        m2_args = [m2_out_channel, m2_kernel, m2_stride, None, m2_activation]
    elif m2 == "InvertedResidualv2":
        m2_c = trial.suggest_int("m2/v2_c", low=16, high=32, step=16)
        m2_t = trial.suggest_int("m2/v2_t", low=1, high=4)
        m2_args = [m2_c, m2_t, m2_stride]
    elif m2 == "InvertedResidualv3":
        m2_kernel = trial.suggest_int("m2/kernel_size", low=3, high=5, step=2)
        m2_t = round(trial.suggest_float("m2/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m2_c = trial.suggest_int("m2/v3_c", low=16, high=40, step=8)
        m2_se = trial.suggest_categorical("m2/v3_se", [0, 1])
        m2_hs = trial.suggest_categorical("m2/v3_hs", [0, 1])
        # k t c SE HS s
        m2_args = [m2_kernel, m2_t, m2_c, m2_se, m2_hs, m2_stride]
    elif m2 == "Fire":
        m2_squeeze = trial.suggest_int("m2/squeeze_planes_size", low=8, high=16, step=4)
        m2_expand1 = trial.suggest_int("m2/expand1x1_size", low=2*m2_squeeze, high=2*2*m2_squeeze, step=4)
        m2_expand3 = trial.suggest_int("m2/expand3x3_size", low=2*m2_squeeze, high=2*2*m2_squeeze, step=4)
        m2_args = [m2_squeeze, m2_expand1, m2_expand3]
    elif m2 == "MBConv":
        m2_t = trial.suggest_int("m2/expand_ratio", low=2, high=14, step=4)
        m2_c = trial.suggest_int("m2/c_size", low=8, high=32, step=8)
        m2_k = trial.suggest_int("m2/kernel_size", low=3, high=5, step=2)
        m2_args = [m2_t, m2_c, m2_stride, m2_k]
    elif m2 == "MaxPool" or m2 == "AvgPool":
        m2_args = [2]
    if not m2 == "Pass":
        if m2_stride == 2:
            n_stride += 1
            if n_stride>=MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([1, m2, m2_args])

    # Module 3
    m3 = trial.suggest_categorical(
        "m3",
        [
        "InvertedResidualv2",
        "InvertedResidualv3",
        "MBConv",
        "Pass"
        ]
        )
    m3_args = []
    m3_repeat = trial.suggest_int("m3/repeat", 1, 3)
    m3_stride = trial.suggest_int("m3/stride", low=1, high=UPPER_STRIDE)
    if m3 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m3_out_channel = trial.suggest_int("m3/out_channels", low=16, high=128, step=16)
        m3_kernel = trial.suggest_int("m3/kernel_size", low=1, high=5, step=2)
        m3_activation = trial.suggest_categorical("m3/activation", ["ReLU", "Hardswish"])
        m3_args = [m3_out_channel, m3_kernel, m3_stride, None, 1, m3_activation]
    elif m3 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m3_out_channel = trial.suggest_int("m3/out_channels", low=16, high=128, step=16)
        m3_kernel = trial.suggest_int("m3/kernel_size", low=1, high=5, step=2)
        m3_activation = trial.suggest_categorical("m3/activation", ["ReLU", "Hardswish"])
        m3_args = [m3_out_channel, m3_kernel, m3_stride, None, m3_activation]
    elif m3 == "InvertedResidualv2":
        m3_c = trial.suggest_int("m3/v2_c", low=8, high=32, step=8)
        m3_t = trial.suggest_int("m3/v2_t", low=1, high=8)
        m3_args = [m3_c, m3_t, m3_stride]
    elif m3 == "InvertedResidualv3":
        m3_kernel = trial.suggest_int("m3/kernel_size", low=3, high=5, step=2)
        m3_t = round(trial.suggest_float("m3/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m3_c = trial.suggest_int("m3/v3_c", low=8, high=40, step=8)
        m3_se = trial.suggest_categorical("m3/v3_se", [0, 1])
        m3_hs = trial.suggest_categorical("m3/v3_hs", [0, 1])
        m3_args = [m3_kernel, m3_t, m3_c, m3_se, m3_hs, m3_stride]
    elif m3 == "Fire":
        m3_squeeze = trial.suggest_int("m3/squeeze_planes_size", low=8, high=16, step=4)
        m3_expand1 = trial.suggest_int("m3/expand1x1_size", low=2*m3_squeeze, high=2*2*m3_squeeze, step=4)
        m3_expand3 = trial.suggest_int("m3/expand3x3_size", low=2*m3_squeeze, high=2*2*m3_squeeze, step=4)
        m3_args = [m3_squeeze, m3_expand1, m3_expand3]
    elif m3 == "MBConv":
        m3_t = trial.suggest_int("m3/expand_ratio", low=2, high=14, step=4)
        m3_c = trial.suggest_int("m3/c_size", low=24, high=64, step=8)
        m3_k = trial.suggest_int("m3/kernel_size", low=3, high=5, step=2)
        m3_args = [m3_t, m3_c, m3_stride, m3_k]
    elif m3 == "MaxPool" or m3 == "AvgPool":
        m3_args = [2]
    if not m3 == "Pass":
        if m3_stride == 2:
            n_stride += 1
            if n_stride>=MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([1, m3, m3_args])

    # Module 4
    m4 = trial.suggest_categorical(
        "m4",
        [
        "InvertedResidualv2",
        "InvertedResidualv3",
        "MBConv",
        "Pass"
        ]
        )
    m4_args = []
    m4_repeat = trial.suggest_int("m4/repeat", 2, 4)
    m4_stride = trial.suggest_int("m4/stride", low=1, high=UPPER_STRIDE)
    # force stride m4
    if n_stride == 1:
        m4_stride = 2
    if m4 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m4_out_channel = trial.suggest_int("m4/out_channels", low=16, high=256, step=16)
        m4_kernel = trial.suggest_int("m4/kernel_size", low=1, high=5, step=2)
        m4_activation = trial.suggest_categorical("m4/activation", ["ReLU", "Hardswish"])
        m4_args = [m4_out_channel, m4_kernel, m4_stride, None, 1, m4_activation]
    elif m4 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m4_out_channel = trial.suggest_int("m4/out_channels", low=16, high=256, step=16)
        m4_kernel = trial.suggest_int("m4/kernel_size", low=1, high=5, step=2)
        m4_activation = trial.suggest_categorical("m4/activation", ["ReLU", "Hardswish"])
        m4_args = [m4_out_channel, m4_kernel, m4_stride, None, m4_activation]
    elif m4 == "InvertedResidualv2":
        m4_c = trial.suggest_int("m4/v2_c", low=8, high=64, step=8)
        m4_t = trial.suggest_int("m4/v2_t", low=1, high=8)
        m4_args = [m4_c, m4_t, m4_stride]
    elif m4 == "InvertedResidualv3":
        m4_kernel = trial.suggest_int("m4/kernel_size", low=3, high=5, step=2)
        m4_t = round(trial.suggest_float("m4/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m4_c = trial.suggest_int("m4/v3_c", low=8, high=80, step=8)
        m4_se = trial.suggest_categorical("m4/v3_se", [0, 1])
        m4_hs = trial.suggest_categorical("m4/v3_hs", [0, 1])
        m4_args = [m4_kernel, m4_t, m4_c, m4_se, m4_hs, m4_stride]
    elif m4 == "Fire":
        m4_squeeze = trial.suggest_int("m4/squeeze_planes_size", low=8, high=16, step=4)
        m4_expand1 = trial.suggest_int("m4/expand1x1_size", low=2*m4_squeeze, high=2*2*m4_squeeze, step=4)
        m4_expand3 = trial.suggest_int("m4/expand3x3_size", low=2*m4_squeeze, high=2*2*m4_squeeze, step=4)
        m4_args = [m4_squeeze, m4_expand1, m4_expand3]
    elif m4 == "MBConv":
        m4_t = trial.suggest_int("m4/expand_ratio", low=2, high=14, step=4)
        m4_c = trial.suggest_int("m4/c_size", low=20, high=120, step=20)
        m4_k = trial.suggest_int("m4/kernel_size", low=3, high=5, step=2)
        m4_args = [m4_t, m4_c, m4_stride, m4_k]
    elif m4 == "MaxPool" or m4 == "AvgPool":
        m4_args = [2]
    if not m4 == "Pass":
        if m4_stride == 2:
            n_stride += 1
            if n_stride>=MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([1, m4, m4_args])

    # Module 5
    m5 = trial.suggest_categorical(
        "m5",
        [
        "InvertedResidualv2",
        "InvertedResidualv3",
        "MBConv",
        "Pass"
        ]
        )
    m5_args = []
    m5_repeat = trial.suggest_int("m5/repeat", 1, 2)
    m5_stride = 1
    if m5 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m5_out_channel = trial.suggest_int("m5/out_channels", low=16, high=256, step=16)
        m5_kernel = trial.suggest_int("m5/kernel_size", low=1, high=5, step=2)
        m5_activation = trial.suggest_categorical("m5/activation", ["ReLU", "Hardswish"])
        m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
        m5_args = [m5_out_channel, m5_kernel, m5_stride, None, 1, m5_activation]
    elif m5 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m5_out_channel = trial.suggest_int("m5/out_channels", low=16, high=256, step=16)
        m5_kernel = trial.suggest_int("m5/kernel_size", low=1, high=5, step=2)
        m5_activation = trial.suggest_categorical("m5/activation", ["ReLU", "Hardswish"])
        m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
        m5_args = [m5_out_channel, m5_kernel, m5_stride, None, m5_activation]
    elif m5 == "InvertedResidualv2":
        m5_c = trial.suggest_int("m5/v2_c", low=16, high=128, step=16)
        m5_t = trial.suggest_int("m5/v2_t", low=1, high=8)
        m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
        m5_args = [m5_c, m5_t, m5_stride]
    elif m5 == "InvertedResidualv3":
        m5_kernel = trial.suggest_int("m5/kernel_size", low=3, high=5, step=2)
        m5_t = round(trial.suggest_float("m5/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m5_c = trial.suggest_int("m5/v3_c", low=16, high=80, step=16)
        m5_se = trial.suggest_categorical("m5/v3_se", [0, 1])
        m5_hs = trial.suggest_categorical("m5/v3_hs", [0, 1])
        m5_stride = trial.suggest_int("m5/stride", low=1, high=UPPER_STRIDE)
        m5_args = [m5_kernel, m5_t, m5_c, m5_se, m5_hs, m5_stride]
    elif m5 == "Fire":
        m5_squeeze = trial.suggest_int("m5/squeeze_planes_size", low=8, high=16, step=4)
        m5_expand1 = trial.suggest_int("m5/expand1x1_size", low=2*m5_squeeze, high=2*2*m5_squeeze, step=4)
        m5_expand3 = trial.suggest_int("m5/expand3x3_size", low=2*m5_squeeze, high=2*2*m5_squeeze, step=4)
        m5_args = [m5_squeeze, m5_expand1, m5_expand3]
    elif m5 == "MBConv":
        m5_t = trial.suggest_int("m5/expand_ratio", low=2, high=14, step=4)
        m5_c = trial.suggest_int("m5/c_size", low=120, high=360, step=120)
        m5_k = trial.suggest_int("m5/kernel_size", low=3, high=5, step=2)
        m5_args = [m5_t, m5_c, m5_stride, m5_k]
    elif m5 == "MaxPool" or m5 == "AvgPool":
        m5_args = [2]
    if not m5 == "Pass":
        if m5_stride == 2:
            n_stride += 1
            if n_stride>=MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m5_repeat, m5, m5_args])

    # Module 6
    m6 = trial.suggest_categorical(
        "m6",
        [
        "InvertedResidualv2",
        "InvertedResidualv3",
        "MBConv",
        "Pass"
        ]
        )
    m6_args = []
    m6_repeat = trial.suggest_int("m6/repeat", 1, 2)
    m6_stride = trial.suggest_int("m6/stride", low=1, high=UPPER_STRIDE)
    # force stride m6
    if n_stride == 2:
        m6_stride = 2
    if m6 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m6_out_channel = trial.suggest_int("m6/out_channels", low=16, high=512, step=16)
        m6_kernel = trial.suggest_int("m6/kernel_size", low=1, high=5, step=2)
        m6_activation = trial.suggest_categorical("m6/activation", ["ReLU", "Hardswish"])
        m6_args = [m6_out_channel, m6_kernel, m6_stride, None, 1, m6_activation]
    elif m6 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m6_out_channel = trial.suggest_int("m6/out_channels", low=16, high=512, step=16)
        m6_kernel = trial.suggest_int("m6/kernel_size", low=1, high=5, step=2)
        m6_activation = trial.suggest_categorical("m6/activation", ["ReLU", "Hardswish"])
        m6_args = [m6_out_channel, m6_kernel, m6_stride, None, m6_activation]
    elif m6 == "InvertedResidualv2":
        m6_c = trial.suggest_int("m6/v2_c", low=16, high=128, step=16)
        m6_t = trial.suggest_int("m6/v2_t", low=1, high=8)
        m6_args = [m6_c, m6_t, m6_stride]
    elif m6 == "InvertedResidualv3":
        m6_kernel = trial.suggest_int("m6/kernel_size", low=3, high=5, step=2)
        m6_t = round(trial.suggest_float("m6/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m6_c = trial.suggest_int("m6/v3_c", low=16, high=160, step=16)
        m6_se = trial.suggest_categorical("m6/v3_se", [0, 1])
        m6_hs = trial.suggest_categorical("m6/v3_hs", [0, 1])
        m6_args = [m6_kernel, m6_t, m6_c, m6_se, m6_hs, m6_stride]
    elif m6 == "Fire":
        m6_squeeze = trial.suggest_int("m6/squeeze_planes_size", low=8, high=16, step=4)
        m6_expand1 = trial.suggest_int("m6/expand1x1_size", low=2*m6_squeeze, high=2*2*m6_squeeze, step=4)
        m6_expand3 = trial.suggest_int("m6/expand3x3_size", low=2*m6_squeeze, high=2*2*m6_squeeze, step=4)
        m6_args = [m6_squeeze, m6_expand1, m6_expand3]
    elif m6 == "MBConv":
        m6_t = trial.suggest_int("m6/expand_ratio", low=2, high=14, step=4)
        m6_c = trial.suggest_int("m6/c_size", low=240, high=360, step=120)
        m6_k = trial.suggest_int("m6/kernel_size", low=3, high=5, step=2)
        m6_args = [m6_t, m6_c, m6_stride, m6_k]
    elif m6 == "MaxPool" or m6 == "AvgPool":
        m6_args = [2]
    if not m6 == "Pass":
        if m6_stride == 2:
            n_stride += 1
            if n_stride>=MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m6_repeat, m6, m6_args])

    # Module 7
    m7 = trial.suggest_categorical(
        "m7",
        [
        "InvertedResidualv2",
        "InvertedResidualv3",
        "MBConv",
        "Pass"
        ]
        )
    m7_args = []
    m7_repeat = trial.suggest_int("m7/repeat", 1, 2)
    m7_stride = trial.suggest_int("m7/stride", low=1, high=UPPER_STRIDE)
    if m7 == "Conv":
        # Conv args: [out_channel, kernel_size, stride, padding, groups, activation]
        m7_out_channel = trial.suggest_int("m7/out_channels", low=128, high=1024, step=128)
        m7_kernel = trial.suggest_int("m7/kernel_size", low=1, high=5, step=2)
        m7_activation = trial.suggest_categorical("m7/activation", ["ReLU", "Hardswish"])
        m7_args = [m7_out_channel, m7_kernel, m7_stride, None, 1, m7_activation]
    elif m7 == "DWConv":
        # DWConv args: [out_channel, kernel_size, stride, padding_size, activation]
        m7_out_channel = trial.suggest_int("m7/out_channels", low=128, high=1024, step=128)
        m7_kernel = trial.suggest_int("m7/kernel_size", low=1, high=5, step=2)
        m7_activation = trial.suggest_categorical("m7/activation", ["ReLU", "Hardswish"])
        m7_args = [m7_out_channel, m7_kernel, m7_stride, None, m7_activation]
    elif m7 == "InvertedResidualv2":
        m7_c = trial.suggest_int("m7/v2_c", low=16, high=160, step=16)
        m7_t = trial.suggest_int("m7/v2_t", low=1, high=8)
        m7_args = [m7_c, m7_t, m7_stride]
    elif m7 == "InvertedResidualv3":
        m7_kernel = trial.suggest_int("m7/kernel_size", low=3, high=5, step=2)
        m7_t = round(trial.suggest_float("m7/v3_t", low=1.0, high=6.0, step=0.1), 1)
        m7_c = trial.suggest_int("m7/v3_c", low=8, high=160, step=8)
        m7_se = trial.suggest_categorical("m7/v3_se", [0, 1])
        m7_hs = trial.suggest_categorical("m7/v3_hs", [0, 1])
        m7_args = [m7_kernel, m7_t, m7_c, m7_se, m7_hs, m7_stride]
    elif m7 == "Fire":
        m7_squeeze = trial.suggest_int("m7/squeeze_planes_size", low=8, high=16, step=4)
        m7_expand1 = trial.suggest_int("m7/expand1x1_size", low=2*m7_squeeze, high=2*2*m7_squeeze, step=4)
        m7_expand3 = trial.suggest_int("m7/expand3x3_size", low=2*m7_squeeze, high=2*2*m7_squeeze, step=4)
        m7_args = [m7_squeeze, m7_expand1, m7_expand3]
    elif m7 == "MBConv":
        m7_t = trial.suggest_int("m7/expand_ratio", low=2, high=14, step=4)
        m7_c = trial.suggest_int("m7/c_size", low=240, high=480, step=120)
        m7_k = trial.suggest_int("m7/kernel_size", low=3, high=5, step=2)
        m7_args = [m7_t, m7_c, m7_stride, m7_k]
    elif m7 == "MaxPool" or m7 == "AvgPool":
        m7_args = [2]
    if not m7 == "Pass":
        if m7_stride == 2:
            n_stride += 1
            if n_stride>=MAX_NUM_STRIDE:
                UPPER_STRIDE = 1
        model.append([m7_repeat, m7, m7_args])

    # last layer
    last_dim = trial.suggest_int("last_dim", low=64, high=128, step=16)
    # We can setup fixed structure as well
    model.append([1, "Conv", [last_dim, 1, 1]])
    model.append([1, "GlobalAvgPool", []])
    model.append([1, "FixedConv", [9, 1, 1, None, 1, None]])

    return model

def objective(trial: optuna.trial.Trial, device) -> Tuple[float, int, float]:
    """Optuna objective.

    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    
    model_config = copy.deepcopy(MODEL_CONFIG)
    data_config = copy.deepcopy(DATA_CONFIG)

    # hyperparams: EPOCHS, IMG_SIZE, n_select, BATCH_SIZE
    hyperparams = {
        "EPOCHS": 30,
        "IMG_SIZE": 32,
        "n_select": 2,
        "BATCH_SIZE": 32
    }#search_hyperparam(trial)

    model_config["input_size"] = [hyperparams["IMG_SIZE"], hyperparams["IMG_SIZE"]]
    model_config["backbone"] = search_model(trial)

    data_config["AUG_TRAIN_PARAMS"]["n_select"] = hyperparams["n_select"]
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["EPOCHS"] = hyperparams["EPOCHS"]
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]

    log_dir = os.path.join("exp", f"{str(trial.number)}_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    
    model_instance = Model(model_config, verbose=False)
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    
    if macs > 3e8:
        raise TrialPruned
    
    os.makedirs(log_dir, exist_ok=True)
    # model_config, data_config
    best_loss, best_f1, best_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
        trial_num=trial.number,
    )
    
    return best_f1, macs


def tune(gpu_id: int, storage: Union[str, None] = None, study_name: str = "pstage_automl"):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.MOTPESampler(n_startup_trials=20)
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage=rdb_storage,
        study_name=study_name,
        sampler=sampler,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=30, interval_steps=10),
    )
    n_trials = 100
    complete_trials = []
    while len(complete_trials) != n_trials:
        study.optimize(lambda trial: objective(trial, device), n_trials=n_trials-len(complete_trials))
        pruned_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        complete_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    #best_trial = get_best_trial_with_condition(study)

if __name__ == "__main__":
    #wandb.init(project="automl")
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--storage", default="postgresql://optuna:nysong2001@localhost/optuna", type=str, help="RDB Storage URL for optuna.")
    parser.add_argument("--study-name", default="automl_partial_module_2", type=str, help="Optuna study name.")
    args = parser.parse_args()
    tune(args.gpu, storage=None if args.storage == "" else args.storage, study_name=args.study_name)
