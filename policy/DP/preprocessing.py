#!/usr/bin/env python
"""Diffusion Policy 的输入/输出预处理：MEAN_STD（图像）+ MIN_MAX（状态/动作）。

lerobot 0.6.x 把归一化放在 processor pipeline 里，policy 只吃归一化后的张量。
这里做同样的事，公式与 `lerobot/processor/normalize_processor.py` 的
`_NormalizationMixin._normalize` 逐行一致（eps 默认 1e-8）：

    MEAN_STD  normalize: (x - mean) / (std + eps)      inverse: x * std + mean
    MIN_MAX   normalize: 2 * (x - min) / (max - min) - 1
              inverse:   (x + 1) / 2 * (max - min) + min

（MIN_MAX 的除数在 min == max 时替换成 eps，见 lerobot 同名字段。）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .modeling_diffusion import ACTION, OBS_IMAGES, DiffusionConfig

__all__ = ["DPNormalizer", "batch_from_sample", "load_dataset_stats"]

DEFAULT_EPS = 1e-8


class DPNormalizer:
    """按 feature 存 min/max/mean/std，支持 MEAN_STD 与 MIN_MAX 两种模式。

    stats 结构：{feature_key: {"mean":..., "std":..., "min":..., "max":...}}，
    feature_key 用数据集里的键名（如 `observation.image` / `observation.state` / `action`）。
    """

    def __init__(
        self,
        config: DiffusionConfig,
        stats: dict[str, dict[str, Tensor]],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        eps: float = DEFAULT_EPS,
    ):
        self.config = config
        self.eps = eps
        self.device = torch.device(device)
        self.dtype = dtype
        self.stats: dict[str, dict[str, Tensor]] = {}
        for key, stat in stats.items():
            self.stats[key] = {
                name: torch.as_tensor(value, dtype=dtype, device=self.device) for name, value in stat.items()
            }

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_dataset_stats(cls, config: DiffusionConfig, dataset_stats: dict[str, dict], **kwargs) -> DPNormalizer:
        """用数据集 `meta/stats.json` 构造（训练时 lerobot 就是这么做的）。"""
        stats = {}
        for key in list(config.input_features) + list(config.output_features):
            if key in dataset_stats:
                stats[key] = dataset_stats[key]
        return cls(config, stats, **kwargs)

    @classmethod
    def from_checkpoint_buffers(
        cls, config: DiffusionConfig, norm_stats: dict[str, dict[str, Tensor]], **kwargs
    ) -> DPNormalizer:
        """用 ckpt 里遗留的归一化 buffer 构造（旧格式 ckpt 才有）。"""
        known = list(config.input_features) + list(config.output_features)
        lookup = {k.replace(".", "_"): k for k in known}
        return cls(config, {lookup.get(k, k): v for k, v in norm_stats.items()}, **kwargs)

    # ------------------------------------------------------------------ 工具
    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> DPNormalizer:
        self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        for stat in self.stats.values():
            for name in stat:
                stat[name] = stat[name].to(device=self.device, dtype=self.dtype)
        return self

    def _mode(self, feature_type: str) -> str:
        return self.config.normalization_mapping.get(feature_type, "IDENTITY")

    def _apply(self, tensor: Tensor, key: str, feature_type: str, inverse: bool) -> Tensor:
        mode = self._mode(feature_type)
        if mode == "IDENTITY" or key not in self.stats:
            return tensor
        stat = self.stats[key]

        if mode == "MEAN_STD":
            mean = stat["mean"].to(device=tensor.device, dtype=tensor.dtype)
            std = stat["std"].to(device=tensor.device, dtype=tensor.dtype)
            return tensor * std + mean if inverse else (tensor - mean) / (std + self.eps)

        if mode == "MIN_MAX":
            min_val = stat["min"].to(device=tensor.device, dtype=tensor.dtype)
            max_val = stat["max"].to(device=tensor.device, dtype=tensor.dtype)
            denom = max_val - min_val
            denom = torch.where(denom == 0, torch.full_like(denom, self.eps), denom)
            if inverse:
                return (tensor + 1) / 2 * denom + min_val
            return 2 * (tensor - min_val) / denom - 1

        raise ValueError(f"unsupported normalization mode {mode!r} for {key}")

    # ------------------------------------------------------------------ 前向
    def normalize(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """观测与动作一起归一化（训练用）。"""
        out = dict(batch)
        for key, feature in self.config.input_features.items():
            if key in out:
                out[key] = self._apply(out[key], key, feature.type, inverse=False)
        for key, feature in self.config.output_features.items():
            if key in out:
                out[key] = self._apply(out[key], key, feature.type, inverse=False)
        return out

    def normalize_observations(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        out = dict(batch)
        for key, feature in self.config.input_features.items():
            if key in out:
                out[key] = self._apply(out[key], key, feature.type, inverse=False)
        return out

    def unnormalize_actions(self, actions: Tensor) -> Tensor:
        feature = self.config.output_features.get(ACTION)
        feature_type = feature.type if feature is not None else "ACTION"
        return self._apply(actions, ACTION, feature_type, inverse=True)

    def state_dict(self) -> dict[str, dict[str, Tensor]]:
        return {k: {n: v.detach().cpu() for n, v in s.items()} for k, s in self.stats.items()}


def batch_from_sample(
    config: DiffusionConfig,
    sample: dict,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """把一条数据集样本整理成 DP 的 batch（加 batch 维）。

    sample 里的观测已按 `n_obs_steps` 取了历史（形状 `(S, ...)`），动作已按
    `horizon` 取成 `(T, action_dim)`（越界处由数据集的 `action_is_pad` 标记）。
    """
    batch: dict[str, Tensor] = {}
    for key, feature in config.input_features.items():
        if key not in sample:
            continue
        value = torch.as_tensor(np.ascontiguousarray(sample[key]), dtype=dtype, device=device)
        if feature.type == "VISUAL":
            if value.ndim == 3:  # (C, H, W) -> (1, C, H, W)，n_obs_steps=1
                value = value.unsqueeze(0)
            if value.max() > 1.0 + 1e-6:
                value = value / 255.0
        else:
            if value.ndim == 1:  # (D,) -> (1, D)
                value = value.unsqueeze(0)
        batch[key] = value.unsqueeze(0)  # 补 batch 维

    for key, feature in config.output_features.items():
        if key not in sample:
            continue
        batch[key] = torch.as_tensor(sample[key], dtype=dtype, device=device).unsqueeze(0)
        pad_key = f"{key}_is_pad"
        if pad_key in sample:
            batch[pad_key] = torch.as_tensor(sample[pad_key], dtype=torch.bool, device=device).unsqueeze(0)

    if OBS_IMAGES in sample:
        batch[OBS_IMAGES] = [
            torch.as_tensor(np.ascontiguousarray(img), dtype=dtype, device=device) for img in sample[OBS_IMAGES]
        ]
    return batch


def load_dataset_stats(root: str | Path) -> dict[str, dict]:
    """读 `meta/stats.json`（v3 数据集自带）。"""
    path = Path(root) / "meta" / "stats.json"
    if not path.exists():
        raise FileNotFoundError(f"no stats.json at {path}")
    return json.loads(path.read_text(encoding="utf-8"))
