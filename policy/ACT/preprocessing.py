#!/usr/bin/env python
"""ACT 的输入/输出预处理：MEAN_STD 归一化（图像 / 机器人状态 / 动作）。

lerobot 0.6.x 把归一化从 policy 挪进了 processor pipeline，policy 本身只吃
已经归一化好的张量。这里做同样的事，好处是对拍时可以"同一份归一化输入喂两边"。

来源：lerobot 0.6.2 `src/lerobot/processor/normalize_processor.py`
（`_NormalizationMixin._normalize` 的 MEAN_STD 分支，eps 默认 1e-8）：

    normalize:   (x - mean) / (std + eps)
    inverse:     x * std + mean

注意两边的除数不一致（归一化带 eps、反归一化不带），这是原实现的写法，
这里保持逐字一致，避免反归一化后的动作出现 1e-8 级别的偏差。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from .modeling_act import ACTION, OBS_IMAGES, ACTConfig

__all__ = ["ACTNormalizer", "batch_from_sample", "load_dataset_stats"]

DEFAULT_EPS = 1e-8


def _sanitize(key: str) -> str:
    """`observation.images.top` -> `observation_images_top`（旧 ckpt 的 buffer 命名）。"""
    return key.replace(".", "_")


class ACTNormalizer:
    """按 feature 存放 mean/std，并据此归一化 batch。

    stats 结构：{feature_key: {"mean": Tensor, "std": Tensor}}，feature_key 用
    数据集里的键名（`observation.images.top` / `observation.state` / `action`）。
    """

    def __init__(
        self,
        config: ACTConfig,
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
                "mean": torch.as_tensor(stat["mean"], dtype=dtype, device=self.device),
                "std": torch.as_tensor(stat["std"], dtype=dtype, device=self.device),
            }

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_checkpoint_buffers(
        cls,
        config: ACTConfig,
        norm_stats: dict[str, dict[str, Tensor]],
        **kwargs,
    ) -> ACTNormalizer:
        """用 ckpt 里遗留的 `normalize_inputs.*` buffer 构造（旧版 lerobot 的产物）。

        `norm_stats` 的键是 `observation_images_top` 这种下划线形式，这里还原成
        数据集键名：先按 feature 名精确匹配，匹配不上再按"下划线归一化"匹配。
        """
        known = list(config.input_features) + list(config.output_features)
        lookup = {_sanitize(k): k for k in known}
        stats: dict[str, dict[str, Tensor]] = {}
        for raw_key, stat in norm_stats.items():
            stats[lookup.get(raw_key, raw_key)] = stat
        return cls(config, stats, **kwargs)

    @classmethod
    def from_dataset_stats(cls, config: ACTConfig, dataset_stats: dict[str, dict], **kwargs) -> ACTNormalizer:
        """用数据集 `meta/stats.json` 构造（新训练时 lerobot 就是这么做的）。"""
        stats = {}
        for key in list(config.input_features) + list(config.output_features):
            if key in dataset_stats:
                stats[key] = dataset_stats[key]
        return cls(config, stats, **kwargs)

    @classmethod
    def from_pretrained_dir(cls, path_or_dir: str | Path, **kwargs) -> ACTNormalizer:
        """从 `weights/` 目录直接读 ckpt buffer（config.json 也在同一目录时）。"""
        from .modeling_act import _read_safetensors, split_normalization_buffers

        root = Path(path_or_dir)
        config = ACTConfig.from_dict(json.loads((root / "config.json").read_text(encoding="utf-8")))
        _, norm_stats = split_normalization_buffers(_read_safetensors(root))
        return cls.from_checkpoint_buffers(config, norm_stats, **kwargs)

    # ------------------------------------------------------------------ 工具
    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> ACTNormalizer:
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
        if self._mode(feature_type) != "MEAN_STD" or key not in self.stats:
            return tensor
        stat = self.stats[key]
        mean = stat["mean"].to(device=tensor.device, dtype=tensor.dtype)
        std = stat["std"].to(device=tensor.device, dtype=tensor.dtype)
        if inverse:
            return tensor * std + mean
        return (tensor - mean) / (std + self.eps)

    # ------------------------------------------------------------------ 前向
    def normalize_observations(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """原地风格地返回新的 batch（图像按 `observation.images.<cam>` 逐个处理）。"""
        out = dict(batch)
        for key, feature in self.config.input_features.items():
            if key not in out:
                continue
            out[key] = self._apply(out[key], key, feature.type, inverse=False)
        return out

    def normalize_actions(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        out = dict(batch)
        for key, feature in self.config.output_features.items():
            if key == ACTION and key in out:
                out[key] = self._apply(out[key], key, feature.type, inverse=False)
        return out

    def normalize(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """观测 + 动作一起归一化（训练用）。"""
        return self.normalize_actions(self.normalize_observations(batch))

    def unnormalize_actions(self, actions: Tensor) -> Tensor:
        """模型输出的是归一化空间的动作，执行前要还原。"""
        feature = self.config.output_features.get(ACTION)
        feature_type = feature.type if feature is not None else "ACTION"
        return self._apply(actions, ACTION, feature_type, inverse=True)

    def state_dict(self) -> dict[str, dict[str, Tensor]]:
        return {k: {n: v.detach().cpu() for n, v in s.items()} for k, s in self.stats.items()}


def batch_from_sample(
    config: ACTConfig,
    sample: dict,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor]:
    """把 `dataset.LeRobotV3Dataset` 的一条样本整理成 ACT 的 batch（加 batch 维）。

    sample 里的键就是数据集特征名：`observation.images.top` / `observation.state` /
    `action` / `action_is_pad`，值为 numpy 数组（图像为 (H, W, C) uint8 或 (C, H, W) float）。
    """
    import numpy as np

    batch: dict[str, Tensor] = {}
    for key, feature in config.input_features.items():
        if key not in sample:
            continue
        value = sample[key]
        if feature.type == "VISUAL":
            arr = value
            if arr.ndim == 3 and arr.shape[-1] == 3 and feature.shape[0] == 3:  # (H, W, C) -> (C, H, W)
                arr = arr.transpose(2, 0, 1)
            tensor = torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype)
            if tensor.max() > 1.0 + 1e-6:
                tensor = tensor / 255.0
            batch[key] = tensor.unsqueeze(0).to(device)
        else:
            batch[key] = torch.as_tensor(value, dtype=dtype, device=device).reshape(1, -1)

    for key, feature in config.output_features.items():
        if key not in sample:
            continue
        batch[key] = torch.as_tensor(sample[key], dtype=dtype, device=device).unsqueeze(0)
        pad_key = f"{key}_is_pad"
        if pad_key in sample:
            batch[pad_key] = torch.as_tensor(sample[pad_key], dtype=torch.bool, device=device).unsqueeze(0)

    if OBS_IMAGES in sample:  # 已经拼好的多相机列表
        batch[OBS_IMAGES] = [
            torch.as_tensor(np.ascontiguousarray(img), dtype=dtype, device=device).unsqueeze(0)
            for img in sample[OBS_IMAGES]
        ]
    return batch


def load_dataset_stats(root: str | Path) -> dict[str, dict]:
    """读 `meta/stats.json`（v3 数据集自带）。"""
    path = Path(root) / "meta" / "stats.json"
    if not path.exists():
        raise FileNotFoundError(f"no stats.json at {path}")
    return json.loads(path.read_text(encoding="utf-8"))
