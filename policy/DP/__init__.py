"""Diffusion Policy（扩散策略）独立实现。

用法见 `policy/DP/README.md`：

    from policy.DP import load_diffusion_policy, DPNormalizer, load_dataset_stats

    policy, _ = load_diffusion_policy("weights/diffusion_pusht")
    normalizer = DPNormalizer.from_dataset_stats(policy.config, load_dataset_stats("dataset/data/pusht"))
"""

from .modeling_diffusion import (
    ACTION,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionConfig,
    DiffusionModel,
    DiffusionPolicy,
    PolicyFeature,
    build_diffusion_from_yaml,
    load_diffusion_policy,
    split_normalization_buffers,
)
from .preprocessing import DPNormalizer, batch_from_sample, load_dataset_stats

__all__ = [
    "ACTION",
    "DDIMScheduler",
    "DDPMScheduler",
    "DPNormalizer",
    "DiffusionConfig",
    "DiffusionModel",
    "DiffusionPolicy",
    "PolicyFeature",
    "batch_from_sample",
    "build_diffusion_from_yaml",
    "load_dataset_stats",
    "load_diffusion_policy",
    "split_normalization_buffers",
]
