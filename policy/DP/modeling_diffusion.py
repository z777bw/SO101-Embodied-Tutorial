#!/usr/bin/env python
"""Diffusion Policy —— 独立实现（不依赖 lerobot / diffusers / einops）。

来源：
  * lerobot 0.6.2 `src/lerobot/policies/diffusion/{modeling_diffusion,configuration_diffusion}.py`
    （Apache-2.0，作为数值对拍的基准）；
  * 官方实现 https://github.com/real-stanford/diffusion_policy
    （`policy/diffusion_unet_hybrid_image_policy.py`、`model/diffusion/conditional_unet1d.py`）。

与官方 / lerobot 的差别只在"外壳"：
  * 噪声调度器（DDPM / DDIM）手写，公式逐行对齐 `diffusers`，因此不再依赖 diffusers；
  * `einops` 的 rearrange 换成等价的 permute/reshape；
  * 视觉主干复用 `policy.ACT.modeling_act._ResNet`（同一套手写 ResNet，已与 torchvision 对拍），
    这里用 `nn.BatchNorm2d`（可训练），可再替换成 GroupNorm（`use_group_norm`）；
  * resize / crop 用 `F.interpolate` + 切片实现，替代 torchvision.transforms；
  * 归一化不在 policy 内（lerobot 0.6.x 已移到 processor），见 `preprocessing.py`。

网络结构（与 lerobot 一致，命名也一致，可直接加载官方 ckpt）：
    DiffusionPolicy
      └── diffusion (DiffusionModel)
            ├── unet (DiffusionConditionalUnet1d)        # 1D conv U-Net + FiLM
            ├── rgb_encoder[.i] (DiffusionRgbEncoder)    # ResNet18 + SpatialSoftmax
            └── noise_scheduler (DDPMScheduler / DDIMScheduler)
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# 单文件复用的公共件：手写 ResNet 主干 + 特征描述
from policy.ACT.modeling_act import _RESNET_ARCHS, PolicyFeature, _ResNet

OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"
OBS_IMAGES = "observation.images"
ACTION = "action"

__all__ = [
    "ACTION",
    "DDIMScheduler",
    "DDPMScheduler",
    "DiffusionConfig",
    "DiffusionModel",
    "DiffusionPolicy",
    "PolicyFeature",
    "build_diffusion_from_yaml",
    "load_diffusion_policy",
    "split_normalization_buffers",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DiffusionConfig:
    """与 lerobot `DiffusionConfig` 同名字段（默认值相同，pusht 配置）。"""

    # 输入 / 输出结构
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 32
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    normalization_mapping: dict[str, str] = field(
        default_factory=lambda: {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}
    )
    # 原实现不采样最后 7 帧，避免大量 padding
    drop_n_last_frames: int = 7

    # 视觉主干
    vision_backbone: str = "resnet18"
    resize_shape: tuple[int, int] | None = None
    crop_ratio: float = 1.0
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = False
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True

    # U-Net
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    gradient_checkpointing: bool = False

    # 噪声调度器
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # 推理
    num_inference_steps: int | None = None

    # 训练
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    do_mask_loss_for_padding: bool = False
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # 运行环境 / 标注（不参与建模）
    device: str = "cpu"
    preset: str | None = None
    fps: float | None = None

    def __post_init__(self):
        self.input_features = _parse_features(self.input_features)
        self.output_features = _parse_features(self.output_features)
        self.resize_shape = tuple(self.resize_shape) if self.resize_shape is not None else None
        self.crop_shape = tuple(self.crop_shape) if self.crop_shape is not None else None
        self.down_dims = tuple(self.down_dims)

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}.")
        if self.prediction_type not in ("epsilon", "sample"):
            raise ValueError(f"`prediction_type` must be epsilon/sample. Got {self.prediction_type}.")
        if self.noise_scheduler_type not in ("DDPM", "DDIM"):
            raise ValueError(f"`noise_scheduler_type` must be DDPM/DDIM. Got {self.noise_scheduler_type}.")
        if self.resize_shape is not None and (
            len(self.resize_shape) != 2 or any(d <= 0 for d in self.resize_shape)
        ):
            raise ValueError(f"`resize_shape` must be a pair of positive integers. Got {self.resize_shape}.")
        if not (0 < self.crop_ratio <= 1.0):
            raise ValueError(f"`crop_ratio` must be in (0, 1]. Got {self.crop_ratio}.")

        if self.resize_shape is not None:
            if self.crop_ratio < 1.0:
                self.crop_shape = (
                    int(self.resize_shape[0] * self.crop_ratio),
                    int(self.resize_shape[1] * self.crop_ratio),
                )
            else:
                self.crop_shape = None
        if self.crop_shape is not None and (self.crop_shape[0] <= 0 or self.crop_shape[1] <= 0):
            raise ValueError(f"`crop_shape` must have positive dimensions. Got {self.crop_shape}.")

        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                f"`horizon` must be a multiple of 2**len(down_dims)={downsampling_factor}. "
                f"Got horizon={self.horizon}, down_dims={self.down_dims}"
            )
        self.validate_features()

    # --- 特征查询 ---
    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        for name, ft in (self.input_features or {}).items():
            if ft.type == "STATE" and name == OBS_STATE:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        for ft in (self.input_features or {}).values():
            if ft.type == "ENV":
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {k: ft for k, ft in (self.input_features or {}).items() if ft.type == "VISUAL"}

    @property
    def action_feature(self) -> PolicyFeature | None:
        for name, ft in (self.output_features or {}).items():
            if ft.type == "ACTION" and name == ACTION:
                return ft
        return None

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")
        if self.resize_shape is None and self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )
        if len(self.image_features) > 0:
            first_key, first_ft = next(iter(self.image_features.items()))
            for key, ft in self.image_features.items():
                if ft.shape != first_ft.shape:
                    raise ValueError(f"`{key}` does not match `{first_key}`, all image shapes must be equal.")

    # --- 数据集采样窗口 ---
    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    # --- 序列化 ---
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiffusionConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> DiffusionConfig:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if "policy" in data:
            fields = dict(data["policy"] or {})
            meta = {k: v for k, v in data.items() if k != "policy"}
        else:
            fields, meta = dict(data), {}
        known = set(cls.__dataclass_fields__)
        unknown = set(fields) - known
        if unknown:
            raise ValueError(f"unknown Diffusion config field(s) in {path}: {sorted(unknown)}")
        fields.setdefault("preset", meta.get("name"))
        fields.setdefault("fps", meta.get("fps"))
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name.endswith("_features"):
                value = {k: {"type": ft.type, "shape": list(ft.shape)} for k, ft in value.items()}
            out[name] = value
        out["type"] = "diffusion"
        return out


def _parse_features(features: dict[str, Any] | None) -> dict[str, PolicyFeature]:
    if not features:
        return {}
    parsed = {}
    for key, ft in features.items():
        if isinstance(ft, PolicyFeature):
            parsed[key] = PolicyFeature(type=ft.type, shape=tuple(ft.shape))
        else:
            parsed[key] = PolicyFeature(type=str(ft["type"]), shape=tuple(ft["shape"]))
    return parsed


# ---------------------------------------------------------------------------
# 噪声调度器（DDPM / DDIM，公式与 diffusers 逐行一致）
# ---------------------------------------------------------------------------
def _betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta: float = 0.999) -> Tensor:
    """cosine 噪声表（`squaredcos_cap_v2`），与 diffusers 的 `betas_for_alpha_bar` 相同。"""

    def alpha_bar_fn(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


def _make_betas(
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
    beta_schedule: str,
) -> Tensor:
    if beta_schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
    if beta_schedule == "scaled_linear":
        return torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
    if beta_schedule == "squaredcos_cap_v2":
        return _betas_for_alpha_bar(num_train_timesteps)
    raise ValueError(f"unsupported beta_schedule {beta_schedule!r}")


def _randn_tensor(
    shape: tuple[int, ...],
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


class _NoiseScheduler:
    """DDPM / DDIM 的公共部分（beta 表、add_noise、timesteps）。"""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
        prediction_type: str = "epsilon",
    ):
        self.config = SimpleNamespace(
            num_train_timesteps=num_train_timesteps,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            prediction_type=prediction_type,
        )
        self.betas = _make_betas(num_train_timesteps, beta_start, beta_end, beta_schedule)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0)
        self.one = torch.tensor(1.0)
        self.init_noise_sigma = 1.0
        self.num_inference_steps: int | None = None
        self.timesteps: Tensor | None = None

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        """均匀取点（`timestep_spacing="leading"`，与 diffusers 默认一致）。"""
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps` ({num_inference_steps}) cannot exceed "
                f"`num_train_timesteps` ({self.config.num_train_timesteps})."
            )
        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise


class DDPMScheduler(_NoiseScheduler):
    """DDPM（Ho et al. 2020），`variance_type="fixed_small"`，采样时逐步注入噪声。"""

    def __init__(self, *args, variance_type: str = "fixed_small", **kwargs):
        super().__init__(*args, **kwargs)
        if variance_type != "fixed_small":
            raise ValueError(f"only variance_type='fixed_small' is implemented, got {variance_type!r}")
        self.variance_type = variance_type

    def previous_timestep(self, timestep: Tensor | int) -> Tensor | int:
        if self.num_inference_steps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                return torch.tensor(-1)
            return self.timesteps[index + 1]
        return timestep - 1

    def _get_variance(self, t: Tensor | int) -> Tensor:
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        return torch.clamp(variance, min=1e-20)

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor | int,
        sample: Tensor,
        generator: torch.Generator | None = None,
    ) -> SimpleNamespace:
        t = timestep
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(f"unsupported prediction_type {self.config.prediction_type!r}")

        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        variance = 0
        if t > 0:
            variance_noise = _randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            variance = (self._get_variance(t) ** 0.5) * variance_noise

        return SimpleNamespace(prev_sample=pred_prev_sample + variance, pred_original_sample=pred_original_sample)


class DDIMScheduler(_NoiseScheduler):
    """DDIM（Song et al. 2020），`eta=0` 时是确定性的。"""

    def __init__(self, *args, set_alpha_to_one: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_alpha_cumprod = self.one if set_alpha_to_one else self.alphas_cumprod[0]

    def _get_variance(self, timestep: Tensor | int, prev_timestep: Tensor | int) -> Tensor:
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return torch.clamp(variance, min=1e-20)

    def step(
        self,
        model_output: Tensor,
        timestep: Tensor | int,
        sample: Tensor,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> SimpleNamespace:
        if self.num_inference_steps is None:
            raise ValueError("call `set_timesteps` before `step`")

        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        else:
            raise ValueError(f"unsupported prediction_type {self.config.prediction_type!r}")

        if self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** 0.5

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        if eta > 0:
            variance_noise = _randn_tensor(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample + std_dev_t * variance_noise

        return SimpleNamespace(prev_sample=prev_sample, pred_original_sample=pred_original_sample)


def _make_noise_scheduler(config: DiffusionConfig) -> _NoiseScheduler:
    kwargs = {
        "num_train_timesteps": config.num_train_timesteps,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "beta_schedule": config.beta_schedule,
        "clip_sample": config.clip_sample,
        "clip_sample_range": config.clip_sample_range,
        "prediction_type": config.prediction_type,
    }
    if config.noise_scheduler_type == "DDPM":
        return DDPMScheduler(**kwargs)
    if config.noise_scheduler_type == "DDIM":
        return DDIMScheduler(**kwargs)
    raise ValueError(f"Unsupported noise scheduler type {config.noise_scheduler_type}")


def _action_dim(config: DiffusionConfig) -> int:
    action_feature = config.action_feature
    if action_feature is None:
        raise ValueError("Diffusion Policy requires an action output feature (FeatureType.ACTION) in `output_features`.")
    return action_feature.shape[0]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class DiffusionPolicy(nn.Module):
    """Diffusion Policy 完整策略：`self.diffusion` 是网络，外面包一层动作队列。"""

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        config.validate_features()
        self.config = config
        self._queues: dict[str, deque[Tensor]] = {}
        self.diffusion = DiffusionModel(config)
        self.reset()

    def get_optim_params(self):
        return self.diffusion.parameters()

    def reset(self):
        """清空观测/动作队列，每个 episode 开始时调用。"""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """离线批（队列为空）或在线滚动（队列非空）都支持，返回 (B, n_action_steps, action_dim)。"""
        queues_populated = any(len(q) > 0 for q in self._queues.values())
        if queues_populated:
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        else:
            batch = dict(batch)
            if self.config.image_features:
                for key in self.config.image_features:
                    if batch[key].ndim == 4:
                        batch[key] = batch[key].unsqueeze(1)
                batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return self.diffusion.generate_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """每次调用吐一个动作；动作块用完之前不重复前向。"""
        if ACTION in batch:
            batch = dict(batch)  # 不要改调用方的 dict
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # 注意：要在把多相机拼成一个 key 之后再入队
        self._queues = _populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        if self.config.image_features:
            batch = dict(batch)
            for key in self.config.image_features:
                if self.config.n_obs_steps == 1 and batch[key].ndim == 4:
                    batch[key] = batch[key].unsqueeze(1)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.diffusion.compute_loss(batch)
        return loss, None


def _populate_queues(queues: dict[str, deque], batch: dict[str, Tensor]) -> dict[str, deque]:
    """第一次调用时把当前观测复制若干份填满队列（等价 lerobot `populate_queues`）。"""
    for key in list(batch):
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            queues[key].append(batch[key])
    return queues


# ---------------------------------------------------------------------------
# 网络本体
# ---------------------------------------------------------------------------
class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config

        robot_state_feature = config.robot_state_feature
        if robot_state_feature is None:
            raise ValueError(
                "Diffusion Policy requires a robot state input feature (FeatureType.STATE) in `input_features`."
            )

        global_cond_dim = robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.noise_scheduler = _make_noise_scheduler(config)
        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    # ---------------- 推理 ----------------
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, _action_dim(self.config)),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample
        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """把状态、图像特征、环境状态拼成 (B, global_cond_dim)。"""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            images = batch[OBS_IMAGES]  # (B, S, N, C, H, W)
            num_cameras = images.shape[2]
            if self.config.use_separate_rgb_encoder_per_camera:
                # b s n ... -> n (b s) ...
                per_camera = images.permute(2, 0, 1, *range(3, images.ndim))
                per_camera = per_camera.reshape(num_cameras * batch_size * n_obs_steps, *images.shape[3:])
                encoded = torch.cat(
                    [encoder(per_camera[i * batch_size * n_obs_steps : (i + 1) * batch_size * n_obs_steps])
                     for i, encoder in enumerate(self.rgb_encoder)],
                    dim=0,
                )
                # (n b s) d -> b s (n d)
                img_features = encoded.reshape(num_cameras, batch_size, n_obs_steps, -1)
                img_features = img_features.permute(1, 2, 0, 3).flatten(2)
            else:
                # b s n ... -> (b s n) ...
                flat = images.reshape(batch_size * n_obs_steps * num_cameras, *images.shape[3:])
                img_features = self.rgb_encoder(flat)
                # (b s n) d -> b s (n d)
                img_features = img_features.reshape(batch_size, n_obs_steps, -1)
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    # ---------------- 训练 ----------------
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        trajectory = batch[ACTION]
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            mask = in_episode_bound.unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            return (loss * mask).sum() / num_valid.clamp_min(1)

        return loss.mean()


class SpatialSoftmax(nn.Module):
    """Spatial soft-argmax：每个通道输出一个归一化到 [-1, 1] 的期望坐标。"""

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # 用 numpy 生成网格：torch.linspace 在这里会有微小数值差异，影响预训练权重表现
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)

        features = features.reshape(-1, self._in_h * self._in_w)  # (B*K, H*W)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid  # (B*K, 2)
        return expected_xy.view(-1, self._out_c, 2)


class DiffusionRgbEncoder(nn.Module):
    """ResNet 主干 + SpatialSoftmax + 线性层，把一帧图像压成一个特征向量。"""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.resize_shape = config.resize_shape
        self.crop_shape = config.crop_shape
        self.crop_is_random = bool(config.crop_is_random and config.crop_shape is not None)

        # 手写 ResNet（BN 可训练），取到 layer4 为止
        if config.vision_backbone not in _RESNET_ARCHS:
            raise ValueError(
                f"unsupported vision_backbone {config.vision_backbone!r}; available: {sorted(_RESNET_ARCHS)}"
            )
        block, layers = _RESNET_ARCHS[config.vision_backbone]
        trunk = _ResNet(block, layers, norm_layer=nn.BatchNorm2d)
        self.backbone = nn.Sequential(*list(trunk.children()))

        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("You can't replace BatchNorm in a pretrained model without ruining the weights!")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        if config.pretrained_backbone_weights:
            _load_torchvision_trunk_weights(self.backbone, config.vision_backbone, config.pretrained_backbone_weights)

        # dry run 得到 layer4 的 feature map 尺寸
        images_shape = next(iter(config.image_features.values())).shape
        if config.crop_shape is not None:
            dummy_hw = config.crop_shape
        elif config.resize_shape is not None:
            dummy_hw = config.resize_shape
        else:
            dummy_hw = images_shape[1:]
        dummy = torch.zeros(1, images_shape[0], *dummy_hw)
        with torch.inference_mode():
            feature_map_shape = tuple(self.backbone(dummy).shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def _apply_crop(self, x: Tensor) -> Tensor:
        if not self.crop_is_random:
            return _center_crop(x, self.crop_shape)
        # 与 torchvision 的 RandomCrop.get_params 相同：先抽 i、再抽 j
        h = x.shape[-2] - self.crop_shape[0] + 1
        w = x.shape[-1] - self.crop_shape[1] + 1
        i = torch.randint(0, h, size=(1,)).item()
        j = torch.randint(0, w, size=(1,)).item()
        return x[..., i : i + self.crop_shape[0], j : j + self.crop_shape[1]]

    def forward(self, x: Tensor) -> Tensor:
        if self.resize_shape is not None:
            x = F.interpolate(x, size=tuple(self.resize_shape), mode="bilinear", align_corners=False, antialias=True)
        if self.crop_shape is not None:
            x = self._apply_crop(x) if self.training else _center_crop(x, self.crop_shape)

        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        return self.relu(self.out(x))


def _center_crop(x: Tensor, crop_shape: tuple[int, int]) -> Tensor:
    ch, cw = crop_shape
    top = (x.shape[-2] - ch) // 2
    left = (x.shape[-1] - cw) // 2
    return x[..., top : top + ch, left : left + cw]


def _replace_submodules(
    root_module: nn.Module,
    predicate,
    func,
) -> nn.Module:
    """把满足 `predicate` 的子模块替换成 `func(module)`（相当于 lerobot 的同名工具）。"""
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


def _load_torchvision_trunk_weights(backbone: nn.Sequential, arch: str, weights_ref: str) -> None:
    """把 torchvision 预训练主干拷进手写 ResNet（只在这条路径上用 torchvision）。"""
    try:
        from torchvision import models
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "loading `pretrained_backbone_weights` requires torchvision as a weight source; "
            "install torchvision or set the field to null"
        ) from exc

    reference = getattr(models, arch)(weights=weights_ref)
    trunk = {
        k: v
        for k, v in reference.state_dict().items()
        if not k.startswith("fc.") and not k.endswith("num_batches_tracked")
    }
    # 目标 backbone 是 nn.Sequential(conv1, bn1, relu, maxpool, layer1..4)，
    # 键名形如 "0.weight" / "4.0.conv1.weight"，与 torchvision 的前缀一一对应
    rename = {"conv1": "0", "bn1": "1", "layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7"}
    mapped = {}
    for key, value in trunk.items():
        head, _, rest = key.partition(".")
        if head in rename:
            mapped[f"{rename[head]}.{rest}" if rest else rename[head]] = value
    backbone.load_state_dict(mapped, strict=False)


class DiffusionSinusoidalPosEmb(nn.Module):
    """扩散步 t 的正弦位置编码。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DiffusionConv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish。"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalResidualBlock1d(nn.Module):
    """带 FiLM 条件调制的 1D 残差块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


class DiffusionConditionalUnet1d(nn.Module):
    """1D 卷积 U-Net，FiLM 调制 global conditioning（无 local conditioning）。"""

    def __init__(self, config: DiffusionConfig, global_cond_dim: int) -> None:
        super().__init__()
        self.config = config
        action_dim = _action_dim(config)

        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        in_out = [(action_dim, config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        common = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common),
                DiffusionConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], action_dim, 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        x = x.permute(0, 2, 1)  # (B, T, D) -> (B, D, T)

        timesteps_embed = self.diffusion_step_encoder(timestep)
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        from torch.utils.checkpoint import checkpoint

        use_gc = self.config.gradient_checkpointing and self.training

        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            if use_gc:
                x = checkpoint(resnet, x, global_feature, use_reentrant=False)
                x = checkpoint(resnet2, x, global_feature, use_reentrant=False)
            else:
                x = resnet(x, global_feature)
                x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = checkpoint(mid_module, x, global_feature, use_reentrant=False) if use_gc else mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            if use_gc:
                x = checkpoint(resnet, x, global_feature, use_reentrant=False)
                x = checkpoint(resnet2, x, global_feature, use_reentrant=False)
            else:
                x = resnet(x, global_feature)
                x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)


# ---------------------------------------------------------------------------
# 权重加载
# ---------------------------------------------------------------------------
_NORM_BUFFER_PREFIXES = ("normalize_inputs.", "normalize_targets.", "unnormalize_outputs.")


def _read_safetensors(path_or_dir: str | Path) -> dict[str, Tensor]:
    from safetensors.torch import load_file

    path = Path(path_or_dir)
    if path.is_dir():
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no *.safetensors in {path}")
        state_dict: dict[str, Tensor] = {}
        for shard in shards:
            state_dict.update(load_file(str(shard)))
        return state_dict
    return load_file(str(path))


def split_normalization_buffers(state_dict: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, dict[str, Tensor]]]:
    """剥离旧 ckpt 里内嵌的归一化 buffer（若有），返回 (模型权重, 统计量)。"""
    import re

    model_sd, norm_sd = {}, {}
    for key, tensor in state_dict.items():
        if key.startswith(_NORM_BUFFER_PREFIXES):
            m = re.match(
                r"^(?:normalize_inputs|normalize_targets|unnormalize_outputs)\.buffer_(.+)?\.(mean|std|min|max)$", key
            )
            if m is None:
                model_sd[key] = tensor
                continue
            norm_sd.setdefault(m.group(1), {})[m.group(2)] = tensor
        else:
            model_sd[key] = tensor
    return model_sd, norm_sd


def load_diffusion_policy(
    path_or_dir: str | Path,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[DiffusionPolicy, dict[str, dict[str, Tensor]]]:
    """从 `config.json + model.safetensors` 构造 DiffusionPolicy 并加载权重。"""
    root = Path(path_or_dir)
    with open(root / "config.json", encoding="utf-8") as f:
        config = DiffusionConfig.from_dict(json.load(f))
    # 所有参数马上会被 ckpt 覆盖，避免为此下载 ImageNet 主干权重
    config.pretrained_backbone_weights = None
    policy = DiffusionPolicy(config)
    policy.to(device=device, dtype=dtype)

    state_dict, norm_stats = split_normalization_buffers(_read_safetensors(root))
    report = policy.load_state_dict(state_dict, strict=False)
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={report.missing_keys[:8]} unexpected={report.unexpected_keys[:8]}"
        )
    policy.eval()
    return policy, norm_stats


def build_diffusion_from_yaml(path: str | Path, *, device: str | None = None) -> DiffusionPolicy:
    """按 `config/DP/<preset>.yaml` 构造（随机初始化）。"""
    config = DiffusionConfig.from_yaml(path)
    policy = DiffusionPolicy(config)
    return policy.to(device=device or config.device)
