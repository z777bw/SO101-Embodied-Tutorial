#!/usr/bin/env python
"""Action Chunking Transformer (ACT) —— 独立实现（不依赖 lerobot / transformers）。

来源：lerobot 0.6.2 `src/lerobot/policies/act/modeling_act.py` 与
`configuration_act.py`（Apache-2.0），按「去依赖 + 去分支」原则搬运：
算子顺序、dtype、mask/dropout 位置、参数命名均与原实现逐行一致，以便
直接加载 hub 上的 `lerobot/act_*` 权重做数值对拍。

与 lerobot 的差异（都是"外壳"层面的）：
  * `ACTConfig` 不再继承 `PreTrainedConfig`，`PolicyFeature` 用本文件内的
    轻量 dataclass 代替（只保留 `type` / `shape`）；
  * 视觉主干（ResNet18/34/50/101/152 + FrozenBatchNorm2d + 取层）全部手写，
    运行时不需要 torchvision；只有显式要求 `pretrained_backbone_weights` 时
    才把 torchvision 当成"权重下载器"用一次；
  * 权重加载手写在 `load_state_dict_from_checkpoint()`，不需要 `safetensors`
    之外的东西；
  * 归一化不再内置在 policy 里（lerobot 0.6.x 已把它移到 processor），
    见 `preprocessing.py`；旧 ckpt 里遗留的 `normalize_inputs.*` /
    `normalize_targets.*` / `unnormalize_outputs.*` buffer 由
    `split_normalization_buffers()` 剥离并返回，可直接用于推理。

batch 约定（与 lerobot 完全一致）：
    batch = {
        "observation.state":          (B, state_dim)                 # 可选
        "observation.environment_state": (B, env_dim)                # 可选
        "observation.images":         list[(B, C, H, W)]             # 至少一项
        "action":                     (B, chunk_size, action_dim)    # 训练时需要
        "action_is_pad":              (B, chunk_size)                # 训练时需要
    }
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# 常量（对应 lerobot.utils.constants）
# ---------------------------------------------------------------------------
OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.environment_state"
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
ACTION = "action"

__all__ = [
    "ACT",
    "ACTION",
    "OBS_ENV_STATE",
    "OBS_IMAGES",
    "OBS_STATE",
    "ACTConfig",
    "ACTDecoder",
    "ACTDecoderLayer",
    "ACTEncoder",
    "ACTEncoderLayer",
    "ACTPolicy",
    "ACTSinusoidalPositionEmbedding2d",
    "ACTTemporalEnsembler",
    "PolicyFeature",
    "build_act_from_config_json",
    "build_act_from_yaml",
    "create_sinusoidal_pos_embedding",
    "get_activation_fn",
    "load_state_dict_from_checkpoint",
    "split_normalization_buffers",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PolicyFeature:
    """`type` 取 "STATE" / "VISUAL" / "ENV" / "ACTION"，`shape` 不含 batch 维。"""

    type: str
    shape: tuple[int, ...]


@dataclass
class ACTConfig:
    """与 lerobot `ACTConfig` 同名的超参集合（默认值相同）。

    多出来的两个字段只做标注、不参与建模：
      * `preset`：这套配置对应哪个本体/数据集（如 "aloha" / "so101"）；
      * `fps`：录制帧率，用来把 `chunk_size` 换算成秒（`chunk_size / fps`）。
    """

    # 输入 / 输出结构
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    normalization_mapping: dict[str, str] = field(
        default_factory=lambda: {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    )

    # 视觉主干
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None
    replace_final_stride_with_dilation: bool = False

    # Transformer
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # 原版 ACT 写的是 7 层，但代码里只用了第 1 层（tonyzhaozh/act#25），故为 1。
    n_decoder_layers: int = 1

    # VAE
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # 推理
    temporal_ensemble_coeff: float | None = None

    # 训练
    dropout: float = 0.1
    kl_weight: float = 10.0
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    # 运行环境（不参与结构，仅用于 .to() 的默认值）
    device: str = "cpu"

    # 仅标注用（不参与建模）：本体名 / 数据集 / 录制帧率
    preset: str | None = None
    fps: float | None = None

    def __post_init__(self):
        # config.json 里的 features 是 {"type": "VISUAL", "shape": [3, 480, 640]} 形式
        self.input_features = _parse_features(self.input_features)
        self.output_features = _parse_features(self.output_features)

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}.")
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError("`n_action_steps` must be 1 when using temporal ensembling.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. "
                f"Got {self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`")
        self.validate_features()

    # --- 特征查询（对应 PreTrainedConfig 的同名 property） ---
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
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    # --- 序列化 ---
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ACTConfig:
        """从 hub 的 config.json（多一个 "type": "act" 键）构造。"""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> ACTConfig:
        """读 `config/ACT/<preset>.yaml`。

        文件结构：

            name: so101            # 元信息：写进 config.preset
            fps: 30                # 元信息：写进 config.fps
            policy:                # 这一段就是 ACTConfig 的字段
              chunk_size: 45
              input_features: {...}

        也接受"没有 policy 段"的扁平写法；出现未知键会直接报错（防止拼写错误被静默忽略）。
        """
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
            raise ValueError(f"unknown ACT config field(s) in {path}: {sorted(unknown)}")
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
        out["type"] = "act"
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
# Policy（训练 loss / 推理接口）
# ---------------------------------------------------------------------------
class ACTPolicy(nn.Module):
    """ACT 的完整策略：`self.model` 是网络本体，外面包一层 loss 与动作队列。"""

    config_class = ACTConfig
    name = "act"

    def __init__(self, config: ACTConfig):
        super().__init__()
        config.validate_features()
        self.config = config
        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    # -- 参数分组：主干单独一个 lr（对应 lerobot get_optim_params） --
    def get_optim_params(self) -> list[dict[str, Any]]:
        return [
            {
                "params": [
                    p for n, p in self.named_parameters() if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """每个 episode 开始时调用。"""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """按环境节奏一次吐一个动作（内部用队列缓存整个 chunk）。"""
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # (B, n_action_steps, D) -> 队列里存 (n_action_steps, B, D)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """一次前向，返回 (B, chunk_size, action_dim)。"""
        self.eval()
        batch = _collate_images(batch, self.config)
        return self.model(batch)[0]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """训练 / 验证一步：L1 重建损失 + KL（VAE 开启时）。"""
        batch = _collate_images(batch, self.config)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae and log_sigma_x2_hat is not None:
            # D_kl(latent_pdf || N(0, I))，逐维求和后在 batch 上取均值（见 VAE 论文附录 B）。
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


def _collate_images(batch: dict[str, Tensor], config: ACTConfig) -> dict[str, Tensor]:
    """把 `observation.images.<cam>` 收成 ACT 需要的 list，放进 OBS_IMAGES。"""
    if config.image_features:
        batch = dict(batch)  # shallow copy，避免污染调用方
        batch[OBS_IMAGES] = [batch[key] for key in config.image_features]
    return batch


class ACTTemporalEnsembler:
    """时间集成：权重 w_i = exp(-c * i)，在线递推求平均（ACT 论文 Algorithm 2）。"""

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """输入 (B, chunk_size, D)，返回当前时刻集成后的 (B, D)。"""
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


# ---------------------------------------------------------------------------
# 网络本体
# ---------------------------------------------------------------------------
class ACT(nn.Module):
    """Action Chunking Transformer。

    术语（与 lerobot 一致）：
      * `vae_encoder`：只在训练时编码目标动作序列，输出 latent 分布参数；
      * `encoder` / `decoder`：真正的 Transformer，decoder 以 chunk 长度的
        query 做 cross-attention，充当 VAE 的解码器（推理时直接用）。
    """

    def __init__(self, config: ACTConfig) -> None:
        super().__init__()
        self.config = config

        action_feature = config.action_feature
        if action_feature is None:
            raise ValueError("ACT requires an action output feature (FeatureType.ACTION) in `output_features`.")

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(action_feature.shape[0], config.dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        if self.config.image_features:
            backbone_model = _build_backbone(config)
            self.backbone = _IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # encoder 的输入 token 顺序：[latent, (state), (env_state), (image patch)]
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(self.config.env_state_feature.shape[0], config.dim_model)
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            # torchvision 里这里写的是 backbone_model.fc.in_features（分类头通道数）；
            # 手写主干没有 fc，用等价的 out_channels（= 512 * block.expansion）
            self.encoder_img_feat_input_proj = nn.Conv2d(backbone_model.out_channels, config.dim_model, kernel_size=1)

        n_1d_tokens = 1  # latent
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # decoder 的 query 位置编码（DETR 风格）
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        self.action_head = nn.Linear(config.dim_model, action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Transformer 部分用 Xavier-uniform（与原版 ACT 一致）。"""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        if self.config.use_vae and self.training:
            assert ACTION in batch, "actions must be provided when using the variational objective in training mode."

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # ---- latent：训练时从 VAE encoder 采样，推理时置零 ----
        if self.config.use_vae and ACTION in batch and self.training:
            cls_embed = self.vae_encoder_cls_embed.weight.repeat(batch_size, 1, 1)  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # key_padding_mask：False 表示不是 padding
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], axis=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # 这里存的是 2log(sigma)，与原实现一致
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # ---- encoder 输入 ----
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            # 多相机时 H、W 可能不同，但 H*W 不变
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = torch.flatten(cam_features.permute(2, 3, 0, 1), start_dim=0, end_dim=1)  # (h w, b, c)
                cam_pos_embed = torch.flatten(cam_pos_embed.permute(2, 3, 0, 1), start_dim=0, end_dim=1)

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)  # (S, B, D)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)  # (B, S, C)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """多层 encoder（可选末端 norm）。"""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # 只要输出，不要 attention 权重
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


# ---------------------------------------------------------------------------
# 位置编码 / 激活函数
# ---------------------------------------------------------------------------
def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D 正弦位置编码，返回 (num_positions, dimension)。"""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D 正弦位置编码，位置索引归一化到 [0, 2π]（下界为 1/H、1/W）。"""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


# ---------------------------------------------------------------------------
# 视觉主干：手写 ResNet（不依赖 torchvision 的模型构造）
#
# 结构与参数名逐字对齐 torchvision.models.resnet*，因此官方 ckpt 的
# `model.backbone.*` 可以 strict 加载；BN 一律换成 _FrozenBatchNorm2d，
# 且只保留到 layer4（分类头 fc 不参与，ACT 只用它的通道数）。
# ---------------------------------------------------------------------------
class _FrozenBatchNorm2d(nn.Module):
    """推理用的 BatchNorm：用 running stats，且不接受梯度更新。"""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        # 注意：官方 ckpt 里没有 num_batches_tracked，这里也不注册，保证键名一一对应。

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class _IntermediateLayerGetter(nn.ModuleDict):
    """返回中间层输出的字典，例如 {"feature_map": layer4_out}。"""

    def __init__(self, model: nn.Module, return_layers: dict[str, str]):
        if not set(return_layers).issubset({name for name, _ in model.named_children()}):
            raise ValueError("return_layers are not present in model")
        self.return_layers = dict(return_layers)
        layers = {}
        remaining = dict(return_layers)
        for name, module in model.named_children():
            layers[name] = module
            remaining.pop(name, None)
            if not remaining:
                break
        super().__init__(layers)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        out = {}
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return out


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _BasicBlock(nn.Module):
    """ResNet-18/34 的残差块：3×3 -> 3×3，expansion = 1。"""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
        norm_layer: type[nn.Module] = _FrozenBatchNorm2d,
    ):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _Bottleneck(nn.Module):
    """ResNet-50/101/152 的残差块：1×1 -> 3×3 -> 1×1，expansion = 4。"""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        dilation: int = 1,
        norm_layer: type[nn.Module] = _FrozenBatchNorm2d,
    ):
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = _conv3x3(planes, planes, stride, dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _ResNet(nn.Module):
    """ResNet 主干（到 layer4 为止），等价于 torchvision 的 `resnet18/34/50/...`。

    `out_channels` 就是 layer4 的输出通道，代替 torchvision 里 `fc.in_features` 的作用。
    """

    def __init__(
        self,
        block: type[_BasicBlock | _Bottleneck],
        layers: list[int],
        *,
        replace_stride_with_dilation: tuple[bool, bool, bool] = (False, False, False),
        norm_layer: type[nn.Module] = _FrozenBatchNorm2d,
    ):
        super().__init__()
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be a 3-element tuple, got {replace_stride_with_dilation}"
            )

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 与 torchvision 一致：replace_stride_with_dilation[i] 对应 layer{i+2}
        self.layer1 = self._make_layer(block, 64, layers[0])
        replace = replace_stride_with_dilation
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace[2])

        self.out_channels = 512 * block.expansion

    def _make_layer(
        self,
        block: type[_BasicBlock | _Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        """逐行照抄 torchvision 的 `_make_layer`（含 `previous_dilation` 这个历史行为）。"""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


_RESNET_ARCHS: dict[str, tuple[type[nn.Module], list[int]]] = {
    "resnet18": (_BasicBlock, [2, 2, 2, 2]),
    "resnet34": (_BasicBlock, [3, 4, 6, 3]),
    "resnet50": (_Bottleneck, [3, 4, 6, 3]),
    "resnet101": (_Bottleneck, [3, 4, 23, 3]),
    "resnet152": (_Bottleneck, [3, 8, 36, 3]),
}


def _build_backbone(config: ACTConfig) -> nn.Module:
    """按 `vision_backbone` 构造手写 ResNet 主干。

    `pretrained_backbone_weights`（如 `"ResNet18_Weights.IMAGENET1K_V1"`）如果给了，
    就用 torchvision **只作为权重来源**：把它的主干 state_dict 拷进来（丢掉分类头
    `fc.*` 和 `num_batches_tracked`）。结构始终是本文件的实现，只有权重来自 torchvision。
    """
    if config.vision_backbone not in _RESNET_ARCHS:
        raise ValueError(
            f"unsupported vision_backbone {config.vision_backbone!r}; "
            f"available: {sorted(_RESNET_ARCHS)}"
        )
    block, layers = _RESNET_ARCHS[config.vision_backbone]
    model = _ResNet(
        block,
        layers,
        replace_stride_with_dilation=(
            False,
            False,
            bool(config.replace_final_stride_with_dilation),
        ),
        norm_layer=_FrozenBatchNorm2d,
    )

    if config.pretrained_backbone_weights:
        _load_torchvision_trunk_weights(model, config.vision_backbone, config.pretrained_backbone_weights)
    return model


def _load_torchvision_trunk_weights(model: _ResNet, arch: str, weights_ref: str) -> None:
    """把 torchvision 预训练主干拷进手写模型（只在这里用到 torchvision）。"""
    try:
        from torchvision import models
    except ImportError as exc:  # pragma: no cover - 只在显式要求预训练权重时才需要
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
    model.load_state_dict(trunk, strict=False)


# ---------------------------------------------------------------------------
# 权重加载
# ---------------------------------------------------------------------------
_NORM_BUFFER_PREFIXES = ("normalize_inputs.", "normalize_targets.", "unnormalize_outputs.")


def split_normalization_buffers(state_dict: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, dict[str, Tensor]]]:
    """剥离旧 ckpt 里内嵌的归一化 buffer。

    返回 (模型权重, {feature_key: {"mean": ..., "std": ...}})，feature_key 采用
    数据集的键名（`observation.images.top` 这类），可直接给 `preprocessing.ACTNormalizer`。
    """
    model_sd, norm_sd = {}, {}
    for key, tensor in state_dict.items():
        if key.startswith(_NORM_BUFFER_PREFIXES):
            # 形如 normalize_inputs.buffer_observation_images_top.mean
            m = re.match(r"^(?:normalize_inputs|normalize_targets|unnormalize_outputs)\.buffer_(.+)?\.(mean|std)$", key)
            if m is None:
                model_sd[key] = tensor
                continue
            feature = m.group(1)
            stat = m.group(2)
            norm_sd.setdefault(feature, {})[stat] = tensor
        else:
            model_sd[key] = tensor
    return model_sd, norm_sd


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


def load_state_dict_from_checkpoint(
    model: nn.Module,
    path_or_dir: str | Path,
    *,
    strict: bool = False,
) -> tuple[list[str], list[str], dict[str, dict[str, Tensor]]]:
    """把 hub 权重灌进 `model`，返回 (missing, unexpected, normalization_stats)。

    缺省 `strict=False`：官方权重常带运行时用不到的 buffer（如归一化统计量），
    这类键会被 `split_normalization_buffers()` 单独拿出来。
    """
    state_dict, norm_stats = split_normalization_buffers(_read_safetensors(path_or_dir))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys), norm_stats


def load_act_policy(
    path_or_dir: str | Path,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[ACTPolicy, dict[str, dict[str, Tensor]]]:
    """从 `config.json + model.safetensors` 构造 ACTPolicy 并加载权重。"""
    root = Path(path_or_dir)
    with open(root / "config.json", encoding="utf-8") as f:
        config = ACTConfig.from_dict(json.load(f))
    # 所有参数马上会被 ckpt 覆盖，没必要再去下载 ImageNet 主干权重
    config.pretrained_backbone_weights = None
    policy = ACTPolicy(config)
    policy.to(device=device, dtype=dtype)
    missing, unexpected, norm_stats = load_state_dict_from_checkpoint(policy, root)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    policy.eval()
    return policy, norm_stats


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------
def build_act_from_config_json(path_or_dir: str | Path) -> ACTPolicy:
    """只按 config.json 构造（随机初始化），用于结构自检。"""
    with open(Path(path_or_dir) / "config.json", encoding="utf-8") as f:
        return ACTPolicy(ACTConfig.from_dict(json.load(f)))


def build_act_from_yaml(path: str | Path, *, device: str | None = None) -> ACTPolicy:
    """按 `config/ACT/<preset>.yaml` 构造（随机初始化），用于在新本体上从零训练。"""
    config = ACTConfig.from_yaml(path)
    policy = ACTPolicy(config)
    return policy.to(device=device or config.device)
