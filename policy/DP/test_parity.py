#!/usr/bin/env python
"""Diffusion Policy 与参考实现的对拍脚本。

1. `--schedulers` 噪声调度器 vs `diffusers`（DDPM / DDIM，含逐步采样链）；
2. `--variants`   tiny 随机权重 vs lerobot 的 diffusion policy（含 separate-encoder 开关、
                  GroupNorm、FiLM、padding mask）；
3. `--real-weights` 真实 ckpt（1.05GB）同一份权重灌进两边，比较 loss / 采样 / select_action；
4. `--configs`   `config/DP/pusht.yaml` 与 ckpt config 一致 + strict 加载；
5. `--real-data` pusht 数据集样本（视频解码 + 归一化）→ loss / 动作块。

用法：
    python -m policy.DP.test_parity
    python -m policy.DP.test_parity --real-data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .modeling_diffusion import (
    DDIMScheduler,
    DDPMScheduler,
    DiffusionConfig,
    DiffusionPolicy,
    _read_safetensors,
    build_diffusion_from_yaml,
    load_diffusion_policy,
    split_normalization_buffers,
)
from .preprocessing import DPNormalizer


# 参考实现在测试里才 import
def _ref_policy(ref_config):
    from lerobot.policies.diffusion.modeling_diffusion import (
        DiffusionPolicy as RefDiffusionPolicy,
    )

    return RefDiffusionPolicy(ref_config)


def _ref_config(features, **kwargs):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import (
        DiffusionConfig as RefDiffusionConfig,
    )

    type_map = {"STATE": FeatureType.STATE, "VISUAL": FeatureType.VISUAL, "ACTION": FeatureType.ACTION}
    return RefDiffusionConfig(
        input_features={k: PolicyFeature(type=type_map[t], shape=tuple(s)) for k, (t, s) in features["inputs"].items()},
        output_features={k: PolicyFeature(type=type_map[t], shape=tuple(s)) for k, (t, s) in features["outputs"].items()},
        device="cpu",
        pretrained_backbone_weights=None,
        **kwargs,
    )


def _my_config(features, **kwargs):
    from .modeling_diffusion import PolicyFeature

    return DiffusionConfig(
        input_features={k: PolicyFeature(type=t, shape=tuple(s)) for k, (t, s) in features["inputs"].items()},
        output_features={k: PolicyFeature(type=t, shape=tuple(s)) for k, (t, s) in features["outputs"].items()},
        pretrained_backbone_weights=None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _compare(name: str, mine, ref, tol: float = 0.0) -> float:
    if mine is None and ref is None:
        print(f"  [SKIP] {name:<28} (both None)")
        return 0.0
    if mine.shape != ref.shape:
        raise AssertionError(f"{name}: shape mismatch {tuple(mine.shape)} vs {tuple(ref.shape)}")
    diff = (mine.double() - ref.double()).abs().max().item()
    status = "OK " if diff <= tol else "BAD"
    print(f"  [{status}] {name:<28} max|diff|={diff:.3e}  shape={tuple(mine.shape)}")
    if diff > tol:
        raise AssertionError(f"{name}: diff {diff:.3e} > tol {tol:.3e}")
    return diff


def _scheduler_kwargs(config: DiffusionConfig) -> dict:
    return {
        "num_train_timesteps": config.num_train_timesteps,
        "beta_start": config.beta_start,
        "beta_end": config.beta_end,
        "beta_schedule": config.beta_schedule,
        "clip_sample": config.clip_sample,
        "clip_sample_range": config.clip_sample_range,
        "prediction_type": config.prediction_type,
    }


def _make_batch(config: DiffusionConfig, batch_size: int, *, seed: int = 0, image_key: str | None = None):
    g = torch.Generator().manual_seed(seed)
    obs_key = image_key or next(iter(config.image_features))
    h, w = config.image_features[obs_key].shape[-2:]
    batch = {
        "observation.state": torch.randn(batch_size, config.n_obs_steps, config.robot_state_feature.shape[0], generator=g),
        obs_key: torch.rand(batch_size, config.n_obs_steps, 3, h, w, generator=g),
        "action": torch.randn(batch_size, config.horizon, config.action_feature.shape[0], generator=g),
    }
    batch["action_is_pad"] = torch.zeros(batch_size, config.horizon, dtype=torch.bool)
    batch["action_is_pad"][:, -max(1, config.horizon // 4) :] = True
    return batch


def _stack_images(batch: dict[str, torch.Tensor], config: DiffusionConfig) -> dict[str, torch.Tensor]:
    batch = dict(batch)
    keys = list(config.image_features)
    if len(keys) == 1 and batch[keys[0]].ndim == 5:
        batch["observation.images"] = batch[keys[0]].unsqueeze(2)  # (B,S,1,C,H,W)
    else:
        batch["observation.images"] = torch.stack([batch[k] for k in keys], dim=-4)
    return batch


# ---------------------------------------------------------------------------
# 1. 调度器
# ---------------------------------------------------------------------------
def run_scheduler_parity() -> None:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler as RefDDIM
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler as RefDDPM

    print("\n" + "=" * 78)
    print("1) 噪声调度器对拍：手写 DDPM/DDIM vs diffusers")
    print("=" * 78)

    kwargs = {
        "num_train_timesteps": 100,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "clip_sample": True,
        "clip_sample_range": 1.0,
        "prediction_type": "epsilon",
    }

    for name, mine_cls, ref_cls in (("DDPM", DDPMScheduler, RefDDPM), ("DDIM", DDIMScheduler, RefDDIM)):
        mine, ref = mine_cls(**kwargs), ref_cls(**kwargs)
        print(f"\n[{name}]")
        _compare("betas", mine.betas, ref.betas)
        _compare("alphas_cumprod", mine.alphas_cumprod, ref.alphas_cumprod)

        for steps in (100, 10):
            mine.set_timesteps(steps)
            ref.set_timesteps(steps)
            _compare(f"timesteps@{steps}", mine.timesteps.float(), ref.timesteps.float())

            x = torch.randn(2, 16, 2)
            noise = torch.randn_like(x)
            ts = torch.tensor([3, 88])
            _compare(f"add_noise@{steps}", mine.add_noise(x, noise, ts), ref.add_noise(x, noise, ts))

            # 逐步采样链：同一个假"模型输出"，同一个 RNG 种子
            sample_mine = x.clone()
            sample_ref = x.clone()
            for t in mine.timesteps:
                out = torch.full_like(x, float(t) / 100.0)
                torch.manual_seed(1234)
                a = mine.step(out, t, sample_mine).prev_sample
                torch.manual_seed(1234)
                b = ref.step(out, t, sample_ref).prev_sample
                sample_mine, sample_ref = a, b
            _compare(f"sample_chain@{steps}", sample_mine, sample_ref)


# ---------------------------------------------------------------------------
# 2. tiny 结构对拍
# ---------------------------------------------------------------------------
VARIANTS = {
    "default": {},
    "separate_encoders": {"use_separate_rgb_encoder_per_camera": True},
    "no_group_norm": {"use_group_norm": False},
    "ddim": {"noise_scheduler_type": "DDIM"},
    "no_mask": {"do_mask_loss_for_padding": False},
}


def run_variants(tol: float = 0.0) -> None:
    print("\n" + "=" * 78)
    print("2) 结构对拍：tiny 配置，比较 rgb encoder / global_cond / unet / loss / 采样")
    print("=" * 78)

    features = {
        "inputs": {"observation.images.cam": ("VISUAL", (3, 64, 64)), "observation.state": ("STATE", (3,))},
        "outputs": {"action": ("ACTION", (2,))},
    }
    tiny = {
        "n_obs_steps": 2,
        "horizon": 8,
        "n_action_steps": 4,
        "down_dims": (32, 64),
        "kernel_size": 3,
        "n_groups": 4,
        "diffusion_step_embed_dim": 16,
        "spatial_softmax_num_keypoints": 8,
        "crop_shape": (60, 60),  # 60/32 -> 2x2 feature map，BatchNorm 在 batch=1 时也能跑
        "crop_is_random": False,  # 训练态也要确定性，随机裁剪交给数据集增强
        "use_group_norm": True,
        "use_film_scale_modulation": True,
        "num_train_timesteps": 10,
        "do_mask_loss_for_padding": True,
    }

    for name, overrides in VARIANTS.items():
        kwargs = {**tiny, **overrides}
        print(f"\n[{name}] overrides={overrides}")
        torch.manual_seed(0)
        mine = DiffusionPolicy(_my_config(features, **kwargs))
        ref = _ref_policy(_ref_config(features, **kwargs))
        report = ref.load_state_dict(mine.state_dict(), strict=False)
        if report.missing_keys or report.unexpected_keys:
            raise AssertionError(f"reference rejected weights: {report}")

        mine.eval()
        ref.eval()
        batch = _make_batch(mine.config, 2, seed=3)
        stacked = _stack_images(batch, mine.config)

        with torch.no_grad():
            # 图像编码
            if mine.config.use_separate_rgb_encoder_per_camera:
                img_mine = mine.diffusion.rgb_encoder[0](stacked["observation.images"][:, 0, 0])
                img_ref = ref.diffusion.rgb_encoder[0](stacked["observation.images"][:, 0, 0])
            else:
                img_mine = mine.diffusion.rgb_encoder(stacked["observation.images"][:, 0, 0])
                img_ref = ref.diffusion.rgb_encoder(stacked["observation.images"][:, 0, 0])
            _compare("rgb_encoder", img_mine, img_ref, tol)

            gc_mine = mine.diffusion._prepare_global_conditioning(stacked)
            gc_ref = ref.diffusion._prepare_global_conditioning(stacked)
            _compare("global_cond", gc_mine, gc_ref, tol)

            # U-Net 单步
            x = torch.randn(2, mine.config.horizon, 2)
            t = torch.tensor([1, 7])
            _compare(
                "unet",
                mine.diffusion.unet(x, t, global_cond=gc_mine),
                ref.diffusion.unet(x, t, global_cond=gc_ref),
                tol,
            )

            # 采样（给定同一个初始噪声，DDPM 每步的随机数用同一 seed）
            noise = torch.randn(2, mine.config.horizon, 2, generator=torch.Generator().manual_seed(11))
            torch.manual_seed(7)
            act_mine = mine.predict_action_chunk(batch, noise=noise.clone())
            torch.manual_seed(7)
            act_ref = ref.predict_action_chunk(batch, noise=noise.clone())
            _compare("actions", act_mine, act_ref, tol)

        # 训练 loss
        mine.train()
        ref.train()
        torch.manual_seed(99)
        loss_mine, _ = mine(batch)
        torch.manual_seed(99)
        loss_ref, _ = ref(batch)
        _compare("loss", loss_mine.detach(), loss_ref.detach(), tol)
        print(f"  [INFO] loss={loss_mine.item():.6f}")

        del mine, ref, batch, stacked, act_mine, act_ref


# ---------------------------------------------------------------------------
# 3. 真实权重
# ---------------------------------------------------------------------------
def run_real_weights(weights_dir: str, tol: float = 0.0) -> None:
    print("\n" + "=" * 78)
    print(f"3) 真实权重对拍：{weights_dir}")
    print("=" * 78)

    mine, _ = load_diffusion_policy(weights_dir)
    config_dict = json.loads((Path(weights_dir) / "config.json").read_text(encoding="utf-8"))
    features = {
        "inputs": {k: (v["type"], tuple(v["shape"])) for k, v in config_dict["input_features"].items()},
        "outputs": {k: (v["type"], tuple(v["shape"])) for k, v in config_dict["output_features"].items()},
    }
    overrides = _config_overrides(config_dict)
    ref = _ref_policy(_ref_config(features, **overrides))

    state_dict, _ = split_normalization_buffers(_read_safetensors(weights_dir))
    report = ref.load_state_dict(state_dict, strict=False)
    print(f"  reference load: missing={len(report.missing_keys)} unexpected={len(report.unexpected_keys)}")
    if report.missing_keys or report.unexpected_keys:
        raise AssertionError(f"reference mismatch: {report.missing_keys[:5]} / {report.unexpected_keys[:5]}")

    mine.eval()
    ref.eval()
    batch = _make_batch(mine.config, 1, seed=5)
    stacked = _stack_images(batch, mine.config)

    with torch.no_grad():
        _compare(
            "global_cond",
            mine.diffusion._prepare_global_conditioning(stacked),
            ref.diffusion._prepare_global_conditioning(stacked),
            tol,
        )

        # 采样（pusht 是 100 步 DDPM，给定同一初始噪声 + 同种子）
        noise = torch.randn(
            1, mine.config.horizon, mine.config.action_feature.shape[0], generator=torch.Generator().manual_seed(21)
        )
        torch.manual_seed(31)
        act_mine = mine.predict_action_chunk(batch, noise=noise.clone())
        torch.manual_seed(31)
        act_ref = ref.predict_action_chunk(batch, noise=noise.clone())
        _compare("actions(pred chunk)", act_mine, act_ref, tol)

        # 逐步推理：队列机制（在线路径吃单帧观测，历史由队列补）
        online = {
            "observation.state": batch["observation.state"][:, -1],
            **{k: batch[k][:, -1] for k in mine.config.image_features},
        }
        mine.reset()
        torch.manual_seed(41)
        step_mine = [mine.select_action(dict(online)) for _ in range(3)]
        ref.reset()
        torch.manual_seed(41)
        step_ref = [ref.select_action(dict(online)) for _ in range(3)]
        for i, (a, b) in enumerate(zip(step_mine, step_ref, strict=True)):
            _compare(f"select_action#{i}", a, b, tol)

    # 训练 loss
    mine.train()
    ref.train()
    torch.manual_seed(77)
    loss_mine, _ = mine(batch)
    torch.manual_seed(77)
    loss_ref, _ = ref(batch)
    _compare("loss", loss_mine.detach(), loss_ref.detach(), tol)
    print(f"  [INFO] loss={loss_mine.item():.6f}")


def _config_overrides(config_dict: dict) -> dict:
    keys = (
        "n_obs_steps",
        "horizon",
        "n_action_steps",
        "drop_n_last_frames",
        "vision_backbone",
        "resize_shape",
        "crop_ratio",
        "crop_shape",
        "crop_is_random",
        "use_group_norm",
        "spatial_softmax_num_keypoints",
        "use_separate_rgb_encoder_per_camera",
        "down_dims",
        "kernel_size",
        "n_groups",
        "diffusion_step_embed_dim",
        "use_film_scale_modulation",
        "gradient_checkpointing",
        "noise_scheduler_type",
        "num_train_timesteps",
        "beta_schedule",
        "beta_start",
        "beta_end",
        "prediction_type",
        "clip_sample",
        "clip_sample_range",
        "num_inference_steps",
        "do_mask_loss_for_padding",
    )
    return {k: config_dict[k] for k in keys if k in config_dict}


# ---------------------------------------------------------------------------
# 4. 预设配置
# ---------------------------------------------------------------------------
def run_config_presets(weights_dir: str) -> None:
    import json

    print("\n" + "=" * 78)
    print("4) 预设配置校验：config/DP/pusht.yaml")
    print("=" * 78)

    ckpt_config = json.loads((Path(weights_dir) / "config.json").read_text(encoding="utf-8"))
    cfg = DiffusionConfig.from_yaml("config/DP/pusht.yaml")
    ignore = {"preset", "fps", "pretrained_backbone_weights", "device", "compile_model", "compile_mode"}
    diffs = {}
    for name in DiffusionConfig.__dataclass_fields__:
        if name in ignore or name not in ckpt_config:
            continue
        mine, theirs = getattr(cfg, name), ckpt_config[name]
        if name.endswith("_features"):
            mine = {k: (ft.type, tuple(ft.shape)) for k, ft in mine.items()}
            theirs = {k: (v["type"], tuple(v["shape"])) for k, v in theirs.items()}
        elif isinstance(theirs, list):
            mine = list(mine) if isinstance(mine, (tuple, list)) else mine
        if mine != theirs:
            diffs[name] = (mine, theirs)
    if diffs:
        raise AssertionError(f"pusht.yaml 与 ckpt config 不一致: {diffs}")
    print("  [OK ] pusht.yaml 与 ckpt config 逐字段一致（preset/fps/初始化/编译开关除外）")

    policy = build_diffusion_from_yaml("config/DP/pusht.yaml", device="cpu")
    state_dict, _ = split_normalization_buffers(_read_safetensors(weights_dir))
    report = policy.load_state_dict(state_dict, strict=True)
    assert not report.missing_keys and not report.unexpected_keys
    print(f"  [OK ] pusht.yaml 严格加载官方权重，参数 {sum(p.numel() for p in policy.parameters()):,}")


# ---------------------------------------------------------------------------
# 5. 真实数据
# ---------------------------------------------------------------------------
def run_real_data(dataset_root: str, weights_dir: str) -> None:
    from dataset.lerobot_v3 import LeRobotV3Dataset

    print("\n" + "=" * 78)
    print(f"5) 真实数据：{dataset_root}")
    print("=" * 78)

    mine, norm_stats = load_diffusion_policy(weights_dir)
    cfg = mine.config
    if norm_stats:
        normalizer = DPNormalizer.from_checkpoint_buffers(cfg, norm_stats)
        print(f"  stats from checkpoint: {sorted(norm_stats)}")
    else:
        stats = json.loads((Path(dataset_root) / "meta" / "stats.json").read_text(encoding="utf-8"))
        normalizer = DPNormalizer.from_dataset_stats(cfg, stats)

    fps = json.loads((Path(dataset_root) / "meta" / "info.json").read_text(encoding="utf-8"))["fps"]
    delta_timestamps = {
        "observation.image": [i / fps for i in cfg.observation_delta_indices],
        "observation.state": [i / fps for i in cfg.observation_delta_indices],
        "action": [i / fps for i in cfg.action_delta_indices],
    }
    ds = LeRobotV3Dataset(dataset_root, delta_timestamps=delta_timestamps)
    print(f"  {ds}")
    sample = ds[100]
    print(
        f"  sample: task={sample['task']!r} image={tuple(sample['observation.image'].shape)} "
        f"state={tuple(sample['observation.state'].shape)} action={tuple(sample['action'].shape)} "
        f"pad={int(sample['action_is_pad'].sum())}"
    )

    batch = {
        "observation.state": sample["observation.state"].unsqueeze(0),
        "observation.image": sample["observation.image"].unsqueeze(0),
        "action": sample["action"].unsqueeze(0),
        "action_is_pad": sample["action_is_pad"].unsqueeze(0),
    }
    batch = normalizer.normalize(batch)

    with torch.no_grad():
        actions = mine.predict_action_chunk(batch)
    print(f"  predict_action_chunk: {tuple(actions.shape)} mean={actions.mean():.4f} std={actions.std():.4f}")

    # 与 lerobot 参考实现比
    config_dict = json.loads((Path(weights_dir) / "config.json").read_text(encoding="utf-8"))
    features = {
        "inputs": {k: (v["type"], tuple(v["shape"])) for k, v in config_dict["input_features"].items()},
        "outputs": {k: (v["type"], tuple(v["shape"])) for k, v in config_dict["output_features"].items()},
    }
    ref = _ref_policy(_ref_config(features, **_config_overrides(config_dict)))
    state_dict, _ = split_normalization_buffers(_read_safetensors(weights_dir))
    ref.load_state_dict(state_dict, strict=False)
    ref.eval()

    noise = torch.randn(1, cfg.horizon, cfg.action_feature.shape[0], generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        mine.eval()
        # DDPM 采样每步都会注入噪声，所以两边各自固定种子
        torch.manual_seed(5)
        a_mine = mine.predict_action_chunk(batch, noise=noise.clone())
        torch.manual_seed(5)
        a_ref = ref.predict_action_chunk(batch, noise=noise.clone())
    _compare("real_data.actions", a_mine, a_ref, 0.0)

    torch.manual_seed(3)
    loss, _ = mine(batch)
    torch.manual_seed(3)
    loss_ref, _ = ref(batch)
    _compare("real_data.loss", loss.detach(), loss_ref.detach(), 1e-9)
    print(f"  loss={loss.item():.6f}")

    # 反归一化后的动作范围（pusht 动作是 2 维末端位移，应在 [-1, 1] 附近）
    act_raw = normalizer.unnormalize_actions(actions)
    print(
        f"  反归一化动作: min={act_raw.min():.3f} max={act_raw.max():.3f} "
        f"first={act_raw[0, 0].tolist()}"
    )
    start = cfg.n_obs_steps - 1
    gt = normalizer.unnormalize_actions(batch["action"])[:, start : start + cfg.n_action_steps]
    pad = sample["action_is_pad"][start : start + cfg.n_action_steps].unsqueeze(0).unsqueeze(-1)
    l1 = ((act_raw - gt).abs() * ~pad).sum() / ((~pad).sum() * gt.shape[-1]).clamp_min(1)
    print(f"  L1(预测, GT) = {l1.item():.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="weights/diffusion_pusht")
    parser.add_argument("--dataset", default="dataset/data/pusht")
    parser.add_argument("--skip-schedulers", action="store_true")
    parser.add_argument("--skip-variants", action="store_true")
    parser.add_argument("--skip-real-weights", action="store_true")
    parser.add_argument("--skip-configs", action="store_true")
    parser.add_argument("--real-data", action="store_true")
    args = parser.parse_args()

    if not args.skip_schedulers:
        run_scheduler_parity()
    if not args.skip_variants:
        run_variants()
    if not args.skip_real_weights:
        run_real_weights(args.weights)
    if not args.skip_configs:
        run_config_presets(args.weights)
    if args.real_data:
        run_real_data(args.dataset, args.weights)

    print("\n全部对拍通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
