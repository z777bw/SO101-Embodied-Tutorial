#!/usr/bin/env python
"""Diffusion Policy 真实权重 + 真实数据 demo。

    python -m policy.DP.run_dp \
        --weights weights/diffusion_pusht \
        --dataset dataset/data/pusht \
        --frame 100 --steps 3

流程：读一条 pusht 样本（2 帧历史 + 16 帧动作窗口）→ 用数据集统计量归一化 →
扩散采样出动作块 → 反归一化 → 与 GT 比 L1；再演示 `select_action` 的动作队列。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dataset.lerobot_v3 import LeRobotV3Dataset

from .modeling_diffusion import load_diffusion_policy
from .preprocessing import DPNormalizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="weights/diffusion_pusht")
    parser.add_argument("--dataset", default="dataset/data/pusht")
    parser.add_argument("--frame", type=int, default=100, help="数据集内的行号（全局帧索引）")
    parser.add_argument("--steps", type=int, default=3, help="select_action 演示的步数")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    policy, norm_stats = load_diffusion_policy(args.weights, device=args.device)
    cfg = policy.config

    if norm_stats:  # ckpt 里带了训练时的统计量，优先用它
        normalizer = DPNormalizer.from_checkpoint_buffers(cfg, norm_stats, device=args.device)
        print(f"stats   : from checkpoint {sorted(norm_stats)}")
    else:
        stats = json.loads((Path(args.dataset) / "meta" / "stats.json").read_text(encoding="utf-8"))
        normalizer = DPNormalizer.from_dataset_stats(cfg, stats, device=args.device)
        print("stats   : from dataset meta/stats.json")

    fps = json.loads((Path(args.dataset) / "meta" / "info.json").read_text(encoding="utf-8"))["fps"]
    ds = LeRobotV3Dataset(
        args.dataset,
        delta_timestamps={
            "observation.image": [i / fps for i in cfg.observation_delta_indices],
            "observation.state": [i / fps for i in cfg.observation_delta_indices],
            "action": [i / fps for i in cfg.action_delta_indices],
        },
    )
    sample = ds[args.frame]
    print(f"dataset : {ds}")
    print(f"task    : {sample['task']!r}")
    print(f"episode : {int(sample['episode_index'].item())} frame_index={int(sample['frame_index'].item())}")
    print(
        f"image   : {tuple(sample['observation.image'].shape)} "
        f"range=[{sample['observation.image'].min():.3f}, {sample['observation.image'].max():.3f}]"
    )
    print(f"state   : {tuple(sample['observation.state'].shape)}")
    print(f"action  : {tuple(sample['action'].shape)} padded={int(sample['action_is_pad'].sum())}")

    batch = {
        "observation.state": sample["observation.state"].unsqueeze(0).to(args.device),
        "observation.image": sample["observation.image"].unsqueeze(0).to(args.device),
        "action": sample["action"].unsqueeze(0).to(args.device),
        "action_is_pad": sample["action_is_pad"].unsqueeze(0).to(args.device),
    }
    batch = normalizer.normalize(batch)

    policy.eval()
    with torch.no_grad():
        pred = policy.predict_action_chunk(batch)  # (1, n_action_steps, action_dim)，归一化空间

    pred_raw = normalizer.unnormalize_actions(pred)
    gt = normalizer.unnormalize_actions(batch["action"])[:, cfg.n_obs_steps - 1 : cfg.n_obs_steps - 1 + cfg.n_action_steps]
    pad = sample["action_is_pad"][cfg.n_obs_steps - 1 : cfg.n_obs_steps - 1 + cfg.n_action_steps].unsqueeze(0).unsqueeze(-1)
    l1 = ((pred_raw - gt).abs() * ~pad).sum() / ((~pad).sum() * gt.shape[-1]).clamp_min(1)

    print(f"\nchunk   : {tuple(pred_raw.shape)} (n_action_steps={cfg.n_action_steps}, horizon={cfg.horizon})")
    print(f"L1(预测, GT) = {l1.item():.4f}")
    print(f"第一步   : 预测 {[round(v, 4) for v in pred_raw[0, 0].tolist()]}")
    print(f"          GT   {[round(v, 4) for v in gt[0, 0].tolist()]}")

    # 训练 loss（加噪 + 去噪一步）
    policy.train()
    torch.manual_seed(0)
    loss, _ = policy(batch)
    print(f"train loss = {loss.item():.2e}")  # 随抽到的扩散步变化很大，看数量级即可

    # 动作队列：n_action_steps 次 select_action 只用一次前向。
    # 注意 select_action 走"在线"路径，吃单帧观测（历史由内部队列补齐）
    online = {
        "observation.state": batch["observation.state"][:, -1],
        **{k: batch[k][:, -1] for k in cfg.image_features},
    }
    policy.eval()
    policy.reset()
    print(f"\nselect_action x{args.steps}（已反归一化）:")
    for i in range(args.steps):
        action = normalizer.unnormalize_actions(policy.select_action(dict(online)))
        print(f"  step {i}: {tuple(action.shape)} {[round(v, 4) for v in action[0].tolist()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
