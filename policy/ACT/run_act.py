#!/usr/bin/env python
"""ACT 真实权重 + 真实数据 demo。

    python -m policy.ACT.run_act \
        --weights weights/act_aloha_sim_insertion_human \
        --dataset dataset/data/aloha_sim_insertion_human \
        --frame 0 --steps 5

流程：读一条数据集样本 → 用 ckpt 内嵌统计量归一化 → 前向得到 chunk →
反归一化 → 与数据集里的动作做 L1 对比；再用 `select_action` 演示动作队列。
"""

from __future__ import annotations

import argparse

import torch

from dataset.lerobot_v3 import LeRobotV3Dataset

from .modeling_act import load_act_policy
from .preprocessing import ACTNormalizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="weights/act_aloha_sim_insertion_human")
    parser.add_argument("--dataset", default="dataset/data/aloha_sim_insertion_human")
    parser.add_argument("--frame", type=int, default=0, help="数据集内的行号（全局帧索引）")
    parser.add_argument("--steps", type=int, default=3, help="select_action 演示的步数")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    policy, norm_stats = load_act_policy(args.weights, device=args.device)
    normalizer = ACTNormalizer.from_checkpoint_buffers(policy.config, norm_stats, device=args.device)

    ds = LeRobotV3Dataset(
        args.dataset,
        delta_timestamps={"action": [i / 50 for i in range(policy.config.chunk_size)]},
    )
    sample = ds[args.frame]
    print(f"dataset : {ds}")
    print(f"task    : {sample['task']!r}")
    print(f"episode : {int(sample['episode_index'].item())} frame_index={int(sample['frame_index'].item())}")
    print(f"image   : {tuple(sample['observation.images.top'].shape)} "
          f"range=[{sample['observation.images.top'].min():.3f}, {sample['observation.images.top'].max():.3f}]")
    print(f"state   : {tuple(sample['observation.state'].shape)}")
    print(f"action  : {tuple(sample['action'].shape)} padded={int(sample['action_is_pad'].sum())}")

    batch = {
        "observation.state": sample["observation.state"].unsqueeze(0).to(args.device),
        "observation.images.top": sample["observation.images.top"].unsqueeze(0).to(args.device),
        "action": sample["action"].unsqueeze(0).to(args.device),
        "action_is_pad": sample["action_is_pad"].unsqueeze(0).to(args.device),
    }
    batch = normalizer.normalize(batch)

    # 前向（eval：latent 全零，无随机性）
    policy.eval()
    with torch.no_grad():
        pred = policy.predict_action_chunk(batch)  # 归一化空间 (1, chunk, action_dim)
    pred_raw = normalizer.unnormalize_actions(pred)  # 回到关节空间

    gt = sample["action"].unsqueeze(0)
    valid = ~sample["action_is_pad"].unsqueeze(-1)
    l1 = ((pred_raw - gt).abs() * valid).sum() / (valid.sum() * gt.shape[-1]).clamp_min(1)
    print(f"\nchunk   : {tuple(pred_raw.shape)} 标准差={pred_raw.std():.4f}")
    print(f"L1(预测, GT) = {l1.item():.4f}")
    print(f"第一步   : 预测 {pred_raw[0, 0, :4].tolist()}")
    print(f"          GT   {gt[0, 0, :4].tolist()}")

    # 训练 loss（打开 VAE 采样）
    policy.train()
    loss, loss_dict = policy(batch)
    print(f"train loss = {loss.item():.4f}  {loss_dict}")

    # 逐 step 推理：队列长度 = n_action_steps
    policy.eval()
    policy.reset()
    print(f"\nselect_action x{args.steps}（已反归一化到关节空间）:")
    for i in range(args.steps):
        action = normalizer.unnormalize_actions(policy.select_action(batch))
        print(f"  step {i}: {tuple(action.shape)} 前 4 维 {action[0, :4].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
