# config/DP —— 按本体/任务划分的 Diffusion Policy 配置

| 文件 | 任务 / 本体 | state / action | 相机 | horizon / n_action_steps / n_obs_steps | 备注 |
| --- | --- | --- | --- | --- | --- |
| `pusht.yaml` | PushT 2D 推块 | 2 / 2 | `observation.image` 96×96 → crop 84×84 | 16 / 8 / 2 @10fps | 与 `weights/diffusion_pusht` 的训练配置逐字段一致，可 strict 加载该 ckpt |

## 用法

```python
from policy.DP import DiffusionConfig, build_diffusion_from_yaml

cfg = DiffusionConfig.from_yaml("config/DP/pusht.yaml")   # 只读配置
policy = build_diffusion_from_yaml("config/DP/pusht.yaml")  # 随机初始化，从头训练

# 已有 ckpt 仍走权重目录（config.json 优先）：
from policy.DP import load_diffusion_policy
policy, _ = load_diffusion_policy("weights/diffusion_pusht")
```

校验：

```bash
python -m policy.DP.test_parity --skip-schedulers --skip-variants --skip-real-weights
```

## 字段说明

```yaml
name: pusht      # 元信息 -> config.preset
fps: 10          # 元信息 -> config.fps
policy:          # 这一段是 DiffusionConfig 的字段，未知键直接报错
  ...
```

和本体强相关的分组：

| 分组 | 关键字段 | 说明 |
| --- | --- | --- |
| 观测窗口 | `n_obs_steps` | 用最近几帧做条件；数据集侧对应 `delta = [1-n_obs, ..., 0]` |
| 动作窗口 | `horizon` / `n_action_steps` / `drop_n_last_frames` | U-Net 一次预测 `horizon` 步，实际执行前 `n_action_steps` 步 |
| 特征 | `input_features` / `output_features` | 键名必须等于数据集特征名；相机形状 `(C,H,W)` |
| 相机预处理 | `resize_shape` / `crop_shape` / `crop_ratio` | 裁剪后尺寸决定 token 数；`use_group_norm=true` 时 `pretrained_backbone_weights` 必须为 null |
| U-Net | `down_dims` | 下采样倍率 = `2**len(down_dims)`，`horizon` 必须是它的整数倍 |
| 扩散 | `num_train_timesteps` / `num_inference_steps` / `beta_schedule` | pusht 用 100 步 cosine；`num_inference_steps: null` 表示与训练步数相同 |

## 换本体 / 换任务的检查清单

1. `state` / `action` 维度 = 数据集 `meta/info.json` 里对应 feature 的 `shape`；
2. 相机键名与 `features` 一致；`crop_shape` 不能大于原图（`validate_features` 会检查）；
3. `horizon % 2**len(down_dims) == 0`，否则 U-Net 的 skip-connection 拼不上；
4. `n_action_steps <= horizon - n_obs_steps + 1`（动作队列从 `n_obs_steps-1` 开始取）；
5. 归一化模式：图像用 `MEAN_STD`（可选 ImageNet 统计量），状态/动作 DP 官方用 `MIN_MAX`；
6. 相机数量或 `state` 维度会改变 `global_cond_dim`（= `state_dim + 2*num_keypoints*相机数`），
   等于换模型，必须重新训练；只改分辨率不改变参数量（SpatialSoftmax 输出维度固定），
   但图像分布变了，仍建议重训。
