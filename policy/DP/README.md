# policy/DP —— Diffusion Policy 的独立实现

从 lerobot 0.6.2 `src/lerobot/policies/diffusion/` 抽取的 Diffusion Policy，
**不依赖 lerobot / diffusers / einops**，运行时只需要 `torch / numpy / safetensors`
（`torchvision` 仅在"要 ImageNet 预训练主干"时作为权重来源用一次）。

参考实现：
* 官方仓库 <https://github.com/real-stanford/diffusion_policy>
  （`policy/diffusion_unet_hybrid_image_policy.py`、`model/diffusion/conditional_unet1d.py`）；
* lerobot 的移植版（数值对拍的基准）。

论文：*Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*
（[2303.04137](https://huggingface.co/papers/2303.04137)）。

## 文件

| 文件 | 内容 |
| --- | --- |
| `modeling_diffusion.py` | `DiffusionConfig` + `DDPMScheduler`/`DDIMScheduler` + `SpatialSoftmax` + RGB 编码器 + `ConditionalUnet1d` + `DiffusionModel`/`DiffusionPolicy` + safetensors 加载 |
| `preprocessing.py` | MEAN_STD（图像）+ MIN_MAX（状态/动作）归一化，样本 → batch |
| `test_parity.py` | 调度器 vs diffusers、结构 vs lerobot、真实权重、预设配置、真实数据 |
| `run_dp.py` | 真实权重 + pusht 真实数据的采样 / loss / `select_action` demo |

按本体划分的配置在 `config/DP/*.yaml`（见 `config/DP/README.md`）。

## 快速开始

```python
from policy.DP import load_diffusion_policy, DPNormalizer

policy, norm_stats = load_diffusion_policy("weights/diffusion_pusht")
normalizer = DPNormalizer.from_checkpoint_buffers(policy.config, norm_stats)   # ckpt 自带统计量

batch = {
    "observation.state": state,        # (B, n_obs_steps, state_dim)
    "observation.image": image,        # (B, n_obs_steps, C, H, W)，像素 [0,1]
}
actions = policy.predict_action_chunk(normalizer.normalize(batch))   # (B, n_action_steps, action_dim)
actions = normalizer.unnormalize_actions(actions)
```

命令行 demo：

```bash
python -m policy.DP.run_dp --weights weights/diffusion_pusht \
    --dataset dataset/data/pusht --frame 100 --steps 3
```

输出（pusht 第 0 集第 100 帧）：

```
chunk   : (1, 8, 2) (n_action_steps=8, horizon=16)
L1(预测, GT) = 0.9793             # 动作量纲是像素位移（~95~235），约 1% 误差
第一步   : 预测 [95.68, 234.77]   GT [96.0, 235.0]
```

## 权重加载约定

`load_diffusion_policy(dir)` 读 `config.json` + `model.safetensors`：

* 参数名与官方 ckpt **逐字对齐**（`diffusion.unet.down_modules.0.0.conv1.block.0.weight` …），
  `strict` 加载 262,709,026 个参数，`missing / unexpected` 均为空；
* ckpt 里的 8 个归一化 buffer（`normalize_inputs.buffer_observation_image.{mean,std}`、
  `..._state.{min,max}`、`normalize_targets.buffer_action.*`、`unnormalize_outputs.buffer_action.*`）
  由 `split_normalization_buffers()` 剥离后返回，可直接构造 `DPNormalizer`，
  **不需要再读数据集的 stats.json**；
* 加载 ckpt 时会显式忽略 `pretrained_backbone_weights`（所有参数都会被覆盖）。

## 与官方 / lerobot 的差异

只改"外壳"，计算顺序、dtype、mask 语义都不变：

1. **噪声调度器手写**：`DDPMScheduler` / `DDIMScheduler` 的 beta 表（含 `squaredcos_cap_v2`）、
   `add_noise`、`step`（含 `clip_sample`、`fixed_small` 方差、DDIM 的 `eta=0` 路径）
   逐行对齐 diffusers，并用 100 步 / 10 步的完整采样链做过对拍。
   `DDIM` 的 `set_alpha_to_one` 默认值跟 diffusers 0.39 一致（`True`）。
2. **视觉主干复用 `policy.ACT.modeling_act._ResNet`**：同一套手写 ResNet（已与 torchvision 对拍），
   这里用可训练的 `nn.BatchNorm2d`，`use_group_norm=true` 时再替换成 `nn.GroupNorm`（键名不变，
   BN 的 `running_*` 也随之消失，与 ckpt 一致）。
3. **resize / crop 用 `F.interpolate` + 切片**替代 `torchvision.transforms`；训练态的随机裁剪
   与 `RandomCrop` 一样先抽 i 再抽 j，保证 RNG 消耗一致。
4. `einops.rearrange` 换成等价的 `permute/reshape`（`b s n ... -> (b s n) ...` 等）。
5. **归一化移出 policy**（lerobot 0.6.x 的做法），放到 `preprocessing.py`。
6. 官方实现训练时维护 EMA 权重、用 robomimic 的 obs encoder；lerobot 都去掉了，本实现跟随 lerobot
   （hub 上的 `lerobot/diffusion_pusht` 也只存了 `diffusion.*`，没有 EMA）。

## 对拍结果

`python -m policy.DP.test_parity --real-data`（CPU，fp32，`torch 2.11`，lerobot 0.6.2，
diffusers 0.39）。全部**逐位相同**，diff 为 `max|a-b|`：

| 用例 | 比较内容 | 结果 |
| --- | --- | --- |
| 调度器 DDPM / DDIM | beta 表、`alphas_cumprod`、`timesteps`(100/10 步)、`add_noise`、整条采样链 | 全部 `0.0` |
| 5 组 tiny 架构开关（separate encoder / 无 GroupNorm / DDIM / 无 padding mask） | rgb_encoder、global_cond、U-Net 单步、采样动作、loss | 全部 `0.0` |
| 真实权重（262.7M 参数） | global_cond、`predict_action_chunk`、`select_action`×3、loss | 全部 `0.0` |
| `config/DP/pusht.yaml` | 与 ckpt config 逐字段一致 + `strict=True` 加载 | 通过 |
| 真实数据（pusht 第 100 帧，含视频解码 + 归一化） | 动作块、loss | 全部 `0.0` |

复现：

```bash
python -m policy.DP.test_parity                 # 调度器 + 结构 + 真实权重 + 配置
python -m policy.DP.test_parity --real-data     # 再加真实数据集
```

对拍时踩到的三个坑（都已处理）：

* **采样必须固定种子**：DDPM 每一步都会注入噪声，只给同一个初始噪声不够；
* **`select_action` 是"在线"接口**：它吃**单帧**观测 `(B, C, H, W)` + `(B, D)` 状态，
  历史由内部队列补齐；把 `(B, n_obs, ...)` 的离线 batch 直接喂进去会多一维（`conv2d` 会报错）；
* **训练态随机裁剪**：两边都关掉（`crop_is_random=False`）才能比 loss，否则差异来自随机数。

## 覆盖 / 未覆盖

* 覆盖：训练 loss（epsilon / sample 两种 target、padding mask 开关）、`predict_action_chunk`、
  `select_action`（动作队列 / `populate_queues`）、DDPM & DDIM 采样、
  多相机（共享 / 每相机独立编码器）、`use_group_norm`、`use_film_scale_modulation`、
  resize+crop 预处理、`gradient_checkpointing`。
* 未覆盖：训练循环 / 优化器 / EMA / 数据增强、`observation.environment_state`（支持但没有真实用例）、
  AMP/bf16、`compile_model`（开关保留，未在本机验证）、官方仓库里的 `MaskGenerator`（lerobot 没有移植）。

## 资产

| 路径 | 内容 | 大小 |
| --- | --- | --- |
| `weights/diffusion_pusht/` | `lerobot/diffusion_pusht`：`model.safetensors` + `config.json` + `train_config.json` + `eval_info.json` | 1.05GB |
| `dataset/data/pusht/` | `lerobot/pusht` v3.0：206 集 / 25650 帧 / 96×96 AV1 视频 | 8MB |

两个都在 `.gitignore` 里。
