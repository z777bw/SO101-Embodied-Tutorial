# policy/ACT —— ACT 的独立实现

从 `lerobot/src/lerobot/policies/act/`（lerobot 0.6.2）抽取的 Action Chunking
Transformer，**不依赖 lerobot / transformers / einops**，运行时只需要
`torch / numpy / safetensors`（`torchvision` 仅在"要 ImageNet 预训练主干"时作为权重来源用一次）。

对应论文：*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*
（[2304.13705](https://huggingface.co/papers/2304.13705)）。

## 文件

| 文件 | 内容 |
| --- | --- |
| `modeling_act.py` | `ACTConfig` + `ACT` + encoder/decoder + `ACTPolicy`（loss / 动作队列 / 时间集成）+ safetensors 加载 |
| `preprocessing.py` | MEAN_STD 归一化（`ACTNormalizer`）、数据集样本 → batch（`batch_from_sample`） |
| `test_parity.py` | 与 lerobot 参考实现的三种对拍（结构 / 真实权重 / 真实数据）+ 预设配置校验 |
| `run_act.py` | 真实权重 + 真实数据的前向、loss、`select_action` demo |

按本体划分的配置在 `config/ACT/*.yaml`（见 `config/ACT/README.md`），
`ACTConfig.from_yaml()` 读取、`build_act_from_yaml()` 直接建模型。

模型代码坚持放在**单文件**里：ACT 的网络结构和 lerobot 一一对应，拆成多文件之后
"和参考实现对拍"会变成在多文件之间跳转，成本远大于单文件的阅读成本。

## 快速开始

```python
from policy.ACT import load_act_policy, ACTNormalizer

policy, norm_stats = load_act_policy("weights/act_aloha_sim_insertion_human")
normalizer = ACTNormalizer.from_checkpoint_buffers(policy.config, norm_stats)

# batch 约定（与 lerobot 一致，键名取 config.input_features 里的名字）
batch = {
    "observation.images.top": img,      # (B, 3, 480, 640) float32, [0,1]
    "observation.state": state,         # (B, 14)
}
actions = policy.predict_action_chunk(normalizer.normalize(batch))   # (B, chunk_size, 14)，归一化空间
actions = normalizer.unnormalize_actions(actions)                     # 回到关节空间
```

换本体（例如 SO-101）时用 YAML 预设，从零训练：

```python
from policy.ACT import build_act_from_yaml, ACTNormalizer, load_dataset_stats

policy = build_act_from_yaml("config/ACT/so101.yaml")          # 6 维 state/action、2 路相机
normalizer = ACTNormalizer.from_dataset_stats(policy.config, load_dataset_stats("dataset/data/<your_dataset>"))
```

命令行 demo：

```bash
python -m policy.ACT.run_act \
    --weights weights/act_aloha_sim_insertion_human \
    --dataset dataset/data/aloha_sim_insertion_human --frame 0
```

## 权重加载约定

`load_act_policy(dir)` 读 `config.json` + `model.safetensors`：

* 参数名与官方 ckpt **逐字对齐**（`model.encoder.layers.0.self_attn.in_proj_weight` …），
  因此 `load_state_dict` 直接成功，`missing / unexpected` 均为空；
* 旧 ckpt 里遗留的 8 个归一化 buffer（`normalize_inputs.*` / `normalize_targets.*` /
  `unnormalize_outputs.*`，lerobot 0.6.x 已把它们移到 processor）由
  `split_normalization_buffers()` 剥离后返回，正好可以直接构造 `ACTNormalizer`；
* 加载失败会抛异常而不是静默吞掉，方便发现 ckpt 与 config 不匹配。

## 与 lerobot 参考实现的差异

只改了"外壳"，没有改计算：

1. `ACTConfig` 不再继承 `PreTrainedConfig`；`PolicyFeature` 换成本文件里的轻量 dataclass
   （只保留 `type` / `shape`）。
2. `einops` 的 `repeat / rearrange` 换成等价的 `Tensor.repeat` 与 `permute + flatten`
   （`b c h w -> (h w) b c` 对应 `permute(2, 3, 0, 1).flatten(0, 1)`）。
3. **视觉主干整体手写**：`_ResNet` + `_BasicBlock`（resnet18/34）+ `_Bottleneck`（resnet50/101/152）
   + `_FrozenBatchNorm2d` + `_IntermediateLayerGetter`。参数名、`downsample.0/1` 的层级、
   `replace_stride_with_dilation` 的语义（含 torchvision 里 `previous_dilation` 的历史行为）
   都逐行对齐，因此官方 ckpt 的 100 个 `backbone.*` 张量可以 `strict=True` 灌进来。
   `torchvision` 只剩一个用途：配置里显式要求 `pretrained_backbone_weights` 时，把它的
   主干 state_dict 拷过来当初值。
4. 归一化移到 `preprocessing.py`（与 lerobot 0.6.x 把 normalize 放进 processor 的做法一致）。

## 对拍结果

`python -m policy.ACT.test_parity --real-data`（CPU，fp32，`torch 2.11`，lerobot 0.6.2）。
所有条目都是**逐位相同**，"diff" 是 `max|a-b|`：

| 用例 | 比较内容 | 结果 |
| --- | --- | --- |
| 主干 vs torchvision（resnet18 / 34 / 50 / 50+空洞） | layer4 feature map（同权重同输入） | 全部 `0.0` |
| 6 组 tiny 架构开关（`pre_norm` / `use_vae=False` / `n_decoder_layers=2` / `gelu` / VAE 层数） | encoder 输入&输出、decoder 输入&输出、action_head、actions、train loss、l1、kld、两处梯度 | 全部 `0.0` |
| 真实权重（51.6M 参数，480×640）eval | actions、encoder/decoder 输出 | 全部 `0.0` |
| 真实权重 train（VAE + dropout 置 0 + 固定种子） | actions、mu、log_sigma_x2、loss | 全部 `0.0` |
| 真实数据（数据集第 0 帧，含视频解码 + 归一化） | predict_action_chunk、train loss、select_action 前两步 | 全部 `0.0` |
| `config/ACT/aloha.yaml` | 与 ckpt config 逐字段一致、`strict=True` 严格加载 51.6M 权重 | 通过 |
| `config/ACT/so101.yaml` | 6 维 state/action、2 路相机、602 个 encoder token 的前向 | 通过 |

复现：

```bash
python -m policy.ACT.test_parity                       # 主干 + 结构 + 真实权重 + 配置预设
python -m policy.ACT.test_parity --real-data           # 再加真实数据集
python -m policy.ACT.test_parity --skip-backbone --skip-variants --skip-real-weights   # 只校验配置
```

对拍里的两个坑（都已处理，改代码时注意）：

* **train 路径要关 dropout**：`nn.MultiheadAttention` 的 dropout 是普通属性而不是
  `nn.Dropout` 子模块，只清 `nn.Dropout` 会留下 residual 差异；
* **VAE 采样要固定种子**：`latent_sample = mu + log_sigma/2 .exp() * randn_like(mu)`，
  两边必须在同一 RNG 状态下前向。

## 覆盖 / 未覆盖

* 覆盖：训练前向 + L1/KL loss、`predict_action_chunk`、`select_action`（动作队列）、
  时间集成（`ACTTemporalEnsembler`）、多相机图像输入、`use_vae=False`、`pre_norm`、
  `n_decoder_layers>1`。
* 未覆盖：训练循环 / 优化器 / 数据增强 / 分布式包装（`_fsdp_wrap_modules`）、
  `observation.environment_state` 之外的其它模态（本实现支持 env_state，但没有真实用例）、
  AMP/bf16 推理（模型全程 fp32；如需半精度，`.to(torch.bfloat16)` 后自行验证容差）。
* 视觉主干支持 `resnet18/34`（BasicBlock）与 `resnet50/101/152`（Bottleneck）；ACT 实际只用 resnet18。
  `replace_final_stride_with_dilation` 在 BasicBlock 上也能用（torchvision 会直接 raise，所以这一项没有对拍基准）。

## 资产

| 路径 | 内容 | 大小 |
| --- | --- | --- |
| `weights/act_aloha_sim_insertion_human/` | `lerobot/act_aloha_sim_insertion_human`：`model.safetensors` + `config.json` + `train_config.json` + `eval_info.json` | 198MB |
| `dataset/data/aloha_sim_insertion_human/` | `lerobot/aloha_sim_insertion_human` v3.0：50 集 / 25000 帧 / 480×640 AV1 视频 | 88MB |

两个都在 `.gitignore` 里（体积大）。权重对应的训练配置：`chunk_size=100`、
`dim_model=512`、4 层 encoder + 1 层 decoder、VAE 开、`use_imagenet_stats=true`
（所以图像用 ImageNet 均值方差，状态/动作用数据集统计量）。

真实数据 demo（`python -m policy.ACT.run_act --frame 0`）的输出片段：

```
chunk   : (1, 100, 14) 标准差=0.5222
L1(预测, GT) = 0.0137          # 100 步动作块与数据集动作的平均绝对误差
第一步   : 预测 [0.0096, -0.9812, 1.1833, -0.0049]
          GT   [0.0092, -0.9342, 1.1735, -0.0046]
train loss = 0.0800  {'l1_loss': 0.0795, 'kld_loss': 5.9e-05}
```

## 参考

* lerobot 0.6.2：`src/lerobot/policies/act/{modeling_act,configuration_act}.py`
  （Apache-2.0）；`src/lerobot/processor/normalize_processor.py`（归一化数学）
* 原版实现：<https://github.com/tonyzhaozh/act>
