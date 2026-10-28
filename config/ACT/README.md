# config/ACT —— 按本体划分的 ACT 配置

每个 YAML 对应"一个本体 + 一套相机 + 一个数据集"的建模参数。换机器人只改这里，
`policy/ACT/modeling_act.py` 不用动。

| 文件 | 本体 | state / action | 相机 | chunk_size @fps | 备注 |
| --- | --- | --- | --- | --- | --- |
| `aloha.yaml` | ALOHA 双臂 | 14 / 14 | `observation.images.top` 480×640 | 100 @50fps = 2.0s | 与 `weights/act_aloha_sim_insertion_human` 的训练配置逐字段一致，可直接加载该 ckpt |
| `so101.yaml` | SO-101 单臂 | 6 / 6 | `front` + `wrist` 480×640 | 45 @30fps = 1.5s | 从零训练；相机键名/数量按实际数据集改 |

## 用法

```python
from policy.ACT import ACTConfig, build_act_from_yaml

cfg = ACTConfig.from_yaml("config/ACT/so101.yaml")      # 只读配置（含 preset / fps 元信息）
policy = build_act_from_yaml("config/ACT/so101.yaml")   # 随机初始化，从头训练

# 已有的 aloha ckpt 仍然走权重目录（config.json 优先）：
from policy.ACT import load_act_policy
policy, norm_stats = load_act_policy("weights/act_aloha_sim_insertion_human")
```

命令行校验：

```bash
python -m policy.ACT.test_parity --skip-variants --skip-real-weights   # 只跑配置校验
```

## 字段说明

```yaml
name: so101        # 元信息 -> config.preset
fps: 30            # 元信息 -> config.fps，用来算 chunk_size/fps；不参与建模
notes: |           # 可选，随便写，解析时忽略
policy:            # 这一段才是 ACTConfig 的字段，出现未知键会直接报错
  ...
```

`policy` 里每个字段的含义见 `policy/ACT/modeling_act.py` 的 `ACTConfig`；和本体强相关的只有四项：

| 字段 | 说明 |
| --- | --- |
| `input_features["observation.state"].shape` | 本体自由度（SO-101 是 5 关节 + 夹爪 = 6） |
| `output_features["action"].shape` | 动作维度，通常与 state 相同 |
| `input_features["observation.images.<cam>"]` | 相机路数与分辨率；键名必须等于数据集里的特征名 |
| `chunk_size` / `n_action_steps` | 动作块长度；用 `fps` 换算成秒来判断是否过长 |

其余字段（`dim_model` / 层数 / 激活 / VAE）与本体无关，换本体时可以保持不变。

## 换新本体的检查清单

1. `observation.state` 维度 = 数据集 `meta/info.json` 里 `observation.state.shape[0]`；
2. `observation.images.*` 的键名、`shape` 与数据集 `features` 完全一致（shape 是 `(C, H, W)`）；
3. 所有相机的 `H×W` 必须相同或至少各自恒定（token 数是 `H/32 × W/32` 相加）；
4. `chunk_size / fps` 控制在 1~3 秒；`n_action_steps ≤ chunk_size`，用时间集成时只能为 1；
5. 归一化用**本数据集**的统计量（`meta/stats.json`），不要沿用别的本体的 buffer；
6. `pretrained_backbone_weights` 默认 `null`：离线可跑。想用 ImageNet 初始化就改成
   `"ResNet18_Weights.IMAGENET1K_V1"`（首次由 torchvision 下载 ~45MB，再拷进手写主干；
   加载已有 ckpt 时该字段被忽略）。

再加一个本体，就是复制一份 YAML 改上面几项，并追加到本 README 的表格里。
