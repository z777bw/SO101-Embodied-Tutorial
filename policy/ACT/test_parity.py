#!/usr/bin/env python
"""ACT 与 lerobot 参考实现的对拍脚本。

三层验证，从便宜到贵：

1. `--variants`     小尺寸随机权重，覆盖架构开关（pre_norm / use_vae / decoder 层数 /
                    激活函数 / chunk_size），逐模块比较 encoder / decoder / action 输出；
2. `--real-weights` 真实 ckpt（206MB）的同一份权重灌进两边，比较 eval 与 train 两条路径
                    （train 路径含 VAE 采样与 loss，dropout 置 0 以保证确定性）；
3. `--real-data`    真实数据集样本（视频解码 + 归一化）→ 前向 / loss / select_action。

用法：
    python -m policy.ACT.test_parity                      # 只跑 1 + 2
    python -m policy.ACT.test_parity --real-data \\
        --dataset dataset/data/aloha_sim_insertion_human \\
        --weights weights/act_aloha_sim_insertion_human
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .modeling_act import (
    ACTConfig,
    ACTPolicy,
    _collate_images,
    _read_safetensors,
    load_act_policy,
    split_normalization_buffers,
)
from .preprocessing import ACTNormalizer


# ---------------------------------------------------------------------------
# lerobot 参考实现（仅在测试里 import，生产代码不依赖）
# ---------------------------------------------------------------------------
def _ref_policy(ref_config):
    from lerobot.policies.act.modeling_act import ACTPolicy as RefACTPolicy

    return RefACTPolicy(ref_config)


def _ref_config(features, **kwargs):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig as RefACTConfig

    type_map = {"STATE": FeatureType.STATE, "VISUAL": FeatureType.VISUAL, "ACTION": FeatureType.ACTION}
    input_features = {
        k: PolicyFeature(type=type_map[t], shape=tuple(s)) for k, (t, s) in features["inputs"].items()
    }
    output_features = {
        k: PolicyFeature(type=type_map[t], shape=tuple(s)) for k, (t, s) in features["outputs"].items()
    }
    return RefACTConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        pretrained_backbone_weights=None,  # 避免测试时联网下载 resnet18
        **kwargs,
    )


def _my_config(features, **kwargs):
    from .modeling_act import PolicyFeature

    input_features = {k: PolicyFeature(type=t, shape=tuple(s)) for k, (t, s) in features["inputs"].items()}
    output_features = {k: PolicyFeature(type=t, shape=tuple(s)) for k, (t, s) in features["outputs"].items()}
    return ACTConfig(
        input_features=input_features,
        output_features=output_features,
        pretrained_backbone_weights=None,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise AssertionError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return (a.double() - b.double()).abs().max().item()


def _compare(name: str, mine, ref, tol: float = 0.0) -> float:
    if mine is None and ref is None:  # eval 模式下没有 VAE，mu/log_sigma 就是 None
        print(f"  [SKIP] {name:<24} (both None)")
        return 0.0
    diff = _max_diff(mine, ref)
    status = "OK " if diff <= tol else "BAD"
    print(f"  [{status}] {name:<24} max|diff|={diff:.3e}  shape={tuple(mine.shape)}")
    if diff > tol:
        raise AssertionError(f"{name}: diff {diff:.3e} > tol {tol:.3e}")
    return diff


class _Buffer:
    """收集 hook 到的中间量。"""

    def __init__(self):
        self.store: dict[str, tuple] = {}

    def hook(self, tag: str):
        def fn(module, args, kwargs, out):
            self.store[tag] = (args, kwargs, out)

        return fn


def _attach(model: ACTPolicy, prefix: str, buf: _Buffer):
    handles = [
        model.model.encoder.register_forward_hook(buf.hook(f"{prefix}.encoder"), with_kwargs=True),
        model.model.decoder.register_forward_hook(buf.hook(f"{prefix}.decoder"), with_kwargs=True),
        model.model.action_head.register_forward_hook(buf.hook(f"{prefix}.head"), with_kwargs=True),
    ]
    if model.config.use_vae:
        handles.append(
            model.model.vae_encoder.register_forward_hook(buf.hook(f"{prefix}.vae_encoder"), with_kwargs=True)
        )
    return handles


def _disable_dropout(model: torch.nn.Module) -> None:
    """对拍 train 路径时把 dropout 关掉：两边都必须确定性，否则差异来自随机数而非实现。

    `nn.MultiheadAttention` 的 dropout 是普通属性（内部走 `F.multi_head_attention_forward`），
    不会以 `nn.Dropout` 子模块的形式出现，所以也要显式清零。
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = 0.0


def _make_batch(config: ACTConfig, batch_size: int, image_shape: tuple[int, int], *, with_actions: bool, seed=0):
    g = torch.Generator().manual_seed(seed)
    batch = {
        "observation.state": torch.randn(batch_size, config.robot_state_feature.shape[0], generator=g),
    }
    for key in config.image_features:
        batch[key] = torch.rand(batch_size, 3, *image_shape, generator=g)
    if with_actions:
        action_dim = config.action_feature.shape[0]
        batch["action"] = torch.randn(batch_size, config.chunk_size, action_dim, generator=g)
        # 末尾 pad 几帧，顺便验证 key_padding_mask 的语义
        pad = torch.zeros(batch_size, config.chunk_size, dtype=torch.bool)
        pad[:, -max(1, config.chunk_size // 8) :] = True
        batch["action_is_pad"] = pad
    return batch


# ---------------------------------------------------------------------------
# 1. 结构对拍：小尺寸 + 架构开关
# ---------------------------------------------------------------------------
VARIANTS = {
    "default": {},
    "pre_norm": {"pre_norm": True},
    "no_vae": {"use_vae": False},
    "two_decoder_layers": {"n_decoder_layers": 2},
    "gelu_kld2": {"feedforward_activation": "gelu", "kl_weight": 2.0},
    "vae_1layer": {"n_vae_encoder_layers": 1, "n_encoder_layers": 2},
}


def run_variants(tol: float = 0.0) -> None:
    features = {
        "inputs": {"observation.images.top": ("VISUAL", (3, 64, 64)), "observation.state": ("STATE", (8,))},
        "outputs": {"action": ("ACTION", (6,))},
    }
    tiny = {
        "dim_model": 32,
        "n_heads": 2,
        "dim_feedforward": 64,
        "chunk_size": 8,
        "n_action_steps": 8,
        "latent_dim": 4,
        "dropout": 0.0,
        "vision_backbone": "resnet18",
    }
    print("=" * 78)
    print("1) 结构对拍：tiny 配置，逐个架构开关比较 encoder/decoder/head 输出")
    print("=" * 78)

    for name, overrides in VARIANTS.items():
        kwargs = {**tiny, **overrides}
        print(f"\n[{name}] overrides={overrides}")
        torch.manual_seed(0)
        mine = ACTPolicy(_my_config(features, **kwargs))
        ref = _ref_policy(_ref_config(features, **kwargs))
        report = ref.load_state_dict(mine.state_dict(), strict=False)
        if report.missing_keys or report.unexpected_keys:
            raise AssertionError(f"reference rejected weights: {report}")

        mine.eval()
        ref.eval()
        batch = _make_batch(mine.config, 2, (64, 64), with_actions=True)

        buf = _Buffer()
        handles = _attach(mine, "mine", buf) + _attach(ref, "ref", buf)
        with torch.no_grad():
            loss_mine, dict_mine = mine(batch)
            loss_ref, dict_ref = ref(batch)
        for h in handles:
            h.remove()

        # encoder / decoder / head 的输入与输出
        for tag in ("encoder", "decoder", "head"):
            key_mine, key_ref = f"mine.{tag}", f"ref.{tag}"
            if key_mine not in buf.store:
                continue
            args_mine, kwargs_mine, out_mine = buf.store[key_mine]
            args_ref, kwargs_ref, out_ref = buf.store[key_ref]
            for idx, (a, b) in enumerate(zip(args_mine, args_ref, strict=True)):
                if torch.is_tensor(a):
                    _compare(f"{tag}.in{idx}", a, b, tol)
            for kw in ("pos_embed", "key_padding_mask", "encoder_pos_embed", "decoder_pos_embed"):
                a, b = kwargs_mine.get(kw), kwargs_ref.get(kw)
                if a is not None and b is not None:
                    _compare(f"{tag}.kw:{kw}", a, b, tol)
            if torch.is_tensor(out_mine):
                _compare(f"{tag}.out", out_mine, out_ref, tol)

        # 动作输出（eval 模式下 latent 全零，无随机性）
        with torch.no_grad():
            actions_mine, _ = mine.model(_collate_images(batch, mine.config))
            actions_ref, _ = ref.model(_collate_images(batch, ref.config))
        _compare("actions", actions_mine, actions_ref, tol)
        print(f"  [INFO] eval 路径 l1={dict_mine['l1_loss']:.6f} (ref {dict_ref['l1_loss']:.6f})")

        # train 路径：VAE 采样 + loss（dropout=0 + 相同种子 => 逐位一致）
        mine.train()
        ref.train()
        torch.manual_seed(1234)
        loss_mine, dict_mine = mine(batch)
        torch.manual_seed(1234)
        loss_ref, dict_ref = ref(batch)
        _compare("train.loss", loss_mine.detach(), loss_ref.detach(), tol)
        _compare("train.l1", torch.tensor(dict_mine["l1_loss"]), torch.tensor(dict_ref["l1_loss"]), tol)
        if "kld_loss" in dict_mine:
            _compare("train.kld", torch.tensor(dict_mine["kld_loss"]), torch.tensor(dict_ref["kld_loss"]), 1e-9)

        # 梯度对拍（只抽几个参数）
        loss_mine.backward()
        loss_ref.backward()
        for pname in ("model.action_head.weight", "model.encoder.layers.0.linear1.weight"):
            pm = dict(mine.named_parameters())[pname].grad
            pr = dict(ref.named_parameters())[pname].grad
            _compare(f"grad:{pname.split('.')[-2]}.{pname.split('.')[-1]}", pm, pr, 1e-9)

        del actions_mine, actions_ref, mine, ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 2. 真实权重的对拍
# ---------------------------------------------------------------------------
def run_real_weights(weights_dir: str, tol: float = 0.0) -> None:
    import json

    print("\n" + "=" * 78)
    print(f"2) 真实权重对拍：{weights_dir}")
    print("=" * 78)

    mine, norm_stats = load_act_policy(weights_dir)
    config_dict = json.loads((Path(weights_dir) / "config.json").read_text(encoding="utf-8"))
    features = {
        "inputs": {k: (v["type"], tuple(v["shape"])) for k, v in config_dict["input_features"].items()},
        "outputs": {k: (v["type"], tuple(v["shape"])) for k, v in config_dict["output_features"].items()},
    }
    overrides = {
        "chunk_size": config_dict["chunk_size"],
        "n_action_steps": config_dict["n_action_steps"],
        "dim_model": config_dict["dim_model"],
        "n_heads": config_dict["n_heads"],
        "dim_feedforward": config_dict["dim_feedforward"],
        "feedforward_activation": config_dict["feedforward_activation"],
        "n_encoder_layers": config_dict["n_encoder_layers"],
        "n_decoder_layers": config_dict["n_decoder_layers"],
        "use_vae": config_dict["use_vae"],
        "latent_dim": config_dict["latent_dim"],
        "n_vae_encoder_layers": config_dict["n_vae_encoder_layers"],
        "pre_norm": config_dict["pre_norm"],
        "dropout": 0.0,  # 让 train 路径可复现（两边一致即可）
        "kl_weight": config_dict["kl_weight"],
        "replace_final_stride_with_dilation": config_dict["replace_final_stride_with_dilation"],
    }
    ref = _ref_policy(_ref_config(features, **overrides))
    state_dict, _ = split_normalization_buffers(_read_safetensors(weights_dir))
    report = ref.load_state_dict(state_dict, strict=False)
    print(f"  reference load: missing={report.missing_keys} unexpected={report.unexpected_keys}")
    if report.missing_keys or report.unexpected_keys:
        raise AssertionError("reference model and checkpoint do not match")

    image_shape = next(iter(mine.config.image_features.values())).shape[-2:]
    batch = _make_batch(mine.config, 1, image_shape, with_actions=True, seed=7)

    mine.eval()
    ref.eval()
    mine_batch = _collate_images(batch, mine.config)
    ref_batch = _collate_images(batch, ref.config)
    buf = _Buffer()
    handles = _attach(mine, "mine", buf) + _attach(ref, "ref", buf)
    with torch.no_grad():
        a_mine, (mu_mine, log_mine) = mine.model(mine_batch)
        a_ref, (mu_ref, log_ref) = ref.model(ref_batch)
    for h in handles:
        h.remove()
    _compare("eval.actions", a_mine, a_ref, tol)
    _compare("eval.mu", mu_mine, mu_ref, tol)
    _compare("eval.log_sigma_x2", log_mine, log_ref, tol)
    _compare("eval.encoder.out", buf.store["mine.encoder"][2], buf.store["ref.encoder"][2], tol)
    _compare("eval.decoder.out", buf.store["mine.decoder"][2], buf.store["ref.decoder"][2], tol)

    # train 路径：VAE + loss（dropout 关掉 + seed 固定 => 两边只有实现差异才会体现在数值上）
    _disable_dropout(mine)
    _disable_dropout(ref)
    mine.train()
    ref.train()
    torch.manual_seed(2024)
    with torch.no_grad():
        a_mine, (mu_mine, log_mine) = mine.model(mine_batch)
    torch.manual_seed(2024)
    with torch.no_grad():
        a_ref, (mu_ref, log_ref) = ref.model(ref_batch)
    _compare("train.actions", a_mine, a_ref, tol)
    _compare("train.mu", mu_mine, mu_ref, tol)
    _compare("train.log_sigma_x2", log_mine, log_ref, tol)

    mine.train()
    ref.train()
    torch.manual_seed(99)
    loss_mine, d_mine = mine(batch)
    torch.manual_seed(99)
    loss_ref, _ = ref(batch)
    _compare("train.loss", loss_mine.detach(), loss_ref.detach(), tol)
    print(f"  [INFO] loss={loss_mine.item():.6f} l1={d_mine['l1_loss']:.6f} kld={d_mine.get('kld_loss', 0):.6f}")

    # 归一化统计量与 ckpt 内嵌 buffer 一致
    print("  normalization buffers:", {k: {n: tuple(v.shape) for n, v in s.items()} for k, s in norm_stats.items()})
    assert "action" in norm_stats and "observation_state" in norm_stats


# ---------------------------------------------------------------------------
# 3. 真实数据：数据加载 + 归一化 + 前向 / loss / select_action
# ---------------------------------------------------------------------------
def run_real_data(dataset_root: str, weights_dir: str, eps: float = 1e-5) -> None:
    from dataset.lerobot_v3 import LeRobotV3Dataset

    print("\n" + "=" * 78)
    print(f"3) 真实数据前向：{dataset_root}")
    print("=" * 78)

    mine, norm_stats = load_act_policy(weights_dir)
    normalizer = ACTNormalizer.from_checkpoint_buffers(mine.config, norm_stats)

    delta_timestamps = {"action": [i / 50 for i in range(mine.config.chunk_size)]}
    ds = LeRobotV3Dataset(dataset_root, delta_timestamps=delta_timestamps)
    print(f"  {ds}")
    sample = ds[0]
    print(f"  sample: task={sample['task']!r} image={tuple(sample['observation.images.top'].shape)} "
          f"state={tuple(sample['observation.state'].shape)} action={tuple(sample['action'].shape)} "
          f"pad={int(sample['action_is_pad'].sum())}")

    batch = {
        "observation.state": sample["observation.state"].unsqueeze(0),
        "observation.images.top": sample["observation.images.top"].unsqueeze(0),
        "action": sample["action"].unsqueeze(0),
        "action_is_pad": sample["action_is_pad"].unsqueeze(0),
    }
    batch = normalizer.normalize(batch)

    with torch.no_grad():
        actions = mine.predict_action_chunk(batch)
    print(f"  predict_action_chunk: {tuple(actions.shape)} mean={actions.mean():.4f} std={actions.std():.4f}")

    # 与 lerobot 参考实现比（同一份权重、同一份输入）
    import json

    config_dict = json.loads((Path(weights_dir) / "config.json").read_text(encoding="utf-8"))
    features = {
        "inputs": {k: (v["type"], tuple(v["shape"])) for k, v in config_dict["input_features"].items()},
        "outputs": {k: (v["type"], tuple(v["shape"])) for k, v in config_dict["output_features"].items()},
    }
    ref = _ref_policy(_ref_config(features, **_ref_overrides(config_dict)))
    state_dict, _ = split_normalization_buffers(_read_safetensors(weights_dir))
    ref.load_state_dict(state_dict, strict=False)
    ref.eval()
    with torch.no_grad():
        actions_ref = ref.predict_action_chunk(batch)
    _compare("real_data.actions", actions, actions_ref, 0.0)

    # 训练 loss（VAE 路径）；关 dropout 让两边只剩实现差异
    _disable_dropout(mine)
    _disable_dropout(ref)
    mine.train()
    torch.manual_seed(3)
    loss, loss_dict = mine(batch)
    torch.manual_seed(3)
    ref.train()
    loss_ref, _ = ref(batch)
    _compare("real_data.loss", loss.detach(), loss_ref.detach(), 1e-9)
    print(f"  loss={loss.item():.6f} {loss_dict}")

    # 逐 step 推理：动作队列应覆盖 n_action_steps 步，且首个动作与 chunk 首帧一致
    mine.eval()
    mine.reset()
    step_actions = [mine.select_action(batch) for _ in range(3)]
    print(f"  select_action: {[tuple(a.shape) for a in step_actions]}")
    _compare("select_action#0", step_actions[0], actions[:, 0], 0.0)
    _compare("select_action#1", step_actions[1], actions[:, 1], 0.0)


def _ref_overrides(config_dict: dict) -> dict:
    return {
        "chunk_size": config_dict["chunk_size"],
        "n_action_steps": config_dict["n_action_steps"],
        "dim_model": config_dict["dim_model"],
        "n_heads": config_dict["n_heads"],
        "dim_feedforward": config_dict["dim_feedforward"],
        "feedforward_activation": config_dict["feedforward_activation"],
        "n_encoder_layers": config_dict["n_encoder_layers"],
        "n_decoder_layers": config_dict["n_decoder_layers"],
        "use_vae": config_dict["use_vae"],
        "latent_dim": config_dict["latent_dim"],
        "n_vae_encoder_layers": config_dict["n_vae_encoder_layers"],
        "pre_norm": config_dict["pre_norm"],
        "dropout": 0.0,
        "kl_weight": config_dict["kl_weight"],
        "replace_final_stride_with_dilation": config_dict["replace_final_stride_with_dilation"],
    }


def run_backbone_parity() -> None:
    """手写 ResNet 主干 vs torchvision：同权重、同输入，逐位比较 feature map。"""
    from torchvision import models as tv_models
    from torchvision.ops.misc import FrozenBatchNorm2d as TvFrozenBN

    from .modeling_act import _RESNET_ARCHS, _FrozenBatchNorm2d, _ResNet

    print("\n" + "=" * 78)
    print("0) 主干对拍：手写 ResNet vs torchvision（含 BasicBlock / Bottleneck）")
    print("=" * 78)

    cases = [
        ("resnet18", (False, False, False)),
        ("resnet34", (False, False, False)),
        ("resnet50", (False, False, False)),
        # 最后一段用空洞卷积代替 stride。torchvision 的 BasicBlock 不支持 dilation>1
        # （会直接 raise），所以这一项只能拿 Bottleneck 比；手写版对两者都支持。
        ("resnet50", (False, False, True)),
    ]
    for arch, dilation in cases:
        block, layers = _RESNET_ARCHS[arch]
        mine = _ResNet(block, layers, replace_stride_with_dilation=dilation, norm_layer=_FrozenBatchNorm2d)
        ref = getattr(tv_models, arch)(
            weights=None,
            norm_layer=TvFrozenBN,
            replace_stride_with_dilation=list(dilation),
        )
        # 手写主干没有分类头，torchvision 的 BatchNorm 还带 num_batches_tracked，都要过滤掉
        trunk = {
            k: v
            for k, v in ref.state_dict().items()
            if not k.startswith("fc.") and not k.endswith("num_batches_tracked")
        }
        report = mine.load_state_dict(trunk, strict=True)
        assert not report.missing_keys and not report.unexpected_keys

        x = torch.rand(2, 3, 128, 128)
        with torch.no_grad():
            out_mine = mine(x)
            y = ref.maxpool(ref.relu(ref.bn1(ref.conv1(x))))
            for stage in ("layer1", "layer2", "layer3", "layer4"):
                y = getattr(ref, stage)(y)
            out_ref = y
        diff = (out_mine.double() - out_ref.double()).abs().max().item()
        tag = f"{arch}{' +dilation' if dilation[-1] else ''}"
        status = "OK " if diff == 0.0 else "BAD"
        print(
            f"  [{status}] {tag:<18} feature_map={tuple(out_mine.shape)} "
            f"params={sum(p.numel() for p in mine.parameters()):>12,} max|diff|={diff:.3e}"
        )
        if diff != 0.0:
            raise AssertionError(f"{tag}: backbone diff {diff:.3e}")
        del mine, ref, out_mine, out_ref


def run_config_presets(weights_dir: str = "weights/act_aloha_sim_insertion_human") -> None:
    """校验 `config/ACT/*.yaml`：aloha 预设必须能严格加载官方 ckpt，so101 预设必须能前向。"""
    import json

    from .modeling_act import build_act_from_yaml

    print("\n" + "=" * 78)
    print("4) 预设配置校验：config/ACT/*.yaml")
    print("=" * 78)

    ckpt_config = json.loads((Path(weights_dir) / "config.json").read_text(encoding="utf-8"))
    cfg = ACTConfig.from_yaml("config/ACT/aloha.yaml")
    ignore = {"preset", "fps", "pretrained_backbone_weights"}  # 元信息 / 初始化策略，不影响结构
    diffs = {}
    for name in ACTConfig.__dataclass_fields__:
        if name in ignore or name not in ckpt_config:
            continue
        mine, theirs = getattr(cfg, name), ckpt_config[name]
        if name.endswith("_features"):
            mine = {k: (ft.type, tuple(ft.shape)) for k, ft in mine.items()}
            theirs = {k: (v["type"], tuple(v["shape"])) for k, v in theirs.items()}
        if mine != theirs:
            diffs[name] = (mine, theirs)
    if diffs:
        raise AssertionError(f"aloha.yaml 与 ckpt config 不一致: {diffs}")
    print("  [OK ] aloha.yaml 与 ckpt config 逐字段一致（preset/fps/初始化策略除外）")

    # 严格加载：结构对不上会直接报 missing/unexpected
    aloha = build_act_from_yaml("config/ACT/aloha.yaml")
    state_dict, _ = split_normalization_buffers(_read_safetensors(weights_dir))
    report = aloha.load_state_dict(state_dict, strict=True)
    assert not report.missing_keys and not report.unexpected_keys
    print(f"  [OK ] aloha.yaml 严格加载官方权重，参数 {sum(p.numel() for p in aloha.parameters()):,}")

    # so101 预设：结构、序列长度、随机前向（device 强制 cpu，保持测试与机器无关）
    so101 = build_act_from_yaml("config/ACT/so101.yaml", device="cpu")
    so101.eval()
    cfg101 = so101.config
    image_shape = next(iter(cfg101.image_features.values())).shape[-2:]
    batch = _make_batch(cfg101, 1, image_shape, with_actions=False, seed=11)
    with torch.no_grad():
        actions = so101.predict_action_chunk(batch)
    tokens = 1 + (1 if cfg101.robot_state_feature else 0) + len(cfg101.image_features) * (
        image_shape[0] // 32 * image_shape[1] // 32
    )
    print(
        f"  [OK ] so101.yaml 前向：state={cfg101.robot_state_feature.shape} action={cfg101.action_feature.shape} "
        f"cameras={len(cfg101.image_features)} encoder_tokens={tokens} out={tuple(actions.shape)} "
        f"params={sum(p.numel() for p in so101.parameters()):,}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="weights/act_aloha_sim_insertion_human")
    parser.add_argument("--dataset", default="dataset/data/aloha_sim_insertion_human")
    parser.add_argument("--skip-variants", action="store_true")
    parser.add_argument("--skip-real-weights", action="store_true")
    parser.add_argument("--skip-configs", action="store_true")
    parser.add_argument("--skip-backbone", action="store_true")
    parser.add_argument("--real-data", action="store_true", help="额外跑真实数据前向（需要已下载数据集）")
    args = parser.parse_args()

    if not args.skip_backbone:
        run_backbone_parity()
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
