"""ACT（Action Chunking Transformer）独立实现。

用法见 `policy/ACT/README.md`：

    from policy.ACT import load_act_policy, ACTNormalizer

    policy, norm_stats = load_act_policy("weights/act_aloha_sim_insertion_human")
    normalizer = ACTNormalizer.from_checkpoint_buffers(policy.config, norm_stats)
"""

from .modeling_act import (
    ACT,
    ACTION,
    ACTConfig,
    ACTPolicy,
    ACTTemporalEnsembler,
    PolicyFeature,
    build_act_from_config_json,
    build_act_from_yaml,
    load_act_policy,
    load_state_dict_from_checkpoint,
    split_normalization_buffers,
)
from .preprocessing import ACTNormalizer, batch_from_sample, load_dataset_stats

__all__ = [
    "ACT",
    "ACTION",
    "ACTConfig",
    "ACTNormalizer",
    "ACTPolicy",
    "ACTTemporalEnsembler",
    "PolicyFeature",
    "batch_from_sample",
    "build_act_from_config_json",
    "build_act_from_yaml",
    "load_act_policy",
    "load_dataset_stats",
    "load_state_dict_from_checkpoint",
    "split_normalization_buffers",
]
