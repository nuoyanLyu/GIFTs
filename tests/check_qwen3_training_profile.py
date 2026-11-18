import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch
from transformers import AutoTokenizer
from verl import DataProto
from hydra import compose, initialize_config_dir

from ragen.llm_agent.ctx_manager import ContextManager
from ragen.llm_agent.es_manager import EnvStateManager
from train import add_dependency_and_validate_config


def task_check(path, tokenizer, tag):
    config = add_dependency_and_validate_config(OmegaConf.load(path))
    assert config.model_path == tokenizer.name_or_path and config.enable_response_mask
    assert config.agent_proxy.response_protocol == "final_answer_with_prefix"
    assert config.agent_proxy.enable_think is False
    assert config.data.apply_chat_template_kwargs.enable_thinking is False
    assert config.algorithm.adv_estimator == "grpo"
    assert config.agent_proxy.reward_normalization.method == "mean_std"
    assert config.actor_rollout_ref.actor.use_kl_loss
    rollout = config.actor_rollout_ref.rollout
    assert rollout.mode == "sync"
    for sampling in (rollout, rollout.val_kwargs):
        assert (sampling.do_sample, sampling.temperature, sampling.top_p, sampling.top_k) == (True, 0.7, 0.8, 20)
    assert config.es_manager.train.env_groups == 8
    assert config.es_manager.train.group_size == 16
    assert list(config.es_manager.train.env_configs.tags) == [tag]

    small = deepcopy(config)
    small.es_manager.train.env_groups = 1
    small.es_manager.train.group_size = 2
    small.es_manager.train.env_configs.n_groups = [1]
    env = EnvStateManager(small)
    context = ContextManager(small, tokenizer)
    observations = env.reset()
    native = tokenizer.apply_chat_template(
        context.get_lm_inputs(deepcopy(observations), prepare_for_update=False)
        .non_tensor_batch["messages_list"][0].tolist(),
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = context.get_lm_inputs(deepcopy(observations), prepare_for_update=False)
    assert inputs.batch["input_ids"][0][inputs.batch["attention_mask"][0].bool()].tolist() == native
    first = env.envs[0]["env"]
    gold = first.question0["extra_info"]["answer"] if tag == "MathTrain" else str(first._get_eq() + 1)
    valid = f"An explanation goes before the answer. <answer>{gold}</answer>"
    invalid = valid + " <answer>2</answer>"
    bodies = [tokenizer.encode(text, add_special_tokens=False) for text in (valid, invalid)]
    padded = [ids + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (max(map(len, bodies)) - len(ids))
              for ids in bodies]
    rollout_output = DataProto.from_dict(
        tensors={"responses": torch.tensor(padded)},
        non_tensors={"env_ids": np.array([0, 1], dtype=object)},
    )
    parsed = context.get_env_inputs(rollout_output)
    assert [row["actions"] for row in parsed] == [[gold], []]
    env.step(parsed)
    state = env.get_rollout_states()
    assert [row["history"][0]["reward"] for row in state] == [1, 0]
    assert [row["penalty"] for row in state] == [0, config.es_manager.format_penalty]
    batch = context.formulate_rollouts(state)
    for index, (text, ids) in enumerate(zip((valid, invalid), bodies)):
        packed = batch.batch["input_ids"][index][batch.batch["attention_mask"][index].bool()].tolist()
        assert packed[:len(native) + len(ids)] == native + ids
        assert batch.non_tensor_batch["messages_list"][index][-1]["content"] == text
        mask = batch.batch["loss_mask"][index][batch.batch["attention_mask"][index, 1:].bool()]
        assert mask[len(native) - 1:len(native) + len(ids) - 1].all()
    return {"task": tag, "train_groups": config.es_manager.train.env_groups,
            "group_size": config.es_manager.train.group_size, "val_groups": config.es_manager.val.env_groups,
            "response_length": rollout.response_length, "max_model_len": rollout.max_model_len,
            "correct_reward": 1, "duplicate_answer_penalty": config.es_manager.format_penalty,
            "prompt_and_sampled_tokens_preserved": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("math", type=Path)
    parser.add_argument("nash", type=Path)
    parser.add_argument("config_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model_path = OmegaConf.load(args.math).model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base=None):
        default = compose(config_name="_10_math")
        preset = compose(config_name="qwen3_4b_nothink")
        custom = compose(config_name="_16_nash_new", overrides=[
            "model_profile=qwen3_4b_nothink", "actor_rollout_ref.rollout.temperature=0.9"
        ])
    assert default.agent_proxy.response_protocol == "legacy" and default.agent_proxy.enable_think
    assert preset.model_path == model_path and preset.agent_proxy.response_protocol == "final_answer_with_prefix"
    assert custom.actor_rollout_ref.rollout.temperature == 0.9
    assert custom.actor_rollout_ref.rollout.val_kwargs.temperature == 0.7
    report = {"no_profile_preserves_legacy": True, "standalone_preset": True, "cli_override": True,
              "tasks": [task_check(args.math, tokenizer, "MathTrain"), task_check(args.nash, tokenizer, "NashNew")],
              "cuda_initialized": torch.cuda.is_initialized()}
    assert not report["cuda_initialized"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
