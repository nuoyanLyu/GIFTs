import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from transformers import AutoConfig, AutoTokenizer
from verl import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import RolloutConfig
from verl.workers.rollout.vllm_rollout import vllm_rollout_spmd as backend

from ragen.workers.fsdp_workers import _backend_config

EXPECTED = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    assert config.data.apply_chat_template_kwargs.enable_thinking is False
    assert config.agent_proxy.enable_think is False
    assert config.agent_proxy.response_protocol == "final_answer_with_prefix"

    rollout = config.actor_rollout_ref.rollout
    assert rollout.mode == "sync"
    assert rollout.do_sample and rollout.val_kwargs.do_sample
    effective = omega_conf_to_dataclass(_backend_config(rollout, RolloutConfig, "rollout"))
    path = config.actor_rollout_ref.model.path
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = SimpleNamespace(
        local_path=path, tokenizer=tokenizer, hf_config=AutoConfig.from_pretrained(path, local_files_only=True),
        trust_remote_code=False, lora_rank=0,
    )
    captured = []

    def generate(prompts, sampling_params, **kwargs):
        captured.append({key: getattr(sampling_params, key) for key in EXPECTED})
        assert sampling_params.max_tokens == rollout.response_length
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[16, tokenizer.eos_token_id])])]

    fake_engine = SimpleNamespace(generate=generate)
    with patch.object(backend, "LLM", return_value=fake_engine), \
            patch.object(backend.torch.distributed, "get_world_size", return_value=rollout.tensor_model_parallel_size):
        worker = backend.vLLMRollout(effective, model, None)
        for validate in (False, True):
            ids = torch.tensor([tokenizer.encode("Sampling configuration check.")])
            inputs = DataProto.from_dict(
                tensors={"input_ids": ids, "attention_mask": torch.ones_like(ids),
                         "position_ids": torch.arange(ids.size(1)).unsqueeze(0)},
                meta_info={"eos_token_id": tokenizer.eos_token_id, "validate": validate, "do_sample": True},
            )
            worker.generate_sequences(inputs)
    assert captured == [EXPECTED, EXPECTED], captured
    report = {
        "model": path, "enable_thinking": False, "response_length": rollout.response_length,
        "train_engine_params": captured[0], "validation_engine_params": captured[1],
        "vllm_rollout_source": backend.__file__, "backend_val_config": asdict(effective.val_kwargs),
        "cuda_initialized": torch.cuda.is_initialized(), "actual_engine_stubbed": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
