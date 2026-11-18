"""CPU checks of the actual composed launch path, without Ray or model weights."""
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import argparse
import sys
root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, default=root / 'logs/cpu_preflight')
parser.add_argument('--config-name', default='qwen3_4b_nothink_smoke')
parser.add_argument('--model-path')
args = parser.parse_args()
out = args.output_dir / 'preflight.json'
out.unlink(missing_ok=True)
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import train
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
from ragen.trainer.agent_trainer import RayAgentTrainer
from ragen.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, _compat_sample_indices

for size in (1, 17, 1024, 16777217, 4022468608):
    indices = _compat_sample_indices(size, 'cpu')
    assert indices.dtype == torch.long
    assert indices.min() >= 0 and indices.max() < size
    assert indices[0] == 0 and indices[-1] == size - 1
    assert bool((indices[1:] >= indices[:-1]).all())

with initialize_config_dir(config_dir=str(root / 'config'), version_base=None):
    config = compose(config_name=args.config_name,
                     overrides=[f'model_path={args.model_path}'] if args.model_path else [])
config = train.add_dependency_and_validate_config(config)
tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
model_config = AutoConfig.from_pretrained(config.model_path, local_files_only=True)
assert type(model_config) in AutoModelForCausalLM._model_mapping
mapping = {Role.ActorRollout: ActorRolloutRefWorker, Role.Critic: CriticWorker,
           Role.RefPolicy: ActorRolloutRefWorker}
trainer = RayAgentTrainer(config=config, tokenizer=tokenizer, role_worker_mapping=mapping,
                          resource_pool_manager=ResourcePoolManager({'global': [1]},
                              {role: 'global' for role in mapping}),
                          reward_fn=train.DummyRewardManager(tokenizer, 0),
                          val_reward_fn=train.DummyRewardManager(tokenizer, 0))
assert not trainer.use_critic
assert trainer.use_reference_policy
assert not torch.cuda.is_initialized()
assert config.data.apply_chat_template_kwargs.enable_thinking is False
assert config.agent_proxy.enable_think is False
result = {'passed': True, 'cuda_initialized': False,
          'python': sys.executable, 'prefix': sys.prefix,
          'ragen_source': str(importlib.import_module('ragen').__file__),
          'backend': str(importlib.import_module('verl').__file__),
          'versions': {name: metadata.version(name) for name in
                       ['verl', 'torch', 'transformers', 'vllm', 'ray', 'tensordict', 'numpy', 'scipy']},
          'architecture': model_config.architectures,
          'algorithm': config.algorithm.adv_estimator,
          'response_mask': config.enable_response_mask,
          'native_thinking': config.data.apply_chat_template_kwargs.enable_thinking,
          'legacy_format_think': config.agent_proxy.enable_think,
          'use_critic': trainer.use_critic, 'use_reference': trainer.use_reference_policy,
          'actor_scheduler': config.actor_rollout_ref.actor.optim.lr_scheduler_type}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2))
OmegaConf.save(config, args.output_dir / 'resolved_launch_config.yaml', resolve=True)
print(json.dumps(result))
