"""Independently inspect the actual final run, checkpoint and exported metadata."""
import importlib.metadata as metadata
import json
import math
from pathlib import Path
import re
import argparse
import sys
import torch
from transformers import AutoTokenizer
from safetensors import safe_open
from urllib.request import Request, urlopen

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', type=Path, default=root / 'logs/gpu_smoke_attempt4')
parser.add_argument('--tmux', default='ragen2_qwen3_gpu4_20261117')
args = parser.parse_args()
run = args.run_dir.resolve()
status = json.loads((run / 'status.json').read_text())
assert status['training']['exit_code'] == 0
assert status['phase'] == 'complete_holding' and status['restoration']['verified']
text = re.sub(r'\x1b\[[0-9;]*m', '', (run / 'training.log').read_text())
steps = {}
for step, fields in re.findall(r'step:(\d+) - ([^\r\n]+)', text):
    values = {}
    for field in fields.split(' - '):
        key, value = field.rsplit(':', 1)
        values[key] = float(value)
    steps[int(step)] = values
assert set(steps) == {1, 2}
for values in steps.values():
    assert all(math.isfinite(value) for value in values.values())
    assert values['compat/parameter_changed_count'] > 0
    assert values['compat/parameter_max_abs_delta'] > 0
    assert values['compat/parameter_sample_finite'] == 1
    assert values['actor/grad_norm'] > 0
assert abs(steps[1]['actor/pg_loss']) > 0
assert steps[1]['critic/advantages/max'] > 0 > steps[1]['critic/advantages/min']

checkpoint = run / 'checkpoints/global_step_2/actor'
assert (run / 'checkpoints/latest_checkpointed_iteration.txt').read_text().strip() == '2'
extra = torch.load(checkpoint / 'extra_state_world_size_1_rank_0.pt', map_location='cpu', weights_only=False)
assert extra['lr_scheduler']['last_epoch'] == 2
weights = torch.load(checkpoint / 'model_world_size_1_rank_0.pt', map_location='meta', weights_only=True, mmap=True)
assert len(weights) == 399
assert json.loads((run / 'merge_status.json').read_text())['exit_code'] == 0
export = run / 'hf_model'
index = json.loads((export / 'model.safetensors.index.json').read_text())
keys = set()
for name in set(index['weight_map'].values()):
    with safe_open(export / name, framework='pt', device='cpu') as reader:
        shard_keys = set(reader.keys())
        assert shard_keys == {key for key, value in index['weight_map'].items() if value == name}
        keys.update(shard_keys)
assert keys == set(index['weight_map'])
original = AutoTokenizer.from_pretrained('/data1/model/Qwen3-4B', local_files_only=True)
for path in (checkpoint / 'huggingface', export):
    saved = AutoTokenizer.from_pretrained(path, local_files_only=True)
    assert saved.chat_template == original.chat_template
source_generation = json.loads(Path('/data1/model/Qwen3-4B/generation_config.json').read_text())
export_generation = json.loads((export / 'generation_config.json').read_text())
source_generation.pop('transformers_version', None)
export_generation.pop('transformers_version', None)
assert source_generation == export_generation

def request(path, body=None):
    req = Request('http://127.0.0.1:18440' + path,
                  data=None if body is None else json.dumps(body).encode(),
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=30) as response:
        payload = response.read()
        return response.status, json.loads(payload) if payload else None

health, _ = request('/health')
models_code, models = request('/v1/models')
model = '/data1/model/Qwen2.5-7B-Instruct'
assert model in [item['id'] for item in models['data']]
inference_code, generated = request('/v1/completions', {'model': model, 'prompt': 'Reply OK.',
                                                      'temperature': 0, 'max_tokens': 4})
assert health == models_code == inference_code == 200 and generated['choices'][0]['text']
assert not torch.cuda.is_initialized()
report = {'passed': True, 'training_exit_code': 0, 'global_steps': 2,
          'python': sys.executable, 'prefix': sys.prefix,
          'versions': {name: metadata.version(name) for name in
                       ('verl', 'torch', 'transformers', 'vllm', 'numpy', 'scipy')},
          'step_metrics': steps, 'checkpoint_path': str(checkpoint),
          'checkpoint_tensor_count': len(weights), 'scheduler_last_epoch': 2,
          'hf_export_path': str(export), 'hf_export_tensor_count': len(keys),
          'tokenizer_template_preserved': True, 'generation_config_preserved': True,
          'holder': {'tmux': args.tmux, 'gpu': 4, 'port': 18440,
                     'model': model, 'supervisor_pid': status['pid'],
                     'holder_pid': status['holder']['pid'], 'health': health,
                     'models': models_code, 'inference': inference_code},
          'limits': ['GPU check: single-card full-parameter GRPO on a local single-turn Bandit game.',
                     'Multi-turn template checked on CPU; multi-GPU/PPO critic/LoRA/PCGrad/CST not GPU-validated.',
                     'Smoke checkpoint saves model and scheduler/RNG, excludes optimizer state.',
                     'Diagnostic format counts are not a final benchmark or proof of RL effectiveness.']}
(run / 'verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps({key: report[key] for key in ('passed', 'global_steps', 'scheduler_last_epoch', 'holder')}))
