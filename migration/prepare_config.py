"""Freeze veRL 0.6.1 required defaults underneath the actual legacy defaults."""
from pathlib import Path
import json
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

root = Path(__file__).resolve().parents[1]
with initialize_config_dir(config_dir=str(root / 'verl/verl/trainer/config'), version_base=None):
    modern = compose(config_name='ppo_trainer')
OmegaConf.set_struct(modern, False)
legacy = OmegaConf.load(root / 'migration/legacy_ppo_trainer.yaml')
merged = OmegaConf.merge(modern, legacy)
for role in ('actor_rollout_ref.actor', 'critic'):
    old_style = OmegaConf.select(legacy, role + '.optim.warmup_style')
    if old_style is not None:
        OmegaConf.update(merged, role + '.optim.lr_scheduler_type', old_style)
destination = root / 'config/ppo_trainer.yaml'
if destination.is_symlink():
    destination.unlink()
OmegaConf.save(merged, destination)
assert OmegaConf.to_container(OmegaConf.merge(modern, legacy), resolve=False) is not None
print(json.dumps({'config': str(destination), 'legacy_defaults_preserved': True,
                  'backend_commit': 'd62da4950573d7a4b7ef2362337952e7ab59e78d'}))
