# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compatibility workers for the original MultiGames algorithms on veRL 0.6.1.

Model initialization, rollout weight synchronization, device-mesh dispatch and
checkpoint I/O follow the pinned veRL backend. The project-local actor/critic
implementations still own loss, masks, PCGrad and diagnostic computations.
No prompt or response tokens are constructed in this module.
"""

import logging
import os
import warnings
from dataclasses import fields, is_dataclass
from typing import get_args, get_type_hints
from typing import Union

import psutil
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.distributed.device_mesh import init_device_mesh

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker as VerlActorRolloutRefWorker,
    CriticWorker as VerlCriticWorker,
    RewardModelWorker as VerlRewardModelWorker,
    create_device_mesh,
    get_sharding_strategy,
)
from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


def _backend_config(config, dataclass_type, path="backend"):
    """Copy only backend schema fields while retaining the original config.

    RAGEN's rollout filters and legacy API knobs are consumed outside veRL.
    veRL 0.6.1 rejects those fields when constructing its strict dataclasses.
    Resolve references against the original config before copying so that
    interpolations such as ${actor_rollout_ref.actor.use_dynamic_bsz} survive.
    Nested backend dataclasses (e.g. multi_turn) need the same treatment.
    """
    values = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    hints = get_type_hints(dataclass_type)
    allowed = {field.name for field in fields(dataclass_type)}
    excluded = sorted(set(values) - allowed)
    if excluded:
        logger.info("RAGEN2 retains project/legacy fields outside %s: %s", path, excluded)
    result = {}
    for key, value in values.items():
        if key not in allowed:
            continue
        field_type = hints.get(key)
        nested_type = field_type if is_dataclass(field_type) else next(
            (kind for kind in get_args(field_type) if is_dataclass(kind)), None
        )
        if nested_type is not None and isinstance(value, dict):
            value = OmegaConf.to_container(_backend_config(value, nested_type, f"{path}.{key}"), resolve=True)
        result[key] = value
    result["_target_"] = f"{dataclass_type.__module__}.{dataclass_type.__name__}"
    return OmegaConf.create(result)


def _compat_sample_indices(numel, device):
    count = min(1024, numel)
    return (torch.arange(count, device=device, dtype=torch.long)
            * (numel - 1) // max(count - 1, 1))


class ActorRolloutRefWorker(VerlActorRolloutRefWorker):
    """veRL model/rollout backend with the unchanged MultiGames actor."""

    def _compat_parameter_sample(self):
        # Optional smoke evidence only; deterministic indexing never consumes RNG.
        samples = []
        with torch.no_grad():
            for parameter in self.actor_module_fsdp.parameters():
                if parameter.requires_grad and parameter.numel():
                    indices = _compat_sample_indices(parameter.numel(), parameter.device)
                    samples.append(parameter.detach().reshape(-1)[indices].float().cpu())
        return torch.cat(samples)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from ragen.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(
                    _backend_config(self.config.actor.fsdp_config, FSDPEngineConfig, "actor.fsdp_config"),
                    dataclass_type=FSDPEngineConfig,
                )
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            # Keep project-specific PCGrad/use_ref/etc. fields in DictConfig.
            # Converting the whole node to veRL's FSDPActorConfig rejects them.
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            actor_cfg = self.config.actor
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(
                    _backend_config(self.config.ref.fsdp_config, FSDPEngineConfig, "ref.fsdp_config"),
                    dataclass_type=FSDPEngineConfig,
                ),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    def _build_rollout(self, trust_remote_code=False):
        # The trainer still sees the complete original filter configuration.
        # Restrict only the temporary configuration passed to veRL construction.
        original_config = self.config
        if original_config.rollout.get("use_fire_sampling", False):
            raise NotImplementedError("Legacy FIRE sampling has no audited veRL 0.6.1 mapping.")
        if original_config.rollout.get("chat_scheduler") is not None:
            raise NotImplementedError("Legacy chat_scheduler requires a separate async rollout migration.")
        if original_config.rollout.get("multi_turn", {}).get("enable", False):
            raise NotImplementedError(
                "Native veRL tool-loop rollout is not part of the preserved RAGEN agent loop."
            )
        backend_config = OmegaConf.create(OmegaConf.to_container(original_config, resolve=False))
        backend_config.rollout = _backend_config(original_config.rollout, RolloutConfig, "rollout")
        backend_config.model = _backend_config(original_config.model, HFModelConfig, "model")
        try:
            self.config = backend_config
            return super()._build_rollout(trust_remote_code=trust_remote_code)
        finally:
            self.config = original_config

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        audit_before = self._compat_parameter_sample() if os.getenv('RAGEN_COMPAT_AUDIT') == '1' else None
        with self.ulysses_sharding_manager:
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            if audit_before is not None:
                audit_after = self._compat_parameter_sample()
                delta = (audit_after - audit_before).abs()
                metrics['compat/parameter_sample_count'] = delta.numel()
                metrics['compat/parameter_changed_count'] = (delta > 0).sum().item()
                metrics['compat/parameter_max_abs_delta'] = delta.max().item()
                metrics['compat/parameter_sample_finite'] = torch.isfinite(audit_after).all().item()
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = torch.cuda.max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def probe_gradient_cosine(self, data: DataProto):
        """Read-only counterpart of update_actor: computes per-task gradient cosine/norm
        (DataParallelPPOActor.probe_task_gradient_cosine) without touching model/optimizer
        state. See write/gradient_conflict_verification_plan.md Stage 1."""
        data = data.to(torch.cuda.current_device())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        with self.ulysses_sharding_manager:
            metrics = self.actor.probe_task_gradient_cosine(data=data)
            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during probe_gradient_cosine", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def probe_cst_chain(self, data: DataProto):
        """CST causal-interference chain probe (DataParallelPPOActor.probe_cst_chain): really
        applies single-task SGD updates and re-measures pairwise gradient cosines along one CST
        cycle. Mutates model weights in-place (never saved -- a fresh checkpoint is loaded per
        run). See write/gradient_conflict_verification_plan.md §5b/§5d."""
        data = data.to(torch.cuda.current_device())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        with self.ulysses_sharding_manager:
            metrics = self.actor.probe_cst_chain(data=data)
            output = DataProto(meta_info={"metrics": metrics})
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during probe_cst_chain", logger=logger)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_log_prob(self, data: DataProto, no_lora=False):
        # when no_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        
        # Support all hardwares
        from contextlib import nullcontext
        adapter_ctx = self.actor.actor_module.disable_adapter() if no_lora else nullcontext()
        with adapter_ctx:
            data = data.to(torch.cuda.current_device())
            # we should always recompute old_log_probs when it is HybridEngine
            data.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size_per_gpu
            data.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
            data.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
            data.meta_info['temperature'] = self.config.rollout.temperature
            # perform recompute log_prob
            with self.ulysses_sharding_manager:
                output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True, no_lora=no_lora)
                output = DataProto.from_dict(
                    tensors={"old_log_probs": output, "entropys": entropys},
                    meta_info={"temperature": self.config.rollout.temperature},
                )


            output = output.to('cpu')

            # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
            # unshard the root FSDP module
            if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
                self.actor.actor_module._handle.reshard(True)

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)

            log_gpu_memory_usage('After compute_log_prob', logger=logger)

            torch.cuda.empty_cache()
            return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data = self.compute_log_prob(data, no_lora=True)
            # this old_log_probs is in fact ref_log_prob
            data = DataProto.from_dict(tensors={'ref_log_prob': data.batch['old_log_probs']})
            return data
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model
        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    # generate_sequences, save_checkpoint and load_checkpoint are inherited from
    # pinned veRL 0.6.1. In sync mode, each generation call transfers current actor
    # weights, generates, then restores trainer mode. All actor/ref forward calls
    # above remain project-local and preserve their original masks/temperature.


class CriticWorker(VerlCriticWorker):
    """Keep the existing token-value loss and optimizer settings."""

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        from verl.utils.model import load_valuehead_model, print_model_size
        from verl.utils.torch_dtypes import PrecisionType

        local_path = copy_to_local(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))
        self.processor = hf_processor(tokenizer_path, trust_remote_code=config.model.get("trust_remote_code", False))

        from omegaconf import OmegaConf

        override_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f"Critic overriding config {override_config_kwargs}")

        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig

        trust_remote_code = config.model.get("trust_remote_code", False)
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1

        init_context = get_init_weight_context_manager(use_meta_tensor=not critic_model_config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            critic_model_config.classifier_dropout = 0.0
            critic_model_config.hidden_dropout = "0"
            # veRL 0.6.1 selects the supported token/value-head architecture.
            critic_model_config._attn_implementation = "flash_attention_2"
            critic_module = load_valuehead_model(
                local_path, torch_dtype, critic_model_config, trust_remote_code
            )

            use_remove_padding = config.model.get("use_remove_padding", False)
            if use_remove_padding or self.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=critic_module, use_remove_padding=use_remove_padding, ulysses_sp_size=self.ulysses_sequence_parallel_size)

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get("enable_gradient_checkpointing", False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if self._is_lora:
            print("Applying LoRA to critic module")
            critic_module.enable_input_require_grads()
            # Convert config to regular Python types before creating PEFT model
            lora_config = {
                'task_type': TaskType.TOKEN_CLS,
                'modules_to_save': ['score'],
                'r': self.config.model.lora_rank,
                'lora_alpha': self.config.model.lora_alpha,
                'target_modules': convert_to_regular_types(self.config.model.target_modules),
                'bias': "none",
            }
            critic_module = get_peft_model(critic_module, LoraConfig(**lora_config))

        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy, is_lora=self.config.model.get('lora_rank', 0) > 0)
        
        log_gpu_memory_usage("Before critic FSDP", logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        if config.strategy == "fsdp":
            critic_module = FSDP(
                critic_module,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                forward_prefetch=False,
                device_mesh=self.device_mesh,
                cpu_offload=None,
            )
        elif config.strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True)
            offload_policy = None
            if fsdp_config.offload_policy:
                self._is_offload_param = False
                self._is_offload_optimizer = False
                offload_policy = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": offload_policy,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = critic_module.state_dict()
            apply_fsdp2(critic_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(critic_module, full_state, fsdp_mesh, offload_policy)
        else:
            raise NotImplementedError(f"Unknown strategy {config.strategy}")

        log_gpu_memory_usage("After critic FSDP", logger=None)

        critic_optimizer = optim.AdamW(
            critic_module.parameters(),
            lr=config.optim.lr,
            betas=config.optim.get("betas", (0.9, 0.999)),
            weight_decay=config.optim.get("weight_decay", 1e-2),
        )

        total_steps = config.optim.get("total_training_steps", 0)
        num_warmup_steps = int(config.optim.get("lr_warmup_steps", -1))
        warmup_style = config.optim.get("warmup_style") or config.optim.get("lr_scheduler_type", "constant")
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = config.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

        from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

        if warmup_style == "constant":
            critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps)
        elif warmup_style == "cosine":
            critic_lr_scheduler = get_cosine_schedule_with_warmup(optimizer=critic_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
        else:
            raise NotImplementedError(f"Warmup style {warmup_style} is not supported")

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from ragen.workers.critic import DataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = DataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    def compute_values(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})

        output = output.to("cpu")
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    def update_critic(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:

            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output


class RewardModelWorker(VerlRewardModelWorker):
    """Learned reward-model backend; game rewards are computed by environments."""


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    """Prevent accidental switching from the preserved synchronous training path."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "RAGEN2 migration currently supports synchronous rollout only; "
            "async agent-loop semantics have not been ported or verified."
        )
