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
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import numpy as np
import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import verl.utils.torch_functional as verl_F
from verl import DataProto
from ragen.trainer.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

from peft import PeftModel
from ragen.utils import safe_logprobs_from_logits

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

        pcgrad_config = self.config.get("pcgrad", {}) or {}
        self.pcgrad_enable = pcgrad_config.get("enable", False)
        self.pcgrad_granularity = pcgrad_config.get("granularity", "layer")

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        # 'responses' mask remove the first token
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:  # False
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                inplace_backward = True
                if calculate_entropy:
                    inplace_backward = False
                # log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled, inplace_backward=inplace_backward)
                log_probs = safe_logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=input_ids_rmpad_rolled,
                    logprobs_fn=logprobs_from_logits,
                    inplace_backward=inplace_backward,
                )

                # compute entropy
                if calculate_entropy:
                    entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                # log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                log_probs = safe_logprobs_from_logits(logits, micro_batch["responses"], logprobs_fn=logprobs_from_logits)
                if calculate_entropy:
                    entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        # print('grad_norm', grad_norm)
        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"ERROR: rank {torch.distributed.get_rank()} actor grad_norm is not finite: {grad_norm}")
            exit(1)
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, no_lora=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]  # False

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        is_peft_model = not no_lora and isinstance(self.actor_module._fsdp_wrapped_module, PeftModel)
        # False，没有这个打印信息
        if is_peft_model:
            print(f"[INFO] Actor is a PeftModel")
            with FSDP.summon_full_params(self.actor_module):
                self.actor_module.merge_adapter()
            print(f"[INFO] Merged adapter actor")

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if is_peft_model:
            print(f"[INFO] Unmerging adapter actor")
            with FSDP.summon_full_params(self.actor_module):
                self.actor_module.unmerge_adapter()
            print(f"[INFO] Unmerged adapter actor")
        

        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs, entropys

    def _compute_actor_loss(self, micro_batch, temperature, metrics):
        """Compute the (unscaled) policy loss for one micro-batch. Does not call backward().

        `metrics` is mutated in place for the kl_loss/kl_coef keys, matching the pre-refactor
        behavior where these were written directly into the outer metrics dict (i.e. only the
        last micro-batch's value survives, not averaged via append_to_dict) rather than logged
        per micro-batch like the other losses.
        """
        response_mask = micro_batch["response_mask"]
        old_log_prob = micro_batch["old_log_probs"]
        advantages = micro_batch["advantages"]

        clip_ratio = self.config.clip_ratio
        clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
        clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
        entropy_coeff = self.config.entropy_coeff
        loss_agg_mode = self.config.loss_agg_mode

        # all return: (bsz, response_length)
        calculate_entropy = entropy_coeff != 0
        entropy, log_prob = self._forward_micro_batch(micro_batch=micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            cliprange=clip_ratio,
            cliprange_low=clip_ratio_low,
            cliprange_high=clip_ratio_high,
            clip_ratio_c=clip_ratio_c,
            loss_agg_mode=loss_agg_mode,
        )

        if entropy_coeff != 0:
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            policy_loss = pg_loss - entropy_loss * entropy_coeff
        else:
            policy_loss = pg_loss

        # 只有kl_loss_coef不为0的时候计算才有意义
        if self.config.use_kl_loss and self.config.kl_loss_coef:
            ref_log_prob = micro_batch["ref_log_prob"]
            kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)
            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
            metrics["actor/kl_loss"] = kl_loss.detach().item()
            metrics["actor/kl_coef"] = self.config.kl_loss_coef

        log_metrics = {
            "actor/pg_loss": pg_loss.detach().item(),
            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
            "actor/ppo_kl": ppo_kl.detach().item(),
            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        }
        if entropy_coeff != 0:
            log_metrics["actor/entropy_loss"] = entropy_loss.detach().item()

        return policy_loss, log_metrics

    def _pcgrad_process_group(self):
        """FSDP1 shards params within its own process group, not necessarily WORLD."""
        return getattr(self.actor_module, "process_group", None)

    def _pcgrad_project_layerwise(self, task_grad_buffer, param_list):
        """MGS: per named-parameter-tensor PCGrad. Uses each task's ORIGINAL (pre-projection)
        gradient as the projection basis for every other task, which makes the result
        independent of task iteration order (unlike the classic evolving-gradient PCGrad)."""
        tags = sorted(task_grad_buffer.keys())
        num_params = len(param_list)
        orig_grads = task_grad_buffer
        corrected = {tag: [g.clone() for g in orig_grads[tag]] for tag in tags}

        pg = self._pcgrad_process_group()
        world_size = torch.distributed.get_world_size(pg) if torch.distributed.is_initialized() else 1

        for ti in tags:
            for tj in tags:
                if ti == tj:
                    continue
                gi_list = orig_grads[ti]
                gj_list = orig_grads[tj]
                dots = torch.stack([torch.sum(gi * gj) for gi, gj in zip(gi_list, gj_list)])
                norms_sq = torch.stack([torch.sum(gj * gj) for gj in gj_list])
                if world_size > 1:
                    # one all_reduce per (task_i, task_j) pair for the whole model, instead of
                    # one per (task_i, task_j, param) triple (~300 tensors for a 1.5B model)
                    packed = torch.cat([dots, norms_sq])
                    torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM, group=pg)
                    dots, norms_sq = packed[:num_params], packed[num_params:]
                for k in range(num_params):
                    if dots[k] < 0:
                        coef = dots[k] / (norms_sq[k] + 1e-12)
                        corrected[ti][k] = corrected[ti][k] - coef * gj_list[k]

        return [sum(corrected[tag][k] for tag in tags) for k in range(num_params)]

    def _pcgrad_project_global(self, task_grad_buffer, param_list):
        """GGS: same algorithm as `_pcgrad_project_layerwise` but on a single flattened
        whole-model gradient vector per task instead of per-tensor."""
        tags = sorted(task_grad_buffer.keys())
        shapes = [g.shape for g in task_grad_buffer[tags[0]]]
        flat = {tag: torch.cat([g.reshape(-1) for g in task_grad_buffer[tag]]) for tag in tags}
        corrected = {tag: flat[tag].clone() for tag in tags}

        pg = self._pcgrad_process_group()
        world_size = torch.distributed.get_world_size(pg) if torch.distributed.is_initialized() else 1

        for ti in tags:
            for tj in tags:
                if ti == tj:
                    continue
                dot = torch.sum(flat[ti] * flat[tj]).reshape(1)
                norm_sq = torch.sum(flat[tj] * flat[tj]).reshape(1)
                if world_size > 1:
                    packed = torch.cat([dot, norm_sq])
                    torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM, group=pg)
                    dot, norm_sq = packed[0:1], packed[1:2]
                if dot.item() < 0:
                    coef = dot / (norm_sq + 1e-12)
                    corrected[ti] = corrected[ti] - coef * flat[tj]

        summed_flat = sum(corrected[tag] for tag in tags)
        final_grads = []
        offset = 0
        for shape in shapes:
            numel = shape.numel()
            final_grads.append(summed_flat[offset : offset + numel].reshape(shape))
            offset += numel
        return final_grads

    def _pcgrad_project(self, task_grad_buffer, param_list):
        if len(task_grad_buffer) <= 1:
            grads = next(iter(task_grad_buffer.values()), None)
            return grads if grads is not None else [torch.zeros_like(p) for p in param_list]
        if self.pcgrad_granularity == "global":
            return self._pcgrad_project_global(task_grad_buffer, param_list)
        return self._pcgrad_project_layerwise(task_grad_buffer, param_list)

    def _update_minibatch_pcgrad(self, micro_batches, temperature, metrics):
        """Task-wise gradient-surgery update for one PPO mini-batch.

        Since `balance_batch` can interleave task_tags within a micro-batch, each micro-batch
        is further split into per-task subgroups. Each subgroup does its own forward+backward,
        its gradient is captured into a per-task accumulator buffer (summed across however many
        subgroups of that task appear in the mini-batch), and the optimizer grad is zeroed
        before the next subgroup — this avoids cross-task gradient contamination via autograd.
        Losses are scaled by 1 / (number of subgroups for that task in this mini-batch) so every
        task contributes equally to its own accumulated gradient regardless of how many
        micro-batches/subgroups it happens to be split across.
        """
        device = torch.cuda.current_device()

        materialized = []
        for mb in micro_batches:
            assert isinstance(mb, DataProto), "pcgrad requires chunk-based DataProto micro-batches"
            tensor_part = mb.batch.to(device)
            non_tensor_part = mb.non_tensor_batch
            task_tags = np.asarray(non_tensor_part["task_tags"])
            materialized.append((tensor_part, non_tensor_part, task_tags))

        task_subgroup_counts = {}
        for _, _, task_tags in materialized:
            for tag in set(task_tags.tolist()):
                task_subgroup_counts[tag] = task_subgroup_counts.get(tag, 0) + 1

        self.actor_optimizer.zero_grad()
        param_list = [p for p in self.actor_module.parameters() if p.requires_grad]
        task_grad_buffer = {}

        for tensor_part, non_tensor_part, task_tags in materialized:
            for tag in sorted(set(task_tags.tolist())):
                idx = torch.as_tensor(np.where(task_tags == tag)[0], dtype=torch.long)
                sub_tensor = tensor_part[idx]
                sub_non_tensor = {k: v[idx.numpy()] for k, v in non_tensor_part.items() if k != "task_tags"}
                sub_data = {**sub_tensor, **sub_non_tensor}

                policy_loss, log_metrics = self._compute_actor_loss(sub_data, temperature, metrics)
                loss = policy_loss / task_subgroup_counts[tag]
                loss.backward()
                append_to_dict(metrics, log_metrics)

                grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in param_list]
                if tag not in task_grad_buffer:
                    task_grad_buffer[tag] = grads
                else:
                    task_grad_buffer[tag] = [buf + g for buf, g in zip(task_grad_buffer[tag], grads)]
                self.actor_optimizer.zero_grad()

        final_grads = self._pcgrad_project(task_grad_buffer, param_list)
        for p, g in zip(param_list, final_grads):
            p.grad = g

        grad_norm = self._optimizer_step()
        return {"actor/grad_norm": grad_norm.detach().item()}

    def probe_task_gradient_cosine(self, data: DataProto) -> dict:
        """Read-only probe: per-task_tag forward-backward over the WHOLE input batch (no PPO
        mini/micro-batch looping across multiple optimizer steps, no projection, no
        optimizer.step()), used to measure pairwise gradient cosine/norm between tasks on a
        single checkpoint (see write/gradient_conflict_verification_plan.md Stage 1). Leaves
        the model's .grad state zeroed on return, exactly like update_policy leaves it after a
        real update, so it is safe to call standalone without perturbing training state.
        """
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "response_mask"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        assert "task_tags" in data.non_tensor_batch.keys(), "probe_task_gradient_cosine requires task_tags in non_tensor_batch (see ctx_manager.get_lm_inputs)"

        num_micro_batches = data.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
        micro_batches = data.select(select_keys, ["task_tags"]).chunk(num_micro_batches)

        device = torch.cuda.current_device()
        materialized = []
        for mb in micro_batches:
            tensor_part = mb.batch.to(device)
            non_tensor_part = mb.non_tensor_batch
            task_tags = np.asarray(non_tensor_part["task_tags"])
            materialized.append((tensor_part, non_tensor_part, task_tags))

        task_subgroup_counts = {}
        for _, _, task_tags in materialized:
            for tag in set(task_tags.tolist()):
                task_subgroup_counts[tag] = task_subgroup_counts.get(tag, 0) + 1

        param_list = [p for p in self.actor_module.parameters() if p.requires_grad]
        for p in param_list:
            p.grad = None
        task_grad_buffer = {}
        metrics = {}

        for tensor_part, non_tensor_part, task_tags in materialized:
            for tag in sorted(set(task_tags.tolist())):
                idx = torch.as_tensor(np.where(task_tags == tag)[0], dtype=torch.long)
                sub_tensor = tensor_part[idx]
                sub_non_tensor = {k: v[idx.numpy()] for k, v in non_tensor_part.items() if k != "task_tags"}
                sub_data = {**sub_tensor, **sub_non_tensor}

                policy_loss, log_metrics = self._compute_actor_loss(sub_data, temperature, metrics)
                loss = policy_loss / task_subgroup_counts[tag]
                loss.backward()
                append_to_dict(metrics, log_metrics)

                grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in param_list]
                if tag not in task_grad_buffer:
                    task_grad_buffer[tag] = grads
                else:
                    task_grad_buffer[tag] = [buf + g for buf, g in zip(task_grad_buffer[tag], grads)]
                for p in param_list:
                    p.grad = None

        pg = self._pcgrad_process_group()
        world_size = torch.distributed.get_world_size(pg) if torch.distributed.is_initialized() else 1
        flat = {tag: torch.cat([g.reshape(-1).float() for g in grads]) for tag, grads in task_grad_buffer.items()}
        if world_size > 1:
            for tag in flat:
                torch.distributed.all_reduce(flat[tag], op=torch.distributed.ReduceOp.SUM, group=pg)

        tags = sorted(flat.keys())
        for tag in tags:
            metrics[f"probe/grad_norm/{tag}"] = flat[tag].norm().item()
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                gi, gj = flat[tags[i]], flat[tags[j]]
                cos = torch.dot(gi, gj) / (gi.norm() * gj.norm() + 1e-12)
                metrics[f"probe/cosine/{tags[i]}__{tags[j]}"] = cos.item()

        self.actor_module.zero_grad()
        return metrics

    # ------------------------------------------------------------------ #
    # Chain probe (write/gradient_conflict_verification_plan.md §5b/§5d)  #
    # ------------------------------------------------------------------ #
    # Unlike probe_task_gradient_cosine (which measures all K task gradients on a SINGLE
    # frozen theta), the chain probe performs the CST causal intervention on ONE checkpoint:
    # it REALLY applies a single-task update U_i(theta)=theta-eta*g_i (SGD, tau steps) and then
    # re-measures the pairwise gradient cosines, walking one CST cycle in the task order. This
    # directly measures the second-order term H_j g_i of Eq.(cst-second-order-basic) via finite
    # difference: g_j(theta - eta g_i) - g_j(theta) ~= -eta H_j g_i. See the plan for why this is
    # a stronger signal than the cross-checkpoint simultaneous-probe trend.

    def _probe_materialize(self, data: DataProto):
        """Split the probe batch into device-resident micro-batches tagged by task (shared by
        the chain-probe helpers below). Mirrors the setup in probe_task_gradient_cosine."""
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "response_mask"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        assert "task_tags" in data.non_tensor_batch.keys(), "probe_cst_chain requires task_tags in non_tensor_batch (see ctx_manager.get_lm_inputs)"

        num_micro_batches = data.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
        micro_batches = data.select(select_keys, ["task_tags"]).chunk(num_micro_batches)

        device = torch.cuda.current_device()
        materialized = []
        for mb in micro_batches:
            tensor_part = mb.batch.to(device)
            non_tensor_part = mb.non_tensor_batch
            task_tags = np.asarray(non_tensor_part["task_tags"])
            materialized.append((tensor_part, non_tensor_part, task_tags))

        task_subgroup_counts = {}
        for _, _, task_tags in materialized:
            for tag in set(task_tags.tolist()):
                task_subgroup_counts[tag] = task_subgroup_counts.get(tag, 0) + 1

        param_list = [p for p in self.actor_module.parameters() if p.requires_grad]
        return materialized, task_subgroup_counts, param_list

    @torch.no_grad()
    def _probe_refresh_old_logprobs(self, materialized, temperature):
        """Reset old_log_probs (and ref_log_prob if used) to the CURRENT theta so the importance
        ratio rho == 1 at every measurement point. This makes the per-task gradient the clean
        policy gradient nabla L(theta_t) on the fixed probe batch -- comparable across phases and
        immune to PPO clipping after a real update has moved theta (see plan §5d.2)."""
        for tensor_part, _, _ in materialized:
            _, log_prob = self._forward_micro_batch(micro_batch=tensor_part, temperature=temperature, calculate_entropy=False)
            tensor_part["old_log_probs"] = log_prob.detach()
            if self.config.use_kl_loss and "ref_log_prob" in tensor_part.keys():
                tensor_part["ref_log_prob"] = log_prob.detach()

    def _probe_task_grads(self, materialized, task_subgroup_counts, param_list, temperature, only_tag=None):
        """One read-only forward-backward per task over its whole sub-batch. Returns
        (task_grad_buffer{tag:[per-param local grad clone]}, task_loss{tag:float}). Leaves
        p.grad zeroed. If only_tag is given, restricts to that single task."""
        for p in param_list:
            p.grad = None
        task_grad_buffer = {}
        task_loss = {}
        metrics = {}
        for tensor_part, non_tensor_part, task_tags in materialized:
            for tag in sorted(set(task_tags.tolist())):
                if only_tag is not None and tag != only_tag:
                    continue
                idx = torch.as_tensor(np.where(task_tags == tag)[0], dtype=torch.long)
                sub_tensor = tensor_part[idx]
                sub_non_tensor = {k: v[idx.numpy()] for k, v in non_tensor_part.items() if k != "task_tags"}
                sub_data = {**sub_tensor, **sub_non_tensor}

                policy_loss, _ = self._compute_actor_loss(sub_data, temperature, metrics)
                loss = policy_loss / task_subgroup_counts[tag]
                loss.backward()
                task_loss[tag] = task_loss.get(tag, 0.0) + loss.item()

                grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in param_list]
                if tag not in task_grad_buffer:
                    task_grad_buffer[tag] = grads
                else:
                    task_grad_buffer[tag] = [buf + g for buf, g in zip(task_grad_buffer[tag], grads)]
                for p in param_list:
                    p.grad = None
        return task_grad_buffer, task_loss

    def _probe_flatten_allreduce(self, task_grad_buffer):
        """Flatten each task's sharded grad to a single fp32 vector and all_reduce across the
        FSDP process group so every rank sees the full gradient (needed for correct cosines)."""
        pg = self._pcgrad_process_group()
        world_size = torch.distributed.get_world_size(pg) if torch.distributed.is_initialized() else 1
        flat = {tag: torch.cat([g.reshape(-1).float() for g in grads]) for tag, grads in task_grad_buffer.items()}
        if world_size > 1:
            for tag in flat:
                torch.distributed.all_reduce(flat[tag], op=torch.distributed.ReduceOp.SUM, group=pg)
        return flat

    def _probe_pairwise(self, flat, task_loss):
        """Pairwise cosine + raw inner product + per-task grad norm/loss for one phase."""
        tags = sorted(flat.keys())
        out = {"grad_norm": {}, "loss": {}, "cosine": {}, "inner_product": {}}
        for tag in tags:
            out["grad_norm"][tag] = flat[tag].norm().item()
            out["loss"][tag] = task_loss.get(tag, None)
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                key = f"{tags[i]}__{tags[j]}"
                gi, gj = flat[tags[i]], flat[tags[j]]
                inner = torch.dot(gi, gj)
                out["inner_product"][key] = inner.item()
                out["cosine"][key] = (inner / (gi.norm() * gj.norm() + 1e-12)).item()
        return out

    def _probe_walk_cycle(self, materialized, task_subgroup_counts, param_list, temperature, order, eta, tau):
        """Walk one CST cycle from the CURRENT theta: baseline measure, then for each task in
        `order` apply tau real SGD steps U_i=theta-eta*g_i and re-measure all pairwise cosines.
        Returns the list of phase records. Mutates theta in place."""
        phases = []
        theta_norm = self._param_norm(param_list)

        # Phase 0: baseline at the current theta_0
        self._probe_refresh_old_logprobs(materialized, temperature)  # rho == 1
        buf, task_loss = self._probe_task_grads(materialized, task_subgroup_counts, param_list, temperature)
        flat = self._probe_flatten_allreduce(buf)
        rec = {"phase": "baseline"}
        rec.update(self._probe_pairwise(flat, task_loss))
        phases.append(rec)
        del buf, flat

        for updating_task in order:
            applied_sq = 0.0  # sum of ||eta*g||^2 over the tau steps (fp32, all-reduced)
            for _ in range(tau):
                self._probe_refresh_old_logprobs(materialized, temperature)  # clean PG for the update
                buf, _ = self._probe_task_grads(materialized, task_subgroup_counts, param_list, temperature, only_tag=updating_task)
                grads = buf[updating_task]
                gnorm = self._probe_flatten_allreduce({updating_task: grads})[updating_task].norm().item()
                applied_sq += (eta * gnorm) ** 2
                with torch.no_grad():
                    for p, g in zip(param_list, grads):
                        p.data.add_(g.to(p.dtype), alpha=-eta)
                del buf, grads
            theta_norm_after = self._param_norm(param_list)
            rec = {"phase": f"after_{updating_task}",
                   "applied_step_norm": applied_sq ** 0.5,
                   "rel_step_norm": (applied_sq ** 0.5) / (theta_norm + 1e-12),
                   "theta_moved_norm": abs(theta_norm_after - theta_norm)}
            self._probe_refresh_old_logprobs(materialized, temperature)
            buf, task_loss = self._probe_task_grads(materialized, task_subgroup_counts, param_list, temperature)
            flat = self._probe_flatten_allreduce(buf)
            rec.update(self._probe_pairwise(flat, task_loss))
            phases.append(rec)
            del buf, flat
            theta_norm = theta_norm_after
        return phases

    def probe_cst_chain(self, data: DataProto) -> dict:
        """CST causal-interference chain probe on ONE checkpoint (reads the checkpoint, but the
        weights ARE mutated in-place by real single-task SGD updates; safe because the probe run
        loads a fresh checkpoint and never saves). Walks one CST cycle in `cst_order`: baseline
        pairwise gradient cosines, then per task a real eta-step update U_i and re-measure -- so
        the caller sees whether updating task i shrinks the angle to the not-yet-updated tasks
        (the H_j g_i second-order term). `cst_probe_eta` may be a SINGLE float or a LIST: for a
        list, theta_0 is snapshotted once and the whole cycle is re-run per eta (same rollouts,
        same theta_0) -- a finite-difference step-size sweep to find where the signal clears the
        noise floor and confirm linearity. See write/gradient_conflict_verification_plan.md §5b/§5d."""
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        # eta is a FINITE-DIFFERENCE probe step, NOT the training LR: large enough that eta*g moves
        # theta above the gradient-noise floor, small enough to stay in the 2nd-order regime.
        eta_cfg = data.meta_info.get("cst_probe_eta", 0.05)
        etas = [float(e) for e in eta_cfg] if isinstance(eta_cfg, (list, tuple)) else [float(eta_cfg)]
        tau = int(data.meta_info.get("cst_probe_tau", 1))
        order = data.meta_info.get("cst_order", None)

        materialized, task_subgroup_counts, param_list = self._probe_materialize(data)
        tags = sorted(task_subgroup_counts.keys())
        if order is None:
            order = tags
        order = [t for t in order if t in tags]  # keep only tasks actually present
        param_dtype = str(param_list[0].dtype) if param_list else "none"

        if len(etas) == 1:
            phases = self._probe_walk_cycle(materialized, task_subgroup_counts, param_list, temperature, order, etas[0], tau)
            self.actor_module.zero_grad()
            return {"order": list(order), "eta": etas[0], "tau": tau, "param_dtype": param_dtype, "phases": phases}

        # eta sweep: snapshot theta_0 on CPU (fp32), restore before each eta so every sweep point
        # starts from the identical checkpoint on the identical probe batch.
        theta0 = [p.data.detach().to("cpu", copy=True) for p in param_list]
        sweep = {}
        for eta in etas:
            with torch.no_grad():
                for p, s in zip(param_list, theta0):
                    p.data.copy_(s.to(p.device))
            sweep[repr(eta)] = self._probe_walk_cycle(materialized, task_subgroup_counts, param_list, temperature, order, eta, tau)
        del theta0
        self.actor_module.zero_grad()
        return {"order": list(order), "etas": etas, "tau": tau, "param_dtype": param_dtype, "sweep": sweep}

    def _param_norm(self, param_list):
        """L2 norm of the full (sharded, all-reduced) parameter vector in fp32, computed
        incrementally to avoid a large concatenation. Used only for the step-size diagnostic."""
        sq = torch.zeros((), dtype=torch.float64, device=torch.cuda.current_device())
        for p in param_list:
            sq += p.data.float().pow(2).sum().double()
        pg = self._pcgrad_process_group()
        if torch.distributed.is_initialized() and torch.distributed.get_world_size(pg) > 1:
            torch.distributed.all_reduce(sq, op=torch.distributed.ReduceOp.SUM, group=pg)
        return sq.sqrt().item()

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages", "response_mask"]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if self.pcgrad_enable:
            assert not self.config.use_dynamic_bsz, "pcgrad is not compatible with use_dynamic_bsz"
            assert "task_tags" in data.non_tensor_batch.keys(), "pcgrad.enable=True requires task_tags in non_tensor_batch (see ctx_manager.get_lm_inputs)"

        # PCGrad needs non_tensor_batch (task_tags) to survive mini/micro-batch splitting, so it
        # reuses the multi-modal path's chunk-based DataProto.select(...).chunk(...) splitting.
        # The legacy TensorDict.split() path below is left completely untouched when disabled.
        use_chunk_split = has_multi_modal_inputs or self.pcgrad_enable
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.pcgrad_enable:
            non_tensor_select_keys.append("task_tags")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if use_chunk_split:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            batch = data.select(batch_keys=select_keys).batch
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if use_chunk_split:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                if self.pcgrad_enable:
                    data = self._update_minibatch_pcgrad(micro_batches, temperature, metrics)
                else:
                    self.actor_optimizer.zero_grad()

                    for data in micro_batches:
                        # Support all hardwares
                        if isinstance(data, DataProto):
                            data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                        else:
                            data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload

                        policy_loss, log_metrics = self._compute_actor_loss(data, temperature, metrics)

                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                        else:
                            loss = policy_loss / self.gradient_accumulation
                        loss.backward()
                        append_to_dict(metrics, log_metrics)

                    grad_norm = self._optimizer_step()
                    data = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
