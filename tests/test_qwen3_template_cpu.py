#!/usr/bin/env python3
"""CPU-only, real-tokenizer audit of RAGEN2's non-thinking template adapter.

Run in the factory environment; no model weights, generation, or CUDA are used.
The source methods are loaded through AST to avoid importing unrelated game/API
dependencies. Every tested method body is the actual production source body.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import re
from types import SimpleNamespace


def source_methods(path, full=False):
    tree = ast.parse(Path(path).read_text())
    wanted = {"_render_chat_template", "_parse_response"}
    if full:
        wanted.update({"get_lm_inputs", "_normalize_score_tensor"})
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {
                "FINAL_ANSWER_WITH_PREFIX", "FINAL_ANSWER_INSTRUCTION", "FINAL_ANSWER_PATTERN"
            } for target in node.targets
        ):
            selected.append(node)
        if full and isinstance(node, ast.FunctionDef) and node.name in {
            "get_special_tokens", "get_masks_and_scores"
        }:
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "ContextManager":
            node.bases = []
            node.body = [item for item in node.body
                         if isinstance(item, ast.FunctionDef) and item.name in wanted]
            selected.append(node)
    tree.body = selected
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.returns = None
        if isinstance(node, ast.arg):
            node.annotation = None
    namespace = {"re": re, "zip_longest": zip_longest}
    if full:
        import numpy as np
        import torch
        from tensordict import TensorDict
        from verl import DataProto
        namespace.update(np=np, torch=torch, TensorDict=TensorDict, DataProto=DataProto)
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace["ContextManager"]


def manager(cls, tokenizer, native_false=True, format_think=False):
    obj = cls.__new__(cls)
    obj.tokenizer = tokenizer
    obj.config = SimpleNamespace(
        data={"apply_chat_template_kwargs": {"enable_thinking": False}} if native_false else {},
        agent_proxy=SimpleNamespace(enable_think=format_think, max_actions_per_turn=5,
                                    max_context_window=-1, use_turn_scores=False,
                                    reward_normalization=SimpleNamespace(grouping="state", method="mean_std")),
        enable_response_mask=True,
    )
    obj.action_sep = "||"
    obj.special_token_list = ["<think>", "</think>", "<answer>", "</answer>",
                              "<|im_start|>", "<|im_end|>"]
    obj.prefix_lookup = {0: "Choose a legal move.", 1: "Choose a legal move."}
    obj.env_config_lookup = {0: {"max_tokens": 200}, 1: {"max_tokens": 200}}
    obj.env_nums = {"Synthetic": 2}
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=Path(__file__).resolve().parents[1] / "ragen/llm_agent/ctx_manager.py")
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--qwen3", default="/data1/model/Qwen3-4B")
    parser.add_argument("--qwen25", default="/data1/model/Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    import torch

    candidate = source_methods(args.candidate, full=True)
    legacy = source_methods(args.legacy, full=True)
    q3 = AutoTokenizer.from_pretrained(args.qwen3, local_files_only=True)
    q25 = AutoTokenizer.from_pretrained(args.qwen25, local_files_only=True)
    report = {"device": "CPU", "source_sha256": {}, "tokenizers": {}, "checks": {}}
    for label, path in (("candidate", args.candidate), ("legacy", args.legacy)):
        report["source_sha256"][label] = hashlib.sha256(path.read_bytes()).hexdigest()
    for label, path in (("qwen3", args.qwen3), ("qwen25", args.qwen25)):
        report["tokenizers"][label] = {"path": str(path), "config_sha256": hashlib.sha256(
            (Path(path) / "tokenizer_config.json").read_bytes()).hexdigest()}

    messages = [
        {"role": "system", "content": "You're a helpful assistant. "},
        {"role": "user", "content": "State: first. Use <answer>...</answer>."},
        {"role": "assistant", "content": "<answer>1,1</answer>"},
        {"role": "user", "content": "Reward: 0. State: second."},
        {"role": "assistant", "content": "<answer>2,2</answer>"},
    ]
    q3_manager = manager(candidate, q3)
    encode = lambda text: q3.encode(text, add_special_tokens=False)
    q3_cases = []
    for size in (3, 5):
        turns = messages[:size]
        update = q3_manager._render_chat_template(turns, False)
        update_ids = encode(update)
        native = q3.apply_chat_template(turns, add_generation_prompt=False, tokenize=False, enable_thinking=False)
        per_turn = []
        for index in range(2, size, 2):
            prefix = q3_manager._render_chat_template(turns[:index], True)
            expected = prefix + turns[index]["content"] + "<|im_end|>\n"
            prefix_ids, expected_ids = encode(prefix), encode(expected)
            assert update.startswith(expected)
            assert update_ids[:len(prefix_ids)] == prefix_ids
            assert update_ids[:len(expected_ids)] == expected_ids
            per_turn.append({"turn": index // 2, "prefix_token_count": len(prefix_ids),
                             "canonical_and_end_token_count": len(expected_ids) - len(prefix_ids),
                             "conditioning_prefix_equal": True})
        if size == 3:
            assert update == native
        else:
            first_native_prompt = q3.apply_chat_template(turns[:2], add_generation_prompt=True,
                                                        tokenize=False, enable_thinking=False)
            assert not native.startswith(first_native_prompt + turns[2]["content"])
            assert native != update
        q3_cases.append({"turn_count": size // 2, "native_whole_trajectory_equal": update == native,
                         "stable_update_text": update, "stock_native_update_text": native,
                         "update_token_ids": update_ids, "per_turn": per_turn})
    report["checks"]["qwen3_single_and_two_turn_conditioning"] = q3_cases
    report["checks"]["stock_native_multiturn_prefix_drift_detected"] = True
    report["checks"]["qwen3_prefix_is_actual_native_false"] = q3_manager._render_chat_template(messages[:2], True) == q3.apply_chat_template(
        messages[:2], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    assert report["checks"]["qwen3_prefix_is_actual_native_false"]

    # The mode must be explicit; RAGEN's output-format switch is independent.
    for mode_name, obj in (("missing_native_false", manager(candidate, q3, native_false=False)),
                           ("legacy_format_think_true", manager(candidate, q3, format_think=True))):
        try:
            obj._render_chat_template(messages[:2], True)
        except ValueError as error:
            report["checks"][mode_name + "_rejected"] = str(error)
        else:
            raise AssertionError(mode_name + " must fail closed")
    modes_seen = []
    original_apply = q3.apply_chat_template
    def recording_apply(*items, **kwargs):
        modes_seen.append(kwargs.get("enable_thinking", "ABSENT"))
        return original_apply(*items, **kwargs)
    q3.apply_chat_template = recording_apply
    try:
        manager(candidate, q3)._render_chat_template(messages, False)
    finally:
        q3.apply_chat_template = original_apply
    assert modes_seen and all(mode is False for mode in modes_seen)
    report["checks"]["every_qwen3_native_call_explicit_false"] = modes_seen

    # Qwen2.5 keeps the original native call and the full old packing/mask/reward path.
    q25_manager = manager(candidate, q25)
    for size in (2, 3, 4, 5):
        generation = size % 2 == 0
        actual = q25_manager._render_chat_template(messages[:size], generation)
        expected = q25.apply_chat_template(messages[:size], add_generation_prompt=generation, tokenize=False)
        assert actual == expected
        assert q25.encode(actual) == q25.encode(expected)
    histories = [
        {"env_id": 0, "group_id": 0, "tag": "Synthetic", "penalty": -0.1, "metrics": {}, "history": [
            {"state": "first state", "actions_left": 2, "llm_response": "<answer>1,1</answer>", "reward": 0.2},
            {"state": "second state", "actions_left": 1, "llm_response": "<answer>2,2</answer>", "reward": 0.3},
            {"state": "final state", "actions_left": 0}]},
        {"env_id": 1, "group_id": 0, "tag": "Synthetic", "penalty": 0.0, "metrics": {}, "history": [
            {"state": "short state", "actions_left": 1, "llm_response": "<answer>0</answer>", "reward": 1.0},
            {"state": "final", "actions_left": 0}]},
    ]
    for update in (False, True):
        old = manager(legacy, q25).get_lm_inputs(deepcopy(histories), prepare_for_update=update)
        new = manager(candidate, q25).get_lm_inputs(deepcopy(histories), prepare_for_update=update)
        assert set(old.batch.keys()) == set(new.batch.keys())
        for key in old.batch.keys():
            assert torch.equal(old.batch[key], new.batch[key]), (update, key)
        for key in old.non_tensor_batch.keys():
            assert old.non_tensor_batch[key].tolist() == new.non_tensor_batch[key].tolist(), key
        assert old.meta_info == new.meta_info
    report["checks"]["qwen25_exact_text_tokens_packing_masks_reward_regression"] = True

    # The user approved precisely this Qwen3 non-thinking instruction change.
    # Compare message bodies independently of Qwen3's historical template rewrite.
    for update in (False, True):
        old = manager(legacy, q3).get_lm_inputs(deepcopy(histories), prepare_for_update=update)
        new = manager(candidate, q3).get_lm_inputs(deepcopy(histories), prepare_for_update=update)
        expected = old.non_tensor_batch['messages_list'].tolist()
        for row in expected:
            for message in row:
                if message['role'] == 'user':
                    message['content'] = message['content'].replace(
                        "Let's think step by step and always output:", 'Always output:')
        assert new.non_tensor_batch['messages_list'].tolist() == expected
    report['checks']['qwen3_only_user_confirmed_step_by_step_removal'] = True

    # Preserve the pre-existing canonicalization, including malformed-response behavior.
    response_cases = ["<answer>1,1</answer>", "extra <answer> 1,1 || 2,2 </answer>",
                      "<answer>0</answer> trailing", "missing tags", "<think>text</think><answer>0</answer>",
                      "<answer>1||2||3||4||5||6</answer>"]
    for thinking in (False, True):
        old, new = manager(legacy, q25, format_think=thinking), manager(candidate, q25, format_think=thinking)
        for raw in response_cases:
            assert old._parse_response(raw) == new._parse_response(raw)
    report["checks"]["legacy_raw_to_canonical_and_actions_unchanged"] = True
    report["limits"] = ["Synthetic protocol examples; no GPU/model generation/optimizer step.",
                         "RAGEN still trains normalized llm_response, not untouched raw output.",
                         "The legacy assistant mask formula is retained; native forced template tokens remain within assistant turns."]
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "output": str(args.output), "check_names": list(report["checks"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
