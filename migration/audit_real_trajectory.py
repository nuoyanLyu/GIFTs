#!/usr/bin/env python3
"""CPU-only audit of saved single-turn Qwen3 non-thinking smoke trajectories.

Reads source through AST rather than importing ragen.env or API credentials.
The frozen pre-migration parser supplies the reference protocol; token packing,
source-role masks, positions and group rewards are recomputed independently.
Optimizer/update completion is a separate check.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parser_node(path):
    tree = ast.parse(Path(path).read_text())
    context = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ContextManager')
    node = deepcopy(next(node for node in context.body if isinstance(node, ast.FunctionDef) and node.name == '_parse_response'))
    node.returns = None
    for child in ast.walk(node):
        if isinstance(child, ast.arg):
            child.annotation = None
    return node


def frozen_parser(legacy_path, candidate_path, config):
    old, current = parser_node(legacy_path), parser_node(candidate_path)
    if ast.dump(old, include_attributes=False) != ast.dump(current, include_attributes=False):
        raise ValueError('The production parser differs from the frozen legacy parser; do not silently change scoring.')
    namespace = {'re': re}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[old], type_ignores=[])), str(legacy_path), 'exec'), namespace)
    context = SimpleNamespace(
        config=SimpleNamespace(agent_proxy=SimpleNamespace(enable_think=False,
            max_actions_per_turn=config['agent_proxy']['max_actions_per_turn'])),
        action_sep=config['agent_proxy']['action_sep'],
        special_token_list=['<think>', '</think>', '<answer>', '</answer>', '<|im_start|>', '<|im_end|>'],
    )
    return lambda raw: namespace['_parse_response'](context, raw)


def expected_rewards(states, grouping, method):
    scores = [sum(float(item.get('reward', 0)) for item in state['history'])
              + float(state.get('penalty', 0)) for state in states]
    groups = {}
    for index, state in enumerate(states):
        if grouping == 'state': key = state['group_id']
        elif grouping == 'inductive': key = state['tag']
        elif grouping == 'batch': key = 1
        else: raise ValueError(f'Unsupported reward grouping: {grouping}')
        groups.setdefault(key, []).append(index)
    result = scores.copy()
    if len(groups) < len(states):
        for indices in groups.values():
            values = [scores[index] for index in indices]
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0
            for index in indices:
                if method == 'identity': value = scores[index]
                elif method == 'mean': value = scores[index] - mean
                elif method in ('mean_std', 'asym_clip'):
                    value = (scores[index] - mean) / (std + 1e-6) if std > 1e-6 else 0.0
                    if method == 'asym_clip': value = min(3, max(-1, value))
                else: raise ValueError(f'Unsupported reward normalization: {method}')
                result[index] = value
    return result


def audit_file(path, tokenizer, parse, config):
    saved = json.loads(path.read_text())
    states, turns, packed = saved['states'], saved['turns'], saved['packed']
    if len(turns) != 1 or turns[0]['turn'] != 0:
        raise ValueError('This audit is scoped to the current single-turn smoke; multi-turn traces require another audit.')
    mode = 'val' if saved['validation'] else 'train'
    env_cfg = config['es_manager'][mode]
    expected_rows = env_cfg['env_groups'] * env_cfg['group_size']
    if len(states) != expected_rows or len({state['env_id'] for state in states}) != len(states):
        raise ValueError(f'Trajectory count/IDs differ from the resolved {mode} config.')
    for name in ('input_ids', 'attention_mask', 'position_ids', 'responses', 'loss_mask', 'rm_scores', 'original_rm_scores'):
        if len(packed[name]) != len(states):
            raise ValueError(f'Packed {name} row count mismatch.')
    turn = turns[0]
    if any(len(turn[name]) != len(states) for name in ('prompt_ids', 'generated_ids', 'parsed_environment_inputs')):
        raise ValueError('The saved generation fields have inconsistent row counts.')
    rn = config['agent_proxy']['reward_normalization']
    rewards = expected_rewards(states, rn['grouping'], rn['method'])
    rows = {state['env_id']: index for index, state in enumerate(states)}
    start_tokens = tokenizer.encode('<|im_start|>', add_special_tokens=False)
    if len(start_tokens) != 1: raise ValueError('The tokenizer no longer has the audited ChatML start token.')
    forced_text = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
    forced_ids = tokenizer.encode(forced_text, add_special_tokens=False)
    report = {'file': str(path), 'sha256': digest(path), 'validation_rollout': saved['validation'],
              'turn_count': 1, 'trajectory_count': len(states), 'rows': []}
    seen = set()
    for prompt, generated, parsed in zip(turn['prompt_ids'], turn['generated_ids'], turn['parsed_environment_inputs']):
        env_id = parsed['env_id']
        if env_id in seen or env_id not in rows: raise ValueError('Generated environment ID is duplicated or missing.')
        seen.add(env_id)
        row = rows[env_id]
        prompt_ids = list(prompt)
        while prompt_ids and prompt_ids[0] == tokenizer.pad_token_id: prompt_ids.pop(0)
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        raw = tokenizer.decode(generated, skip_special_tokens=True)
        canonical, actions = parse(raw)
        ids, attention = packed['input_ids'][row], packed['attention_mask'][row]
        response, mask = packed['responses'][row], packed['loss_mask'][row]
        if len(attention) != len(ids) or len(packed['position_ids'][row]) != len(ids):
            raise ValueError('Input/attention/position lengths differ.')
        if any(len(packed[key][row]) != len(ids) - 1 for key in ('responses', 'loss_mask', 'rm_scores', 'original_rm_scores')):
            raise ValueError('The next-token tensors are not input length minus one.')
        if any(value not in (0, 1, False, True) for value in attention): raise ValueError('Invalid attention mask.')
        valid_ids = [value for value, active in zip(ids, attention) if active]
        expected_ids = tokenizer.encode(prompt_text + canonical + '<|im_end|>\n', add_special_tokens=False)
        roles, role = [], 0
        for token in ids:
            if token == start_tokens[0]: role += 1
            roles.append(role)
        expected_mask = [(role > 1 and role % 2 == 1) for role in roles[:-1]]
        positions, active_count = [], 0
        for active in attention:
            active_count += int(active)
            positions.append(max(active_count - 1, 0))
        system_start, user_start, end = '<|im_start|>system\n', '<|im_start|>user\n', '<|im_end|>\n'
        if not prompt_text.startswith(system_start): raise ValueError('The prompt has no audited system prefix.')
        system, boundary, remainder = prompt_text[len(system_start):].partition(end)
        if not boundary or not remainder.startswith(user_start): raise ValueError('Invalid system/user ChatML boundary.')
        user, boundary, tail = remainder[len(user_start):].partition(end)
        if not boundary or tail != forced_text: raise ValueError('The native non-thinking assistant prefix differs.')
        native = tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                                               add_generation_prompt=True, tokenize=True, enable_thinking=False)
        score_row = packed['rm_scores'][row]
        item = {
            'env_id': env_id, 'native_false_prompt_exact': prompt_ids == native,
            'old_step_by_step_instruction_absent': "Let's think step by step and always output:" not in prompt_text,
            'new_always_output_instruction_present': 'Always output:' in prompt_text,
            'native_false_suffix_exact': prompt_ids[-len(forced_ids):] == forced_ids,
            'raw_decode_exact': raw == parsed['llm_raw_response'],
            'canonical_and_actions_exact': canonical == parsed['llm_response'] and actions == parsed['actions'],
            'packed_matches_prompt_canonical_and_native_end': valid_ids == expected_ids,
            'responses_equal_input_ids_shift1': response == ids[1:],
            'loss_mask_equals_legacy_source_role_rule': mask == expected_mask,
            'position_ids_equal_attention_cumsum_minus1': packed['position_ids'][row] == positions,
            'reward_only_last_target': all(value == 0 for value in score_row[:-1]),
            'normalized_reward_matches_independent_group_calculation': abs(score_row[-1] - rewards[row]) < 1e-6,
            'legacy_original_rm_scores_alias_matches': packed['original_rm_scores'][row] == score_row,
            'supervised_environment_target_tokens': sum(bool(m) and (role <= 2 or role % 2 == 0) for m, role in zip(mask, roles[1:])),
            'supervised_token_count': sum(mask), 'input_token_count': sum(attention),
            'actions': actions, 'raw_equals_canonical': raw == canonical,
            'raw_response': raw, 'canonical_response': canonical,
            'raw_environment_reward': sum(float(item.get('reward', 0)) for item in states[row]['history']),
            'format_penalty': states[row]['penalty'], 'normalized_reward': score_row[-1],
        }
        report['rows'].append(item)
    checks = ('native_false_prompt_exact', 'old_step_by_step_instruction_absent', 'new_always_output_instruction_present',
              'native_false_suffix_exact', 'raw_decode_exact', 'canonical_and_actions_exact',
              'packed_matches_prompt_canonical_and_native_end', 'responses_equal_input_ids_shift1',
              'loss_mask_equals_legacy_source_role_rule', 'position_ids_equal_attention_cumsum_minus1',
              'reward_only_last_target', 'normalized_reward_matches_independent_group_calculation',
              'legacy_original_rm_scores_alias_matches')
    report['all_protocol_checks_pass'] = all(all(row[name] is True for name in checks) for row in report['rows'])
    report['valid_action_count'] = sum(len(row['actions']) for row in report['rows'])
    report['nonzero_normalized_reward_count'] = sum(row['normalized_reward'] != 0 for row in report['rows'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('/data1/lvnuoyan/ragen2_runs/conda_validation_20261117'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--resolved-config', type=Path, help='Otherwise inferred from the run CPU-evidence gate.')
    parser.add_argument('--model-path', type=Path, help='Otherwise use the resolved launch config model_path.')
    parser.add_argument('--legacy-context', type=Path, help='Otherwise use PROJECT_ROOT/migration/legacy_ctx_manager.py.')
    args = parser.parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.project_root = args.project_root.expanduser().resolve()
    output = (args.output_dir or args.run_dir / 'trajectory_review').expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {'passed': False, 'run_dir': str(args.run_dir), 'python': sys.executable, 'prefix': sys.prefix}
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['HF_HUB_OFFLINE'] = '1'
        sys.dont_write_bytecode = True
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        import torch

        status = json.loads((args.run_dir / 'status.json').read_text())
        config_path = args.resolved_config
        if config_path is None:
            cpu_gate = Path(status['gates']['cpu']['path'])
            if not cpu_gate.is_absolute(): cpu_gate = args.project_root / cpu_gate
            config_path = cpu_gate.parent / 'resolved_launch_config.yaml'
        config_path = config_path.expanduser().resolve()
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        if (config['data']['apply_chat_template_kwargs'].get('enable_thinking') is not False
                or config['agent_proxy']['enable_think'] is not False
                or config['agent_proxy']['use_turn_scores'] is not False
                or config['enable_response_mask'] is not True):
            raise ValueError('This audit requires the confirmed non-thinking, response-mask, outcome-score protocol.')
        candidate = args.project_root / 'ragen/llm_agent/ctx_manager.py'
        legacy = args.legacy_context or args.project_root / 'migration/legacy_ctx_manager.py'
        parse = frozen_parser(legacy, candidate, config)
        model_path = args.model_path or Path(config['model_path'])
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        paths = sorted((args.run_dir / 'trajectory_audit').glob('rollout_*.json'))
        if not paths: raise ValueError('No saved rollout files were found; an empty audit cannot pass.')
        reports = []
        for path in paths:
            report = audit_file(path, tokenizer, parse, config)
            (output / f'audit_{path.stem}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            reports.append(report)
        training = [report for report in reports if not report['validation_rollout']]
        expected = int(config['trainer']['total_training_steps'])
        complete = len(training) == expected
        summary.update({
            'passed': complete and all(report['all_protocol_checks_pass'] for report in reports),
            'complete_training_rollout_count': complete, 'expected_training_rollouts': expected,
            'actual_training_rollouts': len(training), 'trajectory_count': sum(report['trajectory_count'] for report in training),
            'valid_actions_by_training_rollout': [report['valid_action_count'] for report in training],
            'nonzero_normalized_rewards_by_training_rollout': [report['nonzero_normalized_reward_count'] for report in training],
            'all_protocol_checks_pass': all(report['all_protocol_checks_pass'] for report in reports),
            'resolved_config': str(config_path), 'resolved_config_sha256': digest(config_path),
            'legacy_parser_source': str(legacy), 'legacy_parser_sha256': digest(legacy),
            'candidate_source': str(candidate), 'candidate_source_sha256': digest(candidate),
            'parser_ast_equal_to_legacy': True, 'model_path': str(model_path),
            'cuda_initialized': torch.cuda.is_initialized(),
            'training_exit_code': status.get('training', {}).get('exit_code'),
            'limits': ['Single-turn saved trajectories only; optimizer/weight-update success is checked separately.',
                       'The legacy mask retains assistant headers and forced empty-think template tokens.',
                       'Original environment reward is read from states/history; legacy original_rm_scores aliases normalized scores.'],
        })
        if summary['cuda_initialized']: raise ValueError('The CPU audit initialized CUDA unexpectedly.')
    except Exception as exc:
        summary['passed'] = False
        summary['error_type'], summary['error'] = type(exc).__name__, str(exc)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
