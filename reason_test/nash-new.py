try:
    from reason_test.model_protocol import ModelProtocol, add_protocol_args
    from reason_test.qwen3_eval import Qwen3Eval, add_qwen3_args, is_qwen3
except ModuleNotFoundError:
    from model_protocol import ModelProtocol, add_protocol_args
    from qwen3_eval import Qwen3Eval, add_qwen3_args, is_qwen3

# NashEnv evaluation script using vLLM-served model
import sys
import os
from ragen.env.nash_new.env import NashNew
from ragen.env.nash_new.config import NashNewConfig
import json
import re
import time
from tqdm import trange
import random
from collections import Counter
import argparse
from pathlib import Path

from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer

# Setup
root_path = '/data1/lvnuoyan/llm_model/gift'
test_round = 100

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="nash-new")
parser.add_argument("--model_name", type=str, default='Qwen2.5-1.5B-Instruct')
parser.add_argument("--device", type=str, default='0')
add_protocol_args(parser)
add_qwen3_args(parser)
args = parser.parse_args()

model_path = args.model_path
model_name = args.model_name
device = args.device
resolved_model_dir = args.model_dir or f"{root_path}/{model_path}/{model_name}"
protocol = ModelProtocol.from_model_path(resolved_model_dir, no_think=args.no_think)
tokenizer = hf_tokenizer(resolved_model_dir)
qwen3_protocol = Qwen3Eval(tokenizer, args) if is_qwen3(resolved_model_dir, args) else None

def load_llm():
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    if qwen3_protocol:
        llm = LLM(resolved_model_dir, **qwen3_protocol.llm_kwargs(max_model_len=6000))
        return llm, qwen3_protocol.sampling_params(max_tokens=600)
    model = resolved_model_dir
    # model = f"{root_path}/{model_name}"
    # ro_config = config.actor_rollout_ref.rollout
    llm = LLM(
		model,
        max_model_len=6000,
	)
    print("LLM initialized")
    sampling_params = SamplingParams(
		max_tokens=600, # ro_config.response_length,
		temperature=0.5,  # ro_config.val_kwargs.temperature,
	)
    return llm, sampling_params

def reformat_prompt(prompt0):
    if qwen3_protocol:
        return qwen3_protocol.trajectory([{"state": prompt0, "actions_left": 1}], env_type="nash_new")
    task_prompt = prompt0
    if protocol.answer_only:
        task_prompt += '\nPut only the action required by the game inside the answer block. Do not add reasoning or commentary.'
    return protocol.render_prompt(
        tokenizer, task_prompt,
        legacy_suffix="Let's think step by step and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens).",
        system_prompt="You're a helpful assistant. ",
    )

def _legacy_extract_action(output):
    """Extract action from <answer>...</answer> tags"""
    # First try to match the full pattern
    pattern = r'<answer>\s*(.*)\s*</answer>'
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    # # Fallback: find any <answer> tag and extract first 1 or 2
    # answer_pattern = r'<answer>(.*?)</answer>'
    # match = re.search(answer_pattern, output, re.DOTALL)
    # if match:
    #     answer_content = match.group(1)
    #     # Find first occurrence of 1 or 2
    #     digit_match = re.search(r'[12]', answer_content)
    #     if digit_match:
    #         return digit_match.group(0)
    
    return None


def extract_action(output):
    if qwen3_protocol:
        actions = qwen3_protocol.actions(output, env_type="nash_new")
        return actions[0] if actions else None
    action = protocol.extract_answer(output, legacy_extractor=_legacy_extract_action)
    # NashNew compares exact action labels; preserve the legacy whitespace rule.
    return action.strip() if protocol.answer_only and action is not None else action

if __name__ == '__main__':
    env = NashNew()
    info_list = []
    llm, sampling_params = load_llm()
    rounds = min(test_round, args.max_samples) if args.max_samples is not None else test_round
    rng = random.Random(args.seed) if qwen3_protocol else random
    for t in trange(rounds):
        seed = rng.randint(1, 1000)
        prompt = env.reset(seed=seed)
        # Format prompt with instruction suffix
        formatted_prompt = reformat_prompt(prompt)
        outputs = llm.generate([formatted_prompt], sampling_params)
        completion = outputs[0].outputs[0]
        output = qwen3_protocol.response_text(completion) if qwen3_protocol else completion.text
        if qwen3_protocol:
            qwen3_protocol.record(completion, prompt=formatted_prompt, game_index=t, game_seed=seed)
        # Extract action from output
        action_str = extract_action(output)
        if action_str is None:
            info_list.append('invalid-format')
            print('invalid-format')
            continue
        # Keep action as string for env.step() (导师版本需要字符串)
        action = action_str
        # Step environment
        prompt, reward, done, info = env.step(action)
        # Record result
        if info.get('success', False):
            info_list.append('success')
            status = 'success'
            print('success')
        else:
            if not info.get('action_is_valid', True):
                info_list.append('invalid-action')
                status = 'invalid-action'
                print('invalid-action')
            else:
                info_list.append('fail')
                status = 'fail'
                print('fail')
    
    # Statistics
    counter = Counter(info_list)
    total = len(info_list)
    
    # Print summary
    print("\n" + "="*20)
    print(f"NashEnv Test Results (n={total})")
    print("="*20)
    for key, value in counter.items():
        print(f"{key}: {value / total:.2%}")
    
    # success_count = counter.get('success', 0)
    # print(f"\nSuccess rate: {success_count / total:.2%}")
    print("="*20)
    
    # Save results to log file
    output_dir = qwen3_protocol.output_dir if qwen3_protocol else Path("reason_test")
    log_path = output_dir / "nashenv-log.txt"
    with log_path.open("a") as f:
        f.write(f"\n=== Model: {args.model_name} ===\n")
        f.write("NashEnv test results:\n")
        for key, value in counter.items():
            f.write(f"{key}: {value / total:.2%}\n")
        # f.write(f"Success rate: {success_count / total:.2%}\n")
    print(f"\nResults saved to {log_path}")
