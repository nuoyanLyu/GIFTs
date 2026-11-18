try:
    from reason_test.model_protocol import ModelProtocol, add_protocol_args
    from reason_test.qwen3_eval import Qwen3Eval, add_qwen3_args, is_qwen3
except ModuleNotFoundError:
    from model_protocol import ModelProtocol, add_protocol_args
    from qwen3_eval import Qwen3Eval, add_qwen3_args, is_qwen3

# 参考tictactoe测试代码生成undercover测试代码
from ragen.env.undercover.config import UndercoverEnvConfig
from ragen.env.undercover.env import UndercoverEnv
# 根据infer得到的结果，运行函数提取模型生成的信息
import json
import re
import time
import datasets
from tqdm import trange
import random
import os, logging
from collections import Counter
from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer
import argparse
from pathlib import Path

# root_path = '/root/autodl-tmp'  # '/data1/lvnuoyan' 
root_path = '/data1/lvnuoyan/llm_model/gift'
test_round = 100
config = UndercoverEnvConfig(
    max_env_try=1,  # 修改最大尝试次数
    player_info=[
        # 固定对手gemini-2.5-flash
        {'model_name': 'gemini-2.5-flash-nothinking'},
        {'model_name': 'gemini-2.5-flash-nothinking'},
        {'model_name': 'gemini-2.5-flash-nothinking'}
    ]
)
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="undercover")
parser.add_argument("--model_name", type=str, default="Qwen2.5-1.5B-Instruct")
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
        llm = LLM(resolved_model_dir, **qwen3_protocol.llm_kwargs(max_model_len=8000))
        return llm, qwen3_protocol.sampling_params(max_tokens=600)
    model = resolved_model_dir
    # model = f"{root_path}/{model_name}"
    # ro_config = config.actor_rollout_ref.rollout
    llm = LLM(
		model,
        max_model_len=8000,
	)
    print("LLM initialized")
    sampling_params = SamplingParams(
		max_tokens=600, # ro_config.response_length,
		temperature=0.5,  # ro_config.val_kwargs.temperature,
	)
    return llm, sampling_params


def reformat_prompt(prompt0):
    task_prompt = prompt0
    if protocol.answer_only:
        task_prompt += '\nPut only the action required by the game inside the answer block. Do not add reasoning or commentary.'
    return protocol.render_prompt(
        tokenizer, task_prompt,
        legacy_suffix="Let's think step by step and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens).",
        system_prompt="You're a helpful assistant. ",
    )

def _legacy_extract_action(output):
    pattern = r'.*<think>(.*?)</think>\s*<answer>(.*?)</answer>$'
    match = re.match(pattern, output, re.DOTALL)
    return match.group(2) if match else None


def extract_action(output):
    return protocol.extract_answer(output, legacy_extractor=_legacy_extract_action)


def classify_action(prompt, reward, done):
    if "Invalid action" in prompt:
        return "trainer-invalid-output"
    if not done:
        return None
    if "env invalid output" in prompt:
        return "env_player-invalid-output"
    return "success" if reward else "fail"


if __name__ == '__main__':
    llm, sampling_params = load_llm()
    env = UndercoverEnv(config=config)
    info_list = []
    rounds = min(test_round, args.max_samples) if args.max_samples is not None else test_round
    rng = random.Random(args.seed) if qwen3_protocol else random
    for t in trange(rounds):
        seed = rng.randint(1, 1000)
        env.reset(seed=seed)
        state = env.render()
        actions_left = 3
        history = []
        while True:
            if qwen3_protocol:
                history.append({"state": state, "actions_left": actions_left})
                prompt = qwen3_protocol.trajectory(history, env_type="undercover")
            else:
                prompt = reformat_prompt(state)
            completion = llm.generate([prompt], sampling_params)[0].outputs[0]
            output = qwen3_protocol.response_text(completion) if qwen3_protocol else completion.text
            if qwen3_protocol:
                qwen3_protocol.record(completion, prompt=prompt, game_index=t, game_seed=seed, turn=len(history))
                actions = qwen3_protocol.actions(output, env_type="undercover")[:actions_left]
            else:
                action = extract_action(output)
                actions = [] if action is None else [action]
            if not actions:
                info_list.append('trainer-invalid-format')
                print('trainer-invalid-format')
                break

            turn_reward = 0
            for action in actions:
                state, reward, done, info = env.step(action)
                actions_left -= 1
                turn_reward += reward
                status = classify_action(state, reward, done)
                if status is not None or done:
                    break

            if status is not None:
                info_list.append(status)
                print(status)
                break
            if done:
                break
            if qwen3_protocol:
                if actions_left == 0:
                    info_list.append('fail')
                    print('fail')
                    break
                history[-1].update(
                    llm_response=output,
                    generated_body_token_ids=qwen3_protocol.body_token_ids(completion),
                    reward=turn_reward,
                )
                state = env.render()
    # 统计info_list中出现的不同情况的次数并计算对应的比例
    counter = Counter(info_list)
    total = len(info_list)
    # 存储结果文件
    output_dir = qwen3_protocol.output_dir if qwen3_protocol else Path("reason_test")
    log_path = output_dir / "undercover-log.txt"
    with log_path.open("a") as f:
        f.write(f"\n=== Model: {model_name} ===\n")
        f.write(f"undercover v.s. {config.player_info[0]['model_name']}\n")
        f.write("undercover test set\n")
        for key, value in counter.items():
            f.write(f"{key}: {value / total:.2%}\n")
    print(f"Results saved to {log_path}")
