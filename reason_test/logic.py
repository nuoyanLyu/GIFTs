try:
    from .model_protocol import ModelProtocol, add_protocol_args, extract_choice_label
    from .qwen3_eval import Qwen3Eval, add_qwen3_args, is_qwen3
except ImportError:
    from model_protocol import ModelProtocol, add_protocol_args, extract_choice_label
    from qwen3_eval import Qwen3Eval, add_qwen3_args, is_qwen3

import json
import re
import time
import tqdm
from math_verify import parse, verify
import os
import logging
from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer
import argparse
from pathlib import Path
from datasets import load_dataset
import random
import string

root_path = '/data1/lvnuoyan/llm_model' 
data_path = '/data1/lvnuoyan/reasoning'
batch_size = 16
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="math")
parser.add_argument("--model_name", type=str, default="math50") 
parser.add_argument("--cuda_id", "--device", dest="cuda_id", type=str, default="2")
parser.add_argument("--gpu_utilize", type=float, default=0.6)
add_protocol_args(parser)
add_qwen3_args(parser)
args = parser.parse_args()
model_path = args.model_path
model_name = args.model_name
gpu_memory_utilization = args.gpu_utilize
cuda_id = args.cuda_id
resolved_model_dir = args.model_dir or f"{root_path}/{model_path}/{model_name}"
tokenizer = hf_tokenizer(resolved_model_dir)
qwen3 = Qwen3Eval(tokenizer, args) if is_qwen3(resolved_model_dir, args) else None
protocol = ModelProtocol.from_model_path(resolved_model_dir, no_think=args.no_think)
# tokenizer = hf_tokenizer(f"{root_path}/{model_name}")
time_str = time.strftime("%m-%d-%H-%M", time.localtime())
# file_name = 'game100-gsm8k-09-23-17-39.json'
# file_name = 'Qwen2.5-1.5B-Instruct-gsm8k-09-23-17-44.json'
# game100不使用原始prompt17-39： 0.4025, strict 0.3161
# Game100使用原始prompt21-55：0.3321——所以改了prompt反而效果更差，，strict 0.2009

def load_llm():
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    engine_kwargs = (
        qwen3.llm_kwargs(max_model_len=6000) if qwen3 is not None
        else {'max_model_len': 6000}
    )
    engine_kwargs['gpu_memory_utilization'] = gpu_memory_utilization
    llm = LLM(resolved_model_dir, **engine_kwargs)
    print('LLM initialized')
    sampling_params = (
        qwen3.sampling_params() if qwen3 is not None
        else SamplingParams(max_tokens=1000, temperature=0.5)
    )
    return llm, sampling_params


def reformat_prompt(prompt0, choice):
    formatted_question = prompt0.strip() + '\n'
    for i, option in enumerate(choice):
        letter = string.ascii_uppercase[i]
        formatted_question += f'{letter}. {option}\n'
    task_prompt = formatted_question
    if qwen3 is not None or protocol.answer_only:
        labels = ", ".join("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(choice)])
        task_prompt += f"\nChoose exactly one option from {labels}. Put only its uppercase letter inside the answer block."
    if qwen3 is not None:
        return qwen3.render(qwen3.messages(task_prompt.rstrip(), max_tokens=200))
    return protocol.render_prompt(
        tokenizer, task_prompt,
        legacy_suffix="Let's think step by step and always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. Max response length: 200 words (tokens).",
        system_prompt="You're a helpful assistant. ",
    )


def _legacy_extract_solution(solution_str):
    pattern = r"<answer>(.*)</answer>"
    # 使用 re.search() 提取最后一个数字
    match = re.search(pattern, solution_str, re.DOTALL)
    if match:
        last_number = match.group(1)  # 提取匹配到的最后一个数字
        solu_strict = last_number
    else:
        solu_strict = None 
    return solu_strict


def extract_solution(solution_str):
    if qwen3 is not None:
        return qwen3.extract(solution_str)
    return protocol.extract_answer(solution_str, legacy_extractor=_legacy_extract_solution)


def extract_choice(text: str, allowed_labels=None):
    if qwen3 is not None:
        return extract_choice_label(text, allowed_labels or 'ABCDEFGHIJ')
    if protocol.answer_only:
        return protocol.extract_choice(text, allowed_labels=allowed_labels or 'ABCDEFGHIJ')

    """
    从文本中提取第一个A-J字母（句首答案），允许格式：
    - A
    - A.
    - A.hello
    - A something
    不匹配：
    - ADHD, BADGE 等单词中嵌入的情况
    """
    text = text.strip()
    match = re.search(r'(?<![A-Za-z])([A-I])(\.|(?:\s|$))', text)
    if match:
        return match.group(1).upper()
    return None


def test_logic(llm, sampling_params, logic):
    answers = []
    accs = {}
    acc_list = []
    for i in tqdm.trange(0, len(logic), batch_size): 
        # 调整prompt内容，之前的格式不太对劲，导致模型输出的最后一个数字不是最后一个数字
        data = logic[i: i + batch_size]
        prompt = [reformat_prompt(data['puzzle'][j]  + '\n' + data['question'][j], data['choices'][j]) for j in range(len(data['question']))]# 模型推理
        outputs = llm.generate(prompt, sampling_params)
        for j, out in enumerate(outputs):
            # print(out.outputs[0].text)
            completion = out.outputs[0]
            response = qwen3.response_text(completion) if qwen3 is not None else completion.text
            solution = extract_solution(response)
            answers.append(completion.text)
            label = data['id'][j][9:12]
            ground_truth = 'ABCDEFGHIJ'[data['choices'][j].index(data['answer'][j])]
            choice = None
            if solution is not None:
                choice = extract_choice(solution, allowed_labels="ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(data['choices'][j])])
            if solution is None or ((qwen3 is not None or protocol.answer_only) and choice is None):
                score = None
            else:
                score = choice == ground_truth
            if qwen3 is not None:
                qwen3.record(completion, prompt=prompt[j], index=i + j,
                             answer=solution, selected_label=choice, expected=ground_truth,
                             score=score)
            if label not in accs:
                accs[label] = []
            accs[label].append(score)
            acc_list.append(score)
            
    return answers, accs, acc_list


def save_results_to_markdown(acc, model_name, output_file='reason_test/ZebraLogic-results.txt'):
    acc_list = []
    rows = []
    print('ZebraLogic, model', model_name)
    # 计算每个 key 的准确率和 invalid 率
    for k in acc.keys():
        total = len(acc[k])
        valid = acc[k].count(1)
        acc_rate = valid / total

        # 构造 markdown 表格每一行
        rows.append(f"| {k} | {acc_rate:.4f} |")

        acc_list += acc[k]

    total_acc = acc_list.count(1) / len(acc_list)
    total_invalid = acc_list.count(None) / len(acc_list)

    with open(output_file, "a") as f:
        f.write(f"\n## Model: {model_name}\n\n")
        f.write(f"| type | {model_name} |\n")
        f.write("|----------|-----------|\n")
        for row in rows:
            f.write(row + "\n")
        f.write(f"| **Total** | **{total_acc:.4f}** |\n\n")
    print(f"total acc: {total_acc:.4f}")
    print(f"Results saved to {output_file}")


def save_sample_results(model_name, acc_list, answers, mmlu_pro_data, 
                       num_samples=100, output_dir="reason_test/results"):
    """
    保存部分正确和错误的测试结果，便于分析
    
    Args:
        model_name: 模型名称
        accs: 准确率列表，按类别分组
        answers: 模型生成的答案列表
        mmlu_pro_data: 测试数据集
        num_samples: 每种类型保存的样本数量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    output_file = f"{output_dir}/{model_name}-ZebraLogic-{time_str}.json"
    
    # 改为保留所有Answer中前100个，这样方便后续对比不同模型的正确-错误原因
    # 遍历所有样本
    # idx = 0
    # for result in acc_list:
    #     if result is None:
    #         invalid_indices.append(idx)
    #     elif result == 1:
    #         correct_indices.append(idx)
    #     else:
    #         incorrect_indices.append(idx)
    #     idx += 1
    
    # # 限制样本数量
    # if len(correct_indices) > num_samples:
    #     correct_indices = correct_indices[:num_samples]
    # if len(incorrect_indices) > num_samples:
    #     incorrect_indices = incorrect_indices[:num_samples]
    # if len(invalid_indices) > num_samples:
    #     invalid_indices = invalid_indices[:num_samples]
    
    # 准备保存的数据
    samples = {
        "info":{"model_name": model_name, 
                 "total_acc": acc_list.count(1) / len(acc_list),
                 "invalid": acc_list.count(None) / len(acc_list)},
        "results":[]
    }
    
    # 添加样本
    for idx in range(min(num_samples, len(mmlu_pro_data))):
        # 获取对应的题目信息
        # data_idx = idx % len(mmlu_pro_data)
        samples["results"].append({
            "question": mmlu_pro_data['question'][idx],
            "choices": mmlu_pro_data['choices'][idx],
            "model_answer": answers[idx],
            "ground_truth": mmlu_pro_data['answer'][idx],
            "type": mmlu_pro_data['id'][idx][9:12],
            "result": acc_list[idx]
        })
    
    # 保存到文件
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=4)
    
    print(f"✅ Sample results saved to {output_file}")
    # print(f"Saved {len(correct_indices)} correct, {len(incorrect_indices)} incorrect and {len(invalid_indices)} invalid samples")


if __name__ == '__main__':
    os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    logging.getLogger("vllm").setLevel(logging.ERROR)
    # 加载 MMLU 数据集，全部数据集all、对应数据字段test
    # mmlu = load_dataset(f"{path0}/mmlu/", 'all')['test']
    zebra = load_dataset(f"{data_path}/ZebraLogic/mc_mode")['test']
    if args.max_samples is not None:
        zebra = zebra.select(range(min(args.max_samples, len(zebra))))
    llm, sampling_params = load_llm()
    # exit(0)
    answers, acc, acc_list = test_logic(llm, sampling_params, zebra)
    output_dir = qwen3.output_dir if qwen3 is not None else (args.output_dir or Path('reason_test'))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results_to_markdown(acc, model_name, output_file=output_dir / 'ZebraLogic-results.txt')
    # 保存部分测试结果，便于分析
    save_sample_results(model_name, acc_list, answers, zebra, num_samples=100, output_dir=output_dir / 'results')
