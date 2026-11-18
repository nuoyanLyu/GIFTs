try:
    from .model_protocol import ModelProtocol, add_protocol_args
    from .qwen3_eval import Qwen3Eval, add_qwen3_args, is_qwen3
except ImportError:
    from model_protocol import ModelProtocol, add_protocol_args
    from qwen3_eval import Qwen3Eval, add_qwen3_args, is_qwen3

# prompt evaluate部分参考 https://github.com/allenai/CommonGen-Eval/tree/main
import json
from collections import Counter

from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer

from openai import OpenAI

import threading
import itertools

import argparse
from pathlib import Path
import tqdm
import random
import re
import os
import logging
import time

# root_path = '/root/autodl-tmp'
root_path = '/data1/lvnuoyan/llm_model/gift'
data_path = '/data1/lvnuoyan'
llm_judge = 'gpt-4o'
batch_size = 16

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="nash-math")
parser.add_argument("--model_name", type=str, default="nm150")
parser.add_argument("--device", type=str, default='0')
add_protocol_args(parser)
add_qwen3_args(parser)
args = parser.parse_args()
model_path = args.model_path
model_name = args.model_name
device = args.device
resolved_model_dir = args.model_dir or f"{root_path}/{model_path}/{model_name}"
tokenizer = hf_tokenizer(resolved_model_dir)
qwen3 = Qwen3Eval(tokenizer, args) if is_qwen3(resolved_model_dir, args) else None
protocol = ModelProtocol.from_model_path(resolved_model_dir, no_think=args.no_think)

'''
400条数据，json文件
"instruction"字段已经构建好了prompt
（可能这里并不需要模板信息了、直接输出结果就可以？试一下）
使用human_annotations作为参考的ground_truth，有两个，做两次对照取平均值
不考虑spacy进行测试等情况
"human_annotations"列表，每一个元素是一个字典，"ref"为对应的人类标注数据
'''
dataset = json.load(open(f"{data_path}/reasoning/commongen_lite_eval/commongen_hard.json"))

gt_count = [len(data['human_annotations']) for data in dataset]
print('no human_annotations', gt_count.count(0))
print('delete no human_annotations data')
dataset = [data for data in dataset if len(data['human_annotations']) > 0]

def load_llm():
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    engine_kwargs = (
        qwen3.llm_kwargs(max_model_len=8000) if qwen3 is not None
        else {'max_model_len': 8000}
    )
    llm = LLM(resolved_model_dir, **engine_kwargs)
    print('LLM initialized')
    sampling_params = (
        qwen3.sampling_params() if qwen3 is not None
        else SamplingParams(max_tokens=600, temperature=0.5)
    )
    return llm, sampling_params

class ThreadSafeCycle:
    def __init__(self, iterable):
        self._lock = threading.Lock()
        self._iterator = itertools.cycle(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            return next(self._iterator)


# load api-keys
keys = json.load(open('ragen/env/api-keys.json'))
dmx_keys = ThreadSafeCycle(keys['dmx'])


def llm_eval(concept_list, candidate_A, candidate_B):
    # prompt = prompt_eval.format(
    #     concept_list=concept_list,
    #     candidate_A=candidate_A,
    #     candidate_B=candidate_B,
    # )
    prompt = f"""
# Data

Given several concepts (i.e., nouns or verbs), we ask models to write a short and simple sentence that contains *all* the required words. 
The sentence should describe a common scene in daily life, and the concepts should be used in a natural way.

Concepts: "{concept_list}"

Model A: "{candidate_A}"

Model B: "{candidate_B}"

# Your Task

Your task is to choose a better sentence from the two candidates. Decide which model's sentence is better in terms of the naturalness and commonness of the scenes they describe. 

## Rules: 
- A better sentence should describe a common scene in daily life, and all concepts should be used in a natural way.
- You should prefer sentences that use all given concepts with correct part-of-speech tags. 
- A simpler and shorter sentence is preferred if it describes the same scene as the other sentence.
- If you think both sentences are equally good or bad, please choose "tie".

Now, please output your choice ("A" or "B" or "tie").

Your choice: 
"""
    client_options = {"max_retries": 0} if qwen3 is not None else {}
    while True:
        try:
            client = OpenAI(
                api_key=next(dmx_keys),  # 每次调用都换一个 key
                base_url='https://www.dmxapi.cn/v1',
                **client_options,
            )
            response = client.chat.completions.create(
                model=llm_judge,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0,
                timeout=10,
            )
            return response.choices[0].message.content
        except Exception as e:
            if qwen3 is not None:
                raise
            print(e)
            time.sleep(1)


def parse_result(result_str):
    if "neither" in result_str.lower():
        return "neither"
    elif "A" in result_str:
        return "A"
    elif "B" in result_str:
        return "B"
    elif "tie" in result_str:
        return "tie"
    else:
        return "Not Matched"


def reformat_prompt(prompt0):
    formatted_question = prompt0
    task_prompt = formatted_question
    if qwen3 is not None or protocol.answer_only:
        task_prompt += '\nPut only the requested sentence inside the answer block.'
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


def test_gen(llm, sampling_params, dataset):
    global common_judge_trace
    common_judge_trace = []
    answers = []
    acc_list = []
    for i in tqdm.trange(0, len(dataset), batch_size):  # len(math['test'])
        # 调整prompt内容，之前的格式不太对劲，导致模型输出的最后一个数字不是最后一个数字
        data = dataset[i: i + batch_size]
        '''
        doc_to_text: "{{context}}\nQuestion: {{question}}\nAnswer:"  
        doc_to_target: "{{answer}}"  # 正确答案的索引  
        doc_to_choice: "{{[answerA, answerB, answerC]}}"  # 三个选项
        '''
        prompt = [reformat_prompt(data[j]['instruction']) for j in range(len(data))]# 模型推理
        outputs = llm.generate(prompt, sampling_params)
        judge_failure = None
        for j, out in enumerate(outputs):
            # answer, choices, subject
            completion = out.outputs[0]
            response = qwen3.response_text(completion) if qwen3 is not None else completion.text
            solution = extract_solution(response)
            answers.append(completion.text)
            trace = {"index": i + j, "raw_answer": completion.text,
                     "extracted_answer": solution, "concept_set": data[j]['concept_set'],
                     "reference": None, "output_id": None, "candidate_A": None,
                     "candidate_B": None, "judge_raw": None, "judge_parsed": None}
            choice = None
            if solution is not None and judge_failure is None:
                ref = random.choice([item['ref'] for item in data[j]['human_annotations']])
                output_id = random.choice(['A', 'B'])
                candidate_a, candidate_b = (solution, ref) if output_id == 'A' else (ref, solution)
                trace.update(reference=ref, output_id=output_id,
                             candidate_A=candidate_a, candidate_B=candidate_b)
                try:
                    judge_response = llm_eval(data[j]['concept_set'], candidate_a, candidate_b)
                    trace['judge_raw'] = judge_response
                    decision = parse_result(judge_response)
                    trace['judge_parsed'] = decision
                    choice = int(decision in (output_id, 'tie'))
                except Exception as error:
                    if qwen3 is None:
                        raise
                    judge_failure = error
                    trace['judge_error'] = type(error).__name__
            elif solution is not None:
                trace['judge_error'] = 'skipped after previous judge failure'
            acc_list.append(choice)
            common_judge_trace.append({**trace, "score": choice})
            if qwen3 is not None:
                qwen3.record(completion, prompt=prompt[j], index=i + j,
                             answer=solution, score=choice, reference=trace['reference'],
                             output_id=trace['output_id'], concept_set=trace['concept_set'],
                             judge_raw=trace['judge_raw'], judge_parsed=trace['judge_parsed'],
                             judge_error=trace.get('judge_error'))
        if judge_failure is not None:
            # Preserve every generated response in this batch before failing the benchmark.
            raise RuntimeError(
                f'CommonGen judge failed ({type(judge_failure).__name__}); trace: {qwen3.trace_path}'
            ) from None
    return answers, acc_list


def save_sample_results(model_name, acc_list, answers, dataset, 
                       num_samples=5, output_dir="reason_test/results"):
    """
    保存部分正确和错误的测试结果，便于分析
    
    Args:
        model_name: 模型名称
        acc_list: 准确率列表、不区分类别
        answers: 模型生成的答案列表
        social: 测试数据集
        ground_truth: 正确答案列表
        num_samples: 每种类型保存的样本数量
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    output_file = f"{output_dir}/{model_name}-CommenGen-{time_str}.json"
    
    # 收集正确和错误的样本索引
    correct_indices = []
    incorrect_indices = []
    invalid_indices = []
    
    # 遍历所有类别和样本
    idx = 0
    for result in acc_list:
        if result is None:
            invalid_indices.append(idx)
        elif result == 1:
            correct_indices.append(idx)
        else:
            incorrect_indices.append(idx)
        idx += 1
    
    # 限制样本数量
    if len(correct_indices) > num_samples:
        correct_indices = correct_indices[:num_samples]
    if len(incorrect_indices) > num_samples:
        incorrect_indices = incorrect_indices[:num_samples]
    if len(invalid_indices) > num_samples:
        invalid_indices = invalid_indices[:num_samples]
    
    # 准备保存的数据
    samples = {
        "model_name": model_name,
        "correct_samples": [],
        "incorrect_samples": [],
        "invalid_samples": []
    }
    
    # 添加正确样本
    for idx in correct_indices:
        # 获取对应的题目信息
        data_idx = idx
        samples["correct_samples"].append({
            "question": dataset[data_idx]['instruction'],
            "model_answer": answers[idx],
            "ground_truth": dataset[data_idx]['human_annotations'][0]['ref'],
        })
    
    # 添加错误样本
    for idx in incorrect_indices:
        # 获取对应的题目信息
        data_idx = idx
        samples["incorrect_samples"].append({
            "question": dataset[data_idx]['instruction'],
            "model_answer": answers[idx],
            "ground_truth": dataset[data_idx]['human_annotations'][0]['ref'],
        })
    
    # 添加无效样本
    for idx in invalid_indices:
        # 获取对应的题目信息
        data_idx = idx
        samples["invalid_samples"].append({
            "question": dataset[data_idx]['instruction'],
            "model_answer": answers[idx],
            "ground_truth": dataset[data_idx]['human_annotations'][0]['ref'],
        })
    
    # 保存到文件
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=4)
    
    print(f"✅ Sample results saved to {output_file}")
    print(f"Saved {len(correct_indices)} correct, {len(incorrect_indices)} incorrect and {len(invalid_indices)} invalid samples")



def save_log(model_name, accs_list, output_file="reason_test/commongen-log.txt"):  
    acc0_strict = accs_list.count(1) / len(accs_list)
    invalid_strict = accs_list.count(None)

    # 写入日志文件
    with open(output_file, "a") as f:
        f.write(f"\n=== Model: {model_name} ===\n")
        f.write("common_gen test set\n")
        f.write(f"total win-tie: {acc0_strict:.4f}\n")
        f.write(f"invalid output: {invalid_strict}\n")
    print(f"✅ Log saved to {output_file}")


if __name__ == '__main__':
    os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    logging.getLogger("vllm").setLevel(logging.ERROR)
    # path0 = f'/root/autodl-tmp/reasoning'
    path0 = f'/data1/lvnuoyan/reasoning'
    if args.max_samples is not None:
        dataset = dataset[:args.max_samples]
    llm, sampling_params = load_llm()
    if qwen3 is not None:
        random.seed(args.seed)
    answers, acc_list = test_gen(llm, sampling_params, dataset)
    # 保存测试结果到日志文件
    output_dir = qwen3.output_dir if qwen3 is not None else (args.output_dir or Path('reason_test'))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_log(model_name, acc_list, output_file=output_dir / 'commongen-log.txt')
    # 保存部分测试结果，便于分析
    save_sample_results(model_name, acc_list, answers, dataset, num_samples=20, output_dir=output_dir / 'results')

    
