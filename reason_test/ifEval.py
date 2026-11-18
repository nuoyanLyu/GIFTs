try:
    from reason_test.model_protocol import ModelProtocol, add_protocol_args
    from reason_test.qwen3_eval import Qwen3Eval, add_qwen3_args, is_qwen3
except ModuleNotFoundError:
    from model_protocol import ModelProtocol, add_protocol_args
    from qwen3_eval import Qwen3Eval, add_qwen3_args, is_qwen3

import json
import re
import time
import tqdm
import os
import logging
from vllm import LLM, SamplingParams
from verl.utils import hf_tokenizer
import argparse
from datasets import load_dataset
import random
import string
from pathlib import Path
try:
    from reason_test import ifeval_batch as _ifeval_support
except ModuleNotFoundError:
    import ifeval_batch as _ifeval_support




root_path = '/data1/lvnuoyan/llm_model' 
data_path = '/data1/lvnuoyan/reasoning'
batch_size = 16
IFEVAL_MAX_TOKENS = 2500
ifeval_source_rows = []
ifeval_predictions = []
ifeval_metrics = {}
ifeval_scorer_identity = {}
ifeval_raw_answers = {}
ifeval_native_report = {}
ifeval_native_outputs = {}
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="weizihao")
parser.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct") 
parser.add_argument("--cuda_id", type=str, default="0")
parser.add_argument("--gpu_utilize", type=float, default=0.7)
add_protocol_args(parser)
add_qwen3_args(parser)
args = parser.parse_args()
model_path = args.model_path
model_name = args.model_name
gpu_memory_utilization = args.gpu_utilize
cuda_id = args.cuda_id
resolved_model_dir = args.model_dir or f"{root_path}/{model_path}/{model_name}"
tokenizer = hf_tokenizer(resolved_model_dir)
qwen3_protocol = Qwen3Eval(tokenizer, args) if is_qwen3(resolved_model_dir, args) else None
protocol = (qwen3_protocol if qwen3_protocol is not None else
            ModelProtocol.from_model_path(resolved_model_dir, no_think=args.no_think))
# tokenizer = hf_tokenizer(f"{root_path}/{model_name}")
time_str = time.strftime("%m-%d-%H-%M", time.localtime())
# file_name = 'game100-gsm8k-09-23-17-39.json'
# file_name = 'Qwen2.5-1.5B-Instruct-gsm8k-09-23-17-44.json'
# game100不使用原始prompt17-39： 0.4025, strict 0.3161
# Game100使用原始prompt21-55：0.3321——所以改了prompt反而效果更差，，strict 0.2009


def load_llm():
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id
    # tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    model = resolved_model_dir
    if qwen3_protocol is not None:
        sampling_params = qwen3_protocol.sampling_params(max_tokens=IFEVAL_MAX_TOKENS)
        llm = LLM(
            model,
            gpu_memory_utilization=gpu_memory_utilization,
            **qwen3_protocol.llm_kwargs(max_model_len=6000),
        )
        print("LLM initialized")
        return llm, sampling_params
    # model = f"{root_path}/{model_name}"
    # ro_config = config.actor_rollout_ref.rollout
    llm = LLM(
		model,
        max_model_len=6000,
        gpu_memory_utilization=gpu_memory_utilization,
	)
    print("LLM initialized")
    sampling_params = SamplingParams(
		max_tokens=1000, # 放大了token数量限制，因为发现了大量invalid的信息
		temperature=0.5,  # ro_config.val_kwargs.temperature,
	)
    return llm, sampling_params


def reformat_prompt(prompt0, *, max_tokens=IFEVAL_MAX_TOKENS):
    if qwen3_protocol is not None:
        messages = qwen3_protocol.messages(prompt0, max_tokens=max_tokens)
        return qwen3_protocol.render(messages)
    formatted_question = prompt0.strip() + '\n'
    task_prompt = formatted_question
    return protocol.render_prompt(
        tokenizer, task_prompt,
        legacy_suffix='',
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
    if qwen3_protocol is not None:
        return qwen3_protocol.extract(solution_str)
    return protocol.extract_answer(solution_str, legacy_extractor=_legacy_extract_solution)


def get_ifeval_dataset(language: str = "en", data_path: str = "/data1/lvnuoyan/reasoning/instruction-following-eval"):
    """
    Load the default dataset for a language from Hugging Face.
    
    Args:
        language: Language code, either "en" or "ru". Defaults to "en".
        
    Returns:
        List of InputExample objects.
        
    Raises:
        ValueError: If the dataset for the specified language cannot be loaded.
    """
    # if language == 'ru':
    #     language = 'ru' if use_legacy_behavior() else "ru_v2"
    try:
        from ifeval.core.evaluation import InputExample

        # Load the dataset from HuggingFace
        data_files = {"test": f"{language}.jsonl"}
        dataset = load_dataset(data_path, data_files=data_files)
        
        # Default to the validation split
        dataset = dataset["test"]
        
        # Parse the examples
        global ifeval_source_rows
        examples = []
        ifeval_source_rows = []
        for index, item in enumerate(dataset):
            # Dataset versions may return the JSON content as either text or an object.
            content = item.get("content", item)
            if isinstance(content, str):
                content = json.loads(content)
            if not isinstance(content, dict):
                raise ValueError(f"IFEval row {index} has invalid content")
            ifeval_source_rows.append({
                "key": content.get("key", index),
                "instruction_id_list": content["instruction_id_list"],
                "prompt": content["prompt"], "kwargs": content["kwargs"],
            })
            
            examples.append(
                InputExample(
                    instruction_id_list=content["instruction_id_list"],
                    prompt=content["prompt"],
                    kwargs=content["kwargs"],
                )
            )
        
        return examples
    
    except Exception as e:
        raise ValueError(f"Failed to load dataset for {language}: {e}")


def test_ifeval(llm, sampling_params, scorer):
    global ifeval_predictions, ifeval_metrics, ifeval_scorer_identity, ifeval_raw_answers
    global ifeval_native_report, ifeval_native_outputs
    # Reuse the recent scorer/resource checks: language seed 0 and legacy=False.
    evaluator, input_class, ifeval_scorer_identity = scorer
    datas = get_ifeval_dataset("en")
    datas = datas[:args.max_samples]
    by_prompt = {row["prompt"]: row for row in ifeval_source_rows}
    rows = [by_prompt[example.prompt] for example in datas]
    if len(rows) != len({row["prompt"] for row in rows}):
        raise ValueError("IFEval prompts must be unique")
    raw_generations = []
    bodies = []
    answers = {}
    for offset in tqdm.trange(0, len(datas), batch_size):
        batch = datas[offset:offset + batch_size]
        prompts = [reformat_prompt(example.prompt, max_tokens=sampling_params.max_tokens)
                   for example in batch]
        outputs = llm.generate(prompts, sampling_params)
        if len(outputs) != len(batch):
            raise ValueError("IFEval generation count differs from the input batch")
        for index, (output, prompt) in enumerate(zip(outputs, prompts), start=offset):
            completion = output.outputs[0]
            raw = completion.text
            row = rows[index]
            raw_generations.append({"index": index, **row, "model_answer": raw,
                "finish_reason": getattr(completion, "finish_reason", None)})
            answers[row["prompt"]] = raw
            if qwen3_protocol is not None:
                body = extract_solution(qwen3_protocol.response_text(completion))
                bodies.append(body)
                qwen3_protocol.record(
                    completion, prompt=prompt, item_index=index, item_key=row["key"],
                    benchmark_prompt=row["prompt"], answer=body,
                )
    ifeval_raw_answers = dict(answers)
    if qwen3_protocol is not None or protocol.answer_only:
        # The shared protocol alone decides validity. Feed its unchanged answer
        # body to the established native IFEval scoring/denominator validator.
        # Canonical wrapping is internal to that helper; original raw outputs
        # are restored in all saved prediction records below.
        if qwen3_protocol is None:
            bodies = [extract_solution(generated["model_answer"]) for generated in raw_generations]
        scorer_generations = []
        for generated, body in zip(raw_generations, bodies):
            canonical = "" if body is None else "<answer>" + body + "</answer>"
            scorer_generations.append({**generated, "model_answer": canonical})
        records, counts, report, all_outputs, answers = _ifeval_support.score_generations(
            rows, scorer_generations, evaluator, input_class)
        for record, raw in zip(records, raw_generations):
            record["model_answer"] = raw["model_answer"]
        ifeval_predictions, ifeval_metrics = records, counts
    else:
        # Opt-out retains the original raw-prompt/full-response protocol.
        report, all_outputs = evaluator.evaluate(datas, answers)
        report = _ifeval_support.json_value(report)
        all_outputs = _ifeval_support.json_value(all_outputs)
        ifeval_predictions = raw_generations
        ifeval_metrics = {"total": len(datas),
            "instruction_total": sum(len(example.instruction_id_list) for example in datas)}
        for mode, native in _ifeval_support.SCORE_MODES.items():
            for kind in ("prompt", "instruction"):
                ifeval_metrics[f"{mode}_{kind}_accuracy"] = report[native][f"{kind}_accuracy"]
    ifeval_native_report, ifeval_native_outputs = report, all_outputs
    evaluator.print_report(report)
    return report, all_outputs, answers


def save_sample_results(model_name, answer_dict, output_dir="reason_test/results"):
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
    output_file = f"{output_dir}/{model_name}-ifeval.json"
    # 保存到文件
    with open(output_file, "w") as f:
        json.dump(answer_dict, f, indent=4)
    # Preserve four metrics, per-instruction native flags and full raw outputs.
    base = Path(output_dir)
    _ifeval_support.atomic_json(base / f"{model_name}-ifeval-raw.json", ifeval_raw_answers)
    _ifeval_support.atomic_json(base / f"{model_name}-ifeval-report.json", ifeval_native_report)
    _ifeval_support.atomic_json(base / f"{model_name}-ifeval-native-outputs.json", ifeval_native_outputs)
    _ifeval_support.atomic_json(base / f"{model_name}-ifeval-predictions.json", ifeval_predictions)
    _ifeval_support.atomic_json(base / f"{model_name}-ifeval-summary.json", {
        "status": "complete", **ifeval_metrics, "protocol": protocol.to_metadata(),
        "scorer_identity": ifeval_scorer_identity,
    })


if __name__ == '__main__':
    os.environ["VLLM_DISABLE_PROGRESS_BAR"] = "1"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    logging.getLogger("vllm").setLevel(logging.ERROR)
    # 加载 MMLU 数据集，全部数据集all、对应数据字段test
    # mmlu = load_dataset(f"{path0}/mmlu/", 'all')['test']
    # zebra = load_dataset(f"{data_path}/ZebraLogic/mc_mode")['test']
    scorer = _ifeval_support.load_scorer()
    llm, sampling_params = load_llm()
    report, all_outputs, answers = test_ifeval(llm, sampling_params, scorer)
    # save answers
    
    print("Strict prompt accuracy:", report["eval_results_strict"]["prompt_accuracy"])
    print("Loose prompt accuracy:", report["eval_results_loose"]["prompt_accuracy"])
    print("Strict instruction accuracy:", report["eval_results_strict"]["instruction_accuracy"])
    print("Loose instruction accuracy:", report["eval_results_loose"]["instruction_accuracy"])
    # save_results_to_markdown(acc, model_name)
    # 保存部分测试结果，便于分析
    output_dir = getattr(qwen3_protocol, "output_dir", None) or args.output_dir
    results_log = "reason_test/ifeval-results.txt"
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_log = output_dir / "ifeval-results.txt"
    with open(results_log, "a") as f:
        f.write(f"\n## Model: {model_name}\n\n")
        f.write(f"Strict prompt accuracy: {report['eval_results_strict']['prompt_accuracy']:.4f}\n")
        f.write(f"Loose prompt accuracy: {report['eval_results_loose']['prompt_accuracy']}\n\n")

    save_sample_results(model_name, answers, output_dir=output_dir or "reason_test/results")
