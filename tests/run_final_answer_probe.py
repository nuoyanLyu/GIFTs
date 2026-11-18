import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
from urllib.request import Request, urlopen

import numpy as np
import pyarrow.parquet as parquet
from transformers import AutoTokenizer
from verl import DataProto
import torch
import yaml

from test_final_answer_protocol import MODEL, manager, states_for
from ragen.llm_agent import ctx_manager

MATH = Path("/data1/lvnuoyan/reasoning/SimpleRL-Zoo-Data/simplelr_abel_level3to5/test.parquet")
TASKS = [
    ("general", "Which planet is known as the Red Planet?", "Mars"),
    ("general", "Translate the English word 'cat' into Spanish. Use the common masculine singular form.", "gato"),
    ("general", "Classify this review as positive or negative: 'The service was excellent and I would return.'", "positive"),
    ("general", "Choose A, B, C or D. Which is a mammal? A. Salmon B. Dolphin C. Octopus D. Trout.", "B"),
    ("general", "Which programming language uses a keyword spelled 'def' to introduce a function: Python or Java?", "Python"),
    ("game", "Legal moves are left, right, and stay. Move toward the exit directly to your right. Give the move name.", "right"),
    ("game", "In rock-paper-scissors your opponent played Rock. Choose from Rock, Paper, or Scissors to win.", "Paper"),
    ("game", "In tic-tac-toe you play X. Board rows: X X . / O . O / . . . . Give the winning position as row,column, numbered from 1.", "1,3"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("port", type=int)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    directory = args.directory
    url = f"http://127.0.0.1:{args.port}/v1/completions"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    table = parquet.read_table(MATH, columns=["extra_info"]).to_pylist()
    indices = random.Random(123).sample(range(len(table)), 8)
    samples = [{"kind": "math_lv3to5", "dataset_index": index,
                "question": table[index]["extra_info"]["question"], "expected": table[index]["extra_info"]["answer"]}
               for index in indices]
    samples += [{"kind": kind, "question": question, "expected": answer}
                for kind, question, answer in TASKS]
    context = manager(tokenizer, samples)
    states = states_for(samples)
    rollout = context.get_lm_inputs(deepcopy(states), prepare_for_update=False)
    sampling = {"temperature": 0.5, "top_p": 1.0, "top_k": -1, "min_p": 0.0}
    max_tokens = 600
    if args.config is not None:
        config = yaml.safe_load(args.config.read_text())
        assert config["data"]["apply_chat_template_kwargs"]["enable_thinking"] is False
        assert config["agent_proxy"]["response_protocol"] == "final_answer_with_prefix"
        rollout_config = config["actor_rollout_ref"]["rollout"]
        validation = rollout_config["val_kwargs"]
        assert validation["do_sample"]
        sampling.update({key: validation[key] for key in ("temperature", "top_p", "top_k")})
        max_tokens = rollout_config["response_length"]
    settings = {"model": MODEL, **sampling, "max_tokens": max_tokens,
                "return_token_ids": True, "add_special_tokens": False}
    eos_ids = [151645, 151643]

    def run(index):
        ids = rollout.batch["input_ids"][index][rollout.batch["attention_mask"][index].bool()].tolist()
        messages = rollout.non_tensor_batch["messages_list"][index].tolist()
        native = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        assert ids == native, index
        payload = {**settings, "prompt": ids, "seed": 123 + index}
        request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
        assert result["model"] == MODEL
        choice = result["choices"][0]
        assert choice["prompt_token_ids"] == ids
        generated = choice["token_ids"]
        assert isinstance(generated, list) and generated
        return {"index": index, **samples[index], "prompt": tokenizer.decode(ids), "prompt_ids": ids,
                "generated_ids": generated, "server_response": result}

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(run, range(len(samples))))
    (directory / "model_raw.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    width = max(len(record["generated_ids"]) for record in records)
    responses = torch.tensor([record["generated_ids"] + [tokenizer.pad_token_id] *
                              (width - len(record["generated_ids"])) for record in records])
    output = DataProto.from_dict(
        tensors={"responses": responses},
        non_tensors={"env_ids": np.array(list(range(len(records))), dtype=object)},
    )
    parsed = context.get_env_inputs(output)
    for state, item in zip(states, parsed):
        state["history"][0].update(item, reward=float(bool(item["actions"])))
        state["history"].append({"state": "Next state", "actions_left": 0})
    batch = context.formulate_rollouts(deepcopy(states))
    for index, (record, item) in enumerate(zip(records, parsed)):
        raw = item["llm_raw_response"]
        actions = item["actions"]
        choice = record["server_response"]["choices"][0]
        packed_messages = batch.non_tensor_batch["messages_list"][index]
        assert packed_messages[-1]["content"] == raw == item["llm_response"], index
        masked = tokenizer.decode(batch.batch["responses"][index][batch.batch["loss_mask"][index]])
        assert raw in masked, index
        packed = batch.batch["input_ids"][index][batch.batch["attention_mask"][index].bool()].tolist()
        generated = record["generated_ids"]
        if generated[-1] in eos_ids:
            generated = generated[:-1]
        strict_ok = bool(re.fullmatch(r"(?:(?!</?answer>).)*<answer>(?:(?!</?answer>).)+</answer>\s*", raw, re.S))
        if strict_ok:
            strict_ok = bool(raw.split("<answer>", 1)[1].split("</answer>", 1)[0].strip())
        record.update(
            raw_response=raw, actions=actions, format_ok=strict_ok,
            prefix_present=("<answer>" in raw and bool(raw.split("<answer>", 1)[0].strip())),
            generated_think_tags=("<think>" in raw or "</think>" in raw),
            finish_reason=choice["finish_reason"], completion_tokens=len(record["generated_ids"]),
            packed_prompt_exact=packed[:len(record["prompt_ids"])] == record["prompt_ids"],
            packed_body_ids_exact=packed[len(record["prompt_ids"]):len(record["prompt_ids"]) + len(generated)] == generated,
        )
    categories = {}
    for name in ("math_lv3to5", "general", "game", "all"):
        subset = records if name == "all" else [record for record in records if record["kind"] == name]
        categories[name] = {
            "total": len(subset), "format_ok": sum(record["format_ok"] for record in subset),
            "parser_accepted": sum(bool(record["actions"]) for record in subset),
            "format_ok_with_prefix": sum(record["format_ok"] and record["prefix_present"] for record in subset),
            "generated_think_tags": sum(record["generated_think_tags"] for record in subset),
            "packed_prompt_exact": sum(record["packed_prompt_exact"] for record in subset),
            "packed_body_ids_exact": sum(record["packed_body_ids_exact"] for record in subset),
            "finish_reasons": Counter(record["finish_reason"] for record in subset),
        }
    report = {"model": MODEL, "enable_thinking": False, "response_protocol": "final_answer_with_prefix",
              "source_sha256": hashlib.sha256(Path(ctx_manager.__file__).read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest() if args.config else None,
              "settings": settings, "dataset": str(MATH), "categories": categories, "records": records}
    (directory / "model_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(categories, ensure_ascii=False, indent=2), flush=True)
    assert all(record["packed_prompt_exact"] and record["packed_body_ids_exact"] for record in records)


if __name__ == "__main__":
    main()
