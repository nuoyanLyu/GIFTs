#!/usr/bin/env python3
"""Run IFEval with the requested existing CoT/XML prompt and answer-only scoring.

The user requested the same CoT/XML instruction as MMLU/CommonGen. The native
IFEval scorer receives the unchanged text inside the legacy greedy answer tag;
missing tags become an empty response and fail every constraint. Generation is
saved before scoring, so a scorer/resource failure never loses GPU outputs.
"""

import argparse
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib.request import Request, urlopen


DEFAULT_DATASET = "/data1/lvnuoyan/reasoning/instruction-following-eval"
SCORE_MODES = {"strict": "eval_results_strict", "loose": "eval_results_loose"}
INSTRUCTION = (
    "Let's think step by step and always output: <think> [Your thoughts] "
    "</think> <answer> [your answer] </answer> with no extra text. Strictly "
    "follow this format."
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported native scoring result: {type(value).__name__}")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_value(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_messages(prompt):
    # Same structure as common.py, with its 200-word suffix removed by request.
    return [{"role": "system", "content": "You're a helpful assistant. "},
            {"role": "user", "content": prompt.strip() + "\n" + INSTRUCTION}]


def extract_scored_response(model_answer):
    # Preserve the greedy/DOTALL extraction used by MMLU and CommonGen.
    match = re.search(r"<answer>(.*)</answer>", model_answer, re.DOTALL)
    return (match.group(1), False) if match else ("", True)


def validate_rows(rows):
    if not rows:
        raise ValueError("IFEval input is empty")
    seen_prompts, seen_keys = set(), set()
    for index, row in enumerate(rows):
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Row {index}: missing prompt")
        if row["prompt"] in seen_prompts:
            raise ValueError(f"Row {index}: duplicate prompt would overwrite an answer")
        seen_prompts.add(row["prompt"])
        key = json.dumps(row.get("key"), ensure_ascii=False, sort_keys=True)
        if key in seen_keys:
            raise ValueError(f"Row {index}: duplicate key")
        seen_keys.add(key)
        ids, kwargs = row.get("instruction_id_list"), row.get("kwargs")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError(f"Row {index}: invalid instruction_id_list")
        if not isinstance(kwargs, list) or len(kwargs) != len(ids) or any(not isinstance(x, dict) for x in kwargs):
            raise ValueError(f"Row {index}: kwargs must align with instruction_id_list")


def load_rows(dataset_path):
    """Read exactly the English JSONL used by the existing dataset loader."""
    path = Path(dataset_path)
    source = path / "en.jsonl" if path.is_dir() else path
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            content = item.get("content", item)
            if isinstance(content, str):
                content = json.loads(content)
            if not isinstance(content, dict):
                raise ValueError(f"Line {line_number}: content is not an object")
            rows.append({"key": content.get("key", len(rows)),
                         "instruction_id_list": content.get("instruction_id_list"),
                         "prompt": content.get("prompt"), "kwargs": content.get("kwargs")})
    validate_rows(rows)
    return rows


def dataset_fingerprint(rows):
    digest = hashlib.sha256()
    for row in rows:
        effective = {key: row.get(key) for key in ("key", "instruction_id_list", "prompt", "kwargs")}
        digest.update(json.dumps(effective, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def model_manifest(model_dir):
    files = {}
    for pattern in ("*.safetensors", "pytorch_model*.bin", "*.json", "tokenizer.*", "merges.txt", "vocab.*"):
        for path in sorted(model_dir.glob(pattern)):
            if path.is_file():
                stat = path.stat()
                files[path.name] = {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return [files[name] for name in sorted(files)]


def load_scorer():
    # These are the exact imports and constructor used in the existing script.
    import ifeval
    from ifeval import Evaluator, instruction_registry
    from ifeval.core.evaluation import InputExample
    from ifeval.core import legacy_behavior, use_legacy_behavior
    from langdetect import DetectorFactory, detect
    import nltk
    # This controls only the rule-based language checker, not model sampling.
    DetectorFactory.seed = 0
    use_legacy_behavior(False)
    if detect("This is a clear English sentence about people and their daily work.") != "en":
        raise ValueError("Language checker resource validation failed")
    if len(nltk.sent_tokenize("First sentence. Second sentence.")) != 2 or nltk.word_tokenize("APPLE tree") != ["APPLE", "tree"]:
        raise ValueError("English NLTK sentence/word tokenizer validation failed")
    punkt = nltk.data.find("tokenizers/punkt/english.pickle")
    with punkt.open() as handle:
        punkt_hash = hashlib.sha256(handle.read()).hexdigest()
    try:
        version = importlib.metadata.version("ifeval")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(ifeval, "__version__", None)
    package_dir = Path(ifeval.__file__).resolve().parent
    source_files = {str(p.relative_to(package_dir)): file_sha256(p)
                    for p in sorted(package_dir.rglob("*.py"))}
    dependencies = {}
    for package in ("nltk", "langdetect", "immutabledict", "absl-py"):
        dependencies[package] = importlib.metadata.version(package)
    identity = {"module": "ifeval", "version": version, "source_files_sha256": source_files,
                "legacy_behavior": legacy_behavior(), "langdetect_seed": DetectorFactory.seed,
                "dependencies": dependencies,
                "resources": {"punkt_english_sha256": punkt_hash,
                              "language_detection_english_preflight": True,
                              "english_sentence_and_word_tokenizer_preflight": True}}
    if identity["legacy_behavior"] is not False:
        raise ValueError("IFEval legacy behavior must be disabled")
    return Evaluator(instruction_registry), InputExample, identity


def check_native_metrics(report, counts):
    for mode, native_key in SCORE_MODES.items():
        native = report[native_key]
        for kind, denominator in (("prompt", counts["total"]), ("instruction", counts["instruction_total"])):
            numerator = counts[f"{mode}_{kind}_correct"]
            name = f"{kind}_accuracy"
            if not isinstance(native.get(name), (int, float)) or not math.isclose(
                    native[name], numerator / denominator, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Native {native_key}.{name} disagrees with full-dataset counts")
            for key, expected in ((f"{kind}_total", denominator), (f"{kind}_correct", numerator)):
                if key in native and native[key] != expected:
                    raise ValueError(f"Native {native_key}.{key} excludes or changes input records")


def score_generations(rows, generated, evaluator, input_class):
    """Use native IFEval scoring and reject any skipped/reordered input."""
    if len(rows) != len(generated):
        raise ValueError("Generation count differs from selected prompt count")
    if [g.get("index") for g in generated] != list(range(len(rows))):
        raise ValueError("Generations must contain all selected indices in order")
    answers, format_invalid = {}, []
    for index, (row, generation) in enumerate(zip(rows, generated)):
        if any(generation.get(k) != row[k] for k in ("key", "instruction_id_list", "prompt", "kwargs")) or not isinstance(generation.get("model_answer"), str):
            raise ValueError(f"Generation {index} does not match input prompt/raw response")
        scored_response, invalid = extract_scored_response(generation["model_answer"])
        answers[row["prompt"]] = scored_response
        format_invalid.append(invalid)
    if len(answers) != len(rows):
        raise ValueError("Not every original prompt has one response")
    examples = [input_class(instruction_id_list=r["instruction_id_list"],
                            prompt=r["prompt"], kwargs=r["kwargs"]) for r in rows]
    report_native, outputs_native = evaluator.evaluate(examples, answers)
    report, outputs = json_value(report_native), json_value(outputs_native)
    records = [{**g, "scored_response": answers[row["prompt"]], "format_invalid": invalid}
               for g, row, invalid in zip(generated, rows, format_invalid)]
    for mode, native_key in SCORE_MODES.items():
        mode_outputs = outputs[native_key]
        if not isinstance(mode_outputs, list) or len(mode_outputs) != len(rows):
            raise ValueError(f"Native {native_key} skipped input prompts")
        for index, (row, scored) in enumerate(zip(rows, mode_outputs)):
            if scored.get("prompt") != row["prompt"] or scored.get("response") != answers[row["prompt"]]:
                raise ValueError(f"Native {native_key} output {index} was skipped/reordered/modified")
            if scored.get("instruction_id_list") != row["instruction_id_list"]:
                raise ValueError(f"Native {native_key} instruction IDs do not match")
            passed = scored.get("follow_instruction_list")
            if not isinstance(passed, list) or len(passed) != len(row["instruction_id_list"]) or any(type(x) is not bool for x in passed):
                raise ValueError(f"Native {native_key} instruction scores are incomplete")
            if type(scored.get("follow_all_instructions")) is not bool or scored["follow_all_instructions"] != all(passed):
                raise ValueError(f"Native {native_key} prompt pass does not match instruction scores")
            if format_invalid[index] and any(passed):
                raise ValueError(f"Native {native_key} accepted a missing-answer-tag response")
            records[index][f"{mode}_follow_instruction_list"] = passed
            records[index][f"{mode}_prompt_pass"] = all(passed)
    counts = {"total": len(rows), "instruction_total": sum(len(r["instruction_id_list"]) for r in rows),
              "format_invalid_count": sum(format_invalid),
              "truncated": sum(r.get("finish_reason") == "length" for r in records)}
    for mode in SCORE_MODES:
        counts[f"{mode}_prompt_correct"] = sum(r[f"{mode}_prompt_pass"] for r in records)
        counts[f"{mode}_instruction_correct"] = sum(sum(r[f"{mode}_follow_instruction_list"]) for r in records)
        counts[f"{mode}_prompt_accuracy"] = counts[f"{mode}_prompt_correct"] / counts["total"]
        counts[f"{mode}_instruction_accuracy"] = counts[f"{mode}_instruction_correct"] / counts["instruction_total"]
    check_native_metrics(report, counts)
    return records, counts, report, outputs, answers


class ApiBackend:
    def __init__(self, args):
        self.args, self.base = args, args.api_base.rstrip("/")
        ids = [x["id"] for x in self.request("models").get("data", [])]
        if args.api_model_id:
            if args.api_model_id not in ids:
                raise ValueError(f"API model {args.api_model_id!r} not in {ids}")
            self.model_id = args.api_model_id
        elif len(ids) == 1:
            self.model_id = ids[0]
        else:
            raise ValueError("Specify --api-model-id for a multi-model service")

    def request(self, endpoint, payload=None):
        request = Request(f"{self.base}/{endpoint}",
                          data=None if payload is None else json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.args.api_timeout) as response:
            return json.load(response)

    def generate(self, batch):
        generated = []
        for row in batch:
            response = self.request("chat/completions", {
                "model": self.model_id, "messages": build_messages(row["prompt"]),
                "temperature": self.args.temperature, "max_tokens": self.args.max_tokens})
            choice = response["choices"][0]
            text = choice["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("API returned non-string content")
            usage = response.get("usage", {})
            generated.append({"model_answer": text, "prompt_tokens": usage.get("prompt_tokens"),
                              "completion_tokens": usage.get("completion_tokens"),
                              "finish_reason": choice.get("finish_reason")})
        return generated


class OfflineBackend:
    def __init__(self, args):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from verl.utils import hf_tokenizer
        from vllm import LLM, SamplingParams
        self.tokenizer = hf_tokenizer(str(args.model_dir))
        self.llm = LLM(model=str(args.model_dir), max_model_len=args.max_model_len,
                       gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
                       max_num_seqs=args.max_num_seqs, enforce_eager=args.enforce_eager)
        self.params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)

    def generate(self, batch):
        prompts = [self.tokenizer.apply_chat_template(build_messages(r["prompt"]),
                   add_generation_prompt=True, tokenize=False) for r in batch]
        outputs = self.llm.generate(prompts, self.params, use_tqdm=False)
        if len(outputs) != len(batch):
            raise RuntimeError("vLLM output count differs from input count")
        return [{"model_answer": out.outputs[0].text, "prompt_tokens": len(out.prompt_token_ids),
                 "completion_tokens": len(out.outputs[0].token_ids),
                 "finish_reason": out.outputs[0].finish_reason} for out in outputs]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--model-label", required=True)
    p.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="5")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=1000)
    p.add_argument("--max-model-len", type=int, default=6000)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--limit", type=int, help="Smoke testing only")
    p.add_argument("--api-base", help="Existing service URL including /v1; smoke only")
    p.add_argument("--api-model-id")
    p.add_argument("--api-timeout", type=float, default=180)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--score-only", action="store_true", help="Score saved generations without loading a model")
    args = p.parse_args(argv)
    for name in ("model_dir", "dataset_path", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not (args.model_dir / "config.json").is_file():
        p.error("--model-dir must contain config.json")
    if not (args.dataset_path / "en.jsonl").is_file():
        p.error("--dataset-path must contain en.jsonl")
    if args.batch_size < 1 or args.max_num_seqs < 1 or args.limit is not None and args.limit < 1:
        p.error("Batch/sequence/limit values must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        p.error("GPU memory utilization must be in (0,1]")
    if not 0 < args.max_tokens < args.max_model_len or args.temperature < 0:
        p.error("Require 0 < max-tokens < max-model-len and temperature >= 0")
    if args.api_base and args.limit is None:
        p.error("--api-base requires --limit; API mode cannot enter the full table")
    return args


def make_identity(args, rows, scorer_identity):
    protocol = {"name": "ifeval_cot_xml_answer_only_v1",
                "scoring_mode": "cot_xml_answer_only",
                "length_constraint_removed_by_user": True,
                "prompt_sha256": hashlib.sha256(json.dumps(build_messages("PROMPT"), ensure_ascii=False).encode()).hexdigest(),
                "scorer": "ifeval.Evaluator(instruction_registry)",
                "response": "legacy_greedy_answer_inner_text_missing_tag_empty",
                "temperature": args.temperature, "max_tokens": args.max_tokens,
                "max_model_len": args.max_model_len, "batch_size": args.batch_size, "seed": args.seed}
    return {"model_dir": str(args.model_dir), "model_label": args.model_label,
            "model_files": model_manifest(args.model_dir), "dataset_path": str(args.dataset_path),
            "dataset_fingerprint": dataset_fingerprint(rows), "dataset_size": len(rows),
            "limit": args.limit, "mode": "api_smoke" if args.api_base else "offline",
            "api_base": args.api_base, "api_model_id": args.api_model_id, "protocol": protocol,
            "engine": {"gpu_memory_utilization": args.gpu_memory_utilization,
                       "max_num_seqs": args.max_num_seqs, "enforce_eager": args.enforce_eager,
                       "vllm_use_v1_environment": os.environ.get("VLLM_USE_V1")},
            "scorer_identity": scorer_identity}


def completed_result(output_dir, identity, expected_total):
    path = output_dir / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    if summary.get("status") != "complete" or summary.get("run_identity") != identity:
        raise ValueError("Existing summary has different/incomplete identity; use a fresh output directory")
    predictions = output_dir / "predictions.jsonl"
    if summary.get("total") != expected_total or not predictions.is_file() or file_sha256(predictions) != summary.get("predictions_sha256"):
        raise ValueError("Existing result failed completeness/hash checks")
    return summary


def main(argv=None):
    args = parse_args(argv)
    all_rows = load_rows(args.dataset_path)
    rows = all_rows if args.limit is None else all_rows[:args.limit]
    evaluator, input_class, scorer_identity = load_scorer()
    identity = make_identity(args, all_rows, scorer_identity)
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.validate_only:
        dummy = "This is a nonempty English response for complete IFEval checker validation."
        def dummy_generations(text):
            return [{"index": index, **row, "model_answer": text, "finish_reason": "stop"}
                    for index, row in enumerate(all_rows)]
        _, full_counts, _, _, _ = score_generations(
            all_rows, dummy_generations("<think>validation</think><answer>" + dummy + "</answer>"), evaluator, input_class)
        _, missing_counts, _, _, _ = score_generations(
            all_rows, dummy_generations("No answer tags in this validation response."), evaluator, input_class)
        if missing_counts["format_invalid_count"] != len(all_rows) or any(
                missing_counts[f"{mode}_{kind}_correct"] for mode in SCORE_MODES for kind in ("prompt", "instruction")):
            raise ValueError("Missing-tag responses must all fail without reducing denominators")
        print(json.dumps({"validated": True, "selected_total": len(rows),
                          "instruction_total": sum(len(r["instruction_id_list"]) for r in rows),
                          "run_identity": identity, "arguments": arguments,
                          "scorer_preflight": {"full_prompts": full_counts["total"],
                              "full_instructions": full_counts["instruction_total"],
                              "missing_answer_tag_fails_all_constraints": True}}, ensure_ascii=False, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cached = completed_result(args.output_dir, identity, len(rows))
    if cached:
        print(f"Already complete: {args.model_label}", flush=True)
        return 0
    previous_run = args.output_dir / "run.json"
    if args.score_only:
        if not previous_run.is_file() or json.loads(previous_run.read_text()).get("run_identity") != identity:
            raise ValueError("--score-only requires a saved matching run identity")
    started_at, started = utc_now(), time.monotonic()
    if args.score_only:
        old = json.loads(previous_run.read_text()); started_at = old["started_at"]
    atomic_json(previous_run, {"status": "running", "started_at": started_at, "pid": os.getpid(),
                              "run_identity": identity, "arguments": arguments})
    atomic_json(args.output_dir / "first_prompt.json", {"index": 0, "messages": build_messages(rows[0]["prompt"])})
    print(f"START {args.model_label}: {len(rows)}/{len(all_rows)} prompts, mode={identity['mode']}", flush=True)
    generations_path = args.output_dir / "generations.jsonl"
    if args.score_only:
        generated = [json.loads(line) for line in generations_path.read_text().splitlines() if line.strip()]
    else:
        backend = ApiBackend(args) if args.api_base else OfflineBackend(args)
        atomic_json(args.output_dir / "engine_ready.json", {"ready_at": utc_now(), "pid": os.getpid(),
                    "model_dir": str(args.model_dir), "model_label": args.model_label, "mode": identity["mode"]})
        generated = []
        with generations_path.open("w", encoding="utf-8") as handle:
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset:offset + args.batch_size]
                outputs = backend.generate(batch)
                if len(outputs) != len(batch):
                    raise RuntimeError("Backend output count differs from input count")
                for index, (row, response) in enumerate(zip(batch, outputs), start=offset):
                    record = {"index": index, **row, **response}
                    generated.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
                print(f"PROGRESS {args.model_label} {len(generated)}/{len(rows)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        atomic_json(args.output_dir / "raw_answers.json", {r["prompt"]: r["model_answer"] for r in generated})
    # Raw generations and answers already persist even if a checker now fails.
    records, metrics, report, outputs, answers = score_generations(rows, generated, evaluator, input_class)
    atomic_json(args.output_dir / "answers.json", answers)
    atomic_json(args.output_dir / "native_report.json", report)
    atomic_json(args.output_dir / "native_outputs.json", outputs)
    predictions = args.output_dir / "predictions.jsonl"
    temporary = predictions.with_name(predictions.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, predictions)
    packages = {}
    for package in ("ifeval", "vllm", "transformers", "torch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    summary = {"status": "complete", "benchmark": "ifeval", "model_dir": str(args.model_dir),
               "model_label": args.model_label, "mode": identity["mode"],
               "is_smoke": args.limit is not None or bool(args.api_base),
               "dataset_path": str(args.dataset_path), "dataset_fingerprint": identity["dataset_fingerprint"],
               "dataset_size": len(all_rows), "run_identity": identity, "protocol": identity["protocol"],
               "scorer_identity": scorer_identity, "arguments": arguments, "packages": packages,
               "started_at": started_at, "completed_at": utc_now(), "elapsed_seconds": time.monotonic()-started,
               "predictions_sha256": file_sha256(predictions),
               "generations_sha256": file_sha256(generations_path),
               "answers_sha256": file_sha256(args.output_dir / "answers.json"),
               "native_report_sha256": file_sha256(args.output_dir / "native_report.json"),
               "native_outputs_sha256": file_sha256(args.output_dir / "native_outputs.json"), **metrics}
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(previous_run, {"status": "complete", "started_at": started_at,
                              "completed_at": summary["completed_at"], "run_identity": identity, "arguments": arguments})
    print(f"COMPLETE {args.model_label}: strict_prompt={metrics['strict_prompt_accuracy']:.6f} "
          f"loose_prompt={metrics['loose_prompt_accuracy']:.6f} "
          f"strict_instruction={metrics['strict_instruction_accuracy']:.6f} "
          f"loose_instruction={metrics['loose_instruction_accuracy']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"EVALUATION FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
