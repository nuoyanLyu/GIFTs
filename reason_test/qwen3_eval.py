"""The RAGEN2 Qwen3 non-thinking training protocol for downstream evaluation."""

from argparse import ArgumentTypeError
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


SYSTEM = "You're a helpful assistant. "
MODEL_LEN = 6000
RESPONSE_TOKENS = 600
SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise ArgumentTypeError("expected a positive integer")
    return number


def add_qwen3_args(parser):
    parser.add_argument("--qwen3", action="store_true", help="Select the native Qwen3 non-thinking training protocol")
    parser.add_argument("--max_samples", type=positive_int, help="Limit the number of benchmark items for validation")
    parser.add_argument("--max_tokens", type=positive_int, help="Override the Qwen3 generation budget (training default: 600)")
    parser.add_argument("--seed", type=int, default=42, help="Qwen3 evaluation seed")
    parser.add_argument("--output_dir", type=Path, help="Output directory for evaluation artifacts")


def is_qwen3(model_path, args):
    config = Path(model_path) / "config.json"
    if args.qwen3:
        return True
    if config.is_file():
        with config.open() as handle:
            return json.load(handle).get("model_type") in {"qwen3", "qwen3_moe"}
    return False


class Qwen3Eval:
    def __init__(self, tokenizer, args):
        from omegaconf import OmegaConf
        from ragen.llm_agent.ctx_manager import (
            ContextManager, FINAL_ANSWER_INSTRUCTION, FINAL_ANSWER_WITH_PREFIX,
        )

        if "enable_thinking" not in (tokenizer.chat_template or ""):
            raise ValueError("Qwen3 evaluation requires an actual tokenizer with native enable_thinking support")
        self.tokenizer = tokenizer
        self.seed = args.seed
        self.override_max_tokens = args.max_tokens
        self.max_tokens = self.override_max_tokens or RESPONSE_TOKENS
        self.instruction = FINAL_ANSWER_INSTRUCTION
        self.protocol_name = FINAL_ANSWER_WITH_PREFIX
        self.context = ContextManager(
            OmegaConf.create({
                "data": {"apply_chat_template_kwargs": {"enable_thinking": False}},
                "agent_proxy": {
                    "enable_think": False,
                    "response_protocol": FINAL_ANSWER_WITH_PREFIX,
                    "action_sep": "||",
                    "max_actions_per_turn": 5,
                },
                "es_manager": {"val": {"group_size": 1, "env_configs": {"n_groups": [], "tags": []}}},
                "custom_envs": {},
            }),
            tokenizer,
            mode="val",
        )
        benchmark = Path(sys.argv[0]).stem
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.output_dir = args.output_dir or Path(__file__).parent / "results" / "qwen3" / f"{benchmark}-{timestamp}"
        self.trace_path = self.output_dir / f"{benchmark}.generations.jsonl"

    def messages(self, observation, system=None, max_actions=1, *, max_tokens=RESPONSE_TOKENS, instruction=""):
        # Same turn layout as ContextManager.get_lm_inputs, for single-turn benchmarks.
        return [
            {"role": "system", "content": SYSTEM if system is None else system},
            {"role": "user", "content": (
                f"{instruction}\nTurn 1:\nState:\n{observation}\n"
                f"You have {max_actions} actions left.\n"
                f"{self.instruction}\nMax response length: {max_tokens} words (tokens).\n"
            )},
        ]

    def render(self, messages):
        return self.context._render_chat_template(messages, add_generation_prompt=True)

    def trajectory(self, history, *, max_tokens=200, instruction="", env_type=None):
        self.context.prefix_lookup[0] = instruction
        self.context.env_config_lookup[0] = {"max_tokens": max_tokens, "env_type": env_type}
        result = self.context.get_lm_inputs(
            [{"env_id": 0, "group_id": 0, "tag": env_type, "history": history}],
            prepare_for_update=False,
        )
        return {"prompt_token_ids": result.batch["input_ids"][0].tolist()}

    def extract(self, raw):
        # The training math branch returns the whole answer body without splitting it on ||.
        _, actions = self.context._parse_response(raw, env_type="math_lv3to5")
        return actions[0] if actions else None

    def actions(self, raw, env_type=None):
        return self.context._parse_response(raw, env_type=env_type)[1]

    def response_text(self, completion):
        return self.tokenizer.decode(completion.token_ids, skip_special_tokens=True)

    def body_token_ids(self, completion):
        ids = list(completion.token_ids)
        stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        while ids and ids[-1] in stop_ids:
            ids.pop()
        return ids

    def sampling_params(self, max_tokens=RESPONSE_TOKENS):
        from vllm import SamplingParams

        self.max_tokens = self.override_max_tokens or max_tokens
        return SamplingParams(max_tokens=self.max_tokens, **SAMPLING)

    def llm_kwargs(self, max_model_len=MODEL_LEN):
        self.max_model_len = max_model_len
        return {
            "max_model_len": max_model_len,
            # Preserve both native Qwen3 EOS IDs. Explicit SamplingParams control decoding.
            "generation_config": "auto",
            "seed": self.seed,
        }

    def to_metadata(self):
        return {
            "response_protocol": self.protocol_name,
            "enable_thinking": False,
            "max_tokens": self.max_tokens,
            "max_model_len": getattr(self, "max_model_len", MODEL_LEN),
            **SAMPLING,
            "seed": self.seed,
            "generation_config": "auto",
        }

    def record(self, completion, *, prompt=None, **fields):
        if not self.trace_path.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = self.trace_path.with_suffix(".metadata.json")
            metadata_path.write_text(json.dumps(self.to_metadata(), indent=2) + "\n")
            print(f"Qwen3 generation trace: {self.trace_path}", flush=True)
        row = {
            **fields,
            "prompt": prompt,
            "raw": completion.text,
            "decoded_for_scoring": self.response_text(completion),
            "generated_token_ids": list(completion.token_ids),
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
        }
        with self.trace_path.open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
