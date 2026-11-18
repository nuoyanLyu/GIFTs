from copy import deepcopy
from types import SimpleNamespace
import unittest

import torch
from transformers import AutoTokenizer
from verl import DataProto

from ragen.llm_agent.ctx_manager import ContextManager

MODEL = "/data1/model/Qwen3-4B"


def manager(tokenizer, samples):
    context = ContextManager.__new__(ContextManager)
    context.tokenizer = tokenizer
    context.config = SimpleNamespace(
        data={"apply_chat_template_kwargs": {"enable_thinking": False}},
        agent_proxy=SimpleNamespace(
            enable_think=False, response_protocol="final_answer_with_prefix",
            max_actions_per_turn=2, max_context_window=-1, use_turn_scores=False,
            reward_normalization=SimpleNamespace(grouping="state", method="identity"),
        ),
        enable_response_mask=True,
    )
    context.action_sep = "||"
    context.special_token_list = [
        "<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>",
    ]
    context.prefix_lookup = {
        index: sample.get("instruction", "") for index, sample in enumerate(samples)
    }
    context.env_config_lookup = {
        index: {"max_tokens": sample.get("max_tokens", 200), "env_type": sample["kind"]}
        for index, sample in enumerate(samples)
    }
    return context


def states_for(samples):
    return [{
        "env_id": index, "group_id": index, "tag": sample["kind"], "metrics": {},
        "history": [{"state": sample["question"], "actions_left": sample.get("actions_left", 1)}],
    } for index, sample in enumerate(samples)]


class FinalAnswerProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        cls.samples = [
            {"question": "Compute the result.", "kind": "math_lv3to5"},
            {"question": "Choose an action.", "kind": "game"},
        ]

    def test_valid_answers(self):
        context = manager(self.tokenizer, self.samples)
        examples = [
            ("<answer>13</answer>", ["13"]),
            ("Reasoning\n<answer> 13 </answer>\n  ", ["13"]),
            ("Explanation.\n<answer>\\frac{1}{2}\n+</answer>", ["\\frac{1}{2}\n+"]),
            ("<answer>||x||</answer>", ["||x||"]),
            ("前置说明。<answer>13</answer>", ["13"]),
        ]
        for response, expected in examples:
            with self.subTest(response=response):
                self.assertEqual(context._parse_response(response, "math_lv3to5"), (response, expected))

    def test_invalid_answers_keep_original_response(self):
        context = manager(self.tokenizer, self.samples)
        examples = [
            "", "13", "<answer> </answer>", "<answer>13", "</answer><answer>",
            "<answer>1</answer> then <answer>13</answer>", "<answer><answer>13</answer></answer>",
            "<answer>13</answer> trailing explanation", "<answer>13</answer>\n</answer>",
            "<answer><think>13</think></answer>", "<Answer>13</Answer>",
        ]
        for response in examples:
            with self.subTest(response=response):
                self.assertEqual(context._parse_response(response, "math_lv3to5"), (response, []))

    def test_game_action_limits_do_not_rewrite_response(self):
        context = manager(self.tokenizer, self.samples)
        response = "The first two are legal. <answer> left || right || up </answer>"
        self.assertEqual(context._parse_response(response, "game"), (response, ["left", "right"]))

    def test_legacy_parser_continues_to_canonicalize(self):
        context = manager(self.tokenizer, self.samples)
        context.config.agent_proxy.response_protocol = "legacy"
        response = "Reasoning. <answer>13</answer>"
        self.assertEqual(context._parse_response(response), ("<answer>13</answer>", ["13"]))
        context.config.agent_proxy.enable_think = True
        self.assertEqual(context._parse_response("<think> reason </think><answer>13</answer>"),
                         ("<think>reason</think><answer>13</answer>", ["13"]))

    def test_prompt_to_parse_to_batch(self):
        context = manager(self.tokenizer, self.samples)
        states = states_for(self.samples)
        rollout = context.get_lm_inputs(deepcopy(states), prepare_for_update=False)
        native_suffix = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        starting_prompt = self.tokenizer.decode(rollout.batch["input_ids"][0][rollout.batch["attention_mask"][0].bool()])
        self.assertTrue(starting_prompt.endswith(native_suffix))
        for messages in rollout.non_tensor_batch["messages_list"]:
            user = messages[-1]["content"]
            self.assertIn("Put only the final answer inside that block.", user)
            self.assertNotIn("with no extra text", user)
            self.assertNotIn("<think>", user)

        replies = ["The norm is the result.\n<answer>||x||</answer>", "Move left. <answer>left</answer>"]
        output = DataProto()
        output.non_tensor_batch = {**rollout.non_tensor_batch, "response_texts": replies}
        parsed = context.get_env_inputs(output)
        self.assertEqual([item["actions"] for item in parsed], [["||x||"], ["left"]])
        for state, item in zip(states, parsed):
            state["history"][0].update(item, reward=1.0)
            state["history"].append({"state": "Next state", "actions_left": 0})
        batch = context.formulate_rollouts(deepcopy(states))
        followup = context.get_lm_inputs(deepcopy(states), prepare_for_update=False)
        for index, reply in enumerate(replies):
            self.assertEqual(batch.non_tensor_batch["messages_list"][index][-1]["content"], reply)
            self.assertEqual(followup.non_tensor_batch["messages_list"][index][-2]["content"], reply)
            masked = self.tokenizer.decode(batch.batch["responses"][index][batch.batch["loss_mask"][index]])
            self.assertIn(reply, masked)
            self.assertEqual(batch.batch["original_rm_scores"][index].sum().item(), 1.0)

    def test_sampled_token_boundary_preserved_in_batch_and_next_turn(self):
        context = manager(self.tokenizer, self.samples[:1])
        states = states_for(self.samples[:1])
        prompt = context.get_lm_inputs(deepcopy(states), prepare_for_update=False)
        encode = self.tokenizer.encode
        response = "Reasoning. <answer>13</answer>"
        sampled_ids = encode("R") + encode("easoning. <answer>13</answer>")
        self.assertEqual(self.tokenizer.decode(sampled_ids), response)
        self.assertNotEqual(sampled_ids, encode(response))
        output = DataProto()
        output.batch = prompt.batch.select("responses")
        output.batch["responses"] = torch.tensor([
            sampled_ids + [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        ])
        output.non_tensor_batch = prompt.non_tensor_batch
        parsed = context.get_env_inputs(output)[0]
        self.assertEqual(parsed["generated_body_token_ids"], sampled_ids)
        self.assertEqual(parsed["actions"], ["13"])
        states[0]["history"][0].update(parsed, reward=1.0)
        states[0]["history"].append({"state": "Next state", "actions_left": 1})
        start_ids = prompt.batch["input_ids"][0].tolist()
        target = start_ids + sampled_ids
        batch = context.formulate_rollouts(deepcopy(states))
        followup = context.get_lm_inputs(deepcopy(states), prepare_for_update=False)
        self.assertEqual(batch.batch["input_ids"][0, :len(target)].tolist(), target)
        self.assertEqual(followup.batch["input_ids"][0, :len(target)].tolist(), target)
        mask = batch.batch["loss_mask"][0, len(start_ids) - 1:len(target) - 1]
        self.assertTrue(mask.all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
