"""Shared prompt and answer protocol for the opt-in non-thinking evaluations.

The default path preserves the legacy prompt and parser.  ``--no_think`` is
supported for local Qwen3/Qwen3.5 checkpoints and passes the native
``enable_thinking=False`` template argument; it is not a textual soft switch.
This module imports neither Transformers nor vLLM and changes no sampling
parameters, token budgets, model weights, or reward values.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import string
from typing import Callable, Iterable


PROTOCOL_VERSION = "answer_body_nothink_v2"
ANSWER_ONLY_INSTRUCTION = (
    "Follow the task requirements. Put your complete final response inside "
    "exactly one <answer>...</answer> block. Do not write anything outside "
    "that block."
)

_QWEN3_TYPES = frozenset({"qwen3", "qwen3_moe"})
_QWEN35_TYPES = frozenset(
    {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
)
_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_STRICT_ANSWER_FORMAT = re.compile(r"\A\s*<answer>(.*?)</answer>\s*\Z", re.DOTALL)
_ANSWER_TAG = re.compile(r"</?answer\b", re.IGNORECASE)
_THINK_TAG = re.compile(r"</?think\b", re.IGNORECASE)
_CHOICE = re.compile(r"(?:([A-Za-z])\.?|\(\s*([A-Za-z])\s*\)|\[\s*([A-Za-z])\s*\])")


def add_protocol_args(parser):
    """Add opt-in protocol flags without changing any legacy defaults."""
    parser.add_argument(
        "--no_think", "--no-think", dest="no_think", action="store_true",
        help="Use Qwen3/Qwen3.5's native enable_thinking=False and one final-answer block.",
    )
    parser.add_argument(
        "--model_dir", "--model-dir", dest="model_dir", default=None,
        help="Local model directory override; otherwise retain the script's legacy model path.",
    )
    return parser


def model_family(model_type: str | None) -> str | None:
    """Identify supported families from config.model_type, not directory names."""
    if not isinstance(model_type, str):
        return None
    normalized = model_type.strip().lower()
    if normalized in _QWEN3_TYPES:
        return "qwen3"
    if normalized in _QWEN35_TYPES:
        return "qwen3.5"
    return None


def read_model_type(model_dir: str | Path) -> str | None:
    """Read a local config; a missing config remains acceptable for legacy use."""
    path = Path(model_dir).expanduser()
    config_path = path / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Model config must be an object: {config_path}")
    value = config.get("model_type")
    return value if isinstance(value, str) else None


def extract_single_answer(text: str | None) -> str | None:
    r"""Return the unique complete nonempty answer body, preserving whitespace.

    Text outside that block is ignored for scoring. Multiple, nested, extra
    or incomplete answer tags remain ambiguous and are rejected, as are think
    tags inside the body. Nothing is repaired and no untagged answer is inferred.
    Use ``is_strict_answer_format`` to measure the separate no-extra-text rule.

    The following two short cases reproduce the structure of the observed
    Qwen3 Logic and IFEval responses (717 and 55 such responses respectively):

    >>> extract_single_answer('To determine the education, analyze the clues.\n<answer>C</answer>')
    'C'
    >>> haiku = "\nMom's a superhero,  \nmother's a wise old tree—  \nboth make me feel free!  \n"
    >>> extract_single_answer('Write a funny haiku about moms.\n\n<answer>' + haiku + '</answer>') == haiku
    True
    >>> extract_single_answer('prefix <answer>  exact text\n</answer> suffix')
    '  exact text\n'
    >>> invalid = [None, '', '2', '<answer>A', 'A</answer>',
    ...            '<answer></answer>', '<answer> \n </answer>',
    ...            '<answer>A</answer><answer>B</answer>',
    ...            '<answer><answer>A</answer></answer>',
    ...            '<answer>A</answer><answer>',
    ...            '</answer><answer>A</answer>',
    ...            '<answer>A</answer></answer>',
    ...            '<answer>A</answer><ANSWER',
    ...            '<answer>A</answer></ANSWER',
    ...            '<answer><think>draft</think>A</answer>']
    >>> all(extract_single_answer(text) is None for text in invalid)
    True
    """
    if not isinstance(text, str):
        return None
    # Count starts, not just well-formed closed blocks: a valid block followed
    # by a dangling opener/closer must not silently become a valid response.
    if len(list(_ANSWER_TAG.finditer(text))) != 2:
        return None
    match = _ANSWER_BLOCK.search(text)
    if match is None:
        return None
    answer = match.group(1)
    if not answer.strip() or _ANSWER_TAG.search(answer) or _THINK_TAG.search(answer):
        return None
    return answer


def is_strict_answer_format(text: str | None) -> bool:
    r"""Report the no-extra-text format rule separately from body-only scoring.

    >>> is_strict_answer_format(' \n<answer>C</answer>\n')
    True
    >>> is_strict_answer_format('Explanation.\n<answer>C</answer>')
    False
    >>> is_strict_answer_format('<answer>C</answer> More text.')
    False
    >>> is_strict_answer_format('<answer>A</answer><answer>B</answer>')
    False
    >>> is_strict_answer_format(None)
    False
    """
    return extract_single_answer(text) is not None and _STRICT_ANSWER_FORMAT.fullmatch(text) is not None


def _choice_labels(allowed_labels: Iterable[str]) -> frozenset[str]:
    labels = frozenset(label.strip().upper() for label in allowed_labels)
    if not labels or any(len(label) != 1 or label not in string.ascii_uppercase for label in labels):
        raise ValueError("allowed_labels must contain one or more single ASCII letters")
    return labels


def extract_choice_label(
    answer: str | None, allowed_labels: Iterable[str] = string.ascii_uppercase,
) -> str | None:
    """Accept a whole answer label, never a letter found in answer prose.

    ``C``, ``c``, ``C.``, ``(C)``, and ``[C]`` identify the same unambiguous
    label. Option text, multiple labels, and XML fragments are rejected.

    >>> extract_choice_label(' (c) ', 'ABC')
    'C'
    >>> extract_choice_label('D', 'ABC') is None
    True
    >>> extract_choice_label('Answer: C', 'ABC') is None
    True
    >>> extract_choice_label('A or C', 'ABC') is None
    True
    """
    labels = _choice_labels(allowed_labels)
    if not isinstance(answer, str):
        return None
    match = _CHOICE.fullmatch(answer.strip())
    if match is None:
        return None
    label = next(value for value in match.groups() if value is not None).upper()
    return label if label in labels else None


@dataclass(frozen=True)
class ModelProtocol:
    model_dir: Path
    model_type: str | None
    answer_only: bool = False

    @classmethod
    def from_model_path(
        cls, model_dir: str | Path, *, no_think: bool = False,
    ) -> "ModelProtocol":
        path = Path(model_dir).expanduser()
        kind = read_model_type(path)
        if no_think and model_family(kind) is None:
            raise ValueError(
                "--no_think requires a local Qwen3/Qwen3.5 config.model_type; "
                f"found {kind!r} at {path / 'config.json'}"
            )
        return cls(model_dir=path, model_type=kind, answer_only=bool(no_think))

    @property
    def no_think(self) -> bool:
        return self.answer_only

    def render_prompt(
        self, tokenizer, task_prompt: str, legacy_suffix: str = "",
        system_prompt: str | None = "You're a helpful assistant. ",
    ) -> str:
        """Render a task with its model's template; retain legacy text exactly."""
        if self.answer_only:
            user_prompt = task_prompt.rstrip() + "\n" + ANSWER_ONLY_INSTRUCTION
        else:
            user_prompt = task_prompt + legacy_suffix
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        kwargs = {"add_generation_prompt": True, "tokenize": False}
        if self.answer_only:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)

    def extract_answer(
        self, text: str | None,
        legacy_extractor: Callable[[str | None], str | None] | None = None,
    ) -> str | None:
        if self.answer_only:
            return extract_single_answer(text)
        if legacy_extractor is not None:
            return legacy_extractor(text)
        if not isinstance(text, str):
            return None
        match = re.search(r"<answer>(.*)</answer>", text, re.DOTALL)
        return match.group(1) if match else None

    def extract_choice(
        self, answer: str | None,
        allowed_labels: Iterable[str] = string.ascii_uppercase,
        legacy_extractor: Callable[[str | None], str | None] | None = None,
    ) -> str | None:
        if self.answer_only:
            return extract_choice_label(answer, allowed_labels)
        if legacy_extractor is not None:
            return legacy_extractor(answer)
        if not isinstance(answer, str):
            return None
        labels = "".join(sorted(_choice_labels(allowed_labels)))
        match = re.search(rf"(?<![A-Za-z])([{labels}])(\.|(?:\s|$))", answer.strip())
        return match.group(1) if match else None

    def to_metadata(self) -> dict:
        return {
            "protocol_version": PROTOCOL_VERSION if self.answer_only else "legacy",
            "model_dir": str(self.model_dir),
            "model_type": self.model_type,
            "model_family": model_family(self.model_type),
            "no_think": self.answer_only,
            "enable_thinking": False if self.answer_only else None,
            "enable_thinking_kwarg_passed": self.answer_only,
            "answer_format": "one_complete_answer_block" if self.answer_only else "legacy",
            "answer_body_whitespace_preserved": True,
            "scoring_scope": "answer_body_only" if self.answer_only else "legacy",
            "outside_answer_text_ignored_for_scoring": True if self.answer_only else None,
            "strict_answer_format_is_separate": True if self.answer_only else None,
        }
