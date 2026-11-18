# RAGEN2 Qwen3 evaluation

Run from the server's RAGEN2 checkout. For a long evaluation, use tmux and
the project's established GPU handoff workflow. The launcher uses the RAGEN2
Conda Python and accepts a local Hugging Face base model or exported checkpoint.
It stops on the first benchmark error and writes a separate directory per test.

```bash
bash reason_test/run_qwen3.sh MODEL_DIR GPU NEW_RUN_DIR --max-samples 2 math nash ifeval logic
```

Omit the test names for all ten tests. Omit `--max-samples` for a full run.
IFEval uses the same scorer as RAGEN. The launcher exposes it and its pinned
dependencies through `IFEVAL_SOURCE` and `IFEVAL_DEPS`; RAGEN2's training
environment is independent. CommonGen and the two opponent-driven game tests
use the existing configured judge and opponents.

The local checkpoint tokenizer renders native `enable_thinking=False`.
The RAGEN2 training parser accepts a single nonempty terminal
`<answer>...</answer>` block and preserves the full generated response in
game history. Benchmark scorers consume the extracted body. Games and Math
keep their training observation layout and response-length instruction.
IFEval keeps the original benchmark instruction as the observation and runs
both native strict and loose scoring against the extracted body.

Default sampling matches the RAGEN2 Qwen3 validation profile: temperature
0.7, top-p 0.8, top-k 20, min-p 0, presence penalty 0. Generation uses 600
tokens except IFEval, where prompts requiring up to 1,200 words motivate a
2,500-token cap. `--max-tokens 600` restores the training generation budget
there. Match budgets when comparing models. The Qwen3 technical report's
non-thinking settings use the same temperature and top-p/top-k and a
presence penalty of 1.5; RAGEN2's training validation uses 0.

Run artifacts include the effective sampling settings, rendered inputs,
original completion texts and token IDs, finish reasons, and benchmark scores.
