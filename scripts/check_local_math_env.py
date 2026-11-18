#!/usr/bin/env python3
"""Check one local MathEnv gold answer on CPU, without API calls or a model.

This is an environment/scorer check, not an LLM evaluation. Run it with the
configured ragen2 interpreter; the editable package must resolve to this repo.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check(args: argparse.Namespace) -> dict:
    # Only this checker process is affected. Prevent an eager environment import
    # from reading the project's credential file or creating a CUDA context.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['RAGEN_API_KEYS_PATH'] = os.devnull
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.dont_write_bytecode = True

    import torch
    import ragen

    project_dir = Path(__file__).resolve().parents[1]
    try:
        Path(ragen.__file__).resolve().relative_to(project_dir / 'ragen')
    except ValueError as exc:
        raise RuntimeError(f'ragen resolved outside this project: {ragen.__file__}') from exc

    from ragen.env.math_lv3to5.config import MathEnvConfig
    from ragen.env.math_lv3to5.env import MathEnv

    module = importlib.import_module('ragen.env.math_lv3to5.env')
    source_path = Path(module.__file__).resolve()
    try:
        source_path.relative_to(project_dir)
    except ValueError as exc:
        raise RuntimeError(f'MathEnv resolved outside this project: {source_path}') from exc

    config = MathEnvConfig()
    if args.data_root is not None:
        config.data_path = str(args.data_root.expanduser().resolve())
    config.mode = args.split
    config.seed = args.seed
    dataset = (Path(config.data_path) / 'reasoning' / config.data_files / f'{config.mode}.parquet')
    if not dataset.is_file():
        raise FileNotFoundError(f'Local MathEnv data is missing: {dataset}')

    env = MathEnv(config)
    env.reset(args.seed)
    question = env.render()
    if not isinstance(question, str) or not question.strip():
        raise AssertionError('MathEnv.render() did not return a nonempty question.')
    gold = env.question0['extra_info']['answer']
    if not isinstance(gold, str):
        raise TypeError('The dataset gold answer must be a string for the existing MathEnv protocol.')
    _, reward, done, info = env.step(gold)
    if float(reward) != 1.0 or done is not True or not info.get('success') or not info.get('action_is_valid'):
        raise AssertionError(f'Gold answer did not pass the existing MathEnv scorer: reward={reward}, info={info}')
    if torch.cuda.is_initialized():
        raise AssertionError('The local MathEnv check unexpectedly initialized CUDA.')

    return {
        'passed': True,
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'python': sys.executable,
        'prefix': sys.prefix,
        'cwd': os.getcwd(),
        'ragen_source': ragen.__file__,
        'math_env_source': str(source_path),
        'math_env_source_sha256': file_sha256(source_path),
        'versions': {name: metadata.version(name) for name in ('torch', 'datasets', 'math-verify')},
        'dataset': {
            'path': str(dataset.resolve()),
            'sha256': file_sha256(dataset),
            'rows': len(env.data_train),
            'split': args.split,
            'seed': args.seed,
        },
        'question_chars': len(question),
        'question_sha256': hashlib.sha256(question.encode()).hexdigest(),
        'gold_score': float(reward),
        'done': bool(done),
        'success': bool(info['success']),
        'action_is_valid': bool(info['action_is_valid']),
        'cuda_initialized': False,
        'api_calls': 0,
        'limit': 'One local gold-answer environment/scorer check; no model generation or training.',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, help='Override MathEnvConfig.data_path for this check only.')
    parser.add_argument('--split', choices=('train', 'test'), default='test')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output', type=Path, help='Optional JSON evidence file.')
    args = parser.parse_args()
    try:
        report = check(args)
        code = 0
    except Exception as exc:
        report = {'passed': False, 'python': sys.executable, 'prefix': sys.prefix,
                  'error_type': type(exc).__name__, 'error': str(exc)}
        code = 1
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end='')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
