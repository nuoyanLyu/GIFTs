#!/usr/bin/env python3
"""CPU-only verification of ragen2's independent environment and import paths.

Run this file with ragen2's Python from outside the repository (e.g. /tmp) to
check editable installs independently of Python's current-directory lookup.
No model weights, API credentials, Ray cluster or CUDA context are used.
Only --output, when provided, writes a JSON report.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import site
import sys
import traceback
from urllib.parse import unquote, urlparse

EXPECTED_VERSIONS = {
    'verl': '0.6.1', 'torch': '2.8.0', 'transformers': '4.56.1',
    'vllm': '0.11.0', 'ray': '2.56.0', 'tensordict': '0.10.0',
    'numpy': '1.26.4', 'scipy': '1.14.1', 'flash-attn': '2.8.1',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def within(path, parent):
    return path == parent or parent in path.parents


def local_url(value):
    parsed = urlparse(value)
    if parsed.scheme == 'file' and parsed.netloc in ('', 'localhost'):
        return Path(unquote(parsed.path)).resolve()
    return None


def inspect_paths(root, prefix):
    """Inspect configuration as data; never execute another editable finder."""
    forbidden = [root.parent / name for name in ('GIFT-LLM', 'EnvFactory', 'RAGEN')]
    forbidden = [path.resolve() for path in forbidden if path.resolve() != root]

    def foreign(path):
        path = Path(path).resolve()
        if within(path, root) or within(path, prefix):
            return False
        return (any(within(path, candidate) for candidate in forbidden)
                or 'site-packages' in path.parts)

    bad = []
    for value in sys.path:
        if value and foreign(value):
            bad.append({'kind': 'sys.path', 'path': value})
    for value in os.environ.get('PYTHONPATH', '').split(os.pathsep):
        if value:
            path = Path(value).resolve()
            if not (within(path, root) or within(path, prefix)):
                bad.append({'kind': 'external PYTHONPATH', 'path': str(path)})
    site_dirs = [Path(value).resolve() for value in site.getsitepackages()]
    require(site_dirs and all(within(path, prefix) for path in site_dirs),
            f'Inherited site-packages detected: {site_dirs}')
    pointer_files = []
    for folder in site_dirs:
        for path in sorted(list(folder.glob('*.pth')) + list(folder.glob('*.egg-link'))):
            pointer_files.append(str(path))
            for line in path.read_text(errors='replace').splitlines():
                value = line.strip()
                if not value or value.startswith('#') or value.startswith('import '):
                    continue
                target = Path(value)
                target = target if target.is_absolute() else folder / target
                target = target.resolve()
                if not (within(target, root) or within(target, prefix)):
                    bad.append({'kind': path.name, 'path': str(target.resolve())})
        for path in folder.glob('__editable__*_finder.py'):
            for node in ast.parse(path.read_text()).body:
                target = None
                if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'MAPPING' for x in node.targets):
                    target = node.value
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'MAPPING':
                    target = node.value
                if target is not None:
                    for value in ast.literal_eval(target).values():
                        target_path = Path(value).resolve()
                        if not (within(target_path, root) or within(target_path, prefix)):
                            bad.append({'kind': path.name, 'path': str(value)})

    installed = {}
    editable_records = []
    for distribution in metadata.distributions(path=[str(path) for path in site_dirs]):
        name = distribution.metadata.get('Name', '').lower().replace('_', '-')
        raw = distribution.read_text('direct_url.json')
        info = json.loads(raw) if raw else {}
        target = local_url(info.get('url', ''))
        editable = info.get('dir_info', {}).get('editable') is True
        if editable:
            # A clone must not keep any live external source dependency.
            if target is None or not (within(target, root) or within(target, prefix)):
                bad.append({'kind': 'external editable', 'package': name,
                            'path': str(target) if target is not None else 'nonlocal editable URL'})
            editable_records.append({'package': name, 'path': str(target)})
        if name in ('ragen', 'verl'):
            installed[name] = {'version': distribution.version, 'editable': editable,
                               'path': str(target) if target is not None else None,
                               'metadata_path': str(distribution._path)}
    require(not bad, 'Foreign package paths remain: ' + json.dumps(bad))
    for name, expected in (('ragen', root), ('verl', (root / 'verl').resolve())):
        require(name in installed, f'{name} is not registered inside the ragen2 site-packages')
        record = installed[name]
        require(record['editable'] and record['path'] == str(expected),
                f'{name} editable install does not point to this checkout: {record}')
    return {'site_packages': [str(path) for path in site_dirs],
            'pointer_files_checked': pointer_files, 'editable_records': editable_records,
            'project_distributions': installed}


def inspect_environment(args, report):
    root = args.project_root.resolve()
    prefix = Path(sys.prefix).resolve()
    require(prefix.name == 'ragen2' and (prefix / 'conda-meta').is_dir(),
            f'Expected independent conda environment ragen2, got {prefix}')
    require(Path(sys.base_prefix).resolve() == prefix, 'A Python venv is active instead of the conda environment')
    require(sys.version_info[:3] == (3, 12, 13), f'Unexpected Python version: {sys.version}')
    require(site.ENABLE_USER_SITE is False, 'Set PYTHONNOUSERSITE=1 before starting Python')
    report.update(python=sys.executable, prefix=str(prefix), cwd=str(Path.cwd()),
                  project_root=str(root), python_version=sys.version.split()[0],
                  user_site_enabled=site.ENABLE_USER_SITE, pythonpath=os.getenv('PYTHONPATH'))
    report['isolation'] = inspect_paths(root, prefix)
    report['sources'] = {}
    for name, expected in (('ragen', root / 'ragen/__init__.py'),
                           ('verl', root / 'verl/verl/__init__.py')):
        spec = importlib.util.find_spec(name)
        require(spec is not None and spec.origin is not None, f'{name} cannot be imported')
        origin = Path(spec.origin).resolve()
        require(origin == expected.resolve(), f'{name} resolves to {origin}, expected {expected.resolve()}')
        report['sources'][name] = str(origin)
    from packaging.version import Version
    report['versions'] = {name: metadata.version(name) for name in EXPECTED_VERSIONS}
    for name, expected in EXPECTED_VERSIONS.items():
        require(Version(report['versions'][name]).public == Version(expected).public,
                f'{name}: expected {expected}, installed {report["versions"][name]}')

    # These third-party imports verify the installed binary linkage without generation.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    modules = {}
    for name in ('torch', 'transformers', 'vllm', 'ray', 'tensordict', 'flash_attn_2_cuda'):
        module = importlib.import_module(name)
        origin = Path(module.__file__).resolve()
        require(within(origin, prefix), f'{name} imports outside the conda prefix: {origin}')
        modules[name] = str(origin)
    torch = importlib.import_module('torch')
    require(torch.version.cuda == '12.8', f'Expected torch CUDA 12.8 build, got {torch.version.cuda}')
    report['torch_cuda_build'] = torch.version.cuda
    report['backend'] = report['sources']['verl']
    report['binary_module_paths'] = modules
    report['versions']['flash-attn'] = metadata.version('flash-attn')

    # Compose the same Hydra configuration without importing game/API modules.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from ragen.utils import register_resolvers
    register_resolvers()
    overrides = list(args.override)
    if args.model_path:
        overrides.append(f'model_path={args.model_path}')
    with initialize_config_dir(config_dir=str(root / 'config'), version_base=None):
        config = compose(config_name=args.config_name, overrides=overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    require(config.data.apply_chat_template_kwargs.enable_thinking is False,
            'The configured native Qwen3 mode is not explicitly False')
    require(config.agent_proxy.enable_think is False, 'RAGEN format-thinking must be False')
    report['config'] = {'name': args.config_name, 'overrides': overrides,
                        'model_path': config.model_path,
                        'algorithm': config.algorithm.adv_estimator,
                        'rollout_mode': config.actor_rollout_ref.rollout.mode,
                        'sha256': hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()}
    report['native_thinking'] = None
    if not args.skip_model_check:
        from transformers import AutoConfig, AutoTokenizer
        model = Path(config.model_path).expanduser()
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        model_config = AutoConfig.from_pretrained(model, local_files_only=True)
        require('Qwen3ForCausalLM' in model_config.architectures, f'Unexpected architecture: {model_config.architectures}')
        messages = [{'role': 'system', 'content': "You're a helpful assistant. "},
                    {'role': 'user', 'content': 'Output exactly <answer>OK</answer>.'}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
        suffix = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        require(rendered.endswith(suffix), 'Native Qwen3 non-thinking prefix differs from the audited protocol')
        report['native_thinking'] = False
        report['model'] = {'path': str(model), 'architecture': model_config.architectures,
                           'native_false_suffix': suffix, 'tokenizer_class': type(tokenizer).__name__,
                           'prompt_token_count': len(tokenizer.encode(rendered, add_special_tokens=False))}
    report['model_checked'] = not args.skip_model_check
    report['cuda_initialized'] = torch.cuda.is_initialized()
    require(report['cuda_initialized'] is False, 'CPU preflight unexpectedly initialized CUDA')
    report['passed'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config-name', default='qwen3_4b_nothink_smoke')
    parser.add_argument('--model-path')
    parser.add_argument('--override', action='append', default=[], help='Hydra override; repeat for each value')
    parser.add_argument('--skip-model-check', action='store_true', help='Only for CI without local model files')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = {'passed': False, 'scope': 'CPU environment, editable-source and native-template checks'}
    try:
        inspect_environment(args, report)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end='')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
