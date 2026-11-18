#!/usr/bin/env bash
# Create/configure only the requested independent environment. Never falls back
# to the active Python, factory, gift1 or the former RAGEN2 .venv.
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
env_name=ragen2
clone_source=''
offline_clone=0
configure_existing=0
flash_attn_wheel=''
requirements_file="$project_dir/requirements-ragen2.txt"
expected_verl_commit=d62da4950573d7a4b7ef2362337952e7ab59e78d
expected_flash_attn=2.8.1

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup_ragen2_env.sh [options]
  --clone-source NAME_OR_PATH  Copy an existing binary environment
  --offline-clone             Use cached conda packages only (optional)
  --configure-existing        Explicitly configure an already-created target
  --flash-attn-wheel FILE      Local cp312 / torch2.8 / CUDA12 compatible wheel
  --requirements FILE         Use a reviewed full requirements snapshot instead
  --help

Default creation refuses an existing target. --configure-existing never creates
or deletes an environment and must be requested explicitly. It is intended for
the new ragen2 copy created as the first step of this setup.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clone-source) clone_source="${2:?Missing source environment}"; shift 2 ;;
    --offline-clone) offline_clone=1; shift ;;
    --configure-existing) configure_existing=1; shift ;;
    --flash-attn-wheel) flash_attn_wheel="${2:?Missing wheel path}"; shift 2 ;;
    --requirements) requirements_file="${2:?Missing requirements path}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ "$configure_existing" == 1 && -n "$clone_source" ]]; then
  echo '--configure-existing and --clone-source are mutually exclusive.' >&2
  exit 2
fi
if [[ "$offline_clone" == 1 && -z "$clone_source" ]]; then
  echo '--offline-clone requires --clone-source.' >&2
  exit 2
fi
if [[ ! -f "$requirements_file" || ! -f "$project_dir/setup.py" || ! -f "$project_dir/verl/setup.py" ]]; then
  echo 'Run from a complete RAGEN2 checkout with its pinned verl source.' >&2
  exit 2
fi
if [[ -n "$flash_attn_wheel" && ( ! -f "$flash_attn_wheel" || "$flash_attn_wheel" != *.whl ) ]]; then
  echo '--flash-attn-wheel must be an existing local wheel.' >&2
  exit 2
fi
conda_exe="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$conda_exe" ]]; then
  echo 'Conda is required. Set CONDA_EXE to its executable if it is not on PATH.' >&2
  exit 2
fi
bootstrap_python="$(command -v python3 || true)"
if [[ -z "$bootstrap_python" ]]; then
  echo 'python3 is required for checking conda environment metadata.' >&2
  exit 2
fi
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

# A git checkout or the recorded immutable source archive is accepted. Refuse an
# unverified backend instead of silently fetching whichever branch is current.
"$bootstrap_python" - "$project_dir" "$expected_verl_commit" <<'PY'
import hashlib, pathlib, subprocess, sys, tarfile
root = pathlib.Path(sys.argv[1]).resolve()
backend = (root / 'verl').resolve()
expected = sys.argv[2]
probe = subprocess.run(['git', '-C', str(backend), 'rev-parse', '--show-toplevel'], text=True, capture_output=True)
if probe.returncode == 0 and pathlib.Path(probe.stdout.strip()).resolve() == backend:
    actual = subprocess.check_output(['git', '-C', str(backend), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise SystemExit(f'verl commit is {actual}; expected {expected}')
    if subprocess.run(['git', '-C', str(backend), 'diff', '--quiet', 'HEAD', '--']).returncode:
        raise SystemExit('The pinned verl checkout has modified tracked files; review them before environment setup.')
else:
    archive = root / 'migration/verl061.tar.gz'
    if not archive.is_file():
        raise SystemExit('Backend is not a git checkout and its fixed source archive is missing.')
    checked = 0
    with tarfile.open(archive) as source:
        for entry in source:
            if not entry.isfile():
                continue
            parts = pathlib.PurePosixPath(entry.name).parts
            if not parts or parts[0] != f'verl-{expected}':
                raise SystemExit('Source archive has an unexpected backend revision.')
            rel = pathlib.Path(*parts[1:])
            file = backend / rel
            with source.extractfile(entry) as payload:
                wanted = hashlib.sha256(payload.read()).digest()
            if not file.is_file() or hashlib.sha256(file.read_bytes()).digest() != wanted:
                raise SystemExit(f'Backend source differs from its pinned archive: {rel}')
            checked += 1
    if not checked:
        raise SystemExit('Backend source archive is empty.')
print('Verified pinned verl source:', backend)
PY

metadata_file="$(mktemp)"
trap 'rm -f "$metadata_file"' EXIT
"$conda_exe" env list --json > "$metadata_file"
target_prefix="$("$bootstrap_python" - "$metadata_file" "$env_name" <<'PY'
import json, pathlib, sys
matches = [p for p in json.load(open(sys.argv[1]))['envs'] if pathlib.Path(p).name == sys.argv[2]]
if len(matches) > 1:
    raise SystemExit('Multiple environments have this name; use a unique name.')
print(matches[0] if matches else '')
PY
)"
if [[ "$configure_existing" == 1 ]]; then
  if [[ -z "$target_prefix" ]]; then
    echo "Target environment does not exist: $env_name" >&2
    exit 2
  fi
else
  if [[ -n "$target_prefix" ]]; then
    echo "Refusing to overwrite existing environment: $target_prefix" >&2
    exit 2
  fi
  if [[ -n "$clone_source" ]]; then
    clone_args=(create --name "$env_name" --clone "$clone_source" --copy --yes)
    if [[ "$offline_clone" == 1 ]]; then clone_args+=(--offline); fi
    "$conda_exe" "${clone_args[@]}"
  else
    # environment.yml contains no site-specific prefix or editable URL.
    "$conda_exe" env create --name "$env_name" --file "$project_dir/environment.yml"
  fi
  target_prefix="$("$conda_exe" run --name "$env_name" python -s -c 'import sys; print(sys.prefix)')"
fi
target_python="$target_prefix/bin/python"
if [[ ! -x "$target_python" ]]; then
  echo "Target interpreter is missing: $target_python" >&2
  exit 2
fi

# Cloning copies editable .pth pointers as ordinary files. Remove only the new
# prefix's registrations; never uninstall or delete source projects outside it.
"$target_python" -s - "$target_prefix" "$project_dir" "$env_name" <<'PY'
import json, pathlib, shutil, site, sys
prefix = pathlib.Path(sys.argv[1]).resolve()
root = pathlib.Path(sys.argv[2]).resolve()
if (pathlib.Path(sys.prefix).resolve() != prefix or prefix.name != sys.argv[3]
        or not (prefix / 'conda-meta').is_dir()):
    raise SystemExit('Refusing to sanitize anything except the target conda prefix.')
if prefix.name in {'factory', 'gift1', 'base'}:
    raise SystemExit('Refusing to alter a protected source environment.')
changes = []
for directory in site.getsitepackages():
    folder = pathlib.Path(directory).resolve()
    if not folder.is_relative_to(prefix):
        raise SystemExit(f'Target still inherits external site-packages: {folder}')
    for metadata in folder.glob('*.dist-info'):
        direct = metadata / 'direct_url.json'
        if direct.is_file() and json.loads(direct.read_text()).get('dir_info', {}).get('editable'):
            changes.append(str(metadata))
            shutil.rmtree(metadata)
    for pattern in ('__editable__*.pth', '__editable__*_finder.py',
                    '_editable_impl_*.pth', '_editable_impl_*.py', '*.egg-link'):
        for pointer in folder.glob(pattern):
            changes.append(str(pointer))
            pointer.unlink()
    for pointer in folder.glob('*.pth'):
        before = pointer.read_text(errors='replace')
        kept = []
        for line in before.splitlines(True):
            value = line.strip()
            if not value or value.startswith('#') or value.startswith('import '):
                kept.append(line)
                continue
            resolved = (folder / value).resolve()
            if resolved.is_relative_to(prefix):
                kept.append(line)
            else:
                changes.append(f'{pointer}: external path registration removed')
        after = ''.join(kept)
        if after != before:
            temporary = pointer.with_name(pointer.name + '.ragen2-tmp')
            temporary.write_text(after)
            temporary.replace(pointer)
print(json.dumps({'target_prefix': str(prefix), 'removed_target_registrations': changes}, indent=2))
PY

# vLLM 0.11 consumes outlines-core. The former outlines package in the source
# factory copy pins an incompatible outlines-core; it is not part of this RL
# stack. Uninstall only when its metadata is inside the new target prefix.
remove_outlines="$($target_python -s - "$target_prefix" <<'PY'
import importlib.metadata as metadata, pathlib, sys
try:
    package = metadata.distribution('outlines')
except metadata.PackageNotFoundError:
    print('no')
else:
    if not pathlib.Path(package._path).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve()):
        raise SystemExit('Refusing to uninstall outlines outside the new environment.')
    print('yes')
PY
)"
if [[ "$remove_outlines" == yes ]]; then
  "$target_python" -s -m pip uninstall --yes outlines
fi

# The copied desktop OpenCV 4.13 requires NumPy 2, while this training backend
# uses NumPy 1.26. Only headless OpenCV is needed. Removing one OpenCV variant
# can remove shared cv2 files, so reinstall the pinned headless wheel afterwards.
remove_cv="$($target_python -s - "$target_prefix" <<'PY'
import importlib.metadata as metadata, pathlib, sys
try:
    package = metadata.distribution('opencv-python')
except metadata.PackageNotFoundError:
    print('no')
else:
    if not pathlib.Path(package._path).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve()):
        raise SystemExit('Refusing to alter OpenCV outside the target environment.')
    print('yes')
PY
)"
if [[ "$remove_cv" == yes ]]; then
  "$target_python" -s -m pip uninstall --yes opencv-python
  "$target_python" -s -m pip install --no-deps --force-reinstall opencv-python-headless==4.11.0.86
fi

# Register an explicitly supplied binary before requirements resolution, so a
# full installed snapshot containing flash-attn cannot trigger a source build.
if [[ -n "$flash_attn_wheel" ]]; then
  "$target_python" -s -m pip install --no-deps "$flash_attn_wheel"
fi
"$target_python" -s -m pip install --requirement "$requirements_file"

# A clone can contain conda metadata for a newer build tool alongside pip's
# older pinned metadata. Repair its package code as well as the stale record.
"$target_python" -s - "$target_prefix" "$requirements_file" <<'PY'
import importlib.metadata as metadata, pathlib, re, shutil, subprocess, sys
prefix = pathlib.Path(sys.argv[1]).resolve()
site = prefix / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
pins = {}
for line in pathlib.Path(sys.argv[2]).read_text().splitlines():
    match = re.fullmatch(r'([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s;]+)', line.strip())
    if match:
        pins[match[1].lower().replace('_', '-')] = match[2]
records = {}
for package in metadata.distributions(path=[str(site)]):
    name = package.metadata['Name'].lower().replace('_', '-')
    records.setdefault(name, []).append(package)
repair = []
for name, packages in records.items():
    if len(packages) > 1 and name in pins:
        for package in packages:
            path = pathlib.Path(package._path).resolve()
            if not path.is_relative_to(prefix):
                raise SystemExit('Refusing to repair metadata outside the target prefix.')
            if package.version != pins[name]:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        repair.append(f'{name}=={pins[name]}')
if repair:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', '--force-reinstall', *repair])
    print('Repaired duplicate copied package records:', repair)
PY

"$target_python" -s - "$expected_flash_attn" <<'PY'
import importlib.metadata as metadata, sys
try:
    actual = metadata.version('flash-attn')
except metadata.PackageNotFoundError:
    raise SystemExit('flash-attn is missing. Supply --flash-attn-wheel for the matched binary; do not change Torch/CUDA.')
if actual != sys.argv[1]:
    raise SystemExit(f'flash-attn is {actual}; expected {sys.argv[1]}. Supply the matched wheel explicitly.')
PY
"$target_python" -s -m pip install --no-deps --no-build-isolation --use-pep517 --editable "$project_dir/verl" --editable "$project_dir"

# These are target-environment variables, not shell-wide or base-env settings.
# Conda restores previous values on deactivate. Absolute paths are derived at
# install time and never embedded in the shareable environment.yml.
"$conda_exe" env config vars set --name "$env_name" \
  "PYTHONPATH=$project_dir/verl:$project_dir" PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  "RAGEN_API_KEYS_PATH=$project_dir/ragen/env/api-keys.json"

cd "$project_dir"
env PYTHONPATH="$project_dir/verl:$project_dir" PYTHONNOUSERSITE=1 \
  "$target_python" -s - "$project_dir" "$target_prefix" <<'PY'
import importlib, importlib.metadata as metadata, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
prefix = pathlib.Path(sys.argv[2]).resolve()
expected = {'ragen': root / 'ragen', 'verl': root / 'verl/verl'}
for name, directory in expected.items():
    module = importlib.import_module(name)
    actual = pathlib.Path(module.__file__).resolve()
    if not actual.is_relative_to(directory.resolve()):
        raise SystemExit(f'{name} imported from an unexpected tree: {actual}')
for name in ('torch', 'transformers', 'vllm', 'flash_attn'):
    module = importlib.import_module(name)
    actual = pathlib.Path(module.__file__).resolve()
    if not actual.is_relative_to(prefix):
        raise SystemExit(f'{name} still imports outside the new prefix: {actual}')
if metadata.version('verl') != '0.6.1':
    raise SystemExit('verl metadata is not the pinned version.')
print('Installed source paths and binary dependency origins verified.')
PY
"$target_python" -s -m pip check
if [[ -f "$project_dir/scripts/check_ragen2_environment.py" && "$env_name" == ragen2 ]]; then
  env PYTHONPATH="$project_dir/verl:$project_dir" PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
    "$target_python" -s "$project_dir/scripts/check_ragen2_environment.py" \
    --project-root "$project_dir" --skip-model-check
fi
echo "Configured independent environment: $env_name ($target_prefix)"
echo "Activate it with: conda activate $env_name"
echo 'Run the project environment check before a GPU smoke or formal training.'
