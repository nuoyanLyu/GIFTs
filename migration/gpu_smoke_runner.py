#!/usr/bin/env python3
"""Run one isolated GPU4 training smoke, then restore the same holder under this supervisor.

Launch this runner inside its own tmux session. It never starts/stops another
GPU, joins an existing Ray cluster, or uses global process-name cleanup.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from urllib.request import Request, urlopen
import uuid

GPU = "4"
GPU_UUID = "GPU-61be0e3d-c360-3047-a51c-de3301fef717"
OLD_RUN = Path("/home/lvnuoyan/RAGEN/logs/creative_writing_v3_20261116")
OLD_SCRIPT = OLD_RUN / "run_creative_writing.py"
MODEL = "/data1/model/Qwen2.5-7B-Instruct"
PORT = 18440
TOKEN_KEY = "RAGEN2_SMOKE_RUN_TOKEN"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def identity(pid):
    directory = Path("/proc") / str(pid)
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    return {"uid": directory.stat().st_uid, "start_ticks": fields[19],
            "ppid": int(fields[1]), "pgrp": int(fields[2]), "state": fields[0]}


def same(pid, recorded):
    try:
        current = identity(pid)
        return (current["uid"] == recorded["uid"]
                and current["start_ticks"] == recorded["start_ticks"]
                and current["state"] not in ("Z", "X"))
    except (OSError, ValueError, IndexError):
        return False


def command(pid):
    return [x.decode(errors="replace") for x in
            (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0") if x]


def environ(pid):
    return dict(x.decode(errors="replace").split("=", 1) for x in
                (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0") if b"=" in x)


def snapshot():
    result = {}
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            result[int(entry.name)] = identity(int(entry.name))
        except (OSError, ValueError, IndexError):
            pass
    return result


def descendants(roots, processes=None):
    processes = snapshot() if processes is None else processes
    included = set(roots)
    while True:
        added = {pid for pid, item in processes.items() if item["ppid"] in included}
        if added <= included:
            return {pid: processes[pid] for pid in included if pid in processes
                    and processes[pid]["uid"] == os.getuid()}
        included.update(added)


def checked_signal(pid, recorded, sig):
    if recorded["uid"] != os.getuid():
        raise RuntimeError(f"Refusing to signal another UID: {pid}")
    if same(pid, recorded):
        os.kill(pid, sig)


def gpu_processes():
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    actual = {x.split(",")[0].strip(): x.split(",")[1].strip()
              for x in query.splitlines() if x.strip()}
    if actual.get(GPU) != GPU_UUID:
        raise RuntimeError(f"GPU4 UUID changed: {actual.get(GPU)!r}")
    query = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"], text=True)
    result = {}
    for line in query.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[0] == GPU_UUID:
            result[int(parts[1])] = parts[2]
    return result


def assert_gpu_owned(allowed):
    # NVML may briefly list a worker after it became a zombie or disappeared.
    # Preserve known ownership during teardown, while still rejecting PID reuse.
    live = set()
    for pid, ident in allowed.items():
        try:
            current = identity(pid)
        except (OSError, ValueError, IndexError):
            live.add(pid)
            continue
        if current['uid'] == ident['uid'] and current['start_ticks'] == ident['start_ticks']:
            live.add(pid)
    actual = gpu_processes()
    unknown = set(actual) - live
    if unknown:
        raise RuntimeError(f"GPU4 has unowned compute processes; refusing handoff/restoration: {sorted(unknown)}")
    return actual


def json_request(path, body=None):
    request = Request(f"http://127.0.0.1:{PORT}{path}",
                      data=None if body is None else json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=12) as response:
        data = response.read()
        return response.status, json.loads(data) if data else None


def discover_original(status_path=OLD_RUN / "status.json", tmux_name="creative_v3_gpu4"):
    """Use current status and the actual tmux/process ancestry, never historic PIDs."""
    status = read_json(status_path)
    if status.get("holder_protocol") == "ragen2_gpu4_v1":
        ready = (status.get("phase") == "complete_holding"
                 and status.get("restoration", {}).get("verified") is True)
        expected_script = str(Path(__file__).resolve())
    else:
        ready = (status.get("phase") == "holding_after_generation"
                 and status.get("generation_complete") is True)
        expected_script = str(OLD_SCRIPT)
    if not ready or str(status.get("gpu")) != GPU:
        raise RuntimeError("Previous queue is not complete and supervising a healthy GPU4 holder")
    supervisor = int(status["pid"])
    sup_ident = identity(supervisor)
    supervisor_cwd = (Path('/proc') / str(supervisor) / 'cwd').resolve()
    script_matches = any(
        str((supervisor_cwd / item).resolve()) == expected_script
        for item in command(supervisor) if item.endswith('.py')
    )
    if sup_ident["uid"] != os.getuid() or not script_matches:
        raise RuntimeError("Original supervisor command/UID does not match")
    panes = subprocess.check_output(
        ["tmux", "list-panes", "-t", tmux_name, "-F", "#{pane_pid}"], text=True)
    pane_roots = [int(x) for x in panes.split()]
    if supervisor not in descendants(pane_roots):
        raise RuntimeError("Previous supervisor no longer belongs to the selected tmux")
    holder = int(status["holder"]["pid"])
    tree = descendants([supervisor])
    if holder not in tree:
        raise RuntimeError("Holder is no longer a descendant of the original supervisor")
    holder_command, holder_env = command(holder), environ(holder)
    required = {"--host": "127.0.0.1", "--port": str(PORT), "--gpu-memory-utilization": "0.90"}
    if ("serve" not in holder_command or MODEL not in holder_command
            or holder_env.get("CUDA_VISIBLE_DEVICES") != GPU
            or not any(Path(x).name == "vllm" for x in holder_command)):
        raise RuntimeError("Holder model/command/device does not match the authorized 7B holder")
    for flag, value in required.items():
        if holder_command.count(flag) != 1 or holder_command[holder_command.index(flag) + 1] != value:
            raise RuntimeError(f"Holder {flag} changed")
    holder_tree = descendants([holder])
    if not same(supervisor, sup_ident) or not same(holder, tree[holder]):
        raise RuntimeError("Supervisor/holder changed while discovering their identity")
    return supervisor, sup_ident, holder, holder_tree


GUARD_CODE = r'''
import json, os, signal, sys, time
from pathlib import Path
import torch
stopping = False
def stop(*_):
    global stopping
    stopping = True
for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, stop)
torch.cuda.set_device(0)
allocation = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
allocation.zero_()
torch.cuda.synchronize()
path = Path(sys.argv[1])
path.write_text(json.dumps({"pid": os.getpid(), "allocated_bytes": allocation.numel(),
                            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}))
while not stopping:
    time.sleep(.5)
del allocation
torch.cuda.empty_cache()
'''


class Runner:
    def __init__(self, args):
        self.args = args
        self.root = Path(__file__).resolve().parents[1]
        self.directory = args.run_dir.absolute()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.token = uuid.uuid4().hex
        self.state = {"pid": os.getpid(), "started_at": now(), "phase": "checking",
                      "gpu": GPU, "gpu_uuid": GPU_UUID, "token": self.token,
                      "training": {}, "restoration": {"verified": False}}
        self.supervisor = None
        self.supervisor_ident = None
        self.original_retired = False
        self.acquired = False
        self.original_tree = {}
        self.guard = None
        self.guard_tree = {}
        self.training = None
        self.holder = None
        self.holder_tree = {}
        self.holder_command = None
        self.holder_env = None
        self.owned = {}
        self.logs = []
        self.lock_handle = None
        self.lock_released = False

    def write(self):
        self.state["updated_at"] = now()
        self.state["owned_processes"] = {str(pid): x for pid, x in self.owned.items()}
        write_json(self.directory / "status.json", self.state)

    def gates(self):
        cpu, template, numeric = map(read_json, [self.args.cpu_evidence, self.args.template_evidence,
                                                self.args.numeric_evidence])
        if (cpu.get("passed") is not True or cpu.get("cuda_initialized") is not False
                or cpu.get("native_thinking") is not False):
            raise RuntimeError("CPU/config/native-nonthinking preflight did not pass")
        if Path(cpu.get('prefix', '')).resolve() != Path(sys.prefix).resolve():
            raise RuntimeError('CPU evidence was produced by another environment')
        if Path(numeric.get('prefix', '')).resolve() != Path(sys.prefix).resolve():
            raise RuntimeError('Numeric evidence was produced by another environment')
        candidate = self.root / 'ragen/llm_agent/ctx_manager.py'
        if template.get('source_sha256', {}).get('candidate') != hashlib.sha256(candidate.read_bytes()).hexdigest():
            raise RuntimeError('Template evidence does not match the active source')
        if template.get("passed") is not True:
            raise RuntimeError("Tokenizer/template regression checks did not pass")
        if (numeric.get("bitwise_equal") is not True or numeric.get("checks", 0) < 18
                or numeric.get("cuda_initialized") is not False
                or numeric.get("singleton_adv_nonzero") is not True):
            raise RuntimeError("Legacy algorithm/gradient numeric equivalence checks did not pass")
        self.state["gates"] = {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in [("cpu", self.args.cpu_evidence), ("template", self.args.template_evidence),
                               ("numeric", self.args.numeric_evidence)]}
        self.state["gates"]["source_sha256"] = {
            str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("ragen", "config") for path in sorted((self.root / folder).rglob("*"))
            if path.is_file() and path.suffix in (".py", ".yaml")}

    def launch(self, command_line, log_name, env):
        log = (self.directory / log_name).open("a")
        self.logs.append(log)
        try:
            return subprocess.Popen(command_line, cwd=self.root, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            # The child owns its duplicated descriptor. Avoid leaking one parent
            # descriptor per long-term holder restart.
            log.close()

    def start_guard(self):
        marker = self.directory / "guard_ready.json"
        marker.unlink(missing_ok=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU_UUID, "PYTHONUNBUFFERED": "1"}
        self.guard = self.launch([str(self.args.guard_python), "-u", "-c", GUARD_CODE, str(marker)],
                                 "guard.log", env)
        self.guard_tree = {self.guard.pid: identity(self.guard.pid)}
        self.state["guard"] = {"pid": self.guard.pid, "identity": self.guard_tree[self.guard.pid],
                               "requested_bytes": 256 * 1024 * 1024}
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            self.guard_tree.update(descendants(
                [pid for pid, ident in self.guard_tree.items() if same(pid, ident)]))
            if self.guard.poll() is not None:
                raise RuntimeError("Bridge guard exited before becoming ready; see guard.log")
            if marker.exists():
                ready = read_json(marker)
                if ready.get("pid") != self.guard.pid or ready.get("cuda_visible_devices") != GPU_UUID:
                    raise RuntimeError("Guard readiness identity/device mismatch")
                self.state["guard"]["ready"] = ready
                self.write()
                return
            self.write()
            time.sleep(.5)
        raise TimeoutError("Bridge guard did not allocate memory in 90 seconds")

    def capture_training(self):
        processes = snapshot()
        roots = [pid for pid, item in self.owned.items() if same(pid, item)]
        if self.training is not None and self.training.pid in processes:
            current = processes[self.training.pid]
            recorded = self.owned.get(self.training.pid)
            if recorded is None or current["start_ticks"] == recorded["start_ticks"]:
                roots.append(self.training.pid)
        candidates = descendants(roots, processes)
        # Ray/vLLM can create their own sessions and reparent on driver exit.
        # The unique inherited marker still proves those workers belong to us.
        for pid, item in processes.items():
            if item["uid"] == os.getuid() and pid not in candidates:
                try:
                    if environ(pid).get(TOKEN_KEY) == self.token:
                        candidates[pid] = item
                except OSError:
                    pass
        for pid, item in candidates.items():
            if pid == os.getpid() or pid == self.supervisor or pid in self.guard_tree:
                continue
            prior = self.owned.get(pid)
            if prior is None or prior["start_ticks"] == item["start_ticks"]:
                self.owned[pid] = item

    def stop_tree(self, tree, refresh=None, first_signal=signal.SIGTERM):
        for sig, duration in ((first_signal, 35), (signal.SIGTERM, 15), (signal.SIGKILL, 10)):
            deadline = time.monotonic() + duration
            signaled = set()
            while True:
                if refresh is not None:
                    refresh()
                alive = {pid: item for pid, item in tree.items() if same(pid, item)}
                if not alive:
                    return
                for pid, item in alive.items():
                    if pid not in signaled:
                        checked_signal(pid, item, sig)
                        signaled.add(pid)
                self.write()
                if time.monotonic() >= deadline:
                    break
                time.sleep(.5)
        raise RuntimeError(f"Owned processes did not terminate: {[p for p,i in tree.items() if same(p,i)]}")

    def discover(self):
        return discover_original(self.args.takeover_status, self.args.takeover_tmux)

    def handoff(self):
        supervisor, sup_ident, holder, tree = self.discover()
        assert_gpu_owned(tree)
        self.start_guard()
        supervisor, sup_ident, holder, tree = self.discover()
        active = assert_gpu_owned({**tree, **self.guard_tree})
        if self.guard.poll() is not None or not (set(self.guard_tree) & set(active)):
            raise RuntimeError("Bridge guard is not alive and resident on GPU4")
        self.supervisor, self.supervisor_ident = supervisor, sup_ident
        self.original_tree = tree
        self.holder_command = command(holder)
        # Preserve the actual gift1 environment in memory; record only relevant,
        # non-secret runtime keys in the status file.
        self.holder_env = environ(holder)
        visible = {k: v for k, v in self.holder_env.items() if k in {
            "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "VLLM_USE_V1",
            "VLLM_WORKER_MULTIPROC_METHOD", "OMP_NUM_THREADS", "HF_ENDPOINT",
            "HF_HUB_OFFLINE", "TOKENIZERS_PARALLELISM", "PYTHONUNBUFFERED",
            "VLLM_ATTENTION_BACKEND", "LD_LIBRARY_PATH"}}
        self.state["original"] = {"supervisor": supervisor, "supervisor_identity": sup_ident,
                                  "holder": holder, "holder_tree": tree,
                                  "command": self.holder_command, "runtime_env": visible,
                                  "status_path": str(self.args.takeover_status),
                                  "tmux": self.args.takeover_tmux}
        # tmux automatically SIGCONTs its stopped pane child. Retire the completed
        # holder-only supervisor instead; never signal its whole process group.
        self.acquired = True
        self.state["phase"] = "retiring_previous_supervisor"
        self.write()
        old_status = read_json(self.args.takeover_status)
        if old_status.get("holder_protocol") == "ragen2_gpu4_v1":
            request = {"supervisor_pid": supervisor, "supervisor_identity": sup_ident,
                       "successor_pid": os.getpid(), "successor_identity": identity(os.getpid()),
                       "guard_pid": self.guard.pid, "guard_identity": self.guard_tree[self.guard.pid],
                       "gpu_uuid": GPU_UUID, "requested_at": now()}
            write_json(self.args.takeover_status.parent / "retire_supervisor.json", request)
        else:
            checked_signal(supervisor, sup_ident, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while same(supervisor, sup_ident):
            if time.monotonic() >= deadline:
                raise RuntimeError("Previous supervisor did not retire; training was not started")
            self.write()
            time.sleep(.25)
        self.original_retired = True
        self.state["original"]["retired_at"] = now()
        self.state["phase"] = "stopping_previous_holder"
        self.write()
        self.stop_tree(self.original_tree, first_signal=signal.SIGINT)
        assert_gpu_owned(self.guard_tree)

    def train(self):
        # Ray's Unix socket paths must stay below the system length limit.
        ray_dir = Path('/tmp') / ('ragen2_' + self.token[:12])
        ray_dir.mkdir()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "PYTHONUNBUFFERED": "1",
               "RAGEN2_ENV_PREFIX": sys.prefix,
               "RAY_ADDRESS": "local", "RAY_TMPDIR": str(ray_dir), TOKEN_KEY: self.token,
               "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false",
               "RAGEN_COMPAT_AUDIT": "1"}
        command_line = ["bash", "run_factory.sh", "train.py", "--config-name",
                        "qwen3_4b_nothink_smoke", f"trainer.default_local_dir={self.directory / 'checkpoints'}"]
        self.state["phase"] = "training"
        self.state["training"] = {"command": command_line, "started_at": now(),
                                  "timeout_seconds": self.args.timeout, "ray_tmpdir": str(ray_dir)}
        self.training = self.launch(command_line, "training.log", env)
        self.owned[self.training.pid] = identity(self.training.pid)
        self.state["training"]["pid"] = self.training.pid
        deadline = time.monotonic() + self.args.timeout
        while True:
            if self.guard.poll() is not None:
                raise RuntimeError("Bridge guard exited during training")
            self.capture_training()
            try:
                assert_gpu_owned({**self.guard_tree, **self.owned})
            except RuntimeError:
                # A CUDA worker may appear between /proc and nvidia-smi scans.
                # Re-establish ownership once before treating it as foreign.
                self.capture_training()
                assert_gpu_owned({**self.guard_tree, **self.owned})
            self.write()
            code = self.training.poll()
            if code is not None:
                self.state["training"].update(exit_code=code, finished_at=now())
                if code != 0:
                    raise RuntimeError(f"Training smoke exited {code}; see training.log")
                return
            if time.monotonic() >= deadline:
                self.state["training"]["timed_out"] = True
                raise TimeoutError(f"Training smoke exceeded {self.args.timeout}s")
            time.sleep(.5)

    def verify_holder(self, holder, tree):
        health, _ = json_request("/health")
        models_status, models = json_request("/v1/models")
        ids = [x["id"] for x in models.get("data", [])]
        if health != 200 or models_status != 200 or MODEL not in ids:
            raise RuntimeError("Holder health/models did not match the original 7B model")
        inference_status, inference = json_request("/v1/completions", {
            "model": MODEL, "prompt": "Reply OK.", "max_tokens": 4, "temperature": 0})
        if inference_status != 200 or not inference.get("choices") or not inference["choices"][0].get("text"):
            raise RuntimeError("Holder short generation failed")
        actual = assert_gpu_owned({**tree, **self.guard_tree})
        self.state["restoration"] = {"verified": True, "verified_at": now(),
            "supervisor": os.getpid() if self.original_retired else self.supervisor,
            "holder": holder, "holder_tree": tree, "model": MODEL, "port": PORT,
            "health_status": health, "models_status": models_status,
            "inference_status": inference_status, "inference": inference,
            "gpu_processes_mib": actual}
        self.write()

    def start_holder(self):
        if self.holder is not None:
            self.holder_tree.update(descendants(
                [pid for pid, ident in self.holder_tree.items() if same(pid, ident)]))
            self.stop_tree(self.holder_tree)
            self.holder.wait(timeout=5)
        assert_gpu_owned(self.guard_tree)
        self.holder = self.launch(self.holder_command, "holder.log", self.holder_env)
        self.holder_tree = {self.holder.pid: identity(self.holder.pid)}
        self.state["holder"] = {"pid": self.holder.pid, "identity": self.holder_tree[self.holder.pid],
            "command": self.holder_command, "started_at": now(), "ready": False,
            "restarts": self.state.get("holder", {}).get("restarts", -1) + 1}
        self.state["restoration"]["verified"] = False
        self.write()

    def restore(self):
        if self.training is not None:
            self.capture_training()
            self.state["phase"] = "cleaning_training"
            self.write()
            self.stop_tree(self.owned, self.capture_training)
            self.training.wait(timeout=5)
        if not self.acquired:
            return
        self.state["phase"] = "restoring_holder"
        self.write()
        if same(self.supervisor, self.supervisor_ident):
            # Retirement failed or was interrupted before the old supervisor
            # exited. Keep its holder intact and verify it; do not race a second.
            supervisor, _, holder, tree = self.discover()
            if supervisor != self.supervisor:
                raise RuntimeError("Previous supervisor identity changed")
            self.verify_holder(holder, tree)
            self.state["restoration"]["previous_supervisor_retained"] = True
            self.write()
            return
        self.original_retired = True
        self.stop_tree(self.original_tree, first_signal=signal.SIGINT)
        assert_gpu_owned(self.guard_tree)
        deadline = time.monotonic() + self.args.restore_timeout
        self.start_holder()
        last_error = None
        while time.monotonic() < deadline:
            try:
                self.holder_tree.update(descendants(
                    [pid for pid, ident in self.holder_tree.items() if same(pid, ident)]))
                if self.holder.poll() is not None:
                    if self.state["holder"]["restarts"] >= 2:
                        raise RuntimeError("Replacement holder failed three startup attempts")
                    self.start_holder()
                self.verify_holder(self.holder.pid, self.holder_tree)
                self.state["holder"]["ready"] = True
                self.state["holder_protocol"] = "ragen2_gpu4_v1"
                self.write()
                return
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self.state["restoration"]["last_error"] = last_error
                self.write()
                time.sleep(2)
        raise TimeoutError(f"Replacement holder not verified within {self.args.restore_timeout}s: {last_error}")

    def retire_request_valid(self):
        path = self.directory / "retire_supervisor.json"
        if not path.exists():
            return False
        value = read_json(path)
        if (value.get("supervisor_pid") != os.getpid() or value.get("gpu_uuid") != GPU_UUID
                or not same(os.getpid(), value["supervisor_identity"])
                or not same(value["successor_pid"], value["successor_identity"])
                or not same(value["guard_pid"], value["guard_identity"])):
            raise RuntimeError("Retirement request has stale/wrong process identity")
        if (value["successor_identity"]["uid"] != os.getuid()
                or value["guard_identity"]["uid"] != os.getuid()
                or environ(value["guard_pid"]).get("CUDA_VISIBLE_DEVICES") != GPU_UUID
                or value["guard_pid"] not in gpu_processes()):
            raise RuntimeError("Successor has not established its own GPU4 guard")
        return True

    def release_handoff_lock(self):
        if self.lock_handle is not None and not self.lock_released:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
            self.lock_released = True
            self.state["handoff_lock_released_at"] = now()

    def supervise_holder(self):
        """Long-term duty; valid successor gate retires only this supervisor."""
        def interrupted(signum, _):
            self.state['last_signal'] = {'number': signum, 'received_at': now()}
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, interrupted)
        unhealthy_since = None
        while True:
            try:
                try:
                    retire = self.retire_request_valid()
                except Exception as exc:
                    self.state["retirement_request_error"] = str(exc)
                    retire = False
                if retire:
                    self.state["phase"] = "retired_for_successor"
                    self.state["retired_at"] = now()
                    self.write()
                    return 0
                if self.holder is not None:
                    self.holder_tree.update(descendants(
                        [pid for pid, ident in self.holder_tree.items() if same(pid, ident)]))
                alive = self.holder is not None and self.holder.poll() is None
                try:
                    health, _ = json_request("/health")
                    ready = alive and health == 200
                except Exception:
                    ready = False
                if ready:
                    if not self.state["restoration"].get("verified"):
                        self.verify_holder(self.holder.pid, self.holder_tree)
                        if self.guard is not None and self.guard.poll() is None:
                            self.stop_tree(self.guard_tree)
                            self.guard.wait(timeout=5)
                    unhealthy_since = None
                    self.state["holder"]["ready"] = True
                    self.state["phase"] = "complete_holding"
                    self.release_handoff_lock()
                else:
                    self.state.setdefault("holder", {})["ready"] = False
                    self.state["restoration"]["verified"] = False
                    unhealthy_since = unhealthy_since or time.monotonic()
                    if not alive or time.monotonic() - unhealthy_since > 600:
                        if self.training is not None:
                            self.capture_training()
                            self.stop_tree(self.owned, self.capture_training)
                        if self.guard is None or self.guard.poll() is not None:
                            self.start_guard()
                        self.start_holder()
                        self.verify_when_ready()
                        self.stop_tree(self.guard_tree)
                        self.guard.wait(timeout=5)
                        unhealthy_since = None
                self.write()
                time.sleep(3)
            except Exception as exc:
                # Unexpected interruptions keep the supervised holder running.
                # A successor uses the explicit identity-bound retirement gate.
                self.state["holder_monitor_error"] = f"{type(exc).__name__}: {exc}"
                self.write()
                time.sleep(3)

    def verify_when_ready(self):
        deadline = time.monotonic() + self.args.restore_timeout
        while time.monotonic() < deadline:
            self.holder_tree.update(descendants(
                    [pid for pid, ident in self.holder_tree.items() if same(pid, ident)]))
            if self.holder.poll() is not None:
                raise RuntimeError("Holder exited during restart; retrying through supervisor")
            try:
                self.verify_holder(self.holder.pid, self.holder_tree)
                self.state["holder"]["ready"] = True
                return
            except Exception as exc:
                self.state["restoration"]["last_error"] = str(exc)
                self.write()
                time.sleep(2)
        raise TimeoutError("Holder restart health/inference timeout")

    def run(self):
        try:
            self.gates()
            self.write()
            if self.args.inspect_only:
                supervisor, sup_ident, holder, tree = self.discover()
                self.state["inspection"] = {"supervisor": supervisor, "holder": holder,
                                             "supervisor_identity": sup_ident,
                                             "gpu_processes_mib": assert_gpu_owned(tree)}
                self.state["phase"] = "inspection_passed"
                self.write()
                return 0
            self.handoff()
            self.train()
        except BaseException as exc:
            self.state["error"] = f"{type(exc).__name__}: {exc}"
            self.state["traceback"] = traceback.format_exc()
        finally:
            # A repeated signal must not interrupt the mandatory resource restoration.
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            try:
                self.restore()
            except BaseException as exc:
                self.state["restoration"]["error"] = f"{type(exc).__name__}: {exc}"
                self.state["restoration"]["traceback"] = traceback.format_exc()
            # Keep the small bridge allocation if handoff began but restoration
            # failed, so the primary agent can recover without an empty GPU.
            keep_guard = (self.acquired and not self.state["restoration"].get("verified"))
            if self.guard is not None and not keep_guard:
                try:
                    self.stop_tree(self.guard_tree)
                    self.guard.wait(timeout=5)
                    self.state["guard"]["stopped_at"] = now()
                except BaseException as exc:
                    self.state.setdefault("cleanup_errors", []).append(str(exc))
            elif self.guard is not None:
                self.state["guard"]["kept_for_recovery"] = True
            if not self.args.inspect_only:
                if self.state["restoration"].get("verified"):
                    self.state["phase"] = "complete_holding" if self.original_retired else "failed_original_holding"
                elif not self.acquired:
                    self.state["phase"] = "refused_before_handoff"
                else:
                    self.state["phase"] = "restore_failed"
            self.state["smoke_phase_finished_at"] = now()
            self.write()
            for log in self.logs:
                log.close()
        if self.original_retired:
            self.state["holder_protocol"] = "ragen2_gpu4_v1"
            self.state["training_process_succeeded"] = self.state.get("training", {}).get("exit_code") == 0
            if self.state["restoration"].get("verified"):
                self.release_handoff_lock()
            self.write()
            return self.supervise_holder()
        return 0 if not self.state.get("error") and self.state["restoration"].get("verified") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cpu-evidence", type=Path, required=True)
    parser.add_argument("--template-evidence", type=Path, required=True)
    parser.add_argument("--numeric-evidence", type=Path, required=True)
    parser.add_argument("--guard-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--restore-timeout", type=int, default=600)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--takeover-status", type=Path, default=OLD_RUN / "status.json")
    parser.add_argument("--takeover-tmux", default="creative_v3_gpu4")
    args = parser.parse_args()
    if args.timeout <= 0 or args.restore_timeout <= 0:
        parser.error("Timeouts must be positive")
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "migration/gpu4_handoff.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        def interrupted(signum, _):
            raise InterruptedError(f"Signal {signum}")
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, interrupted)
        runner = Runner(args)
        runner.lock_handle = lock
        result = runner.run()
        print(json.dumps({"exit_code": result, "phase": runner.state["phase"],
                          "status": str(runner.directory / "status.json")}), flush=True)
        return result


if __name__ == "__main__":
    sys.exit(main())
