import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from urllib.request import urlopen

import gpu_smoke_runner as handoff

ROOT = Path(__file__).resolve().parents[1]
MODEL = "/data1/model/Qwen3-4B"
PORT = 18443
PYTHON = "/home/lvnuoyan/anaconda3/envs/ragen2/bin/python"
OLD_STATUS = Path("/data1/lvnuoyan/ragen2_runs/conda_validation_20261117/status.json")
OLD_TMUX = "ragen2_conda_gpu4_20261117"


class HolderRunner(handoff.Runner):
    def discover(self):
        if self.args.takeover_status == OLD_STATUS:
            return super().discover()
        status = handoff.read_json(self.args.takeover_status)
        assert status["phase"] == "holding" and status["model"] == handoff.MODEL
        assert status["port"] == handoff.PORT and status["gpu_uuid"] == handoff.GPU_UUID
        supervisor = status["pid"]
        holder = status["holder"]["pid"]
        supervisor_identity = handoff.identity(supervisor)
        panes = subprocess.check_output(
            ["tmux", "list-panes", "-t", self.args.takeover_tmux, "-F", "#{pane_pid}"], text=True)
        assert supervisor in handoff.descendants([int(pid) for pid in panes.split()])
        assert any(Path(arg).name == Path(__file__).name for arg in handoff.command(supervisor))
        tree = handoff.descendants([holder])
        assert holder in handoff.descendants([supervisor])
        assert handoff.same(holder, status["holder"]["identity"])
        assert handoff.command(holder) == status["holder"]["command"]
        assert handoff.environ(holder).get("CUDA_VISIBLE_DEVICES") == handoff.GPU
        verify(handoff.PORT, handoff.MODEL, tree)
        return supervisor, supervisor_identity, holder, tree


def verify(port, model, tree):
    for endpoint in ("health", "v1/models"):
        with urlopen(f"http://127.0.0.1:{port}/{endpoint}", timeout=8) as response:
            assert response.status == 200
            if endpoint == "v1/models":
                assert any(item["id"] == model for item in json.load(response)["data"])
    active = handoff.gpu_processes()
    assert set(active) & set(tree)
    return active


def wait_ready(runner, port, model):
    deadline = time.monotonic() + 330
    while runner.holder.poll() is None and time.monotonic() < deadline:
        runner.holder_tree.update(handoff.descendants([runner.holder.pid]))
        try:
            active = verify(port, model, runner.holder_tree)
        except (OSError, ValueError, AssertionError):
            time.sleep(2)
        else:
            handoff.assert_gpu_owned({**runner.holder_tree, **runner.guard_tree})
            return active
    raise RuntimeError(f"Holder failed to load {model}; see holder.log")


def main():
    directory = Path(sys.argv[1]).absolute()
    takeover_status = Path(sys.argv[2]) if len(sys.argv) > 2 else OLD_STATUS
    takeover_tmux = sys.argv[3] if len(sys.argv) > 3 else OLD_TMUX
    args = SimpleNamespace(run_dir=directory, guard_python=PYTHON,
                           takeover_status=takeover_status, takeover_tmux=takeover_tmux)
    runner = HolderRunner(args)
    runner.write()
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": handoff.GPU,
           "VLLM_USE_V1": "1", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONUNBUFFERED": "1", "NO_PROXY": "127.0.0.1,localhost"}
    command = [PYTHON, str(Path(PYTHON).with_name("vllm")), "serve", MODEL,
               "--host", "127.0.0.1", "--port", str(PORT), "--max-model-len", "8192",
               "--gpu-memory-utilization", "0.90", "--enforce-eager",
               "--generation-config", "vllm", "--disable-log-requests"]
    port, model = PORT, MODEL
    with (ROOT / "migration/gpu4_handoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.lock_handle = lock
        with socket.socket() as test_socket:
            test_socket.bind(("127.0.0.1", PORT))
        try:
            runner.handoff()
            runner.holder_command, old_command = command, runner.holder_command
            runner.holder_env, old_env = env, runner.holder_env
            runner.state["phase"] = "loading_qwen3_holder"
            runner.write()
            for attempt in range(2):
                runner.start_holder()
                try:
                    wait_ready(runner, port, model)
                    break
                except RuntimeError:
                    if attempt == 1:
                        raise
        except BaseException as error:
            runner.state["startup_error"] = str(error)
            traceback.print_exc()
            if not runner.original_retired:
                if runner.guard is not None:
                    runner.stop_tree(runner.guard_tree)
                runner.state["phase"] = "refused_before_handoff"
                runner.write()
                return 1
            runner.stop_tree(runner.original_tree)
            runner.holder_command = runner.state["original"]["command"]
            runner.holder_env = old_env if "old_env" in locals() else runner.holder_env
            port, model = handoff.PORT, handoff.MODEL
            runner.start_holder()
            wait_ready(runner, port, model)
        finally:
            if runner.guard is not None:
                runner.stop_tree(runner.guard_tree)
                runner.guard.wait(timeout=5)
        runner.release_handoff_lock()
        runner.state.update(phase="holding", model=model, port=port,
                            gpu_processes=verify(port, model, runner.holder_tree))
        runner.write()
        while True:
            if runner.holder.poll() is not None:
                runner.state["phase"] = "restarting_holder"
                runner.write()
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                runner.lock_released = False
                handoff.assert_gpu_owned(runner.holder_tree)
                for _ in range(2):
                    runner.start_holder()
                    try:
                        active = wait_ready(runner, port, model)
                        break
                    except RuntimeError:
                        continue
                else:
                    runner.state["phase"] = "holder_failed"
                    runner.write()
                    return 1
                runner.release_handoff_lock()
                runner.state.update(phase="holding", gpu_processes=active)
                runner.write()
            time.sleep(3)


if __name__ == "__main__":
    def interrupted(signum, _):
        raise InterruptedError(f"Signal {signum}")

    for mode in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(mode, interrupted)
    sys.exit(main())
