"""
Run the remaining ablations as detached, resumable queues.

  python 13_ablations/run_queue.py cpu    A1, A6, A8 one after another
  python 13_ablations/run_queue.py gpu    A7 networks, each waiting for >= 6 GB free GPU memory

A job is skipped when its result file exists, so relaunching the same command
after an interruption continues where it stopped (the networks also resume
mid-run from their saved state).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
PY = sys.executable
NN = ROOT / "12_scratch_nn"

QUEUES = {
    "cpu": [("a1", [str(HERE / "feature_blocks.py")], HERE / "results" / "a1_feature_blocks.json"),
            ("a6", [str(HERE / "ebm_design.py")], HERE / "results" / "a6_ebm_design.json"),
            ("a8", [str(HERE / "punctuation_artifact.py")], HERE / "results" / "a8_punctuation_artifact.json")],
    "gpu": [("a7_pair_noattn", [str(NN / "nn_scratch.py"), "--variant", "pair_noattn"],
             NN / "results" / "scratch_pair_noattn__sarc_random.json"),
            ("a7_transformer_pair", [str(NN / "nn_scratch.py"), "--variant", "transformer_pair", "--lr", "3e-4",
                                     "--warmup", "2000", "--bs", "256"],
             NN / "results" / "scratch_transformer_pair__sarc_random.json")],
}


def free_gpu_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def main(queue: str):
    for name, args, done in QUEUES[queue]:
        if done.exists():
            print(f"{name}: already done", flush=True)
            continue
        if queue == "gpu":
            while free_gpu_mb() < 6000:
                print(f"{name}: waiting for GPU memory ({free_gpu_mb()} MB free)", flush=True)
                time.sleep(300)
        print(f"{name}: start {time.strftime('%H:%M:%S')}", flush=True)
        with open(LOGS / f"13_{name}.log", "a", encoding="utf-8") as log:
            rc = subprocess.run([PY, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{name}: exit {rc} {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
