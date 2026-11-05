"""
Detached, resumable queues for the final checks.

  python 14_final_checks/run_queue.py cpu    LightGBM seeds + sarc_temporal, EBM seeds + sarc_temporal
  python 14_final_checks/run_queue.py gpu    network seeds, networks on sarc_temporal, Transformer tuning

A job is skipped when its result file exists; the networks resume mid-run from
their saved state. GPU jobs wait for >= 6 GB free memory and cap their own
share with --mem-frac (default 0.27 of 12 GB).
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
NN, NR = ROOT / "12_scratch_nn" / "nn_scratch.py", ROOT / "12_scratch_nn" / "results"
EBM, ER = ROOT / "11_full_convergence" / "ebm_full.py", ROOT / "11_full_convergence" / "results"


def nn(variant, *extra, scheme="sarc_random", tag=None):
    args = [str(NN), "--variant", variant, "--scheme", scheme, *extra]
    return args, NR / f"{tag or 'scratch_' + variant}__{scheme}.json"


TF = ["--variant", "transformer_pair", "--warmup", "2000"]
QUEUES = {
    "cpu": [("lgbm_seeds_temporal", [str(HERE / "lgbm_seeds_temporal.py")], HERE / "results" / "lgbm_seeds_temporal.json.done"),
            ("ebm_s3245", [str(EBM), "--seed", "3245"], ER / "ebm_full_s3245__sarc_random.json"),
            ("ebm_s3246", [str(EBM), "--seed", "3246"], ER / "ebm_full_s3246__sarc_random.json"),
            ("ebm_temporal", [str(EBM), "--scheme", "sarc_temporal"], ER / "ebm_full__sarc_temporal.json")],
    "gpu": [("pair_s3245", *nn("pair", "--seed", "3245", tag="scratch_pair_s3245")),
            ("pair_s3246", *nn("pair", "--seed", "3246", tag="scratch_pair_s3246")),
            ("reply_s3245", *nn("reply", "--seed", "3245", tag="scratch_reply_s3245")),
            ("reply_s3246", *nn("reply", "--seed", "3246", tag="scratch_reply_s3246")),
            ("pair_temporal", *nn("pair", "--fixed-epochs", "7", scheme="sarc_temporal")),
            ("reply_temporal", *nn("reply", "--fixed-epochs", "6", scheme="sarc_temporal")),
            ("tf_cfg2", [str(NN), *TF, "--lr", "1e-3", "--bs", "512", "--tag-suffix", "_cfg2"],
             NR / "scratch_transformer_pair_cfg2__sarc_random.json"),
            ("tf_cfg3", [str(NN), *TF, "--lr", "3e-4", "--bs", "256", "--tf-layers", "2", "--tf-dim", "256",
                         "--tf-ff", "1024", "--tag-suffix", "_cfg3"], NR / "scratch_transformer_pair_cfg3__sarc_random.json"),
            ("tf_cfg4", [str(NN), *TF, "--lr", "3e-4", "--bs", "256", "--tf-drop", "0.2", "--tag-suffix", "_cfg4"],
             NR / "scratch_transformer_pair_cfg4__sarc_random.json")],
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
        with open(LOGS / f"14_{name}.log", "a", encoding="utf-8") as log:
            rc = subprocess.run([PY, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0 and done.suffix == ".done":
            done.touch()
        print(f"{name}: exit {rc} {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
