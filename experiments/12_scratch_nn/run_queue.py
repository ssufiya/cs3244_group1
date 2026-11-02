"""Round 12 queue, run detached: sarc_random pair and reply until convergence,
then sarc_subreddit with the epoch counts chosen on sarc_random. Each job resumes
from its saved state and is skipped once its results json exists."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "logs"


def free_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def run(variant, scheme, extra=()):
    name = f"scratch_{variant}__{scheme}"
    for _ in range(3):
        if (HERE / "results" / f"{name}.json").exists():
            return
        while free_mb() < 6144:   # trainer also caps itself at ~3.2 GB (--mem-frac)
            time.sleep(120)
        with open(LOGS / f"12_{name}.log", "a") as log:
            subprocess.run([sys.executable, str(HERE / "nn_scratch.py"), "--variant", variant,
                            "--scheme", scheme, *extra], stdout=log, stderr=subprocess.STDOUT, cwd=HERE.parent)


for v in ("pair", "reply"):
    run(v, "sarc_random")
for v in ("pair", "reply"):
    best = json.loads((HERE / "results" / f"scratch_{v}__sarc_random.json").read_text())["best_epoch"]
    run(v, "sarc_subreddit", ("--fixed-epochs", str(best)))
