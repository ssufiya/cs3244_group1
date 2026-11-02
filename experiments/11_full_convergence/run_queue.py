"""
Run the remaining round-11 RoBERTa jobs one after another, detached from any
shell session. Each job resumes from its saved state if it was cut off, and is
skipped if its results json already exists. A job only starts once the GPU has
at least 8 GB free (it is shared with other coursework jobs).

  start: Start-Process python run_queue.py (see README)
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "logs"
JOBS = [
    ("roberta_pair_conv__sarc_subreddit", ["--scheme", "sarc_subreddit", "--fixed-epochs", "3"]),
    ("roberta_pair_conv_s4248__sarc_random", ["--scheme", "sarc_random", "--fixed-epochs", "3", "--seed", "4248"]),
]


def free_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


for name, args in JOBS:
    for attempt in range(3):
        if (HERE / "results" / f"{name}.json").exists():
            break
        while free_mb() < 8192:
            time.sleep(120)
        with open(LOGS / f"11_{name}.log", "a") as log:
            subprocess.run([sys.executable, str(HERE / "roberta_converge.py"), *args],
                           stdout=log, stderr=subprocess.STDOUT, cwd=HERE.parent)
