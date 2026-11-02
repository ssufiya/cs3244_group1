"""Run the transformer variants one after another on the GPU (same interpreter)."""
import subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
for variant, scheme in [("pair", "sarc_random"), ("reply", "sarc_random"), ("pair", "sarc_subreddit")]:
    if (HERE / "results" / f"roberta_{variant}__{scheme}.json").exists():
        continue
    with open(HERE.parent / "logs" / f"07_{variant}_{scheme}.log", "w") as log:
        subprocess.run([sys.executable, str(HERE / "run.py"), "--variant", variant, "--scheme", scheme],
                       stdout=log, stderr=subprocess.STDOUT)
