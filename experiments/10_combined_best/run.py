"""
Improvement 9 - everything that helped, in one LightGBM.

60k reply vocabulary + surface (improvement 7) + linguistic cues (1) +
reply/parent contrast (2) + out-of-fold word+char LR score (5) + frozen MiniLM
sentence embeddings with reply/parent similarity (8). LSA (4) is left out
because it hurt. Frozen baseline hyper-parameters, early stopping on
sarc_random/val, tree count frozen for sarc_subreddit.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import write_json  # noqa: E402
from features_extra import contrast_block, cue_block  # noqa: E402
from lgbm_runner import run  # noqa: E402


def load(folder):
    spec = importlib.util.spec_from_file_location(folder, ROOT / folder / "run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


emb = load("09_sentence_embeddings")
lin = load("06_linear_score_as_feature")
(HERE / "results").mkdir(exist_ok=True)
out = {"combined": run("lgbm_combined_best", "lgbm_combined_best",
                       blocks={"cues": cue_block, "contrast": contrast_block, "minilm": emb.emb_block},
                       fitted_block=lin.lr_score_block, vocab=60_000)}
write_json(HERE / "results" / "summary.json", out)
