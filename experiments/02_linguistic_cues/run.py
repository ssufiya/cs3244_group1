"""
Improvement 1 - explicit linguistic sarcasm cues (proposal Q2).

LightGBM reply_tfidf_surface (frozen baseline hyper-parameters) vs the same
model plus 25 cue features: hyperbole/intensity (intensifiers, absolutes,
praise words, superlatives), rhetorical / mock-agreement patterns
("yeah because", "who knew", tag questions, sarcastic openers), understatement
hedges, emphasis markers (*word*, "quoted", emoticons, elongation) and
intra-reply sentiment contrast (VADER positive and negative words together).
Also includes the control run that reproduces the baseline with this runner.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import write_json  # noqa: E402
from features_extra import cue_block  # noqa: E402
from lgbm_runner import run  # noqa: E402

(HERE / "results").mkdir(exist_ok=True)
out = {"control": run("lgbm_base_repro", "lgbm_base_repro"),
       "cues": run("lgbm_cues", "lgbm_cues", blocks={"cues": cue_block})}
write_json(HERE / "results" / "summary.json", out)
