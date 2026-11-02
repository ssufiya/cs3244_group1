"""
Improvement 7 - lift the GBDT vocabulary cap.

The baseline GBDT kept only the top 20,000 reply n-grams, a limit set by the
8 GB GPU used for XGBoost. LightGBM runs on the CPU, so the cap is not needed
for it. Same frozen hyper-parameters, cues + contrast blocks, vocabulary 60,000.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import write_json  # noqa: E402
from features_extra import contrast_block, cue_block  # noqa: E402
from lgbm_runner import run  # noqa: E402

(HERE / "results").mkdir(exist_ok=True)
out = {"vocab60k_cues_contrast": run("lgbm_v60k_cues_contrast", "lgbm_v60k_cues_contrast",
                                     blocks={"cues": cue_block, "contrast": contrast_block}, vocab=60_000)}
write_json(HERE / "results" / "summary.json", out)
