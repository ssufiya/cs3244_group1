"""
Improvement 2 - reply-versus-parent contrast features (proposal Q1).

The baseline's context block (parent TF-IDF + lengths/overlap) added ~0.
Here the parent enters only through 18 explicit contrast features: parent
sentiment and reply-minus-parent / reply-times-parent valence, the
"positive reply to a negative parent" incongruity flag (and its mirror), whether
the parent asked a question and the reply opens with "yeah/sure/oh...", how
much of the reply echoes the parent's words, and parent intensity.
Run alone and together with the cue block from improvement 1.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import write_json  # noqa: E402
from features_extra import contrast_block, cue_block  # noqa: E402
from lgbm_runner import run  # noqa: E402

(HERE / "results").mkdir(exist_ok=True)
out = {"contrast": run("lgbm_contrast", "lgbm_contrast", blocks={"contrast": contrast_block}),
       "cues_contrast": run("lgbm_cues_contrast", "lgbm_cues_contrast",
                            blocks={"cues": cue_block, "contrast": contrast_block})}
write_json(HERE / "results" / "summary.json", out)
