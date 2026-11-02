"""
Score the Glassbox group's delivered EBM (Project Official/models/glassbox/ebm.joblib)
on the FULL sarc_random validation partition (the committed cache only holds a
60k-row sample), on sarc_random/test and on the transfer sets, using the frozen
feature pipeline (glassbox_feature_pipeline.joblib). Nothing is refitted.

The EBM is not scored on sarc_subreddit: it was never refitted on that scheme's
training partition and its subreddit target encoding was learned on
sarc_random/train, so its sarc_subreddit predictions would not be comparable.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OFFICIAL, load_frame, row_ids, save_predictions  # noqa: E402

GB = OFFICIAL / "modelling" / "glassbox"
sys.path.insert(0, str(GB))
spec = importlib.util.spec_from_file_location("gb_build", GB / "20_build_features.py")
gb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gb)
main_mod = sys.modules["__main__"]
for k in ("LexiconScorer", "SubredditEncoder"):          # pickled from __main__
    setattr(main_mod, k, getattr(gb, k))

pipe = joblib.load(OFFICIAL / "models" / "glassbox" / "glassbox_feature_pipeline.joblib")
ebm = joblib.load(OFFICIAL / "models" / "glassbox" / "ebm.joblib")

# sanity: committed test cache reproduces
Xc = np.load(OFFICIAL / ".cache" / "glassbox" / "X_test.npy")
yc = np.load(OFFICIAL / ".cache" / "glassbox" / "y_test.npy")
print("cached sarc_random/test AUC", round(roc_auc_score(yc, ebm.predict_proba(Xc)[:, 1]), 4), flush=True)

for ds, part in [("sarc_random", "val"), ("sarc_random", "test"), ("news_headlines", "test"),
                 ("figlang_reddit", "test"), ("figlang_twitter", "test"), ("tweeteval_irony", "test")]:
    df = load_frame(ds, part)
    X = gb.build(df, pipe["unigrams"], pipe["bigrams"], pipe["lexicon"], pipe["encoder"])
    p = ebm.predict_proba(X)[:, 1]
    y = df["label"].to_numpy()
    save_predictions("ebm", ds, part, y, {"sarc_random__ebm_named340": p}, ids=row_ids(df))
    print(f"{ds}/{part}: AUC {roc_auc_score(y, p):.4f}", flush=True)
