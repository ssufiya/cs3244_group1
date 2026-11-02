"""
Round 11 - interpretable model: EBM retrained on the FULL training split.

The team's EBM (0.7620) was fitted on a 200k-row sample with 340 features that
include a subreddit target encoding. Here it is refitted on all 640k training
rows with 66 named, human-readable features and no subreddit:
  * 17 reply surface counts + 5 parent-context features (baseline pipeline);
  * 25 linguistic cue features (round 02) and 18 reply/parent contrast features (round 03);
  * 1 "text evidence" term: the out-of-fold word+char n-gram LR log-odds (round 06,
    GroupKFold on parent_key). It is itself additive over n-grams, so a single
    prediction decomposes into EBM term contributions, and the text term
    decomposes further into the weights of the n-grams present in the reply.
EBM boosting runs until its internal early stopping (patience 100, up to 20k
rounds), i.e. to convergence. Validation is untouched by fitting; it is used
for the reported val AUC and the precision >= 0.75 threshold.

  python 11_full_convergence/ebm_full.py --scheme sarc_random [--no-text]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import (CONTEXT, SEED, SURFACE, load_frame, row_ids, save_predictions, scores,  # noqa: E402
                    select_threshold, write_json)
from features_extra import CONTRAST_NAMES, CUE_NAMES, contrast_block, cue_block  # noqa: E402
from lgbm_runner import CACHE, TRANSFER, cached_block  # noqa: E402

NAMES = SURFACE + CONTEXT + CUE_NAMES + CONTRAST_NAMES + ["text_ngram_logodds"]


def features(scheme, ds, part, text=True):
    df = load_frame(ds, part)
    cols = [df.reindex(columns=SURFACE + CONTEXT).astype(np.float32).fillna(0).to_numpy(),
            cached_block("cues", cue_block, ds, part), cached_block("contrast", contrast_block, ds, part)]
    if text:
        cols.append(np.load(CACHE / f"lrscore__{scheme}__{ds}__{part}.npy"))
    X = pd.DataFrame(np.hstack(cols), columns=NAMES if text else NAMES[:-1])
    return X, df["label"].to_numpy(), df


def main():
    from interpret.glassbox import ExplainableBoostingClassifier
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="sarc_random")
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    tag = "ebm_full" + ("_notext" if a.no_text else "")
    t0 = time.time()
    Xtr, ytr, _ = features(a.scheme, a.scheme, "train", not a.no_text)
    print(f"[{tag}/{a.scheme}] train {Xtr.shape} ({time.time()-t0:.0f}s)", flush=True)
    ebm = ExplainableBoostingClassifier(
        random_state=SEED, n_jobs=a.jobs, outer_bags=8, inner_bags=0, interactions=15,
        max_bins=256, learning_rate=0.02, early_stopping_rounds=100, max_rounds=20000,
        min_samples_leaf=50, validation_size=0.15)
    ebm.fit(Xtr, ytr)
    fit_s = round(time.time() - t0)
    print(f"[{tag}/{a.scheme}] fitted in {fit_s}s", flush=True)
    out = {"model": tag, "scheme": a.scheme, "train_rows": len(ytr), "n_features": Xtr.shape[1],
           "n_terms": len(ebm.term_names_), "fit_seconds": fit_s,
           "best_rounds_per_bag": np.asarray(getattr(ebm, "best_iteration_", [])).reshape(-1).tolist()}
    joblib.dump(ebm, CACHE / f"11_{tag}__{a.scheme}.joblib")
    keys = [(a.scheme, "val"), (a.scheme, "test")] + ([(t, "test") for t in TRANSFER] if a.scheme == "sarc_random" else [])
    thr = None
    for ds, part in keys:
        X, y, df = features(a.scheme, ds, part, not a.no_text)
        p = ebm.predict_proba(X)[:, 1]
        save_predictions(f"11_{tag}", ds, part, y, {f"{a.scheme}__{tag}": p}, ids=row_ids(df))
        if part == "val":
            out["val_auc"] = roc_auc_score(y, p)
            thr = select_threshold(y, p)
            out["p75_threshold"] = thr
        else:
            s = scores(y, p)
            if ds == a.scheme and thr is not None:
                h = scores(y, p, thr)
                s.update({"p75_precision": h["precision"], "p75_recall": h["recall"]})
            out[ds] = s
        print(f"[{tag}/{a.scheme}] {ds}/{part}: AUC {roc_auc_score(y, p):.4f}", flush=True)
    imp = pd.DataFrame({"term": ebm.term_names_, "importance": ebm.term_importances()}) \
        .sort_values("importance", ascending=False)
    (HERE / "results").mkdir(exist_ok=True)
    imp.to_csv(HERE / "results" / f"{tag}__{a.scheme}__term_importance.csv", index=False)
    out["top_terms"] = imp.head(15).to_dict("records")
    write_json(HERE / "results" / f"{tag}__{a.scheme}.json", out)
    joblib.dump(ebm, CACHE / f"11_{tag}__{a.scheme}.joblib")


if __name__ == "__main__":
    main()
