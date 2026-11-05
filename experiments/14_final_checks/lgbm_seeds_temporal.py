"""
Final robustness checks for the best compliant LightGBM (lgbm_lrscore_cues_contrast).

  seeds      two more seeds (3245, 3246) on sarc_random; LightGBM uses bagging and
             feature subsampling, so the seed changes the model. Early stopping on
             sarc_random/val as in the original run.
  temporal   refit on sarc_temporal/train with the tree count frozen from
             sarc_random (same protocol as sarc_subreddit). The word+char LR that
             feeds the LR-score feature is refitted on sarc_temporal too, and its
             val/test scores are saved as the temporal lr_word_char_surface member.

  python 14_final_checks/lgbm_seeds_temporal.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "13_ablations"))
from common import row_ids, save_predictions, scores, select_threshold, write_json  # noqa: E402
from feature_blocks import BLOCKS, matrices  # noqa: E402
from lgbm_runner import PARAMS, frame  # noqa: E402

MODEL = "lgbm_lrscore_cues_contrast"
SEEDS = [3245, 3246]
(HERE / "results").mkdir(exist_ok=True)


def fit_score(scheme, mats, seed, trees=None):
    X = {p: sp.hstack([mats[(scheme, p)][b] for b in BLOCKS]).tocsr() for p in ("train", "val", "test")}
    y = {p: frame(scheme, p)["label"].to_numpy() for p in ("train", "val", "test")}
    prm = {**PARAMS, "seed": seed}
    dtr = lgb.Dataset(X["train"], y["train"], free_raw_data=True)
    if trees is None:
        bst = lgb.train(prm, dtr, num_boost_round=8000, valid_sets=[lgb.Dataset(X["val"], y["val"], reference=dtr)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        trees = bst.best_iteration
    else:
        bst = lgb.train(prm, dtr, num_boost_round=trees)
    pv, pt = bst.predict(X["val"], num_iteration=trees), bst.predict(X["test"], num_iteration=trees)
    thr = select_threshold(y["val"], pv)
    s = scores(y["test"], pt)
    if thr is not None:
        s["p75_recall"] = scores(y["test"], pt, thr)["recall"]
    return pv, pt, y, {"seed": seed, "n_trees": trees, "val_auc": roc_auc_score(y["val"], pv), "test": s}


def main():
    out = {"seeds": {}, "temporal": {}}
    t0 = time.time()
    mats = matrices("sarc_random")
    for seed in SEEDS:
        pv, pt, y, r = fit_score("sarc_random", mats, seed)
        name = f"{MODEL}_s{seed}"
        for part, yy, p in (("val", y["val"], pv), ("test", y["test"], pt)):
            save_predictions("14_seed_lgbm", "sarc_random", part, yy, {f"sarc_random__{name}": p},
                             ids=row_ids(frame("sarc_random", part)))
        out["seeds"][name] = r
        print(f"[{name}] val {r['val_auc']:.4f} test {r['test']['roc_auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        write_json(HERE / "results" / "lgbm_seeds_temporal.json", out)
    del mats

    scheme = "sarc_temporal"
    mats = matrices(scheme)
    trees = 496
    pv, pt, y, r = fit_score(scheme, mats, 3244, trees)
    for part, yy, p in (("val", y["val"], pv), ("test", y["test"], pt)):
        ids = row_ids(frame(scheme, part))
        save_predictions(MODEL, scheme, part, yy, {f"{scheme}__{MODEL}": p}, ids=ids)
        save_predictions("linear_char", scheme, part, yy,
                         {f"{scheme}__lr_word_char_surface": mats[(scheme, part)]["lrscore"].toarray().ravel()}, ids=ids)
    out["temporal"][MODEL] = r
    print(f"[{MODEL}] {scheme}: val {r['val_auc']:.4f} test {r['test']['roc_auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    write_json(HERE / "results" / "lgbm_seeds_temporal.json", out)


if __name__ == "__main__":
    main()
