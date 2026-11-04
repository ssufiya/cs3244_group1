"""
A1 - leave-one-block-out for the best compliant single model.

Full model (lgbm_lrscore_cues_contrast, 0.8296): reply word TF-IDF (20k) +
17 surface counts + 25 cue features + 18 contrast features + the out-of-fold
word+char LR score. Each variant removes one block and is retrained with the
same frozen LightGBM hyper-parameters, early stopping on sarc_random/val;
sarc_subreddit reuses the tree count chosen on sarc_random. The full model is
refitted here too so every variant comes from the same run.

  python 13_ablations/feature_blocks.py
"""
from __future__ import annotations

import importlib.util
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
from common import SURFACE, paired_bootstrap_auc, row_ids, save_predictions, scores, select_threshold, write_json  # noqa: E402
from features_extra import contrast_block, cue_block  # noqa: E402
from lgbm_runner import PARAMS, base_matrices, cached_block, frame  # noqa: E402

_spec = importlib.util.spec_from_file_location("lrscore", ROOT / "06_linear_score_as_feature" / "run.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
lr_score_block = _mod.lr_score_block

BLOCKS = ["tfidf", "surface", "cues", "contrast", "lrscore"]
VARIANTS = {"full": BLOCKS, **{f"no_{b}": [x for x in BLOCKS if x != b] for b in BLOCKS}}
(HERE / "results").mkdir(exist_ok=True)


def matrices(scheme):
    base = {k: v for k, v in base_matrices(scheme).items() if k[0] == scheme}
    lr = lr_score_block(scheme, list(base))
    n_surf = len(SURFACE)
    out = {}
    for key, X in base.items():
        out[key] = {"tfidf": X[:, :-n_surf], "surface": X[:, -n_surf:],
                    "cues": sp.csr_matrix(cached_block("cues", cue_block, *key)),
                    "contrast": sp.csr_matrix(cached_block("contrast", contrast_block, *key)),
                    "lrscore": sp.csr_matrix(lr[key])}
    return out


def main():
    res, trees, preds = {}, {}, {}
    for scheme in ("sarc_random", "sarc_subreddit"):
        mats = matrices(scheme)
        ytr, yva, yte = (frame(scheme, p)["label"].to_numpy() for p in ("train", "val", "test"))
        for name, blocks in VARIANTS.items():
            t0 = time.time()
            X = {p: sp.hstack([mats[(scheme, p)][b] for b in blocks]).tocsr() for p in ("train", "val", "test")}
            dtr = lgb.Dataset(X["train"], ytr, free_raw_data=True)
            if scheme == "sarc_random":
                bst = lgb.train(PARAMS, dtr, num_boost_round=8000,
                                valid_sets=[lgb.Dataset(X["val"], yva, reference=dtr)],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
                trees[name] = bst.best_iteration
            else:
                bst = lgb.train(PARAMS, dtr, num_boost_round=trees[name])
            pv = bst.predict(X["val"], num_iteration=trees[name])
            pt = bst.predict(X["test"], num_iteration=trees[name])
            preds[(scheme, name)] = pt
            for part, y, p in (("val", yva, pv), ("test", yte, pt)):
                save_predictions("13_feature_blocks", scheme, part, y, {f"{scheme}__a1_{name}": p},
                                 ids=row_ids(frame(scheme, part)))
            thr = select_threshold(yva, pv)
            s = scores(yte, pt)
            if thr is not None:
                s["p75_recall"] = scores(yte, pt, thr)["recall"]
            res.setdefault(name, {"blocks": blocks})[scheme] = {
                "n_features": int(X["train"].shape[1]), "n_trees": trees[name], "val_auc": roc_auc_score(yva, pv),
                "test": s, "seconds": round(time.time() - t0)}
            print(f"[{name}] {scheme}: val {roc_auc_score(yva, pv):.4f} test {s['roc_auc']:.4f} "
                  f"trees {trees[name]} ({time.time()-t0:.0f}s)", flush=True)
        for name in VARIANTS:
            if name != "full":
                res[name][scheme]["vs_full"] = paired_bootstrap_auc(yte, preds[(scheme, name)],
                                                                    preds[(scheme, "full")], n_boot=300)
        write_json(HERE / "results" / "a1_feature_blocks.json", res)


if __name__ == "__main__":
    main()
