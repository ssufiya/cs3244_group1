"""
A6 - EBM design ablation.

Same 66 features and settings as 11_full_convergence/ebm_full.py, with one
design choice changed at a time:
  ebm_no_interactions   main effects only (interactions=0)
  ebm_sample200k        fitted on a 200k-row random sample of train, as the team EBM was
The full EBM (0.8229) and the EBM without the text score (0.7123) come from round 11.

  python 13_ablations/ebm_design.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "11_full_convergence"))
from common import SEED, row_ids, save_predictions, scores, select_threshold, write_json  # noqa: E402
from ebm_full import features  # noqa: E402

VARIANTS = {"ebm_no_interactions": {"interactions": 0, "sample": None},
            "ebm_sample200k": {"interactions": 15, "sample": 200_000}}
(HERE / "results").mkdir(exist_ok=True)


def main():
    from interpret.glassbox import ExplainableBoostingClassifier
    scheme = "sarc_random"
    Xtr, ytr, _ = features(scheme, scheme, "train")
    Xva, yva, dva = features(scheme, scheme, "val")
    Xte, yte, dte = features(scheme, scheme, "test")
    out = {}
    for name, v in VARIANTS.items():
        t0 = time.time()
        X, y = Xtr, ytr
        if v["sample"]:
            idx = np.random.default_rng(SEED).choice(len(ytr), v["sample"], replace=False)
            X, y = Xtr.iloc[idx], ytr[idx]
        ebm = ExplainableBoostingClassifier(
            random_state=SEED, n_jobs=16, outer_bags=8, inner_bags=0, interactions=v["interactions"],
            max_bins=256, learning_rate=0.02, early_stopping_rounds=100, max_rounds=20000,
            min_samples_leaf=50, validation_size=0.15)
        ebm.fit(X, y)
        pv, pt = ebm.predict_proba(Xva)[:, 1], ebm.predict_proba(Xte)[:, 1]
        save_predictions("13_ebm_design", scheme, "val", yva, {f"{scheme}__a6_{name}": pv}, ids=row_ids(dva))
        save_predictions("13_ebm_design", scheme, "test", yte, {f"{scheme}__a6_{name}": pt}, ids=row_ids(dte))
        thr = select_threshold(yva, pv)
        s = scores(yte, pt)
        if thr is not None:
            s["p75_recall"] = scores(yte, pt, thr)["recall"]
        out[name] = {**v, "train_rows": len(y), "n_terms": len(ebm.term_names_), "val_auc": roc_auc_score(yva, pv),
                     "test": s, "fit_seconds": round(time.time() - t0)}
        print(f"[{name}] val {out[name]['val_auc']:.4f} test {s['roc_auc']:.4f} ({out[name]['fit_seconds']}s)", flush=True)
        write_json(HERE / "results" / "a6_ebm_design.json", out)


if __name__ == "__main__":
    main()
