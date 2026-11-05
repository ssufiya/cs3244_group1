"""
Run after both queues finish.

seeds     single-model test AUC over three seeds (3244, 3245, 3246) for the four
          seeded compliant models, and the compliant LR stack refitted once per
          seed with those members swapped to that seed (mean and std over seeds)
temporal  the step-7 ensemble framework (ensemble_v3.run_set) on sarc_temporal with
          every compliant member that has sarc_temporal predictions, plus each
          single model against the LightGBM baseline

  python 14_final_checks/seeds_and_temporal_stack.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_ensemble_voting_mbr"))
sys.path.insert(0, str(ROOT / "11_full_convergence"))
sys.path.insert(0, str(ROOT / "13_ablations"))
import ensemble_v3  # noqa: E402
from common import PRED_DIR, SEED, load_predictions, paired_bootstrap_auc, write_json  # noqa: E402
from ensemble import EPS, calibrate  # noqa: E402
from ensemble_v3 import DROP, crossfit, stack_fit  # noqa: E402
from stacking_and_author import PRETRAINED  # noqa: E402

OUT = HERE / "results"
SEEDED = ["lgbm_lrscore_cues_contrast", "ebm_full", "scratch_pair", "scratch_reply"]
SEEDS = [3244, 3245, 3246]
BASELINE = "lgbm_reply_tfidf_surface"


def seeds():
    extra = [p.name for p in PRED_DIR.glob("14_seed_*")]
    val = load_predictions("sarc_random", "val", scheme="sarc_random", extra=extra)
    test = load_predictions("sarc_random", "test", scheme="sarc_random", extra=extra)
    yv, yt = val["label"].to_numpy(), test["label"].to_numpy()
    base = [c for c in test.columns if c not in ("row_position", "label") and c in val.columns
            and c not in DROP | PRETRAINED and not c.endswith(("_s3245", "_s3246"))]
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(yv))
    single = {m: {} for m in SEEDED}
    stack = {}
    for s in SEEDS:
        cols = {m: (m if s == SEED else f"{m}_s{s}") for m in SEEDED}
        for m, c in cols.items():
            single[m][s] = roc_auc_score(yt, test[c])
        v, t = val[base].copy(), test[base].copy()
        for m, c in cols.items():
            v[m], t[m] = val[c], test[c]
        Pv, Pt = calibrate(v, t, base, yv, folds)
        Lv, Lt = logit(np.clip(Pv, EPS, 1 - EPS)), logit(np.clip(Pt, EPS, 1 - EPS))
        oof = crossfit(lambda a, b: stack_fit(Lv[a], yv[a]).decision_function(Lv[b]), len(yv), folds)
        pt = stack_fit(Lv, yv).decision_function(Lt)
        stack[s] = {"val_auc_oof": roc_auc_score(yv, oof), "test_auc": roc_auc_score(yt, pt),
                    "vs_baseline": roc_auc_score(yt, pt) - roc_auc_score(yt, test[BASELINE])}
        print(f"seed {s}: stack test {stack[s]['test_auc']:.4f}", flush=True)
    summ = {m: {"mean": float(np.mean(list(d.values()))), "std": float(np.std(list(d.values()), ddof=1)),
                "per_seed": d} for m, d in single.items()}
    a = [r["test_auc"] for r in stack.values()]
    summ["compliant LR stack"] = {"mean": float(np.mean(a)), "std": float(np.std(a, ddof=1)),
                                  "per_seed": {s: r["test_auc"] for s, r in stack.items()},
                                  "members": len(base)}
    return summ


def temporal():
    ensemble_v3.OUT = OUT / "ensemble_sarc_temporal"
    ensemble_v3.OUT.mkdir(parents=True, exist_ok=True)
    rows, detail = ensemble_v3.run_set("sarc_temporal", set(PRETRAINED))
    pd.DataFrame(rows).to_csv(ensemble_v3.OUT / "ensemble_results.csv", index=False)
    write_json(ensemble_v3.OUT / "ensemble_details.json", detail)
    val = load_predictions("sarc_temporal", "val", scheme="sarc_temporal")
    test = load_predictions("sarc_temporal", "test", scheme="sarc_temporal")
    yt = test["label"].to_numpy()
    singles = []
    for m in [c for c in test.columns if c not in ("row_position", "label") and c in val.columns]:
        r = {"model": m, "test_auc": roc_auc_score(yt, test[m])}
        if m != BASELINE:
            b = paired_bootstrap_auc(yt, test[m].to_numpy(), test[BASELINE].to_numpy(), n_boot=300)
            r.update({"d_vs_baseline": b["diff"], "ci_low": b["ci_low"], "ci_high": b["ci_high"]})
        singles.append(r)
    pd.DataFrame(singles).sort_values("test_auc", ascending=False).to_csv(OUT / "temporal_single_models.csv", index=False)
    return rows


def main():
    s = seeds()
    write_json(OUT / "seed_variability.json", s)
    rows = temporal()
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(pd.read_csv(OUT / "temporal_single_models.csv").to_string(index=False))
        print(pd.DataFrame(rows)[["method", "n_members", "val_auc_oof", "test_roc_auc", "d_auc_vs_best_single"]].to_string(index=False))
    print(s)


if __name__ == "__main__":
    main()
