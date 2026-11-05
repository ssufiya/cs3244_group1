"""
Two quick checks that need no training.

endings     A8 follow-up: do the class differences in how replies end also appear in
            the human-annotated sets (TweetEval, FigLang) and in news headlines, whose
            labels do not come from a stripped "/s" marker? Reports the last-character
            distribution by label and the AUC of "does not end with a full stop".
team_stack  The team notebook modelling/Ensemble/stacking&voting.ipynb combines
            LightGBM, XGBoost, LR and linear SVM. Its reported AUCs are computed from
            0/1 predictions, which caps them. Here the same four members are combined
            with the step-7 protocol and AUC is computed on scores.

  python 14_final_checks/quick_checks.py
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
from common import SEED, load_frame, load_predictions, paired_bootstrap_auc, text_of, write_json  # noqa: E402
from ensemble import EPS, calibrate  # noqa: E402
from ensemble_v3 import crossfit, stack_fit  # noqa: E402

(HERE / "results").mkdir(exist_ok=True)
SETS = [("sarc_random", "test"), ("figlang_reddit", "test"), ("figlang_twitter", "test"),
        ("tweeteval_irony", "test"), ("news_headlines", "test")]
TEAM = ["lgbm_reply_tfidf_surface", "xgb_reply_tfidf_surface", "lr_reply_tfidf_surface", "svm_tfidf_lex_surface"]


def endings():
    rows = []
    for ds, part in SETS:
        frames = [load_frame(ds, p) for p in (["train", "val", "test"] if ds != "sarc_random" else ["test"])]
        df = pd.concat(frames, ignore_index=True)
        last = text_of(df).str.rstrip().str[-1:].fillna("")
        y = df["label"].to_numpy()
        no_stop = (last != ".").to_numpy().astype(float)
        r = {"set": ds, "rows": len(df), "auc_no_full_stop": roc_auc_score(y, no_stop)}
        for k, mask in {"full stop": last == ".", "exclamation": last == "!", "question": last == "?",
                        "letter or digit": last.str.match(r"[A-Za-z0-9]")}.items():
            m = mask.to_numpy()
            r[f"{k} | not sarcastic"] = float(m[y == 0].mean())
            r[f"{k} | sarcastic"] = float(m[y == 1].mean())
        rows.append(r)
    return rows


def team_stack():
    val = load_predictions("sarc_random", "val", scheme="sarc_random")
    test = load_predictions("sarc_random", "test", scheme="sarc_random")
    yv, yt = val["label"].to_numpy(), test["label"].to_numpy()
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(yv))
    Pv, Pt = calibrate(val, test, TEAM, yv, folds)
    Lv, Lt = logit(np.clip(Pv, EPS, 1 - EPS)), logit(np.clip(Pt, EPS, 1 - EPS))
    stack_t = stack_fit(Lv, yv).decision_function(Lt)
    stack_v = crossfit(lambda a, b: stack_fit(Lv[a], yv[a]).decision_function(Lv[b]), len(yv), folds)
    soft_t = Pt.mean(1)
    hard_t = (Pt >= 0.5).mean(1)
    out = {"members": TEAM,
           "single_test_auc": {m: roc_auc_score(yt, test[m]) for m in TEAM},
           "soft_vote_test_auc": roc_auc_score(yt, soft_t),
           "hard_vote_test_auc_on_vote_share": roc_auc_score(yt, hard_t),
           "stack_lr_val_auc_oof": roc_auc_score(yv, stack_v),
           "stack_lr_test_auc": roc_auc_score(yt, stack_t),
           "stack_lr_test_auc_on_0_1_labels": roc_auc_score(yt, (stack_t >= 0).astype(float)),
           "stack_vs_lgbm_baseline": paired_bootstrap_auc(yt, stack_t, test[TEAM[0]].to_numpy(), n_boot=300)}
    return out


def main():
    e = endings()
    pd.DataFrame(e).to_csv(HERE / "results" / "endings_by_dataset.csv", index=False)
    t = team_stack()
    write_json(HERE / "results" / "team_stack_comparison.json", t)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(e).to_string(index=False))
    print(t)


if __name__ == "__main__":
    main()
