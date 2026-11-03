"""
Ablations that only need saved predictions (no retraining).

A4  stacking members: drop one model family at a time from the compliant stack
A5  stacking design: calibration, logit vs probability inputs, LR vs gated
    LightGBM meta-learner, meta-features, cross-fitting
A9  author overlap: score the main models on test rows whose author never
    appears in the training set

The ablation unit is the L2 logistic-regression stack over calibrated logits
(stack_lr in 11_full_convergence/ensemble_v3.py); every combiner there was
within 0.001 of it. Same protocol: everything is fitted on validation with
5-fold cross-fitting and test is scored once.

  python 13_ablations/stacking_and_author.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_ensemble_voting_mbr"))
sys.path.insert(0, str(ROOT / "11_full_convergence"))
from common import SEED, load_frame, load_predictions, paired_bootstrap_auc, write_json  # noqa: E402
from ensemble import EPS, calibrate  # noqa: E402
from ensemble_v3 import DROP, LGB_META, META, crossfit, lgb_meta, stack_fit  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
K = 5
PRETRAINED = {"roberta_pair", "roberta_reply", "roberta_pair_conv", "roberta_pair_conv_s4248",
              "lgbm_minilm_v60k_cues_contrast", "lgbm_combined_best"}
FAMILIES = {
    "GBDT baseline": ["lgbm_reply_tfidf", "lgbm_reply_tfidf_surface", "xgb_reply_tfidf", "xgb_reply_tfidf_surface"],
    "GBDT improved": ["lgbm_cues", "lgbm_contrast", "lgbm_cues_contrast", "lgbm_v60k_cues_contrast", "lgbm_lrscore",
                      "lgbm_lrscore_cues_contrast", "lgbm_lsa", "lgbm_cues_contrast_lsa"],
    "Linear": ["lr_reply_tfidf", "lr_reply_tfidf_surface", "lr_word_char_surface", "nblr_word_char_surface",
               "svm_tfidf_lex_surface"],
    "Naive Bayes": ["mnb_reply_tfidf", "mnb_reply_tfidf_surface"],
    "Glass-box": ["ebm_full", "ebm_full_notext", "ebm_named340"],
    "Scratch NN": ["scratch_reply", "scratch_pair"],
}
BASELINE = "lgbm_reply_tfidf_surface"
BEST_SINGLE = "lgbm_lrscore_cues_contrast"


def load_set(name):
    val = load_predictions(name, "val", scheme=name)
    test = load_predictions(name, "test", scheme=name)
    members = [c for c in test.columns if c not in ("row_position", "label") and c in val.columns
               and c not in DROP | PRETRAINED]
    return val, test, members


def lr_stack(Lv, Lt, yv, folds):
    oof = crossfit(lambda a, b: stack_fit(Lv[a], yv[a]).decision_function(Lv[b]), len(yv), folds)
    return oof, stack_fit(Lv, yv).decision_function(Lt)


def row(name, label, yv, yt, pv, pt, ref_t=None, **extra):
    r = {"set": name, "variant": label, "val_auc_oof": roc_auc_score(yv, pv), "test_auc": roc_auc_score(yt, pt), **extra}
    if ref_t is not None:
        b = paired_bootstrap_auc(yt, pt, ref_t, n_boot=300)
        r.update({"d_auc_vs_full": b["diff"], "ci_low": b["ci_low"], "ci_high": b["ci_high"]})
    return r


def a4_members(name):
    val, test, members = load_set(name)
    yv, yt = val["label"].to_numpy(), test["label"].to_numpy()
    folds = list(KFold(K, shuffle=True, random_state=SEED).split(yv))
    Pv, Pt = calibrate(val, test, members, yv, folds)
    Lv, Lt = logit(np.clip(Pv, EPS, 1 - EPS)), logit(np.clip(Pt, EPS, 1 - EPS))
    full_v, full_t = lr_stack(Lv, Lt, yv, folds)
    rows = [row(name, "all compliant members", yv, yt, full_v, full_t, n_members=len(members))]
    for fam, ms in FAMILIES.items():
        keep = [j for j, m in enumerate(members) if m not in ms]
        if len(keep) == len(members):
            continue
        pv, pt = lr_stack(Lv[:, keep], Lt[:, keep], yv, folds)
        rows.append(row(name, f"without {fam}", yv, yt, pv, pt, full_t, n_members=len(keep)))
        only = [j for j, m in enumerate(members) if m in ms]
        pv, pt = lr_stack(Lv[:, only], Lt[:, only], yv, folds)
        rows.append(row(name, f"only {fam}", yv, yt, pv, pt, full_t, n_members=len(only)))
        print(f"  {name}: {fam} done", flush=True)
    return rows, (val, test, members, Pv, Pt, Lv, Lt, folds, full_v, full_t)


def a5_design(name, cache):
    val, test, members, Pv, Pt, Lv, Lt, folds, full_v, full_t = cache
    yv, yt = val["label"].to_numpy(), test["label"].to_numpy()
    rows = [row(name, "reference: calibrated logits, LR, cross-fitted", yv, yt, full_v, full_t)]

    # no calibration: raw member scores, logit-transformed when they are probabilities, then standardised
    def raw(df):
        cols = []
        for m in members:
            x = df[m].to_numpy(np.float64)
            cols.append(logit(np.clip(x, EPS, 1 - EPS)) if x.min() >= 0 and x.max() <= 1 else x)
        return np.column_stack(cols)
    Rv, Rt = raw(val), raw(test)
    mu, sd = Rv.mean(0), Rv.std(0) + 1e-9
    pv, pt = lr_stack((Rv - mu) / sd, (Rt - mu) / sd, yv, folds)
    rows.append(row(name, "no calibration", yv, yt, pv, pt, full_t))

    pv, pt = lr_stack(Pv, Pt, yv, folds)
    rows.append(row(name, "probability inputs instead of logits", yv, yt, pv, pt, full_t))

    Mv = np.log1p(load_frame(name, "val", columns=META)[META].astype(np.float32).fillna(0).to_numpy())
    Mt = np.log1p(load_frame(name, "test", columns=META)[META].astype(np.float32).fillna(0).to_numpy())
    for label, Xv, Xt in [("LightGBM meta-learner, no meta-features", Lv, Lt),
                          ("LightGBM meta-learner + meta-features", np.hstack([Lv, Mv]), np.hstack([Lt, Mt]))]:
        cvres = lgb.cv(LGB_META, lgb.Dataset(Xv, yv), num_boost_round=3000, folds=folds, metrics="auc",
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds = len(cvres["valid auc-mean"])
        pv = crossfit(lambda a, b: lgb_meta(Xv[a], yv[a], Xv[b], rounds), len(yv), folds)
        pt = lgb_meta(Xv, yv, Xt, rounds)
        rows.append(row(name, label, yv, yt, pv, pt, full_t, meta_rounds=rounds))

    # no cross-fitting: calibrate and stack on all of validation, report the in-sample val AUC
    Pv_full = np.column_stack([LogisticRegression().fit(Lv[:, [j]], yv).predict_proba(Lv[:, [j]])[:, 1]
                               for j in range(len(members))])
    Lf = logit(np.clip(Pv_full, EPS, 1 - EPS))
    m = stack_fit(Lf, yv)
    r = row(name, "no cross-fitting (in-sample val AUC)", yv, yt, m.decision_function(Lf), full_t)
    r["d_auc_vs_full"] = 0.0
    rows.append(r)
    print(f"  {name}: design variants done", flush=True)
    return rows


def a9_author(cache):
    val, test, members, Pv, Pt, Lv, Lt, folds, full_v, full_t = cache
    train_auth = set(load_frame("sarc_random", "train", columns=["author"])["author"].astype(str))
    te_auth = load_frame("sarc_random", "test", columns=["author"])["author"].astype(str)
    assert len(te_auth) == len(test)
    unseen = ~te_auth.isin(train_auth).to_numpy()
    yt = test["label"].to_numpy()
    cols = {m: test[m].to_numpy() for m in [BASELINE, BEST_SINGLE, "lr_word_char_surface", "ebm_full", "scratch_pair"]}
    cols["compliant LR stack"] = full_t
    full = load_predictions("sarc_random", "test", scheme="sarc_random")
    for m in ["roberta_pair_conv"]:
        cols[f"{m} (reference)"] = full[m].to_numpy()
    rows = []
    for name, p in cols.items():
        r = {"model": name, "auc_all": roc_auc_score(yt, p), "auc_seen_authors": roc_auc_score(yt[~unseen], p[~unseen]),
             "auc_unseen_authors": roc_auc_score(yt[unseen], p[unseen])}
        r["drop_unseen_vs_seen"] = r["auc_unseen_authors"] - r["auc_seen_authors"]
        rows.append(r)
    b = paired_bootstrap_auc(yt[unseen], full_t[unseen], cols[BASELINE][unseen], n_boot=300)
    info = {"test_rows": int(len(yt)), "unseen_author_rows": int(unseen.sum()),
            "unseen_share": float(unseen.mean()), "unseen_sarcastic_rate": float(yt[unseen].mean()),
            "seen_sarcastic_rate": float(yt[~unseen].mean()),
            "stack_vs_baseline_on_unseen": b}
    print("  author check done", flush=True)
    return rows, info


def main():
    a4, a5 = [], []
    for name in ["sarc_random", "sarc_subreddit"]:
        rows, cache = a4_members(name)
        a4 += rows
        if name == "sarc_random":
            a5 += a5_design(name, cache)
            a9_rows, a9_info = a9_author(cache)
    pd.DataFrame(a4).to_csv(OUT / "a4_stacking_members.csv", index=False)
    pd.DataFrame(a5).to_csv(OUT / "a5_stacking_design.csv", index=False)
    pd.DataFrame(a9_rows).to_csv(OUT / "a9_author_overlap.csv", index=False)
    write_json(OUT / "a9_author_overlap.json", a9_info)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        for f in ["a4_stacking_members", "a5_stacking_design", "a9_author_overlap"]:
            print(f"\n{f}\n", pd.read_csv(OUT / f"{f}.csv").to_string(index=False))
    print(a9_info)


if __name__ == "__main__":
    main()
