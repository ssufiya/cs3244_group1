"""
Voting and Minimum-Bayes-Risk (MBR) ensembles over the existing baseline models.

Members (whatever is available on both validation and test of a set):
  GBDT   lgbm / xgb  x  reply_tfidf / reply_tfidf_surface      (committed)
  NB/LR  lr / mnb    x  reply_tfidf / reply_tfidf_surface      (committed; CNB is
         rank-identical to MNB on binary data, so it is dropped to avoid double votes)
  SVM    Linear SVM, TF-IDF + lexicon + surface               (00_extra_baseline_predictions)
  EBM    Glassbox EBM, 340 named features                     (00_extra_baseline_predictions)

Protocol
  * every member's score is Platt-calibrated on validation;
  * all ensemble parameters (calibrators, weights, member subsets, stacking
    coefficients, thresholds, MBR cost ratio) are chosen on validation with
    5-fold cross-fitting, so the validation numbers used for selection are
    out-of-fold; final calibrators/stackers are refitted on the whole
    validation partition and applied once to test;
  * news_headlines is a pure transfer set: the sarc_random models score it and
    every parameter is taken from sarc_random/val.

Methods
  best_single      LightGBM reply_tfidf_surface (the reference baseline)
  hard_vote        majority of calibrated members at 0.5 (score = vote share)
  soft_vote        mean of calibrated probabilities
  rank_avg         mean of within-set rank percentiles (score only)
  weighted_soft    linear pool, non-negative weights summing to 1, fitted for val log-loss
  greedy_select    Caruana forward selection with replacement, maximising val AUC
  stack_lr         logistic regression on member logits (log-linear pool)
  mbr_*            decision rules that minimise expected risk under the
                   calibrated stack_lr posterior (see README / report)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (EVAL_SETS, SCHEME, SEED, load_predictions, paired_bootstrap_auc,  # noqa: E402
                    scores, select_threshold, write_json)

OUT = HERE / "results"
REFERENCE = "lgbm_reply_tfidf_surface"
DROP = {"cnb_reply_tfidf", "cnb_reply_tfidf_surface"}
EPS = 1e-6
K = 5


# --------------------------------------------------------------------------
def to_logit(s: np.ndarray) -> np.ndarray:
    """Probabilities -> logit; margins (outside [0,1]) are used as-is."""
    if s.min() >= 0 and s.max() <= 1:
        return logit(np.clip(s, EPS, 1 - EPS))
    return s


class Platt:
    def fit(self, s, y):
        self.m = LogisticRegression(C=1e6, max_iter=1000).fit(to_logit(s)[:, None], y)
        return self

    def __call__(self, s):
        return self.m.predict_proba(to_logit(s)[:, None])[:, 1]


def calibrate(val: pd.DataFrame, test: pd.DataFrame, members: list[str], y: np.ndarray, folds):
    """Cross-fitted calibrated val probs and full-val calibrated test probs."""
    Pv = np.zeros((len(val), len(members)))
    Pt = np.zeros((len(test), len(members)))
    for j, m in enumerate(members):
        sv = val[m].to_numpy()
        for tr, te in folds:
            Pv[te, j] = Platt().fit(sv[tr], y[tr])(sv[te])
        Pt[:, j] = Platt().fit(sv, y)(test[m].to_numpy())
    return Pv, Pt


def fit_linear_pool(P, y):
    k = P.shape[1]

    def loss(w):
        w = np.abs(w) / np.abs(w).sum()
        return log_loss(y, np.clip(P @ w, EPS, 1 - EPS))
    r = minimize(loss, np.full(k, 1 / k), method="Powell", options={"maxiter": 4000, "xtol": 1e-4})
    w = np.abs(r.x)
    return w / w.sum()


def greedy_select(P, y, steps=30):
    """Caruana et al. (2004) ensemble selection with replacement on AUC."""
    counts = np.zeros(P.shape[1])
    cur = np.zeros(len(y))
    best_auc, history = -1, []
    for t in range(1, steps + 1):
        cand = [roc_auc_score(y, (cur * (t - 1) + P[:, j]) / t) for j in range(P.shape[1])]
        j = int(np.argmax(cand))
        if cand[j] <= best_auc + 1e-6 and t > 1:
            break
        best_auc = cand[j]
        counts[j] += 1
        cur = (cur * (t - 1) + P[:, j]) / t
        history.append(best_auc)
    return counts / counts.sum(), history


def rank_pct(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def mbr_expected_f1(p):
    """Corpus-level MBR for the F1 utility (plug-in approximation, Jansche 2007;
    Nan Ye et al. 2012): label the top-k by posterior, k maximising
    E[F1] ~= 2*sum_{i<=k} p_i / (k + sum_i p_i). Uses no labels."""
    order = np.argsort(-p)
    cs = np.cumsum(p[order])
    k = np.arange(1, len(p) + 1)
    ef1 = 2 * cs / (k + p.sum())
    kb = int(np.argmax(ef1)) + 1
    pred = np.zeros(len(p), bool)
    pred[order[:kb]] = True
    return pred, float(ef1[kb - 1])


def mbr_expected_precision(p, target):
    """Largest prefix of the posterior ranking whose expected precision
    (mean posterior of the labelled set) stays >= target. Uses no labels."""
    order = np.argsort(-p)
    ep = np.cumsum(p[order]) / np.arange(1, len(p) + 1)
    ok = np.flatnonzero(ep >= target)
    kb = int(ok.max()) + 1 if len(ok) else 0
    pred = np.zeros(len(p), bool)
    pred[order[:kb]] = True
    return pred


# --------------------------------------------------------------------------
def run_set(name: str, groups: list[str] | None) -> tuple[list[dict], dict]:
    scheme = SCHEME[name]
    vds, vpart = EVAL_SETS[name]["val"]
    tds, tpart = EVAL_SETS[name]["test"]
    val = load_predictions(vds, vpart, scheme=scheme, groups=groups)
    test = load_predictions(tds, tpart, scheme=scheme, groups=groups)
    members = [c for c in test.columns if c not in ("row_position", "label") and c in val.columns
               and c not in DROP]
    yv, yt = val["label"].to_numpy(), test["label"].to_numpy()
    folds = list(KFold(K, shuffle=True, random_state=SEED).split(yv))
    Pv, Pt = calibrate(val, test, members, yv, folds)
    ref = members.index(REFERENCE)
    print(f"\n=== {name}: {len(members)} members: {members}", flush=True)

    # ---- scores (val = out-of-fold, test = final) for each method
    S = {}
    for j, m in enumerate(members):
        S[f"single:{m}"] = (Pv[:, j], Pt[:, j])
    S["best_single"] = S[f"single:{REFERENCE}"]
    S["hard_vote"] = ((Pv >= 0.5).mean(1), (Pt >= 0.5).mean(1))
    S["soft_vote"] = (Pv.mean(1), Pt.mean(1))
    S["rank_avg"] = (np.mean([rank_pct(Pv[:, j]) for j in range(len(members))], 0),
                     np.mean([rank_pct(Pt[:, j]) for j in range(len(members))], 0))

    # weighted linear pool & greedy selection: weights from out-of-fold val
    w_pool = fit_linear_pool(Pv, yv)
    S["weighted_soft"] = (Pv @ w_pool, Pt @ w_pool)
    w_greedy, hist = greedy_select(Pv, yv)
    S["greedy_select"] = (Pv @ w_greedy, Pt @ w_greedy)

    # stacking on logits, cross-fitted on val
    Lv, Lt = logit(np.clip(Pv, EPS, 1 - EPS)), logit(np.clip(Pt, EPS, 1 - EPS))
    sv = np.zeros(len(yv))
    for tr, te in folds:
        sv[te] = LogisticRegression(C=1.0, max_iter=2000).fit(Lv[tr], yv[tr]).predict_proba(Lv[te])[:, 1]
    stack = LogisticRegression(C=1.0, max_iter=2000).fit(Lv, yv)
    S["stack_lr"] = (sv, stack.predict_proba(Lt)[:, 1])

    rows = []
    for meth, (pv, pt) in S.items():
        is_prob = meth != "rank_avg"
        thr_hi = select_threshold(yv, pv)                      # precision >= 0.75 on val
        r = {"set": name, "method": meth, "n_members": len(members),
             "val_auc_oof": roc_auc_score(yv, pv)}
        t05 = scores(yt, pt, 0.5) if is_prob else scores(yt, pt, pred=pt >= np.median(pv))
        r.update({f"test_{k}": v for k, v in t05.items()})
        if thr_hi is not None:
            hi = scores(yt, pt, thr_hi)
            r.update({"thr_p75": thr_hi, "test_p75_precision": hi["precision"],
                      "test_p75_recall": hi["recall"], "test_p75_f1": hi["f1"]})
        rows.append(r)

    # ---- MBR decision rules on the stacked posterior (calibrated on val)
    pv, pt = S["stack_lr"]
    # (1) 0-1 loss -> Bayes decision p >= 0.5 (== stack_lr row at 0.5)
    # (2) asymmetric cost: FP costs lam x FN -> predict 1 iff p >= lam/(1+lam);
    #     lam chosen on val as the smallest cost ratio that keeps precision >= 0.75
    lam_grid = np.round(np.arange(1.0, 4.01, 0.05), 2)
    lam_star = None
    for lam in lam_grid:
        pred = pv >= lam / (1 + lam)
        if pred.any() and (yv[pred] == 1).mean() >= 0.75:
            lam_star = float(lam)
            break
    extra = []
    for meth, pred_t, note in [
        ("mbr_01loss", pt >= 0.5, "0-1 loss, threshold 0.5 on calibrated posterior"),
        ("mbr_cost", pt >= lam_star / (1 + lam_star), f"C_FP/C_FN = {lam_star} (val: precision >= .75)"),
        ("mbr_expected_f1", mbr_expected_f1(pt)[0], "corpus-level expected-F1 maximiser"),
        ("mbr_expected_prec75", mbr_expected_precision(pt, 0.75), "max coverage s.t. E[precision] >= .75"),
    ]:
        r = {"set": name, "method": meth, "n_members": len(members), "note": note,
             "val_auc_oof": roc_auc_score(yv, pv)}
        r.update({f"test_{k}": v for k, v in scores(yt, pt, pred=pred_t).items()})
        extra.append(r)
    rows += extra

    # significance vs reference on test AUC
    for r in rows:
        if r["method"].startswith("single:") or r["method"] == "best_single":
            continue
        if r["method"] in S:
            b = paired_bootstrap_auc(yt, S[r["method"]][1], S["best_single"][1], n_boot=300)
            r.update({"d_auc_vs_lgbm": b["diff"], "d_auc_ci_low": b["ci_low"],
                      "d_auc_ci_high": b["ci_high"]})

    detail = {"members": members,
              "single_val_auc_raw": {m: float(roc_auc_score(yv, val[m])) for m in members},
              "weights_linear_pool": dict(zip(members, map(float, w_pool))),
              "weights_greedy": dict(zip(members, map(float, w_greedy))), "greedy_auc_path": hist,
              "stack_lr_coef": dict(zip(members, map(float, stack.coef_[0]))),
              "stack_lr_intercept": float(stack.intercept_[0]),
              "mbr_cost_ratio": lam_star,
              "error_correlation_val": pd.DataFrame(Pv - yv[:, None], columns=members).corr().round(3).to_dict()}
    np.save(OUT / f"{name}__test_scores.npy", np.column_stack([S[k][1] for k in
            ("best_single", "soft_vote", "weighted_soft", "greedy_select", "stack_lr")]))
    return rows, detail


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=["svm", "ebm"],
                    help="experiments/predictions folders to add to the committed baselines")
    ap.add_argument("--out", default="results")
    ap.add_argument("--drop", nargs="*", default=[], help="extra member names to leave out")
    a = ap.parse_args()
    OUT = HERE / a.out
    OUT.mkdir(exist_ok=True)
    DROP.update(a.drop)
    all_rows, details = [], {}
    for name in EVAL_SETS:
        rows, det = run_set(name, a.groups)
        all_rows += rows
        details[name] = det
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "ensemble_results.csv", index=False)
    write_json(OUT / "ensemble_details.json", details)
    cols = ["set", "method", "val_auc_oof", "test_roc_auc", "test_accuracy", "test_precision",
            "test_recall", "test_f1", "test_p75_recall", "d_auc_vs_lgbm", "d_auc_ci_low", "d_auc_ci_high"]
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.float_format", "{:.4f}".format):
        print(df[[c for c in cols if c in df]].to_string(index=False))


if __name__ == "__main__":
    main()
