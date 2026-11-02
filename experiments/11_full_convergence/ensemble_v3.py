"""
Round 11 - ensemble frameworks over every model, including the converged
RoBERTa and the full-data EBM.

Same protocol as round 01: members Platt-calibrated on validation, every
ensemble parameter cross-fitted (5 folds) on validation so the reported val AUC
is out-of-fold, then refitted on all of validation and applied once to test.

Methods
  soft_vote         mean calibrated probability
  greedy_select     Caruana forward selection with replacement (val AUC)
  stack_lr          L2 logistic regression on member logits (round 01's winner)
  stack_lr_l1       L1 stacking, C picked by inner CV -> automatic member pruning
  shapley_stack     cooperative-game view: each member is a player, the value of a
                    coalition is the held-out AUC of a logit stack over it; members
                    with a non-positive Monte-Carlo Shapley value are dropped and
                    the stack is refitted on the rest
  stack_lgbm_meta   context-aware (gated) stacking: LightGBM on member logits plus
                    5 meta features (reply/parent length, empty parent, ?, !) so
                    the combination can differ by input type
  blend_meta_lr     mean of stack_lr and stack_lgbm_meta probabilities
MBR decision rules (round 01) are applied to the method chosen on val.

  python 11_full_convergence/ensemble_v3.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "01_ensemble_voting_mbr"))
from common import (SEED, load_frame, load_predictions, paired_bootstrap_auc, scores,  # noqa: E402
                    select_threshold, write_json)
from ensemble import (EPS, calibrate, greedy_select, mbr_expected_f1,  # noqa: E402
                      mbr_expected_precision)

OUT = HERE / "results" / "ensemble"
DROP = {"cnb_reply_tfidf", "cnb_reply_tfidf_surface", "lgbm_base_repro"}   # rank-duplicates
META = ["n_words", "parent_n_words", "parent_is_empty", "n_question", "n_exclam"]
LGB_META = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
            "verbose": -1, "seed": SEED, "num_threads": 8, "deterministic": True}
K = 5


def stack_fit(L, y, C=1.0, penalty="l2"):
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    return LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=3000).fit(L, y)


def crossfit(fit_pred, n, folds):
    out = np.zeros(n)
    for tr, te in folds:
        out[te] = fit_pred(tr, te)
    return out


def shapley_values(L, y, anchor, n_perm=24, seed=SEED, sub=60_000):
    """Monte-Carlo permutation Shapley values of the members for the game
    v(S) = held-out AUC of a logit stack over S + {anchor} (fitted on one val half,
    scored on the other, both directions averaged); v(empty) = AUC of the anchor
    (the best single model) alone. A player's value is what it adds on top of the
    strongest model, so redundant members get ~0 or negative values."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))[: 2 * (min(sub, len(y)) // 2)]
    A, B = idx[: len(idx) // 2], idx[len(idx) // 2:]
    cache: dict = {}

    def v(S):
        key = frozenset(S) | {anchor}
        if key not in cache:
            cols = sorted(key)
            a = roc_auc_score(y[B], stack_fit(L[A][:, cols], y[A]).decision_function(L[B][:, cols]))
            b = roc_auc_score(y[A], stack_fit(L[B][:, cols], y[B]).decision_function(L[A][:, cols]))
            cache[key] = (a + b) / 2
        return cache[key]

    k = L.shape[1]
    players = np.array([j for j in range(k) if j != anchor])
    phi = np.zeros(k)
    base = v([])
    for _ in range(n_perm):
        S, prev = [], base
        for j in rng.permutation(players):
            S.append(int(j))
            cur = v(S)
            phi[j] += cur - prev
            prev = cur
    return phi / n_perm, v(range(k))


def lgb_meta(Xa, ya, Xb, rounds):
    m = lgb.train(LGB_META, lgb.Dataset(Xa, ya), num_boost_round=rounds)
    return m.predict(Xb)


def run_set(name: str, drop: set):
    val = load_predictions(name, "val", scheme=name)
    test = load_predictions(name, "test", scheme=name)
    members = [c for c in test.columns if c not in ("row_position", "label") and c in val.columns
               and c not in DROP | drop]
    yv, yt = val["label"].to_numpy(), test["label"].to_numpy()
    folds = list(KFold(K, shuffle=True, random_state=SEED).split(yv))
    Pv, Pt = calibrate(val, test, members, yv, folds)
    Lv, Lt = logit(np.clip(Pv, EPS, 1 - EPS)), logit(np.clip(Pt, EPS, 1 - EPS))
    print(f"\n=== {name}: {len(members)} members", flush=True)
    single_auc = {m: roc_auc_score(yv, Pv[:, j]) for j, m in enumerate(members)}
    best_m = max(single_auc, key=single_auc.get)
    S, detail = {}, {"members": members, "single_val_auc": single_auc, "best_single": best_m}
    S["best_single"] = (Pv[:, members.index(best_m)], Pt[:, members.index(best_m)])
    S["soft_vote"] = (Pv.mean(1), Pt.mean(1))
    w, _ = greedy_select(Pv, yv)
    S["greedy_select"] = (Pv @ w, Pt @ w)
    detail["greedy_weights"] = {m: float(x) for m, x in zip(members, w) if x > 0}

    full = stack_fit(Lv, yv)
    S["stack_lr"] = (crossfit(lambda a, b: stack_fit(Lv[a], yv[a]).predict_proba(Lv[b])[:, 1], len(yv), folds),
                     full.predict_proba(Lt)[:, 1])
    detail["stack_lr_coef"] = dict(zip(members, map(float, full.coef_[0])))

    t0 = time.time()
    cv = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1, 1.0], penalty="l1", solver="liblinear",
                              cv=folds, scoring="roc_auc", max_iter=3000).fit(Lv, yv)
    C1 = float(cv.C_[0])
    l1 = stack_fit(Lv, yv, C1, "l1")
    S["stack_lr_l1"] = (crossfit(lambda a, b: stack_fit(Lv[a], yv[a], C1, "l1").predict_proba(Lv[b])[:, 1],
                                 len(yv), folds), l1.predict_proba(Lt)[:, 1])
    detail["stack_l1"] = {"C": C1, "kept": [m for m, c in zip(members, l1.coef_[0]) if abs(c) > 1e-8]}
    print(f"  l1 stack C={C1} kept {len(detail['stack_l1']['kept'])} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    anchor = members.index(best_m)
    phi, v_all = shapley_values(Lv, yv, anchor)
    keep = [anchor] + [j for j in range(len(members)) if j != anchor and phi[j] > 0]
    S["shapley_stack"] = (crossfit(lambda a, b: stack_fit(Lv[a][:, keep], yv[a]).predict_proba(Lv[b][:, keep])[:, 1],
                                   len(yv), folds), stack_fit(Lv[:, keep], yv).predict_proba(Lt[:, keep])[:, 1])
    detail["shapley"] = {"values": dict(sorted(zip(members, map(float, phi)), key=lambda t: -t[1])),
                         "v_all": v_all, "kept": [members[j] for j in keep]}
    print(f"  shapley kept {len(keep)}/{len(members)} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    Mv = load_frame(name, "val", columns=META)[META].astype(np.float32).fillna(0).to_numpy()
    Mt = load_frame(name, "test", columns=META)[META].astype(np.float32).fillna(0).to_numpy()
    Xv, Xt = np.hstack([Lv, np.log1p(Mv)]), np.hstack([Lt, np.log1p(Mt)])
    cvres = lgb.cv(LGB_META, lgb.Dataset(Xv, yv), num_boost_round=3000, folds=folds, metrics="auc",
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = len(cvres["valid auc-mean"])
    S["stack_lgbm_meta"] = (crossfit(lambda a, b: lgb_meta(Xv[a], yv[a], Xv[b], rounds), len(yv), folds),
                            lgb_meta(Xv, yv, Xt, rounds))
    detail["stack_lgbm_meta_rounds"] = rounds
    print(f"  lgbm meta stack {rounds} rounds ({time.time()-t0:.0f}s)", flush=True)
    S["blend_meta_lr"] = ((S["stack_lr"][0] + S["stack_lgbm_meta"][0]) / 2,
                          (S["stack_lr"][1] + S["stack_lgbm_meta"][1]) / 2)

    rows = []
    for meth, (pv, pt) in S.items():
        thr = select_threshold(yv, pv)
        r = {"set": name, "method": meth, "n_members": len(members), "val_auc_oof": roc_auc_score(yv, pv)}
        r.update({f"test_{k}": v for k, v in scores(yt, pt).items()})
        if thr is not None:
            h = scores(yt, pt, thr)
            r.update({"test_p75_precision": h["precision"], "test_p75_recall": h["recall"]})
        b = paired_bootstrap_auc(yt, pt, S["best_single"][1], n_boot=300)
        r.update({"d_auc_vs_best_single": b["diff"], "ci_low": b["ci_low"], "ci_high": b["ci_high"]})
        rows.append(r)
    # the method chosen on out-of-fold val AUC gets the MBR operating points
    chosen = max((r for r in rows if r["method"] != "best_single"), key=lambda r: r["val_auc_oof"])["method"]
    detail["chosen_by_val"] = chosen
    pt = S[chosen][1]
    for meth, pred in [("mbr_expected_f1", mbr_expected_f1(pt)[0]),
                       ("mbr_expected_prec75", mbr_expected_precision(pt, 0.75))]:
        r = {"set": name, "method": f"{chosen}+{meth}", "n_members": len(members)}
        r.update({f"test_{k}": v for k, v in scores(yt, pt, pred=pred).items()})
        rows.append(r)
    np.save(OUT / f"{name}__test_{chosen}.npy", pt)
    return rows, detail


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ensemble")
    ap.add_argument("--drop", nargs="*", default=[])
    a = ap.parse_args()
    global OUT
    OUT = HERE / "results" / a.out
    OUT.mkdir(parents=True, exist_ok=True)
    rows, details = [], {}
    for name in ["sarc_random", "sarc_subreddit"]:
        r, d = run_set(name, set(a.drop))
        rows += r
        details[name] = d
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ensemble_results.csv", index=False)
    write_json(OUT / "ensemble_details.json", details)
    cols = ["set", "method", "val_auc_oof", "test_roc_auc", "test_precision", "test_recall", "test_f1",
            "test_p75_recall", "d_auc_vs_best_single", "ci_low", "ci_high"]
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(df[[c for c in cols if c in df]].to_string(index=False))


if __name__ == "__main__":
    main()
