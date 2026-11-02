"""
Improvement 3 - richer text representation for the linear models.

The baseline LR sees word 1-2-gram TF-IDF only. Sarcasm on Reddit is partly
orthographic (elongation "sooo", "!!!", "?!", capitalisation, "yeaaah",
quote marks), which word tokens throw away. Two variants:

  lr_word_char_surface    LR on word 1-2-gram TF-IDF + char_wb 2-5-gram TF-IDF
                          + scaled log surface counts
  nblr_word_char_surface  NB-SVM style (Wang & Manning 2012): binary word+char
                          features scaled by the naive-Bayes log-count ratio r,
                          then LR; interpolates NB's robustness with LR's fit

C is chosen on sarc_random/val (grid below) and frozen for sarc_subreddit.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (SURFACE, load_frame, row_ids, save_predictions, scores,  # noqa: E402
                    select_threshold, text_of, write_json)

TRANSFER = ["news_headlines", "figlang_reddit", "figlang_twitter", "tweeteval_irony"]
#: Validation grids. LR word+char was searched over 0.25/0.5/1/2 in the first (interrupted)
#: run -> .8106/.8139/.8142/.8111, so only the winner is refitted; NB-LR fell from
#: .8082 (0.25) to .8017 (0.5), so its grid continues downwards.
C_GRIDS = {"lr_word_char_surface": [1.0], "nblr_word_char_surface": [0.05, 0.1, 0.25]}
(HERE / "results").mkdir(exist_ok=True)


def build(scheme):
    tr = load_frame(scheme, "train")
    keys = [(scheme, "val"), (scheme, "test")] + ([(t, "test") for t in TRANSFER] if scheme == "sarc_random" else [])
    frames = {k: load_frame(*k) for k in keys}
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=400_000, dtype=np.float32)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, sublinear_tf=True,
                        max_features=400_000, dtype=np.float32, lowercase=False)
    ss = StandardScaler()

    def surf(df):
        return np.log1p(df.reindex(columns=SURFACE).astype(np.float32).fillna(0).clip(lower=0).to_numpy())

    t0 = time.time()
    Xtr = sp.hstack([w.fit_transform(text_of(tr)), c.fit_transform(text_of(tr)),
                     sp.csr_matrix(ss.fit_transform(surf(tr)))]).tocsr()
    X = {k: sp.hstack([w.transform(text_of(d)), c.transform(text_of(d)),
                       sp.csr_matrix(ss.transform(surf(d)))]).tocsr() for k, d in frames.items()}
    print(f"[{scheme}] features {Xtr.shape} ({time.time()-t0:.0f}s)", flush=True)
    y = {k: d["label"].to_numpy() for k, d in frames.items()}
    ids = {k: row_ids(d) for k, d in frames.items()}
    return Xtr, tr["label"].to_numpy(), X, y, ids


def nb_ratio(X, y, alpha=1.0):
    Xb = (X > 0).astype(np.float32)
    p = alpha + Xb[y == 1].sum(0)
    q = alpha + Xb[y == 0].sum(0)
    return np.asarray(np.log((p / p.sum()) / (q / q.sum()))).ravel().astype(np.float32)


def nb_transform(X, r, n_text):
    """Binary text features scaled by r; surface columns (after n_text) kept as-is."""
    T = (X[:, :n_text] > 0).astype(np.float32).multiply(r[:n_text]).tocsr()
    return sp.hstack([T, X[:, n_text:]]).tocsr()


def fit_lr(X, y, C):
    return LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(X, y)


def main():
    out, chosen = {}, {}
    for scheme in ("sarc_random", "sarc_subreddit"):
        Xtr, ytr, X, y, ids = build(scheme)
        n_text = Xtr.shape[1] - len(SURFACE)
        r = nb_ratio(Xtr[:, :n_text], ytr)
        variants = {"lr_word_char_surface": (Xtr, X),
                    "nblr_word_char_surface": (nb_transform(Xtr, r, n_text),
                                               {k: nb_transform(v, r, n_text) for k, v in X.items()})}
        vk = (scheme, "val")
        for name, (A, B) in variants.items():
            t0 = time.time()
            if scheme == "sarc_random":
                trials = {}
                for C in C_GRIDS[name]:
                    m = fit_lr(A, ytr, C)
                    trials[C] = roc_auc_score(y[vk], m.predict_proba(B[vk])[:, 1])
                    print(f"  {name} C={C}: val {trials[C]:.4f}", flush=True)
                chosen[name] = max(trials, key=trials.get)
                out.setdefault(name, {})["c_trials"] = trials
            m = fit_lr(A, ytr, chosen[name])
            pv = m.predict_proba(B[vk])[:, 1]
            thr = select_threshold(y[vk], pv)
            res = {"C": chosen[name], "val_auc": roc_auc_score(y[vk], pv), "seconds": round(time.time() - t0)}
            for k, M in B.items():
                p = m.predict_proba(M)[:, 1]
                save_predictions("linear_char", k[0], k[1], y[k], {f"{scheme}__{name}": p}, ids=ids[k])
                if k[1] == "test":
                    s = scores(y[k], p)
                    if k[0] == scheme and thr is not None:
                        hi = scores(y[k], p, thr)
                        s.update({"p75_precision": hi["precision"], "p75_recall": hi["recall"]})
                    res[k[0]] = s
            out.setdefault(name, {})[scheme] = res
            print(f"[{name}] {scheme}: val {res['val_auc']:.4f} test {res[scheme]['roc_auc']:.4f}", flush=True)
        del Xtr, X, variants
    write_json(HERE / "results" / "summary.json", out)


if __name__ == "__main__":
    main()
