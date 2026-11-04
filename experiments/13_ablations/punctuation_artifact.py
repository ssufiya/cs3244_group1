"""
A8 - is "ends with a full stop -> not sarcastic" a data artefact?

SARC labels come from the "/s" marker, which was stripped from the reply. If
removing it left a systematic difference in how replies end, a model could
learn the ending instead of sarcasm. The full EBM's strongest non-sarcastic
n-grams are a trailing ". " and "! ".

Test: strip trailing punctuation and whitespace from every reply, then
  * score the normally trained word+char LR on stripped val/test;
  * retrain the same LR on stripped train and score stripped val/test.
Surface counts are left out (they are computed from the unstripped text), so
the reference is the word+char LR from A2 (0.8166).

  python 13_ablations/punctuation_artifact.py
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_frame, paired_bootstrap_auc, row_ids, save_predictions, scores, text_of, write_json  # noqa: E402

TRAIL = re.compile(r"[\s.!?,;:…]+$")
(HERE / "results").mkdir(exist_ok=True)


def strip(s: pd.Series) -> pd.Series:
    return s.map(lambda t: TRAIL.sub("", t))


def ending_stats(df):
    t = text_of(df)
    last = t.str.rstrip().str[-1:].fillna("")
    kind = np.select([last == ".", last == "!", last == "?", last.str.match(r"[A-Za-z0-9]")],
                     ["full stop", "exclamation", "question", "letter or digit"], "other")
    tab = pd.crosstab(kind, df["label"], normalize="columns").rename(columns={0: "not sarcastic", 1: "sarcastic"})
    return tab.round(4).to_dict()


def fit_vec(texts):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=400_000, dtype=np.float32)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, sublinear_tf=True,
                        max_features=400_000, dtype=np.float32, lowercase=False)
    X = sp.hstack([w.fit_transform(texts), c.fit_transform(texts)]).tocsr()
    return X, lambda s: sp.hstack([w.transform(s), c.transform(s)]).tocsr()


def main():
    scheme = "sarc_random"
    tr, va, te = (load_frame(scheme, p) for p in ("train", "val", "test"))
    ytr, yva, yte = (d["label"].to_numpy() for d in (tr, va, te))
    out = {"endings_train": ending_stats(tr),
           "share_changed_by_strip": float((strip(text_of(te)) != text_of(te)).mean())}
    preds = {}
    for train_kind in ("original", "stripped"):
        t0 = time.time()
        ttr = text_of(tr) if train_kind == "original" else strip(text_of(tr))
        X, tf = fit_vec(ttr)
        m = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000).fit(X, ytr)
        for test_kind in ("original", "stripped"):
            if train_kind == "stripped" and test_kind == "original":
                continue
            f = (lambda d: text_of(d)) if test_kind == "original" else (lambda d: strip(text_of(d)))
            pv, pt = m.decision_function(tf(f(va))), m.decision_function(tf(f(te)))
            name = f"train_{train_kind}__test_{test_kind}"
            preds[name] = pt
            save_predictions("13_punctuation", scheme, "val", yva, {f"{scheme}__a8_{name}": pv}, ids=row_ids(va))
            save_predictions("13_punctuation", scheme, "test", yte, {f"{scheme}__a8_{name}": pt}, ids=row_ids(te))
            out[name] = {"val_auc": roc_auc_score(yva, pv), "test": scores(yte, pt, 0.0)}
            print(f"[{name}] val {out[name]['val_auc']:.4f} test {out[name]['test']['roc_auc']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    ref = preds["train_original__test_original"]
    for name in preds:
        if name != "train_original__test_original":
            out[name]["vs_reference"] = paired_bootstrap_auc(yte, preds[name], ref, n_boot=300)
    write_json(HERE / "results" / "a8_punctuation_artifact.json", out)


if __name__ == "__main__":
    main()
