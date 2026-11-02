"""
Re-create the Margin group's Linear SVM (C=0.1, word 1-2-gram TF-IDF <=500k +
16 lexicon + 22 surface features, scaled) exactly as in
Project Official/modelling/Margin-Based/margin-based_baseline.ipynb, because its
predictions were never committed. Fitted on sarc_random/train (scores its val,
test and the transfer sets) and on sarc_subreddit/train (scores its val, test).
Raw decision_function margins are saved; the ensemble calibrates them on val.
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (MODEL_SAFE_FEATURES, SEED, load_frame, load_lexicon, parent_of,  # noqa: E402
                    row_ids, save_predictions, text_of)

RE_TOKEN = re.compile(r"[a-z][a-z']+")
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust", "positive", "negative"]
VADER = load_lexicon("vader")
_nrc = load_lexicon("nrc")
NRC = {e: _nrc.get(e, set()) for e in EMOTIONS}


def tokens(t):
    return RE_TOKEN.findall(t.lower())


def _vader(tk):
    vals = [VADER[w] for w in tk if w in VADER]
    if not vals:
        return 0.0, 0.0, 0.0
    a = np.asarray(vals, dtype=np.float32)
    return float(a.mean()), float((a > 0).mean()), float((a < 0).mean())


def lexicon_block(text, parent):
    out = np.zeros((len(text), 16), dtype=np.float32)
    for r, (t, p) in enumerate(zip(text, parent)):
        tk = tokens(t); s = set(tk); n = max(1, len(tk))
        for j, e in enumerate(EMOTIONS):
            out[r, j] = len(s & NRC[e]) / n
        m, pos, neg = _vader(tk)
        pm, _, _ = _vader(tokens(p))
        out[r, 10:16] = [m, pos, neg, pm, 1.0 if (m > 0.05 and pm < -0.05) else 0.0, m * pm]
    return out


def surface_block(df):
    X = df.reindex(columns=MODEL_SAFE_FEATURES).astype(float).fillna(0.0).clip(lower=0.0)
    return np.log1p(X.to_numpy())


class Blocks:
    def fit_transform(self, df):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=500_000)
        self.ls, self.ss = StandardScaler(), StandardScaler()
        t, p = text_of(df), parent_of(df)
        return sp.hstack([self.tfidf.fit_transform(t),
                          sp.csr_matrix(self.ls.fit_transform(lexicon_block(t, p))),
                          sp.csr_matrix(self.ss.fit_transform(surface_block(df)))]).tocsr()

    def transform(self, df):
        t, p = text_of(df), parent_of(df)
        return sp.hstack([self.tfidf.transform(t),
                          sp.csr_matrix(self.ls.transform(lexicon_block(t, p))),
                          sp.csr_matrix(self.ss.transform(surface_block(df)))]).tocsr()


def run(scheme, extra_targets):
    t0 = time.time()
    tr = load_frame(scheme, "train")
    bb = Blocks()
    X = bb.fit_transform(tr)
    print(f"[{scheme}] features {X.shape} ({time.time()-t0:.0f}s)", flush=True)
    m = LinearSVC(C=0.1, dual="auto", max_iter=5000, random_state=SEED).fit(X, tr["label"].to_numpy())
    print(f"[{scheme}] fitted ({time.time()-t0:.0f}s)", flush=True)
    del X
    for ds, part in [(scheme, "val"), (scheme, "test")] + [(t, "test") for t in extra_targets]:
        df = load_frame(ds, part)
        s = m.decision_function(bb.transform(df))
        y = df["label"].to_numpy()
        save_predictions("svm", ds, part, y, {f"{scheme}__svm_tfidf_lex_surface": s}, ids=row_ids(df))
        print(f"  {ds}/{part}: AUC {roc_auc_score(y, s):.4f}", flush=True)


if __name__ == "__main__":
    run("sarc_random", ["news_headlines", "figlang_reddit", "figlang_twitter", "tweeteval_irony"])
    run("sarc_subreddit", [])
