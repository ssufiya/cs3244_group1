"""
Improvement 5 - hybrid linear + tree model ("stacking as a feature").

LightGBM on a 20k-term vocabulary and the linear models on 800k word + char
n-grams see different things: the trees model non-linear surface cues, the
linear model covers the long tail of rare n-grams. Here the out-of-fold score
of the word+char LR from improvement 3 is given to LightGBM as one extra
feature:
  * train rows get 5-fold out-of-fold scores (GroupKFold on parent_key, so
    sibling replies never score each other);
  * val / test / transfer rows get the score of the LR refitted on all of train;
  * C is frozen from improvement 3 (chosen on sarc_random/val).
LightGBM hyper-parameters stay frozen; early stopping on sarc_random/val as before.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import SURFACE, text_of, write_json  # noqa: E402
from features_extra import contrast_block, cue_block  # noqa: E402
from lgbm_runner import CACHE, frame, run  # noqa: E402

# C chosen on sarc_random/val in improvement 3 (log: 0.25 -> .8106, 0.5 -> .8139, 1.0 -> .8142)
C = 1.0


def _surf(df):
    return np.log1p(df.reindex(columns=SURFACE).astype(np.float32).fillna(0).clip(lower=0).to_numpy())


def _fit_matrix(tr):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=400_000, dtype=np.float32)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, sublinear_tf=True,
                        max_features=400_000, dtype=np.float32, lowercase=False)
    ss = StandardScaler()
    X = sp.hstack([w.fit_transform(text_of(tr)), c.fit_transform(text_of(tr)),
                   sp.csr_matrix(ss.fit_transform(_surf(tr)))]).tocsr()

    def tf(df):
        return sp.hstack([w.transform(text_of(df)), c.transform(text_of(df)),
                          sp.csr_matrix(ss.transform(_surf(df)))]).tocsr()
    return X, tf


def lr_score_block(scheme, keys):
    paths = {k: CACHE / f"lrscore__{scheme}__{k[0]}__{k[1]}.npy" for k in keys}
    if all(p.exists() for p in paths.values()):
        return {k: np.load(p) for k, p in paths.items()}
    t0 = time.time()
    tr = frame(scheme, "train")
    y = tr["label"].to_numpy()
    oof = np.zeros(len(tr), np.float32)
    # Vectorisers are refitted inside each fold so held-out rows never shape the vocabulary.
    for i, (a, b) in enumerate(GroupKFold(5).split(tr, y, groups=tr["parent_key"])):
        Xa, tf = _fit_matrix(tr.iloc[a])
        m = LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(Xa, y[a])
        oof[b] = m.decision_function(tf(tr.iloc[b]))
        print(f"  [lr-oof] {scheme} fold {i} ({time.time()-t0:.0f}s)", flush=True)
    X, tf = _fit_matrix(tr)
    m = LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(X, y)
    out = {}
    for k in keys:
        s = oof if k == (scheme, "train") else m.decision_function(tf(frame(*k))).astype(np.float32)
        np.save(paths[k], s[:, None])
        out[k] = s[:, None]
    return out


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    out = {"C": C,
           "lr_feature": run("lgbm_lrscore", "lgbm_lrscore", fitted_block=lr_score_block),
           "lr_feature_cues_contrast": run("lgbm_lrscore_cues_contrast", "lgbm_lrscore_cues_contrast",
                                           blocks={"cues": cue_block, "contrast": contrast_block},
                                           fitted_block=lr_score_block)}
    write_json(HERE / "results" / "summary.json", out)
