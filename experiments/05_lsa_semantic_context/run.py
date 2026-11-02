"""
Improvement 4 - semantic reply/parent interaction through LSA (no deep learning).

The baseline's parent block was a bag of parent words: it can say what the
parent is about, not whether the reply agrees with it. Here reply and parent are
projected into one 128-dimensional latent semantic space (TF-IDF -> truncated
SVD, both fitted on the scheme's training replies + parents), and LightGBM gets
  * cosine(reply, parent) in LSA space and in raw TF-IDF space,
  * the element-wise product reply * parent (128 dims) - where they agree or
    disagree topic by topic, which trees cannot form from two separate bags,
  * |reply - parent| norm.
Added on top of reply_tfidf_surface, and on top of reply_tfidf_surface + cues +
contrast (the best hand-built set).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import SEED, parent_of, text_of, write_json  # noqa: E402
from features_extra import contrast_block, cue_block  # noqa: E402
from lgbm_runner import CACHE, frame, run  # noqa: E402

DIM = 128


def lsa_block(scheme, keys):
    paths = {k: CACHE / f"lsa__{scheme}__{k[0]}__{k[1]}.npy" for k in keys}
    if all(p.exists() for p in paths.values()):
        return {k: np.load(p) for k, p in paths.items()}
    t0 = time.time()
    tr = frame(scheme, "train")
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=5, sublinear_tf=True, max_features=50_000,
                          dtype=np.float32).fit(text_of(tr).tolist() + parent_of(tr).tolist())
    svd = TruncatedSVD(DIM, algorithm="randomized", n_iter=5, random_state=SEED)
    svd.fit(vec.transform(text_of(tr).tolist() + parent_of(tr).tolist()))
    out = {}
    for k in keys:
        df = frame(*k)
        R, P = vec.transform(text_of(df)), vec.transform(parent_of(df))
        cos_tfidf = np.asarray(normalize(R).multiply(normalize(P)).sum(1)).ravel()
        r, p = normalize(svd.transform(R)), normalize(svd.transform(P))
        has_p = (P.getnnz(1) > 0).astype(np.float32)
        X = np.column_stack([(r * p).sum(1) * has_p, cos_tfidf, np.linalg.norm(r - p, axis=1) * has_p,
                             has_p, r * p]).astype(np.float32)
        np.save(paths[k], X)
        out[k] = X
    print(f"[lsa] {scheme} built ({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    out = {"lsa": run("lgbm_lsa", "lgbm_lsa", fitted_block=lsa_block),
           "cues_contrast_lsa": run("lgbm_cues_contrast_lsa", "lgbm_cues_contrast_lsa",
                                    blocks={"cues": cue_block, "contrast": contrast_block},
                                    fitted_block=lsa_block)}
    write_json(HERE / "results" / "summary.json", out)
