"""
A2 - text representation ablation for the linear model.

Same vectorisers and C as 04_char_ngrams_nbsvm (word 1-2-gram TF-IDF,
char_wb 2-5-gram TF-IDF, scaled log surface counts, LR with C=1), trained on
sarc_random with one block removed at a time:

  lr_word_surface        word n-grams + surface
  lr_char_surface        char n-grams + surface
  lr_word_char           word + char n-grams, no surface
  lr_word_char_surface   all three (refit here so every variant shares one run)

  python 13_ablations/text_representation.py
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
from common import (SURFACE, load_frame, paired_bootstrap_auc, row_ids, save_predictions,  # noqa: E402
                    scores, select_threshold, text_of, write_json)

VARIANTS = {"lr_word_surface": ("word", "surface"), "lr_char_surface": ("char", "surface"),
            "lr_word_char": ("word", "char"), "lr_word_char_surface": ("word", "char", "surface")}
(HERE / "results").mkdir(exist_ok=True)


def main():
    scheme = "sarc_random"
    tr, va, te = (load_frame(scheme, p) for p in ("train", "val", "test"))
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=400_000, dtype=np.float32)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, sublinear_tf=True,
                        max_features=400_000, dtype=np.float32, lowercase=False)
    ss = StandardScaler()

    def surf(df):
        return np.log1p(df.reindex(columns=SURFACE).astype(np.float32).fillna(0).clip(lower=0).to_numpy())

    t0 = time.time()
    blocks = {"word": [w.fit_transform(text_of(tr))], "char": [c.fit_transform(text_of(tr))],
              "surface": [sp.csr_matrix(ss.fit_transform(surf(tr)))]}
    for d in (va, te):
        blocks["word"].append(w.transform(text_of(d)))
        blocks["char"].append(c.transform(text_of(d)))
        blocks["surface"].append(sp.csr_matrix(ss.transform(surf(d))))
    print(f"features built ({time.time()-t0:.0f}s)", flush=True)

    yt_, yv, yt = tr["label"].to_numpy(), va["label"].to_numpy(), te["label"].to_numpy()
    preds, out = {}, {}
    for name, parts in VARIANTS.items():
        t0 = time.time()
        Xtr, Xv, Xt = (sp.hstack([blocks[b][i] for b in parts]).tocsr() for i in range(3))
        m = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000).fit(Xtr, yt_)
        pv, pt = m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
        preds[name] = pt
        save_predictions("13_text_repr", scheme, "val", yv, {f"{scheme}__a2_{name}": pv}, ids=row_ids(va))
        save_predictions("13_text_repr", scheme, "test", yt, {f"{scheme}__a2_{name}": pt}, ids=row_ids(te))
        thr = select_threshold(yv, pv)
        s = scores(yt, pt)
        if thr is not None:
            s["p75_recall"] = scores(yt, pt, thr)["recall"]
        out[name] = {"blocks": list(parts), "n_features": int(Xtr.shape[1]), "val_auc": roc_auc_score(yv, pv),
                     "test": s, "seconds": round(time.time() - t0)}
        print(f"{name}: val {out[name]['val_auc']:.4f} test {s['roc_auc']:.4f} ({out[name]['seconds']}s)", flush=True)
    for name in VARIANTS:
        if name != "lr_word_char_surface":
            out[name]["vs_full"] = paired_bootstrap_auc(yt, preds[name], preds["lr_word_char_surface"], n_boot=300)
    write_json(HERE / "results" / "a2_text_representation.json", out)


if __name__ == "__main__":
    main()
