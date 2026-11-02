"""
LightGBM runner that reproduces the baseline "reply_tfidf_surface" model
(Project Official/modelling/gradient_boosting) and lets an experiment append
extra feature blocks.

Protocol, identical to the GBDT baseline:
  * reply word 1-2-gram TF-IDF (top 20k, min_df 3, sublinear) + 17 raw surface counts;
  * hyper-parameters frozen from Project Official/results/gbdt/selected.json;
  * sarc_random: early stopping on sarc_random/val (lr 0.1, patience 100);
  * sarc_subreddit: refit on its own train with the tree count frozen from sarc_random;
  * transfer sets: scored by the sarc_random model, no target-side fitting.

Feature matrices are cached in experiments/.cache so experiments share them.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from common import (HERE, OFFICIAL, SEED, SURFACE, load_frame, row_ids, save_predictions,
                    scores, select_threshold, text_of, write_json)

CACHE = HERE / ".cache"
CACHE.mkdir(exist_ok=True)
SEL = json.loads((OFFICIAL / "results" / "gbdt" / "selected.json").read_text())["lgbm"]
PARAMS = {"objective": "binary", "metric": "auc", "learning_rate": 0.1, "max_bin": 64, "seed": SEED,
          "deterministic": True, "force_col_wise": True, "num_threads": 20, "verbose": -1,
          **SEL["params"]}
TRANSFER = ["news_headlines", "figlang_reddit", "figlang_twitter", "tweeteval_irony"]

_frames: dict = {}


def frame(ds, part):
    key = (ds, part)
    if key not in _frames:
        _frames[key] = load_frame(ds, part)
    return _frames[key]


def base_matrices(scheme: str, vocab: int = 20_000) -> dict:
    """{(ds, part): csr} for the baseline reply_tfidf_surface representation."""
    targets = [(scheme, p) for p in ("train", "val", "test")]
    if scheme == "sarc_random":
        targets += [(t, "test") for t in TRANSFER]
    out, todo = {}, []
    for ds, part in targets:
        p = CACHE / f"base{'' if vocab == 20_000 else vocab}__{scheme}__{ds}__{part}.npz"
        if p.exists():
            out[(ds, part)] = sp.load_npz(p)
        else:
            todo.append((ds, part))
    if todo:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=vocab,
                              dtype=np.float32).fit(text_of(frame(scheme, "train")))
        for ds, part in todo:
            df = frame(ds, part)
            surf = df.reindex(columns=SURFACE).astype(np.float32).fillna(0).to_numpy()
            X = sp.hstack([vec.transform(text_of(df)), sp.csr_matrix(surf)]).tocsr()
            sp.save_npz(CACHE / f"base{'' if vocab == 20_000 else vocab}__{scheme}__{ds}__{part}.npz", X)
            out[(ds, part)] = X
    return out


def cached_block(name: str, fn: Callable[[pd.DataFrame], np.ndarray], ds: str, part: str) -> np.ndarray:
    p = CACHE / f"blk__{name}__{ds}__{part}.npy"
    if p.exists():
        return np.load(p)
    X = fn(frame(ds, part)).astype(np.float32)
    np.save(p, X)
    return X


def run(tag: str, group: str, blocks: dict[str, Callable] | None = None,
        fitted_block: Callable | None = None, schemes=("sarc_random", "sarc_subreddit"),
        params: dict | None = None, vocab: int = 20_000) -> dict:
    """Train/score one LightGBM variant and save its predictions under predictions/<group>/.

    blocks:        name -> row-wise feature function (cached per name)
    fitted_block:  f(scheme, {(ds,part): df-loader}) -> {(ds,part): matrix} for blocks that must be
                   fitted on the scheme's training partition (e.g. an SVD); it receives the keys.
    """
    blocks = blocks or {}
    prm = {**PARAMS, **(params or {})}
    summary, n_trees = {"tag": tag}, None
    for scheme in schemes:
        t0 = time.time()
        base = base_matrices(scheme, vocab)
        mats = {}
        fitted = fitted_block(scheme, list(base)) if fitted_block else {}
        for key, X in base.items():
            parts = [X] + [sp.csr_matrix(cached_block(n, f, *key)) for n, f in blocks.items()]
            if key in fitted:
                parts.append(sp.csr_matrix(fitted[key]))
            mats[key] = sp.hstack(parts).tocsr()
        ytr = frame(scheme, "train")["label"].to_numpy()
        yva = frame(scheme, "val")["label"].to_numpy()
        dtr = lgb.Dataset(mats[(scheme, "train")], ytr, free_raw_data=True)
        if scheme == "sarc_random":
            dva = lgb.Dataset(mats[(scheme, "val")], yva, reference=dtr)
            bst = lgb.train(prm, dtr, num_boost_round=8000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
            n_trees = bst.best_iteration
        else:
            bst = lgb.train(prm, dtr, num_boost_round=n_trees or SEL["n_trees"])
        nt = n_trees or SEL["n_trees"]
        res = {"n_features": mats[(scheme, "train")].shape[1], "n_trees": nt,
               "fit_seconds": round(time.time() - t0, 1)}
        pv = bst.predict(mats[(scheme, "val")], num_iteration=nt)
        thr = select_threshold(yva, pv)
        res["val_auc"] = roc_auc_score(yva, pv)
        for key, X in mats.items():
            if key[1] == "train":
                continue
            p = bst.predict(X, num_iteration=nt)
            df = frame(*key)
            y = df["label"].to_numpy()
            save_predictions(group, key[0], key[1], y, {f"{scheme}__{tag}": p}, ids=row_ids(df))
            if key[1] == "test":
                s = scores(y, p, 0.5)
                if thr is not None and key[0] == scheme:
                    hi = scores(y, p, thr)
                    s.update({"p75_precision": hi["precision"], "p75_recall": hi["recall"]})
                res[key[0]] = s
        summary[scheme] = res
        print(f"[{tag}] {scheme}: val {res['val_auc']:.4f} test {res[scheme]['roc_auc']:.4f} "
              f"trees {nt} ({res['fit_seconds']}s)", flush=True)
        if scheme == "sarc_random":
            imp = pd.Series(bst.feature_importance("gain"))
            summary["gain_share_extra"] = float(imp.iloc[base[(scheme, 'train')].shape[1]:].sum() / imp.sum())
    return summary
