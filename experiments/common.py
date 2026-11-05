"""
Shared helpers for every experiment under `experiments/`.

The repository root holds the read-only baseline: data splits, committed
predictions and models are read from there and nothing is ever written back to
it. All new predictions live under `experiments/predictions/<group>/<dataset>__<split>.parquet`
(one folder per producing script, so parallel runs never write the same file)
with one column per model, keyed by the same `row_id` (blake2b-16 of the reply
text) and `row_position` the baseline handoff tables use.

Protocol (identical to the baselines):
  * shared splits only; hyper-parameters, weights and thresholds are chosen on
    the validation partition, the test partition is scored once;
  * stress scheme sarc_subreddit uses models refitted on its own train;
  * transfer sets (news_headlines, ...) are scored by sarc_random models with no
    target-side fitting, so any ensemble weight/threshold comes from
    sarc_random validation.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

HERE = Path(__file__).resolve().parent
# Inside the repository the baseline is the parent folder; the original
# side-by-side layout (Project Ensemble next to Project Official) still works.
OFFICIAL = HERE.parent if (HERE.parent / "pipeline").exists() else HERE.parent.parent / "Project Official"
# Predictions are ~500 MB and not in git; CS3244_PRED_DIR can point elsewhere.
PRED_DIR = Path(os.environ.get("CS3244_PRED_DIR", HERE / "predictions"))
PRED_DIR.mkdir(exist_ok=True)
# Feature caches and checkpoints (~6 GB, not in git); CS3244_CACHE_DIR can point elsewhere.
CACHE_DIR = Path(os.environ.get("CS3244_CACHE_DIR", HERE / ".cache"))

sys.path.insert(0, str(OFFICIAL / "pipeline"))
from load import load_split, MODEL_SAFE_FEATURES, load_lexicon  # noqa: E402,F401

SEED = 3244
TARGET_PRECISION = 0.75

SURFACE = [c for c in MODEL_SAFE_FEATURES
           if not c.startswith("parent_") and c != "len_ratio_to_parent"]
CONTEXT = ["parent_n_words", "parent_n_chars", "parent_is_empty",
           "parent_jaccard", "len_ratio_to_parent"]

#: (dataset, partition) pairs the ensemble study evaluates. Validation for a
#: transfer set is sarc_random/val (the models were never fitted on the target).
EVAL_SETS = {
    "sarc_random":    {"val": ("sarc_random", "val"),    "test": ("sarc_random", "test")},
    "sarc_subreddit": {"val": ("sarc_subreddit", "val"), "test": ("sarc_subreddit", "test")},
    "news_headlines": {"val": ("sarc_random", "val"),    "test": ("news_headlines", "test")},
}
#: which training scheme's models score each evaluation set
SCHEME = {"sarc_random": "sarc_random", "sarc_subreddit": "sarc_subreddit",
          "news_headlines": "sarc_random"}

BASELINE_DIRS = [OFFICIAL / "results" / "gbdt" / "ensemble_predictions",
                 OFFICIAL / "results" / "nb_logreg" / "ensemble_predictions"]


# --------------------------------------------------------------------------
def text_of(df: pd.DataFrame) -> pd.Series:
    col = "comment_clean" if "comment_clean" in df else "text_clean"
    return df[col].fillna("").astype(str)


def parent_of(df: pd.DataFrame) -> pd.Series:
    for c in ("parent_clean", "context_clean"):
        if c in df:
            return df[c].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def row_ids(df: pd.DataFrame) -> pd.Series:
    return text_of(df).map(lambda s: hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest())


def load_frame(dataset: str, partition: str, columns: list[str] | None = None) -> pd.DataFrame:
    return load_split(dataset, partition, columns=columns).reset_index(drop=True)


# --------------------------------------------------------------------------
def save_predictions(group: str, dataset: str, partition: str, labels: np.ndarray,
                     cols: dict[str, np.ndarray], ids: pd.Series | None = None) -> Path:
    """Merge new model columns into predictions/<group>/<dataset>__<partition>.parquet."""
    (PRED_DIR / group).mkdir(exist_ok=True)
    path = PRED_DIR / group / f"{dataset}__{partition}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        assert len(df) == len(labels) and (df["label"].to_numpy() == labels).all(), path
    else:
        df = pd.DataFrame({"row_position": np.arange(len(labels)), "label": labels.astype(np.int8)})
        if ids is not None:
            df.insert(0, "row_id", ids.to_numpy())
    for k, v in cols.items():
        df[k] = np.asarray(v, dtype=np.float64)
    df.to_parquet(path, index=False)
    return path


#: Prediction groups written by ablations and robustness checks. They are not
#: ensemble members or leaderboard entries, so they are only loaded on request.
EXTRA_GROUP_PREFIXES = ("13_", "14_", "12_scratch_pair_noattn", "12_scratch_transformer")


def is_extra_group(name: str) -> bool:
    return name.startswith(EXTRA_GROUP_PREFIXES)


def load_predictions(dataset: str, partition: str, scheme: str | None = None,
                     include_new: bool = True, groups: list[str] | None = None,
                     extra: list[str] | None = None) -> pd.DataFrame:
    """All model columns for one partition: committed baselines + experiment additions.

    By default the ablation/robustness groups (`is_extra_group`) are skipped; `groups`
    restricts loading to the listed groups, `extra` adds listed extra groups.
    Columns are renamed to drop the `<scheme>__` prefix so the same model has the
    same name on every evaluation set.
    """
    frames = []
    for d in BASELINE_DIRS:
        p = d / f"{dataset}__{partition}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if include_new:
        for p in sorted(PRED_DIR.glob(f"*/{dataset}__{partition}.parquet")):
            g = p.parent.name
            if (groups is not None and g in groups) or (groups is None and not is_extra_group(g))                     or (extra is not None and g in extra):
                frames.append(pd.read_parquet(p))
    base = frames[0][["row_position", "label"]].copy()
    for f in frames:
        assert len(f) == len(base) and (f["label"].to_numpy() == base["label"].to_numpy()).all()
        for c in f.columns:
            if c in ("row_id", "row_position", "label"):
                continue
            name = c.split("__", 1)[1] if "__" in c else c
            if scheme is not None and "__" in c and not c.startswith(scheme + "__"):
                continue
            base[name] = f[c].to_numpy(np.float64)
    return base


# --------------------------------------------------------------------------
def select_threshold(y, p, target=TARGET_PRECISION):
    """Maximise recall subject to precision >= target (None if unreachable)."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    good = np.flatnonzero(precision[:-1] >= target)
    if not len(good):
        return None
    return float(thresholds[good[np.argmax(recall[good])]])


def scores(y, p, threshold=0.5, pred=None) -> dict:
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    if pred is None:
        pred = p >= threshold
    out = {"roc_auc": float(roc_auc_score(y, p)),
           "pr_auc": float(average_precision_score(y, p)),
           "accuracy": float(accuracy_score(y, pred)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0))}
    if p.min() >= 0 and p.max() <= 1:
        out["brier"] = float(brier_score_loss(y, p))
    return out


def paired_bootstrap_auc(y, p_a, p_b, n_boot=500, seed=SEED) -> dict:
    """Paired bootstrap of AUC(a) - AUC(b) on the same rows: mean, 95% CI, P(diff <= 0)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p_a = np.asarray(p_a); p_b = np.asarray(p_b)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx])
    return {"diff": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "p_le_0": float((diffs <= 0).mean())}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n",
                          encoding="utf-8")
