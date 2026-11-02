"""
Shared helpers for the step-by-step notebooks.

The notebooks do not retrain anything. They read the predictions and result
files that the scripts under `experiments/` (and the team baselines under
`modelling/`) saved, and recompute every metric the same way, so each table in
`docs/` can be traced back to a cell.

Where the files are looked up:
  * baseline splits, models and committed baseline predictions: the repository
    root (`pipeline/`, `results/`);
  * experiment predictions (~500 MB, not in git): `$CS3244_PRED_DIR`, else
    `experiments/predictions/`, else the original side-by-side folder
    `../Project Ensemble/predictions/`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NB_DIR = Path(__file__).resolve().parent
ROOT = NB_DIR.parent
OFFICIAL = ROOT if (ROOT / "pipeline").exists() else ROOT.parent / "Project Official"
EXP = ROOT / "experiments"
RESULTS = ROOT / "results" / "ensemble"
FIGURES = ROOT / "figures" / "steps"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def _find_pred_dir() -> Path:
    candidates = [os.environ.get("CS3244_PRED_DIR"), EXP / "predictions",
                  ROOT.parent / "Project Ensemble" / "predictions"]
    for c in candidates:
        if c and Path(c).exists() and any(Path(c).glob("*/sarc_random__test.parquet")):
            return Path(c)
    raise FileNotFoundError(
        "Experiment predictions not found. Set CS3244_PRED_DIR or copy the "
        "predictions folder to experiments/predictions (see experiments/README.md).")


PRED_DIR = _find_pred_dir()
os.environ["CS3244_PRED_DIR"] = str(PRED_DIR)
sys.path.insert(0, str(EXP))
import common

common.PRED_DIR = PRED_DIR
from common import (load_predictions, paired_bootstrap_auc, scores,
                    select_threshold)

BASELINE = "lgbm_reply_tfidf_surface"

# name -> (readable label, family, step that produced it, course-compliant)
# "Compliant" = no pretrained model anywhere in the pipeline (course rule).
MODELS = {
    "mnb_reply_tfidf":            ("Multinomial NB, word TF-IDF", "Naive Bayes", "02 baselines", True),
    "mnb_reply_tfidf_surface":    ("Multinomial NB, + surface", "Naive Bayes", "02 baselines", True),
    "cnb_reply_tfidf":            ("Complement NB, word TF-IDF", "Naive Bayes", "02 baselines", True),
    "cnb_reply_tfidf_surface":    ("Complement NB, + surface", "Naive Bayes", "02 baselines", True),
    "lr_reply_tfidf":             ("Logistic regression, word TF-IDF", "Linear", "02 baselines", True),
    "lr_reply_tfidf_surface":     ("Logistic regression, + surface", "Linear", "02 baselines", True),
    "svm_tfidf_lex_surface":      ("Linear SVM, TF-IDF + lexicon + surface", "Linear", "02 baselines", True),
    "lgbm_reply_tfidf":           ("LightGBM, word TF-IDF", "GBDT", "02 baselines", True),
    "lgbm_reply_tfidf_surface":   ("LightGBM, + surface (baseline)", "GBDT", "02 baselines", True),
    "xgb_reply_tfidf":            ("XGBoost, word TF-IDF", "GBDT", "02 baselines", True),
    "xgb_reply_tfidf_surface":    ("XGBoost, + surface", "GBDT", "02 baselines", True),
    "ebm_named340":               ("Team EBM (200k rows, 340 features)", "Glass-box", "02 baselines", True),
    "lgbm_base_repro":            ("LightGBM baseline, reproduced", "GBDT", "03 features", True),
    "lgbm_cues":                  ("LightGBM + linguistic cues", "GBDT", "03 features", True),
    "lgbm_contrast":              ("LightGBM + reply/parent contrast", "GBDT", "03 features", True),
    "lgbm_cues_contrast":         ("LightGBM + cues + contrast", "GBDT", "03 features", True),
    "lr_word_char_surface":       ("LR, word + char n-grams", "Linear", "03 features", True),
    "nblr_word_char_surface":     ("NB-SVM, word + char n-grams", "Linear", "03 features", True),
    "lgbm_lsa":                   ("LightGBM + LSA interactions", "GBDT", "03 features", True),
    "lgbm_cues_contrast_lsa":     ("LightGBM + cues + contrast + LSA", "GBDT", "03 features", True),
    "lgbm_lrscore":               ("LightGBM + LR score", "GBDT", "03 features", True),
    "lgbm_lrscore_cues_contrast": ("LightGBM + LR score + cues + contrast", "GBDT", "03 features", True),
    "lgbm_v60k_cues_contrast":    ("LightGBM, 60k vocab + cues + contrast", "GBDT", "03 features", True),
    "lgbm_minilm_v60k_cues_contrast": ("LightGBM + MiniLM embeddings", "GBDT", "03 features", False),
    "lgbm_combined_best":         ("LightGBM, all features incl. MiniLM", "GBDT", "03 features", False),
    "ebm_full":                   ("Full-data EBM (66 features + text score)", "Glass-box", "04 EBM", True),
    "ebm_full_notext":            ("Full-data EBM, no text score", "Glass-box", "04 EBM", True),
    "scratch_reply":              ("From-scratch BiLSTM, reply only", "Scratch NN", "05 scratch NN", True),
    "scratch_pair":               ("From-scratch ESIM, reply + parent", "Scratch NN", "05 scratch NN", True),
    "roberta_reply":              ("RoBERTa, reply only, 1 epoch", "Pretrained", "06 reference", False),
    "roberta_pair":               ("RoBERTa, reply + parent, 1 epoch", "Pretrained", "06 reference", False),
    "roberta_pair_conv":          ("RoBERTa, reply + parent, converged", "Pretrained", "06 reference", False),
    "roberta_pair_conv_s4248":    ("RoBERTa, converged, 2nd seed", "Pretrained", "06 reference", False),
}

#: (test set, validation set used for thresholds, training scheme of the models)
EVAL_SETS = [("sarc_random", "sarc_random", "sarc_random"),
             ("sarc_subreddit", "sarc_subreddit", "sarc_subreddit"),
             ("figlang_reddit", "sarc_random", "sarc_random"),
             ("figlang_twitter", "sarc_random", "sarc_random"),
             ("tweeteval_irony", "sarc_random", "sarc_random"),
             ("news_headlines", "sarc_random", "sarc_random")]


def info(model: str) -> dict:
    label, family, step, ok = MODELS.get(model, (model, "?", "?", True))
    return {"model": model, "label": label, "family": family, "step": step, "compliant": ok}


def predictions(test_set: str = "sarc_random") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(test frame, validation frame) with one probability/score column per model."""
    _, val_set, scheme = next(e for e in EVAL_SETS if e[0] == test_set)
    return (load_predictions(test_set, "test", scheme=scheme),
            load_predictions(val_set, "val", scheme=scheme))


def model_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("row_position", "label")]


def evaluate(test: pd.DataFrame, val: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """ROC-AUC etc. at 0.5, plus test recall at the val-chosen precision >= 0.75 threshold."""
    y, yv = test["label"].to_numpy(), val["label"].to_numpy()
    rows = []
    for m in models or model_columns(test):
        if m not in test or m not in val:
            continue
        p, pv = test[m].to_numpy(), val[m].to_numpy()
        r = info(m)
        r.update(scores(y, p, 0.5 if (p.min() >= 0 and p.max() <= 1) else 0.0))
        thr = select_threshold(yv, pv)
        if thr is not None:
            hi = scores(y, p, thr)
            r.update({"p75_precision": hi["precision"], "p75_recall": hi["recall"]})
        rows.append(r)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def bootstrap_vs(test: pd.DataFrame, models: list[str], ref: str = BASELINE, n_boot: int = 200) -> pd.DataFrame:
    """Paired bootstrap AUC difference of each model against `ref` on the same rows."""
    y = test["label"].to_numpy()
    rows = []
    for m in models:
        if m == ref or m not in test:
            continue
        b = paired_bootstrap_auc(y, test[m].to_numpy(), test[ref].to_numpy(), n_boot=n_boot)
        rows.append({"model": m, "vs": ref, "d_auc": b["diff"], "ci_low": b["ci_low"], "ci_high": b["ci_high"]})
    return pd.DataFrame(rows)


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def savefig(fig, name: str):
    path = FIGURES / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def show(df: pd.DataFrame, digits: int = 4):
    """Round floats for display."""
    return df.round(digits)
