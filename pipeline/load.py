"""
pipeline/load.py — one import away from a baseline.

    import sys; sys.path.insert(0, "pipeline")
    from load import load_split, list_datasets, MODEL_SAFE_FEATURES

    tr, va, te = load_split("sarc_random")
    X, y = tr[MODEL_SAFE_FEATURES], tr["label"]

Everything here is read-only. It does not clean, split or modify anything —
run `pipeline/04_build_splits.py` for that.

Why `MODEL_SAFE_FEATURES` exists
--------------------------------
The cleaned files keep every column, including several that must NOT be fed to
a model. `ups`/`downs` are placeholder -1 for most rows and their missingness
correlates with the label — a collection artefact, not sarcasm. `score` is only
known after the comment has been voted on, so it is unavailable at prediction
time. `n_emoji` is identically zero on 2009-2016 Reddit. The audit columns
(`had_slash_s`, ...) record what the cleaner removed and would leak the label
directly. This constant is the allow-list; see docs/data_dictionary.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "splits"
CATALOG_PATH = ROOT / "data" / "catalog.json"

SPLIT_NAMES = ("train", "val", "test")

# --------------------------------------------------------------------------
# Column groups
# --------------------------------------------------------------------------
TEXT_PRIMARY = "comment_clean"
TEXT_SUPPLEMENTARY = "text_clean"
CONTEXT_PRIMARY = "parent_clean"
CONTEXT_SUPPLEMENTARY = "context_clean"
LABEL = "label"

#: Engineered numeric features that are safe to use as model input.
MODEL_SAFE_FEATURES = [
    "n_chars", "n_words", "n_sentences", "avg_word_len",
    "n_exclam", "n_question", "n_ellipsis", "n_repeat_punct",
    "n_allcaps_words", "allcaps_ratio", "n_elongation", "n_scare_quotes",
    "n_urls", "n_mentions", "n_hashtags",
    "n_interjections", "starts_with_interjection",
    "parent_n_words", "parent_n_chars", "parent_is_empty",
    "parent_jaccard", "len_ratio_to_parent",
]

#: Columns that exist in the files but must never be model input, and why.
EXCLUDED_FEATURES = {
    "ups": "placeholder -1 for most rows; missingness is a collection artefact",
    "downs": "placeholder -1 for most rows; missingness is a collection artefact",
    "ups_is_missing": "encodes the same artefact",
    "downs_is_missing": "encodes the same artefact",
    "score": "post-hoc — unknown at prediction time; analysis only",
    "is_controversial": "derived from score",
    "n_emoji": "identically 0 on 2009-2016 Reddit; dead weight",
    "hour_utc": "flat within 4pp — no signal",
    "had_slash_s": "audit column: records the label marker the cleaner removed",
    "had_label_hashtag": "audit column: direct label leak",
    "had_explicit_sarcasm_word": "audit column: direct label leak",
    "en_score": "cleaning diagnostic",
    "lang_method": "cleaning diagnostic",
    "author": "254k values; use only for a deliberate author-holdout experiment",
    "created_utc": "use `year`/`month` if you want time; raw timestamp overfits",
    "date": "redundant with created_utc",
}

#: Categorical columns worth encoding — but read the note.
CATEGORICAL_FEATURES = {
    "subreddit": "STRONG predictor (8%-79% sarcasm rate). On sarc_random it "
                 "partly measures memorised community priors; the honest test "
                 "is sarc_subreddit, where test communities are unseen.",
    "year": "sarcasm rate drifts 11pp across 2009-2016.",
    "month": "weak.",
    "weekday": "weak.",
}


# --------------------------------------------------------------------------
def catalog() -> dict:
    """The machine-readable catalogue written by pipeline/04_build_splits.py."""
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def list_datasets(verbose: bool = True) -> pd.DataFrame:
    """Every available split set, its size and what it is for."""
    rows = []
    for e in catalog()["entries"]:
        if not e.get("splits"):
            continue
        rows.append({
            "name": e["name"],
            "rows": e["rows_total"],
            **{f"{s}_rows": e["splits"].get(s, {}).get("rows", 0)
               for s in SPLIT_NAMES},
            "sarcastic_pct": e["splits"].get("train", {}).get("sarcastic_pct"),
            "stage": e["project_stage"],
            "role": e["role"],
        })
    df = pd.DataFrame(rows)
    if verbose:
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(df.drop(columns=["role"]).to_string(index=False))
    return df


def _resolve(name: str) -> Path:
    for base in (DATA / "primary", DATA / "supplementary"):
        p = base / name
        if p.is_dir():
            return p
    avail = sorted(p.name for base in (DATA / "primary", DATA / "supplementary")
                   if base.is_dir() for p in base.iterdir() if p.is_dir())
    raise FileNotFoundError(f"unknown dataset {name!r}; available: {avail}")


def load_split(name: str, which: str | None = None, columns: list[str] | None = None):
    """
    Load a split set by name.

    `load_split("sarc_random")`          -> (train, val, test)
    `load_split("sarc_random", "test")`  -> just the test DataFrame

    `columns` is passed through to the parquet reader, so loading only the
    text and label off the 914k-row corpus is cheap.
    """
    d = _resolve(name)
    if which is not None:
        if which not in SPLIT_NAMES:
            raise ValueError(f"which must be one of {SPLIT_NAMES}, got {which!r}")
        return pd.read_parquet(d / f"{which}.parquet", columns=columns)
    return tuple(pd.read_parquet(d / f"{s}.parquet", columns=columns)
                 for s in SPLIT_NAMES)


def text_column(df: pd.DataFrame) -> str:
    """The target-text column, whichever schema this frame follows."""
    return TEXT_PRIMARY if TEXT_PRIMARY in df.columns else TEXT_SUPPLEMENTARY


def context_column(df: pd.DataFrame) -> str | None:
    for c in (CONTEXT_PRIMARY, CONTEXT_SUPPLEMENTARY):
        if c in df.columns:
            return c
    return None


def xy(df: pd.DataFrame, *, features: list[str] | None = None,
       with_text: bool = True, with_context: bool = False):
    """
    Split a frame into (X, y) using only model-safe columns.

    `with_context=True` concatenates the parent/context text in front of the
    target text, separated by ' [SEP] ' — the simplest possible context
    encoding, and the right thing to compare against for sub-question 1.
    """
    feats = [c for c in (features or MODEL_SAFE_FEATURES) if c in df.columns]
    X = df[feats].copy()

    if with_text:
        t = df[text_column(df)].fillna("")
        ctx = context_column(df)
        if with_context and ctx:
            t = df[ctx].fillna("") + " [SEP] " + t
        X.insert(0, "text", t)

    return X, df[LABEL]


def load_lexicon(name: str):
    """
    Load one of the four bundled lexicons into a convenient structure.

    name: empath | vader | nrc | sentiwordnet
    """
    d = DATA / "lexicons"
    if name == "empath":
        out = {}
        with (d / "empath_categories.tsv").open(encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if parts and parts[0]:
                    out[parts[0]] = set(p for p in parts[1:] if p)
        return out                                   # category -> {words}
    if name == "vader":
        out = {}
        with (d / "vader_lexicon.txt").open(encoding="utf-8") as fh:
            for line in fh:
                p = line.split("\t")
                if len(p) >= 2:
                    try:
                        out[p[0]] = float(p[1])
                    except ValueError:
                        continue
        return out                                   # token -> valence
    if name == "nrc":
        out: dict[str, set[str]] = {}
        with (d / "nrc_emotion_lexicon.txt").open(encoding="utf-8") as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) == 3 and p[2] == "1":
                    out.setdefault(p[1], set()).add(p[0])
        return out                                   # emotion -> {words}
    if name == "sentiwordnet":
        return pd.read_csv(d / "SentiWordNet_3.0.0.txt", sep="\t", comment="#",
                           names=["POS", "ID", "PosScore", "NegScore",
                                  "SynsetTerms", "Gloss"])
    raise ValueError("name must be empath | vader | nrc | sentiwordnet")


if __name__ == "__main__":
    print("Available split sets\n")
    list_datasets()
    print("\nExample:")
    tr = load_split("sarc_random", "train", columns=["label", "comment_clean"])
    print(f"  sarc_random/train -> {len(tr):,} rows")
    print(f"  {tr['comment_clean'].iloc[0][:80]!r}  label={tr['label'].iloc[0]}")
