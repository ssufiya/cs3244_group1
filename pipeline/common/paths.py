"""Every path the project uses, resolved from the repository root."""
from __future__ import annotations

from pathlib import Path

# pipeline/common/paths.py -> pipeline/common -> pipeline -> repo root
ROOT = Path(__file__).resolve().parents[2]

# --- data (large, regenerable, git-ignored except the manifest and samples) --
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
SAMPLES = DATA / "samples"

RAW_PRIMARY = RAW / "primary" / "reddit_sarc"
RAW_NEWS = RAW / "supplementary" / "news_headlines"
RAW_TWEETS = RAW / "supplementary" / "twitter_sarcasm"
RAW_IRONY = RAW / "supplementary" / "tweeteval_irony"
RAW_FIGLANG = RAW / "supplementary" / "figlang2020_context"
RAW_LEXICONS = RAW / "external" / "lexicons"

# --- the split package the modelling code reads ----------------------------
DS_PRIMARY = SPLITS / "primary"
DS_SUPP = SPLITS / "supplementary"
DS_LEXICONS = SPLITS / "lexicons"
DS_REJECTED = SPLITS / "_rejected"
DS_SAMPLES = SAMPLES
CATALOG = DATA / "catalog.json"

# --- outputs (small, committed) --------------------------------------------
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CLEAN_REPORTS = RESULTS / "cleaning"
EDA_RESULTS = RESULTS / "eda"
SPLIT_RESULTS = RESULTS / "splits"
EDA_FIGURES = FIGURES / "eda"

ALL_DIRS = [
    RAW_PRIMARY, RAW_NEWS, RAW_TWEETS, RAW_IRONY, RAW_FIGLANG, RAW_LEXICONS,
    INTERIM, PROCESSED, SAMPLES, EDA_FIGURES, CLEAN_REPORTS, EDA_RESULTS,
    SPLIT_RESULTS,
]
SPLIT_DIRS = [DS_PRIMARY, DS_SUPP, DS_LEXICONS, DS_REJECTED, DS_SAMPLES]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def ensure_dataset_dirs() -> None:
    for d in SPLIT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
