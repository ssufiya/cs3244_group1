#!/usr/bin/env python
"""
01_clean_primary.py — clean the SARC balanced Reddit corpus.

Pipeline
--------
 1. load + type-coerce
 2. drop rows with an unusable comment ([deleted]/[removed]/empty/NaN)
 3. normalise unicode, strip Reddit markdown, replace URLs/mentions
 4. remove label leakage (/s, \\s, "#sarcasm", "<sarcasm>")   <-- critical
 5. drop too-short / absurdly long comments
 6. English filter (lexical heuristic + optional langdetect adjudication)
 7. de-duplicate: exact, then near-duplicate on (text, label)
 8. engineer surface + context + temporal features
 9. write parquet + a stratified CSV sample + an audit report

Usage
-----
    python scripts/01_clean_primary.py --sample 200000     # sampled run
    python scripts/01_clean_primary.py                     # full ~1.01M rows
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import CLEAN_REPORTS, PROCESSED, RAW_PRIMARY, SAMPLES, ensure_dirs  # noqa: E402
from common.report import CleaningLog                                                  # noqa: E402
from common import text_cleaning as tc                                                 # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = RAW_PRIMARY / "train-balanced-sarcasm.csv"
NAME = "reddit_sarc_balanced"

DTYPES = {
    "label": "int8", "comment": "string", "author": "string",
    "subreddit": "string", "score": "float32", "ups": "float32",
    "downs": "float32", "date": "string", "parent_comment": "string",
}

MIN_WORDS = 2
MAX_WORDS = 300          # 99.99th pct is ~130; beyond this it is copypasta
MAX_CHARS = 2000


def load(sample: int | None, seed: int) -> tuple[pd.DataFrame, int]:
    print(f"[load] {SRC}")
    df = pd.read_csv(SRC, dtype=DTYPES, parse_dates=["created_utc"],
                     on_bad_lines="warn", low_memory=False)
    n_full = len(df)
    print(f"       {n_full:,} rows x {df.shape[1]} cols")
    if sample and sample < n_full:
        # Stratify on the label so the class balance of the sample matches
        # the corpus; the corpus is balanced by construction, so this just
        # guarantees we don't introduce skew by luck.
        parts = [g.sample(n=min(len(g), int(round(sample * len(g) / n_full))),
                          random_state=seed)
                 for _, g in df.groupby("label", sort=True)]
        df = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(f"[load] stratified sample -> {len(df):,} rows")
    return df, n_full


def clean(df: pd.DataFrame, log: CleaningLog, use_langdetect: bool) -> pd.DataFrame:
    # -- 2. unusable comments ---------------------------------------------
    bad = tc.is_deleted(df["comment"]) | df["comment"].isna()
    df = df[~bad]
    log.drop("empty / [deleted] / [removed] comment", len(df))

    # -- 3-4. text pipeline (normalise, strip markup, de-leak) ------------
    print("[clean] normalising comment text ...")
    ctext = tc.clean_series(df["comment"], keep_subreddit=True)
    log.note("leak_slash_s_in_comment", int(ctext["had_slash_s"].sum()))
    log.note("leak_label_hashtag_in_comment", int(ctext["had_label_hashtag"].sum()))
    log.note("leak_explicit_word_in_comment", int(ctext["had_explicit_sarcasm_word"].sum()))

    # Leak rate broken down by class tells us whether the marker really is
    # the label in disguise.
    leak_by_label = (ctext["had_slash_s"] | ctext["had_explicit_sarcasm_word"]) \
        .groupby(df["label"]).mean().round(6).to_dict()
    log.note("leak_rate_by_label", {int(k): float(v) for k, v in leak_by_label.items()})

    print("[clean] normalising parent_comment text ...")
    ptext = tc.clean_series(df["parent_comment"].fillna(""), keep_subreddit=True)

    df = df.assign(
        comment_clean=ctext["text_clean"].values,
        parent_clean=ptext["text_clean"].values,
        **{c: ctext[c].values for c in ctext.columns
           if c not in ("text_clean", "text_norm")},
    )
    df["parent_n_words"] = ptext["n_words"].values
    df["parent_n_chars"] = ptext["n_chars"].values
    df["parent_is_empty"] = (ptext["n_words"].values == 0)

    # Text can become empty once the marker and markup are gone — e.g. a
    # comment that was literally just "/s".
    empty_after = df["comment_clean"].str.strip().eq("")
    log.note("became_empty_after_cleaning", int(empty_after.sum()))
    df = df[~empty_after]
    log.drop("empty after markup/leak removal", len(df))

    # -- 5. length bounds --------------------------------------------------
    too_short = df["n_words"] < MIN_WORDS
    df = df[~too_short]
    log.drop(f"fewer than {MIN_WORDS} words", len(df))

    too_long = (df["n_words"] > MAX_WORDS) | (df["n_chars"] > MAX_CHARS)
    df = df[~too_long]
    log.drop(f"more than {MAX_WORDS} words / {MAX_CHARS} chars", len(df))

    # -- 6. language -------------------------------------------------------
    print("[clean] language filtering ...")
    df = tc.add_language_flags(df, "comment_clean", use_langdetect=use_langdetect)
    log.note("lang_method_counts", df["lang_method"].value_counts().to_dict())
    df = df[df["is_english"]]
    log.drop("non-English (heuristic + langdetect)", len(df))

    # -- 7. de-duplication -------------------------------------------------
    print("[clean] de-duplicating ...")
    df["_key"] = tc.dedup_key(df["comment_clean"])

    before = len(df)
    df = df.drop_duplicates(subset=["_key", "label", "parent_clean"], keep="first")
    log.drop("exact dup (comment+label+parent)", len(df),
             "same comment under the same parent with the same label")

    # A comment whose normalised form appears with BOTH labels is
    # irreducibly ambiguous — the identical string is sarcastic in one thread
    # and sincere in another. Keeping both sides only teaches the model noise.
    conflict = df.groupby("_key")["label"].transform("nunique") > 1
    log.note("label_conflicting_texts", int(df.loc[conflict, "_key"].nunique()))
    log.note("label_conflicting_rows", int(conflict.sum()))
    df = df[~conflict]
    log.drop("same text carries both labels", len(df))

    df = df.drop_duplicates(subset=["_key"], keep="first")
    log.drop("near-dup (normalised text, any parent)", len(df),
             f"{before - len(df):,} total duplicate rows removed")
    df = df.drop(columns=["_key"])

    return df


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Context + temporal + score features used by the EDA and the models."""
    print("[feat] context & temporal features ...")
    df["parent_jaccard"] = [
        tc.jaccard(a, b) for a, b in zip(df["comment_clean"], df["parent_clean"])
    ]
    df["len_ratio_to_parent"] = (
        df["n_words"] / df["parent_n_words"].replace(0, np.nan)
    ).astype("float32")

    ts = pd.to_datetime(df["created_utc"], errors="coerce")
    df["year"] = ts.dt.year.astype("Int16")
    df["month"] = ts.dt.month.astype("Int8")
    df["hour_utc"] = ts.dt.hour.astype("Int8")
    df["weekday"] = ts.dt.dayofweek.astype("Int8")

    # `ups`/`downs` are -1 for most of the corpus (Reddit stopped exposing
    # them); `score` is the field that actually carries information.
    df["score"] = df["score"].fillna(0).astype("float32")
    df["ups_is_missing"] = df["ups"].fillna(-1).eq(-1)
    df["downs_is_missing"] = df["downs"].fillna(-1).eq(-1)
    df["is_controversial"] = df["score"].between(-1, 1)

    df["subreddit"] = df["subreddit"].fillna("<unknown>").astype("string")
    df["author"] = df["author"].fillna("<unknown>").astype("string")

    # Crude sentiment-incongruity proxy: positive surface words in a comment
    # that got downvoted. Replaced by the lexicon features in step 04.
    df["has_positive_interjection"] = df["starts_with_interjection"] & (df["n_words"] > 2)
    return df


ORDERED_COLS = [
    "label", "comment_clean", "parent_clean", "subreddit", "author",
    "score", "ups", "downs", "ups_is_missing", "downs_is_missing",
    "is_controversial", "date", "created_utc", "year", "month", "hour_utc",
    "weekday", "n_chars", "n_words", "n_sentences", "avg_word_len",
    "n_exclam", "n_question", "n_ellipsis", "n_repeat_punct",
    "n_allcaps_words", "allcaps_ratio", "n_emoji", "n_elongation",
    "n_scare_quotes", "n_urls", "n_mentions", "n_hashtags",
    "n_interjections", "starts_with_interjection", "has_positive_interjection",
    "parent_n_words", "parent_n_chars", "parent_is_empty", "parent_jaccard",
    "len_ratio_to_parent", "had_slash_s", "had_label_hashtag",
    "had_explicit_sarcasm_word", "en_score", "lang_method",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=None,
                    help="stratified sample size (default: full corpus)")
    ap.add_argument("--seed", type=int, default=3244)
    ap.add_argument("--no-langdetect", action="store_true",
                    help="skip langdetect adjudication of the ambiguous band")
    ap.add_argument("--sample-out", type=int, default=5000,
                    help="rows written to the human-inspectable CSV sample")
    args = ap.parse_args()

    ensure_dirs()
    df, n_full = load(args.sample, args.seed)

    log = CleaningLog(NAME, len(df))
    log.note("source_file", str(SRC))
    log.note("rows_in_source_file", n_full)
    log.note("sampled", bool(args.sample))

    df = clean(df, log, use_langdetect=not args.no_langdetect)
    df = engineer(df)

    df = df[[c for c in ORDERED_COLS if c in df.columns]].reset_index(drop=True)

    log.note("final_label_balance", df["label"].value_counts().to_dict())
    log.note("n_subreddits", int(df["subreddit"].nunique()))
    log.note("n_authors", int(df["author"].nunique()))
    log.note("date_range", [str(df["created_utc"].min()), str(df["created_utc"].max())])

    suffix = f"_sample{args.sample}" if args.sample else "_full"
    out_parquet = PROCESSED / f"{NAME}{suffix}.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"\n[save] {out_parquet}  ({out_parquet.stat().st_size / 1e6:.1f} MB)")

    csv_sample = SAMPLES / f"{NAME}{suffix}_head{args.sample_out}.csv"
    df.sample(n=min(args.sample_out, len(df)), random_state=args.seed) \
      .sort_index().to_csv(csv_sample, index=False, encoding="utf-8")
    print(f"[save] {csv_sample}")

    log.print_table()
    rep = CLEAN_REPORTS / f"{NAME}{suffix}_cleaning_report.json"
    log.save(rep)
    print(f"[save] {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
