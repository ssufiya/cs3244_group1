#!/usr/bin/env python
"""
02_clean_supplementary.py — clean the four supplementary corpora into one
shared schema so they can be compared with each other and with SARC.

Unified schema
--------------
    dataset   str   news_headlines | twitter_sarcasm | tweeteval_irony | figlang_*
    split     str   train | validation | test
    platform  str   news | twitter | reddit
    label     int8  1 = sarcastic/ironic, 0 = not
    text      str   cleaned target text
    context   str   cleaned conversational context ("" when none exists)
    + the same surface features produced for the primary corpus

Label-leak note
---------------
The Twitter corpora are labelled BY the hashtag (#sarcasm/#irony/#not) and
FigLang responses may still carry "/s". Those tokens are stripped here and the
strip rate is reported per class — for the hashtag-labelled sets this is the
difference between a real task and a lookup table.

Usage
-----
    python scripts/02_clean_supplementary.py
    python scripts/02_clean_supplementary.py --only news twitter
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import (  # noqa: E402
    CLEAN_REPORTS, PROCESSED, RAW_FIGLANG, RAW_IRONY, RAW_NEWS, RAW_TWEETS,
    SAMPLES, ensure_dirs,
)
from common.report import CleaningLog          # noqa: E402
from common import text_cleaning as tc         # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FEATURE_COLS = [
    "n_chars", "n_words", "n_sentences", "avg_word_len", "n_exclam",
    "n_question", "n_ellipsis", "n_repeat_punct", "n_allcaps_words",
    "allcaps_ratio", "n_emoji", "n_elongation", "n_scare_quotes", "n_urls",
    "n_mentions", "n_hashtags", "n_interjections", "starts_with_interjection",
]


# --------------------------------------------------------------------------
# loaders -> (raw DataFrame with columns: split, label, text, context)
# --------------------------------------------------------------------------
def load_news() -> pd.DataFrame:
    frames = []
    for split, fn in [("train", "train.json"), ("test", "test.json")]:
        rows = []
        with (RAW_NEWS / fn).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        d = pd.DataFrame(rows)
        d = d.rename(columns={"is_sarcastic": "label", "headline": "text"})
        d["split"] = split
        # The article URL is a perfect label leak (theonion.com vs
        # huffingtonpost.com), so it is dropped rather than kept as context.
        d["context"] = ""
        frames.append(d[["split", "label", "text", "context"]])
    return pd.concat(frames, ignore_index=True)


def load_twitter() -> pd.DataFrame:
    d = pd.read_csv(RAW_TWEETS / "sarcasm_tweets.csv")
    d = d.rename(columns={"Tweet": "text", "Sarcasm (yes/no)": "label"})
    d["label"] = (d["label"].astype(str).str.strip().str.lower()
                  .map({"yes": 1, "no": 0}))
    d["split"] = "train"
    d["context"] = ""
    return d[["split", "label", "text", "context"]]


def load_irony() -> pd.DataFrame:
    frames = []
    for split in ("train", "validation", "test"):
        d = pd.read_parquet(RAW_IRONY / f"{split}.parquet")
        d["split"] = split
        d["context"] = ""
        frames.append(d[["split", "label", "text", "context"]])
    return pd.concat(frames, ignore_index=True)


def load_figlang(platform: str) -> pd.DataFrame:
    frames = []
    for split, fn in [("train", f"{platform}_training.jsonl"),
                      ("test", f"{platform}_testing.jsonl")]:
        rows = []
        with (RAW_FIGLANG / fn).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        d = pd.DataFrame(rows)
        d["split"] = split
        d["label"] = d.get("label", pd.Series([None] * len(d))) \
                      .map({"SARCASM": 1, "NOT_SARCASM": 0})
        d["text"] = d["response"]
        # Context is a list of turns, oldest first. Join newest-last with a
        # separator so a model can still see turn boundaries.
        d["context"] = d["context"].map(
            lambda c: " || ".join(c) if isinstance(c, list) else (c or ""))
        d["n_context_turns"] = d["context"].map(lambda c: c.count(" || ") + 1 if c else 0)
        frames.append(d[["split", "label", "text", "context", "n_context_turns"]])
    return pd.concat(frames, ignore_index=True)


DATASETS = {
    "news_headlines":   ("news",    load_news),
    "twitter_sarcasm":  ("twitter", load_twitter),
    "tweeteval_irony":  ("twitter", load_irony),
    "figlang_reddit":   ("reddit",  lambda: load_figlang("reddit")),
    "figlang_twitter":  ("twitter", lambda: load_figlang("twitter")),
}


# --------------------------------------------------------------------------
def clean_one(name: str, platform: str, raw: pd.DataFrame,
              use_langdetect: bool) -> tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog(name, len(raw))
    log.note("platform", platform)
    log.note("splits_in", raw["split"].value_counts().to_dict())

    df = raw.copy()

    # Unlabelled rows (FigLang's public test split ships without labels).
    unlabelled = df["label"].isna()
    log.note("unlabelled_rows", int(unlabelled.sum()))
    df = df[~unlabelled]
    log.drop("no usable label", len(df))

    df["label"] = df["label"].astype("int8")

    bad = tc.is_deleted(df["text"]) | df["text"].isna()
    df = df[~bad]
    log.drop("empty / placeholder text", len(df))

    print(f"  [{name}] normalising text ...")
    t = tc.clean_series(df["text"], keep_subreddit=True)
    log.note("leak_slash_s", int(t["had_slash_s"].sum()))
    log.note("leak_label_hashtag", int(t["had_label_hashtag"].sum()))
    log.note("leak_explicit_word", int(t["had_explicit_sarcasm_word"].sum()))
    any_leak = t["had_slash_s"] | t["had_label_hashtag"] | t["had_explicit_sarcasm_word"]
    log.note("leak_rate_by_label",
             {int(k): round(float(v), 6)
              for k, v in any_leak.groupby(df["label"].values).mean().items()})

    ctx = tc.clean_series(df["context"].fillna(""), keep_subreddit=True)

    df = df.assign(text_clean=t["text_clean"].values,
                   context_clean=ctx["text_clean"].values,
                   **{c: t[c].values for c in FEATURE_COLS})
    df["context_n_words"] = ctx["n_words"].values
    df["has_context"] = df["context_clean"].str.strip().ne("")

    empty_after = df["text_clean"].str.strip().eq("")
    log.note("became_empty_after_cleaning", int(empty_after.sum()))
    df = df[~empty_after]
    log.drop("empty after markup/leak removal", len(df))

    df = df[df["n_words"] >= 2]
    log.drop("fewer than 2 words", len(df))

    df = tc.add_language_flags(df, "text_clean", use_langdetect=use_langdetect)
    log.note("lang_method_counts", df["lang_method"].value_counts().to_dict())
    df = df[df["is_english"]]
    log.drop("non-English", len(df))

    df["_key"] = tc.dedup_key(df["text_clean"])
    df = df.drop_duplicates(subset=["_key", "label", "split"], keep="first")
    log.drop("exact dup within split", len(df))

    conflict = df.groupby("_key")["label"].transform("nunique") > 1
    log.note("label_conflicting_rows", int(conflict.sum()))
    df = df[~conflict]
    log.drop("same text carries both labels", len(df))

    # Cross-split leakage is fatal for the held-out evaluation: keep the
    # training copy, drop the test copy.
    order = {"train": 0, "validation": 1, "test": 2}
    df = df.sort_values("split", key=lambda s: s.map(order))
    df = df.drop_duplicates(subset=["_key"], keep="first")
    log.drop("duplicate across splits (test copy dropped)", len(df))
    df = df.drop(columns=["_key"])

    df.insert(0, "dataset", name)
    df.insert(1, "platform", platform)
    log.note("final_label_balance", df["label"].value_counts().to_dict())
    log.note("splits_out", df["split"].value_counts().to_dict())
    return df.reset_index(drop=True), log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, choices=list(DATASETS))
    ap.add_argument("--no-langdetect", action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    wanted = args.only or list(DATASETS)
    combined, summary = [], []

    for name in wanted:
        platform, loader = DATASETS[name]
        print(f"\n=== {name} ({platform}) ===")
        raw = loader()
        print(f"  loaded {len(raw):,} rows")
        df, log = clean_one(name, platform, raw, use_langdetect=not args.no_langdetect)

        out = PROCESSED / f"{name}_clean.parquet"
        df.to_parquet(out, index=False)
        print(f"  [save] {out}  ({len(df):,} rows)")
        df.head(300).to_csv(SAMPLES / f"{name}_head300.csv", index=False, encoding="utf-8")

        log.print_table()
        log.save(CLEAN_REPORTS / f"{name}_cleaning_report.json")

        keep = ["dataset", "platform", "split", "label", "text_clean",
                "context_clean", "has_context", "context_n_words",
                "n_context_turns"] + FEATURE_COLS
        combined.append(df[[c for c in keep if c in df.columns]])
        summary.append(log.to_dict())

    if combined:
        allsup = pd.concat(combined, ignore_index=True)
        out = PROCESSED / "supplementary_all_clean.parquet"
        allsup.to_parquet(out, index=False)
        print(f"\n[save] combined -> {out}  ({len(allsup):,} rows)")

        (CLEAN_REPORTS / "supplementary_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")
        print("\n=== supplementary overview ===")
        print(allsup.groupby(["dataset", "platform", "label"]).size()
              .unstack(fill_value=0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
