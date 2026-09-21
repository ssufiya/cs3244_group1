#!/usr/bin/env python
"""
00_download_datasets.py — fetch every dataset used by the project.

All sources are public HuggingFace / GitHub mirrors, so no Kaggle API token is
required. Files are streamed to disk with resume-friendly size checks and a
SHA256 manifest is written so the download is reproducible.

Usage
-----
    python scripts/00_download_datasets.py                # everything
    python scripts/00_download_datasets.py --only primary # just the main corpus
    python scripts/00_download_datasets.py --force        # re-download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import (  # noqa: E402
    RAW_FIGLANG, RAW_IRONY, RAW_LEXICONS, RAW_NEWS, RAW_PRIMARY, RAW_TWEETS,
    RAW, ensure_dirs,
)

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


@dataclass
class Source:
    group: str          # primary | news | tweets | irony | figlang | lexicon
    dest_dir: Path
    filename: str
    url: str
    role: str           # what this file is for, recorded in the manifest
    notes: str = ""
    tags: list[str] = field(default_factory=list)


SOURCES: list[Source] = [
    # ---------------------------------------------------------------- PRIMARY
    # SARC (Self-Annotated Reddit Corpus), balanced split. This is the exact
    # `train-balanced-sarcasm.csv` distributed on Kaggle as `danofer/sarcasm`,
    # mirrored on HuggingFace so it downloads without a Kaggle token.
    Source(
        group="primary",
        dest_dir=RAW_PRIMARY,
        filename="train-balanced-sarcasm.csv",
        url=HF.format(repo="daniel2588/sarcasm", path="train-balanced-sarcasm.csv"),
        role="Main corpus: ~1.01M balanced Reddit comments, self-annotated with /s.",
        notes="Columns: label, comment, author, subreddit, score, ups, downs, "
              "date, created_utc, parent_comment.",
        tags=["reddit", "self-annotated", "context", "metadata"],
    ),

    # ----------------------------------------------------------- SUPPLEMENTARY
    # 1. News headlines (Misra & Arora). Professionally written -> almost no
    #    typos/slang. Used as a low-noise contrast domain.
    Source(
        group="news",
        dest_dir=RAW_NEWS,
        filename="train.json",
        url=HF.format(repo="raquiba/Sarcasm_News_Headline", path="train.json"),
        role="Low-noise domain: TheOnion (sarcastic) vs HuffPost (serious) headlines.",
        tags=["news", "headline", "clean-text", "editorially-labelled"],
    ),
    Source(
        group="news",
        dest_dir=RAW_NEWS,
        filename="test.json",
        url=HF.format(repo="raquiba/Sarcasm_News_Headline", path="test.json"),
        role="Held-out split of the news-headline corpus.",
        tags=["news", "headline"],
    ),

    # 2. Twitter #sarcasm corpus named in the proposal.
    Source(
        group="tweets",
        dest_dir=RAW_TWEETS,
        filename="sarcasm_tweets.csv",
        url=HF.format(repo="nikesh66/Sarcasm-dataset", path="sarcasm_tweets.csv"),
        role="Cross-platform transfer test: hashtag-labelled tweets with slang.",
        tags=["twitter", "hashtag-labelled", "slang", "noisy"],
    ),

    # 3. SemEval-2018 Task 3 irony, via TweetEval. HUMAN annotated, not
    #    hashtag/-/s self-labelled -> the cleanest external yardstick we have
    #    for how much of the model is learning the /s convention itself.
    Source(
        group="irony",
        dest_dir=RAW_IRONY,
        filename="train.parquet",
        url=HF.format(repo="cardiffnlp/tweet_eval", path="irony/train-00000-of-00001.parquet"),
        role="Gold human-annotated irony benchmark (SemEval-2018 Task 3A) - train.",
        tags=["twitter", "human-annotated", "benchmark"],
    ),
    Source(
        group="irony",
        dest_dir=RAW_IRONY,
        filename="validation.parquet",
        url=HF.format(repo="cardiffnlp/tweet_eval", path="irony/validation-00000-of-00001.parquet"),
        role="SemEval-2018 Task 3A - validation.",
        tags=["twitter", "human-annotated", "benchmark"],
    ),
    Source(
        group="irony",
        dest_dir=RAW_IRONY,
        filename="test.parquet",
        url=HF.format(repo="cardiffnlp/tweet_eval", path="irony/test-00000-of-00001.parquet"),
        role="SemEval-2018 Task 3A - test.",
        tags=["twitter", "human-annotated", "benchmark"],
    ),

    # 4. FigLang-2020 shared task: every example carries the FULL conversation
    #    chain, not just one parent. Directly supports sub-question 1
    #    ("is thread context necessary?") with more than one turn of context.
    Source(
        group="figlang",
        dest_dir=RAW_FIGLANG,
        filename="reddit_training.jsonl",
        url=HF.format(repo="tasksource/figlang2020-sarcasm",
                      path="sarcasm_detection_shared_task_reddit_training.jsonl"),
        role="Multi-turn context ablation (Reddit) - train.",
        tags=["reddit", "multi-turn-context"],
    ),
    Source(
        group="figlang",
        dest_dir=RAW_FIGLANG,
        filename="reddit_testing.jsonl",
        url=HF.format(repo="tasksource/figlang2020-sarcasm",
                      path="sarcasm_detection_shared_task_reddit_testing.jsonl"),
        role="Multi-turn context ablation (Reddit) - test.",
        tags=["reddit", "multi-turn-context"],
    ),
    Source(
        group="figlang",
        dest_dir=RAW_FIGLANG,
        filename="twitter_training.jsonl",
        url=HF.format(repo="tasksource/figlang2020-sarcasm",
                      path="sarcasm_detection_shared_task_twitter_training.jsonl"),
        role="Multi-turn context ablation (Twitter) - train.",
        tags=["twitter", "multi-turn-context"],
    ),
    Source(
        group="figlang",
        dest_dir=RAW_FIGLANG,
        filename="twitter_testing.jsonl",
        url=HF.format(repo="tasksource/figlang2020-sarcasm",
                      path="sarcasm_detection_shared_task_twitter_testing.jsonl"),
        role="Multi-turn context ablation (Twitter) - test.",
        tags=["twitter", "multi-turn-context"],
    ),

    # ------------------------------------------------------- LEXICON RESOURCES
    # LIWC alternatives named in the proposal, for the "linguistic cue" features.
    Source(
        group="lexicon",
        dest_dir=RAW_LEXICONS,
        filename="empath_categories.tsv",
        url="https://raw.githubusercontent.com/Ejhfast/empath-client/master/empath/data/categories.tsv",
        role="Empath: 194 psycholinguistic word categories (free LIWC substitute).",
        tags=["lexicon", "empath", "liwc-alternative"],
    ),
    Source(
        group="lexicon",
        dest_dir=RAW_LEXICONS,
        filename="vader_lexicon.txt",
        url="https://raw.githubusercontent.com/cjhutto/vaderSentiment/master/vaderSentiment/vader_lexicon.txt",
        role="VADER valence lexicon: sentiment polarity + intensity (booster words).",
        tags=["lexicon", "sentiment", "intensity"],
    ),
    Source(
        group="lexicon",
        dest_dir=RAW_LEXICONS,
        filename="nrc_emotion_lexicon.txt",
        url="https://raw.githubusercontent.com/dinbav/LeXmo/master/"
            "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
        role="NRC EmoLex: 8 emotions + 2 sentiments per word, for incongruity features.",
        tags=["lexicon", "emotion"],
    ),
    Source(
        group="lexicon",
        dest_dir=RAW_LEXICONS,
        filename="SentiWordNet_3.0.0.txt",
        url="https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/"
            "SentiWordNet_3.0.0.txt",
        role="SentiWordNet 3.0: per-synset positivity/negativity, the 'general word net' "
             "named in the proposal.",
        tags=["lexicon", "sentiment", "wordnet"],
    ),
]


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(src: Source, force: bool) -> dict:
    dest = src.dest_dir / src.filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"  [skip] {src.filename:<34} already present ({human(dest.stat().st_size)})")
    else:
        print(f"  [get ] {src.filename:<34} <- {src.url}")
        t0 = time.time()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(src.url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done, next_mark = 0, 10
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total and 100 * done / total >= next_mark:
                        print(f"         {next_mark:3d}%  {human(done)}/{human(total)}")
                        next_mark += 10
        tmp.replace(dest)
        print(f"         done in {time.time() - t0:.1f}s")

    return {
        "group": src.group,
        "file": str(dest.relative_to(RAW.parent.parent)).replace("\\", "/"),
        "bytes": dest.stat().st_size,
        "sha256": sha256_of(dest),
        "url": src.url,
        "role": src.role,
        "notes": src.notes,
        "tags": src.tags,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["primary", "news", "tweets", "irony", "figlang", "lexicon"],
                    help="restrict to these groups")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    ensure_dirs()
    wanted = [s for s in SOURCES if args.only is None or s.group in args.only]

    manifest, failures = [], []
    current = None
    for src in wanted:
        if src.group != current:
            current = src.group
            print(f"\n=== {current.upper()} ===")
        try:
            manifest.append(download(src, args.force))
        except Exception as exc:                      # noqa: BLE001
            print(f"  [FAIL] {src.filename}: {exc}")
            failures.append({"file": src.filename, "url": src.url, "error": str(exc)})

    out = RAW / "MANIFEST.json"
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": manifest,
        "failures": failures,
    }
    # merge with any previous manifest so partial runs don't lose entries
    if out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
            seen = {f["file"] for f in manifest}
            payload["files"] = manifest + [f for f in prev.get("files", [])
                                           if f["file"] not in seen]
        except Exception:                             # noqa: BLE001
            pass
    payload["files"].sort(key=lambda f: (f["group"], f["file"]))
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    total = sum(f["bytes"] for f in payload["files"])
    print(f"\nManifest -> {out}")
    print(f"{len(payload['files'])} files, {human(total)} on disk, {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
