#!/usr/bin/env python
r"""
scrape_twitter_sarcasm.py — collect #sarcasm / #irony tweets for the
cross-platform evaluation set described in the proposal.

*** THIS SCRIPT IS PROVIDED BUT NOT RUN. ***
It needs an X/Twitter API bearer token, and the free tier is rate-limited to
roughly 100 posts per month, which is not enough for a usable evaluation set.
Read the "Which route should we actually take" section below before using it.

--------------------------------------------------------------------------
Which route should we actually take?
--------------------------------------------------------------------------
Route A (recommended) - skip scraping entirely.
    `tweeteval_irony` (SemEval-2018 Task 3A, already downloaded by
    scripts/00_download_datasets.py) is 4.6k human-annotated tweets. It is a
    published benchmark, it is citable, and it has no ToS or rate-limit
    problem. For "does the model transfer off Reddit?" it answers the
    question better than anything we could scrape in a week.

Route B - X API v2 recent search. What this script implements.
    Needs: a developer account, and at minimum the Basic tier
    (~USD 100/month) for a meaningful volume. `recent search` only reaches
    back 7 days; `full archive` needs the Academic/Pro tier.

Route C - Reddit via PRAW (see scrape_reddit_sarcasm.py in this folder).
    Free, well-documented API, and it extends the SAME distribution as our
    primary corpus, so it is the right choice for a *fresh-period holdout*
    (e.g. 2024-2025 comments) rather than for cross-platform transfer.

--------------------------------------------------------------------------
Labelling caveat that matters more than the plumbing
--------------------------------------------------------------------------
A tweet is labelled sarcastic because its author appended #sarcasm. That
hashtag IS the label, so it must be removed from the text before modelling —
otherwise a classifier reaches ~100% accuracy by memorising one token and
learns nothing about sarcasm. This script strips it at write time via
`common.text_cleaning.strip_label_markers`, and records that it did.

The negative class is the harder problem: tweets *without* #sarcasm are not
reliably non-sarcastic, they are merely unlabelled. Sampling negatives from a
neutral stream gives a topic-shifted control — a model can then separate the
classes on topic alone. Mitigations, in order of preference:
  1. draw negatives from the same authors' non-hashtagged tweets;
  2. match negatives to positives on topic/keyword;
  3. at minimum, report the confound honestly in the write-up.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    set TWITTER_BEARER_TOKEN=...          # PowerShell: $env:TWITTER_BEARER_TOKEN="..."
    python scripts/scrapers/scrape_twitter_sarcasm.py --max-tweets 2000

Output
    data/raw/supplementary/twitter_scraped/tweets_<UTC timestamp>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import RAW  # noqa: E402
from common import text_cleaning as tc  # noqa: E402

OUT_DIR = RAW / "supplementary" / "twitter_scraped"
SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

# Only original, English, non-retweet text. Retweets duplicate content and
# would otherwise dominate the sample.
BASE_FILTER = "-is:retweet -is:reply lang:en"

POSITIVE_QUERY = f"(#sarcasm OR #sarcastic OR #irony OR #ironic) {BASE_FILTER}"
# Deliberately topic-neutral, high-volume, to keep the control group broad.
NEGATIVE_QUERY = f"(the OR and OR because) {BASE_FILTER} -#sarcasm -#irony -#sarcastic -#not"

TWEET_FIELDS = "id,text,created_at,author_id,public_metrics,conversation_id,lang"


def search(token: str, query: str, want: int, pause: float = 3.0) -> list[dict]:
    """Page through /2/tweets/search/recent until `want` tweets are collected."""
    headers = {"Authorization": f"Bearer {token}",
               "User-Agent": "cs3244-group1-sarcasm-research"}
    out: list[dict] = []
    next_token = None

    while len(out) < want:
        params = {
            "query": query,
            "max_results": min(100, want - len(out)),
            "tweet.fields": TWEET_FIELDS,
        }
        if next_token:
            params["next_token"] = next_token

        r = requests.get(SEARCH_URL, headers=headers, params=params, timeout=60)

        if r.status_code == 429:
            # Respect the reset header rather than hammering the endpoint.
            reset = int(r.headers.get("x-rate-limit-reset", time.time() + 900))
            wait = max(5, reset - int(time.time()) + 1)
            print(f"  rate limited; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if r.status_code != 200:
            print(f"  HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
            break

        payload = r.json()
        batch = payload.get("data", [])
        if not batch:
            break
        out.extend(batch)
        print(f"  +{len(batch)} (total {len(out)})")

        next_token = payload.get("meta", {}).get("next_token")
        if not next_token:
            break
        time.sleep(pause)

    return out[:want]


def to_record(raw: dict, label: int, query: str) -> dict:
    text = raw.get("text", "")
    clean, had_s, had_tag, had_word = tc.strip_label_markers(tc.normalise_unicode(text))
    clean = tc.replace_entities(clean)
    metrics = raw.get("public_metrics", {}) or {}
    return {
        # The raw text is kept so the label-stripping stays auditable, but it
        # must never be fed to a model.
        "text_raw": text,
        "text": " ".join(clean.split()),
        "label": label,
        "label_source": "hashtag" if label == 1 else "absence_of_hashtag",
        "stripped_label_hashtag": bool(had_tag),
        "stripped_slash_s": bool(had_s),
        "stripped_explicit_word": bool(had_word),
        "tweet_id": raw.get("id"),
        "author_id": raw.get("author_id"),
        "conversation_id": raw.get("conversation_id"),
        "created_at": raw.get("created_at"),
        "lang": raw.get("lang"),
        "like_count": metrics.get("like_count"),
        "retweet_count": metrics.get("retweet_count"),
        "reply_count": metrics.get("reply_count"),
        "query": query,
        "collected_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-tweets", type=int, default=1000,
                    help="per class; total collected is roughly 2x this")
    ap.add_argument("--positives-only", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not token:
        print("TWITTER_BEARER_TOKEN is not set.\n"
              "This script cannot run without X API credentials. Consider Route A\n"
              "in the module docstring: tweeteval_irony is already downloaded and\n"
              "is a better cross-platform benchmark than anything we can scrape.",
              file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"tweets_{stamp}.jsonl"

    records: list[dict] = []
    print(f"[+] positives: {POSITIVE_QUERY}")
    records += [to_record(t, 1, POSITIVE_QUERY)
                for t in search(token, POSITIVE_QUERY, args.max_tweets)]

    if not args.positives_only:
        print(f"[-] negatives: {NEGATIVE_QUERY}")
        records += [to_record(t, 0, NEGATIVE_QUERY)
                    for t in search(token, NEGATIVE_QUERY, args.max_tweets)]

    seen, deduped = set(), []
    for rec in records:
        if rec["tweet_id"] in seen:
            continue
        seen.add(rec["tweet_id"])
        deduped.append(rec)

    with out_path.open("w", encoding="utf-8") as fh:
        for rec in deduped:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_pos = sum(r["label"] == 1 for r in deduped)
    stripped = sum(r["stripped_label_hashtag"] for r in deduped)
    print(f"\nwrote {len(deduped):,} tweets ({n_pos:,} positive) -> {out_path}")
    print(f"label hashtags stripped from {stripped:,} tweets")
    print("Reminder: use the `text` field, never `text_raw`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
