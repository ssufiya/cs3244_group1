#!/usr/bin/env python
r"""
scrape_reddit_sarcasm.py — collect fresh /s-annotated Reddit comments with
their parent context, using the official Reddit API via PRAW.

*** THIS SCRIPT IS PROVIDED BUT NOT RUN. ***
It needs Reddit API credentials (free: https://www.reddit.com/prefs/apps,
choose "script"). Set them as environment variables before running.

--------------------------------------------------------------------------
Why this is worth having, when we already have 1M SARC comments
--------------------------------------------------------------------------
SARC stops at 2016-12. A model trained on it may be fitting 2016 memes,
2016 politics and 2016 subreddit norms rather than sarcasm. Scraping a few
thousand comments from 2024-2025 gives a *temporal holdout*: same platform,
same labelling convention, different era. If accuracy collapses on it, the
model learned the period, not the phenomenon — which is exactly the kind of
finding the "explainable ML" half of this project is for.

This is a different question from the Twitter scrape, which tests transfer
across *platforms*. Both are worth one paragraph in the report; neither is
worth blocking the project on.

--------------------------------------------------------------------------
How the labels are built (mirrors SARC's construction)
--------------------------------------------------------------------------
positive : the comment ends with a standalone "/s" token
negative : a sibling reply, under the same parent, with no "/s" anywhere

Sampling negatives as SIBLINGS rather than from a random stream is the whole
trick. It holds subreddit, thread topic and time constant, so the classifier
cannot separate the classes on topic alone. This is what SARC did, and it is
why SARC is a harder and more honest benchmark than a hashtag scrape.

--------------------------------------------------------------------------
Politeness / ToS
--------------------------------------------------------------------------
* PRAW honours Reddit's rate limits automatically; do not bypass them.
* Set a descriptive user agent that identifies the project (required).
* Author usernames are pseudonymous but still personal data. This script
  stores a salted hash instead of the raw username by default; pass
  --keep-usernames only if the project genuinely needs author-level features,
  and do not redistribute that column.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    pip install praw
    $env:REDDIT_CLIENT_ID="..."
    $env:REDDIT_CLIENT_SECRET="..."
    $env:REDDIT_USER_AGENT="cs3244-group1-sarcasm-research by /u/<your-username>"
    python scripts/scrapers/scrape_reddit_sarcasm.py --subreddits politics nba AskReddit --limit 200

Output
    data/raw/supplementary/reddit_scraped/comments_<UTC timestamp>.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import RAW  # noqa: E402
from common import text_cleaning as tc  # noqa: E402

OUT_DIR = RAW / "supplementary" / "reddit_scraped"

DEFAULT_SUBREDDITS = [
    # The subreddits that dominate SARC, so the temporal holdout stays
    # comparable to the training distribution.
    "politics", "worldnews", "nba", "nfl", "AskReddit", "funny", "pics",
    "todayilearned", "leagueoflegends", "news", "movies", "gaming",
]

SALT = os.environ.get("AUTHOR_HASH_SALT", "cs3244-group1")


def hash_author(name: str) -> str:
    return hashlib.sha256((SALT + name).encode("utf-8")).hexdigest()[:16]


def has_slash_s(text: str) -> bool:
    return bool(tc.RE_SLASH_S.search(text or ""))


def build_record(comment, parent_body: str, submission, label: int,
                 keep_usernames: bool) -> dict:
    raw = comment.body or ""
    clean, had_s, had_tag, had_word = tc.strip_label_markers(tc.normalise_unicode(raw))
    clean = " ".join(tc.replace_entities(tc.strip_markup(clean)).split())

    author = str(comment.author) if comment.author else "[deleted]"
    return {
        "comment_raw": raw,              # audit only - never feed to a model
        "comment": clean,
        "label": label,
        "label_source": "slash_s" if label == 1 else "sibling_without_slash_s",
        "stripped_slash_s": bool(had_s),
        "stripped_label_hashtag": bool(had_tag),
        "stripped_explicit_word": bool(had_word),
        "parent_comment": " ".join(
            tc.replace_entities(tc.strip_markup(
                tc.normalise_unicode(parent_body or ""))).split()),
        "subreddit": str(comment.subreddit),
        "submission_title": submission.title if submission else None,
        "author": author if keep_usernames else hash_author(author),
        "author_is_hashed": not keep_usernames,
        "score": comment.score,
        "created_utc": datetime.fromtimestamp(
            comment.created_utc, tz=timezone.utc).isoformat(),
        "comment_id": comment.id,
        "parent_id": comment.parent_id,
        "link_id": comment.link_id,
        "collected_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subreddits", nargs="*", default=DEFAULT_SUBREDDITS)
    ap.add_argument("--limit", type=int, default=100,
                    help="submissions to scan per subreddit")
    ap.add_argument("--sort", default="top", choices=["top", "hot", "new"])
    ap.add_argument("--time-filter", default="month",
                    choices=["hour", "day", "week", "month", "year", "all"],
                    help="only used with --sort top")
    ap.add_argument("--max-pairs", type=int, default=5000,
                    help="stop after this many (sarcastic, sibling) pairs")
    ap.add_argument("--keep-usernames", action="store_true",
                    help="store raw usernames instead of salted hashes")
    args = ap.parse_args()

    try:
        import praw
    except ImportError:
        print("praw is not installed:  pip install praw", file=sys.stderr)
        return 2

    missing = [v for v in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
                           "REDDIT_USER_AGENT") if not os.environ.get(v)]
    if missing:
        print("Missing environment variables: " + ", ".join(missing) + "\n"
              "Create a 'script' app at https://www.reddit.com/prefs/apps",
              file=sys.stderr)
        return 2

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
        check_for_async=False,
    )
    reddit.read_only = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"comments_{stamp}.jsonl"

    n_pairs = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for sub_name in args.subreddits:
            if n_pairs >= args.max_pairs:
                break
            print(f"\n=== r/{sub_name} ===")
            sub = reddit.subreddit(sub_name)
            listing = (sub.top(time_filter=args.time_filter, limit=args.limit)
                       if args.sort == "top"
                       else getattr(sub, args.sort)(limit=args.limit))

            for submission in listing:
                if n_pairs >= args.max_pairs:
                    break
                try:
                    submission.comments.replace_more(limit=0)
                except Exception as exc:                      # noqa: BLE001
                    print(f"  skip {submission.id}: {exc}", file=sys.stderr)
                    continue

                # Group every comment by its parent so we can pick siblings.
                by_parent: dict[str, list] = {}
                for c in submission.comments.list():
                    if not getattr(c, "body", None) or c.body in ("[deleted]", "[removed]"):
                        continue
                    by_parent.setdefault(c.parent_id, []).append(c)

                for parent_id, siblings in by_parent.items():
                    if n_pairs >= args.max_pairs:
                        break
                    pos = [c for c in siblings if has_slash_s(c.body)]
                    neg = [c for c in siblings if not has_slash_s(c.body)]
                    if not pos or not neg:
                        continue

                    try:
                        parent = reddit.comment(id=parent_id.split("_", 1)[1]) \
                            if parent_id.startswith("t1_") else None
                        parent_body = parent.body if parent else submission.selftext
                    except Exception:                          # noqa: BLE001
                        parent_body = ""

                    # One negative per positive keeps the file balanced.
                    for p, n in zip(pos, neg):
                        fh.write(json.dumps(build_record(p, parent_body, submission, 1,
                                                         args.keep_usernames),
                                            ensure_ascii=False) + "\n")
                        fh.write(json.dumps(build_record(n, parent_body, submission, 0,
                                                         args.keep_usernames),
                                            ensure_ascii=False) + "\n")
                        n_pairs += 1
                print(f"  {submission.id}: running total {n_pairs} pairs")

    print(f"\nwrote {n_pairs:,} balanced pairs ({2 * n_pairs:,} rows) -> {out_path}")
    print("Feed this through scripts/01_clean_primary.py-style cleaning before use;")
    print("use the `comment` field, never `comment_raw`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
