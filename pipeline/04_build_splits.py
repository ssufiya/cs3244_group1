#!/usr/bin/env python
"""
04_build_splits.py — turn the cleaned corpora into documented train/val/test
splits and assemble the split package under data/splits/.

Why three splits of the same primary corpus
-------------------------------------------
The proposal asks three ablation questions, and they need different controls.
One split cannot serve all of them honestly:

  sarc_random     grouped-random. The number comparable to published SARC work.
                  Use this for the baseline and for hyper-parameter tuning.
  sarc_temporal   train <2016, val 2016-H1, test 2016-H2. The EDA found an
                  11pp drift in sarcasm rate across 2009-2016, so a random
                  split quietly leaks the period. This one measures whether
                  the model survives it.
  sarc_subreddit  no subreddit appears in two splits. The EDA found sarcasm
                  rate spanning 8%-79% by subreddit, so a random split lets
                  the model memorise community priors. This is the honest
                  test for sub-question 3.

Leakage control applied to every scheme
---------------------------------------
SARC pairs a sarcastic comment with a sibling reply under the SAME parent.
If the two land on opposite sides of a split, the parent text appears in both
train and test and any context-using model is reading its own training data at
evaluation time. Every split here is therefore **grouped by parent comment**,
not by row. `sarc_subreddit` additionally groups by subreddit, which implies
parent grouping.

Usage
-----
    python pipeline/04_build_splits.py
    python pipeline/04_build_splits.py --only primary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import (  # noqa: E402
    CATALOG, DS_LEXICONS, DS_PRIMARY, DS_REJECTED, DS_SAMPLES, DS_SUPP,
    PROCESSED, RAW_LEXICONS, ROOT, ensure_dataset_dirs,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 3244
RATIOS = (0.70, 0.15, 0.15)

CATALOG: list[dict] = []


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def parent_key(s: pd.Series) -> pd.Series:
    """
    Stable id for a thread position, so siblings never straddle a split.

    Rows with an empty parent get a unique key: they are not one thread, they
    are many comments that happen to have lost their parent, and grouping them
    together would wedge all of them into a single split for no reason.
    """
    txt = s.fillna("").astype(str).str.strip().str.lower()
    keys = txt.map(lambda t: hashlib.blake2b(t.encode("utf-8"),
                                             digest_size=8).hexdigest())
    empty = txt.eq("")
    if empty.any():
        keys.loc[empty] = [f"__noparent_{i}" for i in range(int(empty.sum()))]
    return keys


def enforce_group_disjoint(df: pd.DataFrame, key_col: str,
                           split_col: str = "split") -> tuple[pd.DataFrame, int]:
    """
    Guarantee that no group spans two splits, whatever produced the assignment.

    Needed because only `sarc_random` groups on this key directly. A temporal
    boundary cuts through recurring threads, and the same parent *text* can
    appear under several subreddits, so both of the other schemes leak without
    this pass. Priority is train > val > test: the evaluation sets give way,
    so nothing a model trained on can reappear at scoring time.
    """
    rank = {"train": 0, "val": 1, "test": 2}
    owner = (df.assign(_r=df[split_col].map(rank))
               .groupby(key_col, sort=False)["_r"].min())
    keep = df[split_col].map(rank).to_numpy() == df[key_col].map(owner).to_numpy()
    return df[keep], int((~keep).sum())


def grouped_split(df: pd.DataFrame, group_col: str, ratios=RATIOS,
                  seed: int = SEED) -> pd.Series:
    """
    Assign each row to train/val/test so that a whole group stays together.

    Groups are shuffled and then greedily packed to hit the target row
    proportions — simple, deterministic, and it keeps large groups from
    blowing the ratios the way a naive per-group coin flip would.
    """
    sizes = df.groupby(group_col, sort=False).size()
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.index.to_numpy())
    sizes = sizes.reindex(order)

    n_total = int(sizes.sum())
    targets = [ratios[0] * n_total, ratios[1] * n_total, ratios[2] * n_total]
    filled = [0, 0, 0]
    names = ["train", "val", "test"]
    assign: dict = {}

    for g, n in sizes.items():
        # put the group wherever it is furthest from its target, in rows
        deficits = [targets[i] - filled[i] for i in range(3)]
        i = int(np.argmax(deficits))
        assign[g] = names[i]
        filled[i] += int(n)

    return df[group_col].map(assign)


def write_split(df: pd.DataFrame, out_dir: Path, name: str, *,
                role: str, stage: str, use_when: str, notes: str = "",
                text_col: str = "comment_clean",
                sample_rows: int = 200) -> dict:
    """Write train/val/test parquet + a CSV preview, and catalogue them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, files = {}, {}

    for split in ("train", "val", "test"):
        part = df[df["split"] == split].drop(columns=["split"])
        if part.empty:
            continue
        p = out_dir / f"{split}.parquet"
        part.to_parquet(p, index=False)
        counts[split] = {
            "rows": int(len(part)),
            "sarcastic_pct": round(float(part["label"].mean()) * 100, 2),
        }
        files[split] = str(p.relative_to(ROOT)).replace("\\", "/")

    prev = df.sample(n=min(sample_rows, len(df)), random_state=SEED).sort_index()
    prev_path = DS_SAMPLES / f"{name}_preview{sample_rows}.csv"
    prev.to_csv(prev_path, index=False, encoding="utf-8")

    entry = {
        "name": name,
        "dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "files": files,
        "preview_csv": str(prev_path.relative_to(ROOT)).replace("\\", "/"),
        "rows_total": int(len(df)),
        "splits": counts,
        "text_column": text_col,
        "label_column": "label",
        "role": role,
        "project_stage": stage,
        "use_when": use_when,
        "notes": notes,
    }
    CATALOG.append(entry)
    print(f"  [{name}] " + "  ".join(
        f"{k}={v['rows']:,} ({v['sarcastic_pct']}%)" for k, v in counts.items()))
    return entry


# --------------------------------------------------------------------------
# primary
# --------------------------------------------------------------------------
def build_primary() -> None:
    src = PROCESSED / "reddit_sarc_balanced_full.parquet"
    if not src.exists():
        raise SystemExit(f"missing {src} — run 01_clean_primary.py first")
    print(f"\n=== PRIMARY — {src.name} ===")
    df = pd.read_parquet(src)
    df["parent_key"] = parent_key(df["parent_clean"])
    df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce")
    print(f"  {len(df):,} rows, {df['parent_key'].nunique():,} distinct parents, "
          f"{df['subreddit'].nunique():,} subreddits")

    # -- A. grouped random -------------------------------------------------
    a = df.copy()
    a["split"] = grouped_split(a, "parent_key")
    write_split(
        a, DS_PRIMARY / "sarc_random", "sarc_random",
        role="DEFAULT training corpus. Grouped-random 70/15/15, siblings kept "
             "together so the parent comment never spans two splits.",
        stage="Week 7 — base models + hyper-parameter tuning",
        use_when="Use this for the three baseline algorithms and all tuning. "
                 "Report its test score as the headline number; it is the one "
                 "comparable to published SARC results.",
        notes="Optimistic relative to the two stress splits: subreddit and "
              "period are both shared across train/test by design.",
    )

    # -- B. temporal -------------------------------------------------------
    b = df.copy()
    ts = b["created_utc"]
    b["split"] = np.where(ts < "2016-01-01", "train",
                  np.where(ts < "2016-07-01", "val", "test"))
    # A time boundary cuts straight through recurring threads, so this is
    # where the leak would be; the audit in 05_verify_splits.py fails without it.
    b, n_straddle = enforce_group_disjoint(b, "parent_key")
    print(f"  [sarc_temporal] dropped {n_straddle:,} val/test rows whose parent "
          f"thread also appears in an earlier split")
    write_split(
        b, DS_PRIMARY / "sarc_temporal", "sarc_temporal",
        role="STRESS TEST — time generalisation. train <2016, val 2016-H1, "
             "test 2016-H2.",
        stage="Week 8 — ablation study",
        use_when="Run the tuned model from sarc_random on this without "
                 "re-tuning. The gap vs sarc_random is how much of the score "
                 "was period memorisation.",
        notes=f"{n_straddle:,} rows removed to keep parent threads on one side. "
              "Label rate differs by design (pre-2016 ~55% sarcastic, 2016 "
              "~48%) — that drift is the thing being measured, not a bug.",
    )

    # -- C. subreddit-disjoint --------------------------------------------
    c = df.copy()
    c["split"] = grouped_split(c, "subreddit", seed=SEED + 1)
    # Subreddit grouping does NOT imply parent grouping: the same parent text
    # ("What's your favourite X?") recurs across communities.
    c, n_cross = enforce_group_disjoint(c, "parent_key")
    print(f"  [sarc_subreddit] dropped {n_cross:,} rows whose parent text also "
          f"occurs in another subreddit's split")
    write_split(
        c, DS_PRIMARY / "sarc_subreddit", "sarc_subreddit",
        role="STRESS TEST — community generalisation. No subreddit appears in "
             "more than one split.",
        stage="Week 8 — ablation study (sub-question 3)",
        use_when="This is the split that actually answers 'is the subreddit "
                 "topic necessary?'. Train with and without the subreddit "
                 "feature here and compare; on sarc_random the comparison is "
                 "confounded because the model sees the same communities twice.",
        notes=f"{n_cross:,} rows removed so no parent thread spans splits. "
              "Subreddit sarcasm rate ranges 8%-79%, so train/test label "
              "balance will not match exactly. Expect the lowest scores of the "
              "three schemes; that is the point.",
    )


# --------------------------------------------------------------------------
# supplementary
# --------------------------------------------------------------------------
SUPP_SPEC = {
    "news_headlines": dict(
        role="Cross-domain evaluation: professionally edited prose "
             "(TheOnion vs HuffPost).",
        stage="Week 8 — ablation / generalisation",
        use_when="Train on SARC, test here to show how much of the model is "
                 "Reddit-register rather than sarcasm. Also usable as a small "
                 "in-domain training set of its own for comparison.",
        notes="RE-SPLIT BY US. The upstream train/test split was 100% "
              "contaminated (every test headline also appeared in train); "
              "these are fresh stratified 70/15/15 splits of the deduplicated "
              "rows. `article_link` was dropped — it is a perfect label leak.",
        resplit=True),
    "tweeteval_irony": dict(
        role="PRIMARY cross-platform test set. SemEval-2018 Task 3A, "
             "human-annotated.",
        stage="Week 7 (report) and Week 8 (ablation)",
        use_when="The headline transfer number. Do not train on it — train on "
                 "SARC and evaluate here. Official splits are preserved so "
                 "results stay comparable to published work; use `test` for "
                 "the reported figure and concatenate all three only if you "
                 "want a larger evaluation sample.",
        notes="Only corpus here whose labels were checked by humans rather "
              "than inferred from an author-written tag. 436 rows still "
              "contained #irony/#sarcasm/#not — stripped by the cleaner.",
        resplit=False),
    "figlang_reddit": dict(
        role="Multi-turn context ablation (Reddit).",
        stage="Week 8 — ablation (sub-question 1)",
        use_when="SARC gives exactly one parent turn, so it can only test "
                 "'context vs no context'. Use this to sweep 0 / 1 / n turns "
                 "and answer sub-question 1 as a curve.",
        notes="`context_clean` joins turns oldest-first with ' || '; "
              "`n_context_turns` gives the count. Val carved from the "
              "official train split.",
        resplit=False),
    "figlang_twitter": dict(
        role="Multi-turn context ablation (Twitter).",
        stage="Week 8 — ablation (sub-question 1)",
        use_when="Same as figlang_reddit, on the other platform — lets you "
                 "separate 'context helps' from 'context helps on Reddit'.",
        notes="Same context encoding. Val carved from the official train split.",
        resplit=False),
}


def stratified_split(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Label-stratified 70/15/15 for corpora with no grouping constraint."""
    from sklearn.model_selection import train_test_split
    idx = df.index.to_numpy()
    tr, rest = train_test_split(idx, train_size=RATIOS[0], random_state=seed,
                                stratify=df["label"].to_numpy())
    rel = RATIOS[1] / (RATIOS[1] + RATIOS[2])
    va, te = train_test_split(rest, train_size=rel, random_state=seed,
                              stratify=df.loc[rest, "label"].to_numpy())
    out = pd.Series("test", index=df.index)
    out.loc[tr] = "train"
    out.loc[va] = "val"
    return out


def build_supplementary() -> None:
    src = PROCESSED / "supplementary_all_clean.parquet"
    if not src.exists():
        raise SystemExit(f"missing {src} — run 02_clean_supplementary.py first")
    print(f"\n=== SUPPLEMENTARY — {src.name} ===")
    allsup = pd.read_parquet(src)

    for name, spec in SUPP_SPEC.items():
        d = allsup[allsup["dataset"] == name].copy().reset_index(drop=True)
        if d.empty:
            print(f"  [{name}] absent from the cleaned file — skipped")
            continue

        if spec["resplit"]:
            d["split"] = stratified_split(d)
        else:
            # Keep the upstream split; carve a val out of train if there is none.
            d["split"] = d["split"].replace({"validation": "val"})
            if "val" not in set(d["split"]):
                tr = d.index[d["split"] == "train"].to_numpy()
                from sklearn.model_selection import train_test_split
                keep, va = train_test_split(
                    tr, test_size=0.15, random_state=SEED,
                    stratify=d.loc[tr, "label"].to_numpy())
                d.loc[va, "split"] = "val"

        write_split(d, DS_SUPP / name, name,
                    role=spec["role"], stage=spec["stage"],
                    use_when=spec["use_when"], notes=spec["notes"],
                    text_col="text_clean")

    # rejected corpus — kept so the audit is reproducible, fenced off so it
    # cannot be picked up by a glob over the split folders
    rej = allsup[allsup["dataset"] == "twitter_sarcasm"]
    if not rej.empty:
        p = DS_REJECTED / "twitter_sarcasm_REJECTED.parquet"
        rej.to_parquet(p, index=False)
        (DS_REJECTED / "README.md").write_text(
            "# Rejected sources\n\n"
            "## `twitter_sarcasm_REJECTED.parquet` "
            "(`nikesh66/Sarcasm-dataset`)\n\n"
            "**Do not train or evaluate on this.** It is kept only so the "
            "rejection is reproducible.\n\n"
            "Named in the project proposal. Audit result: 5,000 raw rows "
            "contain **240 unique texts**, generated from a small template "
            f"set; {len(rej):,} survive deduplication.\n\n"
            "```\nx49  \"Can't wait for more of artists.\"\n"
            "x48  \"Can't wait for more of musicians.\"\n"
            "x43  \"Can't wait for more of writers.\"\n```\n\n"
            "It is synthetic, not collected, and cannot support a "
            "cross-platform claim. Use `supplementary/tweeteval_irony` "
            "instead — human-annotated and 20x larger after cleaning.\n\n"
            "Reporting the audit is worth more than the data: \"we checked a "
            "proposed source and rejected it with evidence\" is a stronger "
            "methods section than quietly using it.\n",
            encoding="utf-8")
        CATALOG.append({
            "name": "twitter_sarcasm_REJECTED",
            "dir": str(DS_REJECTED.relative_to(ROOT)).replace("\\", "/"),
            "files": {"all": str(p.relative_to(ROOT)).replace("\\", "/")},
            "rows_total": int(len(rej)),
            "splits": {},
            "text_column": "text_clean",
            "label_column": "label",
            "role": "REJECTED — synthetic/templated, not usable.",
            "project_stage": "n/a",
            "use_when": "Never. Cite the audit in the report; do not model on it.",
            "notes": "5,000 rows -> 240 unique texts. See _rejected/README.md.",
        })
        print(f"  [twitter_sarcasm] -> _rejected/ ({len(rej):,} rows, not for use)")


# --------------------------------------------------------------------------
def copy_lexicons() -> None:
    print("\n=== LEXICONS ===")
    spec = {
        "empath_categories.tsv": ("194 psycholinguistic word categories "
                                  "(free LIWC substitute)",
                                  "Week 8 — linguistic-cue features (sub-question 2)"),
        "vader_lexicon.txt": ("valence + intensity per token, incl. emoticons",
                              "Week 8 — hyperbole / intensifier features"),
        "nrc_emotion_lexicon.txt": ("8 emotions + 2 sentiments per word",
                                    "Week 8 — sentiment-incongruity features"),
        "SentiWordNet_3.0.0.txt": ("per-synset positive/negative scores",
                                   "Week 8 — sentiment-incongruity features"),
    }
    for fn, (what, stage) in spec.items():
        src = RAW_LEXICONS / fn
        if not src.exists():
            print(f"  [miss] {fn}")
            continue
        dst = DS_LEXICONS / fn
        shutil.copy2(src, dst)
        CATALOG.append({
            "name": f"lexicon:{fn}",
            "dir": str(DS_LEXICONS.relative_to(ROOT)).replace("\\", "/"),
            "files": {"all": str(dst.relative_to(ROOT)).replace("\\", "/")},
            "rows_total": None,
            "splits": {},
            "text_column": None,
            "label_column": None,
            "role": what,
            "project_stage": stage,
            "use_when": "Feature engineering only — never as a label source.",
            "notes": f"{dst.stat().st_size/1e6:.2f} MB",
        })
        print(f"  [copy] {fn}  ({dst.stat().st_size/1e6:.2f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["primary", "supplementary", "lexicons"])
    args = ap.parse_args()

    ensure_dataset_dirs()
    want = args.only or ["primary", "supplementary", "lexicons"]
    if "primary" in want:
        build_primary()
    if "supplementary" in want:
        build_supplementary()
    if "lexicons" in want:
        copy_lexicons()

    out = CATALOG
    out.write_text(json.dumps({
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "ratios": {"train": RATIOS[0], "val": RATIOS[1], "test": RATIOS[2]},
        "entries": CATALOG,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {out}  ({len(CATALOG)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
