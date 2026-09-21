# Data Cleaning Report

Produced by `pipeline/01_clean_primary.py` and `pipeline/02_clean_supplementary.py`.
Machine-readable per-step audits live beside this file as `results/cleaning/*_cleaning_report.json`.

---

## 1. Primary corpus — SARC balanced (Reddit)

`data/raw/primary/reddit_sarc/train-balanced-sarcasm.csv` → `data/processed/reddit_sarc_balanced_full.parquet`

**1,010,826 → 913,769 rows (90.4% retained), 183 s.**

| # | filter | removed | % of original | remaining |
|---|---|---:|---:|---:|
| 1 | empty / `[deleted]` / `[removed]` comment | 63 | 0.01% | 1,010,763 |
| 2 | empty after markup + leak removal | 79 | 0.01% | 1,010,684 |
| 3 | fewer than 2 words | 44,173 | 4.37% | 966,511 |
| 4 | more than 300 words / 2000 chars | 33 | 0.00% | 966,478 |
| 5 | non-English | 3,034 | 0.30% | 963,444 |
| 6 | exact duplicate (comment + label + parent) | 441 | 0.04% | 963,003 |
| 7 | **same text carries both labels** | **28,881** | **2.86%** | 934,122 |
| 8 | near-duplicate (normalised text) | 20,353 | 2.01% | 913,769 |

Final balance: 465,407 sarcastic / 448,362 not (50.9%).
13,970 subreddits, 254,321 authors, 2009-01-01 → 2016-12-31.

### What the steps do

**Unicode & markup.** Mojibake repair (`â€™` → `'`), double HTML unescaping
(Reddit dumps are frequently `&amp;gt;`), zero-width and control-character
removal, NFKC, quote/dash unification. Then Reddit markdown: code blocks,
`>` quote lines (parent text quoted by the author — not their own words),
`[anchor](url)` → `anchor`, `**bold**`, `^superscript`.

**Entity replacement.** URLs → `<URL>`, `/u/name` and `@handle` → `<USER>`.
`/r/subreddit` keeps the bare subreddit word, because topical signal is
exactly what sub-question 3 is about.

**Label-leak removal — the step that matters most.** Reddit labels come from
the author writing `/s`. If that token survives, a classifier scores ~100% by
memorising one bigram and the explainability half of the project is dead.

> **Result: SARC is already clean.** Only 16 comments in 1.01M still carried a
> `/s`, 4 a label hashtag, 1 an explicit `<sarcasm>`. Leak rate by class:
> 0.0028% (label 0) vs 0.0006% (label 1) — *equal within noise, and slightly
> higher on the negative class*, which is the opposite of what leakage looks
> like. The Kaggle authors stripped it. The check was still necessary, and it
> is a result worth stating in the write-up.

**Step 7 is the headline finding.** 4,043 distinct normalised texts appear with
*both* labels, covering 28,881 rows. Identical strings such as short generic
replies are sarcastic under one parent and sincere under another. Keeping both
copies teaches the model noise; removing them is correct *and* tells us the
achievable accuracy is bounded well below 100%.

**Language filter** is two-stage: a lexical heuristic (English function-word
rate + ASCII ratio) resolves the clear cases, and `langdetect` adjudicates only
the ambiguous band. 960,731 rows were decided by heuristic, 5,744 sent to
`langdetect`, 3 rejected outright. A full `langdetect` pass over 1M short
comments takes ~20 min and is unreliable below ~5 tokens; this keeps it to 0.6%
of rows.

### Engineered columns (46 total)

Surface: `n_chars n_words n_sentences avg_word_len n_exclam n_question
n_ellipsis n_repeat_punct n_allcaps_words allcaps_ratio n_emoji n_elongation
n_scare_quotes n_urls n_mentions n_hashtags n_interjections
starts_with_interjection`
Context: `parent_clean parent_n_words parent_n_chars parent_is_empty
parent_jaccard len_ratio_to_parent`
Temporal: `year month hour_utc weekday`
Audit: `had_slash_s had_label_hashtag had_explicit_sarcasm_word en_score
lang_method`

Features are computed **before** lowercasing, because capitalisation and
punctuation are the markers under investigation.

---

## 2. Supplementary corpora

| corpus | in | out | retained | verdict |
|---|---:|---:|---:|---|
| `news_headlines` | 55,328 | 28,375 | 51.3% | usable — **but re-split it yourself** |
| `tweeteval_irony` | 4,601 | 4,583 | 99.6% | **best cross-platform benchmark** |
| `figlang_reddit` | 6,200 | 6,144 | 99.1% | usable — multi-turn context |
| `figlang_twitter` | 6,800 | 6,454 | 94.9% | usable — multi-turn context |
| `twitter_sarcasm` | 5,000 | 224 | 4.5% | **synthetic — drop it** |

`figlang_reddit` loses 44 rows to "empty after markup": those responses consist
*entirely* of `>` quote lines re-stating the parent, with nothing of the
author's own. They are correctly unusable as target text.

### Three findings from the audit

**(a) `news_headlines` has no real test split.** 26,477 of its test rows were
already in train. Checked directly: of 26,602 unique test headlines,
**26,602 (100.0%) also appear in `train.json`.** The HuggingFace repo's split
is not a split. Concatenate and re-split.

**(b) `twitter_sarcasm` (`nikesh66/Sarcasm-dataset`) is generated, not
collected.** 5,000 rows contain **240 unique texts**, built from a small
template set:

```
x49  "Can't wait for more of artists."
x48  "Can't wait for more of musicians."
x43  "Can't wait for more of writers."
x43  "Can't wait for more of doctors."
```

94.9% of rows are exact duplicates; 224 survive deduplication. It cannot
support a cross-platform claim. It was named in the proposal, so the right
move is to say in the report that it was audited and rejected — and to use
`tweeteval_irony` (SemEval-2018 Task 3A, human-annotated) instead.

**(c) Even the human-annotated irony benchmark leaks hashtags.** 436
`tweeteval_irony` rows still contain a label hashtag, and they are enriched in
the positive class:

| hashtag | in label 0 | in label 1 |
|---|---:|---:|
| `#irony` | 12 | 101 |
| `#sarcasm` | 16 | 95 |
| `#not` | 96 | 117 |

TweetEval's irony set was harvested by hashtag and then human-verified, so
residual hashtags remain. They are stripped by `strip_label_markers`. Had they
been left in, ~2–3 percentage points of "cross-platform performance" would have
been pure label lookup.

### Unified supplementary schema

`data/processed/supplementary_all_clean.parquet` (45,780 rows):
`dataset platform split label text_clean context_clean has_context
context_n_words n_context_turns` + the 18 surface features, so any supplementary
corpus is directly comparable to the primary one.

---

## 3. Reproducing

```bash
python pipeline/00_download_datasets.py
python pipeline/01_clean_primary.py            # full; add --sample 200000 to iterate faster
python pipeline/02_clean_supplementary.py
python pipeline/03_eda.py
```

`data/MANIFEST.json` records a SHA256 for every downloaded file.
