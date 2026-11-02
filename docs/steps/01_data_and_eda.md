# Step 1 — Data, cleaning, splits and EDA

Notebook: [`notebooks/01_data_and_eda.ipynb`](../../notebooks/01_data_and_eda.ipynb) · Pipeline code: `pipeline/notebooks/01–05` · Detailed reports: [`cleaning_report.md`](../cleaning_report.md), [`eda_report.md`](../eda_report.md), [`split_card.md`](../split_card.md), [`datasets.md`](../datasets.md), [`data_dictionary.md`](../data_dictionary.md)

## What was done

1. **Download** (`pipeline/00_download_datasets.py`): 15 source files, 278 MB, each with URL, size and SHA256 in `data/MANIFEST.json`. SARC comes from a HuggingFace mirror of the balanced Kaggle file, so no Kaggle token is needed.
2. **Cleaning** (`01_clean_primary`, `02_clean_supplementary`): empty or deleted comments, duplicates, non-English text and leaked `/s` markers are removed; 1,010,826 → 913,769 SARC rows (90.4% kept). The synthetic `twitter_sarcasm` set (240 unique texts) was rejected.
3. **EDA** (`03_eda`): ten figures in `figures/eda/`, statistics in `results/eda/eda_statistics.json`.
4. **Splits** (`04_build_splits`, `05_verify_splits`): three primary schemes over the same rows plus four supplementary sets; a 33-check leakage audit passes (`results/splits/split_verification.json`).

## Splits used by every later step

| Split set | train | val | test | Role |
|---|---:|---:|---:|---|
| `sarc_random` | 639,638 | 137,066 | 137,065 | default; all tuning |
| `sarc_subreddit` | 639,838 | 135,123 | 135,534 | stress test: disjoint communities |
| `sarc_temporal` | 484,447 | 182,441 | 242,091 | stress test across time (baselines only) |
| `figlang_reddit` / `figlang_twitter` / `tweeteval_irony` / `news_headlines` | — | — | 1,756 / 1,715 / 780 / 4,257 | transfer only, no target-side fitting |

## Key findings

- No label leakage (16 stray `/s` in 1.01M rows, equal in both classes), but **2.9% of texts appear under both labels**: a hard ceiling on accuracy and an argument for context.
- **Length is not a shortcut**: median 9 words in both classes, because SARC samples negatives as siblings under the same parent.
- Only two surface markers carry signal: exclamation marks (2.81×) and interjections (1.97×). ALL-CAPS, ellipsis and repeated punctuation are about 1.05×.
- Distinctive sarcastic vocabulary is **mock agreement**: *yeah, because, obviously, totally, clearly, sure, wow*.
- Subreddit is strongly predictive (8% sarcastic in r/RoastMe, 79% in r/creepyPMs) but mostly reflects the `/s` convention, so later steps avoid it.

## Rules carried into modelling

- Use `sarc_subreddit` (or `sarc_temporal`) for any generalisation claim; never mix the three primary schemes, because they share rows.
- Drop `ups`, `downs`, `score`, `hour_utc`, `n_emoji` and all `had_*` audit columns; `MODEL_SAFE_FEATURES` in `pipeline/load.py` already does this.
- Cross-validate inside train with `GroupKFold(groups=parent_key)` to avoid sibling leakage.
- Disclose author overlap: 84% of `sarc_random` test authors also appear in training.
