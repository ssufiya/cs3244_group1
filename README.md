# For Real? Identifying markers of sarcasm through explainable machine learning

CS3244 Group 1. Sarcasm detection on Reddit comments, with the emphasis on
working out *which* signals a model uses rather than squeezing out the last point
of accuracy.

Start with [`docs/eda_report.md`](docs/eda_report.md) for what the data looks
like, or [`docs/experiment_report.md`](docs/experiment_report.md) for the first
base model.

---

## What is here

| Stage | Status |
|---|---|
| Source selection (primary, supplementary, scrapers) | done |
| Download — 278 MB, 15 files, SHA256 manifest | done |
| Cleaning — 1,010,826 → 913,769 rows | done |
| Exploratory analysis — 10 figures, written report | done |
| Train/val/test splits — 3 primary schemes, 4 supplementary | done |
| Leakage audit — 33 checks, all passing | done |
| Base Model Algorithm 2 (LightGBM) + tuning | done |
| Base Models 1 and 3, Week 8 ablations | next |

---

## Quick start

```bash
pip install -r requirements.txt
python pipeline/00_download_datasets.py    # ~278 MB, no Kaggle token needed
python pipeline/01_clean_primary.py        # 1.01M → 914k rows, ~3 min
python pipeline/02_clean_supplementary.py  # 4 corpora, ~30 s
python pipeline/03_eda.py                  # 10 figures + statistics
python pipeline/04_build_splits.py         # build data/splits/
python pipeline/05_verify_splits.py        # leakage audit + reference floor
```

Then, to train:

```python
import sys; sys.path.insert(0, "pipeline")
from load import load_split, xy

train, val, test = load_split("sarc_random")
X, y = xy(train)          # text + 22 numeric features, unsafe columns excluded
```

The Algorithm 2 experiment:

```bash
python modelling/10_build_features.py
python modelling/11_tune.py
python modelling/12_evaluate.py
python modelling/13_explain.py
python modelling/14_tuning_analysis.py
python modelling/15_stress_splits.py
```

---

## Layout

```
pipeline/            data acquisition, cleaning, EDA, splitting
  common/              text_cleaning, paths, report, viz
  00–05_*.py           the pipeline, in order
  scrapers/            provided but not run (need API credentials)
  load.py              dataset loader + the model-safe feature allow-list

modelling/           Base Model Algorithm 2 (LightGBM)
  config.py            every hyper-parameter and protocol constant
  10–15_*.py           features → tune → evaluate → explain → stress splits

docs/                datasets, data dictionary, split card, reports
figures/             eda/F01–F10, algo2/L01–L07
results/             machine-readable output from every stage
data/                manifest, catalogue, CSV previews (bulk is git-ignored)
```

---

## The data

| Split set | train | val | test | Role |
|---|---:|---:|---:|---|
| **`sarc_random`** | 639,638 | 137,066 | 137,065 | Default — base models and tuning |
| `sarc_temporal` | 484,447 | 182,441 | 242,091 | Stress test: generalising across time |
| `sarc_subreddit` | 639,838 | 135,123 | 135,534 | Stress test: generalising across communities |
| `news_headlines` | 19,862 | 4,256 | 4,257 | Cross-domain (re-split by us) |
| **`tweeteval_irony`** | 2,853 | 950 | 780 | Cross-platform test set |
| `figlang_reddit` | 3,729 | 659 | 1,756 | Multi-turn context ablation |
| `figlang_twitter` | 4,028 | 711 | 1,715 | Multi-turn context ablation |
| ~~`twitter_sarcasm`~~ | — | — | — | Rejected: synthetic, 240 unique texts |

The primary corpus is SARC (Khodak et al., 2018) via the balanced Kaggle file,
mirrored on HuggingFace so no Kaggle token is required. Full provenance in
[`docs/datasets.md`](docs/datasets.md); column-level detail in
[`docs/data_dictionary.md`](docs/data_dictionary.md).

Raw downloads, cleaned parquet and fitted models are not in the repository —
they are large and fully reproducible from the scripts. `data/MANIFEST.json`
records a URL, size and SHA256 for every source file.

---

## Main findings so far

From the [EDA](docs/eda_report.md):

- SARC carries no label leakage — 16 stray `/s` markers in 1.01M rows, at equal
  rates in both classes. Worth checking anyway; the Twitter corpora were a
  different story.
- 2.9% of the corpus contradicts itself: 4,043 texts appear under both labels.
  That is a real ceiling on accuracy and the best argument that context matters.
- Length is not the shortcut it looks like. Median is 9 words in both classes,
  because SARC sampled its negatives as siblings under the same parent.
- Two surface markers carry signal — exclamation marks at 2.81× and interjections
  at 1.97×. ALL-CAPS, ellipsis and repeated punctuation are all around 1.05×,
  which is nothing. Scare quotes run backwards at 0.67×.
- The distinctive vocabulary is mock-agreement machinery: *yeah*, *because*,
  *obviously*, *totally*, *clearly*, *sure*, *wow*.
- Subreddit is a very strong predictor — 8% in r/RoastMe to 79% in r/creepyPMs —
  but what it predicts is the `/s` convention, not sarcasm.

From the [Algorithm 2 experiment](docs/experiment_report.md):

- LightGBM ties TF-IDF + logistic regression (0.7059 vs 0.7060 accuracy). The
  bottleneck is the representation: SVD retains 15% of the text variance.
- Grouped permutation importance prices each information source. Context is worth
  +0.0099 AUC, surface markers +0.0685, subreddit +0.0207.
- Cross-platform transfer fails rather than degrades. On news headlines the AUC
  is 0.4865 — below chance.

---

## Traps worth knowing about

1. Split by subreddit or by time, never at random, for any generalisation claim.
   Both stress splits ship ready to use.
2. Never mix the three primary split schemes. They partition the same 913,769
   rows, so 70.1% of `sarc_temporal/test` and `sarc_subreddit/test` sits inside
   `sarc_random/train`. `modelling/15_stress_splits.py` shows the correct
   pattern.
3. Drop `ups`, `downs` and their missingness flags (collection artefacts),
   `score` (only known after the fact), `hour_utc` (no signal), `n_emoji` (always
   zero on 2009–2016 Reddit), and every `had_*` audit column (direct label leak).
   `MODEL_SAFE_FEATURES` already excludes them.
4. Bin `n_words`; the effect is non-monotonic.
5. Use `GroupKFold(groups=train["parent_key"])` for cross-validation inside the
   training set, or the folds reintroduce the sibling leak the splits prevent.
6. Control for subreddit in feature attribution, or the top "sarcasm markers"
   come back as *women*, *racist*, *white* — political-thread signal, not sarcasm.
7. Disclose the author overlap: 84% of `sarc_random` test authors also appear in
   training. SARC cannot avoid this.

---

## Environment

Python 3.12. `pip install -r requirements.txt`.

`langdetect` is optional — without it the language filter falls back to the
lexical heuristic alone. `scikit-learn` is needed from step 04 onward, and
`lightgbm` only for `modelling/`.

---

## References

- Khodak, Saunshi & Vodrahalli (2018). *A Large Self-Annotated Corpus for
  Sarcasm.* LREC.
- Misra & Arora (2023). *Sarcasm Detection using News Headline Dataset.* AI Open.
- Van Hee, Lefever & Hoste (2018). *SemEval-2018 Task 3: Irony Detection in
  English Tweets.*
- Ghosh, Vajpayee & Muresan (2020). *A Report on the 2020 Sarcasm Detection
  Shared Task.* FigLang @ ACL.
- Monroe, Colaresi & Quinn (2008). *Fightin' Words: Lexical Feature Selection and
  Evaluation for Identifying the Content of Political Conflict.* Political
  Analysis.
