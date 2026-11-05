# For Real? Identifying markers of sarcasm through explainable machine learning

CS3244 Group 1. Sarcasm detection on Reddit comments, with the emphasis on
working out *which* signals a model uses rather than squeezing out the last point
of accuracy.

Start with [`docs/SUMMARY.md`](docs/SUMMARY.md) for the whole project on one
page, then follow the step-by-step notebooks.

## Project walkthrough

| Step | Notebook | Document |
|---|---|---|
| 1 Data, cleaning, splits, EDA | `notebooks/01_data_and_eda.ipynb` | `docs/steps/01_data_and_eda.md` |
| 2 Team baselines | `notebooks/02_baseline_models.ipynb` | `docs/steps/02_baseline_models.md` |
| 3 Feature engineering | `notebooks/03_feature_engineering.ipynb` | `docs/steps/03_feature_engineering.md` |
| 4 Interpretable EBM | `notebooks/04_interpretable_ebm.ipynb` | `docs/steps/04_interpretable_ebm.md` |
| 5 From-scratch neural networks | `notebooks/05_scratch_neural_networks.ipynb` | `docs/steps/05_scratch_neural_networks.md` |
| 6 Pretrained reference (not submitted) | `notebooks/06_pretrained_reference.ipynb` | `docs/steps/06_pretrained_reference.md` |
| 7 Ensembles | `notebooks/07_ensembles.ipynb` | `docs/steps/07_ensembles.md` |
| 8 Model comparison | `notebooks/08_model_comparison.ipynb` | `docs/steps/08_model_comparison.md` |
| 9 Ablations | `notebooks/09_ablations.ipynb` | `docs/steps/09_ablations.md` |
| 10 Final checks: seeds, time split, reproduction | `notebooks/10_final_checks.ipynb` | `docs/steps/10_final_checks.md` |

Headline (sarc_random test AUC): LightGBM baseline 0.8074, best compliant single
model 0.8296, interpretable EBM 0.8229, **compliant stacking 0.8420**. The
notebooks recompute every number from saved predictions; training code for
steps 3–7 is in `experiments/` (see `experiments/README.md`).

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
| Team baselines: NB, LR, SVM, SGD, LightGBM, XGBoost, trees, EBM | done |
| Feature engineering, full-data EBM, from-scratch networks | done |
| Ensembles (voting, stacking, MBR) | done |
| Ablations A1, A2, A4–A9; seeds and `sarc_temporal` checks | done |

---

## Quick start

1. Install requirements 

```bash
pip install -r requirements.txt
```

2. Download data files  into local repository (saved in data/raw/primary)

```bash 
python pipeline/00_download_datasets.py    # ~278 MB, no Kaggle token needed
```

2. Run these notebooks in pipeline/notebooks in order.

- `01_clean_primary.ipynb`  - cleaned main reddit sarcasm data , saved clean data in data/processed
- `02_clean_supplementary.ipynb` - cleaned supplementary data, saved cleaned data in data/processed
- `03_eda.py.ipynb` - exploratory analysis of data with figures 
- `04_build_splits.ipynb` - split data into test train validation, saved split data in data/splits
- `05_verify_splits.ipynb` - verify if split data is usable for modelling 


Then follow the step-by-step notebooks in `notebooks/` (table above). They
recompute every number from saved predictions; the training code for steps
3–9 is in `experiments/` (see `experiments/README.md`). The team baseline
notebooks are in `modelling/`.

Experiment dependencies (LightGBM, XGBoost, interpret, PyTorch, tokenizers)
are listed in `requirements-experiments.txt`.

---

## Layout

```
pipeline/            data acquisition, cleaning, EDA, splitting (step 1)
  common/              text_cleaning, paths, report, viz
  00–05_*.py           the pipeline, in order
  scrapers/            provided but not run (need API credentials)
  load.py              dataset loader + the model-safe feature allow-list
  notebooks/           the same pipeline as notebooks

modelling/           team baselines (step 2): probabilistic_linear, Margin-Based,
                     gradient_boosting, tree-based, glassbox, Ensemble
experiments/         training code for steps 3–9 and their small result files
notebooks/           step-by-step notebooks 01–10
docs/                SUMMARY.md, steps/01–10, datasets, data dictionary, split card, reports
figures/             eda/, gbdt/, glassbox/, nb_logreg/, margin-based/, tree-based/, steps/
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

From the step-by-step study ([`docs/SUMMARY.md`](docs/SUMMARY.md)):

- The bottleneck is the representation, not the classifier: LightGBM and LR
  baselines are within 0.005 AUC. Character n-grams and feeding an LR score to
  the trees lift the best compliant single model to 0.8296.
- Context helps only with models that compare reply and parent: +0.003 for
  hand-made contrast features, +0.006 for a from-scratch ESIM, +0.012 for RoBERTa
  (reference only).
- Stacking 24 compliant models reaches 0.8420; the gain comes from members that
  err differently, not from the combiner.
- Results hold on unseen communities and unseen authors. About 0.01 AUC depends
  on how Reddit replies end.
- Cross-platform transfer stays weak: news headlines are near 0.5 for every model.

---

## Traps worth knowing about

1. Split by subreddit or by time, never at random, for any generalisation claim.
   Both stress splits ship ready to use.
2. Never mix the three primary split schemes. They partition the same 913,769
   rows, so 70.1% of `sarc_temporal/test` and `sarc_subreddit/test` sits inside
   `sarc_random/train`. the step notebooks show the correct
   pattern (each split set is trained and tested on its own).
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
lexical heuristic alone. `scikit-learn` is needed from step 04 onward. The
modelling and experiment steps also need `pip install -r requirements-experiments.txt`.

The notebooks read saved model predictions (~500 MB) that are not in git. Put
them at `experiments/predictions/` or point `CS3244_PRED_DIR` at them;
retraining additionally uses feature caches from `CS3244_CACHE_DIR` (default
`experiments/.cache/`).

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
