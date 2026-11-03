# experiments/ — training code for steps 3–7

These scripts produced every model after the team baselines. The step notebooks in `notebooks/` do not rerun them; they read the predictions the scripts saved. Rerun a script only to reproduce or extend a result.

## Layout

| Path | Step | What it does |
|---|---|---|
| `common.py` | all | paths, prediction I/O keyed by `row_id`, metrics, validation threshold for precision ≥ 0.75, paired bootstrap |
| `features_extra.py` | 3–5 | linguistic cue block (25 features) and reply/parent contrast block (18 features) |
| `lgbm_runner.py` | 3 | reproduces the baseline LightGBM (frozen hyper-parameters from `results/gbdt/selected.json`) and appends feature blocks |
| `collect_results.py` | 8 | older leaderboard script (`results/ensemble/results_master_r00_10.csv`); superseded by notebook 08 |
| `00_extra_baseline_predictions/` | 2 | re-scores the team SVM and EBM so they have saved predictions |
| `01_ensemble_voting_mbr/` | 7 | first ensemble round: voting, Caruana selection, stacking, MBR decision rules |
| `02_linguistic_cues/` | 3 | LightGBM + cue features |
| `03_reply_parent_contrast/` | 3 | LightGBM + contrast features (and cues + contrast) |
| `04_char_ngrams_nbsvm/` | 3 | word + character n-gram LR and NB-SVM |
| `05_lsa_semantic_context/` | 3 | LSA reply × parent interactions (negative result) |
| `06_linear_score_as_feature/` | 3 | out-of-fold LR score as a LightGBM feature |
| `07_transformer_finetune/` | 6 | RoBERTa fine-tuning, 1 epoch — **pretrained, reference only** |
| `08_bigger_vocabulary/` | 3 | 60k vocabulary |
| `09_sentence_embeddings/` | 3 | frozen MiniLM embeddings — **pretrained, reference only** |
| `10_combined_best/` | 3 | all features combined incl. MiniLM — **reference only** |
| `11_full_convergence/` | 4, 6, 7 | full-data EBM (`ebm_full.py`, `explain_ebm.py`), RoBERTa to convergence (`roberta_converge.py`, reference only), final stacking framework (`ensemble_v3.py`) |
| `12_scratch_nn/` | 5 | from-scratch BiLSTM and ESIM (`nn_scratch.py`) |
| `13_ablations/` | 9 | `stacking_and_author.py` (A4, A5, A9, saved predictions only), `text_representation.py` (A2) |

Each folder's `results/` holds small metric files (JSON/CSV) that the notebooks read.

## Data that is not in git

| What | Size | Where the code looks |
|---|---|---|
| Model predictions (`<group>/<dataset>__<partition>.parquet`) | ~500 MB | `$CS3244_PRED_DIR`, else `experiments/predictions/`, else `../Project Ensemble/predictions/` |
| Feature caches, checkpoints (`.cache/`) | ~6 GB | `experiments/.cache/` |
| Baseline splits | see `data/README.md` | `data/splits/` |

## Running

Use the project conda environment (Python 3.12, PyTorch with CUDA for steps 6 and 5). Examples:

```bash
python experiments/02_linguistic_cues/run.py
python experiments/06_linear_score_as_feature/run.py
python experiments/11_full_convergence/ebm_full.py --scheme sarc_random
python experiments/12_scratch_nn/nn_scratch.py --variant pair --scheme sarc_random
# submitted compliant ensemble: drop every pretrained member
python experiments/11_full_convergence/ensemble_v3.py --out ensemble_compliant_with_scratch --drop roberta_pair roberta_reply roberta_pair_conv roberta_pair_conv_s4248 lgbm_minilm_v60k_cues_contrast lgbm_combined_best
```

GPU jobs save resumable state, so an interrupted run continues from its last checkpoint when relaunched with the same command. Check each script's `--help` for the exact options.

## Protocol (same for every script)

Shared splits only; hyper-parameters, weights and thresholds on validation; test scored once; `sarc_subreddit` refitted on its own training set with settings frozen from `sarc_random`; transfer sets scored by `sarc_random` models with no target-side fitting.
