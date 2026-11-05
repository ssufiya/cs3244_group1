# Step 10 — Final robustness checks

Notebook: [`notebooks/10_final_checks.ipynb`](../../notebooks/10_final_checks.ipynb) · Code: `experiments/14_final_checks/` (queues in `run_queue.py`), seed and tuning options in `experiments/12_scratch_nn/nn_scratch.py` and `experiments/11_full_convergence/ebm_full.py`

## 10.1 Reproduction

Ablation and robustness predictions are now kept out of ensembles and leaderboards by default (`common.is_extra_group`). Rerunning `ensemble_v3.py` from scratch with the pretrained members dropped reproduces the saved results to within 3e-5 AUC: **0.8420 on sarc_random and 0.8340 on sarc_subreddit**. Notebooks 07 and 08 were rerun and give the same numbers and leaderboard as before.

## 10.2 Random seeds (sarc_random test AUC)

| Model | Seed 3244 | Seed 3245 | Seed 3246 | Mean ± std |
|---|---:|---:|---:|---:|
| Best compliant LightGBM | 0.8296 | 0.8302 | 0.8300 | 0.8299 ± 0.0003 |
| Full EBM | 0.8229 | 0.8229 | 0.8229 | 0.8229 ± 0.0000 |
| From-scratch ESIM | 0.8242 | 0.8235 | 0.8235 | 0.8237 ± 0.0004 |
| From-scratch BiLSTM (reply) | 0.8180 | 0.8173 | 0.8187 | 0.8180 ± 0.0007 |
| **Compliant LR stack** (24 members, seeded members swapped) | 0.8416 | 0.8420 | 0.8414 | **0.8417 ± 0.0003** |

Seed variation is at most 0.0007, an order of magnitude below every gain reported in steps 3–9. The reported single-seed numbers are representative.

## 10.3 Generalising across time (`sarc_temporal`)

Each model is refitted on the time split's training years with settings frozen from sarc_random (LightGBM tree count, network epochs, LR C); test comments are later in time.

| Model | Test AUC | Δ vs LightGBM baseline [95% CI] |
|---|---:|---|
| Baseline LightGBM | 0.7918 | — |
| **Best compliant LightGBM** | **0.8126** | +0.0209 [+0.0199, +0.0219] |
| Full EBM | 0.8054 | +0.0136 [+0.0125, +0.0147] |
| From-scratch ESIM | 0.8050 | +0.0133 [+0.0119, +0.0146] |
| word + char LR | 0.8011 | +0.0093 [+0.0081, +0.0104] |
| From-scratch BiLSTM (reply) | 0.8010 | +0.0092 [+0.0079, +0.0106] |
| **Stack (13 compliant members, final framework)** | **0.8240** | **+0.032** |

All scores are lower than on sarc_random (later years are harder), but every improvement keeps its sign and roughly its size, and the ensemble adds +0.0114 [+0.0108, +0.0119] over the best single model, as on the other splits. Only 13 members have time-split predictions (team baselines plus the five models above), so this stack is smaller than the submitted one.

## 10.4 Tuning the from-scratch Transformer (A7 follow-up)

| Run | Layers × dim | Dropout | lr | Batch | Test AUC |
|---|---|---:|---:|---:|---:|
| A7 original | 4 × 192 | 0.1 | 3e-4 | 256 | 0.8052 |
| cfg2 | 4 × 192 | 0.1 | 1e-3 | 512 | 0.8056 |
| cfg3 | 2 × 256 | 0.1 | 3e-4 | 256 | 0.8053 |
| **cfg4** | 4 × 192 | 0.2 | 3e-4 | 256 | **0.8067** |
| ESIM (reference) | | | | | 0.8242 |

The best configuration gains only +0.0015 and stays 0.018 below the ESIM. The conclusion of A7 stands: without pretraining a Transformer is weaker than the BiLSTM-based ESIM here, so RoBERTa's lead is pretraining.

## 10.5 How replies end on other datasets (A8 follow-up)

| Dataset (label source) | Ends with "." not sarcastic / sarcastic | Ends with "!" not sarcastic / sarcastic | AUC of "no full stop" |
|---|---:|---:|---:|
| SARC test (stripped `/s`) | 52.5% / 40.7% | 4.6% / 13.5% | 0.559 |
| FigLang Reddit (Reddit) | 56.2% / 36.9% | 4.1% / 14.3% | 0.596 |
| FigLang Twitter | 22.3% / 36.8% | 5.9% / 6.1% | 0.428 |
| TweetEval irony (human-annotated) | 22.7% / 21.7% | 5.5% / 7.3% | 0.505 |
| News headlines | 1.6% / 0.3% | 0.7% / 0.1% | 0.506 |

The pattern exists on both Reddit datasets but not on human-annotated TweetEval, and it reverses on FigLang Twitter. It is a Reddit-specific signal, consistent with a labelling or platform artefact rather than a general property of sarcasm. Together with A8 (about 0.01 AUC depends on it) the report should flag it as a likely artefact.

## 10.6 The team ensemble notebook

`modelling/Ensemble/stacking&voting.ipynb` combines LightGBM, XGBoost, LR and linear SVM, but computes AUC from 0/1 predictions, which caps it (0.71–0.74). With the same four members and the step-7 protocol, AUC on scores:

| Method | Test AUC |
|---|---:|
| Best member (LightGBM) | 0.8074 |
| Hard vote (vote share) | 0.7702 |
| Soft vote | 0.8137 |
| **LR stack** | **0.8142** (+0.0068 [+0.0062, +0.0073]) |
| LR stack scored on 0/1 labels | 0.7374 |

The team's ensemble is a valid but smaller version of step 7: four correlated members add +0.007, while the 24-member compliant stack adds +0.035. The report should quote AUCs computed on scores.

## Remaining caveats for the report

- VADER (a hand-written lexicon with rules) is used in the cue and contrast features; its status under the no-pretrained-models rule needs the course's confirmation. Removing both feature groups costs about 0.003 (A1).
- A3 (a converged reply-only RoBERTa) was not run; it would only refine a reference number.
