# Step 8 — Model comparison, transfer and research questions

Notebook: [`notebooks/08_model_comparison.ipynb`](../../notebooks/08_model_comparison.ipynb) · Output: `results/ensemble/leaderboard_all_models.csv` · Summary: sections 0, 8, 9, 10, 11

## One leaderboard

The notebook recomputes ROC-AUC, PR-AUC, accuracy, precision, recall, F1 and recall at the validation-chosen precision ≥ 0.75 threshold for every saved model on all six evaluation sets, with the same code path. It replaces the older `results_master_r00_10.csv`, which predates steps 4–6.

## Where the gains come from (sarc_random test AUC)

| Stage | AUC | Increment | Main source |
|---|---:|---:|---|
| Baseline LightGBM | 0.8074 | | |
| Best compliant single model | 0.8296 | +0.022 | character n-grams, LR score as tree feature, cues |
| **Compliant stacking (submitted)** | **0.8420** | +0.012 | complementary families, mainly the from-scratch ESIM |
| RoBERTa single (reference) | 0.8809 | +0.039 | large-scale pretraining |
| Stacking with RoBERTa (reference) | 0.8861 | +0.005 | |

## Transfer (trained on sarc_random, no target-side fitting)

| Model | figlang_reddit | figlang_twitter | tweeteval_irony | news_headlines |
|---|---:|---:|---:|---:|
| Baseline LightGBM | 0.7450 | 0.6106 | 0.5986 | 0.4475 |
| word + char LR | 0.7729 | 0.6224 | 0.5956 | 0.4437 |
| Best compliant LightGBM | 0.7700 | 0.6225 | 0.6140 | 0.4508 |
| Full EBM | 0.7706 | 0.6366 | 0.5855 | 0.4342 |
| From-scratch ESIM | 0.7589 | 0.6377 | 0.6484 | 0.4769 |
| RoBERTa converged (reference) | 0.8276 | 0.7479 | 0.6973 | 0.5057 |

Reddit-to-Reddit transfer holds up; Twitter is much harder for compliant models; news headlines are near chance for every model, because headlines have neither conversational context nor Reddit's mock-agreement patterns.

## Research questions

| Question | Evidence | Answer |
|---|---|---|
| Q1 Is the parent comment needed? | contrast features +0.003; from-scratch ESIM +0.006; RoBERTa +0.012; parent bag-of-words ≈ 0; LSA −0.006 | **Yes**, but only models that compare reply and parent token by token can use it |
| Q2 Do linguistic cues help? | cues +0.0045 on both splits; character n-grams +0.011 | Yes, robustly |
| Q3 Is subreddit needed? | gains identical on unseen communities; dropping the subreddit member costs ≈ 0.0006 | No |

## Limitations and next steps

One seed for compliant models; 84% author overlap between train and test; a possible trailing-punctuation artefact; VADER's status under the course rule needs confirming. The ablation plan (A1–A9) is in `docs/SUMMARY.md` section 11, starting with the cheap stacking ablations and the author-disjoint check.
