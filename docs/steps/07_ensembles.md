# Step 7 — Ensembles: voting, stacking and MBR decisions

Notebook: [`notebooks/07_ensembles.ipynb`](../../notebooks/07_ensembles.ipynb) · Code: `experiments/01_ensemble_voting_mbr/ensemble.py`, `experiments/11_full_convergence/ensemble_v3.py` · Summary: section 6

## Framework

1. **Level 0.** Every base model scores validation and test with the same `row_id`s.
2. **Calibration.** Each member is Platt-calibrated on validation.
3. **Level 1, trained on validation only** with 5-fold cross-fitting (so the validation AUC used to choose a combiner is out-of-fold):
   - A: logistic regression on calibrated logits (a log-linear pool that can give redundant members negative weight);
   - B: LightGBM on logits + 5 meta-features (reply words, parent words, empty parent, question marks, exclamation marks), which can learn case-dependent gating.
   - Final output: the average of A and B, refitted on all of validation and applied to test once.
4. **Decision layer (optional).** MBR rules choose the operating point from the posterior without test labels: maximise expected F1, or maximise coverage subject to expected precision ≥ 0.75.

The team's first version is `modelling/Ensemble/stacking&voting.ipynb`; this step extends it with more members and cross-fitted meta-learners.

Also compared: hard and soft voting, a weighted linear pool, Caruana greedy selection, L1 stacking, and Shapley-value pruning.

## Results (test AUC)

| Setting | Members | Best single | Soft vote | Greedy | Stacking (L2) | **Final** |
|---|---:|---:|---:|---:|---:|---:|
| **Compliant + from-scratch networks (submitted), sarc_random** | 24 | 0.8296 | 0.8288 | 0.8384 | 0.8416 | **0.8420** |
| same, without the subreddit-using member | 23 | 0.8296 | | | 0.8412 | 0.8414 |
| compliant, no from-scratch networks | 22 | 0.8296 | 0.8252 | 0.8323 | 0.8346 | 0.8350 |
| **Submitted setting on sarc_subreddit** | 22 | 0.8226 | 0.8228 | 0.8326 | 0.8335 | **0.8340** |
| With RoBERTa (reference only) | 28 | 0.8809 | 0.8476 | 0.8838 | 0.8859 | 0.8861 |

Submitted model at 0.5: precision 0.785, recall 0.733, F1 0.758; recall 0.779 at the validation threshold for precision ≥ 0.75. MBR expected-F1 raises F1 to 0.772 (P 0.722 / R 0.830).

Δ against the baseline LightGBM: **+0.0346 [+0.0335, +0.0358]**. Δ from adding the from-scratch networks: +0.0070 [+0.0065, +0.0077].

The notebook re-runs the logistic-regression stacking live on the compliant members and checks it against the saved 0.8416.

## Takeaways

- Soft voting does not beat the best single model; learned weights (including negative ones) are required.
- All learned combiners land within 0.001 of each other: the combiner is saturated. Gains come from members that **err differently** (the from-scratch ESIM has the largest Shapley value, +0.0062; every other member is ≤ +0.0007).
- Removing the only subreddit-using member costs about 0.0006, so the result can be stated as subreddit-free.
- Greedy selection keeps just two members (ESIM and the best LightGBM, 50/50), which is a compact alternative if a simpler submission is preferred.
