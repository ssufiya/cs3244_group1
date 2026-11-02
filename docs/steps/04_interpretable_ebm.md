# Step 4 — Interpretable model: full-data EBM

Notebook: [`notebooks/04_interpretable_ebm.ipynb`](../../notebooks/04_interpretable_ebm.ipynb) · Code: `experiments/11_full_convergence/ebm_full.py`, `explain_ebm.py` · Summary: section 5

## Why an EBM

An Explainable Boosting Machine is a generalised additive model with boosted shape functions and a few pairwise interactions. Every prediction is exactly *intercept + one contribution per term*, so no approximate explainer such as SHAP is needed. The team EBM (0.7620) was trained on a 200k sample and used a subreddit encoding; this step retrains it on all data with better features and no subreddit.

## Setup

- All 639,638 `sarc_random` training rows.
- 66 named features: 17 reply surface, 5 parent-context, 25 linguistic cues (step 3), 18 reply/parent contrast (step 3), and `text_ngram_logodds`, the out-of-fold word+char LR score (step 3, GroupKFold by parent).
- 8 outer bags, 15 pairwise interactions, learning rate 0.02, max_bins 256, min_samples_leaf 50, 15% internal early stopping (patience 100, cap 20k rounds). Every bag stopped before the cap. 81 terms in total. Fit time about 4.5 minutes on 24 CPU cores.

## Results

| Model | val | sarc_random test | sarc_subreddit | recall at precision ≥ .75 |
|---|---:|---:|---:|---:|
| Team EBM (340 features) | — | 0.7620 | — | |
| Full EBM without text score | 0.7097 | 0.7123 | — | 0.462 |
| **Full EBM** | 0.8185 | **0.8229** | **0.8145** | 0.738 |
| word + char LR (for reference) | | 0.8184 | 0.8101 | 0.729 |
| best compliant LightGBM (for reference) | | 0.8296 | 0.8226 | 0.751 |

Full EBM vs team EBM: +0.0609 [+0.0591, +0.0627]; vs word+char LR: +0.0045 [+0.0040, +0.0049].

## What it learned

- Term importance (mean |log-odds contribution|): text score 1.15, parent characters 0.24, reply characters 0.11, parent words 0.10, sarcastic opener 0.10, length ratio 0.08. Top interactions: reply length × text score, average word length × text score.
- Strongest sarcastic n-grams: *yes because, yeah because, but but, good thing, how dare, everyone knows, would never* (mock agreement). Strongest non-sarcastic: a reply ending with a full stop or exclamation mark, brackets, *not sure / pretty sure / for sure*.
- Worked examples, including one error, are in `experiments/11_full_convergence/results/ebm_explanations.md` and rendered in the notebook.

## Takeaways

- The full EBM is the **interpretable pick**: 0.8229, only about 0.007 below the best compliant LightGBM, fully additive, and free of subreddit.
- The n-gram text score carries most of the signal; the hand-made features add about +0.0045 and make explanations readable.
- The "ends with a full stop → not sarcastic" signal may be an artefact of how SARC stripped the `/s` marker. Ablation A8 in the summary tests it.
