# Step 9 — Ablations

Notebook: [`notebooks/09_ablations.ipynb`](../../notebooks/09_ablations.ipynb) · Code: `experiments/13_ablations/` · Plan: `docs/SUMMARY.md` section 11

This step checks which components the results depend on and whether they can be trusted. The first four ablations (A9, A4, A5, A2) are done; A1, A6, A8 and A7 need more retraining and are still open.

## A9 — Author overlap (credibility check)

87% of `sarc_random` test rows (119,051 of 137,065) were written by authors who also appear in training. If models were partly recognising authors, AUC would drop on the remaining 18,014 rows.

| Model | All test | Seen authors | Unseen authors |
|---|---:|---:|---:|
| Baseline LightGBM | 0.8074 | 0.8071 | 0.8108 |
| Best compliant LightGBM | 0.8296 | 0.8296 | 0.8307 |
| word + char LR | 0.8184 | 0.8184 | 0.8197 |
| Full EBM | 0.8229 | 0.8230 | 0.8234 |
| From-scratch ESIM | 0.8242 | 0.8243 | 0.8249 |
| **Compliant LR stack** | **0.8416** | 0.8416 | **0.8428** |
| RoBERTa converged (reference) | 0.8809 | 0.8804 | 0.8849 |

No model gets worse on unseen authors; every model is slightly better. On unseen authors alone the compliant stack is still +0.0320 [+0.0287, +0.0357] above the baseline. **The results are not driven by author memorisation.**

## A4 — Stacking members

L2 logistic-regression stack over calibrated logits (the unit used in step 7; every combiner there was within 0.001 of it), with one model family removed at a time.

| Removed family | Members left | sarc_random Δ test AUC [95% CI] | sarc_subreddit Δ |
|---|---:|---|---:|
| none (all 24 compliant members) | 24 | 0.8416 | 0.8335 |
| **From-scratch networks** | 22 | **−0.0070 [−0.0076, −0.0064]** | −0.0070 |
| Improved GBDTs (step 3) | 16 | −0.0028 [−0.0032, −0.0025] | −0.0027 |
| Naive Bayes | 22 | −0.0007 [−0.0009, −0.0005] | −0.0005 |
| Linear | 19 | −0.0006 [−0.0007, −0.0005] | −0.0003 |
| Glass-box (EBMs) | 21 | −0.0005 [−0.0006, −0.0003] | −0.0000 |
| Baseline GBDTs | 20 | −0.0001 [−0.0002, −0.0000] | −0.0001 |

Using one family alone: improved GBDTs 0.8314, from-scratch networks 0.8289, glass-box 0.8249, linear 0.8226, baseline GBDTs 0.8090, Naive Bayes 0.7892.

Reading: the stack's gain over the best single model comes from two places — the from-scratch networks (different errors) and the improved GBDTs (strongest single models). The other families each add less than 0.001, and the losses are identical on unseen communities.

## A5 — Stacking design (sarc_random)

| Variant | OOF val AUC | Test AUC | Δ vs reference |
|---|---:|---:|---:|
| Reference: calibrated logits, LR, cross-fitted | 0.8374 | 0.8416 | — |
| No calibration (raw scores, standardised) | 0.8374 | 0.8416 | −0.0000 |
| Probabilities instead of logits | 0.8370 | 0.8411 | −0.0005 |
| LightGBM meta-learner, no meta-features | 0.8371 | 0.8414 | −0.0002 |
| LightGBM meta-learner + meta-features | 0.8373 | 0.8417 | +0.0001 |
| No cross-fitting (in-sample val AUC) | 0.8375 | 0.8416 | 0.0000 |

Every design choice changes test AUC by at most 0.0005. Working in logit space is the only choice with a measurable (small) effect. The in-sample validation AUC is only 0.0001 above the cross-fitted one, so with 137k validation rows the cross-fitting matters little for 24 members. **The stack is robust to its design; what matters is which members it has (A4).**

## A2 — Text representation of the linear model

Same vectorisers and C as `experiments/04_char_ngrams_nbsvm`, retrained on `sarc_random` with one block removed.

| Variant | Features | Test AUC | Δ vs full [95% CI] | Recall at P ≥ .75 |
|---|---:|---:|---|---:|
| word + char + surface (full) | 539,563 | 0.8184 | — | 0.729 |
| word + char | 539,546 | 0.8166 | −0.0018 [−0.0021, −0.0015] | 0.725 |
| word + surface | 299,309 | 0.8008 | −0.0176 [−0.0187, −0.0167] | 0.690 |
| char + surface | 240,271 | 0.7981 | −0.0204 [−0.0213, −0.0193] | 0.680 |

Word and character n-grams are complementary: each alone is about 0.80, together +0.016 to +0.020 more. Character n-grams alone are slightly weaker than words alone, so they add form cues (elongation, punctuation, casing) rather than replacing words. Surface counts add +0.0018 on top.

## Still open

| ID | Ablation | Cost |
|---|---|---|
| A1 | leave one feature block out of the best compliant LightGBM | 4 LightGBM fits, ~1 h CPU |
| A6 | EBM without interactions, and on a 200k sample | 2–3 EBM fits, ~5 min each |
| A8 | strip trailing punctuation and re-evaluate (artefact check) | features + refits, ~1 h CPU |
| A7 | ESIM without cross-attention; same-size from-scratch Transformer | GPU, 20–40 min each |
