# Step 9 — Ablations

Notebook: [`notebooks/09_ablations.ipynb`](../../notebooks/09_ablations.ipynb) · Code: `experiments/13_ablations/`, A7 variants in `experiments/12_scratch_nn/nn_scratch.py` · Plan: `docs/SUMMARY.md` section 11

This step checks which components the results depend on and whether they can be trusted. Every planned ablation is done except A3 (a converged reply-only RoBERTa, which is a reference model only).

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

## A1 — Feature blocks of the best compliant single model

`lgbm_lrscore_cues_contrast` retrained with one block removed (same frozen hyper-parameters; sarc_subreddit reuses the tree count chosen on sarc_random).

| Removed block | Features left | sarc_random test AUC | Δ [95% CI] | sarc_subreddit Δ | Recall at P ≥ .75 |
|---|---:|---:|---|---:|---:|
| none (full model) | 20,061 | 0.8296 | — | — | 0.751 |
| **LR text score** | 20,060 | 0.8143 | **−0.0153 [−0.0163, −0.0143]** | −0.0143 | 0.718 |
| word TF-IDF (20k) | 61 | 0.8240 | −0.0056 [−0.0062, −0.0050] | −0.0067 | 0.739 |
| reply/parent contrast | 20,043 | 0.8276 | −0.0020 [−0.0024, −0.0015] | −0.0021 | 0.746 |
| linguistic cues | 20,036 | 0.8286 | −0.0011 [−0.0015, −0.0007] | −0.0016 | 0.750 |
| surface counts | 20,044 | 0.8286 | −0.0011 [−0.0015, −0.0006] | −0.0019 | 0.749 |

The model depends on the out-of-fold LR score. Without the 20k TF-IDF columns, 61 dense features (LR score + surface + cues + contrast) still reach 0.8240. Every block costs slightly more on unseen communities than on sarc_random, so none of them is a topic shortcut.

## A6 — EBM design

| Variant | Training rows | Terms | Test AUC | Fit time |
|---|---:|---:|---:|---:|
| Full EBM (step 4) | 639,638 | 81 | 0.8229 | ~4.5 min |
| No interactions | 639,638 | 66 | 0.8215 | 2.6 min |
| 200k-row sample | 200,000 | 81 | 0.8225 | 0.5 min |
| No text score (step 4) | 639,638 | 80 | 0.7123 | |

The 15 interactions add +0.0014. With the LR text score as a feature, a 200k sample loses only 0.0004, so the team EBM's gap (0.7620) came from its features, not its sample size. A main-effects-only EBM (0.8215) is the simplest fully readable option at almost no cost.

## A7 — From-scratch network design

| Model | Parameters | Best epoch | sarc_random test AUC | TweetEval |
|---|---:|---:|---:|---:|
| BiLSTM, reply only | 6.75M | 6 | 0.8180 | 0.623 |
| BiLSTM, reply and parent encoded separately (no cross-attention) | 6.90M | 8 | 0.8232 | 0.650 |
| **ESIM (cross-attention)** | 7.79M | 7 | **0.8242** | 0.648 |
| Transformer from scratch, [CLS] reply [SEP] parent (4 layers, d=192) | 7.65M | 6 | 0.8052 | 0.591 |
| RoBERTa converged (pretrained reference) | 125M | | 0.8809 | 0.697 |

- Seeing the parent at all gives +0.0052 [+0.0042, +0.0062]; cross-attention adds only +0.0010 [−0.0001, +0.0020] on top. Most of the compliant context gain comes from encoding the parent, not from word-by-word alignment.
- The same-size Transformer trained from scratch is 0.019 [0.018, 0.020] **below** the ESIM and starts to overfit after epoch 6; RoBERTa is 0.076 above it. RoBERTa's lead over the compliant networks therefore comes from pretraining, not from the Transformer architecture, and the 0.057 gap to the ESIM if anything underestimates what pretraining is worth.
- Caveat: the Transformer was run with one configuration (lr 3e-4, 2,000 warm-up steps, batch 256) and was not tuned.

## A8 — Trailing punctuation (artefact check)

How training replies end:

| Last character | Not sarcastic | Sarcastic |
|---|---:|---:|
| full stop | 52.7% | 40.7% |
| exclamation mark | 4.6% | 13.5% |
| question mark | 12.4% | 11.1% |
| letter or digit | 26.2% | 32.6% |

Word + char LR without surface counts (reference 0.8166), with trailing punctuation and whitespace stripped from every reply (67% of test replies change):

| Train / test text | Test AUC | Δ vs reference [95% CI] |
|---|---:|---|
| original / original | 0.8166 | — |
| original / stripped | 0.8045 | −0.0122 [−0.0130, −0.0114] |
| stripped / stripped | 0.8069 | −0.0097 [−0.0106, −0.0090] |

About 0.01 AUC depends on how a reply ends, and retraining without endings recovers only a quarter of it. The model still reaches 0.807 with no ending information, so endings are a small part of the signal. SARC alone cannot tell whether the difference is an artefact of removing `/s` or a genuine style difference (sarcastic replies are more casual and more often end with `!` or no punctuation). The report should state the ~0.01 dependency and the 0.807 floor.

## Overall reading

- **Credibility:** no author memorisation (A9); about 0.01 AUC rests on reply endings (A8); every component behaves the same on unseen communities.
- **What the compliant results depend on:** the word+char LR score inside the LightGBM (A1), the word/char complementarity (A2), and the from-scratch networks in the stack (A4). The stack design (A5), EBM interactions and sample size (A6) and cross-attention (A7) matter little.
- **Architecture vs pretraining:** a from-scratch Transformer is weaker than the BiLSTM-based ESIM (A7), so the gap to RoBERTa is pretraining.
- **Not run:** A3 (a converged reply-only RoBERTa), which would only refine a reference number.
