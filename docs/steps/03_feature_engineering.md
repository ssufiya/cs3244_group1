# Step 3 — Feature-engineering improvements

Notebook: [`notebooks/03_feature_engineering.ipynb`](../../notebooks/03_feature_engineering.ipynb) · Code: `experiments/02_` … `experiments/10_` (shared code in `experiments/features_extra.py` and `experiments/lgbm_runner.py`) · Summary: sections 3 and 8

## Method

Each change is added to the baseline LightGBM with its **frozen** hyper-parameters and early stopping on `sarc_random` validation. Reproducing the baseline itself gives exactly 0.8074 / 0.8023. Every model is refitted on `sarc_subreddit` with the same settings to check that the gain survives unseen communities.

| Folder | Change |
|---|---|
| `02_linguistic_cues` | 25 cue features: hyperbole, mock-agreement openers (*yeah, sure, oh, wow*), understatement, sentiment co-occurrence, question endings and tag questions |
| `03_reply_parent_contrast` | 18 contrast features: sentiment gap, lexical overlap, length ratio |
| `04_char_ngrams_nbsvm` | word + character n-gram LR (800k dimensions) and NB-SVM |
| `05_lsa_semantic_context` | 128-dimensional LSA reply × parent interactions |
| `06_linear_score_as_feature` | out-of-fold word+char LR score (GroupKFold by parent) as a tree feature |
| `08_bigger_vocabulary` | 60k vocabulary instead of 20k |
| `09_sentence_embeddings`, `10_combined_best` | frozen MiniLM embeddings: **pretrained, reference only** |

## Results (sarc_random test)

| Change | Compliant | AUC | Δ vs baseline [95% CI] | sarc_subreddit |
|---|:---:|---:|---|---:|
| Baseline LightGBM | ✓ | 0.8074 | — | 0.8023 |
| + cues | ✓ | 0.8120 | +0.0046 [+0.0039, +0.0052] | 0.8068 |
| + contrast | ✓ | 0.8103 | +0.0029 [+0.0023, +0.0035] | 0.8050 |
| + cues + contrast | ✓ | 0.8143 | +0.0069 [+0.0061, +0.0076] | 0.8084 |
| word + char LR | ✓ | 0.8184 | +0.0110 [+0.0097, +0.0123] | 0.8101 |
| NB-SVM word + char | ✓ | 0.8169 | +0.0095 [+0.0080, +0.0108] | 0.8084 |
| + LSA | ✓ | 0.8012 | −0.0062 [−0.0070, −0.0053] | 0.7951 |
| + LR score | ✓ | 0.8264 | +0.0189 [+0.0180, +0.0199] | 0.8187 |
| **+ LR score + cues + contrast** | ✓ | **0.8296** | **+0.0222 [+0.0209, +0.0233]** | **0.8226** |
| 60k vocab + cues + contrast | ✓ | 0.8161 | +0.0087 [+0.0077, +0.0095] | 0.8099 |
| + MiniLM | ✗ | 0.8270 | +0.0196 | 0.8208 |
| all combined incl. MiniLM | ✗ | 0.8330 | +0.0256 | 0.8251 |

## Takeaways

- **Best compliant single model: `lgbm_lrscore_cues_contrast`, 0.8296.**
- The LR score as a tree feature is the largest compliant gain: trees model non-linear surface cues while the LR covers the long n-gram tail.
- Character n-grams are cheap and strong: elongations, punctuation runs and casing are lost in a 20k-word vocabulary.
- Cue and contrast gains are the same on both splits, so they are genuine sarcasm markers rather than topic proxies (Q2, Q3).
- LSA is a useful negative result: dense, low-information features take 35% of split gain and still lower test AUC.
- MiniLM numbers only price pretraining; they are excluded from the submission.
