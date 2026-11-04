# Sarcasm detection: baselines, improvements, ensembles and controls — summary report

CS3244 Group 1. This is the one-document summary of the whole project. Each section points to the step notebook that recomputes its numbers (`notebooks/0X_*.ipynb`) and the step document with more detail (`docs/steps/0X_*.md`).

Environment for all experiments: Python 3.12 conda environment, PyTorch 2.11 + CUDA 12.8, one RTX 5070 Ti Laptop GPU (12 GB), 24-core CPU.

---

## 0. Results on one page

1. **Best compliant result (our submission): stacking of 24 compliant models, sarc_random test AUC 0.8420**, +0.0346 [+0.0335, +0.0358] over the LightGBM baseline (0.8074). On `sarc_subreddit`, where train and test communities do not overlap, it reaches 0.8340 (+0.0317). No pretrained model is used anywhere.
2. **Best compliant single model: LightGBM on word TF-IDF + surface features + linguistic cues + reply/parent contrast + a word+char LR score, 0.8296** (+0.0222). Correction to an earlier draft: the "combined LightGBM, 0.8330" used MiniLM sentence embeddings, which are pretrained, so it is not compliant.
3. **Interpretable single model: full-data EBM, 0.8229** (team EBM 0.7620, +0.0609). 66 named features; every prediction splits exactly into per-term contributions, and the text term splits further into n-gram weights. Training takes ~4.5 CPU minutes.
4. **From-scratch neural network: ESIM (reply↔parent cross-attention, no pretraining), 0.8242.** Adding the parent gives +0.0063 over reply-only, so context is usable under the course rules. It is weaker than LightGBM alone but errs differently, and adding it lifts compliant stacking from 0.8350 to 0.8420 (+0.0070).
5. **Pretrained reference: RoBERTa (reply + parent) trained to convergence, 0.8809**, with a second seed at 0.8808. It is ~0.039 above the best compliant result and 0.057 above the from-scratch network that can also compare reply and parent — roughly the value of large-scale pretraining. Stacking with RoBERTa reaches 0.8861, reported as reference only.
6. **The stacking combiner is saturated.** L2 / L1 / Shapley pruning / gated LightGBM meta-learners differ by less than 0.001; gains come from adding members that make different mistakes, not from a more complex combiner.
7. **Research questions.** Context helps, but only with models that compare reply and parent token by token (RoBERTa +0.012, from-scratch ESIM +0.006, hand-made features +0.003). Linguistic cues help robustly. Subreddit is not needed.
8. **Ablations (section 11, step 9):** no model loses AUC on authors unseen in training; about 0.01 AUC depends on how replies end (0.807 without endings); the best LightGBM depends mainly on the LR text score (−0.015 without it); the stack depends on its members, not its design; cross-attention adds only +0.001 over encoding the parent separately; a same-size Transformer from scratch is 0.019 below the ESIM, so the gap to RoBERTa is pretraining.

---

## 1. Task, data and protocol (step 1)

- Task: classify whether a Reddit reply is sarcastic; the parent comment it replies to may be used as context.
- Primary data: SARC, balanced Kaggle release, cleaned from 1,010,826 to 913,769 rows (`docs/cleaning_report.md`). Shared split `sarc_random`: 639,638 train / 137,066 validation / 137,065 test.
- Stress split `sarc_subreddit` (639,838 / 135,123 / 135,534): train and test come from **disjoint subreddits**, to check that models do not just memorise communities.
- Transfer sets, scored by `sarc_random` models with no target-side fitting: figlang_reddit (1,756), figlang_twitter (1,715), tweeteval_irony (780), news_headlines (4,257, The Onion-style headlines).
- Protocol: all hyper-parameters, early stopping, thresholds and ensemble weights are chosen on **validation only**; test is scored once. `sarc_subreddit` models are refitted on their own training set with settings frozen from `sarc_random`.
- Metrics: ROC-AUC (headline, threshold-free); precision / recall / F1 at 0.5; and the proposal's "precision-first" operating point: the validation threshold that keeps precision ≥ 0.75, frozen and applied to test, reporting test recall.
- Differences between two models are given as **paired bootstrap** 95% intervals on the same test rows (200–500 resamples), written Δ [low, high].

EDA highlights (`docs/eda_report.md`): no label leakage; 2.9% of texts appear under both labels; length is not a shortcut (median 9 words in both classes); exclamation marks (2.8×) and interjections (2.0×) are the only strong surface markers; sarcastic vocabulary is mock agreement (*yeah, because, obviously, totally*); subreddit predicts the `/s` convention rather than sarcasm.

### 1.1 Course compliance (important)

The course does not allow pretrained models, so models are split into two groups:

| Group | Contains | Use |
|---|---|---|
| **Compliant** | all bag-of-words / character n-gram models, tree models, EBM, networks trained from scratch | our submitted results |
| **Pretrained reference** | RoBERTa (fine-tuned); MiniLM sentence embeddings (frozen) | upper-bound reference only, to price "what pretraining is worth" |

The sentiment scores inside the cue and contrast features come from VADER, a hand-written lexicon with rules rather than a trained model. It is generally not considered a pretrained model, but this should be confirmed with the course; if not allowed, it can be replaced with a lexicon learned from the training set. Those two feature groups together contribute only about +0.007.

---

## 2. Team baselines (step 2)

| Model | sarc_random test AUC | sarc_subreddit | Note |
|---|---:|---:|---|
| LightGBM (reply TF-IDF 20k + 17 surface features) | **0.8074** | 0.8023 | best baseline; every later Δ is against it |
| LR (word TF-IDF + surface) | 0.8029 | 0.7963 | |
| Linear SVM (TF-IDF + lexicon + surface) | 0.8026 | | |
| XGBoost (reply TF-IDF + surface) | 0.8061 | | |
| SGD (hinge, TF-IDF + lexicon + surface) | 0.7860 | | |
| Multinomial NB (word TF-IDF) | 0.7798 | | surface features hurt NB |
| Histogram gradient boosting / random forest / decision tree | 0.7759 / 0.7603 / 0.6894 | | tree-based notebook |
| EBM (200k sample, 340 features incl. subreddit encoding) | 0.7620 | — | glass-box model |

Detailed reports: `docs/nb_logreg_report.md`, `docs/gbdt_report.md`, `docs/glassbox_report_en.md`.

---

## 3. Feature-engineering improvements (step 3)

All LightGBM variants reuse the baseline's frozen hyper-parameters and early-stop on `sarc_random` validation; the reproduction gives exactly the original 0.8074 / 0.8023.

| # | Change | Compliant | sarc_random AUC | ΔAUC [95% CI] | sarc_subreddit | Recall at precision ≥ .75 |
|---|---|:---:|---:|---|---:|---:|
| — | Baseline LightGBM | ✓ | 0.8074 | — | 0.8023 | 0.704 |
| 1 | + linguistic cues (25: hyperbole, mock-agreement openers, understatement, sentiment co-occurrence, …) | ✓ | 0.8120 | +0.0046 [+0.0039, +0.0052] | 0.8068 | 0.714 |
| 2 | + reply/parent contrast (18) | ✓ | 0.8103 | +0.0029 [+0.0023, +0.0035] | 0.8050 | 0.707 |
| 1+2 | + cues + contrast | ✓ | 0.8143 | +0.0069 [+0.0061, +0.0076] | 0.8084 | 0.718 |
| 3a | word + char n-gram LR (800k dimensions) | ✓ | 0.8184 | +0.0110 [+0.0097, +0.0123] | 0.8101 | 0.729 |
| 3b | NB-SVM (word + char) | ✓ | 0.8169 | +0.0095 [+0.0080, +0.0108] | 0.8084 | 0.723 |
| 4 | + LSA semantic interactions | ✓ | 0.8012 | **−0.0062** [−0.0070, −0.0053] | 0.7951 | 0.689 |
| 5 | + LR score as a tree feature | ✓ | 0.8264 | +0.0189 [+0.0180, +0.0199] | 0.8187 | 0.747 |
| **5+1+2** | **+ LR score + cues + contrast (best compliant single)** | ✓ | **0.8296** | **+0.0222** [+0.0209, +0.0233] | **0.8226** | **0.751** |
| 7 | 60k vocabulary + cues + contrast | ✓ | 0.8161 | +0.0087 [+0.0077, +0.0095] | 0.8099 | 0.722 |
| 8 | 7 + MiniLM sentence embeddings | ✗ | 0.8270 | +0.0196 | 0.8208 | 0.749 |
| 9 | all useful features combined (incl. MiniLM) | ✗ | 0.8330 | +0.0256 | 0.8251 | 0.759 |

Key points:

- **Linguistic cues (Q2)** give the same gain on both splits (+0.0045), so they are cross-community sarcasm markers rather than topic signals. Each single cue has a univariate AUC of only 0.50–0.58; the strongest is the "sarcastic opener" (*yeah / sure / oh / wow…*, 0.578).
- **Reply/parent contrast (Q1)**: +0.003, slightly better than parent bag-of-words (+0.0017), but hand-made sentiment contrast cannot capture real semantic conflict.
- **Character n-grams** are one of the cheapest and strongest changes. A single LR beats every original baseline: elongated words, punctuation combinations and casing are important form cues that a 20k-word bag of words throws away.
- **LSA (negative result)**: 128-dimensional LSA product features take 35% of split gain yet lower test AUC — the trees overfit dense low-information dimensions. Adding "semantic" features does not help by itself; representation quality is what matters.
- **Linear score as a tree feature**: the largest compliant single change (+0.019). LightGBM handles non-linear surface cues, the LR covers the long tail of 800k n-grams. The training-set score is out-of-fold with GroupKFold by parent, so sibling replies under the same parent cannot leak.
- **MiniLM (pretrained, not compliant)**: +0.011; compared with LSA it shows how much better pretrained semantics are than an in-corpus SVD.

---

## 4. Pretrained reference: RoBERTa (step 6)

### 4.1 Input

- Tokenisation: byte-level BPE (vocabulary 50,265); long or elongated words (*sooooo*), emoji and punctuation runs become sub-words instead of unknown tokens — the same problem character n-grams address.
- Pair input: `<s> reply </s></s> parent </s>`, at most 128 tokens, truncating the longer side first, so short replies are usually kept whole. Reply-only control: `<s> reply </s>`, at most 64 tokens.

### 4.2 Model

- 12-layer Transformer encoder (hidden size 768, 12 heads, ~125M parameters); in every layer each token attends to every other token.
- **This is why the parent helps**: with pair input, words in the reply attend directly to words in the parent. If the reply says *Great job* and the parent describes something bad, the model sees "positive wording" and "negative situation" in one representation. Bag-of-words models turn reply and parent into separate count vectors and cannot compare them word by word.
- Classification head: the final `<s>` vector → dense + tanh → linear to 2 classes.

### 4.3 Pretraining and fine-tuning

- **Pretraining (not ours, and the reason it is not compliant)**: RoBERTa-base was pretrained with masked language modelling on ~160 GB of English text, including OpenWebText (pages linked from upvoted Reddit posts) and CC-News.
- **Fine-tuning (ours)**: all parameters, cross-entropy, AdamW (lr 2e-5, weight decay 0.01), 6% linear warm-up then linear decay, batch 64 (4 accumulated micro-batches), bf16, gradient clipping 1.0.

### 4.4 Training to convergence

The schedule allows up to 3 epochs; AUC is computed on the full validation set every half epoch, the best checkpoint is kept, and training stops after 2 evaluations without improvement.

| Epochs | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---:|---:|---:|---:|---:|---:|
| Validation AUC | 0.8553 | 0.8670 | 0.8737 | 0.8771 | 0.8773 | **0.8784** |

After 2 epochs the gain is only 0.0013 and the learning rate decays to zero at 3 epochs, so the run is treated as converged (2.6 hours).

| | val | test | precision | recall | F1 | recall at P ≥ .75 | figlang_reddit | figlang_twitter | tweeteval | news |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa reply only, 1 epoch | | 0.8599 | | | | 0.818 | 0.8192 | 0.6770 | 0.6822 | 0.4709 |
| RoBERTa reply + parent, 1 epoch | 0.8691 | 0.8718 | 0.785 | 0.801 | 0.793 | 0.847 | 0.8246 | 0.7226 | 0.6965 | 0.4689 |
| **RoBERTa reply + parent, converged** | **0.8784** | **0.8809** | 0.788 | 0.822 | **0.804** | **0.864** | **0.8276** | **0.7479** | **0.6973** | **0.5057** |

- Converged vs 1 epoch: +0.0090 [+0.0082, +0.0098]; sarc_subreddit 0.8762 (+0.0092).
- Second seed (4248): validation 0.8782, test 0.8808. The two seeds' logits correlate at 0.97; their average gives 0.8837.

### 4.5 What the reference tells us

| Comparison | Result | Meaning |
|---|---|---|
| Pretrained vs compliant | 0.8809 vs 0.8420 (compliant stacking) / 0.8296 (compliant single) / 0.8242 (from-scratch ESIM) | 0.057 above a from-scratch network that can also compare reply and parent; more than twice all compliant feature engineering (+0.022) |
| Context (Q1), RoBERTa | reply + parent vs reply only (1 epoch): +0.0119 | context does help |
| Context (Q1), compliant hand-made | parent bag-of-words ≈ 0; contrast +0.003; LSA −0.006 | hand-made compliant features barely use context |
| Transfer | RoBERTa: Twitter 0.75, TweetEval 0.70; compliant ≈ 0.62–0.65 | pretraining helps transfer most; news headlines stay ≈ 0.5 for all |

RoBERTa's advantage has two parts: an **architecture** that compares reply and parent token by token, and **pretraining**. Section 7 separates them.

---

## 5. Interpretable single model: full-data EBM (step 4)

### 5.1 Why this model

| Candidate | sarc_random AUC | How one prediction is explained | Training cost | Compliant |
|---|---:|---|---|:---:|
| Best compliant LightGBM | 0.8296 | needs approximations such as SHAP; 20k-dimensional bag of words | medium | ✓ |
| word + char n-gram LR | 0.8184 | directly: sum of n-gram weights | low | ✓ |
| Team EBM | 0.7620 | directly | low | ✓ (but uses subreddit) |
| **Full-data EBM** | **0.8229** | directly, two additive levels: EBM terms, and the "text score" term split into n-gram weights | low: ~4.5 min (24-core CPU), text score reused from cache | ✓ |

The full EBM is +0.0045 [+0.0040, +0.0049] above the LR and ~0.007 below the best compliant LightGBM, but every prediction is exactly "intercept + per-term contributions", with no approximate explainer and no subreddit (consistent with Q3).

### 5.2 Setup

- Training data: all 639,638 `sarc_random` training rows, no sampling.
- 66 named features: 17 reply surface, 5 parent-context, 25 linguistic cues, 18 reply/parent contrast, 1 `text_ngram_logodds` (out-of-fold score of the word+char LR).
- EBM: 8 outer bags, 15 pairwise interactions, learning rate 0.02, max_bins 256, min_samples_leaf 50, internal 15% early stopping with patience 100 and a 20k-round cap. All bags stopped before the cap (converged). 81 terms (66 main effects + 15 interactions).

### 5.3 Results

| | val | sarc_random test | precision | recall | F1 | recall at P ≥ .75 | sarc_subreddit | figlang_reddit | figlang_twitter | tweeteval | news |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Team EBM | — | 0.7620 | | | | | — | 0.6780 | 0.6137 | | 0.4688 |
| Full EBM, no text score | 0.7097 | 0.7123 | 0.702 | 0.556 | 0.621 | 0.462 | — | 0.6397 | 0.5714 | 0.5018 | 0.4904 |
| **Full EBM** | 0.8185 | **0.8229** | 0.771 | 0.713 | 0.741 | 0.738 | **0.8145** | 0.7706 | 0.6366 | 0.5855 | 0.4342 |

- Full EBM vs team EBM: +0.0609 [+0.0591, +0.0627].
- Without the text score only 0.7123 remains: most signal comes from the n-gram text score; hand-made features add ~+0.0045 on top and put readable terms (openers, length, context) into the explanation.

### 5.4 What it learned

- **Most important terms** (mean |log-odds contribution|): text score 1.15, parent characters 0.24, reply characters 0.11, parent words 0.10, sarcastic opener 0.10, reply/parent length ratio 0.08; the largest interactions are "reply length × text score" and "average word length × text score".
- **Strongest n-grams in the text score**: sarcastic — mock-agreement patterns *yes because, yeah because, but but, good thing, how dare, everyone knows, would never*; non-sarcastic — replies ending with a full stop or exclamation mark, brackets, *not sure / pretty sure / for sure*.
- **One prediction**: reply *But that's not racism, that's the trufs!* to a parent complaining that the sub is painted as racist. P(sarcastic) = 0.946: text score +2.56 (`But` opener and `racism` push up, `!` pushes down), sarcastic opener +0.29, parent length +0.16, exclamation interaction −0.36. Every term maps to something visible in the input.
- **Possible artefact to flag**: "sarcastic replies rarely end with a full stop" may come from how SARC was collected (the `/s` marker was stripped), not from sarcasm itself. Ablation A8 tests this.

---

## 6. Stacking framework (step 7)

### 6.1 Structure

```
Level 0: N base models, each trained on train, scoring validation and test
   │
   ├─ each base model is Platt-calibrated on validation (scores → probabilities)
   │
Level 1 (meta-learner, trained on validation only, 5-fold cross-fitting)
   ├─ A: logistic regression on calibrated logits (log-linear pool)
   ├─ B: LightGBM on logits + 5 meta-features
   │        (reply words, parent words, empty parent, question marks, exclamation marks)
   └─ final output = (P_A + P_B) / 2
   │
Decision layer (optional): MBR rule picks the operating point without test labels
   (maximise expected F1, or maximise coverage subject to expected precision ≥ 0.75)
```

Design reasons:

- **Meta-learner on validation**: base-model scores on the training set are overfitted and would over-weight strong models; validation is unseen by every base model.
- **5-fold cross-fitting**: fitting and scoring are separated inside validation, so the validation AUCs used to choose between combiners are out-of-fold. The chosen meta-learner is then refitted on all of validation and scores test once.
- **Logits**: LR on logits is a log-linear pool and can give redundant models negative weights, cancelling repeated errors. Equal-weight soft voting cannot, which is why it falls below the best single model.
- **LightGBM meta-learner**: a linear combiner uses one set of weights for all inputs. With length and context meta-features, LightGBM can learn case-dependent gating ("trust model X on short replies, model Y when there is a parent").

### 6.2 Level-0 members (compliant, 24 on sarc_random)

| Family | Members | Features |
|---|---|---|
| Gradient-boosted trees (baseline) | `lgbm_reply_tfidf`, `lgbm_reply_tfidf_surface`, `xgb_reply_tfidf`, `xgb_reply_tfidf_surface` | reply word 1–2-gram TF-IDF (20k) ± 17 surface features |
| Gradient-boosted trees (improved) | `lgbm_cues`, `lgbm_contrast`, `lgbm_cues_contrast`, `lgbm_v60k_cues_contrast`, `lgbm_lrscore`, `lgbm_lrscore_cues_contrast`, `lgbm_lsa`, `lgbm_cues_contrast_lsa` | baseline + cues / contrast / 60k vocabulary / LR score / LSA |
| Linear | `lr_reply_tfidf`, `lr_reply_tfidf_surface`, `lr_word_char_surface`, `nblr_word_char_surface`, `svm_tfidf_lex_surface` | word TF-IDF, word+char n-grams, NB-SVM, lexicon features |
| Naive Bayes | `mnb_reply_tfidf`, `mnb_reply_tfidf_surface` | word TF-IDF |
| Interpretable | `ebm_full`, `ebm_full_notext`, `ebm_named340` | full EBM, EBM without text score, team EBM |
| **From-scratch networks** | `scratch_reply`, `scratch_pair` | own BPE + random embeddings; BiLSTM (reply) / ESIM (reply↔parent cross-attention) |

(`sarc_subreddit` lacks `ebm_full_notext` and `ebm_named340`: 22 members.) Excluded: every RoBERTa, the two MiniLM LightGBMs, and rank-identical duplicates (CNB, the baseline reproduction).

Learned meta-weights (sarc_random, L2 LR, 24 members): largest are `scratch_pair` +0.38, `lgbm_lrscore_cues_contrast` +0.35, `lgbm_lrscore` +0.23, `lgbm_v60k_cues_contrast` +0.23, `mnb_reply_tfidf` +0.22; `lgbm_reply_tfidf_surface` −0.16 and `lr_reply_tfidf_surface` −0.15 are negative, cancelling overlap. Caruana greedy selection picks only `scratch_pair` and `lgbm_lrscore_cues_contrast`, 50% each. Without the from-scratch networks it picks `lgbm_lrscore_cues_contrast` (50%), `lgbm_v60k_cues_contrast` (25%) and `nblr_word_char_surface` (25%): "trees + a character-level linear model".

### 6.3 Combiners tried

| Method | Description |
|---|---|
| Hard voting | each model votes 0/1; score = vote share |
| Soft voting | equal-weight average of calibrated probabilities |
| Weighted linear pool | non-negative weights minimising validation log-loss |
| Caruana greedy selection | repeatedly add (with replacement) the model that most improves validation AUC |
| **Stacking (L2 LR)** | LR on logits |
| L1 stacking | C chosen by inner CV; prunes redundant members |
| Shapley pruning | members are players; coalition value = held-out stacking AUC on top of the strongest single model; Shapley values from 24 random permutations; drop members with value ≤ 0 |
| Gated LightGBM stacking | logits + 5 meta-features |
| **Average of the two stackers** | best on validation; final method |
| MBR decision rules | change the operating point, not the ranking (AUC) |

### 6.4 Results

| Setting | Members | Best single | Soft vote | Greedy | Stacking (L2) | **Final framework** | Δ vs best single [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---|
| With RoBERTa (reference, not compliant) | 28 | 0.8809 | 0.8476 | 0.8838 | 0.8859 | 0.8861 | +0.0052 [+0.0048, +0.0057] |
| Compliant, no from-scratch networks | 22 | 0.8296 | 0.8252 | 0.8323 | 0.8346 | 0.8350 | +0.0054 [+0.0049, +0.0058] |
| **Compliant, with from-scratch networks** | **24** | 0.8296 | 0.8288 | 0.8384 | 0.8416 | **0.8420** | **+0.0124 [+0.0117, +0.0131]** |
| Compliant, with networks, no subreddit member | 23 | 0.8296 | | | 0.8412 | 0.8414 | +0.0117 [+0.0111, +0.0124] |
| **sarc_subreddit: compliant, with networks** | 22 | 0.8226 | 0.8228 | 0.8326 | 0.8335 | **0.8340** | **+0.0114 [+0.0107, +0.0120]** |
| sarc_subreddit: compliant, no networks | 20 | 0.8226 | 0.8191 | 0.8255 | 0.8265 | 0.8268 | +0.0042 [+0.0038, +0.0046] |

Other metrics of the compliant final framework (sarc_random): at 0.5, precision 0.785, recall 0.733, F1 0.758; with the validation threshold for precision ≥ 0.75, test recall 0.779 (best compliant single 0.751, baseline 0.704). The MBR expected-F1 rule (no test labels) raises F1 to 0.772 (P 0.722 / R 0.830); the MBR expected-precision ≥ 0.75 rule gives precision 0.754, recall 0.783.

Paired comparisons (sarc_random):

| Comparison | ΔAUC [95% CI] |
|---|---|
| Compliant stacking (with networks) vs baseline LightGBM | +0.0346 [+0.0335, +0.0358] |
| Compliant stacking: with vs without networks | +0.0070 [+0.0065, +0.0077] |
| Compliant stacking (no networks) vs full EBM | +0.0121 [+0.0113, +0.0129] |
| Compliant stacking (no networks) vs word+char LR | +0.0166 [+0.0156, +0.0174] |

Reading:

- **Soft voting does not help; stacking is required.** Equal averaging is dragged down by weak models; only learned weights (including negative ones) improve on the best single model.
- **Gains come from complementary model families.** Same-family errors are highly correlated (best LightGBM vs word+char LR: 0.98 on validation residuals), while the from-scratch ESIM correlates only 0.91–0.92 with them. In the Shapley values `scratch_pair` is largest (+0.0062), `scratch_reply` second (+0.0023), every other member ≤ +0.0007.
- **Subreddit contributes almost nothing**: removing the only subreddit-encoded member (team EBM) costs ~0.0006, so the subreddit-free version can be the stated final result.
- **Reference with RoBERTa**: the same framework on 28 members (incl. both converged RoBERTa seeds) gives 0.8861 (sarc_subreddit 0.8791), +0.0115 over the earlier all-model stacking (0.8745).

### 6.5 Is this the best combination without changing the models?

Not provably optimal, but close to the ceiling for these predictions:

- **In theory**, with fixed base models the optimal combination is the true posterior P(sarcastic | all model scores). Stacking estimates it on validation; the LR version assumes additive logits, the LightGBM version can learn non-linear, case-dependent combinations. Together they cover the main ways of combining model outputs.
- **In practice**, with the 24 compliant members every learned combiner (L2, L1, Shapley pruning, gated LightGBM, their average) lands at 0.8373–0.8378 validation AUC and 0.8416–0.8420 test AUC. All Shapley values are positive but small, so pruning removes nothing; the game-theoretic view is better for **explaining** contributions than for raising the score.
- **What could still add a little**: 5-fold cross-trained out-of-fold predictions on the 640k training rows so the meta-learner learns from more data (most expensive, expected ~+0.001), or directly optimising weights for AUC (smaller).
- **A clear gain needs a new base model that errs differently, not a better combiner.** Step 5 confirmed this: adding the from-scratch ESIM raised compliant stacking by +0.0070, an order of magnitude more than any change of combiner.

---

## 7. From-scratch neural networks (step 5)

### 7.1 Purpose

RoBERTa beats the best compliant model by ~0.05 but has two advantages at once: an **architecture** that compares reply and parent token by token, and large-scale **pretraining**. A network with no pretrained component but the same reply↔parent cross-attention separates the two, and its different error pattern may diversify the ensemble.

### 7.2 Setup

- **Tokeniser**: byte-level BPE, 30k vocabulary, fitted only on this split's training replies and parents (not RoBERTa's tokenizer). Reply ≤ 64 tokens, parent ≤ 96.
- **Embeddings**: random initialisation (200 dimensions), no pretrained word vectors.
- `scratch_reply`: BiLSTM (192 per direction) over the reply, mean + max pooling, 2-layer MLP; 6.75M parameters.
- `scratch_pair`: ESIM (Chen et al., 2017). One BiLSTM encodes reply and parent; dot-product attention soft-aligns every reply token with every parent token (bidirectional cross-attention); local inference features [a; ã; a−ã; a⊙ã]; a second BiLSTM composes; mean + max pooling on both sides, then an MLP; 7.79M parameters. With an empty parent the model falls back to the reply.
- **Training**: Adam lr 1e-3 (halved when validation AUC stalls), batch 512, dropout 0.3, gradient clipping 5, bf16. AUC on the full validation set after every epoch; best checkpoint kept; stop after 2 epochs without improvement. `sarc_subreddit` trains for the number of epochs chosen on `sarc_random`.
- **Cost**: ~2 minutes per epoch for `scratch_pair` (~20 minutes total); ~25 seconds per epoch for `scratch_reply`.

### 7.3 Results

| Model | Parameters | Best epoch | val | sarc_random test | precision | recall | F1 | recall at P ≥ .75 | sarc_subreddit test |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scratch_reply (reply only) | 6.75M | 6 | 0.8143 | 0.8180 | 0.763 | 0.714 | 0.738 | 0.730 | 0.8113 |
| **scratch_pair (ESIM)** | 7.79M | 7 | 0.8204 | **0.8242** | 0.749 | 0.751 | 0.750 | 0.743 | **0.8153** |

Validation AUC of `scratch_pair` by epoch: 0.796, 0.809, 0.815, 0.818, 0.818, 0.820, **0.820**, 0.819, 0.820 — no improvement after epoch 7.

### 7.4 Conclusions

1. **Context is usable under the course rules**: ESIM with the parent is +0.0063 [+0.0051, +0.0075] over reply-only (sarc_subreddit +0.0040 [+0.0029, +0.0052]) — twice the hand-made contrast features (+0.003) and about half of RoBERTa's context gain (+0.012).
2. **As a single model it is below the best compliant LightGBM**: 0.8242 vs 0.8296, −0.0054 [−0.0067, −0.0039].
3. **But it is the most valuable new stacking member**: compliant stacking rises from 0.8350 to 0.8420 (+0.0070), sarc_subreddit from 0.8268 to 0.8340 (+0.0072).
4. **Where the gap goes** (sarc_random test AUC):

| Stage | AUC | Increment | Main source |
|---|---:|---:|---|
| Baseline LightGBM | 0.8074 | | |
| Best compliant single (feature engineering) | 0.8296 | +0.022 | character n-grams, linear score as feature, linguistic cues |
| Compliant stacking (with networks) | 0.8420 | +0.012 | complementary model families, mainly the from-scratch network |
| RoBERTa single | 0.8809 | +0.039 | large-scale pretraining (+0.057 vs the comparable from-scratch network 0.8242) |
| Stacking with RoBERTa | 0.8861 | +0.005 | |

ESIM is a BiLSTM and RoBERTa a Transformer. Ablation A7 trained a same-size Transformer from scratch: it reaches only 0.8052, 0.019 below the ESIM, so the +0.057 is pretraining rather than architecture.

5. **Transfer**: on Twitter-style data `scratch_pair` is among the best compliant models (TweetEval 0.648 vs 0.54–0.62 for other compliant models), see section 9.

---

## 8. Research questions

| Question | Evidence | Conclusion |
|---|---|---|
| Q1 Is context necessary? | RoBERTa + parent +0.0119; hand-made contrast +0.003; parent bag-of-words ≈ 0; LSA −0.006; from-scratch ESIM + parent +0.0063 | **Context helps**, but only with models that compare reply and parent semantics token by token; the gain grows with that ability (hand-made +0.003 < from-scratch ESIM +0.006 < RoBERTa +0.012) |
| Q2 Do linguistic cues help? | explicit cues +0.0045 (same on both splits); character n-grams +0.011 | Yes, robustly; character-level form cues are more complete than hand-written rules |
| Q3 Is subreddit needed? | every improvement gains the same on sarc_subreddit as on sarc_random; removing the only subreddit member from stacking costs 0.0008 | No; nearly all gains are available without subreddit |

---

## 9. Cross-platform transfer (models trained on sarc_random, test AUC)

| Model | Compliant | figlang_reddit | figlang_twitter | tweeteval_irony | news_headlines |
|---|:---:|---:|---:|---:|---:|
| Baseline LightGBM | ✓ | 0.7450 | 0.6106 | 0.5986 | 0.4475 |
| Original LR (word + surface) | ✓ | 0.7593 | 0.6466 | 0.5847 | 0.4210 |
| word + char LR | ✓ | 0.7729 | 0.6224 | 0.5956 | 0.4437 |
| Best compliant LightGBM | ✓ | 0.7700 | 0.6225 | 0.6140 | 0.4508 |
| Full EBM | ✓ | 0.7706 | 0.6366 | 0.5855 | 0.4342 |
| Team EBM | ✓ | 0.6780 | 0.6137 | 0.5395 | 0.4688 |
| From-scratch, reply only | ✓ | 0.7676 | 0.5979 | 0.6234 | 0.4803 |
| From-scratch ESIM, reply + parent | ✓ | 0.7589 | 0.6377 | 0.6484 | 0.4769 |
| RoBERTa reply only (1 epoch) | ✗ | 0.8192 | 0.6770 | 0.6822 | 0.4709 |
| RoBERTa reply + parent, converged | ✗ | **0.8276** | **0.7479** | **0.6973** | **0.5057** |

RoBERTa is the only method that clearly improves transfer to Twitter-style data (+0.07 to +0.10). Among compliant models, word+char LR is best on figlang_reddit (0.7729), the original LR on figlang_twitter (0.6466), and the from-scratch ESIM on TweetEval (0.6484).

News headlines (The Onion-style sarcasm) sit around 0.5 for every method: headlines have no conversational context and none of Reddit's mock-agreement patterns, and the Reddit-learned markers even point the wrong way.

---

## 10. Limitations

1. Compliant models (including the from-scratch networks) use one seed; RoBERTa has two seeds on sarc_random with nearly identical results.
2. Author overlap: 84% of test authors appear in training, but A9 shows that no model loses AUC on the unseen-author rows.
3. About 0.01 AUC depends on how replies end (A8); SARC alone cannot tell whether this is a `/s`-removal artefact or genuine style.
4. RoBERTa only had its training length tuned (no learning-rate / batch search); larger models were not tried because the GPU is shared. It is a reference only and does not affect submitted results.
5. The SVM's news-headline score in our re-scoring (0.4231) differs from the original notebook output (0.4453) although in-domain results match exactly; the cause was not found.
6. Whether VADER counts as a pretrained model needs confirmation with the course (see 1.1).

---

## 11. Ablations (step 9)

**Done** (details in `docs/steps/09_ablations.md`):

| ID | Result |
|---|---|
| A9 author overlap | 13% of test rows have authors unseen in training; every model is slightly *better* on them (compliant stack 0.8428 vs 0.8416). Results are not driven by author memorisation. |
| A4 stacking members | removing the from-scratch networks −0.0070, improved GBDTs −0.0028, any other family < 0.001; identical on sarc_subreddit |
| A5 stacking design | calibration, logits vs probabilities, LR vs LightGBM meta-learner, meta-features, cross-fitting: all within 0.0005 |
| A2 text representation | word + char + surface 0.8184; word + char 0.8166; word + surface 0.8008; char + surface 0.7981 |
| A1 feature blocks | removing the LR text score −0.0153, word TF-IDF −0.0056, contrast −0.0020, cues −0.0011, surface −0.0011; 61 dense features alone 0.8240 |
| A6 EBM design | no interactions 0.8215 (−0.0014); 200k sample 0.8225 (−0.0004) |
| A7 network design | ESIM without cross-attention 0.8232 (−0.0010); same-size Transformer from scratch 0.8052 (−0.0190 vs ESIM) |
| A8 trailing punctuation | stripping endings at test −0.0122; retraining on stripped text 0.8069 (−0.0097) |

Not run: A3 (converged reply-only RoBERTa), a reference-only model.

### Original plan

**We recommend moving straight to ablations.** The compliant line has plateaued — the last feature additions bring ~+0.002 each and combiners differ by < 0.001. The report now most needs systematic evidence of each component's contribution and checks of how trustworthy the results are.

| ID | Ablation | How | Existing material | New runs |
|---|---|---|---|---|
| A1 | feature blocks (single model) | leave one block out of the best compliant LightGBM: surface, cues, contrast, LR text score | forward additions already exist (steps 02–06) | 4 LightGBM runs, ~1 h CPU |
| A2 | text representation | word only / word + char / char only n-gram LR | word, word + char exist | char only, 1 run |
| A3 | context (Q1) | every model: reply only vs + parent | RoBERTa (1 epoch) and from-scratch networks exist | converged reply-only RoBERTa (~2.6 h GPU, optional) |
| A4 | stacking members | drop one model family at a time (trees / linear / NB / EBM / from-scratch) | all predictions saved; rerun meta-learner only | ~2 min each |
| A5 | stacking design | calibration on/off; logit vs probability inputs; LR vs LightGBM meta-learner; meta-features on/off; cross-fitting vs direct fit | as above | ~2 min each |
| A6 | EBM design | with/without text score, with/without interactions, full data vs 200k sample | with/without text score exists | 2–3 EBM runs, ~5 min each |
| A7 | from-scratch network design | ESIM without cross-attention (encode separately, concatenate); BiLSTM replaced by a same-size from-scratch Transformer | none | ~20–40 min GPU each |
| A8 | data-artefact check | strip trailing punctuation from replies and re-evaluate (the EBM shows "ends with a full stop" is the strongest non-sarcastic signal) | none | recompute features and predictions, ~1 h CPU |
| A9 | author overlap | evaluate only on authors never seen in training (`author` field available) | none | re-evaluation only, minutes |

Suggested order: (1) A4, A5, A9, A2 — cheap, same day, A9 matters most for credibility; (2) A1, A6, A8 — CPU, half a day; (3) A7 — GPU, a stricter answer to "architecture or pretraining"; (4) if time allows, the converged reply-only RoBERTa in A3.

---

## 12. Where things are

| Path | Contents |
|---|---|
| `notebooks/01–09_*.ipynb` | step-by-step notebooks that recompute every number here from saved predictions |
| `docs/steps/01–09_*.md` | one document per step |
| `pipeline/` | data download, cleaning, EDA, splits (step 1) |
| `modelling/` | team baselines (step 2) |
| `experiments/` | training code for steps 3–7 (`common.py`, `features_extra.py`, `lgbm_runner.py`, numbered folders 00–12) and their small result files |
| `results/ensemble/` | leaderboards written by the notebooks |
| `figures/steps/` | figures written by the notebooks |
| experiment predictions (~500 MB) and caches | not in git; see `experiments/README.md` |
