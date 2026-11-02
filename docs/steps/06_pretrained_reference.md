# Step 6 — Pretrained reference: RoBERTa (not part of submitted results)

Notebook: [`notebooks/06_pretrained_reference.ipynb`](../../notebooks/06_pretrained_reference.ipynb) · Code: `experiments/07_transformer_finetune/`, `experiments/11_full_convergence/roberta_converge.py` · Summary: section 4

> **Course rule.** Pretrained models are not allowed in course results. RoBERTa (and the MiniLM features in step 3) are kept only as a clearly labelled upper bound, to measure what pretraining is worth. None of them is a member of the submitted ensemble.

## Setup

- RoBERTa-base (12 layers, 768 hidden, about 125M parameters), byte-level BPE.
- Pair input `<s> reply </s></s> parent </s>` (128 tokens, longer side truncated first); reply-only control `<s> reply </s>` (64 tokens).
- Full fine-tuning: AdamW lr 2e-5, weight decay 0.01, 6% warm-up then linear decay, batch 64 (4 accumulated micro-batches), bf16, gradient clipping 1.0.
- Step 07 trained one epoch. Round 11 extended the schedule to 3 epochs with full-validation AUC every half epoch, keeping the best checkpoint; state is saved every 2,000 steps so an interrupted run resumes.

## Results

| Model | val | sarc_random test | sarc_subreddit | recall at P ≥ .75 |
|---|---:|---:|---:|---:|
| reply only, 1 epoch | | 0.8599 | | 0.818 |
| reply + parent, 1 epoch | 0.8691 | 0.8718 | 0.8670 | 0.847 |
| **reply + parent, converged (3 epochs)** | **0.8784** | **0.8809** | **0.8762** | **0.864** |
| converged, second seed | 0.8782 | 0.8808 | | |

Validation AUC by epoch: 0.8553, 0.8670, 0.8737, 0.8771, 0.8773, **0.8784**. Training time 2.6 hours.

## What the reference tells us

- **Context**: reply + parent vs reply-only is +0.0119, the largest context gain of any model.
- **Pretraining**: about +0.057 over the from-scratch ESIM (0.8242), which can also cross-attend reply and parent. Part of this gap is Transformer vs BiLSTM architecture (ablation A7).
- **Transfer**: best on Twitter-style data (FigLang Twitter 0.748, TweetEval 0.697). News headlines are about 0.5 for every model.
- With RoBERTa the stacking framework reaches 0.8861; this is a reference number only.
