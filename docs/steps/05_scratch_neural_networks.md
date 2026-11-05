# Step 5 — Neural networks trained from scratch

Notebook: [`notebooks/05_scratch_neural_networks.ipynb`](../../notebooks/05_scratch_neural_networks.ipynb) · Code: `experiments/12_scratch_nn/nn_scratch.py`, `run_queue.py` · Summary: section 7

## Purpose

The pretrained reference (step 6) is about 0.05 AUC above the best compliant model, but it has two advantages at once: an architecture that compares reply and parent token by token, and large-scale pretraining. A network with the same kind of cross-attention but **no pretrained component** separates the two, and it may make different mistakes from the tree and linear models.

## Setup

- Tokeniser: byte-level BPE with a 30k vocabulary, fitted only on this split's training replies and parents. Reply up to 64 tokens, parent up to 96.
- Embeddings: 200-dimensional, randomly initialised; no pretrained word vectors.
- `scratch_reply`: BiLSTM (192 per direction), mean + max pooling, 2-layer MLP; 6.75M parameters.
- `scratch_pair`: ESIM (Chen et al., 2017). A shared BiLSTM encodes reply and parent; dot-product attention aligns every reply token with every parent token in both directions; local inference features [a; ã; a−ã; a⊙ã]; a composition BiLSTM; mean + max pooling of both sides; MLP. 7.79M parameters. With an empty parent it falls back to the reply.
- Training: Adam (lr 1e-3, halved when validation AUC stalls), batch 512, dropout 0.3, gradient clipping 5, bf16; full-validation AUC every epoch; best checkpoint kept; stop after 2 epochs without improvement. `sarc_subreddit` trains for the epoch count chosen on `sarc_random`.
- Runs detached with resumable checkpoints and a GPU memory cap, because the GPU is shared with other jobs. About 20 minutes for ESIM.

## Results

| Model | best epoch | val | sarc_random test | sarc_subreddit test | F1 | recall at P ≥ .75 |
|---|---:|---:|---:|---:|---:|---:|
| scratch_reply (BiLSTM) | 6 | 0.8143 | 0.8180 | 0.8113 | 0.738 | 0.730 |
| **scratch_pair (ESIM)** | 7 | 0.8204 | **0.8242** | **0.8153** | 0.750 | 0.743 |

- Context gain: ESIM vs reply-only **+0.0063 [+0.0051, +0.0075]** on sarc_random and +0.0040 [+0.0029, +0.0052] on sarc_subreddit.
- Against the best compliant LightGBM: −0.0054 [−0.0067, −0.0039].
- Validation residual correlation with the tree, linear and EBM models is 0.93–0.94, while those models correlate 0.98–0.99 with each other.

## Takeaways

- Context is usable under the course rules: the gain is twice that of the hand-made contrast features and about half of RoBERTa's.
- On its own it does not beat feature-engineered LightGBM, but because it errs differently it becomes the **most valuable stacking member** (step 7: +0.0070).
- It is the best compliant model on TweetEval transfer (0.648).
- Ablations (steps 9 and 10): cross-attention adds only +0.001 over encoding the parent separately; a same-size from-scratch Transformer is 0.018–0.019 lower; three seeds give 0.8237 ± 0.0004; on `sarc_temporal` the ESIM is +0.013 over the baseline.
