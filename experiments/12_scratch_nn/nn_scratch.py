"""
Round 12 - neural networks trained FROM SCRATCH (no pretrained weights, no
pretrained tokenizer), as the course-compliant bridge between the bag-of-words
models and the RoBERTa reference.

Tokenizer: byte-level BPE (30k) fitted on this scheme's TRAIN replies + parents only.
Embeddings: randomly initialised.

Variants
  reply   BiLSTM encoder over the reply, max+mean pooling, MLP.
  pair    ESIM-style (Chen et al. 2017): the same BiLSTM encodes reply and parent;
          soft alignment (cross-attention) of every reply token against every
          parent token and vice versa; local inference features [a; b; a-b; a*b];
          a second BiLSTM composes them; max+mean pooling of both sides; MLP.
          An empty parent is a single <pad>-masked position, so the model falls
          back to the reply.

Training: Adam lr 1e-3, batch 512, dropout 0.3, grad clip 5, up to --max-epochs;
validation AUC every epoch, best checkpoint kept, stop after --patience epochs
without improvement (convergence). sarc_subreddit uses --fixed-epochs taken from
the sarc_random run. State is saved every epoch and the run resumes after an
interruption.

  python 12_scratch_nn/nn_scratch.py --variant pair --scheme sarc_random
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import (load_frame, parent_of, row_ids, save_predictions, scores,  # noqa: E402
                    select_threshold, text_of, write_json)

CACHE = ROOT / ".cache" / "12_scratch"
CACHE.mkdir(parents=True, exist_ok=True)
TRANSFER = ["news_headlines", "figlang_reddit", "figlang_twitter", "tweeteval_irony"]
R_LEN, P_LEN, VOCAB = 64, 96, 30_000
SEED = 3244


# --------------------------------------------------------------------------
def tokenizer(scheme):
    from tokenizers import ByteLevelBPETokenizer
    path = CACHE / f"bpe_{scheme}"
    if (path / "vocab.json").exists():
        return ByteLevelBPETokenizer(str(path / "vocab.json"), str(path / "merges.txt"))
    tr = load_frame(scheme, "train")
    tok = ByteLevelBPETokenizer(lowercase=False)
    corpus = text_of(tr).tolist() + parent_of(tr).tolist()
    tok.train_from_iterator(corpus, vocab_size=VOCAB, min_frequency=3, special_tokens=["<pad>", "<unk>"])
    path.mkdir(exist_ok=True)
    tok.save_model(str(path))
    return tok


def encode(tok, texts, max_len):
    out = np.zeros((len(texts), max_len), np.int32)          # 0 = <pad>
    for i, e in enumerate(tok.encode_batch(texts)):
        ids = e.ids[:max_len]
        out[i, :len(ids)] = ids
    return out


def arrays(scheme, ds, part, tok):
    p = CACHE / f"{scheme}__{ds}__{part}.npz"
    if p.exists():
        z = np.load(p)
        return z["r"], z["p"], z["y"]
    df = load_frame(ds, part)
    r, q, y = encode(tok, text_of(df).tolist(), R_LEN), encode(tok, parent_of(df).tolist(), P_LEN), df["label"].to_numpy()
    np.savez(p, r=r, p=q, y=y)
    return r, q, y


# --------------------------------------------------------------------------
def masked_pool(h, m):
    mf = m.unsqueeze(-1).float()
    mean = (h * mf).sum(1) / mf.sum(1).clamp(min=1)
    mx = h.masked_fill(~m.unsqueeze(-1), -1e4).max(1).values
    return torch.cat([mean, mx], -1)


class Net(nn.Module):
    def __init__(self, variant, vocab=VOCAB, emb=200, hid=192, drop=0.3):
        super().__init__()
        self.variant = variant
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.enc = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        d = 2 * hid
        if variant == "pair":
            self.proj = nn.Sequential(nn.Linear(4 * d, hid), nn.ReLU())
            self.comp = nn.LSTM(hid, hid, batch_first=True, bidirectional=True)
            head_in = 4 * d                                   # pooled reply + parent
        else:
            head_in = 2 * d
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(head_in, hid), nn.Tanh(),
                                  nn.Dropout(drop), nn.Linear(hid, 1))

    def encode(self, x):
        m = x != 0
        m[:, 0] |= ~m.any(1)                                  # empty text: keep one position
        h, _ = self.enc(self.drop(self.emb(x)))
        return h, m

    def forward(self, r, p=None):
        a, ma = self.encode(r)
        if self.variant != "pair":
            return self.head(masked_pool(a, ma)).squeeze(-1)
        b, mb = self.encode(p)
        e = torch.bmm(a, b.transpose(1, 2))                   # reply x parent alignment scores
        a_t = torch.bmm(F.softmax(e.masked_fill(~mb.unsqueeze(1), -1e4), -1), b)   # parent summary per reply token
        b_t = torch.bmm(F.softmax(e.masked_fill(~ma.unsqueeze(2), -1e4), 1).transpose(1, 2), a)
        ma_, mb_ = [torch.cat([x, y, x - y, x * y], -1) for x, y in ((a, a_t), (b, b_t))]
        va, _ = self.comp(self.drop(self.proj(ma_)))
        vb, _ = self.comp(self.drop(self.proj(mb_)))
        return self.head(torch.cat([masked_pool(va, ma), masked_pool(vb, mb)], -1)).squeeze(-1)


@torch.no_grad()
def predict(model, r, p, dev, bs=256):
    model.eval()
    out = []
    for s in range(0, len(r), bs):
        rr = torch.from_numpy(r[s:s + bs]).long().to(dev)
        pp = torch.from_numpy(p[s:s + bs]).long().to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(rr, pp).float()).cpu().numpy())
    model.train()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["reply", "pair"], required=True)
    ap.add_argument("--scheme", default="sarc_random")
    ap.add_argument("--max-epochs", type=int, default=12)
    ap.add_argument("--fixed-epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mem-frac", type=float, default=0.27,
                    help="cap on this process's share of GPU memory (the GPU is shared)")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    tag = f"scratch_{a.variant}"
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = "cuda"
    tok = tokenizer(a.scheme)
    rtr, ptr, ytr = arrays(a.scheme, a.scheme, "train", tok)
    rva, pva, yva = arrays(a.scheme, a.scheme, "val", tok)
    model = Net(a.variant).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=0)
    n_params = sum(p.numel() for p in model.parameters())
    ckpt, state_p = CACHE / f"{tag}__{a.scheme}.pt", CACHE / f"{tag}__{a.scheme}__state.pt"
    log = {"variant": a.variant, "scheme": a.scheme, "params": n_params, "evals": []}
    ep0, best, bad, t0 = 0, -1.0, 0, time.time()
    if state_p.exists():
        st = torch.load(state_p, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        ep0, best, bad, log = st["epoch"], st["best"], st["bad"], st["log"]
        rng.bit_generator.state = st["rng"]
        t0 -= st["seconds"]
        print(f"[{tag}/{a.scheme}] resumed after epoch {ep0}", flush=True)
    epochs = a.fixed_epochs or a.max_epochs
    model.train()
    for ep in range(ep0 + 1, epochs + 1):
        perm, tot = rng.permutation(len(ytr)), 0.0
        for s in range(0, len(perm), a.bs):
            b = perm[s:s + a.bs]
            r = torch.from_numpy(rtr[b]).long().to(dev)
            p = torch.from_numpy(ptr[b]).long().to(dev)
            y = torch.from_numpy(ytr[b]).float().to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.binary_cross_entropy_with_logits(model(r, p).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b)
        auc = roc_auc_score(yva, predict(model, rva, pva, dev))
        log["evals"].append({"epoch": ep, "train_loss": tot / len(ytr), "val_auc": auc,
                             "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0)})
        print(f"[{tag}/{a.scheme}] epoch {ep} loss {tot/len(ytr):.4f} val AUC {auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if a.fixed_epochs or auc > best + 1e-4:
            best, bad = max(best, auc), 0
            log["best_epoch"] = ep
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        if not a.fixed_epochs:
            sched.step(auc)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best": best, "bad": bad, "log": log, "rng": rng.bit_generator.state,
                    "seconds": time.time() - t0}, state_p)
        if not a.fixed_epochs and bad >= a.patience:
            break
    log["train_seconds"] = round(time.time() - t0)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    keys = [(a.scheme, "val"), (a.scheme, "test")] + ([(t, "test") for t in TRANSFER] if a.scheme == "sarc_random" else [])
    thr = None
    for ds, part in keys:
        r, p, y = (rva, pva, yva) if (ds, part) == (a.scheme, "val") else arrays(a.scheme, ds, part, tok)
        pr = predict(model, r, p, dev)
        df_ids = row_ids(load_frame(ds, part))
        save_predictions(f"12_{tag}", ds, part, y, {f"{a.scheme}__{tag}": pr}, ids=df_ids)
        if part == "val":
            log["val_auc"] = roc_auc_score(y, pr)
            thr = select_threshold(y, pr)
        else:
            s = scores(y, pr)
            if ds == a.scheme and thr is not None:
                h = scores(y, pr, thr)
                s.update({"p75_precision": h["precision"], "p75_recall": h["recall"]})
            log[ds] = s
        print(f"[{tag}/{a.scheme}] {ds}/{part}: AUC {roc_auc_score(y, pr):.4f}", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    write_json(HERE / "results" / f"{tag}__{a.scheme}.json", log)
    state_p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
