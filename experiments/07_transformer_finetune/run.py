"""
Improvement 6 - fine-tuned transformer (RoBERTa-base), reply only vs reply + parent.

This is the representation upgrade the baseline analysis pointed to: bag-of-words
cannot see the semantic clash between a reply and the comment it answers. Two
inputs are compared with everything else held fixed:
  reply         "<reply>"                                   max 64 tokens
  pair          "<reply>" </s></s> "<parent>"  (longer side truncated first) max 128 tokens
One epoch, AdamW lr 2e-5, linear warm-up 6%, batch 64, bf16, seed 3244. No
hyper-parameter search; validation is only used to report and for the later
ensemble. sarc_subreddit refits on its own train with the same settings;
transfer sets are scored by the sarc_random model.

Run inside the conda env cs3244-ensemble:
  python 07_transformer_finetune/run.py --variant pair --scheme sarc_random
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (SEED, load_frame, parent_of, row_ids, save_predictions, scores,  # noqa: E402
                    select_threshold, text_of, write_json)

MODEL = "roberta-base"
TRANSFER = ["news_headlines", "figlang_reddit", "figlang_twitter", "tweeteval_irony"]


def batches(tok, df, variant, max_len, bs, shuffle, rng=None):
    r = text_of(df).tolist()
    p = parent_of(df).tolist()
    y = df["label"].to_numpy()
    idx = rng.permutation(len(df)) if shuffle else np.arange(len(df))
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        if variant == "pair":
            enc = tok([r[i] for i in b], [p[i] for i in b], truncation="longest_first",
                      max_length=max_len, padding=True, return_tensors="pt")
        else:
            enc = tok([r[i] for i in b], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        yield enc, torch.tensor(y[b], dtype=torch.long)


@torch.no_grad()
def predict(model, tok, df, variant, max_len, dev):
    model.eval()
    out = []
    for enc, _ in batches(tok, df, variant, max_len, 256, False):
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**enc).logits.float()
        out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    model.train()
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["reply", "pair"], required=True)
    ap.add_argument("--scheme", default="sarc_random")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=64, help="effective batch size")
    ap.add_argument("--accum", type=int, default=4, help="micro-batches per step (bs/accum rows each)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: rows per partition")
    a = ap.parse_args()
    max_len = 128 if a.variant == "pair" else 64
    tag = f"roberta_{a.variant}"
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
    tr = load_frame(a.scheme, "train")
    va = load_frame(a.scheme, "val")
    if a.limit:
        tr, va = tr.head(a.limit), va.head(a.limit)
    steps = int(math.ceil(len(tr) / a.bs) * a.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    t0, step, run_loss = time.time(), 0, 0.0
    model.train()
    while step < steps:
        for enc, y in batches(tok, tr, a.variant, max_len, a.bs, True, rng):
            # split the batch into micro-batches to stay inside the 12 GB laptop GPU
            n = len(y)
            for lo in range(0, n, -(-n // a.accum)):
                hi = min(n, lo + -(-n // a.accum))
                sub = {k: v[lo:hi].to(dev) for k, v in enc.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = torch.nn.functional.cross_entropy(model(**sub).logits.float(), y[lo:hi].to(dev))
                (loss * (hi - lo) / n).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            step += 1
            run_loss = 0.98 * run_loss + 0.02 * loss.item()
            if step % 500 == 0:
                print(f"[{tag}/{a.scheme}] step {step}/{steps} loss {run_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if step >= steps:
                break
    out = {"model": MODEL, "variant": a.variant, "scheme": a.scheme, "steps": steps,
           "train_seconds": round(time.time() - t0)}
    keys = [(a.scheme, "val"), (a.scheme, "test")] + ([(t, "test") for t in TRANSFER] if a.scheme == "sarc_random" else [])
    pv = None
    for ds, part in keys:
        df = va if (ds, part) == (a.scheme, "val") else load_frame(ds, part)
        if a.limit:
            if part != "val":
                continue
        p = predict(model, tok, df, a.variant, max_len, dev)
        y = df["label"].to_numpy()
        if not a.limit:
            save_predictions(f"transformer_{a.variant}", ds, part, y, {f"{a.scheme}__{tag}": p}, ids=row_ids(df))
        if part == "val":
            pv = p
            out["val_auc"] = roc_auc_score(y, p)
            thr = select_threshold(y, p)
        else:
            s = scores(y, p)
            if ds == a.scheme and thr is not None:
                hi = scores(y, p, thr)
                s.update({"p75_precision": hi["precision"], "p75_recall": hi["recall"]})
            out[ds] = s
        print(f"[{tag}/{a.scheme}] {ds}/{part}: AUC {roc_auc_score(y, p):.4f}", flush=True)
    if a.limit:
        return
    (HERE / "results").mkdir(exist_ok=True)
    write_json(HERE / "results" / f"{tag}__{a.scheme}.json", out)


if __name__ == "__main__":
    main()
