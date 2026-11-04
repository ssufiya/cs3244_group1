"""
Round 11 - RoBERTa-base (reply + parent) trained on the full train split until
validation AUC converges.

Round 07 trained one epoch only. Here the linear schedule spans --max-epochs
(default 3) and the model is scored on the full validation split every
--eval-every epochs; the best checkpoint (by val AUC) is kept and training stops
after --patience evaluations without improvement. The test sets are scored once,
with the best checkpoint. For sarc_subreddit, pass --fixed-epochs to train the
number of epochs chosen on sarc_random (no early stopping on the target scheme,
same protocol as the baselines' frozen tree counts).

  python 11_full_convergence/roberta_converge.py --scheme sarc_random
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "07_transformer_finetune"))
from common import (CACHE_DIR, SEED, load_frame, parent_of, row_ids, text_of, save_predictions, scores,  # noqa: E402
                    select_threshold, write_json)
from run import MODEL, TRANSFER, predict  # noqa: E402

MAX_LEN = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="sarc_random")
    ap.add_argument("--max-epochs", type=float, default=3.0)
    ap.add_argument("--fixed-epochs", type=float, default=0.0,
                    help="train exactly this long, no early stopping (sarc_subreddit)")
    ap.add_argument("--eval-every", type=float, default=0.5, help="epochs between val checks")
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--save-every", type=int, default=2000, help="steps between resumable state saves")
    ap.add_argument("--stop-after", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--limit", type=int, default=0, help="smoke test rows")
    a = ap.parse_args()
    tag = "roberta_pair_conv" + ("" if a.seed == SEED else f"_s{a.seed}")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(dev)
    tr, va = load_frame(a.scheme, "train"), load_frame(a.scheme, "val")
    if a.limit:
        tr, va = tr.head(a.limit), va.head(a.limit // 4)
    yv = va["label"].to_numpy()
    epochs = a.fixed_epochs or a.max_epochs
    per_epoch = int(math.ceil(len(tr) / a.bs))
    steps = int(per_epoch * epochs)
    eval_steps = max(1, int(per_epoch * a.eval_every))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    ckpt = CACHE_DIR / f"11_{tag}__{a.scheme}.pt"
    log = {"model": MODEL, "variant": "pair", "scheme": a.scheme, "lr": a.lr, "bs": a.bs,
           "seed": a.seed, "schedule_epochs": epochs, "evals": []}
    best, bad, t0, step, run_loss, stop = -1.0, 0, time.time(), 0, 0.0, False
    # resumable state (the laptop session can be torn down mid-run): saved every
    # --save-every steps; the epoch's shuffle is rebuilt from the rng state saved
    # at the epoch start and the batches already done are skipped
    state_path = CACHE_DIR / f"11_{tag}__{a.scheme}{'_smoke' if a.limit else ''}__state.pt"
    skip, epoch_rng = 0, None
    if state_path.exists():
        st = torch.load(state_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sch.load_state_dict(st["sch"])
        step, best, bad, run_loss, log = st["step"], st["best"], st["bad"], st["run_loss"], st["log"]
        epoch_rng, skip = st["epoch_rng"], st["step"] - st["epoch_start_step"]
        torch.set_rng_state(st["torch_rng"].cpu()); torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        t0 -= st["seconds"]
        print(f"[{tag}/{a.scheme}] resumed at step {step}", flush=True)
    model.train()
    r_txt, p_txt, y_all = text_of(tr).tolist(), parent_of(tr).tolist(), tr["label"].to_numpy()
    while step < steps and not stop:
        if epoch_rng is not None:
            rng.bit_generator.state = epoch_rng
        epoch_rng = rng.bit_generator.state
        epoch_start = step - skip
        perm = rng.permutation(len(tr))       # same shuffle as round 07's batches()
        starts = range(skip * a.bs, len(perm), a.bs)
        skip = 0
        for s0 in starts:
            b = perm[s0:s0 + a.bs]
            enc = tok([r_txt[i] for i in b], [p_txt[i] for i in b], truncation="longest_first",
                      max_length=MAX_LEN, padding=True, return_tensors="pt")
            y = torch.tensor(y_all[b], dtype=torch.long)
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
            if (not a.fixed_epochs and step % eval_steps == 0) or step >= steps:
                if a.fixed_epochs:
                    torch.save(model.state_dict(), ckpt)
                    break
                auc = roc_auc_score(yv, predict(model, tok, va, "pair", MAX_LEN, dev))
                ep = step / per_epoch
                log["evals"].append({"epoch": round(ep, 3), "step": step, "val_auc": auc,
                                     "train_loss_ema": run_loss, "seconds": round(time.time() - t0)})
                print(f"[{tag}/{a.scheme}] epoch {ep:.2f} val AUC {auc:.4f} (best {best:.4f})", flush=True)
                if auc > best + 1e-4:
                    best, bad = auc, 0
                    log["best_epoch"] = round(ep, 3)
                    torch.save(model.state_dict(), ckpt)
                else:
                    bad += 1
                    if bad >= a.patience:
                        stop = True
                        break
            if step % a.save_every == 0 and step < steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(),
                            "step": step, "best": best, "bad": bad, "run_loss": run_loss, "log": log,
                            "epoch_rng": epoch_rng, "epoch_start_step": epoch_start,
                            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                            "seconds": time.time() - t0}, str(state_path) + ".tmp")
                for _ in range(60):        # Windows: a scanner may hold the new file briefly
                    try:
                        Path(str(state_path) + ".tmp").replace(state_path)
                        break
                    except PermissionError:
                        time.sleep(2)
                if a.stop_after and step >= a.stop_after:
                    sys.exit(0)            # testing the resume path
            if step >= steps:
                break
        epoch_rng = None
    state_path.unlink(missing_ok=True)
    log["train_seconds"] = round(time.time() - t0)
    log["stopped_early"] = stop
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    keys = [(a.scheme, "val"), (a.scheme, "test")] + ([(t, "test") for t in TRANSFER] if a.scheme == "sarc_random" else [])
    thr = None
    for ds, part in keys:
        if a.limit and part != "val":
            continue
        df = va if (ds, part) == (a.scheme, "val") else load_frame(ds, part)
        p = predict(model, tok, df, "pair", MAX_LEN, dev)
        y = df["label"].to_numpy()
        if not a.limit:
            save_predictions(f"11_{tag}", ds, part, y, {f"{a.scheme}__{tag}": p}, ids=row_ids(df))
        if part == "val":
            log["val_auc"] = roc_auc_score(y, p)
            thr = select_threshold(y, p)
        else:
            s = scores(y, p)
            if ds == a.scheme and thr is not None:
                h = scores(y, p, thr)
                s.update({"p75_precision": h["precision"], "p75_recall": h["recall"]})
            log[ds] = s
        print(f"[{tag}/{a.scheme}] {ds}/{part}: AUC {roc_auc_score(y, p):.4f}", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    write_json(HERE / "results" / f"{tag}__{a.scheme}{'_smoke' if a.limit else ''}.json", log)


if __name__ == "__main__":
    main()
