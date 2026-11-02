"""
Improvement 8 - pretrained sentence embeddings (frozen, no fine-tuning).

all-MiniLM-L6-v2 (sentence-transformers) embeds every reply and parent once
(texts are de-duplicated across splits). LightGBM gets, on top of the best
hand-built set (60k vocabulary + cues + contrast):
  * the 384-d reply embedding (meaning beyond exact n-grams: paraphrases of
    "yeah, great idea", topic, tone),
  * cosine(reply, parent) and ||reply - parent|| (semantic agreement with the
    comment being answered), plus a has-parent flag.
The encoder is frozen, so nothing is fitted on labels outside LightGBM.
Encodes on the CPU by default (EMB_DEVICE=cuda to use the GPU).
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import parent_of, text_of, write_json  # noqa: E402
from features_extra import contrast_block, cue_block  # noqa: E402
from lgbm_runner import CACHE, TRANSFER, frame, run  # noqa: E402

ENC = "sentence-transformers/all-MiniLM-L6-v2"
EMB_FILE = CACHE / "minilm_embeddings.npz"


def h(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=12).hexdigest()


def all_keys():
    keys = []
    for scheme in ("sarc_random", "sarc_subreddit"):
        keys += [(scheme, p) for p in ("train", "val", "test")]
    return keys + [(t, "test") for t in TRANSFER]


def build_embeddings():
    if EMB_FILE.exists():
        z = np.load(EMB_FILE, allow_pickle=False)
        return dict(zip(z["keys"].tolist(), range(len(z["keys"])))), z["emb"]
    from sentence_transformers import SentenceTransformer
    texts = {}
    for k in all_keys():
        df = frame(*k)
        for s in text_of(df).tolist() + parent_of(df).tolist():
            if s:
                texts.setdefault(h(s), s)
    keys = list(texts)
    # EMB_DEVICE=cuda once the laptop GPU is out of its power-saving state (it ran at 180 MHz here)
    dev = os.environ.get("EMB_DEVICE", "cpu")
    torch.set_num_threads(20)
    model = SentenceTransformer(ENC, device=dev)
    model.max_seq_length = 64
    order = np.argsort([len(texts[k]) for k in keys])           # length-sorted batches are faster
    t0 = time.time()
    emb = model.encode([texts[keys[i]] for i in order], batch_size=256, normalize_embeddings=True,
                       show_progress_bar=False, convert_to_numpy=True).astype(np.float16)
    out = np.empty_like(emb)
    out[order] = emb
    print(f"[emb] {len(keys):,} unique texts on {dev} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(EMB_FILE, keys=np.array(keys), emb=out)
    return dict(zip(keys, range(len(keys)))), out


IDX, EMB = None, None


def emb_block(df):
    global IDX, EMB
    if IDX is None:
        IDX, EMB = build_embeddings()
    zero = np.zeros(EMB.shape[1], np.float32)

    def look(series):
        return np.stack([EMB[IDX[h(s)]].astype(np.float32) if s else zero for s in series])
    R, P = look(text_of(df).tolist()), look(parent_of(df).tolist())
    has_p = (np.abs(P).sum(1) > 0).astype(np.float32)
    cos = (R * P).sum(1) * has_p
    dist = np.linalg.norm(R - P, axis=1) * has_p
    return np.column_stack([cos, dist, has_p, R]).astype(np.float32)


if __name__ == "__main__":
    (HERE / "results").mkdir(exist_ok=True)
    IDX, EMB = build_embeddings()
    out = {"minilm_v60k_cues_contrast": run("lgbm_minilm_v60k_cues_contrast", "lgbm_minilm",
                                            blocks={"cues": cue_block, "contrast": contrast_block,
                                                    "minilm": emb_block}, vocab=60_000)}
    write_json(HERE / "results" / "summary.json", out)
