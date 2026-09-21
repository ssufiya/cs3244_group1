#!/usr/bin/env python
"""
05_verify_splits.py — prove the splits are safe before anyone trains on them.

Two independent things happen here:

1. LEAKAGE AUDIT (the point of the script).
   For every split folder: no text may appear in two splits, no parent thread
   may straddle a boundary, and for `sarc_subreddit` no subreddit may either.
   A failure here invalidates every number produced downstream, so this is a
   hard gate, not a report.

2. REFERENCE FLOOR (a sanity check, NOT the project's baseline).
   A TF-IDF + logistic-regression fit, so we know the data is learnable and
   roughly where the floor sits. The proposal assigns the real base models to
   team members in Week 7 — this is a smoke test to confirm the files load and
   behave, not a substitute for that work. It is deliberately untuned.

Usage
-----
    python pipeline/05_verify_splits.py              # audit + floor
    python pipeline/05_verify_splits.py --audit-only
    python pipeline/05_verify_splits.py --max-train 150000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.paths import DS_PRIMARY, DS_SUPP, SPLIT_RESULTS  # noqa: E402
from common import text_cleaning as tc                                # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 3244
SPLITS = ("train", "val", "test")


def load_split(d: Path) -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(d / f"{s}.parquet")
            for s in SPLITS if (d / f"{s}.parquet").exists()}


def text_col(df: pd.DataFrame) -> str:
    return "comment_clean" if "comment_clean" in df.columns else "text_clean"


# --------------------------------------------------------------------------
# 1. leakage audit
# --------------------------------------------------------------------------
def audit(name: str, d: Path) -> dict:
    parts = load_split(d)
    if not parts:
        return {"dataset": name, "status": "MISSING", "checks": []}

    col = text_col(next(iter(parts.values())))
    checks, failed = [], False

    def add(check: str, bad: int, detail: str = "") -> None:
        nonlocal failed
        ok = bad == 0
        failed = failed or not ok
        checks.append({"check": check, "overlap": int(bad),
                       "status": "PASS" if ok else "FAIL", "detail": detail})

    keys = {s: set(tc.dedup_key(p[col])) for s, p in parts.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in keys and b in keys:
            add(f"text overlap {a}/{b}", len(keys[a] & keys[b]))

    if "parent_clean" in next(iter(parts.values())).columns:
        pk = {s: set(p["parent_clean"].fillna("").str.strip().str.lower())
              for s, p in parts.items()}
        for s in pk:                       # the empty parent is not a thread
            pk[s].discard("")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if a in pk and b in pk:
                add(f"parent-thread overlap {a}/{b}", len(pk[a] & pk[b]))

    if name == "sarc_subreddit":
        sr = {s: set(p["subreddit"]) for s, p in parts.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            add(f"subreddit overlap {a}/{b}", len(sr[a] & sr[b]))

    # Author overlap is reported, never failed: SARC has 254k authors over
    # 914k rows, so disjoint authors are not achievable without gutting the
    # corpus. It is recorded so the write-up can acknowledge it.
    author_note = ""
    if "author" in next(iter(parts.values())).columns:
        au = {s: set(p["author"]) for s, p in parts.items()}
        if "train" in au and "test" in au:
            shared = len(au["train"] & au["test"])
            author_note = (f"{shared:,} authors appear in both train and test "
                           f"({100*shared/max(1,len(au['test'])):.0f}% of test "
                           f"authors) — inherent to SARC, disclose in the report")

    stats = {s: {"rows": int(len(p)),
                 "sarcastic_pct": round(float(p["label"].mean()) * 100, 2)}
             for s, p in parts.items()}

    return {"dataset": name, "status": "FAIL" if failed else "PASS",
            "checks": checks, "splits": stats, "author_note": author_note}


# --------------------------------------------------------------------------
# 2. reference floor
# --------------------------------------------------------------------------
def floor(name: str, d: Path, max_train: int) -> dict | None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.dummy import DummyClassifier

    parts = load_split(d)
    if "train" not in parts or "test" not in parts:
        return None
    col = text_col(parts["train"])

    tr = parts["train"]
    if len(tr) > max_train:
        tr = tr.sample(max_train, random_state=SEED)
    te = parts["test"]

    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                          sublinear_tf=True, strip_accents="unicode")
    Xtr = vec.fit_transform(tr[col].fillna(""))
    Xte = vec.transform(te[col].fillna(""))

    dummy = DummyClassifier(strategy="most_frequent").fit(Xtr, tr["label"])
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, tr["label"])
    pred = clf.predict(Xte)

    res = {
        "dataset": name,
        "train_rows_used": int(len(tr)),
        "test_rows": int(len(te)),
        "majority_class_accuracy": round(float(accuracy_score(
            te["label"], dummy.predict(Xte))), 4),
        "tfidf_logreg": {
            "accuracy": round(float(accuracy_score(te["label"], pred)), 4),
            "precision": round(float(precision_score(te["label"], pred, zero_division=0)), 4),
            "recall": round(float(recall_score(te["label"], pred, zero_division=0)), 4),
            "f1": round(float(f1_score(te["label"], pred, zero_division=0)), 4),
        },
        "n_features": int(Xtr.shape[1]),
        "fit_seconds": round(time.time() - t0, 1),
    }
    m = res["tfidf_logreg"]
    print(f"  [{name:<18}] acc={m['accuracy']:.4f}  P={m['precision']:.4f}  "
          f"R={m['recall']:.4f}  F1={m['f1']:.4f}   "
          f"(majority {res['majority_class_accuracy']:.4f}, {res['fit_seconds']}s)")
    return res


# --------------------------------------------------------------------------
# 3. cross-corpus transfer floor
# --------------------------------------------------------------------------
def transfer(max_train: int) -> list[dict]:
    """
    Train once on SARC, score on every other corpus's test split.

    This is the sub-question-4 number in its cheapest form. Having it now means
    the Week-7 models have something to beat, and it tells us in advance
    whether cross-platform transfer is a real experiment or a formality.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    src = DS_PRIMARY / "sarc_random"
    if not (src / "train.parquet").exists():
        return []
    tr = pd.read_parquet(src / "train.parquet")
    if len(tr) > max_train:
        tr = tr.sample(max_train, random_state=SEED)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000,
                          sublinear_tf=True, strip_accents="unicode")
    Xtr = vec.fit_transform(tr["comment_clean"].fillna(""))
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr, tr["label"])

    out = []
    targets = [("sarc_random (in-domain)", DS_PRIMARY / "sarc_random" / "test.parquet")]
    targets += [(p.name, p / "test.parquet") for p in sorted(DS_SUPP.iterdir())
                if p.is_dir() and (p / "test.parquet").exists()]

    for name, path in targets:
        te = pd.read_parquet(path)
        col = text_col(te)
        pred = clf.predict(vec.transform(te[col].fillna("")))
        r = {
            "target": name,
            "test_rows": int(len(te)),
            "accuracy": round(float(accuracy_score(te["label"], pred)), 4),
            "precision": round(float(precision_score(te["label"], pred, zero_division=0)), 4),
            "recall": round(float(recall_score(te["label"], pred, zero_division=0)), 4),
            "f1": round(float(f1_score(te["label"], pred, zero_division=0)), 4),
            "majority_baseline": round(float(max(te["label"].mean(),
                                                 1 - te["label"].mean())), 4),
        }
        out.append(r)
        print(f"  SARC -> {name:<26} acc={r['accuracy']:.4f}  F1={r['f1']:.4f}   "
              f"(majority {r['majority_baseline']:.4f})")
    return out


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--max-train", type=int, default=200_000,
                    help="cap on training rows for the reference floor")
    args = ap.parse_args()

    targets = [(p.name, p) for p in sorted(DS_PRIMARY.iterdir()) if p.is_dir()]
    targets += [(p.name, p) for p in sorted(DS_SUPP.iterdir()) if p.is_dir()]

    print("=== LEAKAGE AUDIT ===")
    audits, n_fail = [], 0
    for name, d in targets:
        a = audit(name, d)
        audits.append(a)
        bad = [c for c in a["checks"] if c["status"] == "FAIL"]
        n_fail += len(bad)
        print(f"  [{a['status']}] {name:<18} {len(a['checks'])} checks"
              + (f"  <-- {len(bad)} FAILED" if bad else ""))
        for c in bad:
            print(f"        FAIL {c['check']}: {c['overlap']:,} overlapping")
        if a.get("author_note"):
            print(f"        note: {a['author_note']}")

    floors = []
    if not args.audit_only:
        print("\n=== REFERENCE FLOOR (TF-IDF + logistic regression, untuned) ===")
        print("  NOT the project baseline — a smoke test that the splits load "
              "and are learnable.")
        for name, d in targets:
            try:
                r = floor(name, d, args.max_train)
                if r:
                    floors.append(r)
            except Exception as exc:                          # noqa: BLE001
                print(f"  [{name}] floor failed: {exc}")

    transfers = []
    if not args.audit_only:
        print("\n=== CROSS-CORPUS TRANSFER (train on SARC, test elsewhere) ===")
        try:
            transfers = transfer(args.max_train)
        except Exception as exc:  # noqa: BLE001
            print(f"  transfer check failed: {exc}")

    SPLIT_RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audit": audits,
        "reference_floor": floors,
        "cross_corpus_transfer": transfers,
        "audit_failures": n_fail,
    }
    p = SPLIT_RESULTS / "split_verification.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[save] {p}")
    print(f"{'ALL CHECKS PASSED' if n_fail == 0 else str(n_fail) + ' CHECK(S) FAILED'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
