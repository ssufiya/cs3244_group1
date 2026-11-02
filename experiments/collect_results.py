"""
Build one leaderboard from the saved predictions, so every number in the
report is computed the same way.

For each model column on sarc_random/test, sarc_subreddit/test and the transfer
sets: ROC-AUC, PR-AUC, accuracy / precision / recall / F1 at 0.5, and recall at
the validation-chosen precision >= 0.75 threshold (sarc_random/val for the
transfer sets). The paired bootstrap compares each model with the LightGBM
baseline on the same rows.

Output: results_master.csv
"""
from __future__ import annotations

import pandas as pd

from common import HERE, load_predictions, paired_bootstrap_auc, scores, select_threshold

SETS = [("sarc_random", "sarc_random", "sarc_random"), ("sarc_subreddit", "sarc_subreddit", "sarc_subreddit"),
        ("news_headlines", "sarc_random", "sarc_random"), ("figlang_reddit", "sarc_random", "sarc_random"),
        ("figlang_twitter", "sarc_random", "sarc_random"), ("tweeteval_irony", "sarc_random", "sarc_random")]
REF = "lgbm_reply_tfidf_surface"


def main():
    rows = []
    for test_ds, val_ds, scheme in SETS:
        te = load_predictions(test_ds, "test", scheme=scheme)
        va = load_predictions(val_ds, "val", scheme=scheme)
        y, yv = te["label"].to_numpy(), va["label"].to_numpy()
        for m in te.columns:
            if m in ("row_position", "label") or m not in va:
                continue
            p, pv = te[m].to_numpy(), va[m].to_numpy()
            is_prob = p.min() >= 0 and p.max() <= 1
            r = {"set": test_ds, "model": m, "n": len(y)}
            r.update(scores(y, p, 0.5 if is_prob else 0.0))
            thr = select_threshold(yv, pv)
            if thr is not None:
                hi = scores(y, p, thr)
                r.update({"p75_precision": hi["precision"], "p75_recall": hi["recall"]})
            if REF in te and m != REF and test_ds in ("sarc_random", "sarc_subreddit"):
                b = paired_bootstrap_auc(y, p, te[REF].to_numpy(), n_boot=200)
                r.update({"d_auc": b["diff"], "ci_low": b["ci_low"], "ci_high": b["ci_high"]})
            rows.append(r)
        print(test_ds, "done", flush=True)
    df = pd.DataFrame(rows).sort_values(["set", "roc_auc"], ascending=[True, False])
    df.to_csv(HERE / "results_master.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(df[df.set.isin(["sarc_random", "sarc_subreddit", "news_headlines"])]
              [["set", "model", "roc_auc", "accuracy", "precision", "recall", "f1", "p75_recall", "d_auc",
                "ci_low", "ci_high"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
