"""
Round 11 - how a single prediction of the full-data EBM is explained.

A prediction's log-odds = intercept + sum of EBM term contributions. The
`text_ngram_logodds` term is a shape function of the word+char LR score, and that
score is itself sum(coef * tf-idf) over the n-grams in the reply, so the text term
can be traced to the n-grams that produced it. The LR is refitted here on the full
sarc_random train with round 06's settings (the val/test scores it gives match the
cached feature) and kept in .cache for reuse.

Writes results/ebm_explanations.md with the global term ranking, the strongest
n-grams of the LR, and local explanations of a few test replies.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "06_linear_score_as_feature"))
sys.path.insert(0, str(HERE))
from common import load_frame, parent_of, text_of  # noqa: E402
from ebm_full import features  # noqa: E402
from lgbm_runner import CACHE  # noqa: E402
from run import C, _fit_matrix  # noqa: E402

SCHEME = "sarc_random"


def lr_model():
    path = CACHE / "11_lr_word_char__sarc_random.joblib"
    if path.exists():
        m, w, c, ss = joblib.load(path)
    else:
        tr = load_frame(SCHEME, "train")
        X, tf = _fit_matrix(tr)
        m = LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(X, tr["label"].to_numpy())
        # _fit_matrix's transform is a closure (not picklable): keep its vectorisers instead
        cells = dict(zip(tf.__code__.co_freevars, (x.cell_contents for x in tf.__closure__)))
        w, c, ss = cells["w"], cells["c"], cells["ss"]
        joblib.dump((m, w, c, ss), path)
    import scipy.sparse as sp
    from run import _surf

    def tf(df):
        return sp.hstack([w.transform(text_of(df)), c.transform(text_of(df)),
                          sp.csr_matrix(ss.transform(_surf(df)))]).tocsr()
    names = [f"w:{t}" for t in w.get_feature_names_out()] + [f"c:{t!r}" for t in c.get_feature_names_out()]
    return m, tf, names


def main():
    m, tf, names = lr_model()
    coef = m.coef_[0]
    ebm = joblib.load(CACHE / f"11_ebm_full__{SCHEME}.joblib")
    X, y, df = features(SCHEME, SCHEME, "test")
    p = ebm.predict_proba(X)[:, 1]
    cached = X["text_ngram_logodds"].to_numpy()
    lines = ["# Full-data EBM: explanation examples (sarc_random)", ""]

    imp = pd.read_csv(HERE / "results" / f"ebm_full__{SCHEME}__term_importance.csv")
    lines += ["## Global: EBM term importance (mean |contribution to log-odds|)", "", "| Term | Importance |", "|---|---:|"]
    lines += [f"| {r.term} | {r.importance:.3f} |" for r in imp.head(20).itertuples()]
    order = np.argsort(coef[: len(names)])
    lines += ["", "## Largest n-gram weights inside the text score (LR coefficients)", "",
              "Sarcastic direction: " + ", ".join(f"`{names[i]}` {coef[i]:+.2f}" for i in order[::-1][:20]),
              "", "Non-sarcastic direction: " + ", ".join(f"`{names[i]}` {coef[i]:+.2f}" for i in order[:20]), ""]

    # pick: most confident correct sarcastic, most confident correct non-sarcastic, a confident miss
    picks = {"Correct sarcastic (high confidence)": np.flatnonzero((y == 1) & (p > 0.9)),
             "Correct non-sarcastic (high confidence)": np.flatnonzero((y == 0) & (p < 0.1)),
             "Error: non-sarcastic predicted as sarcastic": np.flatnonzero((y == 0) & (p > 0.85))}
    rng = np.random.default_rng(0)
    idx = [int(rng.choice(v)) for v in picks.values()]
    local = ebm.explain_local(X.iloc[idx], y[idx])
    texts, parents = text_of(df), parent_of(df)
    for k, (title, pool) in enumerate(picks.items()):
        i = idx[k]
        d = local.data(k)
        contrib = sorted(zip(d["names"], d["scores"], d["values"]), key=lambda t: -abs(t[1]))[:8]
        row = tf(df.iloc[[i]])
        nz = row.indices[row.indices < len(names)]
        ng = sorted(((names[j], coef[j] * row[0, j]) for j in nz), key=lambda t: -abs(t[1]))[:8]
        lines += [f"## Example {k + 1}: {title}", "",
                  f"- Reply: `{texts.iloc[i][:200]}`", f"- Parent: `{parents.iloc[i][:200]}`",
                  f"- True label {y[i]}, EBM P(sarcastic) {p[i]:.3f} (intercept {d['extra']['scores'][0]:+.2f})", "",
                  "| EBM term | Value | Contribution to log-odds |", "|---|---|---:|"]
        lines += [f"| {n} | {v if isinstance(v, str) else round(float(v), 3)} | {s:+.3f} |" for n, s, v in contrib]
        lines += ["", f"Text score {cached[i]:+.2f} comes mainly from: " + ", ".join(f"`{n}` {s:+.2f}" for n, s in ng), ""]
    (HERE / "results" / "ebm_explanations.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
