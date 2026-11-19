"""
Instance-level error analysis for the LightGBM baseline (text + surface).

Person 4 · CS3244 Group 1

Nothing is retrained. The script reads the saved test-set predictions in
`results/gbdt/predictions/`, joins them back to the comment text, and:

1. checks that every saved prediction matches its row (row_id hash and label);
2. labels every row TP / TN / FP / FN at threshold 0.50;
3. measures error rates on simple slices (length, cue words, punctuation,
   how many words the model knows, ...) to find *where* the model fails;
4. picks example comments by a fixed rule and explains each one with Saabas
   path attributions (the same method as `docs/gbdt_report.md` §6).

Models: `sarc_random`, `sarc_subreddit` and `sarc_temporal` are each scored by
the LightGBM model trained on their own training partition. The four transfer
sets are scored by the `sarc_random` model, unchanged.

Run from the repository root:

    python modelling/gradient_boosting/error_analysis.py

Outputs go to `results/gbdt/error_analysis/`:

    confusion.csv        TP/TN/FP/FN and accuracy per test set
    slices.csv           error, FP and FN rates on simple slices
    error_drivers.csv    features that separate errors from correct predictions
                         of the same true class (FP vs TN, FN vs TP; Saabas)
    examples.csv         example comments by a fixed rule, with their top features
    placeholder_test.csv Twitter: score again without '<USER>'/'<URL>' (no retraining)
    dead_features.csv    surface counts that are constant on a test set
    group_errors.csv     accuracy by community (sarc_subreddit) and year (sarc_temporal)
    checks.json          saved vs recomputed probabilities; Saabas sums
    review_sample_raw.csv  80 random confident sarc_random errors to read by hand
                           (our hand-labelled copy is manual_review.csv)

Report: docs/gbdt_error_analysis.md
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401  (needed to unpickle the builders)

ROOT = Path(os.environ.get("CS3244_ROOT", Path(__file__).resolve().parents[2]))
SPLITS = ROOT / "data" / "splits"
MODELS = ROOT / "models" / "gbdt"
PRED = ROOT / "results" / "gbdt" / "predictions"
OUT = ROOT / "results" / "gbdt" / "error_analysis"

REP = "reply_tfidf_surface"
THRESHOLD = 0.50
SEED = 3244
SURFACE = ["n_chars", "n_words", "n_sentences", "avg_word_len",
           "n_exclam", "n_question", "n_ellipsis", "n_repeat_punct",
           "n_allcaps_words", "allcaps_ratio", "n_elongation", "n_scare_quotes",
           "n_urls", "n_mentions", "n_hashtags",
           "n_interjections", "starts_with_interjection"]

# (test set, model that scores it)
RUNS = [("sarc_random", "sarc_random"),
        ("sarc_subreddit", "sarc_subreddit"),
        ("sarc_temporal", "sarc_temporal"),
        ("figlang_reddit", "sarc_random"),
        ("figlang_twitter", "sarc_random"),
        ("tweeteval_irony", "sarc_random"),
        ("news_headlines", "sarc_random")]

# Words the EDA and the attribution analysis flagged (gbdt_report.md §6)
MOCK_WORDS = ["obviously", "clearly", "totally", "sure", "yeah", "wow", "because"]
# Examples shown to teammates skip comments with slurs or strong profanity
BLOCK = re.compile(r"\b(?:fuck\w*|shit\w*|cunt\w*|nigg\w*|fag\w*|retard\w*|bitch\w*|whore\w*|"
                   r"slut\w*|rap(?:e|ed|ist)|kys|dick\w*|cock\w*|puss(?:y|ies)|asshole\w*|kill yourself)\b", re.I)


# --------------------------------------------------------------------------
# The feature builder class must exist under __main__ to unpickle the saved
# builders (they were saved from the notebook). Only `transform` is needed.
class FeatureBuilder:
    def transform(self, df):
        blocks = [self.reply_.transform(text_of(df))]
        if self.surface:
            blocks.append(sparse.csr_matrix(df[SURFACE].to_numpy(np.float32)))
        return sparse.hstack(blocks, format="csr", dtype=np.float32)

    def feature_names(self):
        names = ["tok:" + t for t in self.reply_.get_feature_names_out()]
        return names + (list(SURFACE) if self.surface else [])


sys.modules["__main__"].FeatureBuilder = FeatureBuilder


def text_of(df):
    col = "comment_clean" if "comment_clean" in df else "text_clean"
    return df[col].fillna("").astype(str)


def row_ids(df):
    return text_of(df).map(lambda s: hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest())


def load_test(dataset):
    primary = dataset.startswith("sarc_")
    folder = SPLITS / ("primary" if primary else "supplementary") / dataset
    cols = ["comment_clean" if primary else "text_clean", "label"] + SURFACE
    cols += ["parent_clean", "subreddit", "year", "parent_is_empty"] if primary else ["context_clean"]
    return pd.read_parquet(folder / "test.parquet", columns=cols).reset_index(drop=True)


# --------------------------------------------------------------------------
class Saabas:
    """Saabas path attributions for a LightGBM booster (same maths as the
    notebook's `lgbm_saabas`; the tree dump is cached so many rows are cheap).
    Each split's change in node value is credited to the split's feature; the
    root values form the bias, so a row's values sum to its raw score."""

    def __init__(self, booster):
        self.booster = booster
        self.trees = booster.dump_model()["tree_info"]

    def __call__(self, x):
        n, n_features = x.shape
        leaves = self.booster.predict(x, pred_leaf=True)
        contrib = np.zeros((n, n_features + 1))
        for t, tree in enumerate(self.trees):
            root = tree["tree_structure"]
            if "leaf_index" in root:
                contrib[:, n_features] += root["leaf_value"]
                continue
            contrib[:, n_features] += root["internal_value"]
            col = leaves[:, t]
            order = np.argsort(col, kind="stable")
            bounds = np.searchsorted(col[order], np.arange(tree["num_leaves"] + 1))
            stack = [(root, [], [])]
            while stack:
                node, feats, deltas = stack.pop()
                if "leaf_index" in node:
                    rows = order[bounds[node["leaf_index"]]:bounds[node["leaf_index"] + 1]]
                    if len(rows) and feats:
                        f, inv = np.unique(np.array(feats), return_inverse=True)
                        contrib[np.ix_(rows, f)] += np.bincount(inv, weights=np.array(deltas))
                    continue
                for child in (node["left_child"], node["right_child"]):
                    value = child["leaf_value"] if "leaf_index" in child else child["internal_value"]
                    stack.append((child, feats + [node["split_feature"]],
                                  deltas + [value - node["internal_value"]]))
        return contrib


def describe(contrib_row, x_row, names, k=4):
    """Top-k features pushing towards sarcastic and towards sincere."""
    c = contrib_row[:-1]
    x = x_row.toarray().ravel()

    def fmt(i):
        name = names[i]
        if name.startswith("tok:"):
            label = f'"{name[4:]}"' if x[i] else f'no "{name[4:]}"'
        else:
            v = x[i]
            label = f"{name}={int(v) if float(v).is_integer() else round(float(v), 2)}"
        return f"{label} ({c[i]:+.2f})"

    pos = [fmt(i) for i in np.argsort(-c)[:k] if c[i] > 0.02]
    neg = [fmt(i) for i in np.argsort(c)[:k] if c[i] < -0.02]
    return "; ".join(pos), "; ".join(neg)


# --------------------------------------------------------------------------
def slice_table(df, dataset):
    """Error rate and FP/FN rates on simple, interpretable slices."""
    words = df["n_words"]
    text = text_of(df).str.lower()
    slices = {
        "all": np.ones(len(df), bool),
        "length 1-5 words": words.between(1, 5),
        "length 6-10 words": words.between(6, 10),
        "length 11-20 words": words.between(11, 20),
        "length 21-40 words": words.between(21, 40),
        "length 41+ words": words >= 41,
        "has '!'": df["n_exclam"] > 0,
        "has interjection": df["n_interjections"] > 0,
        "has mock-agreement word": text.str.contains(r"\b(?:" + "|".join(MOCK_WORDS) + r")\b", regex=True),
        "has 'lol'": text.str.contains(r"\blol\b", regex=True),
        "known words: 0": df["known_terms"] == 0,
        "known words: 1-3": df["known_terms"].between(1, 3),
        "known words: 4+": df["known_terms"] >= 4,
    }
    if "n_mentions" in df and df["n_mentions"].gt(0).any():
        slices["has @mention"] = df["n_mentions"] > 0
    if df["n_hashtags"].gt(0).any():
        slices["has #hashtag"] = df["n_hashtags"] > 0
    if "parent_is_empty" in df:
        slices["no parent comment"] = df["parent_is_empty"] == 1
    rows = []
    for name, m in slices.items():
        m = np.asarray(m, bool)
        d = df[m]
        if len(d) == 0:
            continue
        pos, neg = d.label == 1, d.label == 0
        rows.append({"dataset": dataset, "slice": name, "rows": int(m.sum()),
                     "share_of_rows": m.mean(),
                     "sarcastic_rate": d.label.mean(),
                     "predicted_sarcastic_rate": d.pred.mean(),
                     "error_rate": (d.pred != d.label).mean(),
                     "fp_rate": (d.pred[neg] == 1).mean() if neg.any() else np.nan,
                     "fn_rate": (d.pred[pos] == 0).mean() if pos.any() else np.nan})
    return rows


def pick_examples(df, rng):
    """Fixed rule: comments of 4-35 words without slurs/strong profanity.
    TP/TN: the 2 most confident. FP/FN: the 3 most confident + 2 random."""
    ok = df["n_words"].between(4, 35) & ~text_of(df).str.contains(BLOCK) & df.known_terms.gt(0)
    d = df[ok]
    picks = []
    for outcome, n_conf, n_rand in (("TP", 2, 0), ("TN", 2, 0), ("FP", 3, 2), ("FN", 3, 2)):
        g = d[d.outcome == outcome]
        if g.empty:
            continue
        g = g.sort_values("confidence", ascending=False)
        conf = g.head(n_conf).assign(pick="most confident")
        rest = g.iloc[n_conf:]
        rand = rest.sample(min(n_rand, len(rest)), random_state=rng).assign(pick="random") if n_rand else rest.head(0)
        picks.append(pd.concat([conf, rand]))
    return pd.concat(picks)


# --------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    confusion, slices, examples, checks = [], [], [], []
    drivers, placeholder, groups, dead = [], [], [], []
    cache = {}
    for dataset, trained_on in RUNS:
        df = load_test(dataset)
        pred = pd.read_parquet(PRED / f"{trained_on}__lgbm_{REP}__{dataset}__test.parquet")
        assert len(pred) == len(df) and (pred.row_position.to_numpy() == np.arange(len(df))).all()
        ids_ok = bool((row_ids(df).to_numpy() == pred.row_id.to_numpy()).all())
        labels_ok = bool((df.label.to_numpy() == pred.label.to_numpy()).all())
        assert ids_ok and labels_ok, f"{dataset}: saved predictions do not line up with the split"

        if trained_on not in cache:
            booster = lgb.Booster(model_file=str(MODELS / f"{trained_on}__lgbm_{REP}.txt"))
            builder = joblib.load(MODELS / f"{trained_on}__{REP}__builder.joblib")
            cache[trained_on] = (booster, builder, Saabas(booster), builder.feature_names())
        booster, builder, saabas, names = cache[trained_on]

        df["p"] = pred.probability.to_numpy()
        df["pred"] = (df.p >= THRESHOLD).astype(int)
        df["outcome"] = np.select(
            [(df.label == 1) & (df.pred == 1), (df.label == 0) & (df.pred == 0),
             (df.label == 0) & (df.pred == 1)], ["TP", "TN", "FP"], "FN")
        df["confidence"] = (df.p - 0.5).abs()
        df["correct"] = (df.pred == df.label).astype(int)
        df["known_terms"] = np.diff(builder.reply_.transform(text_of(df)).indptr)

        c = df.outcome.value_counts()
        err = df[df.outcome.isin(["FP", "FN"])]
        confusion.append({"dataset": dataset, "model_trained_on": trained_on, "rows": len(df),
                          "TP": int(c.get("TP", 0)), "TN": int(c.get("TN", 0)),
                          "FP": int(c.get("FP", 0)), "FN": int(c.get("FN", 0)),
                          "accuracy": (df.pred == df.label).mean(),
                          "sarcastic_rate": df.label.mean(),
                          "predicted_sarcastic_rate": df.pred.mean(),
                          "errors_with_p_0.4_to_0.6": err.p.between(0.4, 0.6).mean(),
                          "errors_with_p_beyond_0.2_0.8": ((err.p >= 0.8) | (err.p <= 0.2)).mean(),
                          "mean_known_terms": df.known_terms.mean()})
        slices += slice_table(df, dataset)

        ex = pick_examples(df, np.random.RandomState(SEED)).copy()
        x = builder.transform(ex)
        contrib = saabas(x)
        raw = booster.predict(x, raw_score=True)
        p_again = booster.predict(x)
        checks.append({"dataset": dataset,
                       "max_prob_diff_vs_saved": float(np.abs(p_again - ex.p.to_numpy()).max()),
                       "max_saabas_sum_error": float(np.abs(contrib.sum(1) - raw).max())})
        for i, (_, r) in enumerate(ex.iterrows()):
            pos, neg = describe(contrib[i], x[i], names)
            examples.append({"dataset": dataset, "model_trained_on": trained_on,
                             "outcome": r.outcome, "pick": r.pick,
                             "label": "sarcastic" if r.label else "not sarcastic",
                             "p_sarcastic": round(float(r.p), 3),
                             "text": text_of(ex).iloc[i],
                             "context": (r.get("parent_clean") if "parent_clean" in ex else r.get("context_clean")) or "",
                             "subreddit": r.get("subreddit", ""),
                             "known_terms": int(r.known_terms),
                             "pushes_to_sarcastic": pos, "pushes_to_not_sarcastic": neg})
        print(f"{dataset:16s} rows={len(df):7d}  acc={confusion[-1]['accuracy']:.4f}  "
              f"ids_ok={ids_ok} labels_ok={labels_ok}  examples={len(ex)}", flush=True)

        drivers += error_drivers(df, builder, saabas, names, dataset)
        splits = dict(zip(names, booster.feature_importance("split")))
        dead += [{"dataset": dataset, "feature": c, "constant_value": float(df[c].iloc[0]),
                  "splits_on_it_in_model": int(splits[c])}
                 for c in SURFACE if df[c].nunique() == 1]
        if dataset == "sarc_random":
            review_sample(df, names)
        if dataset in ("figlang_twitter", "tweeteval_irony"):
            placeholder += placeholder_test(df, booster, builder, saabas, names, dataset)
        if dataset == "sarc_subreddit":
            groups += group_errors(df, "subreddit", dataset, top=15)
        if dataset == "sarc_temporal":
            groups += group_errors(df, "year", dataset)

    pd.DataFrame(confusion).to_csv(OUT / "confusion.csv", index=False)
    pd.DataFrame(slices).to_csv(OUT / "slices.csv", index=False)
    pd.DataFrame(examples).to_csv(OUT / "examples.csv", index=False)
    pd.DataFrame(drivers).to_csv(OUT / "error_drivers.csv", index=False)
    pd.DataFrame(placeholder).to_csv(OUT / "placeholder_test.csv", index=False)
    pd.DataFrame(groups).to_csv(OUT / "group_errors.csv", index=False)
    pd.DataFrame(dead).to_csv(OUT / "dead_features.csv", index=False)
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")


def error_drivers(df, builder, saabas, names, dataset, n=2000, k=8):
    """What separates the errors from the correct predictions of the same true class.

    Sincere comments: FP vs TN. Sarcastic comments: FN vs TP. For every feature
    we compare its mean Saabas push in the two groups; the features with the
    largest gap in the wrong direction are what drive the errors. Up to `n`
    random rows per group (all rows on the small test sets)."""
    rows = []
    names = np.array(names)
    for err, ok, sign in (("FP", "TN", 1), ("FN", "TP", -1)):
        groups = {}
        for o in (err, ok):
            g = df[df.outcome == o]
            if len(g) > n:
                g = g.sample(n, random_state=SEED)
            x = builder.transform(g)
            groups[o] = (saabas(x)[:, :-1], np.asarray((x != 0).mean(0)).ravel(), len(g))
        gap = (groups[err][0].mean(0) - groups[ok][0].mean(0)) * sign   # > 0: pushes the errors the wrong way
        for i in np.argsort(-gap)[:k]:
            rows.append({"dataset": dataset, "error_type": err, "compared_with": ok,
                         "feature": names[i].replace("tok:", ""),
                         "gap_in_wrong_direction": round(float(gap[i]), 4),
                         "mean_push_in_errors": round(float(groups[err][0][:, i].mean()), 4),
                         "mean_push_in_correct": round(float(groups[ok][0][:, i].mean()), 4),
                         "share_nonzero_in_errors": round(float(groups[err][1][i]), 4),
                         "share_nonzero_in_correct": round(float(groups[ok][1][i]), 4),
                         "rows_errors": groups[err][2], "rows_correct": groups[ok][2]})
    return rows


def review_sample(df, names, n=40):
    """Random confident errors from sarc_random for a manual read."""
    pick = []
    for outcome, cond in (("FP", df.p >= 0.9), ("FN", df.p <= 0.1)):
        g = df[(df.outcome == outcome) & cond]
        pick.append(g.sample(n, random_state=SEED).assign(pool_size=len(g)))
    out = pd.concat(pick)
    out = out.assign(text=text_of(out))[["outcome", "p", "label", "subreddit", "parent_clean", "text", "pool_size"]]
    out.to_csv(OUT / "review_sample_raw.csv", index=False)


def placeholder_test(df, booster, builder, saabas, names, dataset):
    """Twitter text keeps the cleaner's '<USER>' and '<URL>' placeholders.
    TF-IDF reads them as the words 'user' and 'url', and the surface counts
    were computed with them in ('USER' counts as a word and an ALL-CAPS word).
    SARC, the training data, almost never contains them. Score again (same
    model, no retraining) with the placeholders removed (a) from the TF-IDF
    input only and (b) from the TF-IDF input and the surface counts.

    For (b) each count is shifted by the change the pipeline's own
    `surface_features` gives when the placeholders are removed, so the stored
    definitions are kept and only the placeholder effect is taken out."""
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, str(ROOT / "pipeline"))
    from common.text_cleaning import surface_features
    rows = []
    x = builder.transform(df)
    has_user = text_of(df).str.contains("<USER>").to_numpy()
    j = names.index("tok:user")
    contrib_user = saabas(x[has_user])[:, j] if has_user.any() else np.array([0.0])
    col = "comment_clean" if "comment_clean" in df else "text_clean"
    stripped = text_of(df).str.replace(r"<USER>|<URL>", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    text_only = df.copy()
    text_only[col] = stripped
    both = text_only.copy()
    before = pd.DataFrame([surface_features(t) for t in text_of(df)]).astype(float)
    after = pd.DataFrame([surface_features(t) for t in stripped]).astype(float)
    for c in SURFACE:
        both[c] = df[c].astype(float).to_numpy() + (after[c] - before[c]).to_numpy()
    versions = (("as saved (with placeholders)", df),
                ("removed from TF-IDF only", text_only),
                ("removed from TF-IDF and surface counts", both))
    for version, frame in versions:
        p = booster.predict(builder.transform(frame))
        pred = (p >= THRESHOLD).astype(int)
        y = frame.label.to_numpy()
        rows.append({"dataset": dataset, "version": version,
                     "share_with_USER": has_user.mean(),
                     "share_with_allcaps_word": float((frame.n_allcaps_words > 0).mean()),
                     "mean_push_of_word_user": float(contrib_user.mean()),
                     "roc_auc": roc_auc_score(y, p), "accuracy": (pred == y).mean(),
                     "predicted_sarcastic_rate": pred.mean(),
                     "fn_rate": (pred[y == 1] == 0).mean(), "fp_rate": (pred[y == 0] == 1).mean()})
    return rows


def group_errors(df, key, dataset, top=None):
    g = df.groupby(key, observed=True).agg(rows=("label", "size"), sarcastic_rate=("label", "mean"),
                                           predicted_sarcastic_rate=("pred", "mean"),
                                           accuracy=("correct", "mean")).reset_index()
    if top:
        g = g.sort_values("rows", ascending=False).head(top)
    g.insert(0, "dataset", dataset)
    return g.rename(columns={key: "group"}).assign(group_by=key).to_dict("records")


if __name__ == "__main__":
    main()
