# Split card

Produced by `pipeline/04_build_splits.py`, checked by `pipeline/05_verify_splits.py`.
Seed 3244, target ratios 70/15/15. Machine-readable results in
`results/splits/split_verification.json`.

---

## 1. The constraint that applies to every split: group by parent comment

SARC was built by taking a comment ending in `/s` and pairing it with a sibling
reply under the same parent that has no `/s`. That design is why SARC is harder
and more honest than a hashtag scrape — topic, thread and time are all held
constant between the two classes.

It also creates a trap. If the two siblings land on opposite sides of a split,
the parent text appears in both training and test, and any model that reads
`parent_clean` is looking at its own training data when it is scored.

So every split here groups on `parent_key` — a blake2b hash of the parent text —
rather than on rows. Rows with no parent each get a unique key; they are not one
thread, they are a lot of comments that lost their parent.

`enforce_group_disjoint()` runs afterwards as a backstop, with priority
train > val > test. The evaluation sets give way, so nothing the model trained on
can reappear at scoring time.

> This was not right the first time. The audit caught two cases:
> `sarc_temporal` had 340 parent threads spanning val and test, and
> `sarc_subreddit` had 2,518 spanning splits. The second one was the
> counter-intuitive part — grouping by subreddit does *not* imply grouping by
> parent, because the same parent text ("What's your favourite X?") recurs
> across communities. Fixing it cost 4,790 and 3,274 rows respectively.

---

## 2. The three primary splits

### `sarc_random` — the default

| | train | val | test |
|---|---:|---:|---:|
| Rows | 639,638 | 137,066 | 137,065 |
| Sarcastic | 50.91% | 50.91% | 51.07% |

Shuffle all 892,045 parent keys, then greedily pack each group into whichever
bucket is furthest from its target row count. This is steadier than flipping a
coin per group, where a few large groups can skew the ratios.

Use it for the three base models and all hyper-parameter tuning. Its test score
is the headline number and the only one comparable to published SARC work.

It is optimistic by construction: subreddits and years are shared between train
and test. That is the point — it measures how strong the model is within one
world, and the other two measure what survives a change of world.

### `sarc_temporal` — generalising across time

| | train | val | test |
|---|---:|---:|---:|
| Period | before 2016-01-01 | 2016 H1 | 2016 H2 |
| Rows | 484,447 | 182,441 | 242,091 |
| Sarcastic | 53.82% | 50.38% | 45.79% |

The class balance differs across the splits on purpose. The EDA found sarcasm
rate drifting from 58.6% in 2009 to 47.7% in 2016; a random split quietly leaks
that drift to the model. This split exists to measure it.

Take the model tuned on `sarc_random` and run it here unchanged. The drop is how
much of the in-domain score came from knowing the period.

A side benefit: this split has the lowest author overlap of the three (36%), so
it is also the closest thing available to an author holdout.

### `sarc_subreddit` — generalising across communities

| | train | val | test |
|---|---:|---:|---:|
| Rows | 639,838 | 135,123 | 135,534 |
| Sarcastic | 51.11% | 51.19% | 49.93% |
| Shared subreddits | 0 | 0 | 0 |

Same greedy packing as `sarc_random`, but the grouping key is `subreddit`
(13,970 of them), with the `parent_key` backstop applied afterwards.

This is the only split that can answer sub-question 3 honestly. Compare
with-subreddit and without-subreddit models here. Running that comparison on
`sarc_random` proves nothing, because the model sees the same communities twice
and only has to memorise their priors.

Context from the EDA: across the 268 subreddits with at least 500 comments, the
sarcasm rate spans 8.1% (r/RoastMe) to 79.1% (r/creepyPMs).

---

## 3. Supplementary splits

| Corpus | train | val | test | Origin of the split |
|---|---:|---:|---:|---|
| `news_headlines` | 19,862 | 4,256 | 4,257 | **Ours** — stratified 70/15/15 |
| `tweeteval_irony` | 2,853 | 950 | 780 | Official, kept as-is |
| `figlang_reddit` | 3,729 | 659 | 1,756 | Official test; val carved from train |
| `figlang_twitter` | 4,028 | 711 | 1,715 | Same |

`news_headlines` had to be re-split. All 26,602 unique headlines in the upstream
`test.json` also appear in `train.json` — that is not a split. We deduplicated to
28,375 rows and made a fresh stratified one. The `article_link` column was
dropped during cleaning; `theonion.com` versus `huffingtonpost.com` is a perfect
label leak.

`tweeteval_irony` keeps its official split so numbers stay comparable to the
published benchmark. In this project, though, its job is to be a pure test set:
train on SARC, evaluate here, don't train on it.

---

## 4. Leakage audit

```bash
python pipeline/05_verify_splits.py --audit-only
```

| Split set | Checks | Result |
|---|---:|---|
| `sarc_random` | 6 | pass |
| `sarc_temporal` | 6 | pass |
| `sarc_subreddit` | 9 | pass |
| `news_headlines` | 3 | pass |
| `tweeteval_irony` | 3 | pass |
| `figlang_reddit` | 3 | pass |
| `figlang_twitter` | 3 | pass |

Three families of check, all returning zero overlap:

1. Normalised text across train/val, train/test, val/test
2. Parent threads across the same three pairs
3. Subreddits across the same three pairs, for `sarc_subreddit` only

### Author overlap — a limitation, disclosed rather than fixed

| Split | Test authors also seen in train | Share of test authors |
|---|---:|---:|
| `sarc_random` | 78,836 | 84% |
| `sarc_subreddit` | 55,446 | 72% |
| `sarc_temporal` | 41,379 | 36% |

SARC has 254,321 authors across 913,769 rows. Disjoint authors would mean
gutting the corpus, so the audit reports this rather than failing on it. The
model may be picking up individual writing style as much as sarcasm, and the
write-up should say so. `sarc_temporal` is the closest available approximation
to an author holdout.

---

## 5. Reference floor

TF-IDF (1–2 grams, min_df=3, 200k features, sublinear) with untuned logistic
regression at C=1, capped at 200,000 training rows. This is a smoke test — it
confirms the files load and the data is learnable, and gives the Week 7 base
models something to beat. It is not the project baseline.

### Within each split set

Majority baseline here means fitting the majority class on train and applying it
to test.

| Split set | Acc | P | R | F1 | Majority |
|---|---:|---:|---:|---:|---:|
| `sarc_random` | 0.7060 | 0.7238 | 0.6862 | 0.7045 | 0.5107 |
| `sarc_subreddit` | 0.7017 | 0.7138 | 0.6720 | 0.6923 | 0.4993 |
| `sarc_temporal` | 0.6939 | 0.6555 | 0.6983 | 0.6763 | 0.4579 |
| `news_headlines` | 0.8344 | 0.8171 | 0.8393 | 0.8281 | 0.5248 |
| `figlang_twitter` | 0.6449 | 0.6024 | 0.7436 | 0.6656 | 0.4752 |
| `tweeteval_irony` | 0.6372 | 0.5354 | 0.6581 | 0.5904 | 0.3974 |
| `figlang_reddit` | 0.6054 | 0.6154 | 0.5498 | 0.5808 | 0.5028 |

The 0.706 on `sarc_random` lines up with published bag-of-words baselines on
SARC, which is reassuring in both directions: the cleaning did not make the task
artificially hard, and no leakage survived — a leak would have pushed this
number noticeably higher.

The spread across the three schemes is only 1.2 points (0.7060 → 0.6939). A
bag-of-words model does not lean much on community or period. A model that uses
the subreddit feature directly may well lose more, which is what Week 8 is for.

### Transferring off SARC

Here the majority column is the target corpus's own majority share, since the
source-domain prior is meaningless in a transfer setting. Note this is a
different definition from the table above.

| Target | Acc | F1 | Target majority | Gain |
|---|---:|---:|---:|---:|
| `sarc_random` (in-domain) | 0.7060 | 0.7045 | 0.5107 | +19.5 pts |
| `figlang_reddit` | 0.6686 | 0.6589 | 0.5028 | +16.6 pts |
| `figlang_twitter` | 0.6146 | 0.5374 | 0.5248 | +9.0 pts |
| `tweeteval_irony` | 0.6167 | 0.4390 | 0.6026 | +1.4 pts |
| `news_headlines` | 0.4609 | 0.3135 | 0.5248 | −6.4 pts |

Cross-platform transfer does not degrade gracefully — it fails. On news
headlines the model is worse than always guessing the majority class. On
human-annotated irony tweets it beats the target's majority share by 1.4 points,
with an F1 of 0.44. Only the same-platform corpus holds up.

This matches the EDA exactly: exclamation marks carry a 2.81× lift on Reddit and
0.25× on news headlines — the same cue pointing in opposite directions.

The proposal planned to "evaluate model performance on cross-platform data".
There is now a number for it, and the negative result is worth more than a
flattering positive one would have been. Write it up as a finding, not a defect.

---

## 6. Reproducing

```bash
python pipeline/04_build_splits.py     # rebuild every split
python pipeline/05_verify_splits.py    # audit + reference floor
```

Seed 3244 throughout. `04_build_splits.py` is deterministic: the same
`data/processed/` input always yields the same splits.
