# Error analysis — LightGBM, comment by comment

Person 4 · CS3244 Group 1
Script: [`modelling/gradient_boosting/error_analysis.py`](../modelling/gradient_boosting/error_analysis.py) ·
Outputs: [`results/gbdt/error_analysis/`](../results/gbdt/error_analysis/) ·
Main results: [`gbdt_report.md`](gbdt_report.md)

---

## Summary

We read the LightGBM model's mistakes one comment at a time, on all seven test
sets, and looked for the reasons behind them. Five patterns explain most errors:

1. **Many confident "false positives" are really sarcastic.** In a hand-read
   sample of 40 confident false positives on `sarc_random`, 25 (62%, 95% CI
   47–76%) read as sarcastic ("Yeah, because it's so hard to know..."). The
   author just did not add `/s`, so the label says *not sarcastic*. Part of the
   measured error is label noise.
2. **The model trusts a few cues too much.** Among comments with `!`, it calls
   84% sarcastic when 75% are; a sincere comment with `!` becomes a false
   positive 58% of the time (24% overall). Mock-agreement words (*obviously,
   clearly, sure, yeah, because*) and topic words (*racist, women, misogyny*)
   cause the same kind of false alarm.
3. **Sarcasm without cues is missed.** Missed sarcastic comments rarely carry
   the model's cues (`!` in 3% of them vs 17% of the sarcastic comments it
   catches). In the hand-read sample, 26 of 40 confident misses (65%, CI
   50–78%) need the parent comment or world knowledge to be seen as sarcastic
   ("Good luck finding a t-shirt store in Brooklyn").
4. **Sincere-sounding words win.** *iirc, imo, unfortunately, sadly, lol* push
   hard towards *not sarcastic* (up to −2.2 log-odds), even inside a joke.
5. **Off Reddit, the cues break.** Twitter sarcasm uses different cues, and
   Twitter users use `!` sincerely. A preprocessing mismatch adds to this: the
   `<USER>` placeholder in 99% of tweets is read as the word *user* and as an
   extra ALL-CAPS word, and it hides interjections at the start of a tweet.
   On news headlines, The Onion's satire is deadpan (no cues), while real
   headlines contain topic words the Reddit model links to sarcasm, so the
   ranking is worse than chance (ROC-AUC 0.45).

About a third of the errors are close calls: 35% of `sarc_random` errors have a
predicted probability between 0.4 and 0.6, and only 15% are confident (below
0.2 or above 0.8).

We also found a pipeline issue: `n_urls`, `n_mentions` and `n_hashtags` are 0 on
every row of every dataset, so 3 of the 17 surface features carry no
information (§6.3).

---

## 1. Method

- **Models.** LightGBM on reply TF-IDF plus the 17 surface counts (our best
  model, `reply_tfidf_surface`). `sarc_random`, `sarc_subreddit` and
  `sarc_temporal` are each scored by the model trained on their own training
  partition. The four transfer sets are scored by the `sarc_random` model,
  unchanged.
- **No retraining.** We reuse the saved test predictions and join them back to
  the text. Every row's ID hash and label match, and re-scoring the example
  rows gives exactly the saved probabilities (`checks.json`).
- **Threshold 0.50.** Each comment is a TP, TN, FP or FN. *Positive* means
  *sarcastic*.
- **Slices.** Error, false-positive and false-negative rates by length,
  punctuation, cue words, and how many of the comment's terms (words or word
  pairs) are in the model's vocabulary (`slices.csv`).
- **Why each comment was predicted this way.** Saabas path attributions in
  log-odds, the same method as `gbdt_report.md` §6. Positive values push
  towards *sarcastic*.
- **What separates errors from correct predictions.** For each test set we
  compare the average attribution of every feature between false positives
  and true negatives (both truly sincere), and between false negatives and
  true positives (both truly sarcastic). The features with the largest gap are
  what drive the errors (`error_drivers.csv`; up to 2,000 random rows per group).
- **Examples by a fixed rule.** Per test set: the 2 most confident TPs and TNs,
  and the 3 most confident plus 2 random FPs and FNs. We only consider
  comments of 4–35 words with at least one known term and no slurs or common
  profanity (98 comments in `examples.csv`).
- **Hand reading.** 40 random confident FPs (p ≥ 0.9) and 40 random confident
  FNs (p ≤ 0.1) from `sarc_random`, each sorted into a category by one reader
  (`manual_review.csv`). One reader is subjective, so treat the counts as rough;
  we give 95% Wilson confidence intervals.

---

## 2. Overview

| Test set | Model trained on | Rows | TP | TN | FP | FN | Accuracy | Truly sarcastic | Predicted sarcastic |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sarc_random` | `sarc_random` | 137,065 | 49,160 | 51,193 | 15,872 | 20,840 | 0.732 | 51.1% | 47.4% |
| `sarc_subreddit` | `sarc_subreddit` | 135,534 | 46,742 | 51,909 | 15,949 | 20,934 | 0.728 | 49.9% | 46.3% |
| `sarc_temporal` | `sarc_temporal` | 242,091 | 77,134 | 96,947 | 34,297 | 33,713 | 0.719 | 45.8% | 46.0% |
| `figlang_reddit` | `sarc_random` | 1,756 | 602 | 602 | 281 | 271 | 0.686 | 49.7% | 50.3% |
| `figlang_twitter` | `sarc_random` | 1,715 | 313 | 693 | 207 | 502 | 0.587 | 47.5% | 30.3% |
| `tweeteval_irony` | `sarc_random` | 780 | 130 | 342 | 128 | 180 | 0.605 | 39.7% | 33.1% |
| `news_headlines` | `sarc_random` | 4,257 | 495 | 1,552 | 682 | 1,528 | 0.481 | 47.5% | 27.6% |

On `sarc_random` and `sarc_subreddit` the model makes about 30% more FNs than
FPs; on `sarc_temporal` the two are about equal. `figlang_reddit` is balanced
too. On Twitter and news headlines the model calls far too few texts
sarcastic, so most sarcastic ones are missed.

---

## 3. Where the errors are (slices)

`sarc_random` test. FP rate = share of sincere comments called sarcastic; FN
rate = share of sarcastic comments missed.

| Slice | Share of rows | Truly sarcastic | Predicted sarcastic | FP rate | FN rate |
|---|---:|---:|---:|---:|---:|
| All comments | 100% | 51% | 47% | 24% | 30% |
| 1–5 words | 22% | 45% | 36% | 18% | **43%** |
| 41+ words | 0.8% | 27% | 18% | 7% | **51%** |
| Contains `!` | 9% | 75% | **84%** | **58%** | 7% |
| Contains an interjection | 20% | 68% | **73%** | **41%** | 11% |
| Contains a mock-agreement word ¹ | 14% | 76% | **84%** | **54%** | 6% |
| Contains *lol* | 0.8% | 37% | 24% | 9% | **50%** |
| No known term ² | 0.6% | 26% | 2% | 2% | **98%** |
| 1–3 known terms ² | 9% | 40% | 26% | 14% | **56%** |

¹ *obviously, clearly, totally, sure, yeah, wow, because* ·
² terms = words or word pairs in the model's 20,000-term vocabulary

- **The model over-uses the cues.** Comments with `!` or mock-agreement words
  really are sarcastic more often (75–76%), so using them is right. But the
  model calls 84% of them sarcastic, 8–9 points too many, and more than
  half of the sincere ones among them become false positives. Part of this FP
  rate is label noise: these are exactly the comments where a missing `/s` is
  most likely (§4).
- **Little evidence means "not sarcastic".** Short comments and comments with
  few known terms are predicted sarcastic less often than they are, so the
  sarcastic ones among them are missed. With no known term at all (781
  comments), 98% of the sarcastic ones are missed.
- **The same pattern holds on the stress splits.** `sarc_subreddit` and
  `sarc_temporal` give nearly the same numbers (`slices.csv`). On
  `sarc_subreddit`, accuracy across the 15 largest unseen communities is fairly
  even, from 0.668 (r/CoDCompetitive) to 0.757 (r/SubredditDrama)
  (`group_errors.csv`). The `sarc_temporal` test set is all from 2016, so it
  cannot be split by year.
- **Off Reddit, `!` misleads more.** FP rate for comments with `!`: 48% on
  `figlang_twitter` (23% overall), 65% on `tweeteval_irony` (27% overall),
  74% on `figlang_reddit` (32% overall).

---

## 4. Hand reading of confident errors (`sarc_random`)

The sampling pools are 766 FPs with p ≥ 0.9 (5% of all FPs) and 414 FNs with
p ≤ 0.1 (2% of all FNs). These are the model's most confident mistakes, not
typical ones.

| Error | Category | Count / 40 | Share (95% CI) |
|---|---|---:|---:|
| FP | Reads as sarcastic: label probably missing `/s` | **25** | 62% (47–76%) |
| FP | Sincere, but uses a cue the model links to sarcasm (`!`, *obviously*, *sure*, *racist*) | 9 | 22% (12–38%) |
| FP | Unclear | 6 | |
| FN | Sarcastic, but no cue words: needs the parent or world knowledge | **26** | 65% (50–78%) |
| FN | Sarcastic, but sincere-sounding words dominate (*iirc, unfortunately, honestly*) | 4 | 10% (4–23%) |
| FN | Label doubtful (a joke or sincere, not clearly sarcastic) | 10 | 25% (14–40%) |

**Diagnosis.**

- **FP side, label noise.** SARC labels come from the author's own `/s`. When
  an author writes sarcasm without `/s`, the comment is labelled sincere, and a
  good model is counted as wrong. Examples: "Yeah, because it's so hard to know
  the general rough MSRP of RIAA CDs in the US." and "Oh no, better put in more
  ads to generate the lost revenue". So the measured precision is a lower
  bound, and part of the gap to 100% cannot be closed by any model.
  `tweeteval_irony` is the only test set whose labels were checked by people,
  so its errors are the most likely to be real model errors.
- **FN side, missing context.** Most confident misses are deadpan. You only see
  the sarcasm if you know the parent or the world. Examples:
  - "I agree, two Iranian tanks went by my house yesterday" (reply to a claim
    that Americans need assault weapons against an Iranian invasion)
  - "Good luck finding a t-shirt store in Brooklyn" (Brooklyn is full of shops)
  - "General Secretary Debbie Wasserman Schulz has decreed that the
    DNC-sanctioned debates shall not be online, nor televised…"

  Our parent-context block (parent words and length) added only +0.0017 AUC
  (`gbdt_report.md` §3), because a bag of words cannot compare the meaning of
  the reply with the parent.
- **Labels are noisy on the FN side too.** 10 of 40 confident misses do not
  clearly read as sarcastic (a joke, or a sincere remark), so some "misses" are
  also label problems.

---

## 5. What separates errors from correct predictions

Each row compares errors with correct predictions **of the same true class**
(`error_drivers.csv`). "12% vs 2%" means the feature is present in 12% of the
errors and 2% of the correct ones.

| Test set | False positives vs true negatives (both sincere) | False negatives vs true positives (both sarcastic) |
|---|---|---|
| `sarc_random` | `!` 12% vs 2%; interjection 23% vs 12%; starts with interjection 13% vs 4%; *because* 6% vs 2%; longer | `!` 3% vs 17%; interjection 8% vs 33%; starts with interjection 3% vs 20%; *because* 2% vs 10% |
| `sarc_subreddit` | `!` 12% vs 3%; interjection 24% vs 11%; longer | `!` 3% vs 18%; interjection 11% vs 35%; *because* 2% vs 10% |
| `sarc_temporal` | `!` 12% vs 2%; interjection 23% vs 9%; longer | `!` 3% vs 17%; interjection 9% vs 31%; *because* 2% vs 8% |
| `figlang_reddit` | `!` 21% vs 3%; starts with interjection 14% vs 4% | `!` 3% vs 27%; interjection 12% vs 28% |
| `figlang_twitter` | `!` 49% vs 16%; interjection 37% vs 18% | `!` 7% vs 36%; interjection 25% vs 49%; longer tweets |
| `tweeteval_irony` | `!` 37% vs 7% | `!` 6% vs 28%; *yay* 0% vs 8% |
| `news_headlines` | *women* 6% vs 0%; *obama* 4% vs 1%; *white* 3% vs 0% | *man* 8% vs 14%; *woman* 2% vs 6%; interjection 3% vs 8% |

- **On every Reddit and Twitter set, the same small group of cues decides the
  outcome in both directions.** Sincere comments that happen to contain `!` or
  an interjection become false positives; sarcastic comments without them
  become false negatives. The model has learned "sarcasm = these markers", and
  most of its errors are the comments that break that rule.
- **On news headlines the decisive features are topic words**, not markers:
  HuffPost headlines about women, race or Obama are called sarcastic. The
  Onion headlines it catches contain the Onion's "man / woman" set-up ("local
  man…") more often than the ones it misses (*man* 14% vs 8%).

Strong sincere words that override sarcasm (from the example attributions):
*iirc* (−2.2), *unfortunately* (−2.0), *sadly* (−1.7), *afaik* (−1.6),
*imo* (−0.9 to −1.6), *lmao* (−0.8). For example, "afaik tl;dr iirc" is a joke
reply to "What is a sign of an experienced Redditor?" (p = 0.006).

---

## 6. Transfer-specific diagnosis

### 6.1 Twitter: the `<USER>` placeholder

The cleaning pipeline replaces every @mention with `<USER>`. 99% of
`figlang_twitter` tweets (38% of `tweeteval_irony`) contain it, while SARC, the
training data, almost never does (0.03% of `sarc_random` test comments). It changes three inputs:

- **TF-IDF** reads it as the word *user*, which leans sincere (mean push −0.26
  log-odds on every tweet that has it).
- **Surface counts** were computed with it in: `USER` counts as a word and as an
  ALL-CAPS word. 99.8% of `figlang_twitter` tweets have an ALL-CAPS word as
  stored, but only 19.5% once the placeholders are removed.
- **"Starts with an interjection"** is 0 for every `figlang_twitter` tweet,
  because the tweet starts with `<USER>`. Without the placeholders, 10.6%
  start with one.

Test: we removed `<USER>` and `<URL>` and scored again with the same model, no
retraining (`placeholder_test.csv`). For the surface counts we used the
pipeline's own `surface_features`, shifting each stored count by the change the
removal causes.

| Test set | Placeholders | ROC-AUC | Accuracy | Predicted sarcastic | FN rate | FP rate |
|---|---|---:|---:|---:|---:|---:|
| `figlang_twitter` | kept (as saved) | 0.611 | 0.587 | 30.3% | 61.6% | 23.0% |
| `figlang_twitter` | removed from TF-IDF | 0.621 | 0.598 | 37.9% | 52.4% | 29.1% |
| `figlang_twitter` | removed from TF-IDF and surface counts | 0.624 | 0.599 | 40.5% | 49.6% | 31.4% |
| `tweeteval_irony` | kept (as saved) | 0.599 | 0.605 | 33.1% | 58.1% | 27.2% |
| `tweeteval_irony` | removed from TF-IDF | 0.588 | 0.594 | 35.5% | 56.5% | 30.2% |
| `tweeteval_irony` | removed from TF-IDF and surface counts | 0.587 | 0.586 | 36.3% | 56.5% | 31.5% |

Removing the placeholder moves predictions towards sarcastic: on
`figlang_twitter` the FN rate falls from 62% to 50%, but the FP rate rises from
23% to 31%. The ranking barely changes (AUC +0.013 on `figlang_twitter`, −0.012
on `tweeteval_irony`). So the placeholder explains part of the
under-prediction, not the main transfer problem. The main problem is that
Twitter sarcasm uses different words and cues (§5): sincere tweets use `!`
freely, and sarcastic tweets often have no Reddit-style marker.

### 6.2 News headlines: deadpan satire vs. topic words

ROC-AUC is 0.45 and accuracy 0.48: the ranking is worse than chance.
Sarcastic headlines here come from The Onion and sincere ones from HuffPost,
and both break the Reddit cues:

- **Missed satire (FN rate 75%).** Onion headlines copy the neutral style of
  real news. They have no `!`, interjections or "yeah because". Example: "former
  trump advisor carter page found dumb in d.c. hotel room" (p = 0.045), a play
  on the usual "found dead in hotel room" news line.
- **Real news called sarcastic.** HuffPost covers race, gender and politics,
  and those topic words lean sarcastic on Reddit. Example: "women in business
  q&a: star jones, president, professional diversity network" (p = 0.86,
  *women* +1.50).

### 6.3 Pipeline issue: three surface counts are always zero

`n_urls`, `n_mentions` and `n_hashtags` are 0 on every row of all seven test
sets, and none of the three models ever splits on them (`dead_features.csv`).
Before the surface counts are computed, the cleaner replaces links and
mentions with `<URL>` / `<USER>` and drops the `#` from hashtags (to remove
label hashtags such as #sarcasm), so the three patterns never match
(`pipeline/common/text_cleaning.py`, `clean_series`).
In effect the model has 14 surface features, not 17. This affects every slot
that uses `MODEL_SAFE_FEATURES`, not only ours.

---

## 7. Examples (one per outcome and test set)

p = predicted probability of *sarcastic*. Pushes are Saabas log-odds: + towards
sarcastic, − towards not sarcastic. All 98 examples, with the parent or
context text, are in `examples.csv`.

| Test set | Outcome | p | Comment | Main pushes | Diagnosis |
|---|---|---:|---|---|---|
| `sarc_random` | TP | 0.999 | Yes because everyone knows that taking vaccines makes you autistic | *vaccines* +1.70, *yes because* +1.23, *because* +1.10 | Classic "yes because" mock agreement |
| `sarc_random` | TN | 0.005 | I don't know, but this reminds me of the opening scene in Prometheus IIRC | *iirc* −2.17, *know but* −0.97 | Hedge words, sincere tone |
| `sarc_random` | FP | 0.998 | Misogyny Sexist Racist Bigot | *racist* +2.31, *misogyny* +1.69, *sexist* +1.60 | Sincere answer to "What word got so overused…?"; topic words only |
| `sarc_random` | FN | 0.020 | General Secretary Debbie Wasserman Schulz has decreed that the DNC-sanctioned debates shall not be online, nor televised, nor broadcast by radio but will be available in vinyl format for purchase beginning in 2062. | *but will* −0.85, length (33 words) −0.66, *nor* −0.44 | Long deadpan joke; no cue words |
| `sarc_subreddit` | TP | 0.999 | Yeah because all game devs are elite professional gamers | *because* +1.22, *yeah because* +1.09, *gamers* +0.88 | Mock agreement |
| `sarc_subreddit` | TN | 0.003 | It's kind of beside the point, but there are generally considered to be three Australian English accents… | *generally* −0.97, *it kind* −0.61, length −0.48 | Informative, long |
| `sarc_subreddit` | FP | 0.995 | yeah, because that detail makes all the difference. | *yeah because* +1.14, *because* +1.14 | Reads as sarcastic: label probably missing `/s` |
| `sarc_subreddit` | FN | 0.016 | I've yet to find any tainted lands, sadly. | *sadly* −1.71, *ve* −0.56 | *sadly* pulls it to sincere; the irony needs game knowledge |
| `sarc_temporal` | TP | 1.000 | Yeah, because every call of duty game is totally original! | *yeah because* +1.02, *call of* +0.92, interjections +0.64 | Several cues at once |
| `sarc_temporal` | TN | 0.004 | At first I thought this was the Time Traveler's Wife, then I realized it's "Claire" rather than "Clare" xD | *at first* −0.86, *first thought* −0.74 | Personal story |
| `sarc_temporal` | FP | 0.996 | Because clearly, no woman has ever told a single lie in her life. | *clearly* +0.85, *because* +0.82, *ever* +0.76 | Reads as sarcastic: label probably missing `/s` |
| `sarc_temporal` | FN | 0.014 | Not really I usually stay up till 333am and then sleep until 1:24pm | *not really* −1.00, *usually* −0.99 | Joke that only works with the parent ("trouble sleeping?"); sincere words pull it down |
| `figlang_reddit` | TP | 0.997 | Yeah who needs the EPA right? I'm sure those big businesses won't do anything that the common folk wouldn't appreciate! | *who needs* +0.72, interjections +0.71, `!` +0.64 | Several cues at once |
| `figlang_reddit` | TN | 0.020 | Honestly, I haven't in a long time. How are they? Still rampant cheaters or mostly just rage reports? Edit: What a depressing time. | *mostly* −0.88, *honestly* −0.60, *still* −0.55 | Sincere question |
| `figlang_reddit` | FP | 0.964 | Yeah, we all know cops aren't allowed to lie. | starts with interjection +0.49, *all* +0.48, *cops* +0.40 | Reads as sarcastic: label probably wrong |
| `figlang_reddit` | FN | 0.018 | Unfortunately, I've always hated Scooby-Doo. I won't apologize for it. | *unfortunately* −1.99, *ve* −0.47 | Sincere-sounding word dominates |
| `figlang_twitter` | TP | 0.975 | &lt;USER&gt; &lt;USER&gt; No no no You don't understand , Jesus isnt as important as this tweet | *no no* +1.22, *no you* +0.63 | Strong cue beats the *user* push (−0.14) |
| `figlang_twitter` | TN | 0.019 | &lt;USER&gt; &lt;USER&gt; I wish mate but many don't understand the platform which is sad , … | *sad* −0.48, *many* −0.44, *rarely* −0.43 | Sincere complaint |
| `figlang_twitter` | FP | 0.559 | &lt;USER&gt; &lt;USER&gt; &lt;USER&gt; Thank you to absolutely incredible AwesomeFriends for the this AwesomeList and I wish everyone an day &lt;URL&gt; | *thank* +0.60, interjection +0.52, *everyone* +0.45 | Sincere thanks; enthusiasm looks like sarcasm |
| `figlang_twitter` | FN | 0.026 | &lt;USER&gt; &lt;USER&gt; Pretty much . Full of brexiteers most likely | *pretty* −0.80, *likely* −0.70, *user* −0.56 | Dry jab; hedge words and *user* pull it down |
| `tweeteval_irony` | TP | 0.964 | Yay I just love this time of the month...! | *yay* +1.14, `!` +0.97, *just love* +0.90 | Cue words transfer |
| `tweeteval_irony` | TN | 0.028 | I won't be surprised if shelling across pak border intensifies during Delhi Elections to change issues of election's. | *intensifies* −0.84, *be surprised* −0.68 | Sincere news comment |
| `tweeteval_irony` | FP | 0.750 | Hey &lt;USER&gt; some of your geeky/nerdy followers need marker on posts like that. Otherwise, they'll start doing exactly the same :) | *hey* +1.37, *exactly the* +0.36 | Friendly interjection read as sarcasm |
| `tweeteval_irony` | FN | 0.035 | Decided I'd go to bed "early" for once. I've been laying here since 2 am. Yet the total amount of sleep I've had all week is 10 hrs. | *ve* −0.53, length −0.43 | Situational irony; no cue words |
| `news_headlines` | TP | 0.947 | trees planted in poor neighborhood mature just in time for gentrification | *mature* +1.70, *poor* +0.66 | Right, partly by luck (word *mature*) |
| `news_headlines` | TN | 0.038 | how to make salt-roasted fish with spicy orange salsa | *spicy* −1.50, *salt* −0.76 | Recipe headline |
| `news_headlines` | FP | 0.858 | women in business q&a: star jones, president, professional diversity network | *women* +1.50, *diversity* +0.68 | Topic words from Reddit |
| `news_headlines` | FN | 0.045 | former trump advisor carter page found dumb in d.c. hotel room | *hotel* −0.58, *former* −0.41 | Deadpan satire in news style |

---

## 8. What this means

- **Report precision with a caveat.** A large share of confident false
  positives look like real sarcasm without `/s`. The true precision on SARC is
  likely higher than measured, and no model can reach 100% on these labels.
- **The next gain needs context, not more cues.** Most confident misses need
  the parent or world knowledge, and most errors come from the few marker cues
  the model relies on. A model that reads the reply and the parent together
  (for example a fine-tuned transformer with both texts as input) attacks this
  directly. More bag-of-words features will not.
- **Fix the preprocessing.** Compute the surface counts before replacing URLs,
  mentions and hashtags (this revives three dead features), and strip
  placeholders such as `<USER>` / `<URL>` before TF-IDF. This is cheap, but it
  will not fix transfer alone (§6.1).
- **For transfer, adapt to the target.** Be careful with surface cues like `!`
  off Reddit (the surface block already costs 0.02–0.03 AUC on Twitter,
  `gbdt_report.md` §5). Headlines need some target-domain training data,
  because the Reddit cues point the wrong way there.

## Limitations

- The hand reading is one reader on 80 comments, drawn only from the most
  confident errors, so the categories and counts are rough (see the
  confidence intervals in §4).
- Examples skip comments with slurs or common profanity (a regular-expression
  filter; unusual spellings can slip through) and keep 4–35 words. This makes
  them readable but not fully representative.
- Saabas attributions are not exact Shapley values (`gbdt_report.md` §6).
  They show the direction and rough size of each push.
- The placeholder test keeps the trained model. A model retrained on
  placeholder-free inputs could behave differently.

## Reproducing

```bash
python modelling/gradient_boosting/error_analysis.py   # a few minutes on a CPU
```

It needs the saved models in `models/gbdt/` and predictions in
`results/gbdt/predictions/` (both git-ignored; run the notebook first if they
are missing). The hand labels in `manual_review.csv` are not produced by the
script: it writes `review_sample_raw.csv` with the same 80 comments to label.
