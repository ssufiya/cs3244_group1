# Dataset Catalogue

Every source used by the project, why it is here, and what it costs to use.
All of them download without a Kaggle token — see `pipeline/00_download_datasets.py`.

---

## Primary corpus

### SARC — Self-Annotated Reddit Corpus (balanced split)

| | |
|---|---|
| **File** | `data/raw/primary/reddit_sarc/train-balanced-sarcasm.csv` (243 MB) |
| **Rows** | 1,010,826 → **913,769** after cleaning |
| **Source** | `huggingface.co/datasets/daniel2588/sarcasm` — a mirror of the Kaggle `danofer/sarcasm` file the proposal names |
| **Origin** | Khodak, Saunshi & Vodrahalli (2018), *A Large Self-Annotated Corpus for Sarcasm*, LREC |
| **Labelling** | Self-annotation: the author appended `/s` |
| **Columns** | `label comment author subreddit score ups downs date created_utc parent_comment` |

**Why this one.** It is the dataset the proposal is built around, and it is the
only candidate that carries *all three* of the variables the sub-questions need:
thread context (`parent_comment`), community (`subreddit`) and enough scale for
the linguistic-cue analysis. The balanced split also samples negatives as
**siblings under the same parent**, which controls topic, thread and time — this
is why length turns out not to be a shortcut, and it is what makes SARC a
harder and more honest benchmark than any hashtag scrape.

**Known limitations.**
* Ends 2016-12. A 2016-era model may be fitting 2016 memes and politics.
* `ups` / `downs` are the placeholder `-1` in most rows. Do not use them.
* Sarcasm rate drifts 11pp across 2009–2016 — split by time, not at random.
* ~2.9% of rows are texts that appear with both labels (see the cleaning report).

---

## Supplementary corpora

Grouped by the question each one is there to answer.

### A. Cross-domain: does it work on clean, edited prose?

**News Headlines Dataset for Sarcasm Detection** — `data/raw/supplementary/news_headlines/`

| | |
|---|---|
| **Rows** | 55,328 → **28,375** after cleaning |
| **Source** | `huggingface.co/datasets/raquiba/Sarcasm_News_Headline` |
| **Origin** | Misra & Arora (2023), *Sarcasm Detection using News Headline Dataset*, AI Open |
| **Labels** | TheOnion = sarcastic, HuffPost = not. Editorial, not crowd-sourced. |

Professionally written, no typos, no slang, no `/s` — the cleanest possible
contrast with Reddit. Named in the proposal.

> ⚠️ **The shipped split is 100% contaminated**: all 26,602 unique `test.json`
> headlines also appear in `train.json`. Concatenate and re-split yourself.
> The `article_link` column is a perfect label leak (`theonion.com` vs
> `huffingtonpost.com`) and is dropped by the cleaner.

### B. Cross-platform: does it transfer off Reddit?

**TweetEval — Irony (SemEval-2018 Task 3A)** — `data/raw/supplementary/tweeteval_irony/`

| | |
|---|---|
| **Rows** | 4,601 → **4,583** after cleaning |
| **Source** | `huggingface.co/datasets/cardiffnlp/tweet_eval`, config `irony` |
| **Origin** | Van Hee, Lefever & Hoste (2018), SemEval-2018 Task 3 |
| **Labels** | **Human-annotated** after hashtag harvesting |

**This is the recommended cross-platform evaluation set**, and it is the one
addition to the proposal's list. It is the only corpus here whose labels were
checked by people rather than inferred from a tag the author wrote, which makes
it the right yardstick for "how much of our model is learning sarcasm versus
learning the `/s` convention". It is a published benchmark, so results are
comparable to prior work.

> Note: 436 rows still contain `#irony`/`#sarcasm`/`#not`, enriched ~7× in the
> positive class. Stripped by the cleaner.

### C. Context ablation: is one parent turn enough?

**FigLang 2020 Sarcasm Shared Task** — `data/raw/supplementary/figlang2020_context/`

| | |
|---|---|
| **Rows** | 6,200 Reddit + 6,800 Twitter → **6,144 + 6,454** |
| **Source** | `huggingface.co/datasets/tasksource/figlang2020-sarcasm` |
| **Origin** | Ghosh, Vajpayee & Muresan (2020), FigLang@ACL shared task |
| **Structure** | Each row: `response` + `context` as a **list of prior turns** |

The reason to include it: SARC gives exactly one parent turn, so it can only
test "context vs no context". FigLang supplies the full chain (median 31–33
words of context on Reddit, 65–70 on Twitter), which lets sub-question 1 be
answered as a *curve* — 0 turns vs 1 vs n — instead of a single comparison.
That is a materially better answer to the question the proposal actually asks.

### D. Audited and rejected

**`nikesh66/Sarcasm-dataset`** — `data/raw/supplementary/twitter_sarcasm/`

Named in the proposal. 5,000 rows contain **240 unique texts**, generated from
a handful of templates (`"Can't wait for more of artists."` ×49). 4.5% survives
deduplication. It is synthetic, not collected, and cannot support a
cross-platform claim.

**Recommendation: drop it, and say so in the report.** Keeping the audit is
worth more than keeping the data — "we checked a proposed source and rejected
it with evidence" is a stronger methods section than quietly using it.

It is still downloaded and cleaned so the finding is reproducible.

---

## Lexicon resources (LIWC alternatives)

`data/raw/external/lexicons/` — all free, all named in or adjacent to the proposal.

| file | what it gives | use |
|---|---|---|
| `empath_categories.tsv` | 194 psycholinguistic word categories | the free LIWC substitute the proposal asks for |
| `vader_lexicon.txt` | valence + intensity per token, incl. emoticons | sentiment polarity, and *booster* words for hyperbole |
| `nrc_emotion_lexicon.txt` | 8 emotions + 2 sentiments per word | sentiment **incongruity** features (positive words in a negative thread) |
| `SentiWordNet_3.0.0.txt` | per-synset positive/negative scores | the "more general word net" the proposal links to |

The incongruity feature these enable — positive-valence text under a
negative-valence parent — is the single most-cited engineered feature in the
sarcasm literature and is the natural candidate for sub-question 2.

---

## Scrapers (provided, not run)

| script | status | verdict |
|---|---|---|
| `pipeline/scrapers/scrape_twitter_sarcasm.py` | needs `TWITTER_BEARER_TOKEN` | **not recommended** — free tier is ~100 posts/month; `tweeteval_irony` answers the same question better and is citable |
| `pipeline/scrapers/scrape_reddit_sarcasm.py` | needs Reddit API app (free) | **worth running** if time allows — gives a 2024-25 *temporal* holdout that SARC cannot provide |

Both scripts strip label markers at write time and keep the raw text in a
separate audit-only field. The Reddit scraper samples negatives as siblings
under the same parent, mirroring SARC's construction, and hashes author names
by default.

---

## Provenance

`data/MANIFEST.json` records URL, byte size, SHA256, role and tags for all
15 downloaded files (278 MB total).
