# Data dictionary

There are two schemas. The **primary** one covers the SARC splits under
`data/splits/primary/`; the **supplementary** one covers everything under
`data/splits/supplementary/`. The surface-feature columns are named and
computed identically in both, so the corpora can be compared directly.

`MODEL_SAFE_FEATURES` in `pipeline/load.py` is exactly the list of columns
marked usable below.

Legend: **usable** as a feature · **careful** · **never** · **key/metadata**

---

## Primary schema (SARC) — 47 columns

### Label and text

| Column | Type | Description | |
|---|---|---|---|
| `label` | int8 | Target. 1 = sarcastic (author wrote `/s`), 0 = not | target |
| `comment_clean` | str | Cleaned comment body. This is the model input | usable |
| `parent_clean` | str | Cleaned parent comment; the context for sub-question 1 | usable |

`comment_clean` has already been through: unicode normalisation, mojibake
repair, two rounds of HTML unescaping, Reddit markdown stripping, `>` quote-line
removal, URLs replaced with `<URL>`, `@name` and `/u/name` replaced with
`<USER>`, `/r/name` reduced to the bare subreddit word, and removal of `/s` and
other label markers. Case and punctuation are **preserved** — they are the
markers the project is studying.

### Context metadata

| Column | Type | Description | |
|---|---|---|---|
| `parent_n_words` | int64 | Words in the parent comment | usable |
| `parent_n_chars` | int64 | Characters in the parent comment | usable |
| `parent_is_empty` | bool | Parent comment missing | usable |
| `parent_jaccard` | float64 | Token overlap between comment and parent | usable |
| `len_ratio_to_parent` | float32 | Comment words ÷ parent words | usable |
| `parent_key` | str | blake2b hash of the parent text | key |

`parent_key` is not a feature, but it matters. When cross-validating inside the
training set, pass it as `GroupKFold(groups=train["parent_key"])` — otherwise
the folds reintroduce the sibling leak the splits were built to prevent.

The EDA found that none of these statistics carry much signal: `parent_jaccard`
is the fifth-strongest feature overall at r = −0.065, and the length ratio is
flat at 45–52% across every bin. If context helps, it will not be through these.

### Surface markers

| Column | Type | Description | EDA lift | |
|---|---|---|---:|---|
| `n_exclam` | int64 | Count of `!` | 2.81× | usable |
| `n_interjections` | int64 | Interjections and stance words (oh/wow/yeah/sure…) | 1.97× | usable |
| `starts_with_interjection` | bool | Comment opens with an interjection | — | usable |
| `n_elongation` | int64 | Character elongation (sooo) | 1.49× | usable |
| `n_repeat_punct` | int64 | Repeated punctuation runs (!! ??) | 1.05× | usable |
| `n_ellipsis` | int64 | Ellipses | 1.05× | usable |
| `n_allcaps_words` | int64 | ALL-CAPS words | 1.05× | usable |
| `allcaps_ratio` | float64 | Share of words in caps | — | usable |
| `n_question` | int64 | Count of `?` | 0.89× | usable |
| `n_scare_quotes` | int64 | Short quoted spans | 0.67× | usable |
| `n_chars`, `n_words`, `n_sentences` | int64 | Length | non-monotonic | careful |
| `avg_word_len` | float64 | Mean word length | — | usable |
| `n_urls`, `n_mentions`, `n_hashtags` | int64 | Entity counts, taken before replacement | — | usable |
| `n_emoji` | int64 | Emoji count | n/a | never |
| `has_positive_interjection` | bool | Derived from `starts_with_interjection` | — | careful |

Three things to know here:

`n_scare_quotes` points the wrong way. On Reddit, quotation marks are mostly
used to cite the parent comment rather than to sneer, so quoted spans are more
common in sincere comments.

`n_emoji` is identically zero across 2009–2016 Reddit. It is dead weight on this
corpus, though it is live (and still points away from irony) on the Twitter sets.

Bin `n_words` rather than using it linearly. The sarcasm rate runs 40% at 1–3
words, peaks at 56% around 9–12, then falls to 27% past 40.

`has_positive_interjection` correlates almost perfectly with
`starts_with_interjection` (r = 0.160 vs 0.158). Pick one.

### Categorical and temporal

| Column | Type | Description | |
|---|---|---|---|
| `subreddit` | string | Community name, 13,970 distinct | careful |
| `year` | Int16 | 2009–2016 | careful |
| `month` | Int8 | 1–12 | careful |
| `weekday` | Int8 | 0 = Monday | careful |
| `hour_utc` | Int8 | 0–23 | never |
| `date` | string | `YYYY-MM`; redundant with `created_utc` | key |
| `created_utc` | datetime64 | Raw timestamp | key |
| `author` | string | 254,321 distinct | careful |

`subreddit` is the single strongest predictor in the corpus — sarcasm rate runs
from 8% to 79% across communities — and also the most dangerous. On
`sarc_random` it partly measures memorised community priors. Use
`sarc_subreddit` when the question is whether topic actually helps.

`hour_utc` varies by only 4 percentage points across the day. Drop it.

`author` is fine for a deliberate author-holdout experiment and badly
overfit-prone as a feature.

### Vote counts — all unusable

| Column | Type | Description | |
|---|---|---|---|
| `score` | float32 | Upvotes − downvotes | never |
| `ups`, `downs` | float32 | Placeholder −1 in most rows | never |
| `ups_is_missing`, `downs_is_missing` | bool | Whether the placeholder applies | never |
| `is_controversial` | bool | `score` in [−1, 1] | never |

`score` genuinely correlates with the label — comments below −5 are 67%
sarcastic — but it is information that only exists after the comment has been
voted on, so it is unavailable at prediction time.

`downs_is_missing` correlates with the label at r = −0.059. That is an artefact
of when Reddit stopped exposing vote counts, not a property of sarcasm. Use this
group for analysis; keep it out of models.

### Cleaning audit columns — never use as features

| Column | Type | Description | |
|---|---|---|---|
| `had_slash_s` | bool | The raw row contained `/s` (since removed) | never |
| `had_label_hashtag` | bool | Contained `#sarcasm` or similar (removed) | never |
| `had_explicit_sarcasm_word` | bool | Contained `<sarcasm>` or similar (removed) | never |
| `en_score` | float64 | English-likeness heuristic score | never |
| `lang_method` | str | How the language decision was made | never |

The first three record the label markers the cleaner stripped. Feeding them to a
model is handing it the answer. They exist so the leakage check stays
reproducible.

---

## Supplementary schema — 26 columns

| Column | Type | Description | |
|---|---|---|---|
| `label` | int8 | 1 = sarcastic or ironic | target |
| `text_clean` | str | Cleaned target text (the analogue of `comment_clean`) | usable |
| `context_clean` | str | Context; multiple turns joined with ` \|\| `, oldest first | usable |
| `has_context` | bool | Whether context exists | usable |
| `context_n_words` | int64 | Words of context | usable |
| `n_context_turns` | float64 | Number of turns; only for `figlang_*`, NaN elsewhere | usable |
| `dataset` | str | Corpus name | key |
| `platform` | str | reddit / twitter / news | key |

Surface features (`n_chars` through `starts_with_interjection`) carry the same
names and definitions as in the primary schema, which is what makes
cross-corpus comparison possible.

`n_context_turns` is a float because it is NaN for the corpora that have no
turn structure. Slice on it for the 0 / 1 / n-turn curve in sub-question 1.

---

## Suggested starting feature set

```python
import sys; sys.path.insert(0, "pipeline")
from load import load_split, xy, MODEL_SAFE_FEATURES

train, val, test = load_split("sarc_random")
X, y = xy(train)                      # text + 22 numeric features
X, y = xy(train, with_context=True)   # parent + " [SEP] " + comment
```

`xy()` returns the 22 columns marked usable above plus a joined `text` column.
Categorical columns such as `subreddit` are deliberately excluded from the
default — add them explicitly, and think about which split you are adding them
on.
