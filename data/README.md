# data/

Most of what belongs here is not in the repository. Raw downloads, cleaned
parquet and the split package come to about 1 GB, and all of it is reproducible
from the scripts, so only the small provenance files and previews are tracked.

## Tracked

| Path | What it is |
|---|---|
| `MANIFEST.json` | URL, byte size, SHA256, role and tags for all 15 downloaded source files |
| `catalog.json` | Machine-readable index of every split set: rows, role, project stage |
| `samples/` | CSV previews, a few hundred rows each, for eyeballing without loading parquet |

## Not tracked — rebuild with

```bash
python pipeline/00_download_datasets.py    # → data/raw/          ~278 MB
python pipeline/01_clean_primary.py        # → data/processed/    ~164 MB
python pipeline/02_clean_supplementary.py  # → data/processed/
python pipeline/04_build_splits.py         # → data/splits/       ~536 MB
```

Which produces:

```
data/
├── raw/                    exactly as downloaded, never edited
│   ├── primary/reddit_sarc/          SARC balanced, 243 MB
│   ├── supplementary/                news headlines, tweeteval irony, figlang
│   └── external/lexicons/            Empath, VADER, NRC EmoLex, SentiWordNet
├── processed/              cleaned full corpora, before splitting
└── splits/                 what the modelling code reads
    ├── primary/            sarc_random | sarc_temporal | sarc_subreddit
    ├── supplementary/      tweeteval_irony | news_headlines | figlang_*
    ├── lexicons/           copies of the four lexicons
    └── _rejected/          audited and rejected — do not use
```

Everything is deterministic under seed 3244, so a rebuild reproduces the same
splits byte for byte. The SHA256 values in `MANIFEST.json` let you confirm the
downloads matched what this analysis ran on.

No Kaggle token is required. Every source is mirrored on HuggingFace or GitHub.
