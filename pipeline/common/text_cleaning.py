"""
text_cleaning.py — shared normalisation, label-leak removal and feature
extraction for every corpus in the project.

Design notes
------------
* The order of operations matters. We count surface features (caps, "!!!",
  emoji, elongation) BEFORE lowercasing/normalising, because those very
  features are the sarcasm markers the project is trying to explain.
* Label leakage is treated as a first-class concern. Reddit labels come from
  the author writing "/s"; Twitter labels come from "#sarcasm". If those
  tokens survive into the text, a classifier scores ~100% by memorising them
  and the whole explainability story collapses. `strip_label_markers` removes
  them and reports how often they occurred.
* Language filtering is two-stage: a cheap lexical heuristic decides the clear
  cases, and (optionally) `langdetect` adjudicates only the ambiguous band.
  A full langdetect pass over 1M short comments takes ~20 min and is unreliable
  on very short strings; this keeps it to a few percent of rows.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Regexes (compiled once; these run over ~1M rows)
# --------------------------------------------------------------------------
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
RE_MD_LINK = re.compile(r"\[([^\]]*)\]\(\s*<?https?://[^)\s]+>?\s*\)")
RE_REDDIT_USER = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_\-]+", re.I)
RE_REDDIT_SUB = re.compile(r"(?<![\w/])/?r/([A-Za-z0-9_]+)", re.I)
RE_TW_MENTION = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{1,15}\b")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_CODEBLOCK = re.compile(r"(?:^ {4}.*$\n?)+|`{1,3}[^`]*`{1,3}", re.M)
# Quote lines are matched AFTER html unescaping, so the marker is a bare ">".
# The escaped forms are kept as alternatives because some rows are triple-
# escaped and one pass of unescaping leaves "&gt;" behind.
RE_QUOTELINE = re.compile(r"^[ \t]*(?:&(?:amp;)*gt;|>)+.*$", re.M)
RE_MD_EMPH = re.compile(r"(\*{1,3}|_{1,3}|~{2})(?=\S)(.+?)(?<=\S)\1", re.S)
RE_SUPER = re.compile(r"\^+")
RE_ENTITY_NUM = re.compile(r"&#x?[0-9a-fA-F]+;")
RE_ZERO_WIDTH = re.compile(r"[​-‏‪-‮﻿­]")
RE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RE_WS = re.compile(r"\s+")

# Label-leak markers -------------------------------------------------------
# Reddit: a trailing "/s", "\s", "(/s)", "[/s]", "-/s", optionally after
# punctuation. We deliberately require it to stand alone as a token so we do
# not destroy legitimate text like "w/s" or a file path.
RE_SLASH_S = re.compile(
    r"""(?:^|(?<=[\s.,!?;:"'\)\]\}]))      # start or after a boundary char
        [\(\[\{\-~]*                        # optional opening decoration
        [/\\]\s?s                           # the marker itself: /s or \s
        [\)\]\}\.\!]*                       # optional closing decoration
        (?=$|[\s.,!?;:"'\(\[\{])            # must end at a boundary
    """,
    re.I | re.X,
)
LABEL_HASHTAGS = {
    "sarcasm", "sarcastic", "sarcastictweet", "sarcasmtweet", "notsarcasm",
    "irony", "ironic", "ironia", "not", "justkidding", "jk", "kidding",
    "sarcastically", "sarcasam",
}

# Surface-marker regexes ---------------------------------------------------
RE_EMOJI = re.compile(
    "[" "\U0001f300-\U0001faff" "\U00002600-\U000027bf"
    "\U0001f1e6-\U0001f1ff" "\U00002190-\U000021ff" "\U00002b00-\U00002bff"
    "\U0000fe0f" "]"
)
RE_ELONG = re.compile(r"([A-Za-z])\1{2,}")
RE_REPEAT_PUNCT = re.compile(r"([!?.])\1{1,}")
RE_ALLCAPS = re.compile(r"\b[A-Z]{2,}\b")
RE_SCARE_QUOTE = re.compile(r"[\"“”'‘’][^\"“”]{1,40}[\"“”]")
RE_WORD = re.compile(r"[A-Za-z']+")

DELETED_MARKERS = {"[deleted]", "[removed]", "deleted", "removed", "nan", "none", ""}

# Interjections / stance markers repeatedly flagged in the sarcasm literature
INTERJECTIONS = {
    "oh", "ah", "wow", "yeah", "yea", "yep", "sure", "right", "great", "nice",
    "gee", "gosh", "huh", "hmm", "well", "ok", "okay", "lol", "haha", "obviously",
    "clearly", "totally", "absolutely", "definitely", "shocking", "shocker",
    "wonderful", "brilliant", "genius", "amazing", "perfect", "fantastic",
}

# High-frequency English function words for the cheap language heuristic.
EN_FUNCTION_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
    "time", "no", "just", "him", "know", "take", "people", "into", "year",
    "your", "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well", "way", "even",
    "new", "want", "because", "any", "these", "give", "day", "most", "us",
    "is", "are", "was", "were", "been", "has", "had", "did", "does", "am",
    "too", "very", "really", "still", "never", "always", "here", "thing",
}


# --------------------------------------------------------------------------
# Step 1: unicode / markup normalisation
# --------------------------------------------------------------------------
def fix_mojibake(text: str) -> str:
    """
    Repair the common UTF-8-read-as-cp1252 damage (â€™ -> ').

    cp1252 is tried first because it is what actually caused the damage;
    latin-1 is the fallback for strings containing bytes cp1252 leaves
    undefined (0x81, 0x8d, 0x8f, 0x90, 0x9d), which are common in this data.
    Both round-trips are strict: if the result is not valid UTF-8 the string
    was not mojibake and is returned untouched.
    """
    if not ("Ã" in text or "â€" in text or "Â" in text):
        return text
    for codec in ("cp1252", "latin-1"):
        try:
            return text.encode(codec, errors="strict").decode("utf-8", errors="strict")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def normalise_unicode(text: str) -> str:
    text = fix_mojibake(text)
    # Reddit dumps are frequently double-escaped: &amp;gt; -> &gt; -> >
    for _ in range(2):
        if "&" in text:
            text = html.unescape(text)
    text = RE_ZERO_WIDTH.sub("", text)
    text = RE_CTRL.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    # Normalise the many unicode quote/dash variants so "scare quote"
    # detection and tokenisation stay consistent across corpora.
    text = (text.replace("‘", "'").replace("’", "'")
                .replace("“", '"').replace("”", '"')
                .replace("–", "-").replace("—", "-")
                .replace("…", "..."))
    return text


def strip_markup(text: str, keep_subreddit: bool = True) -> str:
    """Remove Reddit markdown / quoting, keeping the human-readable content."""
    text = RE_CODEBLOCK.sub(" ", text)
    text = RE_QUOTELINE.sub(" ", text)          # quoted parent text, not the author's
    text = RE_MD_LINK.sub(r"\1", text)          # [anchor](url) -> anchor
    text = RE_MD_EMPH.sub(r"\2", text)          # **bold** -> bold
    text = RE_SUPER.sub("", text)
    text = RE_ENTITY_NUM.sub(" ", text)
    return text


def replace_entities(text: str, keep_subreddit: bool = True) -> str:
    """Swap volatile identifiers for stable placeholders."""
    text = RE_URL.sub(" <URL> ", text)
    text = RE_REDDIT_USER.sub(" <USER> ", text)
    text = RE_TW_MENTION.sub(" <USER> ", text)
    if keep_subreddit:
        # A subreddit name is topical signal for sub-question 3; keep the word.
        text = RE_REDDIT_SUB.sub(r" \1 ", text)
    else:
        text = RE_REDDIT_SUB.sub(" <SUB> ", text)
    return text


# --------------------------------------------------------------------------
# Step 2: label-leak removal
# --------------------------------------------------------------------------
@dataclass
class LeakCounts:
    slash_s: int = 0
    label_hashtag: int = 0
    explicit_word: int = 0


def strip_label_markers(text: str) -> tuple[str, bool, bool, bool]:
    """
    Remove the annotation artefacts that *are* the label.

    Returns (clean_text, had_slash_s, had_label_hashtag, had_explicit_word).
    """
    had_slash_s = bool(RE_SLASH_S.search(text))
    if had_slash_s:
        text = RE_SLASH_S.sub(" ", text)

    had_tag = False

    def _hash(m: re.Match) -> str:
        nonlocal had_tag
        tag = m.group(1)
        if tag.lower() in LABEL_HASHTAGS:
            had_tag = True
            return " "
        return " " + tag + " "          # keep the word, drop the '#'

    text = RE_HASHTAG.sub(_hash, text)

    # "/sarcasm", "<sarcasm>", "(sarcasm)" written out in full
    had_word = bool(re.search(r"[/<(\[]\s*sarcas(m|tic)\s*[>)\]]?", text, re.I))
    if had_word:
        text = re.sub(r"[/<(\[]\s*sarcas(m|tic)\s*[>)\]]?", " ", text, flags=re.I)

    return text, had_slash_s, had_tag, had_word


# --------------------------------------------------------------------------
# Step 3: surface-marker features (computed on the pre-lowercased text)
# --------------------------------------------------------------------------
def surface_features(text: str) -> dict:
    words = RE_WORD.findall(text)
    n_words = len(words)
    caps = RE_ALLCAPS.findall(text)
    lower = [w.lower() for w in words]
    return {
        "n_chars": len(text),
        "n_words": n_words,
        "n_sentences": max(1, len(re.findall(r"[.!?]+", text))),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "n_exclam": text.count("!"),
        "n_question": text.count("?"),
        "n_ellipsis": len(re.findall(r"\.{2,}", text)),
        "n_repeat_punct": len(RE_REPEAT_PUNCT.findall(text)),
        "n_allcaps_words": len(caps),
        "allcaps_ratio": len(caps) / n_words if n_words else 0.0,
        "n_emoji": len(RE_EMOJI.findall(text)),
        "n_elongation": len(RE_ELONG.findall(text)),
        "n_scare_quotes": len(RE_SCARE_QUOTE.findall(text)),
        "n_urls": len(RE_URL.findall(text)),
        "n_mentions": len(RE_TW_MENTION.findall(text)) + len(RE_REDDIT_USER.findall(text)),
        "n_hashtags": len(RE_HASHTAG.findall(text)),
        "n_interjections": sum(1 for w in lower if w in INTERJECTIONS),
        "starts_with_interjection": bool(lower and lower[0] in INTERJECTIONS),
    }


# --------------------------------------------------------------------------
# Step 4: language heuristic
# --------------------------------------------------------------------------
def english_score(text: str) -> float:
    """
    Cheap English-likeness score in [0, 1].

    Combines (a) the share of tokens that are English function words with
    (b) the share of letters that are ASCII. Short texts get a prior nudge so
    that a valid three-word comment is not thrown away for lacking stopwords.
    """
    words = RE_WORD.findall(text.lower())
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    ascii_ratio = sum(1 for c in letters if c.isascii()) / len(letters)
    if not words:
        return 0.35 * ascii_ratio
    fw = sum(1 for w in words if w in EN_FUNCTION_WORDS) / len(words)
    # With <=5 tokens, absence of a function word is uninformative.
    prior = 0.30 if len(words) <= 5 else 0.0
    return min(1.0, 0.65 * ascii_ratio + 0.35 * min(1.0, fw * 2.5 + prior))


def add_language_flags(df: pd.DataFrame, col: str,
                       low: float = 0.55, high: float = 0.72,
                       use_langdetect: bool = True) -> pd.DataFrame:
    """
    Two-stage language decision.

    score >= high  -> English
    score <  low   -> not English
    in between     -> ask langdetect (if installed); otherwise accept.
    """
    df["en_score"] = df[col].map(english_score)
    df["is_english"] = df["en_score"] >= high
    ambiguous = (df["en_score"] >= low) & (df["en_score"] < high)
    df["lang_method"] = np.where(df["en_score"] >= high, "heuristic-high",
                         np.where(df["en_score"] < low, "heuristic-low", "ambiguous"))

    if ambiguous.any() and use_langdetect:
        try:
            from langdetect import DetectorFactory, detect
            DetectorFactory.seed = 0

            def _d(t: str) -> bool:
                try:
                    return detect(t) == "en"
                except Exception:       # noqa: BLE001  (langdetect raises on short/empty)
                    return True         # keep it; the heuristic already liked it enough
            df.loc[ambiguous, "is_english"] = df.loc[ambiguous, col].map(_d)
            df.loc[ambiguous, "lang_method"] = "langdetect"
        except ImportError:
            df.loc[ambiguous, "is_english"] = True
            df.loc[ambiguous, "lang_method"] = "ambiguous-kept"
    elif ambiguous.any():
        df.loc[ambiguous, "is_english"] = True
        df.loc[ambiguous, "lang_method"] = "ambiguous-kept"
    return df


# --------------------------------------------------------------------------
# Step 5: the full pipeline for one text column
# --------------------------------------------------------------------------
def squash_elongation(text: str) -> str:
    """
    "sooooo" -> "soo". Off by default in `clean_series`.

    Trade-off: leaving elongation intact preserves the marker verbatim but
    fragments the vocabulary ("sooo", "soooo", "sooooo" are three types for a
    bag-of-words model). Squashing to two characters keeps the distinction
    from the ordinary word while collapsing the variants — and `n_elongation`
    has already recorded that it happened, so no signal is lost.
    """
    return RE_ELONG.sub(r"\1\1", text)


def clean_series(s: pd.Series, keep_subreddit: bool = True,
                 squash_elong: bool = False) -> pd.DataFrame:
    """
    Run the whole text pipeline over a Series.

    Returns a DataFrame with the cleaned text, the leak flags and all the
    surface features, indexed like the input.

    `squash_elong=True` additionally collapses character elongation in
    `text_clean` (the feature count is taken first, so nothing is lost).
    """
    s = s.fillna("").astype(str)

    norm = s.map(normalise_unicode)
    norm = norm.map(lambda t: strip_markup(t))

    leaks = norm.map(strip_label_markers)
    text = leaks.map(lambda x: x[0])
    out = pd.DataFrame({
        "had_slash_s": leaks.map(lambda x: x[1]),
        "had_label_hashtag": leaks.map(lambda x: x[2]),
        "had_explicit_sarcasm_word": leaks.map(lambda x: x[3]),
    }, index=s.index)

    text = text.map(lambda t: replace_entities(t, keep_subreddit))

    # Features come from the entity-replaced but still case-preserving text.
    feats = pd.DataFrame(list(text.map(surface_features)), index=s.index)

    if squash_elong:
        text = text.map(squash_elongation)
    text = text.map(lambda t: RE_WS.sub(" ", t).strip())
    out["text_clean"] = text
    out["text_norm"] = text.str.lower()            # for dedup / bag-of-words
    return pd.concat([out, feats], axis=1)


def dedup_key(s: pd.Series) -> pd.Series:
    """Aggressive normalisation used only for near-duplicate detection."""
    return (s.str.lower()
             .str.replace(r"[^a-z0-9 ]", "", regex=True)
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())


def is_deleted(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower().isin(DELETED_MARKERS)


def jaccard(a: str, b: str) -> float:
    """Token overlap between a comment and its parent — a cheap proxy for
    'is the reply echoing the parent', one of the classic sarcasm cues."""
    ta = set(RE_WORD.findall(a.lower()))
    tb = set(RE_WORD.findall(b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
