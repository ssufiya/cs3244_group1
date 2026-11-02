"""
Hand-built feature blocks used by the improvement experiments.

cue_block(df)       linguistic sarcasm cues promised in the proposal (Q2):
                    hyperbole / intensity, rhetorical & mock-agreement
                    patterns, understatement, intra-reply sentiment contrast.
contrast_block(df)  reply-versus-parent contrast features (Q1): sentiment
                    incongruity, echo / overlap, question-answer shape.

Everything is a deterministic function of one row (no fitting), so there is no
train/test leakage. All patterns are lower-cased word-boundary regexes.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import load_lexicon, parent_of, text_of

RE_TOKEN = re.compile(r"[a-z][a-z']+")

INTENSIFIERS = ["so", "totally", "absolutely", "literally", "really", "very", "extremely",
                "completely", "utterly", "incredibly", "super", "truly", "seriously", "definitely",
                "certainly", "surely", "clearly", "obviously", "entirely", "highly", "insanely"]
ABSOLUTES = ["always", "never", "everyone", "everybody", "nobody", "nothing", "everything",
             "all", "none", "every", "forever", "anyone", "anything", "noone", "nowhere", "everywhere"]
PRAISE = ["amazing", "brilliant", "genius", "wonderful", "fantastic", "perfect", "great", "awesome",
          "love", "best", "excellent", "incredible", "fabulous", "glorious", "superb", "flawless",
          "shocking", "shocked", "surprised", "surprising", "wow"]
HEDGES = [r"a bit", r"a little", r"slightly", r"kind of", r"kinda", r"sort of", r"sorta", r"somewhat",
          r"not exactly", r"not really", r"hardly", r"barely", r"mildly", r"a tad", r"just a"]
MOCK_AGREE = [r"yeah,? because", r"yes,? because", r"sure,? because", r"oh,? (?:yeah|yes|sure|wow|great|good)",
              r"of course", r"because (?:that|this|it)'?s", r"good thing", r"thank god", r"thanks obama",
              r"i'?m sure", r"must be", r"how dare", r"who would(?:'ve| have) thought", r"who knew",
              r"what could (?:possibly )?go wrong", r"everyone knows", r"we all know", r"totally not",
              r"because reasons", r"yeah right", r"nice try", r"good job", r"well done", r"so much for",
              r"clearly the", r"obviously the", r"but but", r"no no", r"silly me", r"my bad"]
TAG_Q = [r"right\?", r"isn'?t it\?", r"aren'?t they\?", r"don'?t you\?", r"amirite", r"am i right",
         r"didn'?t you know", r"you know\?", r"huh\?", r"eh\?"]
OPENERS = ["yeah", "yes", "yea", "sure", "oh", "well", "wow", "ah", "ahh", "because", "but", "so",
           "right", "great", "nice", "clearly", "obviously", "totally", "duh", "gee", "golly"]
EMOTICON = re.compile(r"(?:[;:]-?[)pPdD]|\bxd\b|\^_\^|\b:3\b)")


def _rx(words, plain=True):
    alt = "|".join(re.escape(w) if plain else w for w in words)
    return re.compile(rf"\b(?:{alt})\b")


RX_INT, RX_ABS, RX_PRAISE = _rx(INTENSIFIERS), _rx(ABSOLUTES), _rx(PRAISE)
RX_HEDGE, RX_MOCK = _rx(HEDGES, False), _rx(MOCK_AGREE, False)
RX_TAGQ = re.compile("(?:" + "|".join(TAG_Q) + ")")
RX_OPEN = re.compile(rf"^\W*(?:{'|'.join(OPENERS)})\b")
RX_SUPER = re.compile(r"\b(?:[a-z]{3,}est|most \w+|best|worst|greatest|of all time|ever)\b")
RX_STAR = re.compile(r"\*[^*\s][^*]{0,40}\*")
RX_QUOTED = re.compile(r"\"[^\"]{1,40}\"")
RX_ELONG = re.compile("(?:" + "|".join(f"{ch}{{3,}}" for ch in "abcdefghijklmnopqrstuvwxyz") + ")")  # RE2: no backrefs

_VADER = None


def _vader():
    global _VADER
    if _VADER is None:
        _VADER = load_lexicon("vader")
    return _VADER


def _sent_stats(texts: pd.Series) -> np.ndarray:
    """pos_sum, neg_sum, max_pos, min_neg, n_pos, n_neg, mean for each text."""
    V = _vader()
    out = np.zeros((len(texts), 7), np.float32)
    for i, t in enumerate(texts):
        vals = [V[w] for w in RE_TOKEN.findall(t) if w in V]
        if vals:
            a = np.asarray(vals, np.float32)
            pos, neg = a[a > 0], a[a < 0]
            out[i] = [pos.sum(), neg.sum(), pos.max() if len(pos) else 0, neg.min() if len(neg) else 0,
                      len(pos), len(neg), a.mean()]
    return out


CUE_NAMES = ["cue_intensifiers", "cue_absolutes", "cue_praise", "cue_superlatives", "cue_hedges",
             "cue_mock_agree", "cue_tag_question", "cue_ends_question", "cue_opener",
             "cue_star_emphasis", "cue_quoted", "cue_emoticon", "cue_elongated",
             "cue_intensity_rate", "cue_praise_rate",
             "sent_pos_sum", "sent_neg_sum", "sent_max_pos", "sent_min_neg", "sent_n_pos", "sent_n_neg",
             "sent_mean", "sent_mixed", "sent_polarity_span", "cue_praise_x_neg"]


def cue_block(df: pd.DataFrame) -> np.ndarray:
    raw = text_of(df)
    t = raw.str.lower()
    nw = t.str.count(r"\S+").clip(lower=1).to_numpy(np.float32)
    c = lambda rx: t.str.count(rx).to_numpy(np.float32)  # noqa: E731
    inten, absol, praise = c(RX_INT), c(RX_ABS), c(RX_PRAISE)
    s = _sent_stats(t)
    cols = [inten, absol, praise, c(RX_SUPER), c(RX_HEDGE), c(RX_MOCK), c(RX_TAGQ),
            t.str.rstrip().str.endswith("?").to_numpy(np.float32),
            t.str.contains(RX_OPEN).to_numpy(np.float32),
            raw.str.count(RX_STAR).to_numpy(np.float32), raw.str.count(RX_QUOTED).to_numpy(np.float32),
            raw.str.count(EMOTICON).to_numpy(np.float32), c(RX_ELONG),
            (inten + absol) / nw, praise / nw]
    mixed = ((s[:, 4] > 0) & (s[:, 5] > 0)).astype(np.float32)
    span = s[:, 2] - s[:, 3]
    praise_x_neg = (praise > 0).astype(np.float32) * (s[:, 5] > 0)
    X = np.column_stack(cols + [s, mixed, span, praise_x_neg]).astype(np.float32)
    assert X.shape[1] == len(CUE_NAMES)
    return X


CONTRAST_NAMES = ["par_sent_mean", "par_sent_pos_sum", "par_sent_neg_sum", "rep_minus_par_sent",
                  "rep_x_par_sent", "pos_reply_neg_parent", "neg_reply_pos_parent",
                  "par_ends_question", "par_has_question", "q_answered_with_opener",
                  "echo_share", "echo_count", "parent_contains_reply", "rep_par_len_ratio_log",
                  "par_exclam", "par_n_words_log", "par_intensifiers", "par_absolutes"]


def contrast_block(df: pd.DataFrame) -> np.ndarray:
    rt = text_of(df).str.lower()
    pt = parent_of(df).str.lower()
    rs, ps = _sent_stats(rt), _sent_stats(pt)
    rmean, pmean = rs[:, 6], ps[:, 6]
    echo_share = np.zeros(len(df), np.float32)
    echo_cnt = np.zeros(len(df), np.float32)
    for i, (r, p) in enumerate(zip(rt, pt)):
        rtok = RE_TOKEN.findall(r)
        if not rtok or not p:
            continue
        pset = set(RE_TOKEN.findall(p))
        hit = sum(1 for w in rtok if w in pset and len(w) > 3)
        echo_cnt[i] = hit
        echo_share[i] = hit / len(rtok)
    par_q_end = pt.str.rstrip().str.endswith("?").to_numpy(np.float32)
    opener = rt.str.contains(RX_OPEN).to_numpy(np.float32)
    rn = rt.str.count(r"\S+").to_numpy(np.float32)
    pn = pt.str.count(r"\S+").to_numpy(np.float32)
    X = np.column_stack([
        pmean, ps[:, 0], ps[:, 1], rmean - pmean, rmean * pmean,
        ((rmean > 0.05) & (pmean < -0.05)).astype(np.float32),
        ((rmean < -0.05) & (pmean > 0.05)).astype(np.float32),
        par_q_end, pt.str.contains("?", regex=False).to_numpy(np.float32), par_q_end * opener,
        echo_share, echo_cnt,
        np.array([bool(r) and r in p for r, p in zip(rt, pt)], np.float32),
        np.log1p(rn) - np.log1p(pn), pt.str.count("!").to_numpy(np.float32), np.log1p(pn),
        pt.str.count(RX_INT).to_numpy(np.float32), pt.str.count(RX_ABS).to_numpy(np.float32),
    ]).astype(np.float32)
    assert X.shape[1] == len(CONTRAST_NAMES)
    return X
