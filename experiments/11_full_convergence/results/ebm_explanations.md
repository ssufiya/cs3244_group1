# Full-data EBM: explanation examples (sarc_random)

## Global: EBM term importance (mean |contribution to log-odds|)

| Term | Importance |
|---|---:|
| text_ngram_logodds | 1.153 |
| parent_n_chars | 0.241 |
| n_chars | 0.113 |
| parent_n_words | 0.102 |
| cue_opener | 0.102 |
| len_ratio_to_parent | 0.084 |
| par_n_words_log | 0.073 |
| n_chars & text_ngram_logodds | 0.054 |
| starts_with_interjection | 0.049 |
| rep_par_len_ratio_log | 0.046 |
| n_words | 0.045 |
| parent_jaccard | 0.038 |
| avg_word_len | 0.034 |
| avg_word_len & text_ngram_logodds | 0.030 |
| cue_mock_agree | 0.029 |
| par_sent_neg_sum | 0.023 |
| echo_share | 0.022 |
| n_words & text_ngram_logodds | 0.021 |
| cue_ends_question & text_ngram_logodds | 0.020 |
| sent_mean | 0.020 |

## Largest n-gram weights inside the text score (LR coefficients)

Sarcastic direction: `w:yes because` +7.34, `c:'t '` +7.03, `w:but but` +5.58, `w:yeah because` +5.12, `c:'] '` +4.99, `w:good thing` +4.90, `c:'s '` +4.87, `w:not like` +4.45, `c:'What,'` +4.40, `w:how dare` +4.28, `w:but thought` +4.24, `w:or anything` +4.20, `w:would never` +4.16, `w:oh but` +4.14, `w:right because` +4.13, `w:everyone knows` +4.03, `c:'.] '` +3.95, `c:'.]'` +3.91, `c:'e '` +3.90, `c:' t'` +3.83

Non-sarcastic direction: `c:'. '` -10.50, `c:'! '` -9.64, `c:' ('` -6.37, `c:') '` -6.07, `c:' :'` -5.98, `w:not sure` -4.79, `c:'!"'` -4.72, `c:'!" '` -4.59, `w:pretty sure` -4.11, `w:make sure` -4.10, `c:'..'` -3.94, `c:' but '` -3.84, `w:doesn mean` -3.81, `w:for sure` -3.74, `w:the original` -3.62, `w:not really` -3.62, `c:'/ '` -3.56, `c:'( '` -3.53, `c:'- '` -3.50, `w:totally forgot` -3.41

## Example 1: Correct sarcastic (high confidence)

- Reply: `But that's not racism, that's the trufs!`
- Parent: `"They love to paint this sub as racist...just ignore the Nazi propaganda that is stickied to the frontpage!"`
- True label 1, EBM P(sarcastic) 0.946 (intercept +0.14)

| EBM term | Value | Contribution to log-odds |
|---|---|---:|
| text_ngram_logodds | 2.791 | +2.562 |
| n_exclam & par_exclam |  | -0.356 |
| cue_opener | 1.0 | +0.291 |
| parent_n_chars | 108.0 | +0.157 |
| len_ratio_to_parent | 0.368 | -0.088 |
| n_chars | 40.0 | +0.079 |
| parent_n_words | 19.0 | -0.071 |
| par_sent_neg_sum | -5.5 | +0.062 |

Text score +2.79 comes mainly from: `c:'! '` -0.62, `w:racism that` -0.35, `c:'t '` +0.33, `w:racism` +0.28, `c:'But '` +0.26, `c:' t'` +0.25, `c:'But'` +0.24, `w:that not` +0.24

## Example 2: Correct non-sarcastic (high confidence)

- Reply: `To be fair, missing the start of the college fall semester seems a pretty horrible mis-step, given that (a) college students seem to have an infinite amount of spare cash in September and (b) so many `
- Parent: `Announcing it now and not selling it by black friday or Christmas don't make much sense either. Nothing make sense atm. Maybe microsoft will do the unthinkable, host an event to anounce partners' new `
- True label 0, EBM P(sarcastic) 0.038 (intercept +0.14)

| EBM term | Value | Contribution to log-odds |
|---|---|---:|
| text_ngram_logodds | -2.723 | -2.450 |
| parent_n_chars | 218.0 | +0.361 |
| n_words | 43.0 | -0.302 |
| n_chars | 233.0 | -0.253 |
| n_chars & text_ngram_logodds |  | -0.175 |
| parent_n_words | 36.0 | -0.141 |
| par_n_words_log | 3.638 | -0.104 |
| n_words & text_ngram_logodds |  | -0.101 |

Text score -2.72 comes mainly from: `c:' ('` -0.41, `c:') '` -0.38, `w:many` -0.23, `c:'t '` +0.15, `w:amount` -0.15, `w:so many` +0.13, `w:college` -0.13, `c:'. '` -0.13

## Example 3: Error: non-sarcastic predicted as sarcastic

- Reply: `Maybe he's into pegging only, and is therefore a technical virgin.`
- Parent: `I think Haley Atwell has said that he isn't, which is... an interesting dynamic.`
- True label 0, EBM P(sarcastic) 0.919 (intercept +0.14)

| EBM term | Value | Contribution to log-odds |
|---|---|---:|
| text_ngram_logodds | 2.335 | +2.123 |
| parent_n_words | 14.0 | +0.065 |
| avg_word_len & text_ngram_logodds |  | +0.063 |
| cue_opener | 0.0 | -0.062 |
| par_n_words_log | 2.708 | +0.057 |
| n_words | 11.0 | +0.057 |
| n_chars & text_ngram_logodds |  | +0.053 |
| avg_word_len | 4.909 | +0.046 |

Text score +2.34 comes mainly from: `w:therefore` +0.54, `c:'. '` -0.23, `w:into` -0.22, `c:'s '` +0.15, `c:'eref'` +0.14, `c:'erefo'` +0.14, `c:' i'` +0.14, `w:maybe he` +0.13
