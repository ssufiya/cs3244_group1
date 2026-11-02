# Step 2 — Team baseline models

Notebook: [`notebooks/02_baseline_models.ipynb`](../../notebooks/02_baseline_models.ipynb) · Training notebooks: `modelling/probabilistic_linear/`, `modelling/Margin-Based/`, `modelling/gradient_boosting/`, `modelling/glassbox/`, `modelling/tree-based/` · Detailed reports: [`nb_logreg_report.md`](../nb_logreg_report.md), [`gbdt_report.md`](../gbdt_report.md), [`glassbox_report_en.md`](../glassbox_report_en.md)

## Models

| Family | Models | Features |
|---|---|---|
| Probabilistic | Multinomial NB, Complement NB | reply word TF-IDF ± surface features |
| Linear | logistic regression; linear SVM | word TF-IDF ± surface; the SVM adds lexicon features |
| Gradient-boosted trees | LightGBM, XGBoost | reply word 1–2-gram TF-IDF (20k) ± 17 surface features |
| Glass-box | EBM (+ mined rules) | 340 named features on a 200k sample, including a subreddit encoding |
| Tree-based | decision tree, random forest, histogram gradient boosting | `modelling/tree-based/` feature pipeline |
| Margin-based | linear SVM, SGD (hinge) | TF-IDF + lexicon + surface |

All hyper-parameters and thresholds were chosen on `sarc_random` validation, and test was scored once. Each notebook saved its validation and test predictions keyed by `row_id`, which is what makes the later ensembles possible. The SVM and EBM predictions were re-scored in `experiments/00_extra_baseline_predictions/`.

## Results (sarc_random test, recomputed in the notebook)

| Model | ROC-AUC | sarc_subreddit |
|---|---:|---:|
| **LightGBM, TF-IDF + surface (reference baseline)** | **0.8074** | 0.8023 |
| XGBoost, TF-IDF + surface | 0.8061 | 0.8014 |
| Logistic regression, TF-IDF + surface | 0.8029 | 0.7963 |
| Linear SVM, TF-IDF + lexicon + surface | 0.8026 | 0.7958 |
| LightGBM, TF-IDF only | 0.7964 | 0.7927 |
| Multinomial NB, TF-IDF | 0.7798 | 0.7716 |
| SGD (hinge), TF-IDF + lexicon + surface | 0.7860 | |
| Histogram gradient boosting (tree-based notebook) | 0.7759 | |
| Team EBM | 0.7620 | — |
| Random forest | 0.7603 | |
| Decision tree | 0.6894 | |

## Takeaways

- LightGBM and logistic regression are within 0.005 of each other: **the bottleneck is the representation, not the classifier.** This motivates step 3.
- Surface features help trees and linear models (about +0.010) but hurt Naive Bayes, whose multinomial assumption does not fit dense scaled features.
- Single trees and random forests are clearly weaker than boosted trees on sparse text features.
- The team also built a first voting/stacking ensemble in `modelling/Ensemble/stacking&voting.ipynb`; step 7 extends it.
- `lgbm_reply_tfidf_surface` is the reference for every Δ reported later, always as a paired bootstrap on the same test rows.
