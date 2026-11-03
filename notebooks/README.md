# notebooks/ — the project step by step

Read and run these in order. Each notebook has a matching document in `docs/steps/`, and `docs/SUMMARY.md` ties them together.

| Notebook | Step | Main result |
|---|---|---|
| `01_data_and_eda.ipynb` | data, cleaning, splits, EDA | 913,769 cleaned SARC rows; leakage audit passes; mock-agreement vocabulary |
| `02_baseline_models.ipynb` | team baselines (NB, LR, SVM, LightGBM, XGBoost, EBM) | LightGBM baseline 0.8074 |
| `03_feature_engineering.ipynb` | cues, contrast, char n-grams, LSA, LR score, vocabulary | best compliant single model 0.8296 |
| `04_interpretable_ebm.ipynb` | full-data EBM and its explanations | 0.8229, fully additive |
| `05_scratch_neural_networks.ipynb` | from-scratch BiLSTM and ESIM | ESIM 0.8242; parent +0.0063 |
| `06_pretrained_reference.ipynb` | RoBERTa — **reference only, not submitted** | 0.8809 |
| `07_ensembles.ipynb` | voting, stacking, MBR | **compliant stacking 0.8420 (submitted)** |
| `08_model_comparison.ipynb` | full leaderboard, transfer, research questions | `results/ensemble/leaderboard_all_models.csv` |
| `09_ablations.ipynb` | author overlap, stacking members and design, text representation | no drop on unseen authors |

All AUCs are on `sarc_random` test.

## How they work

The notebooks **do not retrain models**. Training happened in `pipeline/`, `modelling/` and `experiments/`; those scripts saved validation and test predictions, and the notebooks recompute every metric from them with one shared code path (`nbutils.py`). Notebook 07 also re-runs the core stacking step live, which takes about a minute.

Requirements:

- the data splits under `data/splits/` (see `data/README.md`);
- the experiment predictions (~500 MB, not in git): set `CS3244_PRED_DIR`, or place them at `experiments/predictions/`;
- Python 3.12 with pandas, scikit-learn, matplotlib, pyarrow, and `nbformat`/`ipykernel` to run them.

Figures are written to `figures/steps/` and tables to `results/ensemble/`. Running all nine takes about 10 minutes on a laptop CPU.
