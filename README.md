# Fake Job Posting Fraud Detection

A notebook project for screening potentially fraudulent job listings in a heavily imbalanced dataset. The experiments compare text-only, structured, neural, and feature-fusion approaches, with attention to duplicate-text leakage, threshold choice, and the precision–recall trade-off.

## Problem

Fake listings can expose job seekers to payment and identity scams. The positive class is rare, so accuracy alone can hide missed fraud. This project treats fraud as the positive class and reports precision, recall, F1, ROC-AUC, and PR-AUC where the saved experiment supports them.

## Workflow

1. Load the CSV locally and inspect class balance and missing fields.
2. Clean text and derive posting metadata and text features.
3. Split by identical combined posting text using `StratifiedGroupKFold` in the fusion experiment to reduce exact-duplicate leakage.
4. Compare TF-IDF + Logistic Regression, structured models, feature fusion, BiLSTM/GloVe, DistilBERT, and XGBoost experiments.
5. Evaluate model scores and threshold trade-offs; inspect Logistic Regression features with SHAP.

The fusion model combines TF-IDF text with structured metadata and engineered signals such as email/URL presence, punctuation, missing company profile, and missing salary information.

## Results

The following table is transcribed from saved outputs in `feature_fusion_experiments.ipynb`. That notebook uses a grouped train/validation/test split (11,443 / 2,861 / 3,576 postings in the recorded run). Notebook outputs are cleared in this repository, and the source dataset is not included.

| Model / evaluation | Accuracy | Fraud precision | Fraud recall | Fraud F1 | ROC-AUC | PR-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.9656 | 0.6263 | 0.7168 | 0.6685 | 0.9613 | 0.7503 | 0.50 |
| TF-IDF + structured Logistic Regression | 0.9622 | 0.5841 | 0.7630 | 0.6617 | 0.9668 | 0.7657 | 0.50 |
| TF-IDF + structured + engineered signals Logistic Regression | 0.9748 | 0.8168 | 0.6185 | **0.7039** | 0.9635 | 0.7712 | 0.50 |
| DistilBERT v2 | 0.9740 | 0.8571 | 0.5549 | 0.6737 | 0.9386 | 0.7478 | 0.50 |
| Structured XGBoost | 0.9706 | 0.7656 | 0.5665 | 0.6512 | 0.9550 | 0.6963 | 0.83* |

`grouped_test_results.csv` contains these recorded values. The fusion Logistic Regression is the best reported F1 among these rows; the table is a single saved evaluation, not a claim of production performance.

The notebook also records 5-fold `StratifiedGroupKFold` results for the fraud-signal Logistic Regression: **Fraud F1 0.7667 ± 0.0407** and **ROC-AUC 0.9654 ± 0.0127**. Fold-level scores are available in `grouped_cv_results.csv`.

`separate_metrics_snapshot.csv` preserves another CSV found in the project folder. Its values differ from the grouped notebook results (for example, tuned TF-IDF + Logistic Regression F1 0.7851 versus 0.6685 in the grouped test table), and the CSV does not include split/run metadata. Treat it as a separate, unverified snapshot; do not combine or compare it directly with the grouped evaluation until its producing run is identified.

The baseline chart in `reports/results/baseline_model_fraud_f1.png` covers the six baseline models only; it does not show the later engineered-feature fusion result.

## Data

The project notes identify the source as the Kaggle Fake Job Postings dataset (17,880 postings; 866 fraudulent). The original source URL and exact redistribution terms were not recorded in the project files, so the dataset is excluded. Obtain a copy only from a source whose terms permit your use and place `fake_job_postings.csv` beside the notebook.

## Run the notebooks

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter lab
```

Open `fake_job_detection.ipynb` for the cleaned baseline workflow or `feature_fusion_experiments.ipynb` for grouped splitting and feature fusion. Optional neural-model cells require the packages in `requirements-neural.txt`. The fusion notebook may download GloVe vectors when running the GloVe experiment; these vectors are not bundled.

## Repository layout

```text
.
├── fake_job_detection.ipynb
├── feature_fusion_experiments.ipynb
├── reports/
│   └── results/
│       ├── baseline_model_fraud_f1.png
│       ├── grouped_cv_results.csv
│       ├── grouped_test_results.csv
│       └── separate_metrics_snapshot.csv
├── requirements.txt
└── requirements-neural.txt
```

## Limitations

- Dataset provenance and redistribution terms need confirmation.
- Saved metrics come from more than one project run; the standalone CSV lacks run metadata and is explicitly kept separate.
- The notebooks are research workflows, not a packaged service. No trained model, prediction API, or deployment artifact is included.
- Reproducing results requires documenting the dataset version, split, random seed, package versions, and threshold-selection procedure.
- Fraud predictions should support human review; false positives can affect legitimate job listings.

## Technologies

Python, Jupyter, pandas, NumPy, scikit-learn, SciPy sparse matrices, TF-IDF, Logistic Regression, Random Forest, SVM, XGBoost, TensorFlow/Keras, GloVe, DistilBERT, SHAP.
