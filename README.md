# Fake Job Posting Detection

A notebook-based binary classification study for identifying potentially fraudulent job postings. The notebook compares text and tabular feature approaches, explores class imbalance and decision thresholds, and includes neural text models and explainability experiments.

## Data

The notebook expects a local `fake_job_postings.csv` file in the project directory. The dataset is not included in this repository. Obtain it from a source whose terms permit your use, and place it beside the notebook. The notebook includes fallback paths for common notebook runtimes.

## Run

Create a Python environment with Jupyter, NumPy, pandas, Matplotlib, scikit-learn, TensorFlow, and SHAP, then open `fake_job_detection.ipynb`. Some cells install optional dependencies and download GloVe vectors from Stanford when needed. Those external assets are not bundled here.

The notebook was saved without cell outputs. Its saved metrics differed from another project summary, so this repository makes no headline performance claim. Run the notebook with a documented dataset version and split before citing results.

## Scope and limitations

This is an exploratory notebook, not a packaged application. Results depend on dataset provenance, split strategy, duplicate handling, threshold choice, and environment versions. Audit those choices before comparing models or using predictions operationally.

## Technologies

Python, pandas, NumPy, scikit-learn, TF-IDF, Logistic Regression, Random Forest, SVM, TensorFlow/Keras, GloVe, SHAP, Jupyter.
