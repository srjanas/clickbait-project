from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from preprocess import preprocess_dataframe

import pandas as pd
import os

# -----------------------------
# 1) Build Pipeline
# -----------------------------
def build_pipeline(C=0.1):
    """
    Create a TF-IDF + Linear SVM pipeline for text classification.

    Args:
        C: Regularization strength for LinearSVC.

    Returns:
        pipeline: Configured sklearn pipeline.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1,2),
            min_df=2,
            max_df=0.9
        )),
        ("svm", LinearSVC(C=C))
    ])

    return pipeline

# -----------------------------
# 2) Train Model
# -----------------------------
def train_model(pipeline, train_df):
    """
    Train pipeline on cleaned data

    Args:
        pipeline: model pipeline
        train_df: training dataframe
    
    Returns:
        pipeline: trained model
    """
    pipeline.fit(train_df["text"], train_df["label"])
    return pipeline

# -----------------------------
# 3) Evaluate Model
# -----------------------------
def evaluate_model(model, df, name):
    """
    Evaluate model performance and print metrics

    Args:
        model: trained model
        df: dataset with cleaned text
        name: dataset name for printing
    """
    preds = model.predict(df["clean_text"])
    print(f"\n{name} Performance:")
    print(classification_report(df["label"], preds))
    return preds

# -----------------------------
# 4) Hyperparameter Tuning
# -----------------------------
def tune_hyperparameters(pipeline, train_df):
    """
    Tune SVM regularization using GridSearchCV

    Args:
        pipeline: base pipeline
        train_df: train dataframe
    
    Returns:
        pipelien: best-performing model
    """
    param_grid = {
    "svm__C": [0.01, 0.1, 1, 10]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
    grid.fit(train_df["text"], train_df["label"])
    print("\nBest Parameters:", grid.best_params_)

    return grid.best_estimator_

def main():
    """
    Full workflow
    """
    # Preprocess
    train_df = preprocess_dataframe(pd.read_csv("data/train.csv"))
    val_df = preprocess_dataframe(pd.read_csv("data/validation.csv"))
    test_df = preprocess_dataframe(pd.read_csv("data/test.csv"))

    # Build pipeline
    pipeline = build_pipeline()

    # Train model
    model = train_model(pipeline, train_df)

    # Evaluate baseline model
    evaluate_model(model, val_df, "Validation")

    # Tune hyperparameter
    best_model = tune_hyperparameters(pipeline, train_df)

    # Final evaluation
    y_test_pred = evaluate_model(best_model, test_df, "Test")
    pred_df = pd.DataFrame({
        "text": test_df["text"],
        "true_label": test_df["label"],
        "pred_label": y_test_pred
    })

    os.makedirs("outputs", exist_ok=True)
    pred_df.to_csv("outputs/svm_test_predictions.csv", index=False)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()
