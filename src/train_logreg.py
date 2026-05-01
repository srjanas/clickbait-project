import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from preprocess import preprocess_dataframe


def main():
    train_df = preprocess_dataframe(pd.read_csv("data/train.csv"))
    val_df = preprocess_dataframe(pd.read_csv("data/validation.csv"))
    test_df = preprocess_dataframe(pd.read_csv("data/test.csv"))

    X_train_text = train_df["clean_text"]
    y_train = train_df["label"]

    X_val_text = val_df["clean_text"]
    y_val = val_df["label"]

    X_test_text = test_df["clean_text"]
    y_test = test_df["label"]

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    print("Validation Metrics:")
    print_metrics(y_val, val_preds)

    print("\nTest Metrics:")
    print_metrics(y_test, test_preds)

    os.makedirs("outputs", exist_ok=True)
    pred_df = pd.DataFrame({
    "text": test_df["text"],
    "true_label": y_test,
    "pred_label": test_preds,
    })

    pred_df.to_csv("outputs/logreg_test_predictions.csv", index=False)
    print("Saved predictions to outputs/logreg_test_predictions.csv")

    print("\nClassification Report (Test):")
    print(classification_report(y_test, test_preds))

def print_metrics(y_true, y_pred):
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")

os.makedirs("outputs", exist_ok=True)

if __name__ == "__main__":
    main()