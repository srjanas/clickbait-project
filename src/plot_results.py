import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(csv_path):
    df = pd.read_csv(csv_path)

    y_true = df["true_label"]
    y_pred = df["pred_label"]

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }


def save_results_table():
    logreg_metrics = compute_metrics("outputs/logreg_test_predictions.csv")
    lstm_metrics = compute_metrics("outputs/lstm_test_predictions.csv")
    svm_metrics = compute_metrics("outputs/svm_test_predictions.csv")
    bert_metrics = compute_metrics("outputs/bert_test_predictions.csv")

    results = pd.DataFrame([
        {"Model": "Logistic Regression", **logreg_metrics},
        {"Model": "LSTM", **lstm_metrics},
        {"Model": "SVM", **svm_metrics},
        {"Model": "BERT", **bert_metrics}
    ])

    os.makedirs("outputs", exist_ok=True)
    results.to_csv("outputs/results_table.csv", index=False)

    print("\nResults table:")
    print(results)
    print("\nSaved to outputs/results_table.csv")

    return results


def plot_metric_comparison(results):
    metrics = ["Accuracy", "Precision", "Recall", "F1"]

    plt.figure(figsize=(8, 5))

    x = np.arange(len(metrics))

    logreg_values = results.loc[results["Model"] == "Logistic Regression", metrics].values[0]
    lstm_values = results.loc[results["Model"] == "LSTM", metrics].values[0]
    svm_values = results.loc[results["Model"] == "SVM", metrics].values[0]
    bert_values = results.loc[results["Model"] == "BERT", metrics].values[0]

    width = 0.2
    plt.bar(x - 1.5*width, logreg_values, width, label="Logistic Regression", color="#9ecae1")
    plt.bar(x - 0.5*width, svm_values, width, label="SVM", color="#3182bd")
    plt.bar(x + 0.5*width, lstm_values, width, label="LSTM", color="#6baed6")
    plt.bar(x + 1.5*width, bert_values, width, label="BERT", color="#08519c")

    plt.xticks(list(x), metrics)
    plt.ylim(0.85, 1.0)
    plt.ylabel("Score (%)")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig("outputs/metric_comparison.png", dpi=300)
    plt.show()

    print("Saved plot to outputs/metric_comparison.png")


def plot_confusion(csv_path, model_name, output_path):
    df = pd.read_csv(csv_path)

    cm = confusion_matrix(df["true_label"], df["pred_label"])
    labels = ["Not Clickbait (0)", "Clickbait (1)"]

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix: {model_name}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved confusion matrix to {output_path}")


def save_error_examples():
    os.makedirs("outputs", exist_ok=True)

    for model_name, path in [
        ("logreg", "outputs/logreg_test_predictions.csv"),
        ("lstm", "outputs/lstm_test_predictions.csv"),
    ]:
        df = pd.read_csv(path)

        errors = df[df["true_label"] != df["pred_label"]].copy()
        errors.to_csv(f"outputs/{model_name}_errors.csv", index=False)

        print(f"Saved {model_name} errors to outputs/{model_name}_errors.csv")
        print(f"{model_name} number of errors: {len(errors)}")


def plot_bert_metrics():
    df = pd.read_csv("outputs/bert_metrics.csv")

    plt.figure(figsize=(9, 5))

    plt.plot(df["epoch"], df["train_loss"], marker='o', label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker='o', label="Validation Loss")

    plt.title("BERT Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("outputs/bert_metrics.png", dpi=300)
    plt.show()


def main():
    results = save_results_table()
    plot_metric_comparison(results)

    plot_confusion(
        "outputs/logreg_test_predictions.csv",
        "Logistic Regression",
        "outputs/logreg_confusion_matrix.png",
    )

    plot_confusion(
        "outputs/lstm_test_predictions.csv",
        "LSTM",
        "outputs/lstm_confusion_matrix.png",
    )

    plot_bert_metrics()

    save_error_examples()


if __name__ == "__main__":
    main()