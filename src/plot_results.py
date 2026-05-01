import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score


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

    results = pd.DataFrame([
        {"Model": "Logistic Regression", **logreg_metrics},
        {"Model": "LSTM", **lstm_metrics},
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

    x = range(len(metrics))
    width = 0.35

    logreg_values = results.loc[results["Model"] == "Logistic Regression", metrics].values[0]
    lstm_values = results.loc[results["Model"] == "LSTM", metrics].values[0]

    plt.bar([i - width / 2 for i in x], logreg_values, width=width, label="Logistic Regression")
    plt.bar([i + width / 2 for i in x], lstm_values, width=width, label="LSTM")

    plt.xticks(list(x), metrics)
    plt.ylim(0.85, 1.0)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()

    plt.savefig("outputs/metric_comparison.png", dpi=300)
    plt.show()

    print("Saved plot to outputs/metric_comparison.png")


def plot_confusion(csv_path, model_name, output_path):
    df = pd.read_csv(csv_path)

    cm = confusion_matrix(df["true_label"], df["pred_label"])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NOT", "CLICKBAIT"])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

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

    save_error_examples()


if __name__ == "__main__":
    main()