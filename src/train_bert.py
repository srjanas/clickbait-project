from transformers import BertTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import numpy as np
import pandas as pd 
import os

# -----------------------------
# 1) Load Tokenizer
# -----------------------------
def load_tokenizer(model_name="bert-base-uncased"):
    """
    Load a pretrained BERT tokenizer
    """
    return BertTokenizer.from_pretrained(model_name)

# -----------------------------
# 2) Tokenization Function
# -----------------------------
def tokenize_function(examples, tokenizer, max_length=64):
    """
    Tokenize text data for BERT input

    Applies padding and truncation to ensure uniform sequence length.

    Args:
        examples: Batch of examples containing "text"
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
    
    Returns:
        dict: Tokenized output including input_ids and attention_mask
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

# -----------------------------
# 3) Prepare Dataset
# -----------------------------
def prepare_dataset(dataset, tokenizer):
    """
    Apply tokenization and set PyTorch format for a dataset

    Args:
        dataset: Hugging Face dataset split
        tokenizer: Tokenizer instance

    Returns:
        Dataset: Tokenized dataset
    """
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    return dataset

# -----------------------------
# 4) Load Model
# -----------------------------
def load_model(model_name="bert-base-uncased", num_labels=2):
    """
    Load a pretrained BERT model for sequence classification

    Args:
        model_name: Hugging Face model name
        num_labels: Number of output classes

    Returns:
        AutoModelForSequenceClassification: BERT classification model
    """
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

# -----------------------------
# 5) Compute Metrics
# -----------------------------
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for Trainer

    Args:
        eval_pred: (logits, labels)

    Returns:
        dict: Dictionary containing accuracy, F1, precision and recall
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return{
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# -----------------------------
# 6) Train Model
# -----------------------------
def train_model(model, train_ds, val_ds, training_args):
    """
    Train a BERT model using Hugging Face Trainer.

    Args:
        model: BERT model
        train_ds: Training dataset
        val_ds: Validation dataset
        training_args: TrainingArguments object

    Returns:
        Trainer: Trained Trainer object
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer

def evaluate_model(model, dataset, texts, name, batch_size=16):
    """
    Evaluate a trained BERT model on a dataset.

    Args:
        model: Trained model
        dataset: Dataset to evaluate
        batch_size (int): Batch size for evaluation

    Returns:
        dict: Dictionary with accuracy, precision, recall, and F1 score
    """
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary"
    )
    accuracy = accuracy_score(all_labels, all_preds)

    pred_df = pd.DataFrame({
        "text": texts,
        "true_label": all_labels,
        "pred_label": all_preds
    })

    os.makedirs("outputs", exist_ok=True)
    pred_df.to_csv(f"outputs/bert_{name}_predictions.csv", index=False)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

def main():
    """
    Full workflow for BERT 
    """
    # Load tokenizer
    tokenizer = load_tokenizer()

    # Load datasets
    dataset = load_dataset("christinacdl/clickbait_detection_dataset")
    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    test_ds = dataset["test"]

    val_texts = val_ds["text"]
    test_texts = test_ds["text"]

    # Prepare datasets
    train_ds = prepare_dataset(train_ds, tokenizer)
    val_ds = prepare_dataset(val_ds, tokenizer)
    test_ds = prepare_dataset(test_ds, tokenizer)

    # Load model
    model = load_model()

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=False
    )

    # Train model
    trainer = train_model(model, train_ds, val_ds, training_args)
    log_history = trainer.state.log_history

    # convert to dataframe
    log_df = pd.DataFrame(log_history)

    log_df = log_df[log_df["epoch"].notna()]

    # separate train vs eval rows
    train_df = log_df[log_df["loss"].notna()].copy()
    val_df = log_df[log_df["eval_loss"].notna()].copy()

    # merge them on epoch
    bert_metrics_df = pd.merge(
        train_df[["epoch", "loss"]],
        val_df[["epoch", "eval_loss", "eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]],
        on="epoch",
        how="left"
    )

    # rename columns nicely
    bert_metrics_df = bert_metrics_df.rename(columns={
        "loss": "train_loss",
        "eval_loss": "val_loss",
        "eval_accuracy": "val_accuracy",
        "eval_f1": "val_f1",
        "eval_precision": "val_precision",
        "eval_recall": "val_recall",
    })

    os.makedirs("outputs", exist_ok=True)
    bert_metrics_df.to_csv("outputs/bert_metrics.csv", index=False)

    print("\nValidation Results:")
    evaluate_model(trainer.model, val_ds, val_texts, "val")

    print("\nTest Results:")
    evaluate_model(trainer.model, test_ds, test_texts, "test")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    main()
