import re
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import preprocess_dataframe


# -----------------------------
# 1) Tokenization + Vocabulary
# -----------------------------
def tokenize(text):
    return text.split()


def build_vocab(texts, min_freq=2):
    counter = Counter()

    for text in texts:
        counter.update(tokenize(text))

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokenize(text)]


def pad_sequence(seq, max_len, pad_idx=0):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


# -----------------------------
# 2) Dataset
# -----------------------------
class ClickbaitDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=20):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = encode_text(self.texts[idx], self.vocab)
        padded = pad_sequence(encoded, self.max_len, pad_idx=self.vocab["<PAD>"])

        x = torch.tensor(padded, dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y


# -----------------------------
# 3) LSTM Classifier
# -----------------------------
class ClickbaitLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_layers=1, dropout=0.2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        embedded = self.embedding(x)             # [B, T, E]
        _, (hidden, _) = self.lstm(embedded)    # hidden: [num_layers, B, H]
        last_hidden = hidden[-1]                # [B, H]
        logits = self.fc(self.dropout(last_hidden))  # [B, 2]
        return logits


# -----------------------------
# 4) Evaluation
# -----------------------------
def evaluate_model(model, data_loader, device, return_preds=False):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    if return_preds:
        return acc, prec, rec, f1, all_labels, all_preds

    return acc, prec, rec, f1


# -----------------------------
# 5) Training
# -----------------------------
def train_model(model, train_loader, val_loader, epochs=5, lr=0.001):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch + 1}:")
        print(f"  Training Loss: {total_loss / len(train_loader):.4f}")
        print(f"  Validation Accuracy:  {val_acc:.4f}")
        print(f"  Validation Precision: {val_prec:.4f}")
        print(f"  Validation Recall:    {val_rec:.4f}")
        print(f"  Validation F1:        {val_f1:.4f}")
        print()

    return model, device


# -----------------------------
# 6) Main
# -----------------------------
def main():
    train_df = preprocess_dataframe(pd.read_csv("data/train.csv"))
    val_df = preprocess_dataframe(pd.read_csv("data/validation.csv"))
    test_df = preprocess_dataframe(pd.read_csv("data/test.csv"))

    vocab = build_vocab(train_df["clean_text"], min_freq=1)
    print("Vocab size:", len(vocab))

    max_len = 20
    batch_size = 16

    train_dataset = ClickbaitDataset(train_df["clean_text"], train_df["label"], vocab, max_len=max_len)
    val_dataset = ClickbaitDataset(val_df["clean_text"], val_df["label"], vocab, max_len=max_len)
    test_dataset = ClickbaitDataset(test_df["clean_text"], test_df["label"], vocab, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ClickbaitLSTM(
        vocab_size=len(vocab),
        embedding_dim=64,
        hidden_dim=96,
        num_layers=1,
        dropout=0.2,
    )

    model, device = train_model(
        model,
        train_loader,
        val_loader,
        epochs=8,
        lr=0.005,
    )

    train_acc, train_prec, train_rec, train_f1 = evaluate_model(
    model, train_loader, device
)

    print("Training Metrics:")
    print(f"Accuracy:  {train_acc:.4f}")
    print(f"Precision: {train_prec:.4f}")
    print(f"Recall:    {train_rec:.4f}")
    print(f"F1 Score:  {train_f1:.4f}")
    print()

    test_acc, test_prec, test_rec, test_f1, test_labels, test_preds = evaluate_model(
        model, test_loader, device, return_preds=True
    )
    import os

    os.makedirs("outputs", exist_ok=True)

    pred_df = pd.DataFrame({
        "text": test_df["text"].tolist(),
        "true_label": test_labels,
        "pred_label": test_preds,
    })

    pred_df.to_csv("outputs/lstm_test_predictions.csv", index=False)
    print("Saved predictions to outputs/lstm_test_predictions.csv")

    print("Test Metrics:")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")


if __name__ == "__main__":
    main()