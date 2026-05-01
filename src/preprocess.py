import re
import string
import pandas as pd

def clean_text(text: str) -> str:
    """
    Basic text cleaning for clickbait headlines.
    """
    if pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"<.*?>", "", text)                   # remove HTML tags
    text = re.sub(r"\d+", "", text)                     # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()            # normalize whitespace
    return text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["text"])
    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = df["text"].apply(clean_text)
    return df