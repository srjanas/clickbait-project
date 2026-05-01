from datasets import load_dataset
import pandas as pd
from pathlib import Path
 
def load_clickbait_dataset():
    dataset = load_dataset("christinacdl/clickbait_detection_dataset")
    return dataset

def save_splits_to_csv(dataset, output_dir="data"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split in dataset.keys():
        df = pd.DataFrame(dataset[split])
        df.to_csv(output_path / f"{split}.csv", index=False)
        print(f"Saved {split} to {output_path / f'{split}.csv'}")

if __name__ == "__main__":
    ds = load_clickbait_dataset()
    save_splits_to_csv(ds)