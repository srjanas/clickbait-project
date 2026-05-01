train = pd.read_csv("data/train.csv")

print(train.head())
print("\nShape:", train.shape)
print("\nLabel counts:")
print(train["label"].value_counts())