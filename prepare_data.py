import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/dataset.csv")

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['author']
)

train_df.to_csv("results/train.csv", index=False)
test_df.to_csv("results/test.csv", index=False)

print(f"Дані розділено на train ({len(train_df)}), test ({len(test_df)}).")