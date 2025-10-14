# split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv", sep=";")

# Keep the target column
X = df.drop("Target", axis=1)
y = df["Target"]

# Create one consistent split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save them for reuse
X_train["Target"] = y_train
X_test["Target"] = y_test

X_train.to_csv("train.csv", index=False)
X_test.to_csv("test.csv", index=False)

print("✅ Data split saved: train.csv & test.csv")

