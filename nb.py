import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load pre-split data
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

X_train = train_df.drop("Target", axis=1)
y_train = train_df["Target"]
X_test  = test_df.drop("Target", axis=1)
y_test  = test_df["Target"]

# Encode categorical target
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)

# Scale continuous features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

# Evaluate
y_pred = nb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
