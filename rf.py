import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load consistent split
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

X_train = train_df.drop("Target", axis=1)
y_train = train_df["Target"]
X_test  = test_df.drop("Target", axis=1)
y_test  = test_df["Target"]

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
