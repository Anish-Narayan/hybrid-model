import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, chi2
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pickle  # Import pickle module

# ------------------ Load consistent split ------------------
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

X_train = train_df.drop("Target", axis=1)
y_train = train_df["Target"]
X_test  = test_df.drop("Target", axis=1)
y_test  = test_df["Target"]

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)

# ------------------ NB Feature Selection ------------------
def nb_feature_selection(X, y, top_k=20):
    scores = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    nb = GaussianNB()
    nb.fit(X_scaled, y)

    for col in X.columns:
        # Mutual Information
        mi = mutual_info_classif(X[[col]], y, discrete_features='auto')[0]
        # Pearson correlation
        try:
            corr, _ = pearsonr(X[col], y)
            corr_score = abs(corr)
        except:
            corr_score = 0
        # Chi-square
        try:
            chi_val = chi2(X[[col]], y)[0][0]
        except:
            chi_val = 0
        # NB variance ratio
        idx = list(X.columns).index(col)
        var_ratio = 1.0 / (nb.var_[:, idx].mean() + 1e-6)
        # Weighted normalized score
        total = (0.4 * mi) + (0.3 * corr_score) + (0.2 * var_ratio) + (0.1 * chi_val)
        scores[col] = total

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_features = [f for f, _ in ranked[:top_k]]
    return top_features, ranked

selected_features, ranking = nb_feature_selection(X_train, y_train, top_k=20)
print("Naive Bayes Selected Features:", selected_features)

# ------------------ SMOTE for balance ------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train[selected_features], y_train)

# ------------------ Random Forest on NB features ------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
)
rf.fit(X_train_res, y_train_res)

# ------------------ Evaluation ------------------
y_pred = rf.predict(X_test[selected_features])
print("\n📊 Model Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------ Explainability ------------------
rf_imp = pd.Series(rf.feature_importances_, index=selected_features).sort_values(ascending=False)
rank_df = pd.DataFrame(ranking, columns=["Feature","NB_score"]).set_index("Feature").loc[selected_features]
rank_df["RF_importance"] = rf_imp
rank_df.to_csv("feature_ranking_nb_rf.csv")
print("\n🔑 NB vs RF Feature Ranking saved → feature_ranking_nb_rf.csv")

plt.figure(figsize=(10,6))
sns.barplot(x=rf_imp.values, y=rf_imp.index, color="skyblue")
plt.title("Random Forest Importances (NB-selected features)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("rf_feature_importances.png")
plt.show()

# ------------------ Save the model, scaler, and encoder ------------------

# Create a dictionary to save the objects
model_objects = {
    'random_forest_model': rf,
    'label_encoder': le,
    'scaler': StandardScaler().fit(X_train[selected_features]),  # Recreate the scaler
}

# Save the objects to a pickle file
with open("model_objects.pkl", "wb") as f:
    pickle.dump(model_objects, f)

print("🔐 Model, scaler, and encoder saved to model_objects.pkl")
