# nb_guided_rf_plus.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, chi2
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --------------------------
# STEP 1: Load data
# --------------------------
df = pd.read_csv("data.csv", sep=";")

# Encode target
le = LabelEncoder()
df['Target'] = le.fit_transform(df['Target'])

X = df.drop("Target", axis=1)
y = df["Target"]

# --------------------------
# STEP 2: Custom NB Feature Scoring
# --------------------------
def nb_feature_selection(X, y, top_k=15):
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

        # Chi-square (categorical relevance)
        chi_val = 0
        try:
            chi_val = chi2(X[[col]], y)[0][0]
        except:
            pass

        # Variance score from NB
        idx = list(X.columns).index(col)
        var_ratio = 1.0 / (nb.var_[:, idx].mean() + 1e-6)

        # Normalize all scores
        total_score = (mi / (mi + 1e-6)) * 0.4 \
                    + (corr_score / (corr_score + 1e-6)) * 0.3 \
                    + (var_ratio / (var_ratio + 1e-6)) * 0.2 \
                    + (chi_val / (chi_val + 1e-6)) * 0.1

        scores[col] = total_score

    sorted_feats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_features = [f for f, s in sorted_feats[:top_k]]

    return top_features, sorted_feats

# --------------------------
# STEP 3: Feature selection
# --------------------------
selected_features, ranking = nb_feature_selection(X, y, top_k=20)
print("Naive Bayes Selected Features:", selected_features)

# --------------------------
# STEP 4: Train/Test Split + SMOTE
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X[selected_features], y, test_size=0.2, random_state=42, stratify=y
)

# Oversample minorities
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --------------------------
# STEP 5: Train Random Forest restricted to NB features
# --------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train_res, y_train_res)

# --------------------------
# STEP 6: Evaluation
# --------------------------
y_pred = rf.predict(X_test)

print("\n📊 Model Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --------------------------
# STEP 7: Explainability
# --------------------------
rf_importances = pd.Series(rf.feature_importances_, index=selected_features).sort_values(ascending=False)

# Compare NB vs RF ranking
ranking_df = pd.DataFrame(ranking, columns=["Feature", "NB_score"])
ranking_df = ranking_df.set_index("Feature").loc[selected_features]
ranking_df["RF_importance"] = rf_importances

print("\n🔑 Feature Importance Comparison (Top NB-selected)")
print(ranking_df.sort_values("NB_score", ascending=False).head(15))

# Save ranking
ranking_df.to_csv("feature_ranking_nb_rf.csv")

# --------------------------
# STEP 8: Plot
# --------------------------
plt.figure(figsize=(10,6))
sns.barplot(x=rf_importances.values, y=rf_importances.index, color="skyblue")
plt.title("Random Forest Importances (NB-selected features)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("rf_feature_importances.png")
plt.show()
