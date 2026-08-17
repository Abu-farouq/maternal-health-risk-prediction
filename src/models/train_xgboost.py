import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# Load the cleaned data
df = pd.read_csv(r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\Maternal_Health_Risk_Data_Set_clean.csv")

# Features and target
X = df[["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]]
y = df["RiskLevel"]

# Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", list(le.classes_))

# Same split as before, for fair comparison across all four models
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

# XGBoost - gradient-boosted trees, builds trees sequentially to correct previous errors
model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="mlogloss")
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("\n=== ACCURACY ===")
print(accuracy_score(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== FEATURE IMPORTANCE ===")
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importance)