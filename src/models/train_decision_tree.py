import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the cleaned data
df = pd.read_csv(r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\Maternal_Health_Risk_Data_Set_clean.csv")

# Features and target
X = df[["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]]
y = df["RiskLevel"]

# Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", list(le.classes_))

# Split into train/test sets - SAME split as Logistic Regression for fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

# Decision Tree does NOT need scaling - it splits on raw thresholds, not distances
model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("\n=== ACCURACY ===")
print(accuracy_score(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

# Feature importance - which vitals matter most to this model?
print("\n=== FEATURE IMPORTANCE ===")
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importance)