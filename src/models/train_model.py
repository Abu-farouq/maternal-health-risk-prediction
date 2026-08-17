import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_splistratify=y_encoded
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

# Scale features - fit ONLY on training data, then apply to both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print("\n=== DEBUG: FIRST ROW BEFORE/AFTER SCALING ===")
print("Before:", X_train.iloc[0].values)
print("After:", X_train_scaled[0])
X_test_scaled = scaler.transform(X_test)

# Train baseline model on scaled data
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)

print("\n=== ACCURACY ===")
print(accuracy_score(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))t(
    X, y_encoded, test_size=0.2, random_state=42, 