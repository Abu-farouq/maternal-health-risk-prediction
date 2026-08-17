import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the cleaned data
df = pd.read_csv(r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\Maternal_Health_Risk_Data_Set_clean.csv")

# Features and target
X = df[["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]]
y = df["RiskLevel"]

# Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Same split as before, for consistency with our documented results
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train the final Random Forest model (same settings as our evaluated version)
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6)
model.fit(X_train, y_train)

# Save the trained model AND the label encoder together
# We need the label encoder too, so predictions (0/1/2) can be converted back to "low risk"/"mid risk"/"high risk"
joblib.dump(model, r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\maternal_risk_model.pkl")
joblib.dump(le, r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\label_encoder.pkl")

print("Model and label encoder saved successfully.")
print("Model file: maternal_risk_model.pkl")
print("Encoder file: label_encoder.pkl")