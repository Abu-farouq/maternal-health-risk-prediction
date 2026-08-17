import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

df = pd.read_csv(r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\Maternal Health Risk Data Set.csv")
df = df.drop_duplicates()
df = df[df['HeartRate'] != 7]

X = df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
y = df['RiskLevel']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", dict(zip(le.classes_, range(len(le.classes_)))))

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nBefore SMOTE - training class counts: {pd.Series(y_train).value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - training class counts: {pd.Series(y_train_res).value_counts().to_dict()}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              random_state=42, eval_metric='mlogloss'),
}

scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

for name, model in models.items():
    print(f"\n{'='*60}\n{name} (with SMOTE)\n{'='*60}")
    if name == "Logistic Regression":
        model.fit(X_train_res_scaled, y_train_res)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train_res, y_train_res)
        preds = model.predict(X_test)
    print(classification_report(y_test, preds, target_names=le.classes_, digits=3))