import pandas as pd

df = pd.read_csv(r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\Maternal_Health_Risk_Data_Set_clean.csv")

features = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

# Compare summary stats for low risk vs mid risk specifically
low = df[df["RiskLevel"] == "low risk"]
mid = df[df["RiskLevel"] == "mid risk"]

print("=== LOW RISK STATS ===")
print(low[features].describe().loc[["mean", "50%", "std"]])

print("\n=== MID RISK STATS ===")
print(mid[features].describe().loc[["mean", "50%", "std"]])

# Check the overlap directly: for each feature, what % of mid risk values
# fall within the low risk interquartile range (25th-75th percentile)?
print("\n=== OVERLAP CHECK: % of MID RISK values inside LOW RISK's IQR ===")
for feature in features:
    q1 = low[feature].quantile(0.25)
    q3 = low[feature].quantile(0.75)
    in_range = mid[feature].between(q1, q3).mean() * 100
    print(f"{feature}: low risk IQR = [{q1}, {q3}]  ->  {in_range:.1f}% of mid risk values fall inside it")