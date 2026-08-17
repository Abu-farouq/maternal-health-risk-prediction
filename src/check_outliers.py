import pandas as pd

df = pd.read_csv(r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\Maternal_Health_Risk_Data_Set_clean.csv")

# Look at the lowest HeartRate values
print("=== LOWEST HEART RATE VALUES ===")
print(df.nsmallest(10, "HeartRate")[["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate", "RiskLevel"]])

# Look at the full range
print("\n=== HEART RATE STATS ===")
print(df["HeartRate"].describe())

# Also check Age, since we saw some odd outliers there (60-70 yrs)
print("\n=== HIGHEST AGE VALUES ===")
print(df.nlargest(10, "Age")[["Age", "RiskLevel"]])
# Drop the row with the impossible HeartRate value
df = df[df["HeartRate"] != 7]

print(f"After removing invalid HeartRate row: {len(df)} rows")

df.to_csv(r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\Maternal_Health_Risk_Data_Set_clean.csv", index=False)
print("Saved final cleaned file.")