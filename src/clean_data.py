import pandas as pd

df = pd.read_csv(r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\Maternal Health Risk Data Set.csv")

print(f"Before: {len(df)} rows")

df = df.drop_duplicates()

print(f"After: {len(df)} rows")

df.to_csv(r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\Maternal_Health_Risk_Data_Set_clean.csv", index=False)

print("Saved cleaned file.")