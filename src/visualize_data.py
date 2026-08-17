import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
df = pd.read_csv(r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\Maternal_Health_Risk_Data_Set_clean.csv")

# 1. Class balance chart
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="RiskLevel", order=["low risk", "mid risk", "high risk"], palette="viridis")
plt.title("Risk Level Class Balance")
plt.xlabel("Risk Level")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\chart_class_balance.png")
plt.show()

# 2. Feature distributions by risk level (boxplots)
features = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.boxplot(data=df, x="RiskLevel", y=feature, order=["low risk", "mid risk", "high risk"], ax=axes[i], palette="viridis")
    axes[i].set_title(f"{feature} by Risk Level")

plt.tight_layout()
plt.savefig(r"C:\Users\MY PC\Desktop\prof emu\PROJECTS\chart_feature_distributions.png")
plt.show()