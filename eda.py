import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("gas_well_data.csv")

# Descriptive statistics
print("Descriptive Statistics:")
print(data.describe())

# Correlation matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Pairplot
sns.pairplot(data, diag_kind="hist", plot_kws={"alpha": 0.5, "edgecolor": "k"})
plt.suptitle("Pairplot of Variables", y=1.02)
plt.show()

# Distribution plots
for col in data.columns:
    sns.histplot(data[col], kde=True, bins=30, color="blue", edgecolor="black")
    plt.title(f"Distribution of {col}")
    plt.show()
