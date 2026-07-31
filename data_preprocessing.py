import pandas as pd

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("C:/Users/Bhagyashree/OneDrive/Desktop/MovieIQ_Project/data/movies.csv")
print("Dataset Loaded Successfully")
print("-" * 50)

# -----------------------------
# Display Basic Information
# -----------------------------
print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nInformation:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# -----------------------------
# Check Missing Values
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Remove Missing Values
# -----------------------------
df = df.dropna()

# -----------------------------
# Remove Invalid Budget
# -----------------------------
df = df[df["budget"] > 0]

# -----------------------------
# Remove Invalid Revenue
# -----------------------------
df = df[df["revenue"] > 0]

# -----------------------------
# Create Success Column
# -----------------------------
df["success"] = (df["revenue"] > df["budget"]).astype(int)

print("\nSuccess Distribution:")
print(df["success"].value_counts())

# -----------------------------
# Save Clean Dataset
# -----------------------------
df.to_csv("data/clean_movies.csv", index=False)

print("\nClean dataset saved successfully!")