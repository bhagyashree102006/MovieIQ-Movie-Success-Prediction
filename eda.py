import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------------------
# Create assets folder if it doesn't exist
# ---------------------------------------
os.makedirs("assets", exist_ok=True)

# ---------------------------------------
# Load Clean Dataset
# ---------------------------------------
df = pd.read_csv("C:/Users/Bhagyashree/OneDrive/Desktop/MovieIQ_Project/data/clean_movies.csv")

print("Dataset Loaded Successfully")

# ---------------------------------------
# Dataset Information
# ---------------------------------------
print(df.head())
print(df.describe())

# ---------------------------------------
# Scatter Plot
# Budget vs Revenue
# ---------------------------------------
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=df,
    x="budget",
    y="revenue",
    alpha=0.6
)

plt.title("Budget vs Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")

plt.tight_layout()
plt.savefig("assets/budget_vs_revenue.png")
plt.show()

# ---------------------------------------
# Most Common Genres
# ---------------------------------------

genre = df["genres"].str.split("|").explode()

genre.value_counts().head(10).plot(
    kind="bar",
    figsize=(10,6)
)

plt.title("Top 10 Movie Genres")
plt.xlabel("Genre")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("assets/top_genres.png")
plt.show()

# ---------------------------------------
# Success Distribution
# ---------------------------------------

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="success"
)

plt.title("Movie Success Distribution")

plt.tight_layout()
plt.savefig("assets/success_distribution.png")
plt.show()

# ---------------------------------------
# Popularity vs Success
# ---------------------------------------

plt.figure(figsize=(8,6))

sns.boxplot(
    data=df,
    x="success",
    y="popularity"
)

plt.title("Popularity vs Success")

plt.tight_layout()
plt.savefig("assets/popularity_vs_success.png")
plt.show()

# ---------------------------------------
# Runtime vs Success
# ---------------------------------------

plt.figure(figsize=(8,6))

sns.boxplot(
    data=df,
    x="success",
    y="runtime"
)

plt.title("Runtime vs Success")

plt.tight_layout()
plt.savefig("assets/runtime_vs_success.png")
plt.show()

# ---------------------------------------
# Vote Average vs Success
# ---------------------------------------

plt.figure(figsize=(8,6))

sns.boxplot(
    data=df,
    x="success",
    y="vote_average"
)

plt.title("Vote Average vs Success")

plt.tight_layout()
plt.savefig("assets/vote_average_vs_success.png")
plt.show()

# ---------------------------------------
# Correlation Heatmap
# ---------------------------------------

numeric = df.select_dtypes(include="number")

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("assets/correlation_heatmap.png")
plt.show()

print("\nAll graphs saved successfully inside assets folder.")