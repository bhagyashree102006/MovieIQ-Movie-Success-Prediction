import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
import ast

# -------------------------------------
# Load Clean Dataset
# -------------------------------------

df = pd.read_csv("C:/Users/Bhagyashree/OneDrive/Desktop/MovieIQ_Project/data/clean_movies.csv")

print("="*60)
print("STATISTICAL TESTING")
print("="*60)

# =====================================
# T - TEST
# =====================================

print("\nT-TEST")
print("-"*40)

success_movies = df[df["success"] == 1]["popularity"]
failure_movies = df[df["success"] == 0]["popularity"]

t_stat, p_value = ttest_ind(success_movies, failure_movies)

print("T Statistic :", round(t_stat,4))
print("P Value     :", round(p_value,4))

print("\nNull Hypothesis (H0):")
print("Popularity has no significant effect on movie success.")

print("\nAlternative Hypothesis (H1):")
print("Popularity significantly affects movie success.")

alpha = 0.05

if p_value < alpha:
    print("\nResult : Reject H0")
    print("Conclusion : Popularity is significantly related to success.")
else:
    print("\nResult : Fail to Reject H0")
    print("Conclusion : Popularity is NOT significantly related to success.")

# =====================================
# CHI-SQUARE TEST
# =====================================

print("\n")
print("="*60)
print("CHI-SQUARE TEST")
print("="*60)

# Convert genre string into first genre name
def get_first_genre(x):
    try:
        genre_list = ast.literal_eval(x)
        if len(genre_list) > 0:
            return genre_list[0]["name"]
        else:
            return "Unknown"
    except:
        return "Unknown"

df["genre"] = df["genres"].apply(get_first_genre)

contingency_table = pd.crosstab(df["genre"], df["success"])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Chi Square Statistic :", round(chi2,4))
print("P Value              :", round(p,4))

print("\nNull Hypothesis (H0):")
print("Genre and Success are independent.")

print("\nAlternative Hypothesis (H1):")
print("Genre and Success are associated.")

if p < alpha:
    print("\nResult : Reject H0")
    print("Conclusion : Genre is associated with movie success.")
else:
    print("\nResult : Fail to Reject H0")
    print("Conclusion : Genre is NOT associated with movie success.")

print("\n")
print("="*60)
print("Statistical Testing Completed Successfully")
print("="*60)