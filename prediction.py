import joblib
import pandas as pd

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("models/random_forest_model.pkl")

print("="*50)
print("MOVIE SUCCESS PREDICTION")
print("="*50)

# -----------------------------
# User Input
# -----------------------------
budget = float(input("Enter Budget : "))
popularity = float(input("Enter Popularity : "))
runtime = float(input("Enter Runtime : "))
vote_average = float(input("Enter Vote Average : "))

# -----------------------------
# Create DataFrame
# -----------------------------
movie = pd.DataFrame({
    "budget":[budget],
    "popularity":[popularity],
    "runtime":[runtime],
    "vote_average":[vote_average]
})

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(movie)

print("\nPrediction Result")

if prediction[0] == 1:
    print("✅ Movie is likely to be SUCCESSFUL")
else:
    print("❌ Movie is likely to be NOT SUCCESSFUL")