import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -------------------------------------
# Create models folder if it doesn't exist
# -------------------------------------
os.makedirs("models", exist_ok=True)

# -------------------------------------
# Load Dataset
# -------------------------------------
df = pd.read_csv("data/clean_movies.csv")

print("=" * 60)
print("RANDOM FOREST MODEL TRAINING")
print("=" * 60)

# -------------------------------------
# Select Features
# -------------------------------------
X = df[[
    "budget",
    "popularity",
    "runtime",
    "vote_average"
]]

# Target Variable
y = df["success"]

print("\nFeatures Selected:")
print(X.head())

# -------------------------------------
# Train Test Split
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# -------------------------------------
# Random Forest Model
# -------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# -------------------------------------
# Prediction
# -------------------------------------
y_pred = model.predict(X_test)

# -------------------------------------
# Accuracy
# -------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

# -------------------------------------
# Classification Report
# -------------------------------------
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# -------------------------------------
# Confusion Matrix
# -------------------------------------
print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------
# Feature Importance
# -------------------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance\n")
print(importance)

# -------------------------------------
# Save Model
# -------------------------------------
joblib.dump(model, "models/random_forest_model.pkl")

print("\nModel Saved Successfully!")
print("Location : models/random_forest_model.pkl")