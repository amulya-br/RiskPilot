import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("data/risk_dataset.csv")

print("Dataset Shape:", df.shape)

# Features
X = df[
    [
        "people_count",
        "density",
        "moving_people",
        "stationary_people",
        "avg_speed",
    ]
]

# Target
y = df["risk"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# Save model
joblib.dump(model, "models/risk_model.pkl")

print("\nModel Saved Successfully!")