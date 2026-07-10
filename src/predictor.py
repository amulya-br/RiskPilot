import joblib
import numpy as np

risk_model = joblib.load("models/risk_model.pkl")

def predict(
    people,
    density,
    moving,
    stationary,
    speed
):

    X = np.array([[

        people,
        density,
        moving,
        stationary,
        speed

    ]])

    return risk_model.predict(X)[0]