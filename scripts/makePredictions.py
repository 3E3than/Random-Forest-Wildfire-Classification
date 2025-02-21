import joblib
import numpy as np

def loadModel():
    model = joblib.load("models/random_forest_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    return model, scaler

def predictFire(features):
    model, scaler = loadModel()
    scaled_features = scaler.transform(np.array([features]))
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0].max()
    return prediction, probability