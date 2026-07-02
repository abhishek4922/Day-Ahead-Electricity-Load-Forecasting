from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features])
    return prediction.tolist()