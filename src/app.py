from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# load model
model = pickle.load(open("artifacts/model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "ML API is running 🚀"}

@app.post("/predict")
def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)

    return {"prediction": float(prediction[0])}