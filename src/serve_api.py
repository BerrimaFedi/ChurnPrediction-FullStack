# -*- coding: utf-8 -*-
import mlflow.sklearn
import pandas as pd
import uvicorn
import os
from fastapi import FastAPI
from pydantic import BaseModel

os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

app = FastAPI(title="Churn Prediction API", version="1.0")

print("Chargement du modele depuis MLflow Registry...")
model = mlflow.sklearn.load_model("models:/churn_adaboost_production/Production")
print("Modele charge avec succes !")

class PredictRequest(BaseModel):
    dataframe_split: dict

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/invocations")
def predict(request: PredictRequest):
    columns = request.dataframe_split["columns"]
    data = request.dataframe_split["data"]
    df = pd.DataFrame(data, columns=columns)
    predictions = model.predict(df).tolist()
    probas = model.predict_proba(df)[:, 1].tolist()
    return {"predictions": predictions, "probabilities": probas}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=1234)