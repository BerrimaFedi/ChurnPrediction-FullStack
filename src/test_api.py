# -*- coding: utf-8 -*-
import requests
import pandas as pd
from preprocessing import load_and_prepare

DATA_PATH = r"C:\project_ChurnPrediction\data\raw\Telecom Customers Churn.csv"
X_train, X_test, y_train, y_test = load_and_prepare(DATA_PATH)
X_sample = X_test.head(5)

payload = {
    "dataframe_split": {
        "columns": list(X_sample.columns),
        "data": X_sample.values.tolist()
    }
}

resp = requests.post("http://localhost:1234/invocations", json=payload)
print("Statut :", resp.status_code)
print("Predictions :", resp.json())