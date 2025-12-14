import argparse
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.data_path)

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.joblib")

    mlflow.log_artifact("artifacts/model.joblib")
    mlflow.sklearn.log_model(model, "model")
