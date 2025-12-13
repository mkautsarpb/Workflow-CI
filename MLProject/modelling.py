import os
import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR,
    "heart_disease_preprocessing",
    "heart_disease_clean.csv"
)

# =========================
# DAGSHUB INIT (ONLINE MLFLOW)
# =========================
dagshub.init(
    repo_owner="mkautsarpb",
    repo_name="Workflow-CI",
    mlflow=True
)

mlflow.set_experiment("Heart Disease - CI Retraining")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN + TUNING
# =========================
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear"]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring="accuracy"
)

with mlflow.start_run():
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    # MODEL ARTEFAK
    model_path = os.path.join(BASE_DIR, "model.joblib")
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)

    # CONFUSION MATRIX ARTEFAK
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")
