import os
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# PATH CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(
    BASE_DIR,
    "heart_disease_preprocessing",
    "heart_disease_clean.csv"
)

# =========================
# DAGSHUB (MLFLOW ONLINE)
# =========================
dagshub.init(
    repo_owner="mkautsarpb",
    repo_name="Workflow-CI",   
    mlflow=True
)
mlflow.set_tracking_uri(
    "https://dagshub.com/mkautsarpb/Workflow-CI.mlflow"
)
mlflow.set_experiment("Heart Disease - CI Retraining")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# HYPERPARAMETER TUNING
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

# =========================
# TRAINING + LOGGING
# =========================
with mlflow.start_run():
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    # =========================
    # LOG PARAMETERS
    # =========================
    mlflow.log_params(grid.best_params_)

    # =========================
    # LOG METRICS
    # =========================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    # =========================
    # 🔥 WAJIB: LOG MODEL (UNTUK CI + DOCKER)
    # =========================
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

    # =========================
    # ARTEFAK TAMBAHAN 1: MODEL FILE
    # =========================
    model_path = os.path.join(BASE_DIR, "model.joblib")
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path)

    # =========================
    # ARTEFAK TAMBAHAN 2: CONFUSION MATRIX
    # =========================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
