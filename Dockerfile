FROM python:3.10-slim

WORKDIR /app

# copy MLProject ke container
COPY MLProject /app/MLProject

# install dependency
RUN pip install --no-cache-dir \
    mlflow \
    scikit-learn \
    pandas \
    numpy \
    joblib

# default command
CMD ["mlflow", "run", "MLProject", "--env-manager=local"]
