from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from src.lib.datasets import DATASETS_MAP
from src.lib.models import MODELS_MAP
from src.lib.trainer import Trainer, Metrics
from src.lib.datasets.interfaces import Dataset
from src.lib.web.interfaces import TrainRequest, PredictRequest
from src.lib.web.handlers import train_handler, predict_handler, list_model_artifacts_handler

app = FastAPI()


@app.get("/health")
async def health_check():
    """
    Endpoint to check the health status of the application.

    Returns:
        dict: A dictionary indicating the health status.
    """
    return {"status": "healthy"}


@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message.

    Returns:
        dict: A dictionary containing the welcome message.
    """
    return {"message": "Hello World"}


@app.get("/datasets")
async def get_datasets():
    """
    Retrieve a list of available datasets.

    Returns:
        dict: A dictionary containing the list of dataset names.
    """
    return {"datasets": list(DATASETS_MAP.keys())}


@app.get("/models")
async def get_models():
    """
    Retrieve a list of available models.

    Returns:
        dict: A dictionary containing the list of model names.
    """
    return {"models": list(MODELS_MAP.keys())}


@app.post("/models/train")
async def train(request: TrainRequest):
    """
    Endpoint to train a new model based on the provided TrainRequest.

    Args:
        request (TrainRequest): The request containing dataset and model configurations.
    
    Returns:
        dict: A dictionary containing the training results.
    """
    return train_handler(request)


@app.post("/models/predict")
async def predict(request: PredictRequest):
    """
    Endpoint to make predictions using a specified model artifact.

    Args:
        request (PredictRequest): The request containing data and the model artifact name.
    
    Returns:
        dict: A dictionary containing the predictions.
    """
    return predict_handler(request)


@app.get("/models/artifacts")
async def list_model_artifacts():
    """
    Endpoint to list all available model artifacts.

    Returns:
        dict: A dictionary containing a list of artifact names.
    """
    return list_model_artifacts_handler()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
