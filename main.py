from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from src.lib.datasets import DATASETS_MAP
from src.lib.models import MODELS_MAP
from src.lib.trainer import Trainer, Metrics
from src.lib.datasets.interfaces import Dataset
from src.lib.web.interfaces import TrainRequest, PredictRequest
from src.lib.web.handlers import train_handler, predict_handler

app = FastAPI()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/datasets")
async def get_datasets():
    return {"datasets": list(DATASETS_MAP.keys())}


@app.get("/models")
async def get_models():
    return {"models": list(MODELS_MAP.keys())}


@app.post("/models/train")
async def train(request: TrainRequest):
    return train_handler(request)


@app.post("/models/predict")
async def predict(request: PredictRequest):
    return predict_handler(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
