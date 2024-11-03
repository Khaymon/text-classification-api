from pydantic import BaseModel, field_validator

from src.lib.datasets import DATASETS_MAP
from src.lib.models import MODELS


class DatasetOptions(BaseModel):
    name: str

    @field_validator('name')
    def validate_dataset_name(cls, v):
        if v not in DATASETS_MAP:
            raise ValueError(f'Dataset must be one of {DATASETS_MAP.keys()}')
        return v

class ModelOptions(BaseModel):
    name: str
    config: dict

    @field_validator('name')
    def validate_model_name(cls, v):
        if v not in MODELS:
            raise ValueError(f'Model must be one of {MODELS}')
        return v
    
    @field_validator('config')
    def validate_model_config(cls, v):
        # TODO: validate model config based on the model name
        if not isinstance(v, dict):
            raise ValueError('Model config must be a dictionary')
        return v


class TrainRequest(BaseModel):
    dataset: DatasetOptions
    model: ModelOptions


class PredictRequest(BaseModel):
    dataset: DatasetOptions
    model: ModelOptions
