from pydantic import BaseModel, field_validator

from src.lib.datasets import DATASETS_MAP
from src.lib.models import MODELS_MAP
from src.lib.models.interfaces import ModelConfig

class DatasetOptions(BaseModel):
    name: str

    @field_validator('name')
    def validate_dataset_name(cls, v):
        if v not in DATASETS_MAP:
            raise ValueError(f'Dataset must be one of {DATASETS_MAP.keys()}')
        return v

class ModelOptions(BaseModel):
    name: str
    configuration: ModelConfig

    @field_validator('name')
    def validate_model_name(cls, v):
        if v not in MODELS_MAP:
            raise ValueError(f'Model must be one of {MODELS_MAP.keys()}')
        return v
    
    @field_validator('configuration')
    def validate_model_config(cls, v):
        # TODO: validate model config based on the model name
        return v


class TrainRequest(BaseModel):
    dataset: DatasetOptions
    model: ModelOptions


class PredictRequest(BaseModel):
    dataset: list[str]
    model_artifact_name: str

    @field_validator("model_artifact_name")
    def validate_model_artifact_name(cls, v):
        # TODO: validate model artifact name
        return v
