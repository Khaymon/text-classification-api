from pydantic import BaseModel, field_validator

from src.lib.datasets import DATASETS_MAP
from src.lib.models import MODELS_MAP
from src.lib.models.interfaces import ModelConfig

class DatasetOptions(BaseModel):
    name: str

    @field_validator('name')
    def validate_dataset_name(cls, v):
        """
        Validate that the dataset name exists in DATASETS_MAP.

        Args:
            v (str): The name of the dataset.
        
        Returns:
            str: The validated dataset name.
        
        Raises:
            ValueError: If the dataset name is not in DATASETS_MAP.
        """
        if v not in DATASETS_MAP:
            raise ValueError(f'Dataset must be one of {DATASETS_MAP.keys()}')
        return v

class ModelOptions(BaseModel):
    name: str
    configuration: ModelConfig

    @field_validator('name')
    def validate_model_name(cls, v):
        """
        Validate that the model name exists in MODELS_MAP.

        Args:
            v (str): The name of the model.
        
        Returns:
            str: The validated model name.
        
        Raises:
            ValueError: If the model name is not in MODELS_MAP.
        """
        if v not in MODELS_MAP:
            raise ValueError(f'Model must be one of {MODELS_MAP.keys()}')
        return v
    
    @field_validator('configuration')
    def validate_model_config(cls, v):
        """
        Validate the model configuration based on the model name.

        Args:
            v (ModelConfig): The configuration of the model.
        
        Returns:
            ModelConfig: The validated model configuration.
        
        # TODO: validate model config based on the model name
        """
        return v

class TrainRequest(BaseModel):
    """
    Request model for training a new machine learning model.

    Attributes:
        dataset (DatasetOptions): The dataset configuration.
        model (ModelOptions): The model configuration.
    """
    dataset: DatasetOptions
    model: ModelOptions

class PredictRequest(BaseModel):
    """
    Request model for making predictions using a trained model.

    Attributes:
        data (list[str]): The input data for prediction.
        model_artifact_name (str): The name of the model artifact to use for prediction.
    """
    data: list[str]
    model_artifact_name: str

    @field_validator("model_artifact_name")
    def validate_model_artifact_name(cls, v):
        """
        Validate the model artifact name.

        Args:
            v (str): The name of the model artifact.
        
        Returns:
            str: The validated model artifact name.
        
        # TODO: validate model artifact name
        """
        return v
