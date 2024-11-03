from src.lib.web.interfaces import TrainRequest
from src.lib.datasets import DATASETS_MAP
from src.lib.models import MODELS_MAP
from src.lib.trainer import Trainer, Metrics
from src.lib.datasets.interfaces import Dataset
from src.lib.models.interfaces import ModelInterface, ModelConfig
from src.lib.datasets.interfaces import Data
from src.lib.web.interfaces import PredictRequest
from src.lib.storage.local_artifact_storage import LocalArtifactStorage

STORAGE = LocalArtifactStorage()


def train_handler(request: TrainRequest) -> dict:
    """
    Handle the training of a new model based on the provided TrainRequest.

    Args:
        request (TrainRequest): The request containing dataset and model configurations.
    
    Returns:
        dict: A dictionary containing the artifact name and evaluation metrics.
    """
    train_dataset: Dataset = DATASETS_MAP[request.dataset.name].load(split="train")
    test_dataset: Dataset | None = DATASETS_MAP[request.dataset.name].load(split="test")
    model: ModelInterface = MODELS_MAP[request.model.name](request.model.configuration)
    trainer = Trainer(model, train_dataset, test_dataset)
    model: ModelInterface = trainer.fit()
    metrics = trainer.evaluate()
    artifact_name = STORAGE.save(model, request.dataset.name)
    return {
        "artifact_name": artifact_name,
        "metrics": metrics.model_dump()
    }

def predict_handler(request: PredictRequest) -> dict:
    """
    Handle prediction requests using a specified model artifact.

    Args:
        request (PredictRequest): The request containing data and the model artifact name.
    
    Returns:
        dict: A dictionary containing the predictions.
    """
    model: ModelInterface = STORAGE.load(request.model_artifact_name)
    data = Data(request.data)
    predictions = model.predict(data)
    return {
        "predictions": predictions.to_list()
    }

def list_model_artifacts_handler() -> dict:
    """
    List all available model artifacts stored locally.

    Returns:
        dict: A dictionary containing a list of artifact names.
    """
    return {"artifacts": list(STORAGE.list())}
