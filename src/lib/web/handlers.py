from src.lib.datasets import DATASETS_MAP
from src.lib.models import MODELS_MAP
from src.lib.trainer import Trainer, Metrics
from src.lib.datasets.interfaces import Dataset
from src.lib.models.interfaces import ModelInterface, ModelConfig
from src.lib.datasets.interfaces import Data
from src.lib.web.interfaces import PredictRequest, PredictResponse, TrainResponse, TrainRequest, ListModelArtifactsResponse
from src.lib.storage.local_artifact_storage import LocalArtifactStorage

STORAGE = LocalArtifactStorage()


def train_handler(request: TrainRequest) -> TrainResponse:
    """
    Handle the training of a new model based on the provided TrainRequest.

    Args:
        request (TrainRequest): The request containing dataset and model configurations.

    Returns:
        TrainResponse: A response containing the artifact name and evaluation metrics.
    """
    train_dataset: Dataset = DATASETS_MAP[request.dataset.name].load(split="train")
    test_dataset: Dataset | None = DATASETS_MAP[request.dataset.name].load(split="test")
    model: ModelInterface = MODELS_MAP[request.model.name](request.model.configuration)
    trainer = Trainer(model, train_dataset, test_dataset)
    model: ModelInterface = trainer.fit()
    metrics = trainer.evaluate()
    artifact_name = STORAGE.save(model, request.dataset.name)
    return TrainResponse(
        artifact_name=artifact_name,
        metrics=metrics,
    )

def predict_handler(request: PredictRequest) -> PredictResponse:
    """
    Handle prediction requests using a specified model artifact.

    Args:
        request (PredictRequest): The request containing data and the model artifact name.

    Returns:
        PredictResponse: A response containing the predictions.
    """
    model: ModelInterface = STORAGE.load(request.model_artifact_name)
    data = Data(request.data)
    predictions = model.predict(data)
    return PredictResponse(
        predictions=predictions.to_list()
    )

def list_model_artifacts_handler() -> ListModelArtifactsResponse:
    """
    List all available model artifacts stored locally.

    Returns:
        ListModelArtifactsResponse: A response containing a list of artifact names.
    """
    return ListModelArtifactsResponse(
        artifacts=list(STORAGE.list())
    )
