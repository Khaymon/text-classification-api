from src.lib.web.interfaces import TrainRequest
from src.lib.datasets import DATASETS_MAP
from src.lib.models import MODELS_MAP
from src.lib.trainer import Trainer, Metrics
from src.lib.datasets.interfaces import Dataset
from src.lib.models.interfaces import ModelInterface, ModelConfig
from src.lib.datasets.interfaces import Data
from src.lib.web.interfaces import PredictRequest
from src.lib.storage.local_artifact_storage import LocalArtifactStorage

storage = LocalArtifactStorage()


def train_handler(request: TrainRequest) -> dict:
    train_dataset: Dataset = DATASETS_MAP[request.dataset.name].load(split="train")
    test_dataset: Dataset | None = DATASETS_MAP[request.dataset.name].load(split="test")
    model: ModelInterface = MODELS_MAP[request.model.name](request.model.configuration)
    trainer = Trainer(model, train_dataset, test_dataset)
    model: ModelInterface = trainer.fit()
    metrics = trainer.evaluate()
    artifact_name = storage.save(model, request.dataset.name)
    return {
        "artifact_name": artifact_name,
        "metrics": metrics.model_dump()
    }
