from src.lib.web.interfaces import TrainRequest
from src.lib.datasets import DATASETS_MAP
from src.lib.models import MODELS_MAP
from src.lib.trainer import Trainer, Metrics
from src.lib.datasets.interfaces import Dataset
from src.lib.models.interfaces import ModelInterface, ModelConfig


def train_handler(request: TrainRequest) -> dict:
    train_dataset: Dataset = DATASETS_MAP[request.dataset.name].load(split="train")
    test_dataset: Dataset | None = DATASETS_MAP[request.dataset.name].load(split="test")
    model: ModelInterface = MODELS_MAP[request.model.name](request.model.configuration)
    trainer = Trainer(model, train_dataset, test_dataset)
    metrics: Metrics = trainer.fit()
    return {
        "metrics": metrics.model_dump()
    }
