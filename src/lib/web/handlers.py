from src.lib.web.interfaces import TrainRequest
from src.lib.datasets import DATASETS_MAP
from src.lib.trainer import Trainer, Metrics
from src.lib.datasets.interfaces import Dataset


def train_handler(request: TrainRequest) -> dict:
    train_dataset: Dataset = DATASETS_MAP[request.dataset.name].load(split="train")
    test_dataset: Dataset | None = DATASETS_MAP[request.dataset.name].load(split="test")
    trainer = Trainer(request.model.name, train_dataset, test_dataset)
    metrics: Metrics = trainer.fit()
    return {
        "metrics": {
            "accuracy": metrics.accuracy,
            "f1": metrics.f1,
            "precision": metrics.precision,
            "recall": metrics.recall,
        }
    }

def predict_handler(request: PredictRequest) -> dict:
    raise NotImplementedError()