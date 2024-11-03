from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import src.common.utils as utils
from src.lib.models.interfaces import ModelInterface
from src.lib.datasets.interfaces import Dataset, Data, Targets
from src.lib.trainer.interfaces import Metrics


LOGGER = utils.initialize_logging(__name__)


class Trainer:
    def __init__(self, model: ModelInterface, train_dataset: Dataset, test_dataset: Dataset | None = None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def fit(self) -> ModelInterface:
        LOGGER.info(f"Fit model {self.model.NAME} with dataset {self.train_dataset.NAME}")
        self.model = self.model.fit(self.train_dataset)
        return self.model

    def predict(self, data: Data | None = None) -> Targets:
        data = data or self.test_dataset.data if self.test_dataset is not None else None
        if data is None:
            raise ValueError("You should provide data for prediction if there is no test dataset")

        LOGGER.info(f"Predict with model {self.model.NAME}")

        return self.model.predict(data)
    
    def evaluate(self, dataset: Dataset | None = None) -> Metrics | None:
        if dataset is None:
            dataset = self.test_dataset

        predictions = self.model.predict(dataset.data)
        return self.compute_metrics(predictions, dataset.targets)

    def compute_metrics(self, predicted: Targets, true: Targets) -> Metrics:
        true_list = true.to_list()
        predicted_list = predicted.to_list()
        return Metrics(
            accuracy=float(accuracy_score(true_list, predicted_list)),
            f1=float(f1_score(true_list, predicted_list)),
            precision=float(precision_score(true_list, predicted_list)),
            recall=float(recall_score(true_list, predicted_list)),
        )
