from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.lib.models.interfaces import ModelInterface
from src.lib.datasets.interfaces import Dataset, Data, Targets
from src.lib.trainer.interfaces import Metrics


class Trainer:
    def __init__(self, model: ModelInterface, train_dataset: Dataset, test_dataset: Dataset | None = None):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def fit(self) -> Metrics | None:
        self.model = self.model.fit(self.train_dataset)
        if self.test_dataset:
            predictions = self.model.predict(self.test_dataset.data)
            return self.evaluate(predictions, self.test_dataset.targets)

    def predict(self, data: Data | None = None) -> Targets:
        data = data or self.test_dataset.data
        return self.model.predict(data)
    
    @staticmethod
    def evaluate(predicted: Targets, true: Targets) -> Metrics:
        true = true.to_list()
        predicted = predicted.to_list()
        return Metrics(
            accuracy=accuracy_score(true, predicted),
            f1=f1_score(true, predicted),
            precision=precision_score(true, predicted),
            recall=recall_score(true, predicted),
        )
